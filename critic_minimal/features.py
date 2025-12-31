# critic_minimal/features.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# These appear in your captures for "no chip" / invalid slots.
INVALID_CHIP_IDS = {255, 65535}

# Controller inputs included as per-timestep ACTION features.
BUTTON_KEYS = [
    "DPAD_UP",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "START",
    "BACK",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "EAST",
    "SOUTH",  # EAST=A, SOUTH=B usually
]
ACTION_DIM = len(BUTTON_KEYS)

# Scalars in cache (already normalized in precache to roughly [0,1]).
SCALAR_NAMES = ["p_hp", "e_hp", "p_charge", "e_charge", "cust_gauge"]
SCALAR_DIM = len(SCALAR_NAMES)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, list):
            return float(v[0]) if v else float(default)
        return float(v)
    except Exception:
        return default


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def _btn01(v: Any) -> float:
    # captures can be scalar or 1-element list
    x = _as_float(v, 0.0)
    return 1.0 if x > 0.5 else 0.0


def _chip_id_norm(v: Any) -> int:
    x = _as_int(v, 0)
    if x in INVALID_CHIP_IDS or x < 0:
        return 0
    return x


def _pos_norm(v: Any) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _as_int(v[0], 0), _as_int(v[1], 0)
    if isinstance(v, str) and "," in v:
        parts = v.split(",")
        if len(parts) >= 2:
            return _as_int(parts[0], 0), _as_int(parts[1], 0)
    return 0, 0


def _tile_norm(v: Any) -> int:
    x = _as_int(v, 0)
    return 0 if x < 0 else x


def _owner_norm(v: Any) -> int:
    # expected: 0->P, 1->E, else unknown=2
    x = _as_int(v, 2)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 2


# -----------------------------------------------------------------------------
# Position -> 0..17 Grid Index (6 cols x 3 rows)
# -----------------------------------------------------------------------------

def pos_to_grid_idx(x: float, y: float) -> int:
    """
    Data-driven mapping of (x,y) to 0-17 grid index.
    Based on dataset analysis:
      X clusters ~ 20, 60, 100, 140, 180, 220 (40px spacing)
      Y clusters ~ 260, 515, 770
    """
    col = int(float(x) // 40.0)
    col = max(0, min(5, col))

    row = 1
    if float(y) < 387.0:
        row = 0
    elif float(y) > 642.0:
        row = 2

    return int(row * 6 + col)


def grid_idx_to_row_col(idx: int) -> Tuple[int, int]:
    idx = int(idx)
    return (idx // 6), (idx % 6)


def rel_pe_index(p_idx: int, e_idx: int) -> int:
    """
    Relative offset index from player->enemy.
    dx in [-5..5], dy in [-2..2] => 11*5 = 55 bins => [0..54]
    """
    pr, pc = grid_idx_to_row_col(p_idx)
    er, ec = grid_idx_to_row_col(e_idx)

    dx = max(-5, min(5, ec - pc))
    dy = max(-2, min(2, er - pr))

    dx_i = dx + 5
    dy_i = dy + 2
    return int(dy_i * 11 + dx_i)


# -----------------------------------------------------------------------------
# Actions: aggregate across held window
# -----------------------------------------------------------------------------

def aggregate_action(frames: Sequence[Dict[str, Any]], raw_i: int, hold: int) -> torch.Tensor:
    """
    Aggregate button presses over [raw_i, raw_i+hold).
    We use max (OR) so any press in the held interval counts.

    Returns: float32 [ACTION_DIM] in BUTTON_KEYS order.
    """
    n = len(frames)
    a = torch.zeros((ACTION_DIM,), dtype=torch.float32)
    lo = max(0, int(raw_i))
    hi = min(n, int(raw_i) + max(1, int(hold)))

    for j in range(lo, hi):
        f = frames[j]
        for k_idx, k in enumerate(BUTTON_KEYS):
            # max() in python is fine here; ACTION_DIM is tiny
            a[k_idx] = max(float(a[k_idx].item()), float(_btn01(f.get(k, 0.0))))
    return a


# -----------------------------------------------------------------------------
# Bellman HP-return on sampled timeline (Nitrogen-style)
# -----------------------------------------------------------------------------

def bellman_hp_return_sampled(
    frames: Sequence[Dict[str, Any]],
    sample_raw_indices: Sequence[int],
    battle_mask_sample: Sequence[bool],
    *,
    gamma: float,
    reward_scale: float,
    positive_only: bool,
    ema_alpha: float,
) -> List[float]:
    """
    Nitrogen-style value target, computed on sampled timeline:

      reward_t = (max(0, dEHP) - max(0, dPHP)) * reward_scale
      return_t = reward_t + gamma * return_{t+1}

    dEHP/dPHP are computed between successive sampled points.
    Return resets to 0 outside battle_mask_sample.
    Optional EMA smoothing applies only inside battle segments.

    Output length == len(sample_raw_indices).
    """
    S = int(len(sample_raw_indices))
    if S == 0:
        return []

    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    # HP at sampled points
    p_hp = [0.0] * S
    e_hp = [0.0] * S
    last_p = 0.0
    last_e = 0.0
    for t, ri in enumerate(sample_raw_indices):
        if 0 <= ri < len(frames):
            last_p = _as_float(frames[ri].get("player_health"), last_p)
            last_e = _as_float(frames[ri].get("enemy_health"), last_e)
        p_hp[t] = float(last_p)
        e_hp[t] = float(last_e)

    # rewards (sampled), reward[0]=0
    r = [0.0] * S
    for t in range(1, S):
        prev_p, prev_e = p_hp[t - 1], e_hp[t - 1]
        cur_p, cur_e = p_hp[t], e_hp[t]

        dmg_taken = max(0.0, prev_p - cur_p)
        dmg_dealt = max(0.0, prev_e - cur_e)

        rt = (dmg_dealt - dmg_taken) * float(reward_scale)
        if positive_only:
            rt = max(0.0, rt)
        r[t] = float(rt)

    # discounted return backwards, reset outside battle
    out = [0.0] * S
    nxt = 0.0
    for t in range(S - 1, -1, -1):
        if not bool(battle_mask_sample[t]):
            nxt = 0.0
            out[t] = 0.0
            continue
        nxt = float(r[t]) + g * nxt
        out[t] = float(nxt)

    # optional EMA smoothing inside battle segments
    a = float(ema_alpha)
    if a > 0.0:
        ema: Optional[float] = None
        for t in range(S):
            if not bool(battle_mask_sample[t]):
                ema = None
                out[t] = 0.0
                continue
            ema = float(out[t]) if ema is None else (a * float(out[t]) + (1.0 - a) * float(ema))
            out[t] = float(ema)

    return out


def compute_done_from_valid(valid: torch.Tensor) -> torch.Tensor:
    """
    valid: [T] bool
    done[t] = valid[t] and not valid[t+1] (or end)
    """
    if valid.ndim != 1:
        raise ValueError(f"valid must be [T], got {tuple(valid.shape)}")
    T = int(valid.shape[0])
    done = torch.zeros((T,), dtype=torch.bool)
    if T == 0:
        return done
    if T == 1:
        done[0] = bool(valid[0].item())
        return done

    vn = torch.zeros_like(valid)
    vn[:-1] = valid[1:]
    done = valid & (~vn)
    return done


# -----------------------------------------------------------------------------
# Target normalization + weighting
# -----------------------------------------------------------------------------

def normalize_target_tanh(y_raw: torch.Tensor, *, norm_factor: float) -> torch.Tensor:
    """
    y_norm = tanh(y_raw / norm_factor)
    Keeps targets bounded in [-1,1] and stabilizes training.
    """
    nf = max(1e-6, float(norm_factor))
    return torch.tanh(y_raw / nf)


def value_weight_tanh(
    y_raw: torch.Tensor,
    *,
    norm_factor: float,
    tanh_clip: float,
    scale: float,
) -> torch.Tensor:
    """
    Nitrogen-flavored weighting from |raw target|:

      w = tanh( clamp(|y| / norm_factor, 0..tanh_clip) * scale )

    Returns w in [0, 1). You typically use (1 + w) or (eps + w).
    """
    nf = max(1e-6, float(norm_factor))
    tc = max(0.0, float(tanh_clip))
    sc = float(scale)

    a = (y_raw.abs() / nf)
    if tc > 0.0:
        a = torch.clamp(a, 0.0, tc)
    a = a * sc
    return torch.tanh(a)


__all__ = [
    "INVALID_CHIP_IDS",
    "BUTTON_KEYS",
    "ACTION_DIM",
    "SCALAR_NAMES",
    "SCALAR_DIM",
    "_as_int",
    "_as_float",
    "_as_list",
    "_btn01",
    "_chip_id_norm",
    "_pos_norm",
    "_tile_norm",
    "_owner_norm",
    "pos_to_grid_idx",
    "grid_idx_to_row_col",
    "rel_pe_index",
    "aggregate_action",
    "bellman_hp_return_sampled",
    "compute_done_from_valid",
    "normalize_target_tanh",
    "value_weight_tanh",
]
