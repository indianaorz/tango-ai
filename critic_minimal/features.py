# critic_minimal/features.py
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
import torch

INVALID_CHIP_IDS = {255, 65535}
BUTTON_KEYS = [
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
    "START", "BACK", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "EAST", "SOUTH"
]
ACTION_DIM = len(BUTTON_KEYS)

# ---- basic helpers ----
def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []

def _btn01(v: Any) -> float:
    return 1.0 if _as_float(v) > 0.5 else 0.0

# ---- normalization helpers ----
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _chip_id_norm01(v: Any) -> float:
    x = _as_int(v, 0)
    if x in INVALID_CHIP_IDS or x < 0:
        return 0.0
    # chip ids are 0..255
    return _clamp01(float(x) / 255.0)

def _emo_norm01(v: Any) -> float:
    # emotions in your mapping look like 0..22-ish; clamp to safe range
    x = _as_int(v, 0)
    return _clamp01(float(max(0, min(22, x))) / 22.0)

def _tile_norm01(v: Any) -> float:
    # tile types in your UI are 1..7-ish (plus sometimes 0)
    x = _as_int(v, 0)
    return _clamp01(float(max(0, min(7, x))) / 7.0)

def _owner_norm01(v: Any) -> float:
    # 0=Player, 1=Enemy, 2=Neutral/Unknown
    x = _as_int(v, 2)
    return _clamp01(float(max(0, min(2, x))) / 2.0)

def _pos_norm(v: Any) -> Tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return 0.0, 0.0

# ---- grid helpers ----
def pos_to_grid_idx(x: float, y: float) -> int:
    col = int(x // 40.0)
    col = max(0, min(5, col))
    row = 1
    if y < 387.0:
        row = 0
    elif y > 642.0:
        row = 2
    return int(row * 6 + col)

def rel_pe_index(p_idx: int, e_idx: int) -> float:
    pr, pc = p_idx // 6, p_idx % 6
    er, ec = e_idx // 6, e_idx % 6
    dx = max(-5, min(5, ec - pc))
    dy = max(-2, min(2, er - pr))
    # (dy+2) in [0..4], (dx+5) in [0..10] => [0..54]
    return float((dy + 2) * 11 + (dx + 5))

# ---- action aggregation ----
def aggregate_action(frames: Sequence[Dict[str, Any]], raw_i: int, hold: int) -> torch.Tensor:
    n = len(frames)
    a = torch.zeros((ACTION_DIM,), dtype=torch.float32)
    lo = max(0, int(raw_i))
    hi = min(n, int(raw_i) + max(1, int(hold)))

    for j in range(lo, hi):
        f = frames[j]
        for k_idx, k in enumerate(BUTTON_KEYS):
            val = _btn01(f.get(k, 0.0))
            if val > a[k_idx]:
                a[k_idx] = val
    return a

# ---- reward (unchanged here) ----
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
    S = len(sample_raw_indices)
    if S == 0:
        return []

    p_hp = [0.0] * S
    e_hp = [0.0] * S

    last_p, last_e = 0.0, 0.0
    for t, ri in enumerate(sample_raw_indices):
        if 0 <= ri < len(frames):
            last_p = _as_float(frames[ri].get("player_health"), last_p)
            last_e = _as_float(frames[ri].get("enemy_health"), last_e)
        p_hp[t] = last_p
        e_hp[t] = last_e

    out = [0.0] * S
    nxt = 0.0

    for t in range(S - 1, -1, -1):
        if not battle_mask_sample[t]:
            nxt = 0.0
            out[t] = 0.0
            continue

        if t < S - 1:
            dmg_taken = max(0.0, p_hp[t] - p_hp[t + 1])
            dmg_dealt = max(0.0, e_hp[t] - e_hp[t + 1])
        else:
            dmg_taken, dmg_dealt = 0.0, 0.0

        r = (dmg_dealt - dmg_taken) * reward_scale
        if positive_only:
            r = max(0.0, r)

        nxt = r + gamma * nxt
        out[t] = nxt

    if ema_alpha > 0.0:
        ema = None
        for t in range(S):
            if not battle_mask_sample[t]:
                ema = None
                continue
            ema = out[t] if ema is None else (ema_alpha * out[t] + (1 - ema_alpha) * ema)
            out[t] = ema

    return out

# ---- feature extractor ----
def extract_flow_features(f: Dict[str, Any]) -> List[float]:
    # 1) scalars (0..1)
    s = [
        _clamp01(_as_float(f.get("player_health")) / 2500.0),
        _clamp01(_as_float(f.get("enemy_health")) / 2500.0),
        _clamp01(_as_float(f.get("player_charge")) / 2.0),
        _clamp01(_as_float(f.get("enemy_charge")) / 2.0),
        _clamp01(_as_float(f.get("cust_gauge")) / 100.0),
    ]

    # 2) categoricals normalized (0..1)
    cats = [
        _emo_norm01(f.get("player_game_emotion")),
        _emo_norm01(f.get("enemy_game_emotion")),
        _chip_id_norm01(f.get("player_chip")),
    ]

    # 3) grid (18 tiles * 2)
    gs = _as_list(f.get("grid_state"))
    go = _as_list(f.get("grid_owner_state"))
    grid_feats: List[float] = []
    for i in range(18):
        grid_feats.append(_tile_norm01(gs[i] if i < len(gs) else 0))
        grid_feats.append(_owner_norm01(go[i] if i < len(go) else 2))

    # 4) positions normalized (0..1)
    px, py = _pos_norm(f.get("player_pos"))
    ex, ey = _pos_norm(f.get("enemy_pos"))
    pidx = pos_to_grid_idx(px, py)
    eidx = pos_to_grid_idx(ex, ey)
    rel = rel_pe_index(pidx, eidx)

    pos_feats = [
        float(pidx) / 17.0,
        float(eidx) / 17.0,
        float(rel) / 54.0,
    ]

    return s + cats + grid_feats + pos_feats
