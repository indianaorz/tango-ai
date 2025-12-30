# critic_rl/dataset.py
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from viewer.derived_state import compute_derived

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# These appear in your captures for "no chip" / invalid slots.
INVALID_CHIP_IDS = {255, 65535}

# Controller inputs included as per-timestep ACTION features (NOT state).
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

# State-only scalar vector (keep minimal and semantic)
# 0..15:
#   0  p_hp_norm
#   1  e_hp_norm
#   2  p_charge_norm
#   3  e_charge_norm
#   4  cust_norm
#   5  inside_window
#   6  turn_idx_norm
#   7  p_x_norm
#   8  p_y_norm
#   9  e_x_norm
#   10 e_y_norm
#   11 window_happened
#   12 window_selected_any
#   13 window_beast_selected
#   14 player_cross_entered
#   15 enemy_cross_entered
STATE_SCALAR_DIM = 16


# -----------------------------------------------------------------------------
# Position -> 0..17 Grid Index (6 cols x 3 rows)
# -----------------------------------------------------------------------------

def pos_to_grid_idx(x: float, y: float) -> int:
    """
    Data-driven mapping of (x,y) to 0-17 grid index.
    Based on dataset analysis:
      X clusters: 20, 60, 100, 140, 180, 220 (Standard 40px spacing)
      Y clusters: 260, 515, 770
    """
    col = int(x // 40)
    col = max(0, min(5, col))

    row = 1
    if y < 387:
        row = 0
    elif y > 642:
        row = 2

    return (row * 6) + col


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
# Helpers (robust parsing)
# -----------------------------------------------------------------------------

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
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def _chip_id_norm(v: Any) -> int:
    try:
        x = int(v)
    except Exception:
        return 0
    if x in INVALID_CHIP_IDS or x < 0:
        return 0
    return x


def _code_norm(v: Any, max_code: int = 63) -> int:
    try:
        x = int(v)
    except Exception:
        return 0
    if x < 0:
        return 0
    if x > max_code:
        return max_code
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
    if v is None:
        return 0
    try:
        x = int(v)
    except Exception:
        return 0
    if x < 0:
        return 0
    return x


def _owner_norm(v: Any) -> int:
    # expected: 0->P, 1->E, else unknown=2
    x = _as_int(v, 2)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 2


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_actions_jsonl(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return frames


# -----------------------------------------------------------------------------
# Reward (stride-aware)
# -----------------------------------------------------------------------------

def _hp_delta_series(frames: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    p = 0
    e = 0
    for f in frames:
        if "player_health" in f:
            p = _as_int(f.get("player_health"), p)
        if "enemy_health" in f:
            e = _as_int(f.get("enemy_health"), e)
        out.append(int(p - e))
    return out


def _reward_stride_hp_delta(hp_delta: List[int], raw_i: int, stride: int) -> float:
    """
    Reward for a sampled timestep raw_i with stride sampling:
      r = hp_delta[min(raw_i+stride, last)] - hp_delta[raw_i]

    This matches the sampled transition s_t -> s_{t+1} when your dataset uses
    raw indices: raw_i = start + k*stride.
    """
    if not hp_delta:
        return 0.0
    n = len(hp_delta)
    if raw_i < 0:
        raw_i = 0
    if raw_i >= n:
        raw_i = n - 1
    raw_next = raw_i + int(stride)
    if raw_next >= n:
        raw_next = n - 1
    return float(hp_delta[raw_next] - hp_delta[raw_i])


def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask


def _action_vec(frame: Dict[str, Any]) -> torch.Tensor:
    """
    Action-at-time-t as a dense vector (float32) aligned with BUTTON_KEYS.
    Handles scalar or 1-element list captures.
    """
    a = torch.zeros((ACTION_DIM,), dtype=torch.float32)
    for i, k in enumerate(BUTTON_KEYS):
        v = frame.get(k, 0.0)
        if isinstance(v, list):
            v = v[0] if v else 0.0
        a[i] = float(_as_float(v, 0.0))
    return a


# -----------------------------------------------------------------------------
# Tensorization
# -----------------------------------------------------------------------------

def tensorize_frame(
    frame: Dict[str, Any],
    d: Dict[str, Any],
    static: Dict[str, Any],
    *,
    folder_len: int = 30,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict of tensors for ONE timestep.

    Key design:
      - 'scalars' are STATE-ONLY (no buttons).
      - 'action' is per-timestep button vector.
      - The model predicts Q(s_t, a_t) (SARSA TD(λ) targets in train.py).
    """

    # Basic scalars
    p_hp = _as_int(frame.get("player_health"), 0)
    e_hp = _as_int(frame.get("enemy_health"), 0)
    p_chg = _as_int(frame.get("player_charge"), 0)
    e_chg = _as_int(frame.get("enemy_charge"), 0)
    cust = _as_int(frame.get("cust_gauge"), 0)
    inside = 1.0 if _as_bool(frame.get("inside_window"), False) else 0.0
    turn_idx = _as_int(d.get("turn_index"), 0)

    # Raw positions (ints)
    px_i, py_i = _pos_norm(frame.get("player_pos"))
    ex_i, ey_i = _pos_norm(frame.get("enemy_pos"))

    # Discrete grid idx (0..17)
    p_grid_idx = pos_to_grid_idx(float(px_i), float(py_i))
    e_grid_idx = pos_to_grid_idx(float(ex_i), float(ey_i))
    rel_idx = rel_pe_index(p_grid_idx, e_grid_idx)  # 0..54

    # Occupancy (18)
    p_occ = [0.0] * 18
    e_occ = [0.0] * 18
    p_occ[p_grid_idx] = 1.0
    e_occ[e_grid_idx] = 1.0

    # Continuous normalized positions (optional but useful)
    MAX_POS = 750.0
    px = max(0.0, min(float(px_i), MAX_POS)) / MAX_POS
    py = max(0.0, min(float(py_i), MAX_POS)) / MAX_POS
    ex = max(0.0, min(float(ex_i), MAX_POS)) / MAX_POS
    ey = max(0.0, min(float(ey_i), MAX_POS)) / MAX_POS

    # Scalars vector (STATE ONLY)
    scalars = torch.zeros(STATE_SCALAR_DIM, dtype=torch.float32)
    scalars[0] = float(p_hp) / 2500.0
    scalars[1] = float(e_hp) / 2500.0
    scalars[2] = float(p_chg) / 2.0
    scalars[3] = float(e_chg) / 2.0
    scalars[4] = float(cust) / 100.0
    scalars[5] = float(inside)
    scalars[6] = float(turn_idx) / 50.0
    scalars[7] = float(px)
    scalars[8] = float(py)
    scalars[9] = float(ex)
    scalars[10] = float(ey)

    # Derived scalar flags
    p_derived = d.get("player", {}) if isinstance(d.get("player", {}), dict) else {}
    e_derived = d.get("enemy", {}) if isinstance(d.get("enemy", {}), dict) else {}

    wc = p_derived.get("window_commit", {}) if isinstance(p_derived.get("window_commit", {}), dict) else {}
    scalars[11] = 1.0 if wc.get("happened") else 0.0
    scalars[12] = 1.0 if wc.get("selected_any") else 0.0
    scalars[13] = 1.0 if wc.get("beast_selected") else 0.0

    p_evt = p_derived.get("cross_event", {}) if isinstance(p_derived.get("cross_event", {}), dict) else {}
    e_evt = e_derived.get("cross_event", {}) if isinstance(e_derived.get("cross_event", {}), dict) else {}
    scalars[14] = 1.0 if p_evt.get("entered") else 0.0
    scalars[15] = 1.0 if e_evt.get("entered") else 0.0

    # Action vector (buttons at THIS timestep)
    action = _action_vec(frame)

    # Grid tokens (18)
    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))
    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0] * (18 - min(18, len(grid_state)))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2] * (18 - min(18, len(grid_owner)))

    # Hand (10 slots)
    chip_slots = _as_list(frame.get("chip_slots"))
    chip_codes = _as_list(frame.get("chip_codes"))
    vis = _as_int(frame.get("chip_visible_count"), 5)

    hand_id = [0] * 10
    hand_code = [0] * 10
    hand_vis = [0.0] * 10
    for i in range(10):
        if i < len(chip_slots):
            hand_id[i] = _chip_id_norm(chip_slots[i])
            hand_code[i] = _code_norm(chip_codes[i] if i < len(chip_codes) else 0)
        hand_vis[i] = 1.0 if i < vis else 0.0

    # Folder
    p_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("player_folder_ids"))]
    e_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("enemy_folder_ids"))]
    p_ids, p_mask = _pad_ids(p_folder_ids, folder_len)
    e_ids, e_mask = _pad_ids(e_folder_ids, folder_len)

    p_used_mask_raw = _as_list(p_derived.get("folder_used_mask"))
    e_used_mask_raw = _as_list(e_derived.get("folder_used_mask"))
    p_used = [1.0 if bool(p_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(p_used_mask_raw)))] + [0.0] * (
        folder_len - min(folder_len, len(p_used_mask_raw))
    )
    e_used = [1.0 if bool(e_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(e_used_mask_raw)))] + [0.0] * (
        folder_len - min(folder_len, len(e_used_mask_raw))
    )

    # Held (battle hand) [5]
    held = _as_list(p_derived.get("held_chips"))
    held_id = [0] * 5
    held_code = [0] * 5
    held_mask = [False] * 5
    for i in range(min(5, len(held))):
        h = held[i] if isinstance(held[i], dict) else {}
        held_id[i] = _chip_id_norm(h.get("id"))
        held_code[i] = _code_norm(h.get("code"))
        held_mask[i] = True

    # Cross & Beast
    p_used_cross = _as_list(p_derived.get("used_cross_mask"))
    e_used_cross = _as_list(e_derived.get("used_cross_mask"))
    used_cross = [0.0] * 22
    for i in range(min(11, len(p_used_cross))):
        used_cross[i] = 1.0 if bool(p_used_cross[i]) else 0.0
    for i in range(min(11, len(e_used_cross))):
        used_cross[11 + i] = 1.0 if bool(e_used_cross[i]) else 0.0

    p_active = p_derived.get("active_cross") if isinstance(p_derived.get("active_cross"), dict) else None
    e_active = e_derived.get("active_cross") if isinstance(e_derived.get("active_cross"), dict) else None
    p_idx = _as_int(p_active.get("idx"), 11) if isinstance(p_active, dict) else 11
    e_idx = _as_int(e_active.get("idx"), 11) if isinstance(e_active, dict) else 11

    p_beast = p_derived.get("beast") if isinstance(p_derived.get("beast"), dict) else {}
    e_beast = e_derived.get("beast") if isinstance(e_derived.get("beast"), dict) else {}
    p_b_active = 1.0 if bool(p_beast.get("active")) else 0.0
    e_b_active = 1.0 if bool(e_beast.get("active")) else 0.0
    p_b_ever = 1.0 if bool(p_beast.get("ever")) else 0.0
    e_b_ever = 1.0 if bool(e_beast.get("ever")) else 0.0
    p_ts = p_beast.get("turns_since")
    e_ts = e_beast.get("turns_since")
    p_ts_f = float(_as_int(p_ts, 999)) / 50.0 if p_ts is not None else 9.99
    e_ts_f = float(_as_int(e_ts, 999)) / 50.0 if e_ts is not None else 9.99
    beast_feats = torch.tensor([p_b_active, p_ts_f, p_b_ever, e_b_active, e_ts_f, e_b_ever], dtype=torch.float32)

    # Last used + on deck
    last_p = _chip_id_norm(p_derived.get("used_chip_id"))
    last_e = _chip_id_norm(e_derived.get("used_chip_id"))
    curr_p = _chip_id_norm(frame.get("player_chip"))
    curr_e = _chip_id_norm(frame.get("enemy_chip"))

    # -------------------------------------------------------------------------
    # Option B: CNN-ready board tensor [C, 3, 6] (float)
    # -------------------------------------------------------------------------
    tile_norm = torch.tensor(gs, dtype=torch.float32) / 31.0  # coarse scale
    owner_norm = torch.tensor(go, dtype=torch.float32) / 2.0
    p_occ_t = torch.tensor(p_occ, dtype=torch.float32)
    e_occ_t = torch.tensor(e_occ, dtype=torch.float32)
    board_feat = torch.stack([tile_norm, owner_norm, p_occ_t, e_occ_t], dim=0).reshape(4, 3, 6)

    out: Dict[str, torch.Tensor] = {
        # --- state/action split ---
        "scalars": scalars,  # [STATE_SCALAR_DIM]
        "action": action,  # [ACTION_DIM]

        # existing towers
        "grid_tile": torch.tensor(gs, dtype=torch.int64),  # [18]
        "grid_owner": torch.tensor(go, dtype=torch.int64),  # [18]
        "hand_id": torch.tensor(hand_id, dtype=torch.int64),  # [10]
        "hand_code": torch.tensor(hand_code, dtype=torch.int64),  # [10]
        "hand_vis": torch.tensor(hand_vis, dtype=torch.float32),  # [10]
        "folder_id_p": torch.tensor(p_ids, dtype=torch.int64),  # [F]
        "folder_used_p": torch.tensor(p_used, dtype=torch.float32),  # [F]
        "folder_mask_p": torch.tensor(p_mask, dtype=torch.bool),  # [F]
        "folder_id_e": torch.tensor(e_ids, dtype=torch.int64),  # [F]
        "folder_used_e": torch.tensor(e_used, dtype=torch.float32),  # [F]
        "folder_mask_e": torch.tensor(e_mask, dtype=torch.bool),  # [F]
        "held_id": torch.tensor(held_id, dtype=torch.int64),  # [5]
        "held_code": torch.tensor(held_code, dtype=torch.int64),  # [5]
        "held_mask": torch.tensor(held_mask, dtype=torch.bool),  # [5]
        "used_cross": torch.tensor(used_cross, dtype=torch.float32),  # [22]
        "active_cross_idx_p": torch.tensor(p_idx, dtype=torch.int64),
        "active_cross_idx_e": torch.tensor(e_idx, dtype=torch.int64),
        "beast_feats": beast_feats,  # [6]
        "last_used_id_p": torch.tensor(last_p, dtype=torch.int64),
        "last_used_id_e": torch.tensor(last_e, dtype=torch.int64),
        "current_chip_p": torch.tensor(curr_p, dtype=torch.int64),
        "current_chip_e": torch.tensor(curr_e, dtype=torch.int64),

        # Option A/C features
        "p_grid_idx": torch.tensor(p_grid_idx, dtype=torch.int64),
        "e_grid_idx": torch.tensor(e_grid_idx, dtype=torch.int64),
        "rel_pe_idx": torch.tensor(rel_idx, dtype=torch.int64),
        "grid_p_occ": torch.tensor(p_occ, dtype=torch.float32),  # [18]
        "grid_e_occ": torch.tensor(e_occ, dtype=torch.float32),  # [18]

        # Option B feature
        "board_feat": board_feat,  # [4,3,6]
    }

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def _stack_time(xs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[torch.Tensor]] = {}
    for x in xs:
        for k, v in x.items():
            out.setdefault(k, []).append(v)
    return {k: torch.stack(vs, dim=0) for k, vs in out.items()}


def _pad_sequence_time(seq: Dict[str, torch.Tensor], target_len: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not seq:
        raise ValueError("empty seq")
    T = int(next(iter(seq.values())).shape[0])
    valid = torch.zeros((target_len,), dtype=torch.bool)
    n = min(T, target_len)
    valid[:n] = True

    if T >= target_len:
        return {k: v[:target_len] for k, v in seq.items()}, valid

    padded: Dict[str, torch.Tensor] = {}
    for k, v in seq.items():
        pad_shape = (target_len - T,) + tuple(v.shape[1:])
        padded[k] = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)], dim=0)
    return padded, valid


# -----------------------------------------------------------------------------
# Cache Builder
# -----------------------------------------------------------------------------

def _cache_params_dict(stride: int, folder_len: int, require_cust_gt0: bool, seq_len: int) -> Dict[str, Any]:
    # Bump format because reward/done semantics are different from v7.
    return {
        "format": "critic_rl_td_lambda_v8_action_conditioned_stride_reward",
        "stride": int(stride),
        "folder_len": int(folder_len),
        "require_cust_gt0": bool(require_cust_gt0),
        "seq_len": int(seq_len),
        "reward": "hp_delta_stride_diff",
        "target": "td_lambda_sarsa_bootstrap_in_train",
        "state_scalar_dim": int(STATE_SCALAR_DIM),
        "action_dim": int(ACTION_DIM),
        "action_keys": list(BUTTON_KEYS),
    }


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _process_replay_task(args_pack: Tuple[str, str, Dict[str, Any], int, int, int, bool, bool]) -> str:
    (child_str, out_root_str, params, stride, folder_len, seq_len, require_cust_gt0, overwrite) = args_pack
    child = Path(child_str)
    out_root = Path(out_root_str)
    name = child.name

    actions_path = child / "actions.jsonl"
    static_path = child / "static_data.json"
    cache_path = out_root / f"{name}.pt"

    if cache_path.exists() and not overwrite:
        try:
            ck = torch.load(cache_path, map_location="cpu")
            if isinstance(ck, dict) and ck.get("params") == params:
                return f"[SKIP] {name}"
        except Exception:
            pass

    frames = _read_actions_jsonl(actions_path)
    if not frames:
        return f"[EMPTY] {name}"
    static = _read_json(static_path) or {}

    try:
        derived = compute_derived(frames, static)
    except Exception as e:
        return f"[ERR] {name}: {e}"

    hp_delta = _hp_delta_series(frames)

    starts: List[int] = []
    for fi in range(0, len(frames), stride):
        if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
            continue
        starts.append(fi)

    if not starts:
        _atomic_torch_save(
            {
                "replay": name,
                "params": params,
                "start_frame": torch.empty((0,), dtype=torch.int64),
                "r": torch.empty((0, seq_len), dtype=torch.float32),
                "done": torch.empty((0, seq_len), dtype=torch.bool),
                "valid": torch.empty((0, seq_len), dtype=torch.bool),
                "x": {},
            },
            cache_path,
        )
        return f"[FILTERED] {name}"

    xs: Dict[str, List[torch.Tensor]] = {}
    rs: List[torch.Tensor] = []
    dones: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    n_frames = len(frames)

    for start in starts:
        per_x = []
        per_r = []
        per_done = []

        for k in range(seq_len):
            raw_i = start + k * stride
            if raw_i >= n_frames:
                break
            
            # --- FIX STARTS HERE ---
            # Strict check: If we enter an invalid state mid-sequence, ABORT.
            # This ensures the sequence is Pure Combat.
            if require_cust_gt0 and _as_int(frames[raw_i].get("cust_gauge"), 0) <= 0:
                break
            # -----------------------

            di = min(raw_i, len(derived) - 1) if derived else 0
            per_x.append(
                tensorize_frame(
                    frames[raw_i],
                    derived[di] if derived else {},
                    static,
                    folder_len=folder_len,
                )
            )

            # stride-aware reward matches sampled transition raw_i -> raw_i+stride
            per_r.append(_reward_stride_hp_delta(hp_delta, raw_i, stride))

            # done only when the sampled next would hit terminal
            raw_next = min(raw_i + stride, n_frames - 1)
            per_done.append(bool(raw_next >= n_frames - 1))

        if not per_x:
            continue

        seq = _stack_time(per_x)
        seq, valid = _pad_sequence_time(seq, target_len=seq_len)

        r_t = torch.zeros((seq_len,), dtype=torch.float32)
        d_t = torch.zeros((seq_len,), dtype=torch.bool)
        for i in range(min(len(per_r), seq_len)):
            r_t[i] = float(per_r[i])
        for i in range(min(len(per_done), seq_len)):
            d_t[i] = bool(per_done[i])

        for kk, vv in seq.items():
            xs.setdefault(kk, []).append(vv)
        rs.append(r_t)
        dones.append(d_t)
        valids.append(valid)

    if not rs:
        return f"[EMPTY] {name}"

    x_stacked = {k: torch.stack(vs, dim=0) for k, vs in xs.items()}
    _atomic_torch_save(
        {
            "replay": name,
            "params": params,
            "start_frame": torch.tensor(starts[: len(rs)], dtype=torch.int64),
            "r": torch.stack(rs, dim=0),
            "done": torch.stack(dones, dim=0),
            "valid": torch.stack(valids, dim=0),
            "x": x_stacked,
        },
        cache_path,
    )
    return f"[DONE] {name}"


def build_rl_sequence_cache(
    *,
    dataset_dir: str,
    cache_dir: str,
    stride: int,
    folder_len: int,
    seq_len: int,
    require_cust_gt0: bool,
    overwrite: bool,
    num_workers: int,
) -> None:
    root = Path(dataset_dir)
    out_root = Path(cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    params = _cache_params_dict(
        stride=stride,
        folder_len=folder_len,
        require_cust_gt0=require_cust_gt0,
        seq_len=seq_len,
    )

    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "actions.jsonl").exists()]
    if not dirs:
        raise RuntimeError(f"No replays in {root}")

    print(f"[cache] Starting parallel build with {num_workers} workers...")
    t0 = time.time()

    args = [(str(d), str(out_root), params, stride, folder_len, seq_len, require_cust_gt0, overwrite) for d in dirs]

    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_process_replay_task, args, chunksize=1), total=len(dirs)):
            pass

    manifest = {"params": params, "num_replay_dirs": len(dirs), "built_at": time.time()}
    _atomic_torch_save(manifest, out_root / "_manifest.pt")
    print(f"[cache] Done in {time.time() - t0:.1f}s")


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayMeta:
    name: str
    path: Path
    n_frames: int
    static: Dict[str, Any]


class _ReplayCache:
    def __init__(self, *, max_replays: int = 2):
        self.max_replays = int(max_replays)
        self._order: List[str] = []
        self._frames: Dict[str, List[Dict[str, Any]]] = {}
        self._derived: Dict[str, List[Dict[str, Any]]] = {}
        self._hp_delta: Dict[str, List[int]] = {}

    def get(self, meta: ReplayMeta) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
        key = meta.name
        if key in self._frames and key in self._derived and key in self._hp_delta:
            self._touch(key)
            return self._frames[key], self._derived[key], self._hp_delta[key]

        frames = _read_actions_jsonl(meta.path / "actions.jsonl")
        derived = compute_derived(frames, meta.static)
        hp_delta = _hp_delta_series(frames)

        self._frames[key] = frames
        self._derived[key] = derived
        self._hp_delta[key] = hp_delta
        self._touch(key)
        self._evict_if_needed()
        return frames, derived, hp_delta

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.max_replays:
            victim = self._order.pop(0)
            self._frames.pop(victim, None)
            self._derived.pop(victim, None)
            self._hp_delta.pop(victim, None)
    
class StreamingCachedDataset(torch.utils.data.IterableDataset):
    """
    Stream-loads cache files one by one.
    - Keeps RAM usage low (only active files loaded).
    - Supports multi-process loading via DataLoader workers.
    - Shuffles file order AND row order within files.
    """
    def __init__(
        self,
        dataset_dir: str,
        *,
        cache_dir: str,
        stride: int,
        folder_len: int,
        seq_len: int,
        require_cust_gt0: bool,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        max_samples: Optional[int] = None,
        boring_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.params = _cache_params_dict(
            stride=int(stride),
            folder_len=int(folder_len),
            require_cust_gt0=bool(require_cust_gt0),
            seq_len=int(seq_len),
        )
        self.boring_drop_rate = 0.0 if split == "val" else float(boring_drop_rate)
        self.seed = seed
        self.split = split

        # 1. Gather all file paths
        all_files = sorted([p for p in self.cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
        if not all_files:
            raise RuntimeError(f"No cache files in {cache_dir}")

        # 2. Deterministic split (Train vs Val)
        rng = __import__("random").Random(seed)
        names = [p.stem for p in all_files]
        rng.shuffle(names)
        
        if float(val_ratio) <= 0:
            n_val = 0
        else:
            n_val = max(1, int(len(names) * float(val_ratio)))
            
        val_set = set(names[:n_val])
        
        # Filter files based on split
        self.files = [p for p in all_files if ((p.stem in val_set) if split == "val" else (p.stem not in val_set))]
        
        if not self.files:
            print(f"WARNING: {split} dataset has 0 files.")

        # 3. accurate length from manifest
        self.total_samples = 0
        manifest_path = self.cache_dir / "_manifest.pt"
        if manifest_path.exists():
            try:
                m = torch.load(manifest_path)
                # If manifest has the count, use it
                if "total_samples" in m:
                    # Scale by ratio of files we are actually using (train vs val split)
                    total_in_cache = float(m["total_samples"])
                    total_files_in_cache = float(m.get("counted_files", len(all_files)))
                    if total_files_in_cache > 0:
                        avg_per_file = total_in_cache / total_files_in_cache
                        self.total_samples = int(len(self.files) * avg_per_file)
            except Exception:
                pass

        # Fallback if manifest is missing or old
        if self.total_samples == 0:
            self.total_samples = len(self.files) * 500  # Fallback guess

        # Adjust for drop rate if needed (approximate)
        if self.boring_drop_rate > 0:
            self.approx_len = int(self.total_samples * (1.0 - self.boring_drop_rate))
        else:
            self.approx_len = self.total_samples

    def __len__(self) -> int:
        return self.approx_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Sharding: Determine which files THIS worker is responsible for
        if worker_info is None:
            # Single-process
            my_files = list(self.files)
        else:
            # Multi-process: split files evenly
            per_worker = int(__import__("math").ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            my_files = self.files[iter_start:iter_end]

        # 2. Shuffle file order (Stochastic per epoch)
        # We use a seed based on epoch if possible, but here we relies on RNG state
        # For reproducibility in workers, we mix in worker_id
        g = torch.Generator()
        g.manual_seed(self.seed + (worker_info.id if worker_info else 0) + int(time.time()))
        
        indices = torch.randperm(len(my_files), generator=g).tolist()
        my_files = [my_files[i] for i in indices]

        # 3. Iterate through assigned files
        for p in my_files:
            try:
                # Load one file into RAM
                ck = torch.load(p, map_location="cpu")
                
                # Check params integrity
                if not isinstance(ck, dict) or ck.get("params") != self.params:
                    continue
                
                rewards = ck["r"] # [N, SeqLen]
                n_rows = int(rewards.shape[0])
                if n_rows == 0:
                    continue

                # Filter indices (boring drop rate)
                keep_indices = []
                # Vectorized boring check for speed
                # Check if ANY reward in sequence is non-zero
                is_interesting = (rewards.abs() > 0.001).any(dim=1) # [N] boolean
                
                # Random mask for boring ones
                if self.boring_drop_rate > 0:
                    rand_vals = torch.rand(n_rows)
                    # Keep if interesting OR (random > drop_rate)
                    keep_mask = is_interesting | (rand_vals >= self.boring_drop_rate)
                else:
                    keep_mask = torch.ones(n_rows, dtype=torch.bool)
                
                valid_idxs = torch.nonzero(keep_mask).squeeze(1) # [M]
                
                if valid_idxs.numel() == 0:
                    continue

                # Local Shuffle: Shuffle rows within this file
                perm = torch.randperm(valid_idxs.numel(), generator=g)
                shuffled_idxs = valid_idxs[perm]

                # Yield Items
                # We access the dict items one by one. 
                # Optimization: Access raw tensors to avoid overhead
                x_dict = ck["x"]
                r_tens = ck["r"]
                d_tens = ck["done"]
                v_tens = ck["valid"]

                for i in shuffled_idxs:
                    # Construct single sample dict
                    x_out = {k: v[i] for k, v in x_dict.items()}
                    yield x_out, r_tens[i], d_tens[i], v_tens[i]

                # cleanup explicitly to help GC in constrained RAM
                del ck, x_dict, r_tens, d_tens, v_tens, keep_mask, valid_idxs
                
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue
    

class CriticRLTDDataset(Dataset):
    """
    On-the-fly dataset (no cache files).
    """

    def __init__(
        self,
        root_dir: str,
        *,
        stride: int = 6,
        seq_len: int = 16,
        max_samples: Optional[int] = None,
        cache_replays: int = 2,
        folder_len: int = 30,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        require_cust_gt0: bool = True,
        boring_drop_rate: float = 0.90,
    ):
        self.root = Path(root_dir)
        self.stride = int(stride)
        self.seq_len = int(seq_len)
        self.max_samples = max_samples
        self.folder_len = int(folder_len)
        self.cache = _ReplayCache(max_replays=int(cache_replays))
        self.require_cust_gt0 = bool(require_cust_gt0)

        all_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir() and (p / "actions.jsonl").exists()]
        rng = __import__("random").Random(seed)
        rng.shuffle(all_dirs)
        if float(val_ratio) <= 0:
            n_val = 0
        else:
            n_val = max(1, int(len(all_dirs) * float(val_ratio)))

        if split == "val":
            target_dirs = all_dirs[:n_val]
            self.boring_drop_rate = 0.0
        else:
            target_dirs = all_dirs[n_val:]
            self.boring_drop_rate = float(boring_drop_rate)

        self.metas = [ReplayMeta(d.name, d, 0, _read_json(d / "static_data.json") or {}) for d in target_dirs]

        self.index: List[Tuple[int, int]] = []
        kept_boring = 0
        dropped_boring = 0
        kept_interesting = 0

        print(f"[{split}] Indexing and filtering (drop_rate={self.boring_drop_rate})...")
        for r_i, m in enumerate(tqdm(self.metas)):
            frames, _, hp_delta = self.cache.get(meta=m)
            if not frames:
                continue
            n_frames = len(frames)

            for fi in range(0, n_frames, self.stride):
                if self.require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                    continue

                # "interesting" if any stride-aware reward within sampled timesteps
                is_interesting = False
                for k in range(self.seq_len):
                    check_idx = fi + (k * self.stride)
                    if check_idx >= n_frames:
                        break
                    if abs(_reward_stride_hp_delta(hp_delta, check_idx, self.stride)) > 0.001:
                        is_interesting = True
                        break

                if is_interesting:
                    self.index.append((r_i, fi))
                    kept_interesting += 1
                else:
                    if rng.random() >= self.boring_drop_rate:
                        self.index.append((r_i, fi))
                        kept_boring += 1
                    else:
                        dropped_boring += 1

                if self.max_samples and len(self.index) >= self.max_samples:
                    break
            if self.max_samples and len(self.index) >= self.max_samples:
                break

        total = len(self.index)
        print(f"[{split}] Dataset Ready. Total: {total}")
        print(f"   Interesting (Kept): {kept_interesting}")
        print(f"   Boring (Kept):      {kept_boring}")
        print(f"   Boring (Dropped):   {dropped_boring}")
        if total > 0:
            print(f"   Signal Density:     {kept_interesting / total:.1%}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        r_i, start = self.index[idx]
        meta = self.metas[r_i]
        frames, derived, hp_delta = self.cache.get(meta)
        n = len(frames)

        per_x: List[Dict[str, torch.Tensor]] = []
        per_r: List[float] = []
        per_done: List[bool] = []

        for k in range(self.seq_len):
            ri = start + k * self.stride
            if ri >= n:
                break

            di = min(ri, len(derived) - 1) if derived else 0
            per_x.append(tensorize_frame(frames[ri], derived[di] if derived else {}, meta.static, folder_len=self.folder_len))

            per_r.append(_reward_stride_hp_delta(hp_delta, ri, self.stride))

            raw_next = min(ri + self.stride, n - 1)
            per_done.append(bool(raw_next >= n - 1))

        if not per_x:
            x0 = tensorize_frame(frames[0], derived[0] if derived else {}, meta.static, folder_len=self.folder_len)
            seq = _stack_time([x0])
            seq, valid = _pad_sequence_time(seq, self.seq_len)
            r_t = torch.zeros(self.seq_len, dtype=torch.float32)
            d_t = torch.zeros(self.seq_len, dtype=torch.bool)
            return seq, r_t, d_t, valid

        seq = _stack_time(per_x)
        seq, valid = _pad_sequence_time(seq, self.seq_len)

        r_t = torch.zeros((self.seq_len,), dtype=torch.float32)
        d_t = torch.zeros((self.seq_len,), dtype=torch.bool)
        for i in range(min(len(per_r), self.seq_len)):
            r_t[i] = float(per_r[i])
        for i in range(min(len(per_done), self.seq_len)):
            d_t[i] = bool(per_done[i])

        return seq, r_t, d_t, valid


class InMemoryCriticRLTDDataset(Dataset):
    """
    Loads ALL cache files into RAM (CPU tensors) at startup.
    """

    def __init__(
        self,
        dataset_dir: str,
        *,
        cache_dir: str,
        stride: int,
        folder_len: int,
        seq_len: int,
        require_cust_gt0: bool,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        max_samples: Optional[int] = None,
        boring_drop_rate: float = 0.90,
    ):
        self.cache_dir = Path(cache_dir)
        self.params = _cache_params_dict(
            stride=int(stride),
            folder_len=int(folder_len),
            require_cust_gt0=bool(require_cust_gt0),
            seq_len=int(seq_len),
        )

        self.boring_drop_rate = 0.0 if split == "val" else float(boring_drop_rate)

        all_files = sorted([p for p in self.cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
        if not all_files:
            raise RuntimeError(f"No cache files in {cache_dir}")

        rng = __import__("random").Random(seed)
        names = [p.stem for p in all_files]
        rng.shuffle(names)
        if float(val_ratio) <= 0:
            n_val = 0
        else:
            n_val = max(1, int(len(names) * float(val_ratio)))
        val_set = set(names[:n_val])
        files = [p for p in all_files if ((p.stem in val_set) if split == "val" else (p.stem not in val_set))]

        print(f"Loading {len(files)} files into RAM for {split} (drop_rate={self.boring_drop_rate})...")

        buffer_x: Dict[str, List[torch.Tensor]] = {}
        buffer_r: List[torch.Tensor] = []
        buffer_done: List[torch.Tensor] = []
        buffer_valid: List[torch.Tensor] = []

        total_loaded = 0
        kept_interesting = 0
        kept_boring = 0
        dropped_boring = 0

        for p in tqdm(files, desc=f"Loading {split} RAM"):
            if max_samples and total_loaded >= max_samples:
                break
            try:
                ck = torch.load(p, map_location="cpu")
                if not isinstance(ck, dict) or ck.get("params") != self.params:
                    continue

                rewards = ck["r"]  # [N, SeqLen]
                n_rows = int(rewards.shape[0])
                if n_rows == 0:
                    continue

                keep_indices: List[int] = []
                for i in range(n_rows):
                    is_interesting = bool((rewards[i].abs() > 0.001).any().item())
                    if is_interesting:
                        keep_indices.append(i)
                        kept_interesting += 1
                    else:
                        if rng.random() >= self.boring_drop_rate:
                            keep_indices.append(i)
                            kept_boring += 1
                        else:
                            dropped_boring += 1

                    if max_samples and (total_loaded + len(keep_indices)) >= max_samples:
                        break

                if not keep_indices:
                    continue

                idxs = torch.tensor(keep_indices, dtype=torch.long)

                for k, v in ck["x"].items():
                    buffer_x.setdefault(k, []).append(v[idxs])

                buffer_r.append(ck["r"][idxs])
                buffer_done.append(ck["done"][idxs])
                buffer_valid.append(ck["valid"][idxs])

                total_loaded += len(keep_indices)

                del ck
                if len(buffer_r) % 50 == 0:
                    gc.collect()

            except Exception:
                continue

        if total_loaded == 0:
            raise RuntimeError("No data loaded! Check cache generation or params.")

        print(f"Concatenating {total_loaded} samples...")

        self.x = {k: torch.cat(v_list, dim=0) for k, v_list in buffer_x.items()}
        self.r = torch.cat(buffer_r, dim=0)
        self.done = torch.cat(buffer_done, dim=0)
        self.valid = torch.cat(buffer_valid, dim=0)

        del buffer_x, buffer_r, buffer_done, buffer_valid
        gc.collect()

        print(f"RAM Dataset Ready ({split}). Total: {total_loaded}")
        print(f"   Interesting: {kept_interesting}")
        print(f"   Boring Kept: {kept_boring}")
        print(f"   Dropped:     {dropped_boring}")
        if total_loaded > 0:
            print(f"   Signal Density: {kept_interesting / total_loaded:.1%}")

    def __len__(self) -> int:
        return int(self.r.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        x = {k: v[idx] for k, v in self.x.items()}
        return x, self.r[idx], self.done[idx], self.valid[idx]

    def get_batch(self, indices: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = indices.to(torch.int64)
        x = {k: v[indices] for k, v in self.x.items()}
        r = self.r[indices]
        done = self.done[indices]
        valid = self.valid[indices]
        return x, r, done, valid


# Alias used by training script
# CachedCriticRLTDDataset = InMemoryCriticRLTDDataset
CachedCriticRLTDDataset = StreamingCachedDataset

def collate_batch(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, rs, ds, vs = zip(*batch)
    out: Dict[str, torch.Tensor] = {}
    for k in xs[0].keys():
        out[k] = torch.stack([x[k] for x in xs], dim=0)
    r = torch.stack(list(rs), dim=0)
    done = torch.stack(list(ds), dim=0)
    valid = torch.stack(list(vs), dim=0)
    return out, r, done, valid
            

__all__ = [
    "pos_to_grid_idx",
    "build_rl_sequence_cache",
    "CriticRLTDDataset",
    "InMemoryCriticRLTDDataset",
    "CachedCriticRLTDDataset",
    "StreamingCachedDataset",
    "collate_batch",
    "BUTTON_KEYS",
    "ACTION_DIM",
    "STATE_SCALAR_DIM",
]
