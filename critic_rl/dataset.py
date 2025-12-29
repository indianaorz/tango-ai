from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import multiprocessing
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from viewer.derived_state import compute_derived

INVALID_CHIP_IDS = {255, 65535}

BUTTON_KEYS = [
    'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT',
    'START', 'BACK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'EAST', 'SOUTH' # EAST=A, SOUTH=B usually in standard mapping
]
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_int(v: Any, default: int = 0) -> int:
    try: return int(v)
    except: return default

def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    return default

def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []

def _chip_id_norm(v: Any) -> int:
    try: x = int(v)
    except: return 0
    if x in INVALID_CHIP_IDS or x < 0: return 0
    return x

def _code_norm(v: Any, max_code: int = 63) -> int:
    try: x = int(v)
    except: return 0
    if x < 0: return 0
    if x > max_code: return max_code
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
    if v is None: return 0
    try: x = int(v)
    except: return 0
    if x < 0: return 0
    return x

def _owner_norm(v: Any) -> int:
    x = _as_int(v, 2)
    if x == 0: return 0
    if x == 1: return 1
    return 2

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def _read_actions_jsonl(path: Path) -> List[Dict[str, Any]]:
    frames = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: frames.append(json.loads(line))
                except: continue
    except: pass
    return frames

def _hp_delta_series(frames: List[Dict[str, Any]]) -> List[int]:
    out = []
    p = 0
    e = 0
    for f in frames:
        if "player_health" in f: p = _as_int(f.get("player_health"), p)
        if "enemy_health" in f: e = _as_int(f.get("enemy_health"), e)
        out.append(int(p - e))
    return out

def _hp_delta_rewards(hp_delta: List[int]) -> List[float]:
    if not hp_delta: return []
    r = [0.0] * len(hp_delta)
    for t in range(len(hp_delta) - 1):
        r[t] = float(hp_delta[t + 1] - hp_delta[t])
    r[-1] = 0.0
    return r

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
        self._rewards: Dict[str, List[float]] = {}

    def get(self, meta: ReplayMeta) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
        key = meta.name
        if key in self._frames and key in self._derived and key in self._rewards:
            self._touch(key)
            return self._frames[key], self._derived[key], self._rewards[key]

        frames = _read_actions_jsonl(meta.path / "actions.jsonl")
        derived = compute_derived(frames, meta.static)
        hp_delta = _hp_delta_series(frames)
        rewards = _hp_delta_rewards(hp_delta)

        self._frames[key] = frames
        self._derived[key] = derived
        self._rewards[key] = rewards
        self._touch(key)
        self._evict_if_needed()
        return frames, derived, rewards

    def _touch(self, key: str) -> None:
        if key in self._order: self._order.remove(key)
        self._order.append(key)

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.max_replays:
            victim = self._order.pop(0)
            self._frames.pop(victim, None)
            self._derived.pop(victim, None)
            self._rewards.pop(victim, None)

def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask

def tensorize_frame(frame, d, static, *, folder_len=30, device=None) -> Dict[str, torch.Tensor]:
    p_hp = _as_int(frame.get("player_health"), 0)
    e_hp = _as_int(frame.get("enemy_health"), 0)
    p_chg = _as_int(frame.get("player_charge"), 0)
    e_chg = _as_int(frame.get("enemy_charge"), 0)
    cust = _as_int(frame.get("cust_gauge"), 0)
    inside = 1.0 if _as_bool(frame.get("inside_window"), False) else 0.0
    turn_idx = _as_int(d.get("turn_index"), 0)

    # Raw positions
    px, py = _pos_norm(frame.get("player_pos"))
    ex, ey = _pos_norm(frame.get("enemy_pos"))
    
    # --- Clamp & Normalize Positions ---
    MAX_POS = 750.0
    px = max(0.0, min(float(px), MAX_POS)) / MAX_POS
    py = max(0.0, min(float(py), MAX_POS)) / MAX_POS
    ex = max(0.0, min(float(ex), MAX_POS)) / MAX_POS
    ey = max(0.0, min(float(ey), MAX_POS)) / MAX_POS

    # --- Scalars (UPDATED TO 48) ---
    scalars = torch.zeros(48, dtype=torch.float32)
    scalars[0] = float(p_hp) / 1000.0
    scalars[1] = float(e_hp) / 1000.0
    scalars[2] = float(p_chg) / 2.0
    scalars[3] = float(e_chg) / 2.0
    scalars[4] = float(cust) / 100.0
    scalars[5] = float(inside)
    scalars[6] = float(turn_idx) / 50.0
    scalars[7] = px
    scalars[8] = py
    scalars[9] = ex
    scalars[10] = ey
    
    # --- Derived Scalars ---
    p_derived = d.get("player", {})
    e_derived = d.get("enemy", {})
    
    wc = p_derived.get("window_commit", {})
    scalars[11] = 1.0 if wc.get("happened") else 0.0
    scalars[12] = 1.0 if wc.get("selected_any") else 0.0
    scalars[13] = 1.0 if wc.get("beast_selected") else 0.0
    
    p_evt = p_derived.get("cross_event", {})
    e_evt = e_derived.get("cross_event", {})
    scalars[14] = 1.0 if p_evt.get("entered") else 0.0
    scalars[15] = 1.0 if e_evt.get("entered") else 0.0

    # --- NEW: Controller Inputs (Indices 16-25) ---
    current_idx = 16
    for k in BUTTON_KEYS:
        val = frame.get(k, 0.0)
        if isinstance(val, list): val = val[0] # Handle [1.0] format
        scalars[current_idx] = float(val)
        current_idx += 1

    # --- Grid ---
    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))
    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0]*(18-len(grid_state))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2]*(18-len(grid_owner))

    # --- Hand (Window Slots) ---
    chip_slots = _as_list(frame.get("chip_slots"))
    chip_codes = _as_list(frame.get("chip_codes"))
    vis = _as_int(frame.get("chip_visible_count"), 5)
    
    hand_id = [0]*10
    hand_code = [0]*10
    hand_vis = [0.0]*10
    for i in range(10):
        if i < len(chip_slots):
            hand_id[i] = _chip_id_norm(chip_slots[i])
            hand_code[i] = _code_norm(chip_codes[i] if i < len(chip_codes) else 0)
            hand_vis[i] = 1.0 if i < vis else 0.0

    # --- Folder ---
    p_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("player_folder_ids"))]
    e_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("enemy_folder_ids"))]
    p_ids, p_mask = _pad_ids(p_folder_ids, folder_len)
    e_ids, e_mask = _pad_ids(e_folder_ids, folder_len)

    p_used_mask_raw = _as_list(d.get("player", {}).get("folder_used_mask"))
    e_used_mask_raw = _as_list(d.get("enemy", {}).get("folder_used_mask"))
    
    p_used = [1.0 if bool(p_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(p_used_mask_raw)))] + [0.0]*(folder_len-len(p_used_mask_raw))
    e_used = [1.0 if bool(e_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(e_used_mask_raw)))] + [0.0]*(folder_len-len(e_used_mask_raw))

    # --- Held (Battle Hand) ---
    held = _as_list(d.get("player", {}).get("held_chips"))
    held_id = [0]*5
    held_code = [0]*5
    held_mask = [False]*5
    for i in range(min(5, len(held))):
        h = held[i] if isinstance(held[i], dict) else {}
        held_id[i] = _chip_id_norm(h.get("id"))
        held_code[i] = _code_norm(h.get("code"))
        held_mask[i] = True

    # --- Cross & Beast ---
    p_used_cross = _as_list(d.get("player", {}).get("used_cross_mask"))
    e_used_cross = _as_list(d.get("enemy", {}).get("used_cross_mask"))
    used_cross = [0.0]*22
    for i in range(min(11, len(p_used_cross))): used_cross[i] = 1.0 if bool(p_used_cross[i]) else 0.0
    for i in range(min(11, len(e_used_cross))): used_cross[11+i] = 1.0 if bool(e_used_cross[i]) else 0.0

    p_active = d.get("player", {}).get("active_cross") or None
    e_active = d.get("enemy", {}).get("active_cross") or None
    p_idx = _as_int(p_active.get("idx"), 11) if isinstance(p_active, dict) else 11
    e_idx = _as_int(e_active.get("idx"), 11) if isinstance(e_active, dict) else 11

    p_beast = d.get("player", {}).get("beast") or {}
    e_beast = d.get("enemy", {}).get("beast") or {}
    p_b_active = 1.0 if bool(p_beast.get("active")) else 0.0
    e_b_active = 1.0 if bool(e_beast.get("active")) else 0.0
    p_b_ever = 1.0 if bool(p_beast.get("ever")) else 0.0
    e_b_ever = 1.0 if bool(e_beast.get("ever")) else 0.0
    p_ts = p_beast.get("turns_since")
    e_ts = e_beast.get("turns_since")
    p_ts_f = float(_as_int(p_ts, 999))/50.0 if p_ts is not None else 9.99
    e_ts_f = float(_as_int(e_ts, 999))/50.0 if e_ts is not None else 9.99

    beast_feats = torch.tensor([p_b_active, p_ts_f, p_b_ever, e_b_active, e_ts_f, e_b_ever], dtype=torch.float32)
    
    # --- Last Used & On Deck ---
    last_p = _chip_id_norm(p_derived.get("used_chip_id"))
    last_e = _chip_id_norm(e_derived.get("used_chip_id"))
    curr_p = _chip_id_norm(frame.get("player_chip"))
    curr_e = _chip_id_norm(frame.get("enemy_chip"))

    out = {
        "scalars": scalars,
        "grid_tile": torch.tensor(gs, dtype=torch.int64),
        "grid_owner": torch.tensor(go, dtype=torch.int64),
        "hand_id": torch.tensor(hand_id, dtype=torch.int64),
        "hand_code": torch.tensor(hand_code, dtype=torch.int64),
        "hand_vis": torch.tensor(hand_vis, dtype=torch.float32),
        "folder_id_p": torch.tensor(p_ids, dtype=torch.int64),
        "folder_used_p": torch.tensor(p_used, dtype=torch.float32),
        "folder_mask_p": torch.tensor(p_mask, dtype=torch.bool),
        "folder_id_e": torch.tensor(e_ids, dtype=torch.int64),
        "folder_used_e": torch.tensor(e_used, dtype=torch.float32),
        "folder_mask_e": torch.tensor(e_mask, dtype=torch.bool),
        "held_id": torch.tensor(held_id, dtype=torch.int64),
        "held_code": torch.tensor(held_code, dtype=torch.int64),
        "held_mask": torch.tensor(held_mask, dtype=torch.bool),
        "used_cross": torch.tensor(used_cross, dtype=torch.float32),
        "active_cross_idx_p": torch.tensor(p_idx, dtype=torch.int64),
        "active_cross_idx_e": torch.tensor(e_idx, dtype=torch.int64),
        "beast_feats": beast_feats,
        "last_used_id_p": torch.tensor(last_p, dtype=torch.int64),
        "last_used_id_e": torch.tensor(last_e, dtype=torch.int64),
        "current_chip_p": torch.tensor(curr_p, dtype=torch.int64),
        "current_chip_e": torch.tensor(curr_e, dtype=torch.int64),
    }
    if device: out = {k: v.to(device) for k,v in out.items()}
    return out

def _stack_time(xs):
    out = {}
    for x in xs:
        for k, v in x.items():
            out.setdefault(k, []).append(v)
    return {k: torch.stack(vs, dim=0) for k, vs in out.items()}

def _pad_sequence_time(seq, target_len):
    if not seq: raise ValueError("empty seq")
    T = int(next(iter(seq.values())).shape[0])
    valid = torch.zeros((target_len,), dtype=torch.bool)
    n = min(T, target_len)
    valid[:n] = True
    if T >= target_len:
        return {k: v[:target_len] for k,v in seq.items()}, valid
    padded = {}
    for k, v in seq.items():
        pad_shape = (target_len - T,) + tuple(v.shape[1:])
        padded[k] = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)], dim=0)
    return padded, valid

def _cache_params_dict(stride, folder_len, require_cust_gt0, seq_len):
    return {
        "format": "critic_rl_td_lambda_v5", # BUMPED FOR ON-DECK/JUST-USED
        "stride": int(stride),
        "folder_len": int(folder_len),
        "require_cust_gt0": bool(require_cust_gt0),
        "seq_len": int(seq_len),
        "reward": "hp_delta_diff",
        "target": "td_lambda_bootstrap_in_train",
    }

def _atomic_torch_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except: pass

def _process_replay_task(args_pack):
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
        except: pass

    frames = _read_actions_jsonl(actions_path)
    if not frames: return f"[EMPTY] {name}"
    static = _read_json(static_path) or {}

    try: derived = compute_derived(frames, static)
    except Exception as e: return f"[ERR] {name}: {e}"

    hp_delta = _hp_delta_series(frames)
    rewards = _hp_delta_rewards(hp_delta)

    starts = []
    for fi in range(0, len(frames), stride):
        if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0: continue
        starts.append(fi)

    if not starts:
        _atomic_torch_save({
            "replay": name, "params": params,
            "start_frame": torch.empty((0,), dtype=torch.int64),
            "r": torch.empty((0, seq_len), dtype=torch.float32),
            "done": torch.empty((0, seq_len), dtype=torch.bool),
            "valid": torch.empty((0, seq_len), dtype=torch.bool),
            "x": {}
        }, cache_path)
        return f"[FILTERED] {name}"

    xs, rs, dones, valids = {}, [], [], []
    n_frames = len(frames)
    
    for start in starts:
        per_x, per_r, per_done = [], [], []
        for k in range(seq_len):
            raw_i = start + k * stride
            if raw_i >= n_frames: break
            di = min(raw_i, len(derived)-1) if derived else 0
            per_x.append(tensorize_frame(frames[raw_i], derived[di] if derived else {}, static, folder_len=folder_len))
            per_r.append(float(rewards[raw_i]))
            per_done.append(bool(raw_i >= n_frames-1))
        
        if not per_x: continue
        seq = _stack_time(per_x)
        seq, valid = _pad_sequence_time(seq, target_len=seq_len)
        r_t = torch.zeros((seq_len,), dtype=torch.float32)
        d_t = torch.zeros((seq_len,), dtype=torch.bool)
        for i in range(min(len(per_r), seq_len)): r_t[i] = per_r[i]
        for i in range(min(len(per_done), seq_len)): d_t[i] = per_done[i]
        if valid.any(): d_t[int(valid.nonzero(as_tuple=False)[-1].item())] = True

        for k, v in seq.items(): xs.setdefault(k, []).append(v)
        rs.append(r_t)
        dones.append(d_t)
        valids.append(valid)
    
    if not rs: return f"[EMPTY] {name}"
    
    x_stacked = {k: torch.stack(vs, dim=0) for k, vs in xs.items()}
    _atomic_torch_save({
        "replay": name, "params": params,
        "start_frame": torch.tensor(starts[:len(rs)], dtype=torch.int64),
        "r": torch.stack(rs, dim=0),
        "done": torch.stack(dones, dim=0),
        "valid": torch.stack(valids, dim=0),
        "x": x_stacked
    }, cache_path)
    return f"[DONE] {name}"

def build_rl_sequence_cache(*, dataset_dir, cache_dir, stride, folder_len, seq_len, require_cust_gt0, overwrite, num_workers):
    root = Path(dataset_dir)
    out_root = Path(cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    params = _cache_params_dict(stride=stride, folder_len=folder_len, require_cust_gt0=require_cust_gt0, seq_len=seq_len)
    
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p/"actions.jsonl").exists()]
    if not dirs: raise RuntimeError(f"No replays in {root}")
    
    print(f"[cache] Starting parallel build with {num_workers} workers...")
    t0 = time.time()
    
    args = [(str(d), str(out_root), params, stride, folder_len, seq_len, require_cust_gt0, overwrite) for d in dirs]
    
    with multiprocessing.Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(_process_replay_task, args, chunksize=1), total=len(dirs)): pass
    
    manifest = {"params": params, "num_replay_dirs": len(dirs), "built_at": time.time()}
    _atomic_torch_save(manifest, out_root / "_manifest.pt")
    print(f"[cache] Done in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------------

class CriticRLTDDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        *, 
        stride=6, 
        seq_len=16, 
        max_samples=None, 
        cache_replays=2, 
        folder_len=30, 
        split="train", 
        val_ratio=0.1, 
        seed=1337, 
        require_cust_gt0=True,
        boring_drop_rate=0.90  # NEW: Drop 90% of zero-reward sequences
    ):
        self.root = Path(root_dir)
        self.stride = stride
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.folder_len = folder_len
        self.cache = _ReplayCache(max_replays=cache_replays)
        self.require_cust_gt0 = require_cust_gt0
        
        all_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir() and (p/"actions.jsonl").exists()]
        rng = __import__("random").Random(seed)
        rng.shuffle(all_dirs)
        n_val = max(1, int(len(all_dirs)*val_ratio))
        
        # Split logic
        if split == "val":
            target_dirs = all_dirs[:n_val]
            # Don't drop boring samples in validation! We want honest metrics.
            self.boring_drop_rate = 0.0 
        else:
            target_dirs = all_dirs[n_val:]
            self.boring_drop_rate = boring_drop_rate

        self.metas = [ReplayMeta(d.name, d, 0, _read_json(d/"static_data.json") or {}) for d in target_dirs]
        
        self.index = []
        kept_boring = 0
        dropped_boring = 0
        kept_interesting = 0

        print(f"[{split}] Indexing and filtering (drop_rate={self.boring_drop_rate})...")
        
        for r_i, m in enumerate(tqdm(self.metas)):
            # Peek at rewards without fully parsing heavy derived state yet if possible,
            # but _ReplayCache does it all. That's fine, it's one-time init.
            # actually we can cheat and just parse actions for speed if needed, 
            # but let's stick to the cache pattern to be safe.
            frames, _, rewards = self.cache.get(meta=m)
            
            if not frames: continue
            
            n_frames = len(frames)
            
            for fi in range(0, n_frames, stride):
                # 1. Cust Gauge Filter
                if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0: 
                    continue
                
                # 2. "Boring" Filter (Lookahead)
                # Check if ANY reward happens in the next seq_len steps
                is_interesting = False
                end_idx = min(fi + (seq_len * stride), n_frames)
                
                # Scan the rewards covered by this sequence
                # rewards array is 1:1 with frames
                # We need to check frames [fi, fi+stride, ..., fi+(seq_len-1)*stride]
                for k in range(seq_len):
                    check_idx = fi + (k * stride)
                    if check_idx >= n_frames: break
                    if abs(rewards[check_idx]) > 0.001: # Epsilon check
                        is_interesting = True
                        break
                
                if is_interesting:
                    self.index.append((r_i, fi))
                    kept_interesting += 1
                else:
                    # It's boring. Roll dice.
                    if rng.random() >= self.boring_drop_rate:
                        self.index.append((r_i, fi))
                        kept_boring += 1
                    else:
                        dropped_boring += 1

                if max_samples and len(self.index) >= max_samples: break
            if max_samples and len(self.index) >= max_samples: break
            
        total = len(self.index)
        print(f"[{split}] Dataset Ready. Total: {total}")
        print(f"   Interesting (Kept): {kept_interesting}")
        print(f"   Boring (Kept):      {kept_boring}")
        print(f"   Boring (Dropped):   {dropped_boring}")
        if total > 0:
            print(f"   Signal Density:     {kept_interesting/total:.1%} (was <1%)")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        # ... (Same as before) ...
        r_i, start = self.index[idx]
        meta = self.metas[r_i]
        frames, derived, rewards = self.cache.get(meta)
        n = len(frames)
        per_x, per_r, per_done = [], [], []
        
        for k in range(self.seq_len):
            ri = start + k*self.stride
            if ri >= n: break
            di = min(ri, len(derived)-1) if derived else 0
            per_x.append(tensorize_frame(frames[ri], derived[di] if derived else {}, meta.static, folder_len=self.folder_len))
            per_r.append(float(rewards[ri]))
            per_done.append(bool(ri >= n-1))
            
        if not per_x: 
            return _stack_time([tensorize_frame(frames[0], derived[0], meta.static, folder_len=self.folder_len)]), torch.zeros(self.seq_len), torch.zeros(self.seq_len, dtype=torch.bool), torch.zeros(self.seq_len, dtype=torch.bool)

        seq = _stack_time(per_x)
        seq, valid = _pad_sequence_time(seq, self.seq_len)
        r_t = torch.zeros((self.seq_len,), dtype=torch.float32)
        d_t = torch.zeros((self.seq_len,), dtype=torch.bool)
        for i in range(min(len(per_r), self.seq_len)): r_t[i] = per_r[i]
        for i in range(min(len(per_done), self.seq_len)): d_t[i] = per_done[i]
        if valid.any(): d_t[int(valid.nonzero(as_tuple=False)[-1].item())] = True
        return seq, r_t, d_t, valid

# ---------------------------------------------------------------------------
# IN-MEMORY DATASET (The 256GB RAM Solution)
# ---------------------------------------------------------------------------

class InMemoryCriticRLTDDataset(Dataset):
    """
    Loads ALL cache files into a single huge CPU tensor dict at startup.
    Eliminates all disk I/O and worker overhead during training.
    """
    def __init__(
        self, 
        dataset_dir, 
        *, 
        cache_dir, 
        stride, 
        folder_len, 
        seq_len, 
        require_cust_gt0, 
        split="train", 
        val_ratio=0.1, 
        seed=1337, 
        max_samples=None,
        boring_drop_rate=0.90  # NEW
    ):
        self.cache_dir = Path(cache_dir)
        self.params = _cache_params_dict(stride=stride, folder_len=folder_len, require_cust_gt0=require_cust_gt0, seq_len=seq_len)
        
        # Split Logic
        if split == "val":
            self.boring_drop_rate = 0.0 # Keep everything for validation
        else:
            self.boring_drop_rate = float(boring_drop_rate)

        all_files = sorted([p for p in self.cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
        if not all_files: raise RuntimeError(f"No cache files in {cache_dir}")
        
        rng = __import__("random").Random(seed)
        names = [p.stem for p in all_files]
        rng.shuffle(names)
        n_val = max(1, int(len(names) * val_ratio))
        val_set = set(names[:n_val])
        files = [p for p in all_files if ((p.stem in val_set) if split == "val" else (p.stem not in val_set))]

        print(f"Loading {len(files)} files into RAM for {split} (drop_rate={self.boring_drop_rate})...")
        
        # Temporary lists to hold data from all files
        buffer_x: Dict[str, List[torch.Tensor]] = {}
        buffer_r: List[torch.Tensor] = []
        buffer_done: List[torch.Tensor] = []
        buffer_valid: List[torch.Tensor] = []
        
        total_loaded = 0
        kept_interesting = 0
        kept_boring = 0
        dropped_boring = 0
        
        # Load files sequentially (fast sequential read)
        for p in tqdm(files, desc=f"Loading {split} RAM"):
            try:
                ck = torch.load(p, map_location="cpu")
                # Basic check
                if ck.get("params") != self.params:
                    continue
                
                # Check rewards to filter
                rewards = ck["r"] # [N, SeqLen]
                n_rows = rewards.shape[0]
                if n_rows == 0: continue

                # Identify rows to keep
                keep_indices = []
                for i in range(n_rows):
                    # Check if sequence has ANY non-zero reward
                    is_interesting = (rewards[i].abs() > 0.001).any().item()
                    
                    if is_interesting:
                        keep_indices.append(i)
                        kept_interesting += 1
                    else:
                        if rng.random() >= self.boring_drop_rate:
                            keep_indices.append(i)
                            kept_boring += 1
                        else:
                            dropped_boring += 1
                
                if not keep_indices:
                    continue

                # Slice tensors
                idxs = torch.tensor(keep_indices, dtype=torch.long)
                
                for k, v in ck["x"].items():
                    buffer_x.setdefault(k, []).append(v[idxs])
                
                buffer_r.append(ck["r"][idxs])
                buffer_done.append(ck["done"][idxs])
                buffer_valid.append(ck["valid"][idxs])
                
                total_loaded += len(keep_indices)
                
                # Cleanup per file
                del ck
                if len(buffer_r) % 50 == 0:
                    gc.collect()

                if max_samples and total_loaded >= max_samples:
                    break
            except Exception:
                pass

        if total_loaded == 0:
            raise RuntimeError("No data loaded! Check cache generation or params.")

        print(f"Concatenating {total_loaded} samples...")
        
        # Concat everything into huge tensors
        self.x = {k: torch.cat(v_list, dim=0) for k, v_list in buffer_x.items()}
        self.r = torch.cat(buffer_r, dim=0)
        self.done = torch.cat(buffer_done, dim=0)
        self.valid = torch.cat(buffer_valid, dim=0)
        
        # Free temp lists
        del buffer_x, buffer_r, buffer_done, buffer_valid
        gc.collect()
        
        print(f"RAM Dataset Ready ({split}). Total: {total_loaded}")
        print(f"   Interesting: {kept_interesting}")
        print(f"   Boring Kept: {kept_boring}")
        print(f"   Dropped:     {dropped_boring}")
        if total_loaded > 0:
            print(f"   Signal Density: {kept_interesting/total_loaded:.1%}")

    def __len__(self) -> int:
        return self.r.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        x = {k: v[idx] for k, v in self.x.items()}
        return x, self.r[idx], self.done[idx], self.valid[idx]

    # --- NEW METHOD: Vectorized Batch Getter ---
    def get_batch(self, indices: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = indices.to(torch.int64)
        x = {k: v[indices] for k, v in self.x.items()}
        r = self.r[indices]
        done = self.done[indices]
        valid = self.valid[indices]
        return x, r, done, valid
CachedCriticRLTDDataset = InMemoryCriticRLTDDataset


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
    "CriticRLTDDataset",
    "InMemoryCriticRLTDDataset",
    "CachedCriticRLTDDataset",
    "build_rl_sequence_cache",
    "collate_batch",
]