from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gc
import torch
import numpy as np

# Assumes training code is in python path
from critic_rl.model import CriticRLConfig, HPDeltaTDLambdaCritic


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticResult:
    values_by_frame: List[Optional[float]]  # len == num_frames
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers (Matches critic_rl.dataset v6)
# ---------------------------------------------------------------------------

INVALID_CHIP_IDS = {255, 65535}

BUTTON_KEYS = [
    'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT',
    'START', 'BACK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'EAST', 'SOUTH'
]

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

def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask

def _tensorize_frame_v5(
    frame: Dict[str, Any],
    d: Dict[str, Any],
    static: Dict[str, Any],
    *,
    folder_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Exact replica of critic_rl.dataset.tensorize_frame (v6 with buttons).
    """
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

    # --- Scalars (Size 48 for Buttons) ---
    scalars = torch.zeros(48, dtype=torch.float32)
    scalars[0] = float(p_hp) / 1000.0
    scalars[1] = float(e_hp) / 1000.0
    
    # FIX: Charge is 0,1,2 -> Normalize by 2.0
    scalars[2] = float(p_chg) / 2.0
    scalars[3] = float(e_chg) / 2.0
    
    scalars[4] = float(cust) / 100.0
    scalars[5] = float(inside)
    scalars[6] = float(turn_idx) / 50.0
    scalars[7] = px
    scalars[8] = py
    scalars[9] = ex
    scalars[10] = ey
    
    # --- Derived Scalars (Window Commit & Events) ---
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
        # Handle list format if present
        if isinstance(val, list):
            val = val[0]
        scalars[current_idx] = float(val)
        current_idx += 1

    # --- Grid ---
    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))
    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0]*(18-len(grid_state))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2]*(18-len(grid_owner))

    # --- Hand ---
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

    p_used_mask_raw = _as_list(p_derived.get("folder_used_mask"))
    e_used_mask_raw = _as_list(e_derived.get("folder_used_mask"))
    
    p_used = [1.0 if bool(p_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(p_used_mask_raw)))] + [0.0]*(folder_len-len(p_used_mask_raw))
    e_used = [1.0 if bool(e_used_mask_raw[i]) else 0.0 for i in range(min(folder_len, len(e_used_mask_raw)))] + [0.0]*(folder_len-len(e_used_mask_raw))

    # --- Held ---
    held = _as_list(p_derived.get("held_chips"))
    held_id = [0]*5
    held_code = [0]*5
    held_mask = [False]*5
    for i in range(min(5, len(held))):
        h = held[i] if isinstance(held[i], dict) else {}
        held_id[i] = _chip_id_norm(h.get("id"))
        held_code[i] = _code_norm(h.get("code"))
        held_mask[i] = True

    # --- Cross & Beast ---
    p_used_cross = _as_list(p_derived.get("used_cross_mask"))
    e_used_cross = _as_list(e_derived.get("used_cross_mask"))
    used_cross = [0.0]*22
    for i in range(min(11, len(p_used_cross))): used_cross[i] = 1.0 if bool(p_used_cross[i]) else 0.0
    for i in range(min(11, len(e_used_cross))): used_cross[11+i] = 1.0 if bool(e_used_cross[i]) else 0.0

    p_active = p_derived.get("active_cross") or None
    e_active = e_derived.get("active_cross") or None
    p_idx = _as_int(p_active.get("idx"), 11) if isinstance(p_active, dict) else 11
    e_idx = _as_int(e_active.get("idx"), 11) if isinstance(e_active, dict) else 11

    p_beast = p_derived.get("beast") or {}
    e_beast = e_derived.get("beast") or {}
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

    return {
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


# ---------------------------------------------------------------------------
# Critic Runner
# ---------------------------------------------------------------------------

class CriticRunner:
    def __init__(
        self,
        *,
        ckpt_path: str,
        device: str = "cuda",
        use_amp: bool = True,
        batch_seqs: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.batch_seqs = int(batch_seqs)

        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Critic checkpoint not found: {p}")

        ckpt = torch.load(str(p), map_location="cpu")
        cfg_dict = ckpt.get("cfg", None) or {}
        cfg = CriticRLConfig(**cfg_dict) if isinstance(cfg_dict, dict) else CriticRLConfig()

        args = (ckpt.get("args", {}) or {}) if isinstance(ckpt.get("args", {}), dict) else {}
        folder_len = int(args.get("folder_len", 30))
        trained_seq_len = int(args.get("seq_len", 16))
        trained_require_cust_gt0 = bool(args.get("require_cust_gt0", False))

        model = HPDeltaTDLambdaCritic(cfg, folder_len=folder_len)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval().to(self.device)

        self.model = model
        self.cfg = cfg
        self.folder_len = folder_len
        self.trained_seq_len = trained_seq_len
        self.trained_require_cust_gt0 = trained_require_cust_gt0

        self.ckpt_meta = {
            "ckpt_path": str(p),
            "cfg": cfg_dict if isinstance(cfg_dict, dict) else {},
            "epoch": ckpt.get("epoch", None),
            "global_step": ckpt.get("global_step", None),
            "args": args,
        }

    def _precompute_all_frames(
        self,
        frames: List[Dict[str, Any]],
        static: Dict[str, Any],
        derived: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert ALL frames to tensors at once.
        Returns dict: {key: Tensor[N_FRAMES, ...]} on DEVICE.
        """
        n_frames = len(frames)
        all_tensors: Dict[str, List[torch.Tensor]] = {}

        for i in range(n_frames):
            di = derived[i] if i < len(derived) else {}
            t_dict = _tensorize_frame_v5(
                frames[i],
                di,
                static,
                folder_len=self.folder_len
            )
            for k, v in t_dict.items():
                all_tensors.setdefault(k, []).append(v)

        device_tensors = {}
        for k, v_list in all_tensors.items():
            t = torch.stack(v_list, dim=0)
            device_tensors[k] = t.to(self.device, non_blocking=True)
        
        return device_tensors

    def infer_from_derived(
        self,
        *,
        frames: List[Dict[str, Any]],
        static: Optional[Dict[str, Any]],
        derived: List[Dict[str, Any]],
        stride: int,
        seq_len: int,
        require_cust_gt0: Optional[bool] = None,
    ) -> CriticResult:
        static = static or {}
        num_frames = int(len(frames))
        if num_frames <= 0:
            return CriticResult(values_by_frame=[], meta={**self.ckpt_meta, "note": "empty replay"})

        s = int(stride)
        T = int(seq_len)
        use_filter = self.trained_require_cust_gt0 if require_cust_gt0 is None else bool(require_cust_gt0)

        starts: List[int] = []
        for fi in range(0, num_frames, s):
            if use_filter and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                continue
            starts.append(fi)

        out: List[Optional[float]] = [None] * num_frames
        if not starts:
            return CriticResult(values_by_frame=out, meta={**self.ckpt_meta, "note": "0 start frames"})

        all_frames_device = None
        
        try:
            # 1. Precompute [N, ...]
            all_frames_device = self._precompute_all_frames(frames, static, derived)

            with torch.no_grad():
                for a in range(0, len(starts), self.batch_seqs):
                    b = min(len(starts), a + self.batch_seqs)
                    start_batch = starts[a:b]
                    B = len(start_batch)

                    base_starts = torch.tensor(start_batch, device=self.device, dtype=torch.long).unsqueeze(1)
                    offsets = (torch.arange(T, device=self.device, dtype=torch.long) * s).unsqueeze(0)
                    
                    raw_indices = base_starts + offsets
                    valid_mask = raw_indices < num_frames
                    clamped_indices = raw_indices.clamp(max=num_frames - 1)

                    xb = {}
                    flat_indices = clamped_indices.view(-1)
                    
                    for k, big_t in all_frames_device.items():
                        dims = big_t.shape[1:]
                        gathered = big_t[flat_indices]
                        xb[k] = gathered.view(B, T, *dims)

                    with torch.autocast(device_type=self.device.type, dtype=torch.float16) if self.use_amp else torch.no_grad():
                        v = self.model(xb)  # [B,T]

                    # Scatter output
                    v_cpu = v.detach().to("cpu", non_blocking=False)
                    valid_cpu = valid_mask.to("cpu", non_blocking=False)
                    raw_indices_cpu = raw_indices.to("cpu", non_blocking=False)
                    
                    self._scatter_frame_values(out, raw_indices_cpu, valid_cpu, v_cpu)
        
        finally:
            del all_frames_device
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        meta = {
            **self.ckpt_meta,
            "mode": "infer_from_derived_v5",
            "num_frames": num_frames,
            "stride": s,
            "seq_len": T,
            "coverage_frames": int(sum(1 for x in out if x is not None)),
        }
        return CriticResult(values_by_frame=out, meta=meta)

    def _scatter_frame_values(self, out, frame_idx, valid, v):
        f_flat = frame_idx.flatten().numpy()
        v_flat = valid.flatten().numpy()
        val_flat = v.flatten().numpy()
        
        for i in range(len(f_flat)):
            if v_flat[i]:
                idx = f_flat[i]
                if 0 <= idx < len(out):
                    out[idx] = float(val_flat[i])

__all__ = ["CriticRunner", "CriticResult"]