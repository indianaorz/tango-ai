from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Assuming your training code is importable as a package named 'critic_rl'
from critic_rl.model import CriticRLConfig, HPDeltaTDLambdaCritic


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticResult:
    values_by_frame: List[Optional[float]]  # len == num_frames
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Null context (for autocast off)
# ---------------------------------------------------------------------------

class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Helpers (MUST match critic_rl.dataset behavior)
# ---------------------------------------------------------------------------

INVALID_CHIP_IDS = {255, 65535}


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
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
    x = _as_int(v, 2)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 2


def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask


def _clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ---------------------------------------------------------------------------
# Tensorization
# ---------------------------------------------------------------------------

def _tensorize_frame_like_training(
    frame: Dict[str, Any],
    d: Dict[str, Any],
    static: Dict[str, Any],
    *,
    folder_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Mirrors critic_rl.dataset.tensorize_frame exactly (for inference).
    Returns tensors on CPU.
    """
    p_hp = _as_int(frame.get("player_health"), 0)
    e_hp = _as_int(frame.get("enemy_health"), 0)
    p_chg = _as_int(frame.get("player_charge"), 0)
    e_chg = _as_int(frame.get("enemy_charge"), 0)
    cust = _as_int(frame.get("cust_gauge"), 0)
    inside = 1.0 if _as_bool(frame.get("inside_window"), False) else 0.0
    turn_idx = _as_int(d.get("turn_index"), 0)

    px, py = _pos_norm(frame.get("player_pos"))
    ex, ey = _pos_norm(frame.get("enemy_pos"))

    scalars = torch.zeros(32, dtype=torch.float32)
    scalars[0] = float(p_hp) / 1000.0
    scalars[1] = float(e_hp) / 1000.0
    scalars[2] = float(p_chg) / 100.0
    scalars[3] = float(e_chg) / 100.0
    scalars[4] = float(cust) / 100.0
    scalars[5] = float(inside)
    scalars[6] = float(turn_idx) / 50.0
    scalars[7] = float(px) / 5.0
    scalars[8] = float(py) / 5.0
    scalars[9] = float(ex) / 5.0
    scalars[10] = float(ey) / 5.0

    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))
    gs = [0] * 18
    go = [2] * 18
    for i in range(min(18, len(grid_state))):
        gs[i] = _tile_norm(grid_state[i])
    for i in range(min(18, len(grid_owner))):
        go[i] = _owner_norm(grid_owner[i])

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

    p_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("player_folder_ids"))]
    e_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("enemy_folder_ids"))]
    p_ids, p_mask = _pad_ids(p_folder_ids, folder_len)
    e_ids, e_mask = _pad_ids(e_folder_ids, folder_len)

    p_used_mask_raw = _as_list((d.get("player", {}) or {}).get("folder_used_mask"))
    e_used_mask_raw = _as_list((d.get("enemy", {}) or {}).get("folder_used_mask"))

    p_used = [0.0] * folder_len
    e_used = [0.0] * folder_len
    for i in range(min(folder_len, len(p_used_mask_raw))):
        p_used[i] = 1.0 if bool(p_used_mask_raw[i]) else 0.0
    for i in range(min(folder_len, len(e_used_mask_raw))):
        e_used[i] = 1.0 if bool(e_used_mask_raw[i]) else 0.0

    held = _as_list((d.get("player", {}) or {}).get("held_chips"))
    held_id = [0] * 5
    held_code = [0] * 5
    held_mask = [False] * 5
    for i in range(min(5, len(held))):
        h = held[i] if isinstance(held[i], dict) else {}
        held_id[i] = _chip_id_norm(h.get("id"))
        held_code[i] = _code_norm(h.get("code"))
        held_mask[i] = True

    p_used_cross = _as_list((d.get("player", {}) or {}).get("used_cross_mask"))
    e_used_cross = _as_list((d.get("enemy", {}) or {}).get("used_cross_mask"))
    used_cross = [0.0] * 22
    for i in range(min(11, len(p_used_cross))):
        used_cross[i] = 1.0 if bool(p_used_cross[i]) else 0.0
    for i in range(min(11, len(e_used_cross))):
        used_cross[11 + i] = 1.0 if bool(e_used_cross[i]) else 0.0

    p_active = (d.get("player", {}) or {}).get("active_cross") or None
    e_active = (d.get("enemy", {}) or {}).get("active_cross") or None
    p_idx = _as_int(p_active.get("idx"), 11) if isinstance(p_active, dict) else 11
    e_idx = _as_int(e_active.get("idx"), 11) if isinstance(e_active, dict) else 11

    # Safety clamp
    p_idx = _clamp_int(p_idx, 0, 10_000_000)
    e_idx = _clamp_int(e_idx, 0, 10_000_000)

    p_beast = (d.get("player", {}) or {}).get("beast") or {}
    e_beast = (d.get("enemy", {}) or {}).get("beast") or {}
    p_b_active = 1.0 if bool(p_beast.get("active")) else 0.0
    e_b_active = 1.0 if bool(e_beast.get("active")) else 0.0
    p_b_ever = 1.0 if bool(p_beast.get("ever")) else 0.0
    e_b_ever = 1.0 if bool(e_beast.get("ever")) else 0.0
    p_ts = p_beast.get("turns_since")
    e_ts = e_beast.get("turns_since")
    p_ts_f = float(_as_int(p_ts, 999)) / 50.0 if p_ts is not None else 9.99
    e_ts_f = float(_as_int(e_ts, 999)) / 50.0 if e_ts is not None else 9.99

    beast_feats = torch.tensor([p_b_active, p_ts_f, p_b_ever, e_b_active, e_ts_f, e_b_ever], dtype=torch.float32)

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
        trained_stride = int(args.get("stride", 6))
        trained_seq_len = int(args.get("seq_len", 16))
        trained_require_cust_gt0 = bool(args.get("require_cust_gt0", False))

        model = HPDeltaTDLambdaCritic(cfg, folder_len=folder_len)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval().to(self.device)

        self.model = model
        self.cfg = cfg
        self.folder_len = folder_len
        self.trained_stride = trained_stride
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

        # 1. CPU Loop: Create individual tensors (unavoidable overhead, but done once)
        # To optimize this further, we'd need to batch the dictionary lookups, but this is usually fast enough.
        for i in range(n_frames):
            di = derived[i] if i < len(derived) else {}
            t_dict = _tensorize_frame_like_training(
                frames[i],
                di,
                static,
                folder_len=self.folder_len
            )
            for k, v in t_dict.items():
                all_tensors.setdefault(k, []).append(v)

        # 2. Stack and move to Device
        device_tensors = {}
        for k, v_list in all_tensors.items():
            # stack [N, ...]
            t = torch.stack(v_list, dim=0)
            # move to device once
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

        # Filter start frames
        starts: List[int] = []
        for fi in range(0, num_frames, s):
            if use_filter and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                continue
            starts.append(fi)

        out: List[Optional[float]] = [None] * num_frames
        if not starts:
            meta = {
                **self.ckpt_meta,
                "mode": "infer_from_derived",
                "note": "0 start frames after filtering",
                "num_frames": num_frames,
                "stride": s,
                "seq_len": T,
                "nseq": 0,
                "coverage_frames": 0,
            }
            return CriticResult(values_by_frame=out, meta=meta)

        # 1. Precompute ALL frames onto GPU [N, ...]
        # This replaces the Python inner-loop tensorization.
        all_frames_device = self._precompute_all_frames(frames, static, derived)

        with torch.no_grad():
            for a in range(0, len(starts), self.batch_seqs):
                b = min(len(starts), a + self.batch_seqs)
                start_batch = starts[a:b]
                B = len(start_batch)

                # 2. Vectorized Batch Construction on GPU
                # We need [B, T] indices into all_frames_device
                # indices[b, t] = start_batch[b] + t * stride
                
                # Base starts: [B, 1]
                base_starts = torch.tensor(start_batch, device=self.device, dtype=torch.long).unsqueeze(1)
                
                # Offsets: [1, T] (0, s, 2s, ...)
                offsets = (torch.arange(T, device=self.device, dtype=torch.long) * s).unsqueeze(0)
                
                # Indices: [B, T]
                raw_indices = base_starts + offsets
                
                # Mask out-of-bounds indices (beyond n_frames)
                valid_mask = raw_indices < num_frames
                
                # Clamp indices to valid range for gathering (will mask output later)
                clamped_indices = raw_indices.clamp(max=num_frames - 1)

                # Gather batch: for each key, index select
                xb = {}
                for k, big_t in all_frames_device.items():
                    # big_t: [N, D...]
                    # clamped_indices: [B, T]
                    # we want: [B, T, D...]
                    
                    # Flatten indices to gather, then reshape
                    # shape of big_t is (N, *dims)
                    # output shape should be (B, T, *dims)
                    
                    dims = big_t.shape[1:]
                    flat_indices = clamped_indices.view(-1) # [B*T]
                    
                    # gather relies on the first dim
                    gathered = big_t[flat_indices] # [B*T, *dims]
                    
                    # reshape back to [B, T, *dims]
                    xb[k] = gathered.view(B, T, *dims)

                # 3. Model Forward
                amp_ctx = (
                    torch.autocast(device_type=self.device.type, dtype=torch.float16)
                    if (self.use_amp and self.device.type == "cuda")
                    else _NullCtx()
                )
                with amp_ctx:
                    v = self.model(xb)  # [B,T]

                v_cpu = v.detach().to("cpu", non_blocking=False)
                valid_cpu = valid_mask.to("cpu", non_blocking=False)
                
                # 4. Scatter results
                # Only valid time steps (within video bounds) are written
                # We reuse the python list scattering logic but passing tensor indices
                # raw_indices is on GPU, move to CPU for scattering loop
                raw_indices_cpu = raw_indices.to("cpu", non_blocking=False)
                
                self._scatter_frame_values(out, raw_indices_cpu, valid_cpu, v_cpu)

        meta = {
            **self.ckpt_meta,
            "mode": "infer_from_derived",
            "num_frames": num_frames,
            "stride": s,
            "seq_len": T,
            "nseq": len(starts),
            "coverage_frames": int(sum(1 for x in out if x is not None)),
        }
        return CriticResult(values_by_frame=out, meta=meta)

    def _tensorize_batch(self, *args, **kwargs):
        # Legacy method, replaced by vectorization above
        pass

    def _scatter_frame_values(
        self,
        out: List[Optional[float]],
        frame_idx: torch.Tensor,
        valid: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        B, T = frame_idx.shape
        # Convert to numpy for faster iteration if needed, but simple loop is fine
        # frame_idx, valid, v are CPU tensors
        
        # Optimization: Flatten everything to simple lists/arrays
        f_flat = frame_idx.flatten().numpy()
        v_flat = valid.flatten().numpy()
        val_flat = v.flatten().numpy()
        
        # We can perform a bulk update if indices are unique, 
        # but since seqs overlap, order matters (later overwrites earlier? or avg?)
        # Standard logic: overwrite is fine, or first write. 
        # With stride=1, every frame is the 'start' of a sequence (t=0).
        # It is also t=1 of the previous sequence.
        # The model predicts V(st). Theoretically V(st) should be similar regardless of being t=0 or t=5 in a seq.
        # However, to be deterministic and match "prediction at time t", we usually prefer the value where t=0 (most context future) or just overwrite.
        # Simple overwrite:
        
        for i in range(len(f_flat)):
            if v_flat[i]:
                idx = f_flat[i]
                if 0 <= idx < len(out):
                    out[idx] = float(val_flat[i])


__all__ = ["CriticRunner", "CriticResult"]