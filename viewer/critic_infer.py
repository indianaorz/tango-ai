# viewer/critic_infer.py
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch

from critic_rl.model import CriticRLConfig, HPDeltaTDLambdaCritic
from critic_rl.dataset import (
    ACTION_DIM,
    STATE_SCALAR_DIM,
    tensorize_frame,
)

# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticResult:
    values_by_frame: List[Optional[float]]  # len == num_frames
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _extract_cust_gauge_series(frames: List[Dict[str, Any]]) -> List[int]:
    """
    Canonical source of truth for cust gauge: raw frame dict.
    This avoids relying on any scalar index / normalization assumptions.
    """
    out: List[int] = []
    for f in frames:
        out.append(_as_int(f.get("cust_gauge"), 0))
    return out


# -----------------------------------------------------------------------------
# Critic Runner
# -----------------------------------------------------------------------------

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
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.batch_seqs = int(batch_seqs)

        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Critic checkpoint not found: {p}")

        ckpt = torch.load(str(p), map_location="cpu")
        cfg_dict = ckpt.get("cfg", None) or {}
        cfg = CriticRLConfig(**cfg_dict) if isinstance(cfg_dict, dict) else CriticRLConfig()

        args = (ckpt.get("args", {}) or {}) if isinstance(ckpt.get("args", {}), dict) else {}
        folder_len = int(args.get("folder_len", 30))
        trained_seq_len = int(args.get("seq_len", 64))
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
            di = min(i, len(derived) - 1) if derived else 0
            d = derived[di] if derived else {}

            t_dict = tensorize_frame(
                frames[i],
                d,
                static,
                folder_len=self.folder_len,
                device=None,  # build on CPU, move once
            )
            for k, v in t_dict.items():
                all_tensors.setdefault(k, []).append(v)

        device_tensors: Dict[str, torch.Tensor] = {}
        for k, v_list in all_tensors.items():
            t = torch.stack(v_list, dim=0)
            device_tensors[k] = t.to(self.device, non_blocking=True)

        # Sanity check dimensions
        sc = device_tensors.get("scalars", None)
        ac = device_tensors.get("action", None)
        if sc is None or sc.ndim != 2 or int(sc.shape[1]) != int(STATE_SCALAR_DIM):
            got = tuple(sc.shape) if sc is not None else None
            raise ValueError(f"Expected scalars [N,{STATE_SCALAR_DIM}], got {got}")
        if ac is None or ac.ndim != 2 or int(ac.shape[1]) != int(ACTION_DIM):
            got = tuple(ac.shape) if ac is not None else None
            raise ValueError(f"Expected action [N,{ACTION_DIM}], got {got}")

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

        # Canonical gauge series (raw frames)
        cust_gauge_series = _extract_cust_gauge_series(frames)

        starts: List[int] = []
        for fi in range(0, num_frames, s):
            if use_filter and cust_gauge_series[fi] <= 0:
                continue
            starts.append(fi)

        out: List[Optional[float]] = [None] * num_frames
        if not starts:
            return CriticResult(values_by_frame=out, meta={**self.ckpt_meta, "note": "0 start frames"})

        all_frames_device: Optional[Dict[str, torch.Tensor]] = None
        cust_gauge_dev: Optional[torch.Tensor] = None

        try:
            all_frames_device = self._precompute_all_frames(frames, static, derived)

            # Build a device gauge tensor once (int32 is enough)
            cust_gauge_dev = torch.tensor(
                cust_gauge_series, device=self.device, dtype=torch.int32
            )

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )

            with torch.no_grad():
                for a in range(0, len(starts), self.batch_seqs):
                    b = min(len(starts), a + self.batch_seqs)
                    start_batch = starts[a:b]
                    B = len(start_batch)

                    base_starts = torch.tensor(start_batch, device=self.device, dtype=torch.long).unsqueeze(1)
                    offsets = (torch.arange(T, device=self.device, dtype=torch.long) * s).unsqueeze(0)

                    raw_indices = base_starts + offsets  # [B,T]

                    # Bounds mask
                    in_bounds = raw_indices < num_frames
                    clamped_indices = raw_indices.clamp(max=num_frames - 1)

                    # STRICT GAUGE FILTERING (raw cust_gauge > 0)
                    gauge_vals = cust_gauge_dev[clamped_indices]  # [B,T]
                    valid_mask = in_bounds & (gauge_vals > 0)

                    flat_indices = clamped_indices.reshape(-1)

                    xb: Dict[str, torch.Tensor] = {}
                    for k, big_t in all_frames_device.items():
                        dims = big_t.shape[1:]
                        gathered = big_t[flat_indices]
                        xb[k] = gathered.view(B, T, *dims)

                    with amp_ctx:
                        v = self.model(xb)

                    v_cpu = v.detach().to("cpu", non_blocking=False)
                    valid_cpu = valid_mask.to("cpu", non_blocking=False)
                    raw_indices_cpu = raw_indices.to("cpu", non_blocking=False)

                    self._scatter_frame_values(out, raw_indices_cpu, valid_cpu, v_cpu)

        finally:
            del all_frames_device
            del cust_gauge_dev
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        meta = {
            **self.ckpt_meta,
            "mode": "infer_from_derived",
            "num_frames": num_frames,
            "stride": s,
            "seq_len": T,
            "require_cust_gt0": bool(use_filter),
            "coverage_frames": int(sum(1 for x in out if x is not None)),
        }
        return CriticResult(values_by_frame=out, meta=meta)

    @staticmethod
    def _scatter_frame_values(
        out: List[Optional[float]],
        frame_idx: torch.Tensor,
        valid: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        f_flat = frame_idx.flatten().numpy()
        m_flat = valid.flatten().numpy()
        val_flat = v.flatten().numpy()

        for i in range(len(f_flat)):
            if not m_flat[i]:
                continue
            idx = int(f_flat[i])
            if 0 <= idx < len(out):
                # Only write if None: earliest-start window wins (max context).
                if out[idx] is None:
                    out[idx] = float(val_flat[i])


__all__ = ["CriticRunner", "CriticResult"]
