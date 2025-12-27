# train_rl.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nitrogen.mm_tokenizers import NitrogenTokenizer

# Shared training / checkpoint utilities (your existing module)
import train_utils as U

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,           # number of anchors per optimizer step
    "lr": 1e-4,
    "epochs": 20,
    "save_every": 5000,
    "num_workers": 8,
    "max_keep_ckpts": 3,

    # Base battle model to start from (offline RL only fine-tunes battle)
    # Override via --base_ckpt or --resume / auto-resume directory.
    "base_ckpt": "checkpoints/battle/step_150000.pt",
}

# Cached RL segments from precache_split_rl.py
DEFAULT_RL_CACHE_DIR = "data/nitrogen_rl_cache"

# Where to store offline RL checkpoints/logs (battle-only)
DEFAULT_CKPT_DIR = "checkpoints/rl_battle"
DEFAULT_LOG_DIR = "logs/rl_battle"


# -----------------------------------------------------------------------------
# Dataset: RL segment cache -> anchor samples
# -----------------------------------------------------------------------------
def _safe_load_pt(path: Path) -> Dict[str, Any]:
    """
    torch.load wrapper that tries mmap first (fast) then falls back.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # Older torch may not support mmap kwarg
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _clamp_f(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _window_indices_end(idx: int, T: int, n: int) -> List[int]:
    """
    Indices for a window of length T ending at idx (inclusive), clamped to [0..n-1].
    """
    idx = max(0, min(int(idx), int(n) - 1))
    start = idx - (T - 1)
    return [max(0, min(start + t, n - 1)) for t in range(T)]


class CachedRLSegmentDataset(Dataset):
    """
    Treat all cached RL segment files as one long stream of anchor indices.
    Each item:
      - frames: (V, 3, 256, 256) float in [-1,1]
      - j_left:  (H, 2)
      - j_right: (H, 2)
      - buttons: (H, B)
      - dropped_frames: (V,) bool (always zeros)
      - weight: scalar float32 (sample weight for offline RL)
      - game: "bn6"
      - meta: optional small dict for debugging
    """

    def __init__(
        self,
        root_dir: str,
        *,
        vision_horizon: int,
        base_seed: int = 42,
        weight_after_event: float = 1.0,
        weight_cap: float = 5.0,
        min_weight: float = 0.05,
        filter_actor: str = "both",        # "player" | "enemy" | "both"
        filter_outcome: str = "both",      # "good" | "bad" | "both"
    ):
        self.root = Path(root_dir)
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.root}")

        self.vision_horizon = int(vision_horizon)
        self.base_seed = int(base_seed) if base_seed > 0 else 0
        self.weight_after_event = float(weight_after_event)
        self.weight_cap = float(weight_cap)
        self.min_weight = float(min_weight)
        self.filter_actor = str(filter_actor)
        self.filter_outcome = str(filter_outcome)

        # Indexing: treat each timestep in each segment as an addressable item.
        self._valid_files: List[Path] = []
        self._file_lengths: List[int] = []
        self._cumulative: List[int] = []
        self._file_meta: List[Dict[str, Any]] = []

        total = 0
        for f in tqdm(self.files, desc="Indexing RL cache"):
            try:
                d = _safe_load_pt(f)
                frames = d.get("frames", None)
                actions = d.get("actions", None)
                meta = d.get("meta", {}) or {}

                if frames is None or actions is None:
                    continue
                n = int(frames.shape[0])
                if n <= 0:
                    continue

                # Basic sanity: actions should align with frames on the first dim
                if int(actions.shape[0]) != n:
                    continue

                # Optional filtering by actor/outcome at file-level (cheap + reduces data skew)
                actor = str(meta.get("actor", ""))
                outcome = str(meta.get("outcome", ""))

                if self.filter_actor != "both" and actor != self.filter_actor:
                    continue
                if self.filter_outcome != "both" and outcome != self.filter_outcome:
                    continue

                self._valid_files.append(f)
                self._file_lengths.append(n)
                total += n
                self._cumulative.append(total)

                # Keep only small meta fields we need for weighting
                kept_meta = {
                    "replay": str(meta.get("replay", "")),
                    "actor": actor,
                    "kind": str(meta.get("kind", "")),
                    "label": str(meta.get("label", "")),
                    "outcome": outcome,
                    "reason": str(meta.get("reason", "")),
                    "weight": float(meta.get("weight", 1.0)),
                    "start_frame": int(meta.get("start_frame", 0)),
                    "event_frame": int(meta.get("event_frame", -1)),
                    "end_frame": int(meta.get("end_frame", n - 1)),
                    # action_indices may exist; helps us locate event within segment
                    "action_indices": meta.get("action_indices", None),
                }
                self._file_meta.append(kept_meta)

            except Exception:
                continue

        if not self._valid_files or total <= 0:
            raise RuntimeError(f"No valid RL cached segments found in {self.root}")

        self.files = self._valid_files  # replace with filtered/validated list

        # Cache last-loaded file for speed
        self._cache_file_idx = -1
        self._cache_data: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        # bisect_right implemented inline to avoid extra import
        gi = int(global_idx)
        lo, hi = 0, len(self._cumulative)
        while lo < hi:
            mid = (lo + hi) // 2
            if gi < self._cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        file_idx = lo
        prev = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        local_idx = gi - prev
        return file_idx, int(local_idx)

    def _load_file(self, file_idx: int) -> None:
        self._cache_data = _safe_load_pt(self.files[file_idx])
        self._cache_file_idx = int(file_idx)

    def _event_local_index(self, meta: Dict[str, Any], n: int) -> int:
        """
        Try to locate the event_frame within the segment indices.
        Falls back to the center-ish of the segment if not found.
        """
        event_frame = int(meta.get("event_frame", -1))
        action_indices = meta.get("action_indices", None)

        if isinstance(action_indices, list) and action_indices:
            # action_indices is list of global indices aligned to segment positions
            try:
                # Find the exact match (fast path)
                pos = action_indices.index(event_frame)
                return int(max(0, min(pos, n - 1)))
            except Exception:
                pass

        # If not available, best-effort: assume event is within [0..n-1] at its clamped position
        # Many segments are [start..end] with contiguous indices, so this often works.
        sf = int(meta.get("start_frame", 0))
        if event_frame >= 0 and event_frame >= sf:
            return int(max(0, min(event_frame - sf, n - 1)))

        return int(max(0, min(n // 2, n - 1)))

    def _sample_weight(self, meta: Dict[str, Any], local_idx: int, n: int) -> float:
        """
        Offline RL weighting heuristic:
          - Use meta["weight"] for frames up to and including the event moment.
          - For frames after the event, scale down to weight_after_event (often 1.0 = neutral BC).
          - Clamp to [min_weight, weight_cap].
        """
        base_w = float(meta.get("weight", 1.0))
        event_local = self._event_local_index(meta, n)

        if local_idx <= event_local:
            w = base_w
        else:
            # after-event frames: usually less informative for "execution"
            w = float(self.weight_after_event)

        return _clamp_f(w, self.min_weight, self.weight_cap)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        file_idx, local_idx = self._locate(global_idx)

        if self._cache_file_idx != file_idx:
            self._load_file(file_idx)

        assert self._cache_data is not None
        d = self._cache_data

        frames_u8: torch.Tensor = d["frames"]  # (N, 3, 256, 256) uint8
        actions: torch.Tensor = d["actions"]   # (N, H, ACTION_DIM) float32
        meta = self._file_meta[file_idx]
        n = int(frames_u8.shape[0])

        anchor = int(max(0, min(local_idx, n - 1)))

        V = int(self.vision_horizon)
        vis_ids = _window_indices_end(anchor, V, n)

        # Normalize frames to [-1, 1]
        frames = frames_u8[vis_ids].clone().float().div(255.0).mul(2.0).sub(1.0)

        # Action window at this anchor
        aw = actions[anchor].clone().float()  # (H, ACTION_DIM)

        out = {
            "frames": frames,
            "j_left": aw[:, 0:2],
            "j_right": aw[:, 2:4],
            "buttons": aw[:, 4:],
            "dropped_frames": torch.zeros(V, dtype=torch.bool),
            "weight": torch.tensor(self._sample_weight(meta, anchor, n), dtype=torch.float32),
            "game": "bn6",
            # Keep meta light; helpful for debugging
            "meta": {
                "file": self.files[file_idx].name,
                "replay": meta.get("replay", ""),
                "actor": meta.get("actor", ""),
                "kind": meta.get("kind", ""),
                "label": meta.get("label", ""),
                "outcome": meta.get("outcome", ""),
                "reason": meta.get("reason", ""),
                "event_frame": int(meta.get("event_frame", -1)),
            },
        }
        return out


# -----------------------------------------------------------------------------
# Training loop (reward-weighted regression / weighted BC)
# -----------------------------------------------------------------------------
def _auto_resume_latest(ckpt_dir: Path) -> Optional[str]:
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def _extract_step_from_path(p: str) -> int:
    try:
        name = Path(p).stem
        if name.startswith("step_"):
            return int(name.split("_")[1])
    except Exception:
        pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL fine-tuning on cached RL segments (battle base model only).")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_RL_CACHE_DIR, help="RL cache directory (*.pt).")
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR, help="Checkpoint output dir.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="TensorBoard log dir.")

    parser.add_argument("--resume", type=str, default="", help="Path to resume checkpoint (overrides auto-resume).")
    parser.add_argument("--base_ckpt", type=str, default=BASE_CONFIG["base_ckpt"], help="Base battle checkpoint to start from.")
    parser.add_argument("--overwrite_resume", action="store_true", help="If set, do not auto-resume; always start from --base_ckpt unless --resume is provided.")

    parser.add_argument("--batch_size", type=int, default=BASE_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=BASE_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"])
    parser.add_argument("--save_every", type=int, default=BASE_CONFIG["save_every"])
    parser.add_argument("--num_workers", type=int, default=BASE_CONFIG["num_workers"])
    parser.add_argument("--max_keep_ckpts", type=int, default=BASE_CONFIG["max_keep_ckpts"])

    # RL sampling knobs
    parser.add_argument("--filter_actor", choices=["player", "enemy", "both"], default="both")
    parser.add_argument("--filter_outcome", choices=["good", "bad", "both"], default="both")
    parser.add_argument("--weight_after_event", type=float, default=1.0, help="Weight for frames after event (default neutral).")
    parser.add_argument("--weight_cap", type=float, default=5.0)
    parser.add_argument("--min_weight", type=float, default=0.05)

    # Determinism
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(BASE_CONFIG["device"])
    U.set_seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Offline RL Training (battle base)")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Checkpoints: {ckpt_dir}")
    print(f"   Logs: {log_dir}")
    print(f"   Actor filter: {args.filter_actor} | Outcome filter: {args.filter_outcome}")

    # -------------------------------------------------------------------------
    # Load model: resume > auto-resume (unless overwrite_resume) > base_ckpt
    # -------------------------------------------------------------------------
    load_path: str
    if args.resume:
        load_path = args.resume
        print(f"🔄 Resuming from explicit checkpoint: {load_path}")
    else:
        if args.overwrite_resume:
            load_path = args.base_ckpt
            print(f"🌱 Starting fresh (overwrite_resume): {load_path}")
        else:
            latest = _auto_resume_latest(ckpt_dir)
            if latest:
                load_path = latest
                print(f"🔄 Auto-resuming latest in ckpt_dir: {load_path}")
            else:
                load_path = args.base_ckpt
                print(f"🌱 Starting from base battle checkpoint: {load_path}")

    loaded = U.load_ng_checkpoint_faithful(load_path, device)

    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.train()

    model = loaded.model
    model.train()

    # Freeze vision tower (keep consistent with your supervised training)
    for name, p in model.named_parameters():
        if "vision" in name or "siglip" in name:
            p.requires_grad = False

    # -------------------------------------------------------------------------
    # Dataset / loader
    # -------------------------------------------------------------------------
    dataset = CachedRLSegmentDataset(
        root_dir=str(dataset_dir),
        vision_horizon=int(getattr(tokenizer, "vision_horizon", 1)),
        base_seed=int(args.seed),
        weight_after_event=float(args.weight_after_event),
        weight_cap=float(args.weight_cap),
        min_weight=float(args.min_weight),
        filter_actor=str(args.filter_actor),
        filter_outcome=str(args.filter_outcome),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=8 if int(args.num_workers) > 0 else None,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    writer = SummaryWriter(str(log_dir))

    # Step init from checkpoint name (best-effort)
    step = _extract_step_from_path(load_path)
    print(f"🔥 Training Loop Start (Continuing from Step {step})")

    # -------------------------------------------------------------------------
    # Training: reward-weighted regression / weighted BC
    #
    # Note: NitroGen returns a single scalar loss for the provided batch. To apply
    # per-sample RL weights robustly without relying on model internals, we do a
    # per-sample forward pass and scale each loss by that sample's weight.
    # -------------------------------------------------------------------------
    for epoch in range(int(args.epochs)):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            step += 1

            bs = int(batch["frames"].shape[0])
            weights = batch["weight"].detach().cpu().float().tolist()

            optimizer.zero_grad(set_to_none=True)

            sum_w = 0.0
            sum_weighted_loss = 0.0
            n_used = 0

            # Process each sample individually (deterministic + correct weighting)
            for i in range(bs):
                w = float(weights[i])
                if not np.isfinite(w) or w <= 0.0:
                    continue

                sample = {
                    "frames": batch["frames"][i].unsqueeze(0),
                    "j_left": batch["j_left"][i].unsqueeze(0),
                    "j_right": batch["j_right"][i].unsqueeze(0),
                    "buttons": batch["buttons"][i].unsqueeze(0),
                    "dropped_frames": batch["dropped_frames"][i].unsqueeze(0),
                    "game": "bn6",
                }

                enc = tokenizer.encode(sample)
                if not enc:
                    continue

                model_input = U.collate_encoded([enc], device=device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(model_input)
                    loss = out["loss"] if isinstance(out, dict) else out[0]

                # Scale and backprop
                (loss * w).backward()

                lv = float(loss.detach().cpu().item())
                sum_w += w
                sum_weighted_loss += lv * w
                n_used += 1

            if n_used == 0:
                continue

            optimizer.step()

            # Logging
            avg_w = sum_w / max(1e-9, float(n_used))
            avg_weighted_loss = sum_weighted_loss / max(1e-9, float(sum_w))

            writer.add_scalar("Train/WeightedLoss", avg_weighted_loss, step)
            writer.add_scalar("Train/AvgWeight", avg_w, step)
            writer.add_scalar("Train/UsedSamples", n_used, step)

            pbar.set_postfix({"wloss": f"{avg_weighted_loss:.4f}", "w": f"{avg_w:.2f}", "n": n_used})

            # Save
            if step % int(args.save_every) == 0:
                save_path = ckpt_dir / f"step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ckpt_config": U.to_dict(loaded.ckpt_config),
                        "tokenizer_cfg": U.to_dict(loaded.tokenizer_cfg),
                    },
                    save_path,
                )
                U.cleanup_old_checkpoints(ckpt_dir, int(args.max_keep_ckpts))

    print("✅ Done.")


if __name__ == "__main__":
    main()
