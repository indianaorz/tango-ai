# train_rl.py
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nitrogen.mm_tokenizers import NitrogenTokenizer
import train_utils as U


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 20,
    "save_every": 5000,
    "num_workers": 8,
    "max_keep_ckpts": 3,
    "base_ckpt": "checkpoints/battle/step_185000.pt",
}

DEFAULT_RL_CACHE_DIR = "data/nitrogen_rl_cache"
DEFAULT_CKPT_DIR = "checkpoints/rl_battle"
DEFAULT_LOG_DIR = "logs/rl_battle"


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _safe_load_pt(path: Path) -> Dict[str, Any]:
    """torch.load wrapper that tries mmap first (fast) then falls back."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _clamp_f(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _extract_step_from_path(p: str) -> int:
    try:
        name = Path(p).stem
        if name.startswith("step_"):
            return int(name.split("_")[1])
    except Exception:
        pass
    return 0


def _auto_resume_latest(ckpt_dir: Path) -> Optional[str]:
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1])
        if "_" in p.stem and p.stem.split("_")[1].isdigit()
        else -1,
    )
    return str(ckpts[-1]) if ckpts else None


# -----------------------------------------------------------------------------
# Dataset: ALL anchors (every timestep) across all cached RL segments
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _FileInfo:
    path: Path
    n: int
    base_w: float
    event_local: int
    actor: str
    outcome: str


def _event_local_index_from_meta(meta: Dict[str, Any], n: int) -> int:
    """
    Cheap contiguous assumption:
      local = clamp(event_frame - start_frame)
    Avoid scanning meta["action_indices"].
    """
    try:
        sf = int(meta.get("start_frame", 0))
        ef = int(meta.get("event_frame", -1))
        if ef >= 0:
            return int(max(0, min(ef - sf, n - 1)))
    except Exception:
        pass
    return int(max(0, min(n // 2, n - 1)))


def _window_indices_end_torch(anchor: int, V: int, n: int) -> torch.Tensor:
    """
    Vectorized window indices [anchor-(V-1) .. anchor], clamped to [0..n-1].
    Returns int64 tensor shape (V,).
    """
    anchor = int(max(0, min(anchor, n - 1)))
    V = int(max(1, V))
    start = anchor - (V - 1)
    idx = torch.arange(start, start + V, dtype=torch.int64)
    return idx.clamp_(0, n - 1)


class CachedRLSegmentAllAnchorsDataset(Dataset):
    """
    Treats every cached segment file as a contiguous sequence of anchor timesteps.
    __len__ == total anchors across all files.

    Each item provides a SINGLE anchor (local timestep) with:
      - frames: (V,3,256,256) float32 in [-1,1]
      - actions: (H, ACTION_DIM) split into j_left/j_right/buttons
      - weight: scalar float32 (reward weight for THIS anchor)
    """

    def __init__(
        self,
        root_dir: str,
        *,
        vision_horizon: int,
        weight_after_event: float = 1.0,
        weight_cap: float = 5.0,
        min_weight: float = 0.05,
        filter_actor: str = "both",   # "player" | "enemy" | "both"
        filter_outcome: str = "both", # "good" | "bad" | "both"
    ):
        self.root = Path(root_dir)
        files = sorted(self.root.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {self.root}")

        self.vision_horizon = int(max(1, vision_horizon))
        self.weight_after_event = float(weight_after_event)
        self.weight_cap = float(weight_cap)
        self.min_weight = float(min_weight)
        self.filter_actor = str(filter_actor)
        self.filter_outcome = str(filter_outcome)

        infos: List[_FileInfo] = []
        cumulative: List[int] = []
        total = 0

        for f in tqdm(files, desc="Indexing RL cache (all anchors)"):
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
                if int(actions.shape[0]) != n:
                    continue

                actor = str(meta.get("actor", ""))
                outcome = str(meta.get("outcome", ""))

                if self.filter_actor != "both" and actor != self.filter_actor:
                    continue
                if self.filter_outcome != "both" and outcome != self.filter_outcome:
                    continue

                base_w = _clamp_f(float(meta.get("weight", 1.0)), self.min_weight, self.weight_cap)
                event_local = _event_local_index_from_meta(meta, n)

                infos.append(
                    _FileInfo(
                        path=f,
                        n=n,
                        base_w=base_w,
                        event_local=int(event_local),
                        actor=actor,
                        outcome=outcome,
                    )
                )
                total += n
                cumulative.append(total)

            except Exception:
                continue

        if not infos or total <= 0:
            raise RuntimeError(f"No valid RL cached segments found in {self.root} after filtering.")

        self._infos = infos
        self._cumulative = cumulative

        # Per-worker cache of last loaded file
        self._cache_file_idx = -1
        self._cache_data: Optional[Dict[str, Any]] = None

    @property
    def cumulative(self) -> List[int]:
        # exposed for batch sampler
        return self._cumulative

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _locate(self, global_idx: int) -> Tuple[int, int]:
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
        return int(file_idx), int(local_idx)

    def _load_file(self, file_idx: int) -> None:
        self._cache_data = _safe_load_pt(self._infos[file_idx].path)
        self._cache_file_idx = int(file_idx)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        file_idx, local_idx = self._locate(global_idx)
        info = self._infos[file_idx]

        if self._cache_file_idx != file_idx:
            self._load_file(file_idx)

        assert self._cache_data is not None
        d = self._cache_data

        frames_u8: torch.Tensor = d["frames"]   # (N,3,256,256) uint8
        actions: torch.Tensor = d["actions"]    # (N,H,ACTION_DIM) float32
        n = int(frames_u8.shape[0])

        anchor = int(max(0, min(local_idx, n - 1)))

        # weight for THIS anchor (pre-event uses base_w, post-event uses weight_after_event)
        if anchor <= int(info.event_local):
            w = float(info.base_w)
        else:
            w = float(_clamp_f(self.weight_after_event, self.min_weight, self.weight_cap))

        # Vision window
        V = int(self.vision_horizon)
        vis_ids = _window_indices_end_torch(anchor, V, n)

        # frames: float32 in [-1,1]
        frames = frames_u8.index_select(0, vis_ids).to(dtype=torch.float32)
        frames.mul_(2.0 / 255.0).sub_(1.0)

        aw = actions[anchor].to(dtype=torch.float32)  # (H, ACTION_DIM)

        return {
            "frames": frames,
            "j_left": aw[:, 0:2],
            "j_right": aw[:, 2:4],
            "buttons": aw[:, 4:],
            "dropped_frames": torch.zeros(V, dtype=torch.bool),
            "weight": torch.tensor(w, dtype=torch.float32),
            "game": "bn6",
        }


# -----------------------------------------------------------------------------
# Batch sampler: keep batches within the same file for cache locality
# -----------------------------------------------------------------------------
class FileGroupedBatchSampler(Sampler[List[int]]):
    """
    Produces batches of global indices grouped by file.
    Shuffles file order per epoch and (optionally) shuffles anchors within each file.

    This drastically reduces tiny-file thrash (torch.load per sample) when you have
    many segment files and millions of anchors.
    """

    def __init__(
        self,
        cumulative: List[int],
        batch_size: int,
        *,
        seed: int,
        shuffle_within_file: bool = True,
        drop_last: bool = False,
    ):
        self.cum = list(map(int, cumulative))
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_file = bool(shuffle_within_file)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.bs <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.cum or self.cum[-1] <= 0:
            raise ValueError("Invalid cumulative list")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + 1009 * self.epoch)

        # Build [start,end) ranges per file
        ranges: List[Tuple[int, int]] = []
        prev = 0
        for c in self.cum:
            ranges.append((prev, c))
            prev = c

        rng.shuffle(ranges)

        for start, end in ranges:
            idxs = list(range(start, end))
            if self.shuffle_within_file:
                rng.shuffle(idxs)

            for i in range(0, len(idxs), self.bs):
                batch = idxs[i : i + self.bs]
                if self.drop_last and len(batch) < self.bs:
                    continue
                yield batch

    def __len__(self) -> int:
        total = int(self.cum[-1])
        if self.drop_last:
            return total // self.bs
        return (total + self.bs - 1) // self.bs


# -----------------------------------------------------------------------------
# Loss extraction: try to get per-sample loss vector (fast). fallback to slow.
# -----------------------------------------------------------------------------
def _extract_per_sample_loss(out: Any) -> Optional[torch.Tensor]:
    """
    Returns a tensor shape (B,) if the model exposes per-sample losses.
    Otherwise returns None.
    """
    if isinstance(out, dict):
        for k in ("loss_per_sample", "per_sample_loss", "losses"):
            v = out.get(k, None)
            if isinstance(v, torch.Tensor) and v.ndim == 1:
                return v
    return None


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL fine-tuning over ALL cached anchors with reward weighting.")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_RL_CACHE_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--base_ckpt", type=str, default=BASE_CONFIG["base_ckpt"])
    parser.add_argument("--overwrite_resume", action="store_true")

    parser.add_argument("--batch_size", type=int, default=BASE_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=BASE_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"])
    parser.add_argument("--save_every", type=int, default=BASE_CONFIG["save_every"])
    parser.add_argument("--num_workers", type=int, default=BASE_CONFIG["num_workers"])
    parser.add_argument("--max_keep_ckpts", type=int, default=BASE_CONFIG["max_keep_ckpts"])

    parser.add_argument("--filter_actor", choices=["player", "enemy", "both"], default="both")
    parser.add_argument("--filter_outcome", choices=["good", "bad", "both"], default="both")
    parser.add_argument("--weight_after_event", type=float, default=1.0)
    parser.add_argument("--weight_cap", type=float, default=5.0)
    parser.add_argument("--min_weight", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    # Optional: freeze vision
    parser.add_argument("--freeze_vision", action="store_true")

    # Optional: force fallback (debug)
    parser.add_argument("--force_slow_weighting", action="store_true", help="Force slow per-sample weighting (debug only).")

    args = parser.parse_args()

    device = torch.device(BASE_CONFIG["device"])
    U.set_seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Offline RL Fine-tune (ALL anchors, reward-weighted)")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Checkpoints: {ckpt_dir}")
    print(f"   Logs: {log_dir}")
    print(f"   Actor filter: {args.filter_actor} | Outcome filter: {args.filter_outcome}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
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

    if args.freeze_vision:
        for name, p in model.named_parameters():
            if "vision" in name or "siglip" in name:
                p.requires_grad = False
        print("🧊 Vision tower frozen.")
    else:
        print("🔥 Vision tower trainable (as requested).")

    # -------------------------------------------------------------------------
    # Dataset / loader: FULL PASS over all anchors
    # -------------------------------------------------------------------------
    dataset = CachedRLSegmentAllAnchorsDataset(
        root_dir=str(dataset_dir),
        vision_horizon=int(getattr(tokenizer, "vision_horizon", 1)),
        weight_after_event=float(args.weight_after_event),
        weight_cap=float(args.weight_cap),
        min_weight=float(args.min_weight),
        filter_actor=str(args.filter_actor),
        filter_outcome=str(args.filter_outcome),
    )

    total_anchors = len(dataset)
    print(f"   Total anchors: {total_anchors:,}")
    print(f"   Effective iters/epoch: {(total_anchors + int(args.batch_size) - 1) // int(args.batch_size):,} (batch_size={int(args.batch_size)})")

    # Batch sampler for locality (REPLACES shuffle=True)
    sampler = FileGroupedBatchSampler(
        dataset.cumulative,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle_within_file=True,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=8 if int(args.num_workers) > 0 else None,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    writer = SummaryWriter(str(log_dir))

    step = _extract_step_from_path(load_path)
    print(f"🔥 Training Loop Start (Continuing from Step {step})")

    warned_slow = False
    warned_no_per_sample = False

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(int(args.epochs)):
        sampler.set_epoch(epoch)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            step += 1

            bs = int(batch["frames"].shape[0])

            # Build encoded batch
            encoded_list: List[Dict[str, Any]] = []
            weights_list: List[float] = []

            for i in range(bs):
                sample = {
                    "frames": batch["frames"][i].unsqueeze(0),
                    "j_left": batch["j_left"][i].unsqueeze(0),
                    "j_right": batch["j_right"][i].unsqueeze(0),
                    "buttons": batch["buttons"][i].unsqueeze(0),
                    "dropped_frames": batch["dropped_frames"][i].unsqueeze(0),
                    "game": "bn6",
                }
                enc = tokenizer.encode(sample)
                if enc:
                    encoded_list.append(enc)
                    weights_list.append(float(batch["weight"][i].item()))

            used = int(len(encoded_list))
            if used == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            # Fast attempt: one forward pass
            model_input = U.collate_encoded(encoded_list, device=device)
            w = torch.tensor(weights_list, dtype=torch.float32, device=device)

            has_per_sample = False
            if not args.force_slow_weighting:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(model_input)
                    per_sample = _extract_per_sample_loss(out)
                    has_per_sample = per_sample is not None

                if per_sample is not None:
                    per_sample = per_sample.to(dtype=torch.float32)
                    denom = torch.clamp(w.sum(), min=1e-12)
                    loss = (per_sample * w).sum() / denom

                    loss.backward()
                    optimizer.step()

                    loss_val = float(loss.detach().cpu().item())

                    # Logging
                    avg_w = float(np.mean(weights_list)) if weights_list else 1.0
                    writer.add_scalar("Train/WeightedLoss", loss_val, step)
                    writer.add_scalar("Train/AvgWeight", avg_w, step)
                    writer.add_scalar("Train/UsedSamples", used, step)
                    writer.add_scalar("Debug/HasPerSampleLoss", 1.0, step)

                    pbar.set_postfix({"wloss": f"{loss_val:.4f}", "avg_w": f"{avg_w:.2f}", "used": used})

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
                    continue

                if not warned_no_per_sample:
                    print("⚠️ Model did not return per-sample losses; using slow per-sample weighting (will be much slower).")
                    warned_no_per_sample = True

            # Slow path: per-sample forward/backward with weights (correct, expensive)
            if not warned_slow:
                print("🐢 Slow weighting path active (one forward/backward per sample).")
                warned_slow = True

            sum_w = 0.0
            sum_weighted_loss = 0.0
            n_used = 0

            optimizer.zero_grad(set_to_none=True)

            for enc_i, w_i in zip(encoded_list, weights_list):
                if not np.isfinite(w_i) or w_i <= 0.0:
                    continue
                mi = U.collate_encoded([enc_i], device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_i = model(mi)
                    loss_i = out_i["loss"] if isinstance(out_i, dict) else out_i[0]
                (loss_i * float(w_i)).backward()

                lv = float(loss_i.detach().cpu().item())
                sum_w += float(w_i)
                sum_weighted_loss += lv * float(w_i)
                n_used += 1

            if n_used == 0 or sum_w <= 0.0:
                continue

            optimizer.step()
            loss_val = float(sum_weighted_loss / max(1e-12, sum_w))

            avg_w = float(np.mean(weights_list)) if weights_list else 1.0

            writer.add_scalar("Train/WeightedLoss", loss_val, step)
            writer.add_scalar("Train/AvgWeight", avg_w, step)
            writer.add_scalar("Train/UsedSamples", used, step)
            writer.add_scalar("Debug/HasPerSampleLoss", 0.0, step)

            pbar.set_postfix({"wloss": f"{loss_val:.4f}", "avg_w": f"{avg_w:.2f}", "used": used})

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
