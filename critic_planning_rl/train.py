#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import PlanningDataset, collate_fn
from .model import PlanningCritic, PlanningConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _pretty_sps(n: int, dt: float) -> float:
    return float(n) / float(dt) if dt > 0 else 0.0


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainCfg:
    data: str
    save: str
    save_ckpt: Optional[str]
    tb_dir: str
    epochs: int
    batch: int
    lr: float
    weight_decay: float
    num_workers: int
    amp: bool
    grad_clip: float
    seed: int
    shuffle_hand: bool
    max_samples: Optional[int]
    # Target
    target_key: str
    target_transform: str
    # Logging
    log_every: int
    save_every: int


def train() -> None:
    ap = argparse.ArgumentParser("Train PlanningCritic (viewer-compatible saving)")

    ap.add_argument("--data", default="data/chipwindows_v2/strategy_v2.jsonl")
    ap.add_argument("--save", default="checkpoints/planning_critic.pt",
                    help="RAW state_dict path (viewer loads this)")
    ap.add_argument("--save_ckpt", default="checkpoints/planning_critic.ckpt",
                    help="Optional wrapped resume checkpoint (set '' to disable)")
    ap.add_argument("--tb_dir", default="runs/planning_critic")

    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--shuffle_hand", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--target_key", default="net_yield_weighted",
                    choices=["net_yield_weighted"],
                    help="Train on weighted value (DPS-weighted signal)")
    ap.add_argument("--target_transform", default="signed_log1p",
                    choices=["signed_log1p", "none"])

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1)

    args = ap.parse_args()

    cfg = TrainCfg(
        data=str(args.data),
        save=str(args.save),
        save_ckpt=(str(args.save_ckpt) if str(args.save_ckpt).strip() else None),
        tb_dir=str(args.tb_dir),
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        shuffle_hand=bool(args.shuffle_hand),
        max_samples=(None if args.max_samples is None else int(args.max_samples)),
        target_key=str(args.target_key),
        target_transform=str(args.target_transform),
        log_every=int(args.log_every),
        save_every=max(1, int(args.save_every)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(cfg.amp and device == "cuda")

    _seed_all(cfg.seed)
    _ensure_dir_for_file(cfg.save)
    if cfg.save_ckpt:
        _ensure_dir_for_file(cfg.save_ckpt)
    os.makedirs(cfg.tb_dir, exist_ok=True)

    # Dataset
    ds = PlanningDataset(
        cfg.data,
        max_samples=cfg.max_samples,
        shuffle_hand=cfg.shuffle_hand,
        target_key=cfg.target_key,              # IMPORTANT
        target_transform=cfg.target_transform,  # IMPORTANT
    )
    if len(ds) == 0:
        raise SystemExit(f"No samples loaded from: {cfg.data}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )

    # Model / opt / loss
    model = PlanningCritic(PlanningConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=cfg.tb_dir)

    print(
        "Training PlanningCritic\n"
        f"  device={device} amp={amp_enabled}\n"
        f"  samples={len(ds)} batch={cfg.batch} workers={cfg.num_workers}\n"
        f"  target={cfg.target_key} transform={cfg.target_transform}\n"
        f"  save(state_dict)={cfg.save}\n"
        f"  save_ckpt(wrapped)={cfg.save_ckpt}\n"
        f"  tb_dir={cfg.tb_dir}\n"
        f"  train_py={__file__}"
    )

    global_step = 0
    best_epoch_loss: Optional[float] = None

    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        ep_loss_sum = 0.0
        ep_count = 0

        last_log_t = time.time()

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Dataset is expected to provide transformed target already.
            # Still safe if you later choose to return raw_target too.
            target = batch["target"].view(-1)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(batch).view(-1)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = int(pred.shape[0])
            loss_val = float(loss.detach().item())

            ep_loss_sum += loss_val * bs
            ep_count += bs
            global_step += 1

            if cfg.log_every > 0 and (global_step % cfg.log_every == 0):
                now = time.time()
                dt = max(1e-6, now - last_log_t)
                last_log_t = now

                with torch.no_grad():
                    mae = float(torch.mean(torch.abs(pred - target)).item())
                    pred_mean = float(pred.mean().item())
                    targ_mean = float(target.mean().item())

                writer.add_scalar("Train/LossHuber", loss_val, global_step)
                writer.add_scalar("Train/MAE", mae, global_step)
                writer.add_scalar("Train/PredMean", pred_mean, global_step)
                writer.add_scalar("Train/TargetMean", targ_mean, global_step)
                writer.add_scalar("Train/LR", opt.param_groups[0]["lr"], global_step)
                writer.add_scalar("Train/SamplesPerSec", _pretty_sps(cfg.log_every * cfg.batch, dt), global_step)

        avg_loss = ep_loss_sum / max(1, ep_count)
        dt_ep = time.time() - t0
        sps = _pretty_sps(ep_count, dt_ep)

        improved = best_epoch_loss is None or avg_loss < best_epoch_loss
        if improved:
            best_epoch_loss = avg_loss

        writer.add_scalar("Epoch/AvgLoss", avg_loss, ep + 1)
        writer.add_scalar("Epoch/SamplesPerSec", sps, ep + 1)
        writer.add_scalar("Epoch/IsBest", 1.0 if improved else 0.0, ep + 1)

        print(f"Epoch {ep+1}/{cfg.epochs}: avg_loss={avg_loss:.4f}  ({sps:.0f} samp/s)")

        # -------------------------
        # SAVE (THIS IS THE FIX)
        # -------------------------
        if ((ep + 1) % cfg.save_every) == 0 or (ep + 1) == cfg.epochs:
            # 1) Viewer-compatible file: RAW state_dict ONLY
            torch.save(model.state_dict(), cfg.save)

            # 2) Optional wrapped checkpoint for resume/debug
            if cfg.save_ckpt:
                wrapped: Dict[str, Any] = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": ep + 1,
                    "global_step": global_step,
                    "best_epoch_loss": best_epoch_loss,
                    "train_cfg": cfg.__dict__,
                    "model_cfg": PlanningConfig().__dict__,
                }
                torch.save(wrapped, cfg.save_ckpt)

    writer.close()
    print(f"Done.\n  Saved state_dict to: {cfg.save}\n  Saved wrapped ckpt to: {cfg.save_ckpt}")


if __name__ == "__main__":
    train()
