# critic/train.py
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from critic.dataset import (
    CriticHPDeltaDataset,
    CachedCriticHPDeltaDataset,
    build_hpdelta_cache,
    collate_batch,
)
from critic.model import CriticConfig, HPDeltaCritic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb = move_batch(xb, device)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        err = pred - yb
        mae_sum += err.abs().sum().item()
        mse_sum += (err * err).sum().item()
        n += yb.numel()
    mae = mae_sum / max(1, n)
    rmse = (mse_sum / max(1, n)) ** 0.5
    return mae, rmse


def _cache_has_any_valid_files(cache_dir: str) -> bool:
    try:
        from pathlib import Path
        p = Path(cache_dir)
        if not p.exists():
            return False
        files = [x for x in p.glob("*.pt") if x.name != "_manifest.pt"]
        return len(files) > 0
    except Exception:
        return False


def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _tb_text(writer: SummaryWriter, tag: str, text: str) -> None:
    try:
        writer.add_text(tag, text)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, default="data/dataset")

    # Cache controls
    ap.add_argument("--cache_dir", type=str, default="", help="If set, use cache_dir/*.pt cache files.")
    ap.add_argument("--build_cache", action="store_true", help="Build cache into --cache_dir and exit.")
    ap.add_argument("--rebuild_cache", action="store_true", help="Overwrite existing cache files when building.")
    ap.add_argument("--folder_len", type=int, default=30, help="Folder length used during tensorization (cache-sensitive).")

    # Training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_delta", type=float, default=50.0)

    # RL return config
    ap.add_argument("--gamma", type=float, default=0.999, help="Discount factor per frame for return targets.")

    ap.add_argument("--run_name", type=str, default="", help="Subfolder name under --save_dir for checkpoints/logs.")
    ap.add_argument("--save_dir", type=str, default="checkpoints/critic_hpdelta")

    # TensorBoard
    ap.add_argument("--tb_dir", type=str, default="", help="TensorBoard log dir. Default: <run_dir>/tb")
    ap.add_argument("--tb_flush_secs", type=int, default=10, help="TensorBoard flush interval seconds.")

    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--require_cust_gt0", action="store_true", help="Train only on frames where cust_gauge > 0.")

    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=5000)

    # JSONL-mode RAM cache (still used if you don't use --cache_dir)
    ap.add_argument("--cache_replays", type=int, default=2)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke_test", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if not (0.0 < float(args.gamma) <= 1.0):
        raise SystemExit(f"--gamma must be in (0,1], got {args.gamma}")

    run_name = (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = (args.tb_dir or os.path.join(run_dir, "tb")).strip()
    _ensure_dir(tb_dir)

    print("=== Critic Value Training (startup) ===", flush=True)
    print(f"device={device} dataset_dir={args.dataset_dir}", flush=True)
    if args.cache_dir:
        print(f"cache_dir={args.cache_dir} build_cache={bool(args.build_cache)} rebuild_cache={bool(args.rebuild_cache)}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"tb_dir={tb_dir}", flush=True)
    print(f"gamma={float(args.gamma)} require_cust_gt0={bool(args.require_cust_gt0)} stride={args.stride} folder_len={args.folder_len}", flush=True)
    print(f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} wd={args.weight_decay} huber_delta={args.huber_delta}", flush=True)
    print(f"num_workers={args.num_workers}", flush=True)

    # ---------------------------------------------------------------------
    # Optional: build cache
    # ---------------------------------------------------------------------
    if args.build_cache or args.rebuild_cache:
        if not args.cache_dir:
            raise SystemExit("--cache_dir is required when using --build_cache/--rebuild_cache")

        build_hpdelta_cache(
            dataset_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=args.stride,
            folder_len=args.folder_len,
            require_cust_gt0=bool(args.require_cust_gt0),
            gamma=float(args.gamma),
            overwrite=bool(args.rebuild_cache),
        )

        print("cache build done.", flush=True)
        return

    # ---------------------------------------------------------------------
    # Datasets (cached if cache_dir has files; else JSONL)
    # ---------------------------------------------------------------------
    use_cache = bool(args.cache_dir) and _cache_has_any_valid_files(args.cache_dir)

    print("building datasets (this can take a bit)...", flush=True)

    if use_cache:
        print("using CACHED dataset.", flush=True)
        train_ds = CachedCriticHPDeltaDataset(
            args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            gamma=float(args.gamma),
            split="train",
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
        val_ds = CachedCriticHPDeltaDataset(
            args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            gamma=float(args.gamma),
            split="val",
            seed=args.seed,
            max_samples=args.max_val_samples,
        )
    else:
        if args.cache_dir:
            print("cache_dir provided but no cache files found; falling back to JSONL dataset.", flush=True)
            print("tip: build cache with --build_cache", flush=True)

        train_ds = CriticHPDeltaDataset(
            args.dataset_dir,
            stride=args.stride,
            gamma=float(args.gamma),
            max_samples=args.max_train_samples,
            cache_replays=args.cache_replays,
            folder_len=args.folder_len,
            split="train",
            seed=args.seed,
            require_cust_gt0=bool(args.require_cust_gt0),
        )
        val_ds = CriticHPDeltaDataset(
            args.dataset_dir,
            stride=max(1, args.stride * 2),
            gamma=float(args.gamma),
            max_samples=args.max_val_samples,
            cache_replays=max(1, args.cache_replays),
            folder_len=args.folder_len,
            split="val",
            seed=args.seed,
            require_cust_gt0=bool(args.require_cust_gt0),
        )

    print("datasets ready.", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=False,
    )

    cfg = CriticConfig()
    model = HPDeltaCritic(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=float(args.huber_delta))

    print("=== Critic Value Training (model) ===", flush=True)
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    print(f"config={asdict(cfg)}", flush=True)

    # ---------------------------------------------------------------------
    # TensorBoard
    # ---------------------------------------------------------------------
    writer = SummaryWriter(log_dir=tb_dir, flush_secs=int(args.tb_flush_secs))
    _tb_text(writer, "Run/Info", f"run_dir={run_dir}\nuse_cache={use_cache}\ncache_dir={args.cache_dir}")
    _tb_text(writer, "Run/Args", "\n".join([f"{k}={v}" for k, v in sorted(vars(args).items())]))
    _tb_text(writer, "Run/ModelConfig", "\n".join([f"{k}={v}" for k, v in sorted(asdict(cfg).items())]))

    try:
        writer.add_hparams(
            {
                "stride": int(args.stride),
                "folder_len": int(args.folder_len),
                "require_cust_gt0": int(bool(args.require_cust_gt0)),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "huber_delta": float(args.huber_delta),
                "gamma": float(args.gamma),
                "use_cache": int(bool(use_cache)),
            },
            {"hparam/placeholder": 0.0},
        )
    except Exception:
        pass

    if args.smoke_test:
        print("running smoke_test: one forward/backward step...", flush=True)
        xb, yb = next(iter(train_loader))
        xb = move_batch(xb, device)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        print(f"[smoke_test] pred_mean={pred.mean().item():.3f} loss={loss.item():.3f}", flush=True)
        writer.add_scalar("SmokeTest/Loss", float(loss.item()), 0)
        writer.flush()
        writer.close()
        return

    best_val = float("inf")
    steps_per_epoch = len(train_loader)
    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n = 0

            t0 = time.time()
            pbar = tqdm(total=steps_per_epoch, desc=f"train e{epoch}/{args.epochs}", dynamic_ncols=True, leave=True)
            last_print = time.time()

            for step, (xb, yb) in enumerate(train_loader, start=1):
                global_step += 1

                xb = move_batch(xb, device)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm: Optional[float] = None
                try:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_norm = float(total_norm.item()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
                except Exception:
                    grad_norm = None

                opt.step()

                running += loss.item() * yb.numel()
                n += yb.numel()

                cur_train = running / max(1, n)
                pbar.update(1)

                writer.add_scalar("Train/HuberLoss", float(loss.item()), global_step)
                writer.add_scalar("Train/HuberLoss_avg", float(cur_train), global_step)
                writer.add_scalar("Train/LR", float(opt.param_groups[0]["lr"]), global_step)
                if grad_norm is not None:
                    writer.add_scalar("Train/GradNorm", float(grad_norm), global_step)

                now = time.time()
                if now - last_print >= 0.5 or step == steps_per_epoch:
                    it_per_s = pbar.format_dict.get("rate", None)
                    pbar.set_postfix(huber=f"{cur_train:.4f}", it_s=f"{it_per_s:.2f}" if it_per_s else "?")
                    last_print = now

            pbar.close()
            train_huber = running / max(1, n)

            writer.add_scalar("Epoch/TrainHuber", float(train_huber), epoch)
            writer.add_scalar("Epoch/GlobalStep", float(global_step), epoch)

            v0 = time.time()
            val_mae, val_rmse = evaluate(model, val_loader, device)
            v1 = time.time()

            writer.add_scalar("Val/MAE", float(val_mae), epoch)
            writer.add_scalar("Val/RMSE", float(val_rmse), epoch)

            print(
                f"epoch={epoch} "
                f"train_huber={train_huber:.4f} "
                f"val_mae={val_mae:.3f} val_rmse={val_rmse:.3f} "
                f"(train_time={time.time()-t0:.1f}s val_time={v1-v0:.1f}s)",
                flush=True,
            )

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
                "args": vars(args),
            }

            torch.save(ckpt, os.path.join(run_dir, "last.pt"))

            if val_rmse < best_val:
                best_val = val_rmse
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  -> new best val_rmse={best_val:.3f}", flush=True)
                writer.add_scalar("Val/BestRMSE", float(best_val), epoch)

            writer.flush()

        print("done.", flush=True)

    finally:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
