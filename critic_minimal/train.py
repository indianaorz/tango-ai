# critic_minimal/train.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from critic_minimal.dataset import StreamingMinimalQDataset, collate_minimal_q
from critic_minimal.features import normalize_target_tanh, value_weight_tanh
from critic_minimal.model import MinimalQConfig, MinimalQCritic


def set_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def move_batch(xb: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in xb.items()}


def _load_norm_factor_from_manifest(cache_dir: str) -> float:
    """
    Prefer abs_p95 median (more robust), then abs_p90 median, else fallback.
    """
    fallback = 325.0
    try:
        mpath = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(mpath):
            return fallback
        m = json.loads(Path(mpath).read_text(encoding="utf-8"))  # type: ignore[name-defined]
        ts = m.get("targets_summary", {}) if isinstance(m, dict) else {}
        if isinstance(ts, dict):
            v = ts.get("abs_p95_median_across_files", None)
            if isinstance(v, (int, float)) and v > 1e-6:
                return float(v)
            v = ts.get("abs_p90_median_across_files", None)
            if isinstance(v, (int, float)) and v > 1e-6:
                return float(v)
        return fallback
    except Exception:
        return fallback


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    y_norm_factor: float,
    tanh_clip: float,
    weight_scale: float,
    use_weighting: bool,
) -> Tuple[float, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0.0

    for xb, y_raw, valid in loader:
        xb = move_batch(xb, device)
        y_raw = y_raw.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        y = normalize_target_tanh(y_raw, norm_factor=y_norm_factor)

        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            q = model(xb)

        mask = valid.float()
        if use_weighting:
            w = value_weight_tanh(y_raw, norm_factor=y_norm_factor, tanh_clip=tanh_clip, scale=weight_scale)
            mask = mask * (1.0 + w)

        err = (q - y) * mask
        mae_sum += err.abs().sum().item()
        mse_sum += (err * err).sum().item()
        n += float(mask.sum().item())

    mae = mae_sum / max(1.0, n)
    rmse = (mse_sum / max(1.0, n)) ** 0.5
    return mae, rmse


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Minimal Q(s,a) critic (regression to Bellman HP-return).")
    ap.add_argument("--cache_dir", type=str, default="data/cache_minimal_q/h4_t192")
    ap.add_argument("--run_name", type=str, default="minq_v2_tanh")
    ap.add_argument("--save_dir", type=str, default="checkpoints/critic_minimal")
    ap.add_argument("--tb_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)

    # NOTE: target is tanh-space in [-1,1]; deltas like 0.1-0.5 make sense.
    ap.add_argument("--huber_delta", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # split / limits
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_train_sequences", type=int, default=None)
    ap.add_argument("--max_val_sequences", type=int, default=5000)

    # AMP
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    # target normalization
    ap.add_argument(
        "--y_norm_factor",
        type=float,
        default=0.0,
        help="If 0, auto-pick from manifest (abs_p95 median). y_norm=tanh(y_raw/y_norm_factor).",
    )

    # weighting (Nitrogen-flavored) based on RAW y
    ap.add_argument("--use_weighting", action="store_true", help="Apply tanh(|y_raw|/norm) weighting in the loss.")
    ap.add_argument("--tanh_clip", type=float, default=5.0)
    ap.add_argument("--weight_scale", type=float, default=2.0)

    # model config overrides
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--time_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--emotion_vocab", type=int, default=256)
    ap.add_argument("--chip_vocab", type=int, default=512)
    ap.add_argument("--tile_vocab", type=int, default=128)

    args = ap.parse_args()

    set_seed(int(args.seed), deterministic=bool(args.deterministic))
    device = torch.device(str(args.device))

    run_name = (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = os.path.join(str(args.save_dir), run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = (args.tb_dir or os.path.join(run_dir, "tb")).strip()
    os.makedirs(tb_dir, exist_ok=True)

    amp_enabled = bool(args.amp) and not bool(args.no_amp) and device.type == "cuda"
    print(f"=== minimal-q train | amp={amp_enabled} device={device} deterministic={bool(args.deterministic)} ===", flush=True)

    # datasets
    train_ds = StreamingMinimalQDataset(
        str(args.cache_dir),
        split="train",
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_sequences=args.max_train_sequences,
    )
    val_ds = StreamingMinimalQDataset(
        str(args.cache_dir),
        split="val",
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_sequences=args.max_val_sequences,
    )

    # norm factor
    y_norm_factor = float(args.y_norm_factor)
    if y_norm_factor <= 0.0:
        y_norm_factor = _load_norm_factor_from_manifest(str(args.cache_dir))
    print(f"   y_norm_factor={y_norm_factor:.3f} (tanh target)", flush=True)

    loader_kwargs: Dict[str, object] = {
        "batch_size": int(args.batch_size),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_minimal_q,
        "num_workers": int(args.num_workers),
        "drop_last": True,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)  # IterableDataset => no shuffle
    val_loader = DataLoader(val_ds, shuffle=False, **{**loader_kwargs, "drop_last": False})

    # model (bind to cache schema)
    schema = train_ds.schema
    cfg = MinimalQConfig(
        max_seq_len=int(schema.seq_len),
        d_model=int(args.d_model),
        time_layers=int(args.time_layers),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
        action_dim=int(schema.action_dim),
        scalar_dim=int(schema.scalar_dim),
        emotion_vocab=int(args.emotion_vocab),
        chip_vocab=int(args.chip_vocab),
        tile_vocab=int(args.tile_vocab),
    )
    model = MinimalQCritic(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.HuberLoss(delta=float(args.huber_delta), reduction="none")
    scaler = GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
    try:
        writer.add_text("Run/Args", str(vars(args)))
        writer.add_text("Run/ModelCfg", str(asdict(cfg)))
        writer.add_text("Run/Schema", str(schema))
        writer.add_text("Run/y_norm_factor", str(y_norm_factor))
    except Exception:
        pass

    best_rmse = float("inf")
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        denom = 0.0

        t0 = time.time()
        pbar = tqdm(desc=f"train e{epoch}/{int(args.epochs)}", dynamic_ncols=True)

        for xb, y_raw, valid in train_loader:
            global_step += 1
            xb = move_batch(xb, device)
            y_raw = y_raw.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            y = normalize_target_tanh(y_raw, norm_factor=y_norm_factor)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                q = model(xb)  # [B,T] (predict tanh-space)
                per = loss_fn(q, y)  # [B,T]

                mask = valid.float()
                if bool(args.use_weighting):
                    w = value_weight_tanh(
                        y_raw,
                        norm_factor=y_norm_factor,
                        tanh_clip=float(args.tanh_clip),
                        scale=float(args.weight_scale),
                    )
                    mask = mask * (1.0 + w)

                loss = (per * mask).sum() / torch.clamp(mask.sum(), min=1.0)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
                opt.step()

            running += float(loss.item()) * float(mask.sum().item())
            denom += float(mask.sum().item())

            if global_step % 20 == 0:
                writer.add_scalar("Train/Loss", float(loss.item()), global_step)
                writer.add_scalar("Train/LR", float(opt.param_groups[0]["lr"]), global_step)
                if amp_enabled:
                    writer.add_scalar("Train/GradScale", float(scaler.get_scale()), global_step)

            pbar.set_postfix(loss=f"{(running / max(1.0, denom)):.4f}")
            pbar.update(1)

            if global_step % 2000 == 0:
                ckpt = {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if amp_enabled else None,
                    "cfg": asdict(cfg),
                    "schema": {"seq_len": schema.seq_len, "action_dim": schema.action_dim, "scalar_dim": schema.scalar_dim},
                    "y_norm_factor": float(y_norm_factor),
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        pbar.close()
        train_loss = running / max(1.0, denom)
        writer.add_scalar("Epoch/TrainLoss", float(train_loss), epoch)

        # validation (tanh-space)
        val_mae, val_rmse = evaluate(
            model,
            val_loader,
            device,
            amp=amp_enabled,
            y_norm_factor=float(y_norm_factor),
            tanh_clip=float(args.tanh_clip),
            weight_scale=float(args.weight_scale),
            use_weighting=bool(args.use_weighting),
        )
        writer.add_scalar("Epoch/ValMAE", float(val_mae), epoch)
        writer.add_scalar("Epoch/ValRMSE", float(val_rmse), epoch)

        # save
        ckpt = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "cfg": asdict(cfg),
            "schema": {"seq_len": schema.seq_len, "action_dim": schema.action_dim, "scalar_dim": schema.scalar_dim},
            "y_norm_factor": float(y_norm_factor),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        elapsed = time.time() - t0
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f} (t={elapsed:.1f}s)",
            flush=True,
        )

        if float(val_rmse) < best_rmse:
            best_rmse = float(val_rmse)
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"  -> new best val_rmse={best_rmse:.4f}", flush=True)

        writer.flush()

    print("done.", flush=True)
    writer.close()


if __name__ == "__main__":
    from pathlib import Path  # local import to keep top clean

    main()
