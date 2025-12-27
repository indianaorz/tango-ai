from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from critic_rl.dataset import (
    CriticRLTDDataset,
    CachedCriticRLTDDataset,
    build_rl_sequence_cache,
    collate_batch,
)
from critic_rl.model import CriticRLConfig, HPDeltaTDLambdaCritic


# ---------------------------------------------------------------------------
# Repro
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# TD(λ) targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def td_lambda_targets(
    *,
    r: torch.Tensor,         # [B,T]
    done: torch.Tensor,      # [B,T] bool
    valid: torch.Tensor,     # [B,T] bool
    v_pred: torch.Tensor,    # [B,T] (used for bootstrap on V_{t+1})
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """
    Computes TD(λ) returns (λ-return) for each timestep:

      G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]   if not done_t
      G_t^λ = r_t                                             if done_t

    We treat padded steps (valid=False) as ignored; their targets are 0.
    We bootstrap with V(s_{t+1}) from the model prediction at the next step inside the sequence.
    At the final timestep, V(s_{T}) is treated as 0.
    """
    if r.ndim != 2:
        raise ValueError(f"r must be [B,T], got {tuple(r.shape)}")
    if done.shape != r.shape or valid.shape != r.shape or v_pred.shape != r.shape:
        raise ValueError("shape mismatch among r/done/valid/v_pred")

    gg = float(gamma)
    ll = float(lam)
    if not (0.0 < gg <= 1.0):
        raise ValueError(f"gamma must be in (0,1], got {gamma}")
    if not (0.0 <= ll <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")

    B, T = r.shape

    # v_next[t] = V_{t+1} for t<T-1, else 0
    v_next = torch.zeros_like(v_pred)
    if T > 1:
        v_next[:, :-1] = v_pred[:, 1:]

    # Apply valid mask to inputs (padded positions -> 0)
    r_eff = torch.where(valid, r, torch.zeros_like(r))
    done_eff = torch.where(valid, done, torch.ones_like(done))  # invalid -> treat as done

    targets = torch.zeros_like(r_eff)

    g = torch.zeros((B,), dtype=r_eff.dtype, device=r_eff.device)  # running return
    for t in range(T - 1, -1, -1):
        rt = r_eff[:, t]
        dt = done_eff[:, t].float()  # 1 if done else 0
        vnt = v_next[:, t]

        # if done: g = r_t
        # else: g = r_t + gamma * ((1-lam) * V_{t+1} + lam * g_next)
        g = rt + gg * (1.0 - dt) * ((1.0 - ll) * vnt + ll * g)
        targets[:, t] = g

    # ensure invalid steps are 0 (safe for loss masking)
    targets = torch.where(valid, targets, torch.zeros_like(targets))
    return targets


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    gamma: float,
    lam: float,
) -> Tuple[float, float]:
    """
    Report MAE/RMSE between V(s_t) and TD(λ) targets on validation sequences.
    """
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for xb, r, done, valid in loader:
        xb = move_batch(xb, device)
        r = r.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        v = model(xb)  # [B,T]
        y = td_lambda_targets(r=r, done=done, valid=valid, v_pred=v, gamma=gamma, lam=lam)

        mask = valid
        err = (v - y)
        err = torch.where(mask, err, torch.zeros_like(err))

        mae_sum += err.abs().sum().item()
        mse_sum += (err * err).sum().item()
        n += int(mask.sum().item())

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

    # Sequence config
    ap.add_argument("--seq_len", type=int, default=16, help="Number of timesteps per training sequence.")
    ap.add_argument("--stride", type=int, default=6, help="Raw-frame stride between sequence timesteps.")
    ap.add_argument("--require_cust_gt0", action="store_true", help="Only include starts where cust_gauge > 0.")

    # TD config
    ap.add_argument("--gamma", type=float, default=0.999, help="Discount factor per timestep (sequence step).")
    ap.add_argument("--lam", type=float, default=0.95, help="TD(lambda) mixing coefficient.")

    # Training
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_delta", type=float, default=50.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)

    ap.add_argument("--run_name", type=str, default="", help="Subfolder name under --save_dir for checkpoints/logs.")
    ap.add_argument("--save_dir", type=str, default="checkpoints/critic_rl")

    # TensorBoard
    ap.add_argument("--tb_dir", type=str, default="", help="TensorBoard log dir. Default: <run_dir>/tb")
    ap.add_argument("--tb_flush_secs", type=int, default=10, help="TensorBoard flush interval seconds.")

    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=5000)

    # JSONL-mode RAM cache (only if not using --cache_dir)
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
    if not (0.0 <= float(args.lam) <= 1.0):
        raise SystemExit(f"--lam must be in [0,1], got {args.lam}")
    if int(args.seq_len) < 2:
        raise SystemExit(f"--seq_len must be >= 2, got {args.seq_len}")

    run_name = (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = (args.tb_dir or os.path.join(run_dir, "tb")).strip()
    _ensure_dir(tb_dir)

    print("=== Critic RL TD(lambda) Training (startup) ===", flush=True)
    print(f"device={device} dataset_dir={args.dataset_dir}", flush=True)
    if args.cache_dir:
        print(f"cache_dir={args.cache_dir} build_cache={bool(args.build_cache)} rebuild_cache={bool(args.rebuild_cache)}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"tb_dir={tb_dir}", flush=True)
    print(
        f"seq_len={int(args.seq_len)} stride={int(args.stride)} require_cust_gt0={bool(args.require_cust_gt0)} "
        f"gamma={float(args.gamma)} lam={float(args.lam)} folder_len={int(args.folder_len)}",
        flush=True,
    )
    print(
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
        f"huber_delta={args.huber_delta} clip_grad_norm={args.clip_grad_norm}",
        flush=True,
    )
    print(f"num_workers={args.num_workers}", flush=True)

    # ---------------------------------------------------------------------
    # Optional: build cache
    # ---------------------------------------------------------------------
    if args.build_cache or args.rebuild_cache:
        if not args.cache_dir:
            raise SystemExit("--cache_dir is required when using --build_cache/--rebuild_cache")

        build_rl_sequence_cache(
            dataset_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
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
        train_ds = CachedCriticRLTDDataset(
            args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            split="train",
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
        val_ds = CachedCriticRLTDDataset(
            args.dataset_dir,
            cache_dir=args.cache_dir,
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            split="val",
            seed=args.seed,
            max_samples=args.max_val_samples,
        )
    else:
        if args.cache_dir:
            print("cache_dir provided but no cache files found; falling back to JSONL dataset.", flush=True)
            print("tip: build cache with --build_cache", flush=True)

        train_ds = CriticRLTDDataset(
            args.dataset_dir,
            stride=int(args.stride),
            seq_len=int(args.seq_len),
            max_samples=args.max_train_samples,
            cache_replays=int(args.cache_replays),
            folder_len=int(args.folder_len),
            split="train",
            seed=int(args.seed),
            require_cust_gt0=bool(args.require_cust_gt0),
        )
        val_ds = CriticRLTDDataset(
            args.dataset_dir,
            stride=max(1, int(args.stride) * 2),
            seq_len=int(args.seq_len),
            max_samples=args.max_val_samples,
            cache_replays=max(1, int(args.cache_replays)),
            folder_len=int(args.folder_len),
            split="val",
            seed=int(args.seed),
            require_cust_gt0=bool(args.require_cust_gt0),
        )

    print("datasets ready.", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=False,
    )

    # Model
    cfg = CriticRLConfig(max_seq_len=max(32, int(args.seq_len)))
    model = HPDeltaTDLambdaCritic(cfg, folder_len=int(args.folder_len)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.HuberLoss(delta=float(args.huber_delta), reduction="none")

    print("=== Critic RL TD(lambda) Training (model) ===", flush=True)
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
                "seq_len": int(args.seq_len),
                "stride": int(args.stride),
                "folder_len": int(args.folder_len),
                "require_cust_gt0": int(bool(args.require_cust_gt0)),
                "gamma": float(args.gamma),
                "lam": float(args.lam),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "huber_delta": float(args.huber_delta),
                "use_cache": int(bool(use_cache)),
            },
            {"hparam/placeholder": 0.0},
        )
    except Exception:
        pass

    if args.smoke_test:
        print("running smoke_test: one forward/backward step...", flush=True)
        xb, r, done, valid = next(iter(train_loader))
        xb = move_batch(xb, device)
        r = r.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        v = model(xb)
        y = td_lambda_targets(r=r, done=done, valid=valid, v_pred=v, gamma=float(args.gamma), lam=float(args.lam))

        # masked huber over valid steps only
        per = loss_fn(v, y)
        mask = valid.float()
        loss = (per * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        print(f"[smoke_test] v_mean={v.mean().item():.3f} loss={loss.item():.3f}", flush=True)
        writer.add_scalar("SmokeTest/Loss", float(loss.item()), 0)
        writer.flush()
        writer.close()
        return

    best_val = float("inf")
    global_step = 0
    steps_per_epoch = len(train_loader)

    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            running = 0.0
            n = 0

            t0 = time.time()
            pbar = tqdm(total=steps_per_epoch, desc=f"train e{epoch}/{int(args.epochs)}", dynamic_ncols=True, leave=True)
            last_print = time.time()

            for step, (xb, r, done, valid) in enumerate(train_loader, start=1):
                global_step += 1

                xb = move_batch(xb, device)
                r = r.to(device, non_blocking=True)
                done = done.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)

                v = model(xb)  # [B,T]

                # targets use bootstrap from v (detached via @no_grad), so this is true TD(λ)
                y = td_lambda_targets(
                    r=r, done=done, valid=valid, v_pred=v.detach(), gamma=float(args.gamma), lam=float(args.lam)
                )

                # masked huber loss
                per = loss_fn(v, y)
                mask = valid.float()
                loss = (per * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm: Optional[float] = None
                try:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
                    grad_norm = float(total_norm.item()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
                except Exception:
                    grad_norm = None

                opt.step()

                # stats
                running += float(loss.item()) * float(mask.sum().item())
                n += int(mask.sum().item())

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
            val_mae, val_rmse = evaluate(
                model, val_loader, device, gamma=float(args.gamma), lam=float(args.lam)
            )
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
                "epoch": int(epoch),
                "global_step": int(global_step),
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

#build
#python -u -m critic_rl.train --dataset_dir data/dataset --cache_dir data/cache_rl_tdlam/v1 --build_cache --rebuild_cache --require_cust_gt0 --stride 6 --seq_len 16 --folder_len 30

#train
#python -u -m critic_rl.train --dataset_dir data/dataset --cache_dir data/cache_rl_tdlam/v1 --run_name tdlam_v1 --require_cust_gt0 --stride 6 --seq_len 16 --gamma 0.999 --lam 0.95 --batch_size 128 --epochs 10 --lr 3e-4 --weight_decay 1e-2 --huber_delta 50 --clip_grad_norm 1.0 --num_workers 4

#smoketest
#python -u -m critic_rl.train --dataset_dir data/dataset --cache_dir data/cache_rl_tdlam/v1 --run_name smoke --require_cust_gt0 --stride 6 --seq_len 16 --gamma 0.999 --lam 0.95 --batch_size 8 --epochs 1 --smoke_test





