# critic_rl/train.py
from __future__ import annotations

import argparse
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

from critic_rl.dataset import (
    CriticRLTDDataset,
    CachedCriticRLTDDataset,
    InMemoryCriticRLTDDataset,
    build_rl_sequence_cache,
    collate_batch,
)
from critic_rl.model import CriticRLConfig, HPDeltaTDLambdaCritic


# ---------------------------------------------------------------------------
# FAST DATALOADER (VECTORIZED)
# ---------------------------------------------------------------------------
class FastDataLoader:
    """
    Lightweight iterator that batches indices and calls dataset.get_batch()
    Bypasses PyTorch DataLoader collation overhead.
    """
    def __init__(self, dataset: InMemoryCriticRLTDDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.length = len(dataset) // self.batch_size

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        n = len(self.dataset)
        indices = torch.randperm(n) if self.shuffle else torch.arange(n)

        for i in range(0, n, self.batch_size):
            batch_idxs = indices[i : i + self.batch_size]
            if len(batch_idxs) < self.batch_size:
                continue
            yield self.dataset.get_batch(batch_idxs)


# ---------------------------------------------------------------------------
# Repro
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# TD(λ) targets (SARSA-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def td_lambda_targets(
    *,
    r: torch.Tensor,
    done: torch.Tensor,
    valid: torch.Tensor,  # Shape [B, T] (bool)
    v_pred: torch.Tensor, # Shape [B, T]
    gamma: float,
    lam: float,
) -> torch.Tensor:
    if r.ndim != 2:
        raise ValueError(f"r must be [B,T], got {tuple(r.shape)}")

    gg = float(gamma)
    ll = float(lam)
    B, T = r.shape

    # 1. Shift v_pred to get v_next (raw)
    v_next = torch.zeros_like(v_pred)
    if T > 1:
        v_next[:, :-1] = v_pred[:, 1:]

    # 2. CRITICAL FIX: Create a mask for "Is the NEXT step valid?"
    # If valid=[T, T, F], valid_next=[T, F, F]
    valid_next = torch.zeros_like(valid)
    if T > 1:
        valid_next[:, :-1] = valid[:, 1:]
    
    # 3. Apply the mask to v_next immediately
    # This prevents bootstrapping off the random output of padded frames
    v_next = v_next * valid_next.float()

    r_eff = torch.where(valid, r, torch.zeros_like(r))
    done_eff = torch.where(valid, done, torch.ones_like(done))

    targets = torch.zeros_like(r_eff)
    g = torch.zeros((B,), dtype=r_eff.dtype, device=r_eff.device)

    for t in range(T - 1, -1, -1):
        rt = r_eff[:, t]
        dt = done_eff[:, t].float()
        
        # vnt is now guaranteed 0.0 if t+1 is padding
        vnt = v_next[:, t] 
        
        g = rt + gg * (1.0 - dt) * ((1.0 - ll) * vnt + ll * g)
        
        # FIXED SAFETY: 
        # We multiply by valid[:, t] to ensure that 'g' (the target) 
        # is reset to 0.0 for EVERY individual sequence in the batch 
        # the moment it enters the padding region.
        g = g * valid[:, t].float()

        targets[:, t] = g

    # Final cleanup
    targets = torch.where(valid, targets, torch.zeros_like(targets))
    return targets


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: FastDataLoader | DataLoader,
    device: torch.device,
    *,
    gamma: float,
    lam: float,
    amp: bool,
) -> Tuple[float, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for xb, r, done, valid in loader:
        xb = move_batch(xb, device)
        r = r.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            v = model(xb)
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
    ap.add_argument("--cache_dir", type=str, default="data/cache_rl/s2_s256_2")
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--folder_len", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--require_cust_gt0", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_delta", type=float, default=10.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--run_name", type=str, default="v10_full")
    ap.add_argument("--save_dir", type=str, default="checkpoints/critic_rl")
    ap.add_argument("--tb_dir", type=str, default="")
    ap.add_argument("--tb_flush_secs", type=int, default=10)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=5000)
    ap.add_argument("--cache_replays", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--drop_rate", type=float, default=0, help="Fraction of boring frames to drop (0.0=keep all)")

    # AMP
    ap.add_argument("--amp", action="store_true", help="Enable AMP (recommended on CUDA).")
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP even on CUDA.")

    # Disable validation entirely
    ap.add_argument(
        "--no_val",
        action="store_true",
        help="Disable validation: do not build val dataset/loader; skip eval; best.pt chosen by train loss.",
    )
    ap.add_argument("--val_every", type=int, default=1, help="Evaluate every N epochs (ignored with --no_val).")

    args = ap.parse_args()

    set_seed(int(args.seed))
    device = torch.device(str(args.device))

    run_name = (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = os.path.join(str(args.save_dir), run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = (args.tb_dir or os.path.join(run_dir, "tb")).strip()
    _ensure_dir(tb_dir)

    amp_enabled = bool(args.amp) and not bool(args.no_amp) and device.type == "cuda"
    print(f"=== Critic RL TD(lambda) Training (startup) | amp={amp_enabled} ===", flush=True)

    # Cache build mode
    if bool(args.build_cache) or bool(args.rebuild_cache):
        if not args.cache_dir:
            raise SystemExit("--cache_dir is required when using --build_cache/--rebuild_cache")
        build_rl_sequence_cache(
            dataset_dir=str(args.dataset_dir),
            cache_dir=str(args.cache_dir),
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            overwrite=bool(args.rebuild_cache),
            num_workers=int(args.num_workers),
        )
        return

    use_cache = bool(args.cache_dir) and _cache_has_any_valid_files(str(args.cache_dir))
    print("building datasets...", flush=True)
    
    # If validation is disabled, give 100% of data to training.
    # Otherwise, use default 0.1 (10%).
    train_val_ratio = 0.0 if bool(args.no_val) else 0.1
    # --- Train dataset ---
    if use_cache:
        train_ds = CachedCriticRLTDDataset(
            str(args.dataset_dir),
            cache_dir=str(args.cache_dir),
            stride=int(args.stride),
            folder_len=int(args.folder_len),
            seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0),
            split="train",
            seed=int(args.seed),
            max_samples=args.max_train_samples,
            boring_drop_rate=float(args.drop_rate), 
            val_ratio=train_val_ratio,
        )
    else:
        train_ds = CriticRLTDDataset(
            str(args.dataset_dir),
            stride=int(args.stride),
            seq_len=int(args.seq_len),
            max_samples=args.max_train_samples,
            cache_replays=int(args.cache_replays),
            folder_len=int(args.folder_len),
            split="train",
            seed=int(args.seed),
            require_cust_gt0=bool(args.require_cust_gt0),
            boring_drop_rate=float(args.drop_rate), 
            val_ratio=train_val_ratio,
        )

    # --- Val dataset (optional) ---
    val_ds = None
    if not bool(args.no_val):
        if use_cache:
            val_ds = CachedCriticRLTDDataset(
                str(args.dataset_dir),
                cache_dir=str(args.cache_dir),
                stride=int(args.stride),
                folder_len=int(args.folder_len),
                seq_len=int(args.seq_len),
                require_cust_gt0=bool(args.require_cust_gt0),
                split="val",
                seed=int(args.seed),
                max_samples=args.max_val_samples,
            )
        else:
            val_ds = CriticRLTDDataset(
                str(args.dataset_dir),
                stride=max(1, int(args.stride) * 2),
                seq_len=int(args.seq_len),
                max_samples=args.max_val_samples,
                cache_replays=max(1, int(args.cache_replays)),
                folder_len=int(args.folder_len),
                split="val",
                seed=int(args.seed),
                require_cust_gt0=bool(args.require_cust_gt0),
            )

    if val_ds is None:
        print(f"datasets ready. train_samples={len(train_ds)} (NO VAL)", flush=True)
    else:
        print(f"datasets ready. train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)





    # ---------------------------------------------------------------------
    # LOADER SELECTION
    # ---------------------------------------------------------------------
    
    # Check if we are using the Streaming Dataset (Iterable)
    from critic_rl.dataset import StreamingCachedDataset
    is_iterable = isinstance(train_ds, StreamingCachedDataset)

    # Use FastDataLoader ONLY if it's the specific In-Memory class (which holds tensors in RAM)
    if isinstance(train_ds, InMemoryCriticRLTDDataset):
        print("Using FastDataLoader (Vectorized In-Memory) for Train. Workers=0 forced.", flush=True)
        train_loader = FastDataLoader(train_ds, int(args.batch_size), shuffle=True)
        val_loader = FastDataLoader(val_ds, int(args.batch_size), shuffle=False) if val_ds is not None else None
    else:
        workers = int(args.num_workers)
        print(f"Using Standard DataLoader. Config: workers={workers}, iterable={is_iterable}", flush=True)
        
        loader_kwargs: Dict[str, object] = {
            "batch_size": int(args.batch_size),
            "pin_memory": (device.type == "cuda"),
            "collate_fn": collate_batch,
            "num_workers": workers,
        }
        
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
            loader_kwargs["persistent_workers"] = True

        # Standard DataLoader configuration
        # IMPORTANT: shuffle=True is forbidden for IterableDatasets
        train_shuffle = True if not is_iterable else False
        
        train_loader = DataLoader(
            train_ds, 
            shuffle=train_shuffle, 
            drop_last=True, 
            **loader_kwargs
        )  # type: ignore[arg-type]
        
        val_loader = DataLoader(
            val_ds, 
            shuffle=False, 
            drop_last=False, 
            **loader_kwargs
        ) if val_ds is not None else None  # type: ignore[arg-type]

    # Model
    cfg = CriticRLConfig(max_seq_len=max(32, int(args.seq_len)))
    model = HPDeltaTDLambdaCritic(cfg, folder_len=int(args.folder_len)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.HuberLoss(delta=float(args.huber_delta), reduction="none")
    scaler = GradScaler(enabled=amp_enabled)

    print("=== Critic RL TD(lambda) Training (model) ===", flush=True)

    writer = SummaryWriter(log_dir=tb_dir, flush_secs=int(args.tb_flush_secs))
    _tb_text(writer, "Run/Args", str(vars(args)))
    _tb_text(writer, "Run/ModelCfg", str(asdict(cfg)))


    # Best tracking:
    # - normal case: best = lowest val_rmse
    # - no_val: best = lowest train_huber (epoch average)
    best_metric = float("inf")
    best_metric_name = "val_rmse" if not bool(args.no_val) else "train_huber"
    global_step = 0
    steps_per_epoch = max(1, len(train_loader))

    # Optional smoke test: just a few steps, no saving best, etc.
    max_epochs = 1 if bool(args.smoke_test) else int(args.epochs)
    max_steps_per_epoch = 5 if bool(args.smoke_test) else None

    try:
        first_batch_reported = False  # Flag to trigger the one-time report

        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            n = 0

            t0 = time.time()
            pbar = tqdm(
                total=steps_per_epoch if max_steps_per_epoch is None else min(steps_per_epoch, max_steps_per_epoch),
                desc=f"train e{epoch}/{max_epochs}",
                dynamic_ncols=True,
                leave=True,
                mininterval=0.5,
            )
            last_print = time.time()

            # -------------------------
            # TRAIN LOOP
            # -------------------------
            for step, (xb, r, done, valid) in enumerate(train_loader, start=1):
                
                # --- INTEGRITY REPORT (Runs once on first successful load) ---
                if not first_batch_reported:
                    print("\n" + "="*50)
                    print(" DATASET INTEGRITY REPORT (Live Batch)")
                    print("="*50)
                    
                    # Move to CPU for stats calculation
                    v_cpu = valid.detach().cpu()
                    r_cpu = r.detach().cpu()
                    
                    total_el = v_cpu.numel()
                    active_el = v_cpu.sum().item()
                    density = (active_el / total_el) * 100
                    
                    active_r = r_cpu[v_cpu]
                    r_mean = active_r.mean().item() if active_el > 0 else 0.0
                    r_max = active_r.abs().max().item() if active_el > 0 else 0.0
                    
                    print(f"Batch Size:         {r.shape[0]}")
                    print(f"Combat Density:     {density:.1f}% (Gameplay vs Padding)")
                    print(f"Avg Reward (HP):    {r_mean:.4f}")
                    print(f"Max Delta:          {r_max:.1f} HP")
                    
                    if density < 10.0:
                        print("!! WARNING: Extremely low combat density !!")
                    elif density > 95.0:
                        print("!! NOTE: High density. Truncation is rare. !!")
                    print("="*50 + "\n")
                    first_batch_reported = True
                # --- END INTEGRITY REPORT ---

                if (max_steps_per_epoch is not None) and (step > max_steps_per_epoch):
                    break

                global_step += 1

                xb = move_batch(xb, device)
                r = r.to(device, non_blocking=True)
                done = done.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # Model prediction Q(s, a)
                    q = model(xb)  # [B, T]

                    # SARSA TD(λ) targets
                    # Detach Q to ensure targets are treated as ground truth labels
                    y = td_lambda_targets(
                        r=r,
                        done=done,
                        valid=valid,
                        v_pred=q.detach(),
                        gamma=float(args.gamma),
                        lam=float(args.lam),
                    )

                    if global_step % 50 == 0:
                        with torch.no_grad():
                            m = valid
                            qq = q[m]
                            yy = y[m]
                            print(
                                f"[dbg] q: mean={qq.mean().item():.2f} std={qq.std().item():.2f} maxabs={qq.abs().max().item():.2f} | "
                                f"y: mean={yy.mean().item():.2f} std={yy.std().item():.2f} maxabs={yy.abs().max().item():.2f}"
                            )

                    # Huber loss for robustness against outliers
                    per = loss_fn(q, y)
                    mask = valid.float()
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
                n += int(mask.sum().item())

                cur_train = running / max(1, n)
                pbar.update(1)

                if global_step % 10 == 0:
                    writer.add_scalar("Train/HuberLoss", float(loss.item()), global_step)
                    writer.add_scalar("Train/LR", float(opt.param_groups[0]["lr"]), global_step)
                    if amp_enabled:
                        writer.add_scalar("Train/GradScaler", float(scaler.get_scale()), global_step)

                if global_step % 1000 == 0:
                    ckpt = {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if amp_enabled else None,
                        "cfg": asdict(cfg),
                        "args": vars(args),
                    }
                    torch.save(ckpt, os.path.join(run_dir, "last.pt"))
                    print(f"[save] Saved intermediate checkpoint at step {global_step}", flush=True)

                now = time.time()
                if now - last_print >= 1.0:
                    it_per_s = pbar.format_dict.get("rate", None)
                    pbar.set_postfix(huber=f"{cur_train:.4f}", it_s=f"{it_per_s:.2f}" if it_per_s else "?")
                    last_print = now

            pbar.close()
            train_huber = running / max(1, n)
            writer.add_scalar("Epoch/TrainHuber", float(train_huber), epoch)

            # -------------------------
            # VALIDATION (optional)
            # -------------------------
            did_val = False
            val_mae = None
            val_rmse = None
            val_time = 0.0

            if (val_loader is not None) and (not bool(args.no_val)) and (int(args.val_every) > 0) and (epoch % int(args.val_every) == 0):
                v0 = time.time()
                val_mae, val_rmse = evaluate(
                    model,
                    val_loader,
                    device,
                    gamma=float(args.gamma),
                    lam=float(args.lam),
                    amp=amp_enabled,
                )
                writer.add_scalar("Val/MAE", float(val_mae), epoch)
                writer.add_scalar("Val/RMSE", float(val_rmse), epoch)
                did_val = True
                val_time = time.time() - v0

            # -------------------------
            # LOGGING
            # -------------------------
            train_time = time.time() - t0
            if did_val:
                print(
                    f"epoch={epoch} train_huber={train_huber:.4f} val_rmse={float(val_rmse):.3f} "
                    f"(train={train_time:.1f}s val={val_time:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch} train_huber={train_huber:.4f} (train={train_time:.1f}s)",
                    flush=True,
                )

            # -------------------------
            # SAVE LAST
            # -------------------------
            ckpt = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "cfg": asdict(cfg),
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))

            # -------------------------
            # BEST SELECTION
            # -------------------------
            if bool(args.no_val) or (val_loader is None) or (not did_val):
                metric = float(train_huber)
                is_best = metric < best_metric
            else:
                metric = float(val_rmse)  # type: ignore[arg-type]
                is_best = metric < best_metric

            if is_best and (not bool(args.smoke_test)):
                best_metric = float(metric)
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  -> new best {best_metric_name}={best_metric:.4f}", flush=True)

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
