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
            
            # Use the vectorized getter!
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
# TD(λ) targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def td_lambda_targets(
    *,
    r: torch.Tensor,
    done: torch.Tensor,
    valid: torch.Tensor,
    v_pred: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    if r.ndim != 2:
        raise ValueError(f"r must be [B,T], got {tuple(r.shape)}")
    
    gg = float(gamma)
    ll = float(lam)
    B, T = r.shape

    v_next = torch.zeros_like(v_pred)
    if T > 1:
        v_next[:, :-1] = v_pred[:, 1:]

    r_eff = torch.where(valid, r, torch.zeros_like(r))
    done_eff = torch.where(valid, done, torch.ones_like(done))

    targets = torch.zeros_like(r_eff)
    g = torch.zeros((B,), dtype=r_eff.dtype, device=r_eff.device)
    
    for t in range(T - 1, -1, -1):
        rt = r_eff[:, t]
        dt = done_eff[:, t].float()
        vnt = v_next[:, t]
        g = rt + gg * (1.0 - dt) * ((1.0 - ll) * vnt + ll * g)
        targets[:, t] = g

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
) -> Tuple[float, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for xb, r, done, valid in loader:
        xb = move_batch(xb, device)
        
        # --- CHANGE HERE: Scale rewards ---
        r = r.to(device, non_blocking=True) * 0.01
        # ----------------------------------
        
        done = done.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

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
        if not p.exists(): return False
        files = [x for x in p.glob("*.pt") if x.name != "_manifest.pt"]
        return len(files) > 0
    except: return False


def _ensure_dir(p: str) -> None:
    if p: os.makedirs(p, exist_ok=True)


def _tb_text(writer: SummaryWriter, tag: str, text: str) -> None:
    try: writer.add_text(tag, text)
    except: pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--folder_len", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--require_cust_gt0", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--huber_delta", type=float, default=50.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--run_name", type=str, default="")
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

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    run_name = (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = (args.tb_dir or os.path.join(run_dir, "tb")).strip()
    _ensure_dir(tb_dir)

    print("=== Critic RL TD(lambda) Training (startup) ===", flush=True)
    
    if args.build_cache or args.rebuild_cache:
        build_rl_sequence_cache(
            dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, stride=int(args.stride),
            folder_len=int(args.folder_len), seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0), overwrite=bool(args.rebuild_cache),
            num_workers=int(args.num_workers),
        )
        return

    use_cache = bool(args.cache_dir) and _cache_has_any_valid_files(args.cache_dir)
    print("building datasets...", flush=True)

    if use_cache:
        # Defaults to InMemory alias
        train_ds = CachedCriticRLTDDataset(
            args.dataset_dir, cache_dir=args.cache_dir, stride=int(args.stride),
            folder_len=int(args.folder_len), seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0), split="train",
            seed=args.seed, max_samples=args.max_train_samples,
        )
        val_ds = CachedCriticRLTDDataset(
            args.dataset_dir, cache_dir=args.cache_dir, stride=int(args.stride),
            folder_len=int(args.folder_len), seq_len=int(args.seq_len),
            require_cust_gt0=bool(args.require_cust_gt0), split="val",
            seed=args.seed, max_samples=args.max_val_samples,
        )
    else:
        # Fallback to RAM dataset
        train_ds = CriticRLTDDataset(
            args.dataset_dir, stride=int(args.stride), seq_len=int(args.seq_len),
            max_samples=args.max_train_samples, cache_replays=int(args.cache_replays),
            folder_len=int(args.folder_len), split="train", seed=int(args.seed),
            require_cust_gt0=bool(args.require_cust_gt0),
        )
        val_ds = CriticRLTDDataset(
            args.dataset_dir, stride=max(1, int(args.stride) * 2), seq_len=int(args.seq_len),
            max_samples=args.max_val_samples, cache_replays=max(1, int(args.cache_replays)),
            folder_len=int(args.folder_len), split="val", seed=int(args.seed),
            require_cust_gt0=bool(args.require_cust_gt0),
        )

    print("datasets ready.", flush=True)

    # ---------------------------------------------------------------------
    # FAST LOADER SELECTION
    # ---------------------------------------------------------------------
    
    if isinstance(train_ds, InMemoryCriticRLTDDataset):
        print(f"Using FastDataLoader (Vectorized In-Memory) for Train. Workers=0 forced.", flush=True)
        train_loader = FastDataLoader(train_ds, int(args.batch_size), shuffle=True)
        # Validate using FastLoader too for speed
        val_loader = FastDataLoader(val_ds, int(args.batch_size), shuffle=False)
    else:
        # Fallback to standard
        workers = int(args.num_workers)
        loader_kwargs = {
            "batch_size": int(args.batch_size),
            "pin_memory": (device.type == "cuda"),
            "collate_fn": collate_batch,
            "num_workers": workers,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
            loader_kwargs["persistent_workers"] = True

        print(f"Using Standard DataLoader. Config: workers={workers}", flush=True)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    # Model
    cfg = CriticRLConfig(max_seq_len=max(32, int(args.seq_len)))
    model = HPDeltaTDLambdaCritic(cfg, folder_len=int(args.folder_len)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.HuberLoss(delta=float(args.huber_delta), reduction="none")

    print("=== Critic RL TD(lambda) Training (model) ===", flush=True)
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)

    writer = SummaryWriter(log_dir=tb_dir, flush_secs=int(args.tb_flush_secs))
    
    # ... (smoke test logic omitted for brevity, assuming standard loop) ...

    best_val = float("inf")
    global_step = 0
    # Estimate steps for progress bar
    steps_per_epoch = len(train_loader)

    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            running = 0.0
            n = 0

            t0 = time.time()
            pbar = tqdm(total=steps_per_epoch, desc=f"train e{epoch}/{int(args.epochs)}", dynamic_ncols=True, leave=True, mininterval=0.5)
            last_print = time.time()

            # Main Loop
            for step, (xb, r, done, valid) in enumerate(train_loader, start=1):
                global_step += 1

                xb = move_batch(xb, device)
                
                # --- CHANGE HERE: Scale rewards ---
                r = r.to(device, non_blocking=True) * 0.01
                # ----------------------------------
                
                done = done.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    v = model(xb)
                    y = td_lambda_targets(r=r, done=done, valid=valid, v_pred=v.detach(), gamma=float(args.gamma), lam=float(args.lam))
                    per = loss_fn(v, y)
                    mask = valid.float()
                    loss = (per * mask).sum() / torch.clamp(mask.sum(), min=1.0)

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

                now = time.time()
                if now - last_print >= 1.0:
                    it_per_s = pbar.format_dict.get("rate", None)
                    pbar.set_postfix(huber=f"{cur_train:.4f}", it_s=f"{it_per_s:.2f}" if it_per_s else "?")
                    last_print = now

            pbar.close()
            train_huber = running / max(1, n)
            writer.add_scalar("Epoch/TrainHuber", float(train_huber), epoch)
            
            v0 = time.time()
            val_mae, val_rmse = evaluate(model, val_loader, device, gamma=float(args.gamma), lam=float(args.lam))
            writer.add_scalar("Val/MAE", float(val_mae), epoch)
            writer.add_scalar("Val/RMSE", float(val_rmse), epoch)

            print(
                f"epoch={epoch} train_huber={train_huber:.4f} val_rmse={val_rmse:.3f} "
                f"(train={time.time()-t0:.1f}s val={time.time()-v0:.1f}s)",
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

            writer.flush()

        print("done.", flush=True)

    finally:
        try: writer.flush(); writer.close()
        except: pass

if __name__ == "__main__":
    main()