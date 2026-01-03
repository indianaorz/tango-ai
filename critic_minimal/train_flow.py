# critic_minimal/train_flow.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from critic_minimal.dataset import FlowDataset
from critic_minimal.model_flow import ActionFlowDiT


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_resume(ckpt_path: Path, device: torch.device):
    """
    Loads a full checkpoint dict if present. Supports:
      - legacy: state_dict only
      - new: {"model":..., "opt":..., "epoch":..., "best_loss":..., "global_step":...}
    """
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    # legacy
    return {
        "model": obj,
        "opt": None,
        "epoch": 0,
        "best_loss": float("inf"),
        "global_step": 0,
    }


def _save_ckpt(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    *,
    epoch: int,
    best_loss: float,
    global_step: int,
) -> None:
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "global_step": int(global_step),
    }
    torch.save(payload, path)


_STEP_RE = re.compile(r"^step_(\d{9})\.pt$")


def _prune_step_checkpoints(save_dir: Path, keep_last: int) -> None:
    """
    Keep only the most recent `keep_last` step_*.pt snapshots.
    Never touches best.pt / last.pt.

    keep_last <= 0 => keep everything.
    """
    if keep_last <= 0:
        return

    items: list[tuple[int, Path]] = []
    for p in save_dir.iterdir():
        if not p.is_file():
            continue
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        items.append((step, p))

    if len(items) <= keep_last:
        return

    items.sort(key=lambda t: t[0])  # ascending by step
    to_delete = items[: max(0, len(items) - keep_last)]

    for _, path in to_delete:
        try:
            path.unlink()
        except Exception:
            # On Windows you can hit transient file locks (AV/indexer).
            # Failing to prune is non-fatal; next prune attempt will try again.
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/cache_flow_v5_eventweighted")
    ap.add_argument("--save_dir", default="checkpoints/flow_v9_eventweighted")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Model config
    ap.add_argument("--p_uncond", type=float, default=0.1)  # CFG dropout prob
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--hist_len", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=18)

    # Return conditioning (must match dataset encoding)
    ap.add_argument("--ret_scale", type=float, default=50.0)
    ap.add_argument("--ret_clip", type=float, default=200.0)

    # Rectified flow / solver training
    ap.add_argument("--beta_a", type=float, default=1.5)
    ap.add_argument("--beta_b", type=float, default=1.0)

    # Edge weighting
    ap.add_argument("--edge_mult", type=float, default=20.0)

    # Resume / seed
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true", help="resume from save_dir/last.pt if present")

    # Step-based saving
    ap.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save checkpoints every N optimizer steps (0 disables).",
    )
    ap.add_argument(
        "--save_last_each_epoch",
        action="store_true",
        help="Also write last.pt at the end of every epoch (in addition to step saves).",
    )
    ap.add_argument(
        "--keep_step_ckpts",
        type=int,
        default=25,
        help="Keep only the most recent N step_*.pt snapshots (0 keeps all). best.pt/last.pt are always kept.",
    )

    # Optional AMP
    ap.add_argument("--amp", action="store_true", help="Enable torch autocast AMP")

    # Optional max_steps (0 = no limit)
    ap.add_argument("--max_steps", type=int, default=0, help="Stop after N optimizer steps (0 disables).")

    args = ap.parse_args()
    device = torch.device(args.device)

    _seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"

    print(f"--- Training Bellman Flow (event-weighted cache) ---")
    print(f"cache_dir={args.cache_dir}")
    print(f"save_dir={args.save_dir}")
    print(f"device={device}")
    print(
        f"seq_len={args.seq_len} hist_len={args.hist_len} batch={args.batch_size} "
        f"save_every_steps={args.save_every_steps} keep_step_ckpts={args.keep_step_ckpts}"
    )
    print(f"amp={bool(args.amp)} max_steps={int(args.max_steps)}")

    ds = FlowDataset(
        args.cache_dir,
        context_len=args.hist_len,
        pred_len=args.seq_len,
        ret_scale=args.ret_scale,
        ret_clip=args.ret_clip,
    )

    sampler = WeightedRandomSampler(
        weights=ds.weights_tensor,
        num_samples=len(ds),
        replacement=True,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = ActionFlowDiT(
        feat_dim=47,
        act_dim=10,
        hist_len=args.hist_len,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float("inf")
    global_step = 0

    if args.resume and last_path.exists():
        print(f"🔄 Resuming from {last_path} ...")
        ckpt = _load_resume(last_path, device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("opt") is not None:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        global_step = int(ckpt.get("global_step", 0))
        print(f"   resumed: epoch={start_epoch} global_step={global_step} best_loss={best_loss:.6f}")

    # beta distribution for t sampling (bias to low t -> more noise)
    beta_dist = torch.distributions.beta.Beta(args.beta_a, args.beta_b)

    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # helps avoid double-saving on resume if global_step already lands on boundary
    last_saved_step = global_step

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Ep {epoch}", dynamic_ncols=True)
        loss_sum = 0.0
        steps_in_epoch = 0

        for batch in pbar:
            hist = batch["history"].to(device, non_blocking=True)   # [B, hist, feat]
            ret = batch["return"].to(device, non_blocking=True)     # [B, 1]
            x1 = batch["action"].to(device, non_blocking=True)      # [B, seq, act]

            B, Seq, Dim = x1.shape

            # Edge weighting: emphasize changes
            diffs = torch.abs(x1[:, 1:] - x1[:, :-1])               # [B, seq-1, act]
            zeros = torch.zeros((B, 1, Dim), device=device)
            edges = torch.cat([zeros, diffs], dim=1)                # [B, seq, act]
            loss_weights = 1.0 + (edges * float(args.edge_mult))

            # CFG dropout: zero out conditioning return sometimes
            if args.p_uncond > 0.0:
                drop = (torch.rand(B, device=device) < args.p_uncond)
                if drop.any():
                    ret = ret.clone()
                    ret[drop] = 0.0

            # Rectified Flow (velocity target)
            x0 = torch.randn_like(x1)
            sample = beta_dist.sample((B,)).to(device)              # in (0,1)
            t = 1.0 - sample                                        # bias toward 0

            t_view = t[:, None, None]
            x_t = t_view * x1 + (1.0 - t_view) * x0
            target_v = x1 - x0

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_v = model(x_t, t, hist, ret)
                raw = (pred_v - target_v) ** 2
                loss = (raw * loss_weights).mean()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            global_step += 1
            steps_in_epoch += 1
            loss_sum += float(loss.item())

            pbar.set_postfix(loss=loss_sum / max(1, steps_in_epoch), step=global_step)

            # STEP-BASED SAVE: fires immediately when hitting multiples of N steps
            if args.save_every_steps and args.save_every_steps > 0:
                if (global_step % args.save_every_steps == 0) and (global_step != last_saved_step):
                    # write last.pt for resume safety
                    _save_ckpt(
                        last_path,
                        model,
                        opt,
                        epoch=epoch,
                        best_loss=best_loss,
                        global_step=global_step,
                    )
                    # snapshot
                    snap_path = save_dir / f"step_{global_step:09d}.pt"
                    _save_ckpt(
                        snap_path,
                        model,
                        opt,
                        epoch=epoch,
                        best_loss=best_loss,
                        global_step=global_step,
                    )

                    # prune old snapshots
                    _prune_step_checkpoints(save_dir, int(args.keep_step_ckpts))

                    last_saved_step = global_step
                    print(f"\n💾 Saved checkpoint at global_step={global_step} -> {snap_path.name}")

            # Optional stop condition
            if args.max_steps and args.max_steps > 0 and global_step >= int(args.max_steps):
                print(f"\n🛑 Reached max_steps={args.max_steps}. Saving last.pt and exiting.")
                _save_ckpt(
                    last_path,
                    model,
                    opt,
                    epoch=epoch,
                    best_loss=best_loss,
                    global_step=global_step,
                )
                return

        avg_loss = loss_sum / max(1, steps_in_epoch)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.6f}")

        # Optional: always write last.pt each epoch end
        if args.save_last_each_epoch:
            _save_ckpt(
                last_path,
                model,
                opt,
                epoch=epoch,
                best_loss=best_loss,
                global_step=global_step,
            )

        # Best model only on improvement (epoch-level)
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_ckpt(
                best_path,
                model,
                opt,
                epoch=epoch,
                best_loss=best_loss,
                global_step=global_step,
            )
            print(f"-> New Best Model Saved ({best_loss:.6f})")


if __name__ == "__main__":
    main()
