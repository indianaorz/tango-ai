#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch
from ng_policy import load_ng_checkpoint, NgNitroGenPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights/ng.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--t", type=int, default=48)
    p.add_argument("--c", type=int, default=3)
    p.add_argument("--h", type=int, default=256)
    p.add_argument("--w", type=int, default=256)

    p.add_argument("--num-actions", type=int, default=16)  # set to len(DISCRETE_ACTIONS) if you want exact
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    dev = torch.device(args.device)

    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load NG
    loaded = load_ng_checkpoint(args.ckpt, device=dev)
    policy = NgNitroGenPolicy(loaded).to(dev).eval()

    # Dummy input: [B,T,C,H,W]
    frames = torch.rand(1, args.t, args.c, args.h, args.w, device=dev, dtype=torch.float32)

    # Warmup
    _ = policy(frames, seed=args.seed, return_continuous=True)

    # Timed runs
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.runs):
        action = policy(frames, seed=args.seed, return_continuous=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    print("OK: NitroGen get_action() succeeded.")
    print(f"action shape: {tuple(action.shape)}  dtype={action.dtype}  device={action.device}")
    print(f"avg forward time: {(t1 - t0)/args.runs*1000:.3f} ms")
    print("action head (first 8 dims):", action[0, :8].tolist())


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
