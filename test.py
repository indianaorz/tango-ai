#!/usr/bin/env python3
"""
Check whether NitrogenTokenizer.encode() produces pixel_values in [0,1] or [-1,1],
using your *actual cached training data directory* (data/dataset_cached/*.pt).

Run (defaults should work):
  python check_siglip_norm_cached.py

Optional:
  python check_siglip_norm_cached.py --dataset_dir data/dataset_cached --pick newest --idx 1500
  NG_CKPT_PATH=weights/ng.pt python check_siglip_norm_cached.py
"""

import argparse
from pathlib import Path
import os
import torch

from ng_policy import load_ng_checkpoint
from nitrogen.mm_tokenizers import NitrogenTokenizer


def window_indices(idx: int, T: int, n: int) -> list[int]:
    idx = max(0, min(idx, n - 1))
    start = idx - (T - 1)
    out = []
    for t in range(T):
        j = start + t
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        out.append(j)
    return out


def stats(name: str, t: torch.Tensor) -> None:
    t = t.detach().float().cpu()
    print(f"{name}: shape={tuple(t.shape)} dtype={t.dtype}")
    print(
        f"  min={t.min().item():.6f} max={t.max().item():.6f} "
        f"mean={t.mean().item():.6f} std={t.std(unbiased=False).item():.6f}"
    )


def pick_pt_file(dataset_dir: Path, pick: str) -> Path:
    pts = sorted(dataset_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files found in {dataset_dir}")

    if pick == "first":
        return pts[0]
    if pick == "newest":
        return max(pts, key=lambda p: p.stat().st_mtime)
    if pick == "largest":
        return max(pts, key=lambda p: p.stat().st_size)
    raise ValueError(f"Unknown --pick {pick!r}")


def load_cached_pt(path: Path) -> dict:
    # Try mmap for speed if supported
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/dataset_cached")
    ap.add_argument("--pick", choices=["first", "newest", "largest"], default="newest")
    ap.add_argument("--idx", type=int, default=-1, help="Frame index. -1 means middle of file.")
    ap.add_argument("--ckpt", default=os.getenv("NG_CKPT_PATH", "weights/ng.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--vision_horizon",
        type=int,
        default=1,
        help="How many frames to include in the vision stream. Keep 1 unless you know token budget fits.",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    pt_path = pick_pt_file(dataset_dir, args.pick)
    print(f"Using cached file: {pt_path}")

    device = torch.device(args.device)
    loaded = load_ng_checkpoint(args.ckpt, device=device)
    if loaded.tokenizer_cfg is None:
        raise RuntimeError("Checkpoint missing tokenizer_cfg; can't test tokenizer safely.")

    tokenizer = NitrogenTokenizer(loaded.tokenizer_cfg)
    tokenizer.eval()

    action_horizon = int(getattr(tokenizer, "action_horizon", 18))
    vision_horizon = int(args.vision_horizon)
    if vision_horizon <= 0:
        vision_horizon = 1

    data = load_cached_pt(pt_path)
    frames_u8 = data["frames"]  # [N,3,H,W] uint8
    actions = data["actions"]   # [N,25] (float/half/etc)

    n = int(frames_u8.shape[0])
    if n <= 0:
        raise RuntimeError("Cached file has zero frames.")

    idx = args.idx
    if idx < 0:
        idx = n // 2

    ids = window_indices(idx, action_horizon, n)

    # Actions window: [1, T_act, ...]
    act_win = actions[ids].float()          # [T_act,25]
    j_left  = act_win[:, 0:2].unsqueeze(0)  # [1,T_act,2]
    j_right = act_win[:, 2:4].unsqueeze(0)  # [1,T_act,2]
    buttons = act_win[:, 4:].unsqueeze(0)   # [1,T_act,21]

    # Vision window: last T_vis frames, in [0,1]
    vis_ids = ids[-vision_horizon:]
    vis_u8 = frames_u8[vis_ids].float() / 255.0  # [T_vis,3,H,W] in [0,1]
    frames = vis_u8.unsqueeze(0)                 # [1,T_vis,3,H,W]

    dropped = torch.zeros((1, vision_horizon), dtype=torch.bool)

    sample = {
        "frames": frames,              # [1,T_vis,3,H,W] in [0,1]
        "j_left": j_left,              # [1,T_act,2]
        "j_right": j_right,            # [1,T_act,2]
        "buttons": buttons,            # [1,T_act,21]
        "dropped_frames": dropped,     # [1,T_vis]
        "game": "bn6",
    }

    enc = tokenizer.encode(sample)

    print("\n--- Horizons ---")
    print(f"action_horizon={action_horizon} vision_horizon={vision_horizon}")

    print("\n--- What you passed into encode() ---")
    stats("frames_in", frames)

    # Find encoded vision tensor
    pv = enc.get("pixel_values", None)
    pv_key = "pixel_values"
    if pv is None:
        for k in ("frames", "images"):
            if k in enc and torch.is_tensor(enc[k]):
                pv = enc[k]
                pv_key = k
                break

    print("\n--- What the model will see (encoded vision) ---")
    if pv is None:
        print("Could not find encoded vision tensor (pixel_values/frames/images). Keys:")
        print(sorted(enc.keys()))
        return

    stats(f"{pv_key}_encoded", pv)

    mn = float(pv.detach().min().item())
    mx = float(pv.detach().max().item())

    print("\n--- Interpretation ---")
    if mn >= -1.2 and mx <= 1.2 and (mx > 0.6) and (mn < -0.6):
        print("Tokenizer is producing ~[-1, 1] pixel values (SigLIP-style normalization already applied).")
        print("=> Feed frames to encode() in [0,1]. Do NOT pre-normalize in the dataset.")
    elif mn >= -0.05 and mx <= 1.05:
        print("Tokenizer is producing ~[0, 1] pixel values (normalization likely NOT applied).")
        print("=> You must normalize exactly once (e.g., in dataset or a preprocessing hook) before vision tower.")
    else:
        print("Encoded pixel range is unusual (not clearly [0,1] or [-1,1]).")
        print("=> Likely double-normalization or unexpected preprocessing; inspect tokenizer/vision processor.")


if __name__ == "__main__":
    main()
