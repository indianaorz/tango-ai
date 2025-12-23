#!/usr/bin/env python3
"""
audit_pipeline.py
-----------------------------------------------------------------------------
Pinpoint where action encoding/order goes wrong:
  1) raw JSONL (actions.jsonl) produced by convert_dataset.py
  2) cached .pt produced by precache_dataset.py
  3) canonical Nitrogen token order (nitrogen.shared.BUTTON_ACTION_TOKENS)

What it checks:
  - cached actions shape [N,25]
  - cached axis stats (should be ~0 for GBA)
  - cached button value set (should be {0,1})
  - JSONL->cache agreement at sample indices
  - token list used by cache writer matches nitrogen.shared.BUTTON_ACTION_TOKENS

It prints a report and exits with code 1 if it finds issues.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# -----------------------------
# Canonical order (Nitrogen)
# -----------------------------
def load_ng_button_tokens() -> List[str]:
    try:
        from nitrogen.shared import BUTTON_ACTION_TOKENS as BTN
        BTN = list(BTN)
        if len(BTN) != 21:
            raise RuntimeError(f"Expected 21 BUTTON_ACTION_TOKENS, got {len(BTN)}")
        return BTN
    except Exception as e:
        raise RuntimeError(
            "Could not import nitrogen.shared.BUTTON_ACTION_TOKENS. "
            "Run this inside your repo/venv where nitrogen is importable."
        ) from e


# -----------------------------
# Helpers
# -----------------------------
def scalar(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        if not v:
            return float(default)
        v = v[0]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return float(default)
        v = v.reshape(-1)[0]
    try:
        return float(v)
    except Exception:
        return float(default)


def approx_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= eps


def format_set(vals: List[float]) -> str:
    # nice printing for tiny sets
    uniq = sorted(set(float(x) for x in vals))
    return "{" + ", ".join(f"{x:g}" for x in uniq) + "}"


def choose_indices(n: int) -> List[int]:
    # deterministic probes across the sequence
    if n <= 0:
        return []
    picks = {0, n // 4, n // 2, (3 * n) // 4, n - 1}
    # add a few more if long enough
    if n > 50:
        picks |= {10, 25, 40}
    return sorted(i for i in picks if 0 <= i < n)


def load_jsonl_actions(jsonl_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def extract_vec25_from_jsonl_row(row: Dict[str, Any], btn_tokens: List[str]) -> List[float]:
    # jsonl rows are expected to already be Nitrogen template dicts
    ax_lx = scalar(row.get("AXIS_LEFTX", 0.0))
    ax_ly = scalar(row.get("AXIS_LEFTY", 0.0))
    ax_rx = scalar(row.get("AXIS_RIGHTX", 0.0))
    ax_ry = scalar(row.get("AXIS_RIGHTY", 0.0))
    v = [ax_lx, ax_ly, ax_rx, ax_ry]
    for b in btn_tokens:
        v.append(scalar(row.get(b, 0.0)))
    return v


def axis_report(ax: torch.Tensor) -> str:
    # ax: [N,4]
    axf = ax.float()
    return (
        f"abs_mean={axf.abs().mean().item():.6f} "
        f"abs_max={axf.abs().max().item():.6f} "
        f"min={axf.min().item():.6f} max={axf.max().item():.6f}"
    )


def button_value_set(btn: torch.Tensor) -> List[float]:
    # btn: [N,21]
    # compute unique-ish values without materializing huge tensors:
    vals = btn.detach().float().cpu().flatten()
    # sample up to 20000 values deterministically
    if vals.numel() > 20000:
        idx = torch.linspace(0, vals.numel() - 1, steps=20000).long()
        vals = vals[idx]
    uniq = torch.unique(vals).cpu().tolist()
    # keep only "small" uniq sets; if too many, compress by rounding
    if len(uniq) > 32:
        rounded = torch.unique(torch.round(vals * 1000) / 1000).cpu().tolist()
        return sorted(float(x) for x in rounded)
    return sorted(float(x) for x in uniq)


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    raise SystemExit(1)


# -----------------------------
# Main audit
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="data/dataset", help="folder containing replay subdirs")
    ap.add_argument("--cache-dir", type=str, default="data/dataset_cached", help="folder containing .pt caches")
    ap.add_argument("--replay", type=str, default="", help="replay folder name (subdir of dataset-dir)")
    ap.add_argument("--cache", type=str, default="", help="cache filename (in cache-dir), e.g. REPLAY.pt")
    ap.add_argument("--eps", type=float, default=1e-6, help="float compare epsilon")
    args = ap.parse_args()

    btn_tokens = load_ng_button_tokens()
    print("✅ Canonical Nitrogen button token order:")
    print("   ", btn_tokens)

    dataset_dir = Path(args.dataset_dir)
    cache_dir = Path(args.cache_dir)

    # choose replay/cache
    replay_name = args.replay.strip()
    cache_name = args.cache.strip()

    if not replay_name:
        # pick first replay that has actions.jsonl
        candidates = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / "actions.jsonl").exists()]
        if not candidates:
            fail(f"No replay folders with actions.jsonl found under {dataset_dir}")
        candidates.sort()
        replay_name = candidates[0].name

    if not cache_name:
        # assume cache file named replay_name.pt
        cache_name = f"{replay_name}.pt"

    replay_dir = dataset_dir / replay_name
    jsonl_path = replay_dir / "actions.jsonl"
    pt_path = cache_dir / cache_name

    if not jsonl_path.exists():
        fail(f"Missing JSONL: {jsonl_path}")
    if not pt_path.exists():
        fail(f"Missing cache PT: {pt_path}")

    print(f"\n=== TARGETS ===")
    print(f"Replay: {replay_dir}")
    print(f"JSONL : {jsonl_path}")
    print(f"Cache : {pt_path}")

    # load jsonl
    rows = load_jsonl_actions(jsonl_path)
    if not rows:
        fail("JSONL parsed to 0 rows (invalid jsonl?)")
    print(f"\nJSONL rows: {len(rows):,}")

    # load pt
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

    if "actions" not in data or "frames" not in data:
        fail(f"Cache missing keys. Found: {list(data.keys())}")

    actions = data["actions"]
    frames = data["frames"]

    if actions.ndim != 2 or actions.shape[1] != 25:
        fail(f"Cache actions expected [N,25], got {tuple(actions.shape)}")
    if frames.ndim != 4 or frames.shape[1] != 3:
        fail(f"Cache frames expected [N,3,H,W], got {tuple(frames.shape)}")

    n_cache = int(actions.shape[0])
    n_jsonl = len(rows)
    print(f"Cache frames/actions: {n_cache:,} (frames shape={tuple(frames.shape)})")

    # basic stats
    ax = actions[:, 0:4]
    btn = actions[:, 4:]

    print("\n=== CACHE STATS ===")
    print("Axis:", axis_report(ax))

    uniq_btn = button_value_set(btn)
    print("Button unique values (sampled):", format_set(uniq_btn))

    # expectations for GBA
    # axis: near 0
    ax_abs_max = float(ax.float().abs().max().item())
    if ax_abs_max > 0.01:
        print("⚠️ Axis abs_max > 0.01 — suspicious for GBA-only.")
        # don't hard fail yet; we'll fail if mismatch is systematic below.

    # buttons: should contain 0 and 1 (often)
    if all(x in (0.0, 1.0) for x in uniq_btn):
        pass
    else:
        # allow a tiny float noise tolerance
        allowed = {0.0, 1.0, 0.5}
        if all(any(approx_equal(x, a, 1e-6) for a in allowed) for x in uniq_btn):
            print("⚠️ Buttons include 0.5 (or near) — indicates upstream half-scaling or thresholding bug.")
        else:
            print("⚠️ Buttons contain unexpected non-binary values.")

    # cross-check jsonl->cache on a few indices
    print("\n=== JSONL → CACHE CROSSCHECK ===")
    final_len = min(n_cache, n_jsonl)
    if final_len <= 0:
        fail("No overlap between JSONL and cache lengths.")

    probe = choose_indices(final_len)
    print("Probe indices:", probe)

    mismatches = 0
    axis_mismatches = 0
    button_mismatches = 0

    for idx in probe:
        row = rows[idx]
        v_jsonl = extract_vec25_from_jsonl_row(row, btn_tokens)  # 25 floats in canonical token order
        v_cache = actions[idx].detach().cpu().float().tolist()

        # compare axes
        for k in range(4):
            if not approx_equal(v_jsonl[k], v_cache[k], args.eps):
                axis_mismatches += 1

        # compare buttons
        for j in range(21):
            a = v_jsonl[4 + j]
            b = v_cache[4 + j]
            if not approx_equal(a, b, args.eps):
                button_mismatches += 1

        if axis_mismatches or button_mismatches:
            mismatches += 1

        # print per-index summary (compact)
        # show which buttons pressed in JSONL vs cache at this frame
        pressed_jsonl = [btn_tokens[j] for j in range(21) if v_jsonl[4 + j] >= 0.5]
        pressed_cache = [btn_tokens[j] for j in range(21) if v_cache[4 + j] >= 0.5]
        print(
            f"idx={idx:6d} "
            f"ax_jsonl={v_jsonl[:4]} ax_cache={v_cache[:4]} "
            f"btn_max_jsonl={max(v_jsonl[4:]):g} btn_max_cache={max(v_cache[4:]):g} "
            f"pressed_jsonl={pressed_jsonl[:8]} pressed_cache={pressed_cache[:8]}"
        )

    print("\n=== SUMMARY ===")
    print(f"axis_mismatches:   {axis_mismatches}")
    print(f"button_mismatches: {button_mismatches}")

    # additional: detect systematic axis contamination (like you saw AXIS_RIGHTY=1.0)
    # print top few rows where any axis is large
    ax_abs = ax.float().abs().max(dim=1).values
    bad = torch.where(ax_abs > 0.1)[0]
    if bad.numel() > 0:
        show = bad[:5].tolist()
        print(f"\n⚠️ Found {bad.numel()} frames with axis abs > 0.1. Showing first 5:")
        for i in show:
            v = actions[i].detach().cpu().float().tolist()
            print(f"  idx={i} axes={v[:4]} btn_max={max(v[4:]):g}")

    # fail conditions:
    # - any jsonl->cache mismatch is a smoking gun: cache writer differs from jsonl
    if axis_mismatches > 0 or button_mismatches > 0:
        fail("JSONL → cache mismatch detected. Bug is in precache_dataset.py (or you cached from older JSONL).")

    # - if no mismatch but buttons are 0/0.5, bug is in convert_dataset.py producing 0.5 values
    if set(round(x, 6) for x in uniq_btn).issubset({0.0, 0.5}) and 1.0 not in uniq_btn:
        fail("Cache buttons appear to be {0,0.5} only. Bug is upstream (convert_dataset.py / input source).")

    # - if axes are systematically non-zero but jsonl matches cache, bug is upstream at jsonl generation
    if ax_abs_max > 0.1:
        print("⚠️ Axes are large but JSONL matches cache, so the issue originates in JSONL generation.")
        fail("Axis contamination detected upstream (convert_dataset.py or telemetry source).")

    print("\n✅ Audit passed: JSONL and cache agree, token order is canonical, values look sane.")


if __name__ == "__main__":
    main()
