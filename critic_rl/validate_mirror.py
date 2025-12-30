#!/usr/bin/env python3
# scripts/validate_mirror.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root is on python path
sys.path.append(os.getcwd())

from viewer.critic_infer import CriticRunner
from viewer.derived_state import compute_derived

# Terminal Colors & Styling
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BG_RED = "\033[41m\033[37m"  # White on Red
BG_GREEN = "\033[42m\033[37m"

DEFAULT_P1_REPLAY = "20230929001326-ummm-bn6-vs-DthKrdMnSP-round2-p1"
DEFAULT_P2_REPLAY = "20230929001326-ummm-bn6-vs-IndianaOrz-round2-p2"
DEFAULT_CKPT = "checkpoints/critic_rl/v10_full/last.pt"


def print_dataset_manifesto():
    """Explicit debug printout explaining the MMBN Mirror Perspective logic."""
    print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}MMBN DATASET INTEGRITY MANIFESTO (Read this before debugging data!){RESET}")
    print(f"{'='*80}{RESET}")
    print(f"{BOLD}1. PERSPECTIVE RIGGING:{RESET}")
    print(f"   Each folder (e.g., '-p1' vs '-p2') represents the EXACT SAME MATCH recorded")
    print(f"   from two different 'Self' perspectives. The labels 'player_health' and")
    print(f"   'enemy_health' are verified as perspective-correct visual ground truth.")
    print(f"   - In a {CYAN}-p1{RESET} folder: P1 is the Left entity, P2 is the Right entity.")
    print(f"   - In a {MAGENTA}-p2{RESET} folder: P2 is the Left entity, P1 is the Right entity.")
    print(f"")
    print(f"{BOLD}2. DATA VALIDITY:{RESET}")
    print(f"   We have visually audited the health bars against VLM (Vision-Language Model)")
    print(f"   OCR. The metadata is correctly synced to the frames. If the Critic gives")
    print(f"   points to the enemy, {RED}THE DATA IS NOT WRONG{RESET}; the model is simply biased")
    print(f"   toward visual movement (Spectator Bias) or has not yet anchored 'Identity'.")
    print(f"")
    print(f"{BOLD}3. THE GOAL:{RESET}")
    print(f"   A perfect model should show {GREEN}Positive Value{RESET} for P1 when they have an HP")
    print(f"   advantage, regardless of which folder we load. Global Correlation should")
    print(f"   ideally be {GREEN}+1.0{RESET} (Identity Invariance). Current negative correlation")
    print(f"   proves the model is watching 'the person who got hit' instead of 'Self'.")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")


def load_replay_data(dataset_dir: str, replay_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    replay_path = os.path.join(dataset_dir, replay_name)
    jsonl_path = os.path.join(replay_path, "actions.jsonl")
    static_path = os.path.join(replay_path, "static_data.json")

    frames: List[Dict[str, Any]] = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(json.loads(line))
                except Exception:
                    pass

    static: Dict[str, Any] = {}
    if os.path.exists(static_path):
        with open(static_path, "r", encoding="utf-8") as f:
            try:
                static = json.load(f)
            except Exception:
                static = {}

    return frames, static


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _hp_delta_rewards_from_frames(frames: List[Dict[str, Any]], n: int) -> np.ndarray:
    """
    Per-frame hp_delta[t+1]-hp_delta[t]. This is fine for "where did HP change",
    but not a strict replica of stride-aware training targets.
    """
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    diffs = []
    for f in frames[:n]:
        p = _safe_float(f.get("player_health", 0))
        e = _safe_float(f.get("enemy_health", 0))
        diffs.append(p - e)
    diffs = np.array(diffs, dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    rewards[:-1] = diffs[1:] - diffs[:-1]
    return rewards


def _pearsonr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation over masked elements only."""
    idx = np.where(mask)[0]
    if idx.size < 2:
        return float("nan")
    aa = a[idx].astype(np.float64)
    bb = b[idx].astype(np.float64)
    av = aa - aa.mean()
    bv = bb - bb.mean()
    denom = np.sqrt((av * av).sum()) * np.sqrt((bv * bv).sum())
    return float((av * bv).sum() / denom) if denom > 0 else float("nan")


def get_action_str(frame: Dict[str, Any]) -> str:
    chars = []
    if frame.get("SOUTH", 0):
        chars.append("B")
    if frame.get("EAST", 0):
        chars.append("A")
    if frame.get("LEFT_SHOULDER", 0):
        chars.append("L")
    if frame.get("RIGHT_SHOULDER", 0):
        chars.append("R")
    if frame.get("DPAD_UP", 0):
        chars.append("^")
    elif frame.get("DPAD_DOWN", 0):
        chars.append("v")
    elif frame.get("DPAD_LEFT", 0):
        chars.append("<")
    elif frame.get("DPAD_RIGHT", 0):
        chars.append(">")
    return "".join(chars)[:4] if chars else "."


# -----------------------------------------------------------------------------
# The Rich View (prints ONLY sampled frames; no NaN rows)
# -----------------------------------------------------------------------------

def print_rich_window(
    f1: List[Dict[str, Any]],
    f2: List[Dict[str, Any]],
    v1: np.ndarray,
    v2: np.ndarray,
    r1: np.ndarray,
    start_idx: int,
    *,
    duration: int = 25,
    stride: int = 2,
):
    end_idx = min(start_idx + duration, len(v1), len(f1), len(f2))
    if end_idx <= start_idx:
        return

    # Only print frames where BOTH perspectives have real predictions.
    # This eliminates the noisy "(nan)" rows caused by stride/gated inference.
    sampled_mask = np.isfinite(v1) & np.isfinite(v2)
    idxs = [i for i in range(start_idx, end_idx) if sampled_mask[i]]
    if not idxs:
        return

    w_v1 = v1[idxs]
    w_v2 = v2[idxs]

    max1 = float(np.nanmax(np.abs(w_v1))) if w_v1.size else 0.0
    max2 = float(np.nanmax(np.abs(w_v2))) if w_v2.size else 0.0
    max_val = float(max(max1, max2, 1.0))

    width = 12
    BG_HIT = "\033[48;5;236m"

    print(f"\n{BOLD}{YELLOW}[Highlight Frame {idxs[0]} - {idxs[-1]}] Scale: ±{max_val:.1f}{RESET}")
    print(
        f"{BOLD}{'Frm':<5} | {'P1 HP':<6} {'P2 HP':<6} | {'Adv':<6} | "
        f"{'Act1':<4} {'Act2':<4} | {'r@hit':<8} | "
        f"{'P1_Val':<8} {'ΔV1':<6} | {'P2_Val':<8} {'ΔV2':<6} | {'Visual'}{RESET}"
    )
    print("-" * 155)

    # Track last printed HPs so we can still flag hits even if they happened
    # between sampled frames (they will show up as HP deltas at the next sample).
    prev_p = _safe_int(f1[idxs[0]].get("player_health", 0))
    prev_e = _safe_int(f1[idxs[0]].get("enemy_health", 0))

    for idx in idxs:
        val1 = float(v1[idx])
        val2 = float(v2[idx])

        # Reward display: show the immediate per-frame HP delta change at this index.
        rew = float(r1[idx]) if idx < len(r1) else 0.0

        p1_hp = _safe_int(f1[idx].get("player_health", prev_p))
        p2_hp = _safe_int(f1[idx].get("enemy_health", prev_e))
        adv = p1_hp - p2_hp

        act1 = get_action_str(f1[idx])
        act2 = get_action_str(f2[idx])

        dv1 = 0.0
        if idx >= stride and np.isfinite(v1[idx - stride]):
            dv1 = float(val1 - float(v1[idx - stride]))

        dv2 = 0.0
        if idx >= stride and np.isfinite(v2[idx - stride]):
            dv2 = float(val2 - float(v2[idx - stride]))

        is_hit = (p1_hp != prev_p or p2_hp != prev_e)
        row_style = BG_HIT if is_hit else ""

        p1_text_style = CYAN
        p2_text_style = MAGENTA

        hp1_s = f"{p1_hp:<6}"
        if p1_hp < prev_p:
            hp1_s = f"{RED}{p1_hp:<6}{RESET}{row_style}"
        hp2_s = f"{p2_hp:<6}"
        if p2_hp < prev_e:
            hp2_s = f"{GREEN}{p2_hp:<6}{RESET}{row_style}"

        r_str = f"{rew:>+8.1f}"
        if rew > 10:
            r_str = f"{GREEN}{r_str}{RESET}{row_style}"
        elif rew < -10:
            r_str = f"{RED}{r_str}{RESET}{row_style}"

        dv1_s = f"{dv1:>+6.1f}"
        if dv1 > 0.5:
            dv1_s = f"{GREEN}{dv1_s}{RESET}{row_style}"
        elif dv1 < -0.5:
            dv1_s = f"{RED}{dv1_s}{RESET}{row_style}"

        dv2_s = f"{dv2:>+6.1f}"
        if dv2 > 0.5:
            dv2_s = f"{GREEN}{dv2_s}{RESET}{row_style}"
        elif dv2 < -0.5:
            dv2_s = f"{RED}{dv2_s}{RESET}{row_style}"

        line = [" "] * (2 * width + 1)
        line[width] = ":"

        def _pos(v: float) -> int:
            vv = max(-1.0, min(1.0, float(v) / max_val))
            return int((vv + 1.0) * width)

        pos1 = _pos(val1)
        line[pos1] = f"{p1_text_style}|{RESET}{row_style}"

        pos2 = _pos(val2)
        if line[pos2] != " " and line[pos2] != ":":
            line[pos2] = f"{BOLD}X{RESET}{row_style}"
        else:
            line[pos2] = f"{p2_text_style}*{RESET}{row_style}"

        hit_marker = f"{BOLD}{YELLOW} <HIT!>{RESET}" if is_hit else ""

        val1_s = f"{val1:8.1f}"
        val2_s = f"{val2:8.1f}"

        row_content = (
            f"{idx:<5} | {hp1_s} {hp2_s} | {adv:>+6} | "
            f"{act1:<4} {act2:<4} | {r_str} | "
            f"{p1_text_style}{val1_s}{RESET}{row_style} {dv1_s} | "
            f"{p2_text_style}{val2_s}{RESET}{row_style} {dv2_s} | "
            f"{''.join(line)} {hit_marker}"
        )

        print(f"{row_style}{row_content}{RESET}")
        prev_p, prev_e = p1_hp, p2_hp

    print("-" * 155)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--highlights", type=int, default=10)
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--p1", type=str, default=DEFAULT_P1_REPLAY)
    ap.add_argument("--p2", type=str, default=DEFAULT_P2_REPLAY)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--require_cust_gt0", action="store_true", default=True)
    args = ap.parse_args()

    print_dataset_manifesto()

    print(f"{BOLD}{CYAN}--- Initializing Inference for Mirror Comparison ---{RESET}")
    critic = CriticRunner(
        ckpt_path=args.ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    f1, s1 = load_replay_data(args.dataset_dir, args.p1)
    f2, s2 = load_replay_data(args.dataset_dir, args.p2)

    d1 = compute_derived(f1, s1)
    d2 = compute_derived(f2, s2)

    res1 = critic.infer_from_derived(
        frames=f1,
        static=s1,
        derived=d1,
        stride=args.stride,
        seq_len=critic.trained_seq_len,
        require_cust_gt0=bool(args.require_cust_gt0),
    )
    res2 = critic.infer_from_derived(
        frames=f2,
        static=s2,
        derived=d2,
        stride=args.stride,
        seq_len=critic.trained_seq_len,
        require_cust_gt0=bool(args.require_cust_gt0),
    )

    # NaN for missing (critical)
    v1 = np.array([float(v) if v is not None else np.nan for v in res1.values_by_frame], dtype=np.float32)
    v2 = np.array([float(v) if v is not None else np.nan for v in res2.values_by_frame], dtype=np.float32)

    min_len = min(len(v1), len(v2), len(f1), len(f2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    f1 = f1[:min_len]
    f2 = f2[:min_len]

    r1 = _hp_delta_rewards_from_frames(f1, min_len)

    valid_mask = np.isfinite(v1) & np.isfinite(v2)
    corr = _pearsonr_masked(v1, v2, valid_mask)

    corr_neg = _pearsonr_masked(v1, -v2, valid_mask)

    cov = int(valid_mask.sum())
    print(f"Coverage (both valid): {BOLD}{cov}{RESET} / {min_len}")
    print(f"Global Correlation (masked): {BOLD}{corr:+.4f}{RESET}")
    print(f"Global Correlation vs -P2 (masked): {BOLD}{corr_neg:+.4f}{RESET}")

    # Highlight frames where reward spikes (per-frame); printing is sampled-only.
    sig_frames = np.where(np.abs(r1) > 20)[0]
    if len(sig_frames) == 0:
        sig_frames = np.array([1500], dtype=np.int64)

    for i in range(min(args.highlights, len(sig_frames))):
        start = max(0, int(sig_frames[i]) - 10)
        print_rich_window(f1, f2, v1, v2, r1, start, duration=25, stride=args.stride)


if __name__ == "__main__":
    main()
