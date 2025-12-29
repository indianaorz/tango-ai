import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import sys

# Ensure viewer/ is in python path
sys.path.append(os.getcwd())

from viewer.critic_infer import CriticRunner
from viewer.derived_state import compute_derived

DEFAULT_P1_REPLAY = "20230929001213-ummm-bn6-vs-DthKrdMnSP-round1-p1"
DEFAULT_P2_REPLAY = "20230929001213-ummm-bn6-vs-IndianaOrz-round1-p2"
DEFAULT_CKPT = "checkpoints/critic_rl/tdlam_complete_v5_final/last.pt"

def load_replay_data(dataset_dir, replay_name):
    replay_path = os.path.join(dataset_dir, replay_name)
    jsonl_path = os.path.join(replay_path, "actions.jsonl")
    static_path = os.path.join(replay_path, "static_data.json")
    
    frames = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    try: frames.append(json.loads(line))
                    except: pass
    static = {}
    if os.path.exists(static_path):
        with open(static_path, "r") as f:
            static = json.load(f)
    return frames, static

def print_ascii_trend(v1, v2, start_idx, steps=20):
    print(f"\n--- Tracking {steps} Frames (Frame {start_idx} to {start_idx+steps}) ---")
    print(f"{'Frm':<5} | {'P1 Val':<8} | {'P2 Val':<8} | {'Diff':<6} | {'Visual (P1=|, P2=*)'}")
    print("-" * 75)
    
    for i in range(steps):
        idx = start_idx + i
        if idx >= len(v1) or idx >= len(v2): break
        
        val1 = v1[idx]
        val2 = v2[idx]
        diff = val1 - val2
        
        # Simple ASCII Bar (-1.0 to 1.0 range usually)
        # Center is roughly 20 chars in
        center = 20
        scale = 10 
        
        p1_pos = int(center + (val1 * scale))
        p2_pos = int(center + (val2 * scale))
        
        line = [" "] * 45
        line[center] = ":" # Zero line
        
        # Place markers (clamped)
        p1_pos = max(0, min(44, p1_pos))
        p2_pos = max(0, min(44, p2_pos))
        
        if line[p1_pos] != " ": line[p1_pos] = "X" # Collision
        else: line[p1_pos] = "|"
            
        if line[p2_pos] == "|": line[p2_pos] = "X"
        else: line[p2_pos] = "*"
            
        visual = "".join(line)
        print(f"{idx:<5} | {val1:6.3f}   | {val2:6.3f}   | {diff:6.3f} | {visual}")

    print("-" * 75)
    print("Legend: '|' = Winner View, '*' = Loser View, ':' = Zero")
    print("Goal:   '|' should be Right (>0), '*' should be Left (<0)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", type=str, default=DEFAULT_P1_REPLAY)
    ap.add_argument("--p2", type=str, default=DEFAULT_P2_REPLAY)
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_start", type=int, default=1000)
    args = ap.parse_args()

    print(f"--- Mirror Validation ---")
    
    # 1. Load Model
    try:
        critic = CriticRunner(ckpt_path=args.ckpt, device=args.device, batch_seqs=256)
    except Exception as e:
        print(f"FATAL: {e}")
        return

    # 2. Process Replays
    print("Processing P1...")
    f1, s1 = load_replay_data(args.dataset_dir, args.p1)
    d1 = compute_derived(f1, s1)
    r1 = critic.infer_from_derived(frames=f1, static=s1, derived=d1, stride=1, seq_len=critic.trained_seq_len, require_cust_gt0=False)
    v1 = np.array([v if v is not None else 0.0 for v in r1.values_by_frame])

    print("Processing P2...")
    f2, s2 = load_replay_data(args.dataset_dir, args.p2)
    d2 = compute_derived(f2, s2)
    r2 = critic.infer_from_derived(frames=f2, static=s2, derived=d2, stride=1, seq_len=critic.trained_seq_len, require_cust_gt0=False)
    v2 = np.array([v if v is not None else 0.0 for v in r2.values_by_frame])

    # 3. Analyze
    min_len = min(len(v1), len(v2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    
    corr, _ = pearsonr(v1, v2)
    print(f"\nGlobal Correlation: {corr:.4f}")
    if corr > 0.5: print(">> STATUS: BIASED (High Positive Correlation)")
    elif corr < -0.5: print(">> STATUS: CORRECT (Negative Correlation)")
    else: print(">> STATUS: NOISY (Low Correlation)")

    # 4. Detailed Print
    print_ascii_trend(v1, v2, args.sample_start, steps=25)

if __name__ == "__main__":
    main()