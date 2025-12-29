import json
import os
import time
import numpy as np
import torch
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Valid imports from your codebase
from critic_rl.dataset import CriticRLTDDataset, collate_batch
from critic_rl.model import HPDeltaTDLambdaCritic, CriticRLConfig

# --- CONFIG ---
DATASET_DIR = "data/dataset"
CHECKPOINT = "checkpoints/critic_rl/tdlam_complete_v5_final/last.pt"
WORKERS = os.cpu_count() or 4
# How many replays to scan for the "Data Truth" (Set to 1000 or len(files) for full scan)
REPLAYS_TO_SCAN = 200 

# -------------------------------------------------------------------------
# 1. FAST WORKER (No Heavy Imports)
# -------------------------------------------------------------------------
def _worker_scan_replay(jsonl_path: Path) -> dict:
    """
    Scans a single replay file for HP changes.
    Skipping 'derived_state' makes this 100x faster.
    """
    rewards = []
    p_prev, e_prev = 0, 0
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                
                # Extract HP (with carry-forward logic implicit in dataset, 
                # but here we just grab what's present or default to 0 for simplicity)
                p = int(row.get("player_health", p_prev))
                e = int(row.get("enemy_health", e_prev))
                
                # Calculate Delta
                # hp_delta[t] = p - e
                # reward[t] = delta[t+1] - delta[t]
                
                # On the very first frame, we assume delta matches current state
                # but we can't calc reward until second frame.
                current_delta = p - e
                prev_delta = p_prev - e_prev
                
                # Reward is change in advantage
                r = current_delta - prev_delta
                rewards.append(r)
                
                p_prev, e_prev = p, e
                
    except Exception as e:
        return {"error": str(e)}

    # Filter out the first frame artifact usually
    if len(rewards) > 1:
        rewards = rewards[1:]
        
    rewards = np.array(rewards)
    return {
        "count": len(rewards),
        "zeros": np.sum(rewards == 0),
        "pos": np.sum(rewards > 0),
        "neg": np.sum(rewards < 0),
        "max": np.max(rewards) if len(rewards) else 0,
        "min": np.min(rewards) if len(rewards) else 0,
    }

# -------------------------------------------------------------------------
# 2. MAIN ROUTINE
# -------------------------------------------------------------------------
def audit():
    print(f"--- TURBO AUDIT: SPARSITY & MODEL ---")
    
    # --- PHASE 1: PARALLEL DATA SCAN ---
    print(f"\n[Phase 1] Scanning Raw Data ({WORKERS} workers)...")
    
    root = Path(DATASET_DIR)
    files = sorted(list(root.rglob("actions.jsonl")))
    
    if REPLAYS_TO_SCAN < len(files):
        files = files[:REPLAYS_TO_SCAN]
        
    print(f"Scanning {len(files)} replays...")
    
    total_frames = 0
    total_zeros = 0
    total_pos = 0
    total_neg = 0
    
    # Parallel execution
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(tqdm(pool.map(_worker_scan_replay, files), total=len(files)))
        
    for res in results:
        if "error" in res: continue
        total_frames += res["count"]
        total_zeros += res["zeros"]
        total_pos += res["pos"]
        total_neg += res["neg"]
        
    sparsity = (total_zeros / total_frames) * 100.0 if total_frames else 0
    
    print(f"\n=== DATA TRUTH (The Haystack) ===")
    print(f"Total Frames: {total_frames}")
    print(f"Zeros (Silence):      {total_zeros} ({sparsity:.2f}%)")
    print(f"Positives (Dealt):    {total_pos} ({(total_pos/total_frames)*100:.2f}%)")
    print(f"Negatives (Taken):    {total_neg} ({(total_neg/total_frames)*100:.2f}%)")
    
    if sparsity > 95.0:
        print(">> VERDICT: Dataset is EXTREMELY SPARSE.")
        print("   The model learns to predict 0.0 because 0.0 is correct >95% of the time.")
    else:
        print(">> VERDICT: Dataset is moderately dense.")

    # --- PHASE 2: MODEL CHECK (Sequential, GPU) ---
    print(f"\n[Phase 2] Checking Model Confidence ({CHECKPOINT})...")
    
    if not os.path.exists(CHECKPOINT):
        print("Checkpoint not found. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load just ONE replay fully using the real dataset class
    # We do this to get valid tensor inputs for the model
    ds = CriticRLTDDataset(
        root_dir=DATASET_DIR,
        stride=1,
        seq_len=64,
        max_samples=256, # Just need a few batches
        cache_replays=1
    )
    
    loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=collate_batch)
    
    # Load Model
    cfg = CriticRLConfig(max_seq_len=64)
    model = HPDeltaTDLambdaCritic(cfg, folder_len=30).to(device)
    
    ckpt = torch.load(CHECKPOINT, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    
    all_preds = []
    
    print("Running Inference Probe...")
    with torch.no_grad():
        for xb, r, done, valid in loader:
            xb = {k: v.to(device) for k, v in xb.items()}
            
            # Predict
            preds = model(xb) # [B, T]
            
            # Mask out padding/invalid frames
            mask = valid.bool().to(device)
            valid_preds = preds[mask]
            
            all_preds.extend(valid_preds.cpu().numpy().tolist())
            
    preds = np.array(all_preds)
    
    print(f"\n=== MODEL BEHAVIOR (N={len(preds)}) ===")
    print(f"Mean: {np.mean(preds):.6f}")
    print(f"Std:  {np.std(preds):.6f}")
    print(f"Range: [{np.min(preds):.6f}, {np.max(preds):.6f}]")
    
    if np.std(preds) < 0.05:
        print("\n>> DIAGNOSIS: 'Mean Reversion Plateau'")
        print("   The model is predicting ~0.0 for everything to stay safe.")
        print("   Keep training until Std increases.")
    else:
        print("\n>> DIAGNOSIS: Active Prediction")
        print("   The model is confidently differentiating states.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    audit()