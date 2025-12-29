import argparse
import torch
import numpy as np
import sys
from critic_rl.dataset import InMemoryCriticRLTDDataset

# --- CONFIGURATION ---
# Expected Normalization Factors (for verification)
SCALAR_MAP = {
    0: ("Player HP", 1000.0),
    1: ("Enemy HP", 1000.0),
    2: ("Player Charge", 2.0),
    3: ("Enemy Charge", 2.0),
    4: ("Cust Gauge", 100.0),
    5: ("Inside Window", 1.0),
    6: ("Turn Index", 50.0),
    7: ("Player X", 750.0),
    8: ("Player Y", 750.0),
    9: ("Enemy X", 750.0),
    10: ("Enemy Y", 750.0),
    11: ("Window Happened", 1.0),
    12: ("Window Selected", 1.0),
    13: ("Window Beast", 1.0),
    14: ("Player Cross Ent", 1.0),
    15: ("Enemy Cross Ent", 1.0),
}

BEAST_FEAT_MAP = {
    0: "P Active",
    1: "P Turns Since",
    2: "P Ever",
    3: "E Active",
    4: "E Turns Since",
    5: "E Ever",
}

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def analyze_tensor(name, t, mask=None, is_discrete=False):
    """Generic analyzer for any tensor."""
    if mask is not None:
        # Flatten and apply mask
        # t: [N, T, ...] -> [N*T, ...]
        flat_mask = mask.view(-1)
        flat_t = t.reshape(-1, *t.shape[2:])
        data = flat_t[flat_mask]
    else:
        data = t.reshape(-1)

    if data.numel() == 0:
        print(f"{name:<20} | [EMPTY]")
        return

    d_type = str(data.dtype).replace("torch.", "")
    shape_str = f"{list(t.shape[1:])}" # Per-sequence shape

    # Fix for Boolean mean crash: Cast to float
    if "bool" in d_type.lower():
        data = data.float()

    if is_discrete or "int" in d_type or "long" in d_type:
        # Discrete Analysis
        uniq = torch.unique(data)
        num_uniq = len(uniq)
        vmin = data.min().item()
        vmax = data.max().item()
        
        # Check for padding zeros if relevant, but basic stats first
        sample = uniq[:8].tolist()
        sample_str = str(sample) + ("..." if num_uniq > 8 else "")
        
        print(f"{name:<25} | {d_type:<10} | Shape:{shape_str:<12} | Range:[{vmin},{vmax}] | Unique:{num_uniq:<5} | Samples:{sample_str}")
    else:
        # Continuous Analysis
        vmin = data.min().item()
        vmax = data.max().item()
        vmean = data.mean().item()
        vstd = data.std().item()
        
        # Anomaly check
        status = "OK"
        if torch.isnan(data).any(): status = "!! NAN !!"
        elif torch.isinf(data).any(): status = "!! INF !!"
        
        print(f"{name:<25} | {d_type:<10} | Shape:{shape_str:<12} | Min:{vmin:<8.2f} Max:{vmax:<8.2f} Mean:{vmean:<8.2f} | {status}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--folder_len", type=int, default=30)
    args = ap.parse_args()

    print(f"--- Loading Cache: {args.cache_dir} ---")
    
    try:
        # Load VALIDATION set (faster)
        ds = InMemoryCriticRLTDDataset(
            args.dataset_dir, 
            cache_dir=args.cache_dir, 
            stride=args.stride, 
            folder_len=args.folder_len, 
            seq_len=args.seq_len, 
            require_cust_gt0=True, 
            split="val",
            val_ratio=0.1
        )
    except Exception as e:
        print(f"FATAL: Could not load dataset. {e}")
        return

    # Global Valid Mask
    # All analysis will only look at 'valid' frames to avoid padding noise
    valid = ds.valid # [N, T]
    
    print_section("1. REWARDS (TARGETS)")
    r = ds.r
    r_flat = r[valid].float()
    
    raw_rmse = (r_flat ** 2).mean().sqrt()
    sym_r = torch.sign(r_flat) * torch.log1p(torch.abs(r_flat))
    sym_rmse = (sym_r ** 2).mean().sqrt()
    
    print(f"Count: {len(r_flat):,}")
    print(f"Min:   {r_flat.min().item():.2f}")
    print(f"Max:   {r_flat.max().item():.2f}")
    print(f"Mean:  {r_flat.mean().item():.4f}")
    print("-" * 40)
    print(f"Zero-Pred RMSE (Linear): {raw_rmse.item():.4f}")
    print(f"Zero-Pred RMSE (SymLog): {sym_rmse.item():.4f}  <-- If Val RMSE > this, model is failing")

    print_section("2. DISCRETE INPUTS (IDs/Indices)")
    # We iterate known keys to group them logically
    
    # Grid
    analyze_tensor("grid_tile", ds.x["grid_tile"], valid, is_discrete=True)
    analyze_tensor("grid_owner", ds.x["grid_owner"], valid, is_discrete=True)
    
    # Chips (IDs)
    analyze_tensor("hand_id", ds.x["hand_id"], valid, is_discrete=True)
    analyze_tensor("held_id", ds.x["held_id"], valid, is_discrete=True)
    analyze_tensor("folder_id_p", ds.x["folder_id_p"], valid, is_discrete=True)
    analyze_tensor("folder_id_e", ds.x["folder_id_e"], valid, is_discrete=True)
    
    # NEW: On Deck & Last Used
    if "current_chip_p" in ds.x:
        analyze_tensor("current_chip_p", ds.x["current_chip_p"], valid, is_discrete=True)
        analyze_tensor("current_chip_e", ds.x["current_chip_e"], valid, is_discrete=True)
    else:
        print("!! MISSING: current_chip_p (On Deck) - Rebuild Cache !!")

    if "last_used_id_p" in ds.x:
        analyze_tensor("last_used_id_p", ds.x["last_used_id_p"], valid, is_discrete=True)
    else:
        print("!! MISSING: last_used_id_p (Just Fired) - Rebuild Cache !!")

    # Codes
    analyze_tensor("hand_code", ds.x["hand_code"], valid, is_discrete=True)
    analyze_tensor("held_code", ds.x["held_code"], valid, is_discrete=True)
    
    # Cross
    analyze_tensor("active_cross_idx_p", ds.x["active_cross_idx_p"], valid, is_discrete=True)
    analyze_tensor("active_cross_idx_e", ds.x["active_cross_idx_e"], valid, is_discrete=True)

    print_section("3. CONTINUOUS / MASK INPUTS")
    
    # Masks (0.0 / 1.0 floats)
    analyze_tensor("hand_vis", ds.x["hand_vis"], valid)
    analyze_tensor("held_mask", ds.x["held_mask"], valid) # bool
    analyze_tensor("folder_used_p", ds.x["folder_used_p"], valid)
    analyze_tensor("folder_used_e", ds.x["folder_used_e"], valid)
    analyze_tensor("used_cross", ds.x["used_cross"], valid) # 22-dim vector

    # Beast Features (6-dim)
    print("\n--- Beast Features (6-dim) Breakdown ---")
    bf = ds.x["beast_feats"] # [N, T, 6]
    flat_bf = bf[valid] # [M, 6]
    for i in range(6):
        name = BEAST_FEAT_MAP.get(i, f"Feat {i}")
        col = flat_bf[:, i]
        print(f"  {i}: {name:<15} | Min:{col.min():.2f} Max:{col.max():.2f} Mean:{col.mean():.2f}")

    print_section("4. SCALAR TOWER (16-dim)")
    scalars = ds.x["scalars"] # [N, T, 32] -> but only 16 used currently
    flat_sc = scalars[valid]
    
    print(f"{'Idx':<4} {'Name':<20} {'NormFactor':<10} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Status'}")
    print("-" * 90)
    
    for i in range(16):
        name, factor = SCALAR_MAP.get(i, ("Unknown", 1.0))
        vals = flat_sc[:, i]
        vmin = vals.min().item()
        vmax = vals.max().item()
        vmean = vals.mean().item()
        vstd = vals.std().item()
        
        status = "OK"
        if vmax > 2.0: status = "(!High)" # Value > 2x expected norm
        if vstd == 0.0: status = "(Constant)"
        
        print(f"{i:<4} {name:<20} {factor:<10.1f} {vmin:<8.2f} {vmax:<8.2f} {vmean:<8.2f} {vstd:<8.2f} {status}")

    print("\nDone.")

if __name__ == "__main__":
    main()