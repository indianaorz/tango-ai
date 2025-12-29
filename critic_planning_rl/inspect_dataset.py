import argparse
import torch
import numpy as np
from .dataset import PlanningDataset

# --- CONFIGURATION ---
# Expected Normalization Factors (for verification)
SCALAR_CHECKS = {
    'p_hp':     ("Player HP", 1000.0), # Should be ~0.0 to ~2.0
    'e_hp':     ("Enemy HP", 1000.0),
    'turn_idx': ("Turn Index", 50.0),  # Should be small (0.02 - 0.5)
}

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def analyze_tensor(name, tensor_list, is_discrete=False):
    """
    Aggregates a list of tensors and prints stats.
    tensor_list: List of Tensors from the dataset loop
    """
    if not tensor_list:
        print(f"{name:<20} | [EMPTY]")
        return

    # Stack into one big tensor for analysis
    data = torch.stack(tensor_list).float() # Convert to float for stats calculation
    
    # Flatten [N, T, ...] -> [N*T]
    data = data.view(-1)

    if is_discrete:
        # Discrete Analysis
        uniq = torch.unique(data)
        num_uniq = len(uniq)
        vmin = data.min().item()
        vmax = data.max().item()
        
        # Sample values
        sample = uniq[:8].tolist()
        sample_str = str(sample) + ("..." if num_uniq > 8 else "")
        
        print(f"{name:<20} | Int/ID     | Range:[{int(vmin)},{int(vmax)}] | Unique:{num_uniq:<5} | Samples:{sample_str}")
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
        elif vmax > 5.0 and "target" not in name: status = "(!High)" # Warning for unnormalized inputs
        
        print(f"{name:<20} | Float      | Min:{vmin:<8.2f} Max:{vmax:<8.2f} Mean:{vmean:<8.2f} | {status}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to strategy_v2.jsonl")
    args = ap.parse_args()
    
    print(f"--- Loading Dataset: {args.jsonl} ---")
    ds = PlanningDataset(args.jsonl)
    print(f"Loaded {len(ds)} samples.")
    
    if len(ds) == 0:
        print("Dataset is empty.")
        return

    # 1. Collect all data into memory (Dataset is small, so this is fine)
    buffer = {}
    
    print("Aggregating statistics...")
    for i in range(len(ds)):
        sample = ds[i]
        for k, v in sample.items():
            buffer.setdefault(k, []).append(v)

    # 2. Analyze Groups
    
    # --- TARGETS ---
    print_section("1. OPTIMIZATION TARGET (Net Yield)")
    # Target is SymLog(Damage Dealt - Damage Taken)
    analyze_tensor("target", buffer['target'])
    
    # Calculate Zero-Baseline RMSE
    t_tensor = torch.stack(buffer['target'])
    rmse = torch.sqrt((t_tensor**2).mean())
    print(f"\n>> Zero-Pred RMSE (Goal to beat): {rmse:.4f}")
    print("   (If validation loss > this, model is guessing zero)")

    # --- SCALARS ---
    print_section("2. GLOBAL CONTEXT (Scalars)")
    for k in ['p_hp', 'e_hp', 'turn_idx']:
        analyze_tensor(k, buffer[k])

    # --- GRID ---
    print_section("3. BOARD STATE (Grid)")
    analyze_tensor("grid_tile", buffer['grid_tile'], is_discrete=True)
    analyze_tensor("grid_owner", buffer['grid_owner'], is_discrete=True)

    # --- HAND & DRAW ---
    print_section("4. AVAILABLE RESOURCES (Hand + Draw)")
    analyze_tensor("held_id", buffer['held_id'], is_discrete=True)
    analyze_tensor("held_code", buffer['held_code'], is_discrete=True)
    analyze_tensor("draw_id", buffer['draw_id'], is_discrete=True)
    analyze_tensor("draw_code", buffer['draw_code'], is_discrete=True)
    
    # Check if Draw has valid chips (non-zero)
    draw_ids = torch.stack(buffer['draw_id']).float()
    non_zero_draws = (draw_ids > 0).float().mean()
    print(f"\n>> Draw Slot Utilization: {non_zero_draws*100:.1f}% (slots with actual chips)")

    # --- ACTION ---
    print_section("5. PLAYER ACTION (Selection)")
    analyze_tensor("sel_id", buffer['sel_id'], is_discrete=True)
    analyze_tensor("sel_code", buffer['sel_code'], is_discrete=True)
    analyze_tensor("sel_cross", buffer['sel_cross'], is_discrete=True)
    analyze_tensor("sel_beast", buffer['sel_beast'], is_discrete=True)

    # --- MASKS ---
    print_section("6. HISTORICAL MASKS (0.0 - 1.0)")
    analyze_tensor("folder_p", buffer['folder_p'])
    analyze_tensor("folder_e", buffer['folder_e'])
    analyze_tensor("cross_hist_p", buffer['cross_hist_p'])
    
    print("\nDone.")

if __name__ == "__main__":
    main()