import os
import json
import numpy as np
import argparse

# --- CONFIG ---
INPUT_DIR = "data/dataset"
OUTPUT_DIR = "data/nitrogen_rl"
LOOKAHEAD_FRAMES = 90   # 1.5 seconds (Chips can be slow)
CHARGE_THRESHOLD = 2    # Max charge level

def process_replay(replay_path):
    frames = []
    try:
        with open(replay_path, 'r') as f:
            for line in f:
                if line.strip():
                    frames.append(json.loads(line))
    except: return []
    
    if not frames: return []

    total_frames = len(frames)
    
    # --- 1. EXTRACT DATA TO NUMPY ---
    # HP & Charge
    e_hp = np.array([f.get('enemy_health', 1000) or 1000 for f in frames])
    p_hp = np.array([f.get('player_health', 1000) or 1000 for f in frames])
    p_charge = np.array([f.get('player_charge', 0) or 0 for f in frames])
    e_charge = np.array([f.get('enemy_charge', 0) or 0 for f in frames])
    
    # Buttons: Extract "A" button (Assume index 0)
    # Shape: [N_Frames]
    a_buttons = np.array([1 if (f.get('joy_buttons') and f['joy_buttons'][0]) else 0 for f in frames])
    
    # Calculate Edges
    # A_Press: 0 -> 1
    a_press_events = np.zeros(total_frames, dtype=bool)
    a_press_events[1:] = (a_buttons[:-1] == 0) & (a_buttons[1:] == 1)

    # Calculate Damage Events (Positive delta = damage taken)
    e_dmg_events = np.maximum(0, e_hp[:-1] - e_hp[1:]) 
    p_dmg_events = np.maximum(0, p_hp[:-1] - p_hp[1:])
    
    # Pad last frame
    e_dmg_events = np.append(e_dmg_events, 0)
    p_dmg_events = np.append(p_dmg_events, 0)

    tagged_samples = []

    # --- 2. FRAME ANALYSIS ---
    for i in range(total_frames - 1):
        weight = 1.0 # Default Baseline
        
        # Lookahead window
        end_idx = min(i + LOOKAHEAD_FRAMES, total_frames)
        
        # Current State
        curr_pc = p_charge[i]
        prev_pc = p_charge[i-1] if i > 0 else 0
        
        # --- EVENT 1: CHARGE SHOT (Release) ---
        # Logic: Charge was high, now it's 0.
        did_release_charge = (prev_pc >= CHARGE_THRESHOLD and curr_pc == 0)
        
        if did_release_charge:
            damage = np.sum(e_dmg_events[i:end_idx])
            if damage > 0:
                weight = 3.0 # High reward for landing charge shot
            else:
                weight = 0.0 # Punish missing charge shot

        # --- EVENT 2: CHIP / BUSTER (Press) ---
        # Logic: We pressed A, and we weren't just releasing a charge.
        did_press_a = a_press_events[i]
        
        if did_press_a and not did_release_charge:
            damage = np.sum(e_dmg_events[i:end_idx])
            
            if damage > 0:
                # Reward scales with damage.
                # Buster (1 dmg) -> 1.01 (Tiny boost)
                # Chip (200 dmg) -> 3.0 (Huge boost)
                weight = 1.0 + (damage / 100.0)
                
                # Cap extremely high rewards to prevent instability
                weight = min(weight, 5.0)
            else:
                # We fired and missed.
                # Strictly punish to teach trigger discipline.
                weight = 0.1 

        # --- EVENT 3: DODGE (Defensive) ---
        # Logic: Enemy released charge
        prev_ec = e_charge[i-1] if i > 0 else 0
        curr_ec = e_charge[i]
        enemy_fired = (prev_ec >= CHARGE_THRESHOLD and curr_ec == 0)
        
        if enemy_fired:
            dodge_win = min(i + 90, total_frames)
            taken = np.sum(p_dmg_events[i:dodge_win])
            
            if taken == 0:
                weight = max(weight, 2.0) # Reward dodge (use max to stack with offense)
            else:
                weight *= 0.5 # Penalty for getting hit

        # Save
        if weight != 1.0:
            tagged_samples.append({
                "replay": os.path.basename(os.path.dirname(replay_path)),
                "frame_idx": i,
                "weight": round(float(weight), 2)
            })

    return tagged_samples

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    all_samples = []
    print(f"Scanning {INPUT_DIR}...")
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file == "actions.jsonl":
                path = os.path.join(root, file)
                samples = process_replay(path)
                all_samples.extend(samples)
    
    # Save Manifest (True JSONL Format)
    out_path = os.path.join(OUTPUT_DIR, "frame_weights.jsonl")
    
    print(f"Writing {len(all_samples)} records to {out_path}...")
    
    with open(out_path, 'w') as f:
        for s in all_samples:
            # Clean, minimal record per line
            record = {
                "key": f"{s['replay']}/{s['frame_idx']}",
                "val": s['weight']
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"✅ Done! Weights saved to {out_path}")

if __name__ == "__main__":
    main()