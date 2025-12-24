import argparse
import json
import os
from typing import List, Dict

# Configuration
DEFAULT_INPUT_DIR = "data/dataset"
OUTPUT_DIR = "data/chipwindows"
OUTPUT_FILENAME = "strategy.jsonl"

def get_files(input_path: str) -> List[str]:
    """
    Recursively find all actions.jsonl files within subfolders.
    Example match: data/dataset/replay_123/actions.jsonl
    """
    if os.path.isfile(input_path):
        return [input_path]
    
    files = []
    if not os.path.exists(input_path):
        print(f"Warning: Input directory '{input_path}' does not exist.")
        return []
        
    # os.walk automatically dives into every subfolder (e.g., data/dataset/{replay_name}/)
    for root, _, filenames in os.walk(input_path):
        for filename in filenames:
            if filename == "actions.jsonl":
                files.append(os.path.join(root, filename))
    return sorted(files)

def process_file(filepath: str) -> List[Dict]:
    """
    Parses a single replay file to extract turns.
    Includes logic to mask hidden chips, TRIM selection, and track USED CROSSES.
    """
    turns = []
    replay_name = os.path.basename(os.path.dirname(filepath))
    
    # State tracking
    in_window = False
    match_used_crosses = set() # <--- NEW: Track used crosses for this match
    
    # Data for the current turn being recorded
    current_turn_input = None     # Hand state at start of window
    current_turn_selection = None # Selection state (chips + cross) at end of window
    
    # Reward tracking
    accumulating_reward = False
    damage_dealt_sum = 0
    damage_taken_sum = 0
    
    # Telemetry for diffing
    prev_p_hp = None
    prev_e_hp = None

    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- 1. Extract Basic Telemetry ---
            curr_in_window = row.get("inside_window", False)
            p_hp = row.get("player_health", 0)
            e_hp = row.get("enemy_health", 0)
            
            if p_hp is None: p_hp = prev_p_hp or 0
            if e_hp is None: e_hp = prev_e_hp or 0

            # --- 2. Calculate Damage ---
            p_dmg = 0
            e_dmg = 0
            if prev_p_hp is not None and prev_e_hp is not None:
                diff_p = prev_p_hp - p_hp
                diff_e = prev_e_hp - e_hp
                if 0 < diff_p < 2000: p_dmg = diff_p
                if 0 < diff_e < 2000: e_dmg = diff_e

            prev_p_hp = p_hp
            prev_e_hp = e_hp

            # --- 3. Accumulate Rewards ---
            if accumulating_reward:
                damage_dealt_sum += e_dmg
                damage_taken_sum += p_dmg

            # --- 4. State Machine ---
            
            # CASE A: Window Just Opened (Battle -> Menu)
            if curr_in_window and not in_window:
                # 1. Finish PREVIOUS turn
                if current_turn_input and current_turn_selection:
                    
                    # Trim indices based on count
                    count = current_turn_selection.get('count', 0)
                    raw_indices = current_turn_selection.get('indices', [])
                    valid_indices = [x for x in raw_indices[:count] if x != 255]

                    turn_record = {
                        "replay_file": replay_name,
                        "frame_start": current_turn_input['frame'],
                        "hand_slots": current_turn_input['chip_slots'],
                        "hand_codes": current_turn_input['chip_codes'],
                        "p_hp_start": current_turn_input['p_hp'],
                        "e_hp_start": current_turn_input['e_hp'],
                        "cust_gauge": current_turn_input['cust_gauge'],
                        
                        # Context
                        "used_crosses": current_turn_input.get('used_crosses', []), # <--- NEW Output
                        
                        # Action
                        "selected_indices": valid_indices,
                        "selected_count": len(valid_indices),
                        "selected_cross": current_turn_selection['cross'],
                        
                        # Outcome
                        "damage_dealt": damage_dealt_sum,
                        "damage_taken": damage_taken_sum,
                        "net_yield": damage_dealt_sum - damage_taken_sum
                    }
                    turns.append(turn_record)

                # 2. Reset for NEW turn
                damage_dealt_sum = 0
                damage_taken_sum = 0
                accumulating_reward = False 
                
                # 3. Capture Input State
                raw_slots = row.get("chip_slots", [])
                raw_codes = row.get("chip_codes", [])
                visible_count = row.get("chip_visible_count", 5)

                masked_slots = []
                masked_codes = []
                for i in range(10): 
                    if i < len(raw_slots) and i < visible_count:
                        masked_slots.append(raw_slots[i])
                        masked_codes.append(raw_codes[i])
                    else:
                        masked_slots.append(255)
                        masked_codes.append(255)

                current_turn_input = {
                    "frame": row.get("frame_idx", line_idx),
                    "chip_slots": masked_slots,
                    "chip_codes": masked_codes,
                    "p_hp": p_hp,
                    "e_hp": e_hp,
                    "cust_gauge": row.get("cust_gauge", 0),
                    "used_crosses": list(match_used_crosses) # <--- NEW: Capture context snapshot
                }
                current_turn_selection = None
                
            # CASE B: Window Just Closed (Menu -> Battle)
            elif not curr_in_window and in_window:
                accumulating_reward = True
                
                # <--- NEW: Update Used Crosses based on the decision just made
                if current_turn_selection:
                    selected_c = current_turn_selection.get('cross', 0)
                    if selected_c > 0:
                        match_used_crosses.add(selected_c)

            # CASE C: Inside Window
            if curr_in_window:
                current_turn_selection = {
                    "indices": row.get("selected_chip_indices", []),
                    "count": row.get("chip_select_count", 0),
                    "cross": row.get("selected_cross_index", 0)
                }

            in_window = curr_in_window

    # End of file cleanup
    if accumulating_reward and current_turn_input and current_turn_selection:
         count = current_turn_selection.get('count', 0)
         raw_indices = current_turn_selection.get('indices', [])
         valid_indices = [x for x in raw_indices[:count] if x != 255]

         turn_record = {
            "replay_file": replay_name,
            "frame_start": current_turn_input['frame'],
            "hand_slots": current_turn_input['chip_slots'],
            "hand_codes": current_turn_input['chip_codes'],
            "p_hp_start": current_turn_input['p_hp'],
            "e_hp_start": current_turn_input['e_hp'],
            "cust_gauge": current_turn_input['cust_gauge'],
            "used_crosses": current_turn_input.get('used_crosses', []), # <--- NEW Output
            "selected_indices": valid_indices,
            "selected_count": len(valid_indices),
            "selected_cross": current_turn_selection['cross'],
            "damage_dealt": damage_dealt_sum,
            "damage_taken": damage_taken_sum,
            "net_yield": damage_dealt_sum - damage_taken_sum
        }
         turns.append(turn_record)
         
    return turns


def main():
    parser = argparse.ArgumentParser(description="Extract Chip Strategy Dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help=f"Path to dataset folder (default: {DEFAULT_INPUT_DIR})")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    print(f"Scanning for replays in: {args.input}")
    files = get_files(args.input)
    print(f"Found {len(files)} replay files.")

    total_turns = 0
    with open(output_path, 'w') as out_f:
        for fpath in files:
            turns = process_file(fpath)
            for t in turns:
                out_f.write(json.dumps(t) + "\n")
            total_turns += len(turns)
            if len(turns) > 0:
                print(f"  + {os.path.basename(os.path.dirname(fpath))}: {len(turns)} turns")

    print(f"\n✅ Done! Extracted {total_turns} strategy samples.")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    main()