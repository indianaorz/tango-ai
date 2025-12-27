import argparse
import json
import os
from typing import List, Dict

# Configuration
DEFAULT_INPUT_DIR = "data/dataset"
OUTPUT_DIR = "data/chipwindows"
OUTPUT_FILENAME = "strategy.jsonl"

def get_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path): return [input_path]
    files = []
    if not os.path.exists(input_path):
        print(f"Warning: Input directory '{input_path}' does not exist.")
        return []
    for root, _, filenames in os.walk(input_path):
        for filename in filenames:
            if filename == "actions.jsonl":
                files.append(os.path.join(root, filename))
    return sorted(files)

def process_file(filepath: str) -> List[Dict]:
    turns = []
    replay_name = os.path.basename(os.path.dirname(filepath))
    
    in_window = False
    match_player_used_crosses = set()
    match_enemy_used_crosses = set()
    
    current_turn_input = None     
    current_turn_selection = None 
    
    accumulating_reward = False
    damage_dealt_sum = 0
    damage_taken_sum = 0
    prev_p_hp = None
    prev_e_hp = None

    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. State Tracking (History)
            p_cross = row.get("player_cross_id", 0)
            if p_cross > 0: match_player_used_crosses.add(p_cross)
            e_cross = row.get("enemy_cross_id", 0)
            if e_cross > 0: match_enemy_used_crosses.add(e_cross)

            # 2. Telemetry
            curr_in_window = row.get("inside_window", False)
            p_hp = row.get("player_health", 0)
            e_hp = row.get("enemy_health", 0)
            
            if p_hp is None: p_hp = prev_p_hp or 0
            if e_hp is None: e_hp = prev_e_hp or 0

            # 3. Reward
            p_dmg = 0
            e_dmg = 0
            if prev_p_hp is not None and prev_e_hp is not None:
                diff_p = prev_p_hp - p_hp
                diff_e = prev_e_hp - e_hp
                if 0 < diff_p < 2000: p_dmg = diff_p
                if 0 < diff_e < 2000: e_dmg = diff_e

            prev_p_hp = p_hp
            prev_e_hp = e_hp

            if accumulating_reward:
                damage_dealt_sum += e_dmg
                damage_taken_sum += p_dmg

            # 4. State Machine
            
            # CASE A: Window OPENED (Battle -> Menu)
            if curr_in_window and not in_window:
                # Finish PREVIOUS turn
                if current_turn_input and current_turn_selection:
                    count = current_turn_selection.get('count', 0)
                    raw_indices = current_turn_selection.get('indices', [])
                    valid_indices = [x for x in raw_indices[:count] if x != 255]

                    turn_record = {
                        "replay_file": replay_name,
                        "frame_start": current_turn_input['frame'],
                        "hand_slots": current_turn_input['chip_slots'],
                        "hand_codes": current_turn_input['chip_codes'],
                        "chip_visible_count": current_turn_input['chip_visible_count'],
                        "p_hp_start": current_turn_input['p_hp'],
                        "e_hp_start": current_turn_input['e_hp'],
                        
                        # Context
                        "player_used_crosses": current_turn_input['player_used_crosses'],
                        "enemy_used_crosses": current_turn_input['enemy_used_crosses'],
                        "current_cross": current_turn_input['current_cross'], # <--- SAVED
                        
                        # Board
                        "grid_state": current_turn_input['grid_state'],
                        "grid_owner_state": current_turn_input['grid_owner_state'],
                        "player_emotion": current_turn_input['player_emotion'],
                        "enemy_emotion": current_turn_input['enemy_emotion'],
                        "player_pos": current_turn_input['player_pos'],
                        "enemy_pos": current_turn_input['enemy_pos'],
                        "beast_mode": current_turn_input['beast_mode'],
                        
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

                # Reset
                damage_dealt_sum = 0
                damage_taken_sum = 0
                accumulating_reward = False 
                
                # Capture Snapshot
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
                    "chip_visible_count": visible_count,
                    "p_hp": p_hp,
                    "e_hp": e_hp,
                    "cust_gauge": row.get("cust_gauge", 0),
                    
                    # Context
                    "player_used_crosses": list(match_player_used_crosses),
                    "enemy_used_crosses": list(match_enemy_used_crosses),
                    # 🚀 CRITICAL ADDITION: Capture the exact state at this frame
                    "current_cross": row.get("player_cross_id", 0),
                    
                    # Board
                    "grid_state": row.get("grid_state", [0]*18),
                    "grid_owner_state": row.get("grid_owner_state", [0]*18),
                    "player_emotion": row.get("player_emotion", 0),
                    "enemy_emotion": row.get("enemy_emotion", 0),
                    "player_pos": row.get("player_pos", [0, 0]),
                    "enemy_pos": row.get("enemy_pos", [0, 0]),
                    "beast_mode": row.get("beast_mode", 0),
                }
                current_turn_selection = None
                
            # CASE B: Window CLOSED
            elif not curr_in_window and in_window:
                accumulating_reward = True
                if current_turn_selection:
                    selected_c = current_turn_selection.get('cross', 0)
                    if selected_c > 0: match_player_used_crosses.add(selected_c)

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
            "chip_visible_count": current_turn_input['chip_visible_count'],
            "p_hp_start": current_turn_input['p_hp'],
            "e_hp_start": current_turn_input['e_hp'],
            "player_used_crosses": current_turn_input['player_used_crosses'],
            "enemy_used_crosses": current_turn_input['enemy_used_crosses'],
            "current_cross": current_turn_input['current_cross'], # <--- SAVED
            "grid_state": current_turn_input['grid_state'],
            "grid_owner_state": current_turn_input['grid_owner_state'],
            "player_emotion": current_turn_input['player_emotion'],
            "enemy_emotion": current_turn_input['enemy_emotion'],
            "player_pos": current_turn_input['player_pos'],
            "enemy_pos": current_turn_input['enemy_pos'],
            "beast_mode": current_turn_input['beast_mode'],
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
            for t in turns: out_f.write(json.dumps(t) + "\n")
            total_turns += len(turns)
            if len(turns) > 0: print(f"  + {os.path.basename(os.path.dirname(fpath))}: {len(turns)} turns")

    print(f"\n✅ Done! Extracted {total_turns} strategy samples.")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    main()