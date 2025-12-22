import json
import sys
import argparse

# GBA Input Bitmasks
KEY_RIGHT = 1 << 4
KEY_LEFT  = 1 << 5
KEY_UP    = 1 << 6
KEY_DOWN  = 1 << 7

def get_input(obj):
    inp = obj.get("input", 0)
    if isinstance(inp, dict): return int(inp.get("local", 0))
    return int(inp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to raw .jsonl file")
    args = parser.parse_args()

    # Stats for Entity A (Player default)
    p_moves = 0
    p_matches = 0

    # Stats for Entity B (Enemy default)
    e_moves = 0
    e_matches = 0
    
    prev_p = None
    prev_e = None
    
    valid_frames = 0
    
    # Scan limit
    MAX_SCAN = 6000

    with open(args.file, "r") as f:
        for line in f:
            if valid_frames > MAX_SCAN: break
            
            try:
                row = json.loads(line)
                
                # --- PARSING ---
                state_wrapper = row.get("state", {})
                st = state_wrapper.get("V1", state_wrapper)
                
                if not isinstance(st, dict): continue
                if st.get("inside_window") is True: continue 
                
                joy = get_input(row)
                p_pos = st.get("player_pos")
                e_pos = st.get("enemy_pos")
                
                if not p_pos or not e_pos: continue
                
                curr_p = (p_pos[0], p_pos[1])
                curr_e = (e_pos[0], e_pos[1])

                if prev_p is not None:
                    # 1. Calculate Deltas
                    dpx = curr_p[0] - prev_p[0]
                    dpy = curr_p[1] - prev_p[1]
                    dex = curr_e[0] - prev_e[0]
                    dey = curr_e[1] - prev_e[1]
                    
                    # 2. Parse Inputs
                    in_r = bool(joy & KEY_RIGHT)
                    in_l = bool(joy & KEY_LEFT)
                    in_u = bool(joy & KEY_UP)
                    in_d = bool(joy & KEY_DOWN)

                    # --- RATIO LOGIC ---
                    
                    # Check Entity A (Player)
                    if abs(dpx) > 0 or abs(dpy) > 0:
                        p_moves += 1
                        # Did the move match the input?
                        is_match = False
                        if dpx > 0 and in_r: is_match = True
                        elif dpx < 0 and in_l: is_match = True
                        elif dpy < 0 and in_u: is_match = True
                        elif dpy > 0 and in_d: is_match = True
                        
                        if is_match: p_matches += 1

                    # Check Entity B (Enemy)
                    if abs(dex) > 0 or abs(dey) > 0:
                        e_moves += 1
                        is_match = False
                        if dex > 0 and in_r: is_match = True
                        elif dex < 0 and in_l: is_match = True
                        elif dey < 0 and in_u: is_match = True
                        elif dey > 0 and in_d: is_match = True
                        
                        if is_match: e_matches += 1

                    valid_frames += 1

                prev_p = curr_p
                prev_e = curr_e

            except Exception:
                continue

    # --- CALCULATION ---
    
    # Avoid division by zero
    ratio_p = (p_matches / p_moves) if p_moves > 10 else 0.0
    ratio_e = (e_matches / e_moves) if e_moves > 10 else 0.0
    
    print(f"   [Detector] P-Ratio: {ratio_p:.2%} ({p_matches}/{p_moves}) | E-Ratio: {ratio_e:.2%} ({e_matches}/{e_moves})")

    # Tie-breaker / Safety
    # If both are 0 (no movement), Keep.
    if ratio_p < 0.01 and ratio_e < 0.01:
        print("   -> Decision: KEEP (No valid movement data)")
        sys.exit(0)

    # Decision
    if ratio_e > ratio_p:
        print("   -> Decision: SWAP (Entity B obeys inputs better)")
        sys.exit(1)
    else:
        print("   -> Decision: KEEP (Entity A obeys inputs better)")
        sys.exit(0)

if __name__ == "__main__":
    main()