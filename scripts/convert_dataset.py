import json
import argparse
from collections import OrderedDict

# --- CONFIGURATION ---
GBA_BITS = {
    0: 'A', 1: 'B', 2: 'SELECT', 3: 'START',
    4: 'RIGHT', 5: 'LEFT', 6: 'UP', 7: 'DOWN',
    8: 'R', 9: 'L'
}

GBA_TO_NITROGEN = {
    'A': 'EAST', 'B': 'SOUTH', 'L': 'LEFT_SHOULDER', 'R': 'RIGHT_SHOULDER',
    'START': 'START', 'SELECT': 'BACK',
    'UP': 'DPAD_UP', 'DOWN': 'DPAD_DOWN', 'LEFT': 'DPAD_LEFT', 'RIGHT': 'DPAD_RIGHT'
}

# Fields that must be physically swapped if the perspective is Player 2
SWAP_PAIRS = [
    ("player_health", "enemy_health"),
    ("player_pos", "enemy_pos"),
    ("player_charge", "enemy_charge"),
    ("player_chip", "enemy_chip"),
    ("player_hand", "enemy_hand"),
    ("player_folder", "enemy_folder"),
    ("player_code_folder", "enemy_code_folder"),
    ("player_tag_chips", "enemy_tag_chips"),
    ("player_reg_chip", "enemy_reg_chip"),
    ("player_navi_cust", "enemy_navi_cust"),
]

# Base Action Template (Nitrogen Standard)
NITROGEN_TEMPLATE = OrderedDict([
    ("WEST", 0.0), ("SOUTH", 0.0), ("BACK", 0.0),
    ("DPAD_DOWN", 0.0), ("DPAD_LEFT", 0.0), ("DPAD_RIGHT", 0.0), ("DPAD_UP", 0.0),
    ("GUIDE", 0.0), ("AXIS_LEFTX", [0.0]), ("AXIS_LEFTY", [0.0]),
    ("LEFT_SHOULDER", 0.0), ("LEFT_TRIGGER", [0.0]),
    ("AXIS_RIGHTX", [0.0]), ("AXIS_RIGHTY", [0.0]),
    ("LEFT_THUMB", 0.0), ("RIGHT_THUMB", 0.0),
    ("RIGHT_SHOULDER", 0.0), ("RIGHT_TRIGGER", [0.0]),
    ("START", 0.0), ("EAST", 0.0), ("NORTH", 0.0),
])

def parse_input_bitmask(bitmask_int: int):
    action = NITROGEN_TEMPLATE.copy()
    if not isinstance(bitmask_int, int): bitmask_int = 0
    for bit, gba_btn in GBA_BITS.items():
        if (bitmask_int >> bit) & 1:
            nitro_key = GBA_TO_NITROGEN.get(gba_btn)
            if nitro_key: action[nitro_key] = 1.0
    return action

def extract_state_v1(obj: dict, swap: bool) -> dict:
    state_wrapper = obj.get("state", {})
    st = state_wrapper.get("V1", state_wrapper) if isinstance(state_wrapper, dict) else {}
    if not st: return {}

    if swap:
        st = st.copy()
        for k1, k2 in SWAP_PAIRS:
            v1 = st.get(k1)
            v2 = st.get(k2)
            st[k1] = v2
            st[k2] = v1
    return st

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--swap-players", action="store_true", help="Swap P1/P2 telemetry data")
    args = parser.parse_args()

    count = 0
    prev_hp = {"p": None, "e": None}

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                
                # 1. Inputs
                joyflags = data.get("input", 0)
                if isinstance(joyflags, dict): joyflags = joyflags.get("local", 0)
                final_row = parse_input_bitmask(int(joyflags or 0))
                
                # 2. Metadata
                final_row["frame_idx"] = data.get("frame", count)
                final_row["tick"] = data.get("tick", count)

                # 3. State Extraction & Swap
                st = extract_state_v1(data, args.swap_players)
                
                # 4. MERGE ALL FIELDS (Passthrough)
                for k, v in st.items():
                    final_row[k] = v

                # 5. Calculate Damage Metrics
                p_hp = final_row.get("player_health")
                e_hp = final_row.get("enemy_health")
                curr_p = p_hp if isinstance(p_hp, (int, float)) else prev_hp["p"]
                curr_e = e_hp if isinstance(e_hp, (int, float)) else prev_hp["e"]

                if curr_p is not None and prev_hp["p"] is not None:
                    diff = prev_hp["p"] - curr_p
                    final_row["player_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["player_damage_taken"] = 0

                if curr_e is not None and prev_hp["e"] is not None:
                    diff = prev_hp["e"] - curr_e
                    final_row["enemy_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["enemy_damage_taken"] = 0

                prev_hp["p"] = curr_p
                prev_hp["e"] = curr_e

                fout.write(json.dumps(final_row) + "\n")
                count += 1

            except Exception:
                pass

    print(f"Converted {count} frames.")

if __name__ == "__main__":
    main()