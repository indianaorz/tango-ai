import json
import argparse
from pathlib import Path
from collections import OrderedDict

# [FIX] Standard GBA Hardware Bitmask
# These are the bits mGBA uses internally
GBA_BITS = {
    0: 'A',
    1: 'B',
    2: 'SELECT',
    3: 'START',
    4: 'RIGHT',
    5: 'LEFT',
    6: 'UP',
    7: 'DOWN',
    8: 'R',         # Fixed: Bit 8 is Right Shoulder
    9: 'L'          # Fixed: Bit 9 is Left Shoulder
}

# Map GBA Buttons -> Nitrogen/Dataset Tokens
GBA_TO_NITROGEN = {
    'A': 'EAST',           # Nintendo A is Right (East)
    'B': 'SOUTH',          # Nintendo B is Bottom (South)
    'L': 'LEFT_SHOULDER',
    'R': 'RIGHT_SHOULDER',
    'START': 'START',
    'SELECT': 'BACK',
    'UP': 'DPAD_UP',
    'DOWN': 'DPAD_DOWN',
    'LEFT': 'DPAD_LEFT',
    'RIGHT': 'DPAD_RIGHT'
}

# Nitrogen Output Template (The schema expected by your model)
NITROGEN_TEMPLATE = OrderedDict([
    ("WEST", 0.0), ("SOUTH", 0.0), ("BACK", 0.0),
    ("DPAD_DOWN", 0.0), ("DPAD_LEFT", 0.0), ("DPAD_RIGHT", 0.0), ("DPAD_UP", 0.0),
    ("GUIDE", 0.0),
    ("AXIS_LEFTX", [0.0]), ("AXIS_LEFTY", [0.0]),
    ("LEFT_SHOULDER", 0.0), ("LEFT_TRIGGER", [0.0]),
    ("AXIS_RIGHTX", [0.0]), ("AXIS_RIGHTY", [0.0]),
    ("LEFT_THUMB", 0.0), ("RIGHT_THUMB", 0.0),
    ("RIGHT_SHOULDER", 0.0), ("RIGHT_TRIGGER", [0.0]),
    ("START", 0.0), ("EAST", 0.0), ("NORTH", 0.0),
])

def parse_input_bitmask(bitmask_int):
    # Initialize clean template
    action = NITROGEN_TEMPLATE.copy()
    # Ensure lists are new objects, not references
    action["AXIS_LEFTX"] = [0.0]
    action["AXIS_LEFTY"] = [0.0]
    action["AXIS_RIGHTX"] = [0.0]
    action["AXIS_RIGHTY"] = [0.0]
    action["LEFT_TRIGGER"] = [0.0]
    action["RIGHT_TRIGGER"] = [0.0]

    # Check every GBA bit
    for bit, gba_btn in GBA_BITS.items():
        if (bitmask_int >> bit) & 1:
            nitro_key = GBA_TO_NITROGEN.get(gba_btn)
            if nitro_key:
                action[nitro_key] = 1.0
                
    return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input raw JSONL from Rust")
    parser.add_argument("--output", type=str, required=True, help="Output Nitrogen JSONL")
    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")
    
    count = 0
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            try:
                data = json.loads(line)
                
                # Convert raw integer -> Nitrogen Dict
                nitrogen_action = parse_input_bitmask(data['input'])
                nitrogen_action['frame_idx'] = data['frame']
                
                fout.write(json.dumps(nitrogen_action) + "\n")
                count += 1
            except Exception as e:
                print(f"Skipping bad line: {e}")
                
    print(f"Converted {count} frames.")

if __name__ == "__main__":
    main()