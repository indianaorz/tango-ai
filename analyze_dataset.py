import os
import glob
import torch
import sys
import time

# --- CONFIG ---
DATASET_DIR = "data/dataset_cached"
PRESS_THRESHOLD = 0.5 

# Exact token list from your strategy_ng.py
BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER',
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST',
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP'
]

# ANSI Colors for Terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
GRAY = '\033[90m'
CYAN = '\033[96m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_dashboard(total_frames, button_counts, axis_counts, no_op_count, num_files, total_files):
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{BOLD}📊 Dataset Bias Analyzer{RESET}")
    print(f"📂 Scanning: {DATASET_DIR}")
    print(f"PROGRESS: [{num_files}/{total_files}] files | {total_frames:,} frames analyzed\n")

    if total_frames == 0: return

    # --- 1. IDLE / NO-OP PERCENTAGE ---
    no_op_pct = (no_op_count / total_frames) * 100
    color = GREEN if no_op_pct < 20 else (YELLOW if no_op_pct < 50 else RED)
    bar_len = int((no_op_pct / 100) * 40)
    bar = "█" * bar_len
    print(f"{CYAN}--- ACTIVITY DENSITY ---{RESET}")
    print(f"NO INPUT       | {color}{bar:<40}{RESET} | {no_op_pct:5.1f}% (Model learns to idle)")
    print("")

    # --- 2. DIRECTIONAL BIAS (Stick + Dpad) ---
    print(f"{CYAN}--- DIRECTIONAL BIAS (Stick + D-Pad) ---{RESET}")
    
    dirs = {
        "UP": (button_counts[4] + axis_counts["UP"]) / total_frames * 100,    # DPAD_UP idx 4
        "DOWN": (button_counts[1] + axis_counts["DOWN"]) / total_frames * 100, # DPAD_DOWN idx 1
        "LEFT": (button_counts[2] + axis_counts["LEFT"]) / total_frames * 100, # DPAD_LEFT idx 2
        "RIGHT": (button_counts[3] + axis_counts["RIGHT"]) / total_frames * 100 # DPAD_RIGHT idx 3
    }

    for name, pct in dirs.items():
        bar_len = int((pct / 100.0) * 40)
        bar = "█" * bar_len
        color = GRAY
        if pct > 10: color = YELLOW
        if pct > 30: color = GREEN
        print(f"{name:<14} | {color}{bar:<40}{RESET} | {pct:5.1f}%")
    
    print("")

    # --- 3. BUTTON BIAS ---
    print(f"{CYAN}--- BUTTON BIAS ---{RESET}")
    
    # Hide D-Pads (shown above) and obscure buttons
    HIDDEN_TOKENS = {'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 
                     'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP',
                     'LEFT_THUMB', 'RIGHT_THUMB', 'GUIDE', 'LEFT_TRIGGER', 'RIGHT_TRIGGER',
                     'WEST', 'NORTH'}

    for i, token in enumerate(BUTTON_TOKENS):
        if token in HIDDEN_TOKENS: continue

        count = button_counts[i].item()
        pct = (count / total_frames) * 100
        
        bar_len = int((pct / 100.0) * 40)
        bar = "█" * bar_len
        
        color = GRAY
        if pct > 5: color = YELLOW
        if pct > 20: color = GREEN
        
        print(f"{token:<14} | {color}{bar:<40}{RESET} | {pct:5.1f}% ({count:,.0f})")

    print("-" * 60)

def main():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.pt")))
    
    if not files:
        print(f"❌ No .pt files found in {DATASET_DIR}")
        return

    # 21 buttons
    total_button_counts = torch.zeros(21, dtype=torch.float32)
    axis_counts = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
    no_op_count = 0
    total_frames = 0

    for i, f in enumerate(files):
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            actions = data["actions"] # [N, 25]
            
            if actions.shape[0] == 0: continue

            # 1. Count Buttons
            # actions[:, 4:] are the 21 buttons
            btn_active = (actions[:, 4:] > PRESS_THRESHOLD)
            total_button_counts += btn_active.float().sum(dim=0)

            # 2. Count Axes
            # 0:LX, 1:LY
            left_x = actions[:, 0]
            left_y = actions[:, 1]
            
            up_mask = (left_y < -PRESS_THRESHOLD)
            down_mask = (left_y > PRESS_THRESHOLD)
            left_mask = (left_x < -PRESS_THRESHOLD)
            right_mask = (left_x > PRESS_THRESHOLD)

            axis_counts["UP"]    += up_mask.float().sum().item()
            axis_counts["DOWN"]  += down_mask.float().sum().item()
            axis_counts["LEFT"]  += left_mask.float().sum().item()
            axis_counts["RIGHT"] += right_mask.float().sum().item()

            # 3. Count NO_OPs (Frames where NOTHING is pressed)
            # A frame is NO_OP if no buttons are pressed AND no axis movement
            # We OR all the masks together. If result is False, it's a NO_OP.
            any_btn = btn_active.any(dim=1)
            any_axis = up_mask | down_mask | left_mask | right_mask
            is_active = any_btn | any_axis
            no_op_count += (~is_active).float().sum().item()

            total_frames += actions.shape[0]

            if i % 5 == 0 or i == len(files) - 1:
                print_dashboard(total_frames, total_button_counts, axis_counts, no_op_count, i+1, len(files))

        except Exception as e:
            print(f"⚠️ Error: {e}")

    print("\n✅ Complete.")

if __name__ == "__main__":
    main()