import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
DATASET_DIR = "data/dataset_cached"
SEQ_LEN = 8  # Total frames to show
PRE_ROLL = 2 # How many frames to show BEFORE the button press (for context)
TARGET_BTN_IDX = 14 # Index of 'SOUTH' in the list below

BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE', 
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER', 
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST', 
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP' 
]

def show_sequence_with_south():
    # 1. Find Files
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.pt")))
    if not files:
        print(f"❌ No .pt files found in {DATASET_DIR}")
        return

    # Shuffle files so we don't always look at the same one
    np.random.shuffle(files)
    
    found_sequence = False
    frames = None
    actions = None
    start_idx = 0
    filename = ""

    print(f"🔍 Scanning for SOUTH (Index {TARGET_BTN_IDX}) presses...")

    # 2. Search for a file containing the action
    for f_path in files:
        try:
            # Load light (map to CPU)
            data = torch.load(f_path, map_location="cpu", weights_only=True)
            temp_actions = data.get('actions') # [T, 25] or [T, 21]
            
            # Slice the specific button column
            # If 25 dims (sticks included), buttons start at 4. So SOUTH is 4 + 14 = 18?
            # Let's double check standard layout:
            # Sticks=4. Then buttons match BUTTON_TOKENS. 
            # So SOUTH is at index 4 + 14 = 18.
            
            if temp_actions.shape[1] == 25:
                # With sticks
                btn_col = temp_actions[:, 4 + TARGET_BTN_IDX]
            else:
                # Buttons only
                btn_col = temp_actions[:, TARGET_BTN_IDX]

            # Find indices where press > 0.4
            # We use > 0.4 to match game threshold
            matches = (btn_col > 0.4).nonzero(as_tuple=True)[0]
            
            if len(matches) > 0:
                # Found one!
                # Pick a random occurrence in this file
                # Ensure we have enough room for SEQ_LEN
                valid_matches = matches[(matches >= PRE_ROLL) & (matches < len(temp_actions) - (SEQ_LEN - PRE_ROLL))]
                
                if len(valid_matches) > 0:
                    center_idx = valid_matches[np.random.randint(0, len(valid_matches))].item()
                    
                    # Set our start point (rewind a bit to see context)
                    start_idx = center_idx - PRE_ROLL
                    frames = data.get('frames')
                    actions = temp_actions
                    filename = os.path.basename(f_path)
                    found_sequence = True
                    break # Stop searching
        except Exception as e:
            print(f"⚠️ Error reading {f_path}: {e}")
            continue

    if not found_sequence:
        print("❌ Could not find ANY 'SOUTH' presses in the scanned files.")
        return

    print(f"✅ Found SOUTH press in {filename} at frame {start_idx + PRE_ROLL}")
    print(f"🎬 Showing frames {start_idx} to {start_idx + SEQ_LEN}")

    # 4. Setup Plot
    fig, axes = plt.subplots(1, SEQ_LEN, figsize=(20, 5))
    if SEQ_LEN == 1: axes = [axes]

    for i in range(SEQ_LEN):
        idx = start_idx + i
        ax = axes[i]
        
        # --- DECODE IMAGE ---
        # 1. Float division [0-255] -> [0.0-1.0]
        img_tensor = frames[idx].float() / 255.0
        # 2. CHW -> HWC
        img = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        
        # --- DECODE ACTIONS ---
        btn_tensor = actions[idx].float()
        if len(btn_tensor) == 25:
            btns = btn_tensor[4:]
        else:
            btns = btn_tensor

        active = []
        is_south_frame = False
        
        for b_i, val in enumerate(btns):
            if val > 0.4:
                name = BUTTON_TOKENS[b_i].replace("DPAD_", "").replace("SHOULDER", "SHLDR")
                active.append(name)
                if b_i == TARGET_BTN_IDX:
                    is_south_frame = True
        
        # --- RENDER ---
        ax.imshow(img)
        ax.axis('off')
        
        # Title Styling
        title = f"#{idx}\n"
        if active:
            title += "\n".join(active)
            if is_south_frame:
                # Highlights specific frame where button is pressed
                ax.set_title(title, fontsize=10, color='red', fontweight='bold', backgroundcolor='#ffeeee')
                # Add border to image
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                ax.set_title(title, fontsize=10, color='black')
        else:
            ax.set_title(f"#{idx}\n(Neutral)", fontsize=10, color='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_sequence_with_south()