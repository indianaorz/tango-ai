# utils.py
import random
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

#import DESCRETE_ACTIONS and KEY_BIT_POSITIONS from your config file
from config import DISCRETE_ACTIONS, KEY_BIT_POSITIONS

def int_to_binary_string(value):
    return format(value, '016b')

def generate_random_action_for_skip_strategy(key_bit_positions, random_action_keys):
    binary_command = 0
    num_keys_to_press = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2], k=1)[0]
    if num_keys_to_press > 0:
        selected_keys = random.sample(random_action_keys, num_keys_to_press)
        for key_name in selected_keys:
            if key_name in key_bit_positions:
                binary_command |= (1 << key_bit_positions[key_name])
    return int_to_binary_string(binary_command)

def map_discrete_action_to_buttons(action_index,
                                   discrete_actions_map,
                                   key_bit_positions_map):
    """
    Maps a discrete action index to a 16‑bit button mask string.

    HOLD_* sets the bit this frame; RELEASE_* returns 0 so the strategy
    can drop the persistent hold.
    """
    if action_index < 0 or action_index >= len(discrete_actions_map):
        print(f"Warning: Action index {action_index} out of bounds. Defaulting to NO_OP.")
        return format(0, "016b")

    name   = discrete_actions_map[action_index]
    mask   = 0
    kb     = key_bit_positions_map  # shorthand

    # ── directional & single‑frame buttons ───────────────────────────
    if name == "UP":       mask |= 1 << kb["UP"]
    elif name == "DOWN":   mask |= 1 << kb["DOWN"]
    elif name == "LEFT":   mask |= 1 << kb["LEFT"]
    elif name == "RIGHT":  mask |= 1 << kb["RIGHT"]
    elif name == "X":      mask |= 1 << kb["X"]
    elif name == "Z":      mask |= 1 << kb["Z"]
    elif name == "LEFT_X":
        mask |= (1 << kb["LEFT"]) | (1 << kb["X"])
    elif name == "RIGHT_X":
        mask |= (1 << kb["RIGHT"]) | (1 << kb["X"])
    elif name == "UP_X":
        mask |= (1 << kb["UP"]) | (1 << kb["X"])
    elif name == "DOWN_X":
        mask |= (1 << kb["DOWN"]) | (1 << kb["X"])

    # ── “sticky” actions (handled by DRLAgentStrategy) ───────────────
    elif name in ("HOLD_X", "HOLD_Z"):
        # holding just sets the bit this frame; persistence added later
        if name.endswith("X"):
            mask |= 1 << kb["X"]
        else:
            mask |= 1 << kb["Z"]

    elif name in ("RELEASE_X", "RELEASE_Z", "NO_OP"):
        mask = 0  # explicit no‑buttons frame

    else:
        print(f"Warning: Unmapped action '{name}'. Sending NO_OP.")

    return format(mask, "016b")



# ----------------------------------------------------------------------
#  Discrete‑action  ⇆  16‑bit button‑mask  conversions
# ----------------------------------------------------------------------
def _build_bitmask_to_idx(discrete_actions, key_bit_positions):
    """Pre‑compute {bitmask:int → action_idx:int} once at start‑up."""
    table = {}
    for idx in range(len(discrete_actions)):
        mask_str = map_discrete_action_to_buttons(
            idx, discrete_actions, key_bit_positions
        )
        table[int(mask_str, 2)] = idx
    return table

# Call **once** when the module is imported
BITMASK_TO_ACTION_IDX = _build_bitmask_to_idx(
    DISCRETE_ACTIONS, KEY_BIT_POSITIONS
)

def bitmask_to_action_index(bitmask: int) -> int:
    """Returns the discrete‑action index for a pressed‑button mask."""
    # Unknown combos fall back to NO_OP (index 0)
    return BITMASK_TO_ACTION_IDX.get(bitmask, 0)




def preprocess_frame(frame_pil_image, height, width):
    """
    Converts a PIL image to a [C,H,W] float tensor in **RGB**.
    Scales pixels to [0,1].  If image is missing, returns zeros.
    """
    if frame_pil_image is None:
        return torch.zeros((3, height, width), dtype=torch.float32)

    img = frame_pil_image.convert("RGB")                # keep colour
    img = TF.resize(img, [height, width], antialias=True)
    img_tensor = TF.to_tensor(img)                      # [3,H,W], float32 0‑1
    return img_tensor

def normalize_value(value, current_max, default_max=1.0):
    """Normalizes a value, updating current_max if value is higher."""
    if value is None: return 0.0
    true_max = max(current_max, value, default_max) # Ensure current_max doesn't shrink below default_max
    return float(value) / true_max, true_max

def get_grid_coordinates(pixel_pos, game_width=240, game_height=160, grid_cols=6, grid_rows=3):
    """Converts pixel coordinates to approximate grid cell coordinates (1-indexed)."""
    # This is a placeholder. You'll need to fine-tune based on your game's actual panel layout.
    # The example data had 'player_grid_position': [6, 2] directly. If that's always available, use it.
    if pixel_pos is None: return None
    
    # Example simple mapping if player_grid_position is not directly available
    # These values (20, 518) seem to be from a different coordinate system or scaled.
    # Assuming your game panels:
    # X: 0-3 (player side), 4-7 (enemy side) -- needs calibration
    # Y: 0-2 (rows)
    
    # If using the server-provided grid positions:
    # return pixel_pos # if pixel_pos is already like [col, row]

    # If you must convert from raw pixel x,y:
    # panel_width = game_width / grid_cols
    # panel_height = game_height / grid_rows
    # col = int(pixel_pos[0] / panel_width) + 1
    # row = int(pixel_pos[1] / panel_height) + 1
    # return [min(grid_cols, max(1, col)), min(grid_rows, max(1, row))]
    return None # Placeholder until coordinates are clear