# utils.py
import random
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

#import DESCRETE_ACTIONS and KEY_BIT_POSITIONS from your config file
from config import DISCRETE_ACTIONS, KEY_BIT_POSITIONS, BUTTON_ALIAS


def _bit(kb: dict, logical: str) -> int:
    """
    Convert logical button name -> actual bit position.
    A -> Z, B -> X, START -> RETURN.
    """
    physical = BUTTON_ALIAS.get(logical, logical)
    if physical not in kb:
        raise KeyError(f"Unknown key '{logical}' (physical='{physical}') in KEY_BIT_POSITIONS")
    return 1 << kb[physical]


def int_to_binary_string(value):
    return format(value, '016b')

def generate_random_action_for_skip_strategy(key_bit_positions, random_action_keys):
    binary_command = 0
    num_keys_to_press = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2], k=1)[0]
    if num_keys_to_press > 0:
        selected_keys = random.sample(random_action_keys, num_keys_to_press)
        for logical in selected_keys:
            physical = BUTTON_ALIAS.get(logical, logical)
            if physical in key_bit_positions:
                binary_command |= (1 << key_bit_positions[physical])
    return int_to_binary_string(binary_command)



def map_discrete_action_to_buttons(action_index,
                                   discrete_actions_map,
                                   key_bit_positions_map):
    """
    Maps a discrete action index to a 16-bit button mask string.

    Supported:
      NO_OP
      UP/DOWN/LEFT/RIGHT
      A (chip) / B (shoot) / START
      UP_A etc, UP_B etc
    """
    if action_index < 0 or action_index >= len(discrete_actions_map):
        return format(0, "016b")

    name = discrete_actions_map[action_index]
    kb = key_bit_positions_map
    mask = 0

    if name == "NO_OP":
        mask = 0

    elif name in ("UP", "DOWN", "LEFT", "RIGHT"):
        mask |= _bit(kb, name)

    elif name in ("A", "B", "START"):
        mask |= _bit(kb, name)

    elif "_" in name:
        # combos: DIR_A / DIR_B
        parts = name.split("_")
        if len(parts) == 2:
            d, btn = parts
            if d in ("UP", "DOWN", "LEFT", "RIGHT") and btn in ("A", "B"):
                mask |= _bit(kb, d)
                mask |= _bit(kb, btn)
            else:
                mask = 0
        else:
            mask = 0
    else:
        mask = 0

    return format(mask, "016b")

# ----------------------------------------------------------------------
#  Discrete-action  ⇆  16-bit button-mask conversions
# ----------------------------------------------------------------------
def _build_bitmask_to_idx(discrete_actions, key_bit_positions):
    table = {}
    for idx in range(len(discrete_actions)):
        mask_str = map_discrete_action_to_buttons(idx, discrete_actions, key_bit_positions)
        table[int(mask_str, 2)] = idx
    return table

BITMASK_TO_ACTION_IDX = _build_bitmask_to_idx(DISCRETE_ACTIONS, KEY_BIT_POSITIONS)

def bitmask_to_action_index(bitmask: int) -> int:
    return BITMASK_TO_ACTION_IDX.get(bitmask, 0)



def preprocess_frame(frame_pil_image, height, width):
    """
    Converts a PIL image to a [C,H,W] float tensor in **RGB**.
    Matches training logic: 
    1. Force Resize to Native GBA (240x160) using Nearest Neighbor.
    2. Center-pad onto the target canvas (e.g., 256x256).
    """
    if frame_pil_image is None:
        return torch.zeros((3, height, width), dtype=torch.float32)

    # 1. Define Native GBA Resolution
    NATIVE_W, NATIVE_H = 240, 160
    
    img = frame_pil_image.convert("RGB")
    
    # 2. Resize to Native (Nearest Neighbor preserves pixel art crispness)
    # This handles cases where the incoming stream might be scaled differently
    if img.size != (NATIVE_W, NATIVE_H):
        img = img.resize((NATIVE_W, NATIVE_H), resample=Image.NEAREST)
    
    # 3. Create Black Canvas (Target Size)
    new_img = Image.new("RGB", (width, height), (0, 0, 0))
    
    # 4. Paste in Center
    left = (width - NATIVE_W) // 2
    top = (height - NATIVE_H) // 2
    new_img.paste(img, (left, top))
    
    # 5. Convert to Tensor [3, H, W] (Scales 0-255 -> 0.0-1.0 automatically)
    img_tensor = TF.to_tensor(new_img)
    
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