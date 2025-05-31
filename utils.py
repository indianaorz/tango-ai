# utils.py
import random
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

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

def map_discrete_action_to_buttons(action_index, discrete_actions_map, key_bit_positions_map):
    """Maps an AI's discrete action index to a button command string."""
    if action_index < 0 or action_index >= len(discrete_actions_map):
        print(f"Warning: Action index {action_index} out of bounds. Defaulting to NO_OP.")
        return int_to_binary_string(0)

    action_name = discrete_actions_map[action_index]
    binary_command = 0

    if action_name == "NO_OP":
        pass
    elif action_name == "UP":
        binary_command |= (1 << key_bit_positions_map['UP'])
    elif action_name == "DOWN":
        binary_command |= (1 << key_bit_positions_map['DOWN'])
    elif action_name == "LEFT":
        binary_command |= (1 << key_bit_positions_map['LEFT'])
    elif action_name == "RIGHT":
        binary_command |= (1 << key_bit_positions_map['RIGHT'])
    elif action_name == "X":
        binary_command |= (1 << key_bit_positions_map['X'])
    elif action_name == "A":
        binary_command |= (1 << key_bit_positions_map['A'])
    elif action_name == "LEFT_X":
        binary_command |= (1 << key_bit_positions_map['LEFT'])
        binary_command |= (1 << key_bit_positions_map['X'])
    elif action_name == "RIGHT_X":
        binary_command |= (1 << key_bit_positions_map['RIGHT'])
        binary_command |= (1 << key_bit_positions_map['X'])
    elif action_name == "UP_X":
        binary_command |= (1 << key_bit_positions_map['UP'])
        binary_command |= (1 << key_bit_positions_map['X'])
    elif action_name == "DOWN_X":
        binary_command |= (1 << key_bit_positions_map['DOWN'])
        binary_command |= (1 << key_bit_positions_map['X'])
    # Add other mappings for new discrete actions here
    else:
        print(f"Warning: Unmapped action name '{action_name}'. Defaulting to NO_OP.")

    return int_to_binary_string(binary_command)



def preprocess_frame(frame_pil_image, height, width):
  """Preprocesses a single PIL image frame for the DRL model."""
  if frame_pil_image is None: # Handle cases where image might be missing
    return torch.zeros((1, height, width), dtype=torch.float32)
  img = frame_pil_image.convert("L") # Grayscale
  img = TF.resize(img, [height, width], antialias=True)
  img_tensor = TF.to_tensor(img) # Converts to [C, H, W] and scales to [0, 1]
  return img_tensor # Shape: [1, H, W]

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