# utils.py
import os
import glob


def get_root_dir():
    return '../TANGO'#'/home/lee/tango' #'../TANGO'#'/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO'

def get_image_memory():
    return 1

def get_threshold():
    return 0.1

def get_threshold_plan():
    return 0.3

def inference_fps():
    return 60

default_checkpoint_path = get_root_dir() + '/checkpoints'
#checkpoint path
def get_checkpoint_path(checkpoint_dir = default_checkpoint_path, image_memory = 1):
    """
    Returns the path to the latest checkpoint in the given directory.
    """
    #path should be checkpoints/10/
    checkpoint_path =  os.path.join(checkpoint_dir, str(image_memory))
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*.pt'))
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_path}")
        return None
    return max(checkpoint_files, key=os.path.getctime)

def get_exponental_amount():
    return 4

# utils.py

def get_exponential_sample(indices_list, current_idx, image_memory):
    """
    Returns a list of frame indices sampled exponentially from the current index.
    Ensures exactly `image_memory` frames by allowing duplicates when necessary.

    The sampling order is:
        current frame, current-1, current-2, current-4, current-8, ...

    If indices go below 0, they are clamped to 0.
    The returned list is ordered from oldest to newest frame.
    """
    if not indices_list:
        # print("get_exponential_sample: indices_list is empty.")
        return []

    # Ensure indices_list is sorted in ascending order
    sorted_indices = sorted(indices_list)
    earliest_idx = sorted_indices[0]
    latest_idx = sorted_indices[-1]

    # print(f"get_exponential_sample: earliest_idx={earliest_idx}, latest_idx={latest_idx}")

    sampled_indices = []
    step = 1
    current = current_idx

    for _ in range(image_memory):
        if current < earliest_idx:
            current = earliest_idx
        sampled_indices.append(current)
        # print(f"get_exponential_sample: Appending index {current}")
        current -= step
        step *= get_exponental_amount()  # Exponentially increase the step (1, 2, 4, 8, ...)

    # Remove any excess frames
    if len(sampled_indices) > image_memory:
        sampled_indices = sampled_indices[:image_memory]

    # To maintain chronological order (oldest to newest), sort the indices
    sampled_indices_sorted = sorted(sampled_indices)
    # print(f"get_exponential_sample: sampled_indices_sorted={sampled_indices_sorted}")

    return sampled_indices_sorted

# utils.py

import math

def position_to_grid(x, y):
    """
    Converts a (x, y) position to a 2D grid with a 1 marking the player's location.

    Grid Configuration:
    - Columns: 3
    - Rows: 6
    - X-axis range: 20 to 220
    - Y-axis range: 258 to 773

    Args:
        x (float): The x-coordinate of the player's position.
        y (float): The y-coordinate of the player's position.

    Returns:
        list[list[int]]: A 6x3 grid with a 1 in the player's cell and 0s elsewhere.
    """
    import math

    # Grid boundaries and dimensions
    GRID_START_X = 20
    GRID_END_X = 220
    GRID_START_Y = 258
    GRID_END_Y = 773

    COLUMNS = 3  # Changed to 3 to match HDF5 expectation
    ROWS = 6     # Changed to 6 to match HDF5 expectation

    GRID_WIDTH = GRID_END_X - GRID_START_X  # 200
    GRID_HEIGHT = GRID_END_Y - GRID_START_Y  # 515

    COLUMN_WIDTH = GRID_WIDTH / COLUMNS  # ~66.666
    ROW_HEIGHT = GRID_HEIGHT / ROWS      # ~85.8333

    # Calculate column index
    col = math.floor((x - GRID_START_X) / COLUMN_WIDTH)
    # Calculate row index
    row = math.floor((y - GRID_START_Y) / ROW_HEIGHT)

    # if x or y is out of bounds, return an empty grid
    if col < 0 or col >= COLUMNS or row < 0 or row >= ROWS:
        return [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]

    # Initialize grid with 0s
    grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]

    # Mark the player's position with a 1
    grid[row][col] = 1

    return grid  # Shape: (6, 3)


# Example Usage
if __name__ == "__main__":
    # Example player position
    player_x = 20
    player_y = 5130

    grid = position_to_grid(player_x, player_y)
    for row in grid:
        print(row)
