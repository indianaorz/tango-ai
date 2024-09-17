# utils.py
import os
import glob


def get_root_dir():
    return '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO'

def get_image_memory():
    return 1

def get_threshold():
    return 0.1


default_checkpoint_path = get_root_dir() + '/checkpoints'
#checkpoint path
def get_checkpoint_path(checkpoint_dir = default_checkpoint_path, image_memory = 1):
    """
    Returns the path to the latest checkpoint in the given directory.
    """
    #path should be checkpoints10
    checkpoint_path = checkpoint_dir + str(image_memory)
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*.pt'))
    if not checkpoint_files:
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

