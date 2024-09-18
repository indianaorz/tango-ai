# cache_data.py
import os
import json
import glob
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm  # For progress bars
import argparse  # For command-line arguments
from utils import get_root_dir, position_to_grid  # Ensure position_to_grid is imported

# Paths
TRAINING_DATA_DIR = get_root_dir() + '/training_data'
TRAINING_CACHE_DIR = get_root_dir() + '/training_cache'
WINDOW_SIZE = 600  # Number of frames to look ahead for rewards/punishments

def process_replay(replay_dir, output_dir=TRAINING_CACHE_DIR):
    # Extract the replay name
    replay_name = os.path.basename(replay_dir)
    cache_dir = os.path.join(output_dir, replay_name)

    # Check if the cache directory already exists and is non-empty
    if os.path.exists(cache_dir) and os.listdir(cache_dir):
        print(f"Cache already exists for {replay_name}, skipping this replay.")
        return  # Skip processing this replay as it is already cached

    os.makedirs(cache_dir, exist_ok=True)

    # Path to winner.json
    winner_file = os.path.join(replay_dir, 'winner.json')
    is_winner = None
    if os.path.exists(winner_file):
        with open(winner_file, 'r') as f:
            try:
                winner_data = json.load(f)
                is_winner = winner_data.get('is_winner', None)
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {winner_file}, skipping this folder.")
                return  # Skip processing this folder due to invalid JSON
    else:
        print(f"winner.json not found in {replay_dir}, skipping this folder.")
        return  # Skip processing this folder

    if is_winner is None:
        print(f"Winner status is undecided in {replay_dir}, skipping this folder.")
        return  # Skip processing this folder

    # Collect all JSON files (exclude winner.json)
    json_files = sorted(glob.glob(os.path.join(replay_dir, '*.json')))
    json_files = [f for f in json_files if not os.path.basename(f) == 'winner.json']

    if not json_files:
        print(f"No JSON files found in {replay_dir}, skipping this folder.")
        return

    # First Pass: Determine max player_health and enemy_health
    max_player_health = 0
    max_enemy_health = 0
    temp_frames = []

    print(f"First pass: Determining max health values in {replay_name}")
    for json_file in tqdm(json_files, desc="Scanning health values"):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {json_file}, skipping this frame.")
                continue  # Skip this frame due to invalid JSON

            player_health = data.get('player_health', 0)
            enemy_health = data.get('enemy_health', 0)

            if isinstance(player_health, (int, float)) and player_health > max_player_health:
                max_player_health = player_health

            if isinstance(enemy_health, (int, float)) and enemy_health > max_enemy_health:
                max_enemy_health = enemy_health

            temp_frames.append(data)

    if max_player_health == 0 or max_enemy_health == 0:
        print(f"Max player_health or enemy_health is zero in {replay_dir}, skipping.")
        return  # Avoid division by zero

    print(f"Max Player Health: {max_player_health}, Max Enemy Health: {max_enemy_health}")

    # Second Pass: Process frames with normalization and additional inputs
    frames = []
    for data in temp_frames:
        # Extract timestamp
        try:
            timestamp_str = os.path.splitext(os.path.basename(data.get('image_path', '0.png')))[0]
            timestamp = int(timestamp_str.split('_')[0])  # Adjust based on filename pattern
        except (ValueError, IndexError):
            timestamp = int(time.time() * 1000)  # Use current time if extraction fails

        image_path = data.get('image_path')
        input_str = data.get('input')
        reward = data.get('reward') or 0
        punishment = data.get('punishment') or 0
        player_health = data.get('player_health', 0) / max_player_health
        enemy_health = data.get('enemy_health', 0) / max_enemy_health
        player_position = data.get('player_position', [0, 0])
        enemy_position = data.get('enemy_position', [0, 0])
        inside_window = data.get('inside_window', False)
        if inside_window is None:
            print(f"Invalid 'inside_window' value in frame with timestamp {timestamp}, skipping this frame.")
            continue  # Skip frames with invalid inside_window


        if image_path is None:
            print(f"No 'image_path' found in frame with timestamp {timestamp}, skipping this frame.")
            continue  # Skip frames without image_path

        # Construct the absolute path to the image
        image_abs_path = os.path.join(replay_dir, os.path.basename(image_path))
        if not os.path.exists(image_abs_path):
            print(f"Image file {image_abs_path} does not exist, skipping this frame.")
            continue  # Skip frames with missing images

        # Convert positions to grids
        player_grid = position_to_grid(player_position[0], player_position[1])
        enemy_grid = position_to_grid(enemy_position[0], enemy_position[1])

        frames.append({
            'timestamp': timestamp,
            'image_path': image_abs_path,
            'input_str': input_str,
            'reward': reward,
            'punishment': punishment,
            'player_health': player_health,
            'enemy_health': enemy_health,
            'player_grid': player_grid,      # 6x3 grid
            'enemy_grid': enemy_grid,        # 6x3 grid
            'inside_window': inside_window   # 0 or 1
        })

    if not frames:
        print(f"No valid frames found after processing in {replay_dir}, skipping this folder.")
        return  # Skip if no valid frames are found

    # Sort frames by timestamp
    frames.sort(key=lambda x: x['timestamp'])

    num_frames = len(frames)

    # Extract rewards and punishments for all frames
    rewards = np.array([frame['reward'] for frame in frames], dtype=np.float32)
    punishments = np.array([frame['punishment'] for frame in frames], dtype=np.float32)

    # Compute net rewards
    net_rewards = rewards - punishments

    # Modify rewards or punishments based on winner status
    if is_winner:
        # Player won, add 1 to rewards for every frame
        net_rewards += 1.0
    else:
        # Player lost, subtract 1 from net rewards for every frame
        net_rewards -= 1.0

    # Linear decay weights
    weights = np.array([(WINDOW_SIZE - k) / WINDOW_SIZE for k in range(WINDOW_SIZE)], dtype=np.float32)

    # Compute net rewards for each frame by convolving with the weights
    cumulative_rewards = []
    for t in range(num_frames):
        # Determine the window range
        start = t
        end = min(t + WINDOW_SIZE, num_frames)
        window_size = end - start
        # Adjust weights for the window size
        window_weights = weights[:window_size]
        # Compute net reward
        net_reward = np.sum(net_rewards[start:end] * window_weights)
        cumulative_rewards.append(net_reward)

    # Process frames and save tensors
    for idx, frame in enumerate(tqdm(frames, desc=f'Processing {replay_name}')):
        input_str = frame['input_str']

        if input_str is None:
            print(f"Input string is None for frame at timestamp {frame['timestamp']}, skipping.")
            continue  # Skip this frame

        # Check for invalid input strings
        if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
            print(f"Invalid input string '{input_str}' at timestamp {frame['timestamp']}, skipping.")
            continue

        # Load image and convert to tensor
        try:
            image = Image.open(frame['image_path']).convert('RGB')
            image_np = np.array(image)
            if image_np.shape != (160, 240, 3):
                print(f"Unexpected image shape {image_np.shape} in {frame['image_path']}, expected (160, 240, 3). Skipping.")
                continue  # Skip frames with unexpected image dimensions
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Shape: (3, 160, 240)
        except Exception as e:
            print(f"Failed to load image {frame['image_path']}: {e}")
            continue  # Skip this frame

        # Convert input string to tensor
        try:
            input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)
        except ValueError:
            print(f"Non-binary character found in input string '{input_str}' at frame {idx}, skipping.")
            continue  # Skip frames with invalid input strings

        # Get net reward
        net_reward = cumulative_rewards[idx]

        if net_reward == 0:
            print(f"Net reward is 0 at frame {idx}, skipping.")
            continue

        # Create sample dictionary with new inputs
        sample = {
            'image': image_tensor,
            'input': input_tensor,
            'net_reward': net_reward,
            'player_health': torch.tensor(frame['player_health'], dtype=torch.float32),
            'enemy_health': torch.tensor(frame['enemy_health'], dtype=torch.float32),
            'player_grid': torch.tensor(frame['player_grid'], dtype=torch.float32),  # Shape: (6, 3)
            'enemy_grid': torch.tensor(frame['enemy_grid'], dtype=torch.float32),    # Shape: (6, 3)
            'inside_window': torch.tensor(frame['inside_window'], dtype=torch.float32)  # Scalar
        }

        # Save sample to cache
        sample_path = os.path.join(cache_dir, f'{idx:06d}.pt')

        try:
            torch.save(sample, sample_path)
        except Exception as e:
            print(f"Failed to save tensor to {sample_path}: {e}")
            continue  # Skip saving this frame due to error

def process_all_replays():
    os.makedirs(TRAINING_CACHE_DIR, exist_ok=True)
    replay_dirs = glob.glob(os.path.join(TRAINING_DATA_DIR, '*'))
    if not replay_dirs:
        print(f"No replay folders found in {TRAINING_DATA_DIR}.")
    for replay_dir in tqdm(replay_dirs, desc='Processing all replays'):
        if os.path.isdir(replay_dir):
            process_replay(replay_dir)
        else:
            print(f"Found non-directory item {replay_dir}, skipping.")

if __name__ == '__main__':
    process_all_replays()
