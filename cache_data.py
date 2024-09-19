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
TRAINING_DATA_DIR = os.path.join(get_root_dir(), 'training_data')
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')
PLANNING_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'planning')
BATTLE_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'battle')
WINDOW_SIZE = 600  # Number of frames to look ahead for rewards/punishments

def process_replay(replay_dir, planning_output_dir=PLANNING_CACHE_DIR, battle_output_dir=BATTLE_CACHE_DIR, rewards_only=True):
    replay_name = os.path.basename(replay_dir)
    planning_cache_dir = os.path.join(planning_output_dir, replay_name)
    battle_cache_dir = os.path.join(battle_output_dir, replay_name)

    if os.path.exists(planning_cache_dir) and os.listdir(planning_cache_dir) and \
       os.path.exists(battle_cache_dir) and os.listdir(battle_cache_dir):
        print(f"Cache already exists for {replay_name}, skipping this replay.")
        return

    os.makedirs(planning_cache_dir, exist_ok=True)
    os.makedirs(battle_cache_dir, exist_ok=True)

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

    for json_file in tqdm(json_files, desc="Scanning health values"):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {json_file}, skipping this frame.")
                continue

            player_health = data.get('player_health', 0)
            enemy_health = data.get('enemy_health', 0)

            if isinstance(player_health, (int, float)) and player_health > max_player_health:
                max_player_health = player_health

            if isinstance(enemy_health, (int, float)) and enemy_health > max_enemy_health:
                max_enemy_health = enemy_health

            temp_frames.append(data)

    if max_player_health == 0 or max_enemy_health == 0:
        print(f"Max player_health or enemy_health is zero in {replay_dir}, skipping.")
        return

    print(f"Max Player Health: {max_player_health}, Max Enemy Health: {max_enemy_health}")

    # Second Pass: Process frames and segment into rounds
    frames = []
    for data in temp_frames:
        # Extract timestamp
        try:
            image_path_field = data.get('image_path', '0.png')
            timestamp_str = os.path.splitext(os.path.basename(image_path_field))[0]
            timestamp = int(timestamp_str.split('_')[0])  # Adjust based on filename pattern
        except (ValueError, IndexError):
            timestamp = int(time.time() * 1000)  # Use current time if extraction fails

        image_path = data.get('image_path')
        input_str = data.get('input')
        reward = data.get('reward') or 0
        punishment = data.get('punishment') or 0
        if rewards_only:
            punishment = 0
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
            'inside_window': inside_window   # Boolean
        })

    if not frames:
        print(f"No valid frames found after processing in {replay_dir}, skipping this folder.")
        return  # Skip if no valid frames are found

    # Sort frames by timestamp
    frames.sort(key=lambda x: x['timestamp'])

    # Segment frames into rounds
    rounds = []
    current_round = {'planning': [], 'battle': []}
    phase = 'waiting'  # Possible phases: 'waiting', 'planning', 'battle'

    for frame in frames:
        if frame['inside_window']:
            if phase in ['waiting', 'battle']:
                if current_round['planning'] or current_round['battle']:
                    # Save the completed round before starting a new one
                    rounds.append(current_round)
                    current_round = {'planning': [], 'battle': []}
                phase = 'planning'
            current_round['planning'].append(frame)
        else:
            if phase == 'planning':
                phase = 'battle'
            if phase == 'battle':
                current_round['battle'].append(frame)

    # Append the last round if it has at least planning and battle phases
    if current_round['planning'] and current_round['battle']:
        rounds.append(current_round)

    if not rounds:
        print(f"No complete rounds found in {replay_dir}, skipping this folder.")
        return

    # Initialize lists to collect all planning and battle frames
    all_planning_frames = []
    all_battle_frames = []

    # Process each round
    for round_idx, round_data in enumerate(tqdm(rounds, desc=f'Processing {replay_name} rounds')):
        planning_frames = round_data['planning']
        battle_frames = round_data['battle']

        if not battle_frames:
            print(f"No battle frames in round {round_idx} of {replay_name}, skipping this round.")
            continue

        # Calculate total damage dealt by player and enemy
        total_reward = sum(frame['reward'] for frame in battle_frames)
        total_punishment = sum(frame['punishment'] for frame in battle_frames)

        # Determine winner based on total damage dealt
        if total_reward > total_punishment:
            is_winner = True  # Player wins
        elif total_reward < total_punishment:
            is_winner = False  # Enemy wins
        else:
            # Handle tie-breaker based on final health values
            last_battle_frame = battle_frames[-1]
            player_health_last = last_battle_frame.get('player_health', 0) / max_player_health
            enemy_health_last = last_battle_frame.get('enemy_health', 0) / max_enemy_health
            is_winner = player_health_last >= enemy_health_last  # Fallback to health comparison

        # Assign flat reward based on is_winner to planning frames
        round_reward = 1.0 if is_winner else -1.0  # You can adjust these values

        for frame in planning_frames:
            frame['assigned_reward'] = round_reward

        # Collect all planning frames
        all_planning_frames.extend(planning_frames)

        # Process battle frames to compute cumulative rewards
        num_battle_frames = len(battle_frames)
        rewards = np.array([frame['reward'] for frame in battle_frames], dtype=np.float32)
        punishments = np.array([frame['punishment'] for frame in battle_frames], dtype=np.float32)
        if rewards_only:
            punishments = np.zeros_like(rewards)

        # Compute net rewards
        net_rewards = rewards - punishments

        # Modify net rewards based on is_winner
        if is_winner:
            net_rewards += 1.0  # Player won
        else:
            net_rewards -= 1.0  # Player lost

        # Linear decay weights
        weights = np.array([(WINDOW_SIZE - k) / WINDOW_SIZE for k in range(WINDOW_SIZE)], dtype=np.float32)

        # Compute cumulative rewards for battle frames
        cumulative_rewards = []
        for t in range(num_battle_frames):
            # Determine the window range
            start = t
            end = min(t + WINDOW_SIZE, num_battle_frames)
            window_size = end - start
            # Adjust weights for the window size
            window_weights = weights[:window_size]
            # Compute net reward
            net_reward = np.sum(net_rewards[start:end] * window_weights)

            # Add punishment if the net reward is zero (indicating time wasted)
            if net_reward == 0:
                punishment_value = -0.5  # You can adjust this value
                net_reward += punishment_value

            cumulative_rewards.append(net_reward)

        # Assign cumulative rewards to battle frames
        for idx, frame in enumerate(battle_frames):
            frame['assigned_reward'] = cumulative_rewards[idx]

        # Collect all battle frames
        all_battle_frames.extend(battle_frames)

    # Save Planning Frames
    for idx, frame in enumerate(tqdm(all_planning_frames, desc=f'Processing Planning Frames for {replay_name}')):
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

        # Get assigned reward
        net_reward = frame.get('assigned_reward', 0.0)

        # Create sample dictionary with new inputs
        sample = {
            'image': image_tensor,
            'input': input_tensor,
            'net_reward': torch.tensor(net_reward, dtype=torch.float32),
            'player_health': torch.tensor(frame['player_health'], dtype=torch.float32),
            'enemy_health': torch.tensor(frame['enemy_health'], dtype=torch.float32),
            'player_grid': torch.tensor(frame['player_grid'], dtype=torch.float32),  # Shape: (6, 3)
            'enemy_grid': torch.tensor(frame['enemy_grid'], dtype=torch.float32),    # Shape: (6, 3)
            'inside_window': torch.tensor(float(frame['inside_window']), dtype=torch.float32)  # Scalar
        }

        # Save sample to Planning cache
        sample_path = os.path.join(planning_cache_dir, f'{idx:06d}.pt')

        try:
            torch.save(sample, sample_path)
        except Exception as e:
            print(f"Failed to save tensor to {sample_path}: {e}")
            continue  # Skip saving this frame due to error

    # Process and save Battle Frames with Window-Based Rewards
    for idx, frame in enumerate(tqdm(all_battle_frames, desc=f'Processing Battle Frames for {replay_name}')):
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

        # Get assigned reward
        net_reward = frame.get('assigned_reward', 0.0)

        # Create sample dictionary with new inputs
        sample = {
            'image': image_tensor,
            'input': input_tensor,
            'net_reward': torch.tensor(net_reward, dtype=torch.float32),
            'player_health': torch.tensor(frame['player_health'], dtype=torch.float32),
            'enemy_health': torch.tensor(frame['enemy_health'], dtype=torch.float32),
            'player_grid': torch.tensor(frame['player_grid'], dtype=torch.float32),  # Shape: (6, 3)
            'enemy_grid': torch.tensor(frame['enemy_grid'], dtype=torch.float32),    # Shape: (6, 3)
            'inside_window': torch.tensor(float(frame['inside_window']), dtype=torch.float32)  # Scalar
        }

        # Save sample to Battle cache
        sample_path = os.path.join(battle_cache_dir, f'{idx:06d}.pt')

        try:
            torch.save(sample, sample_path)
        except Exception as e:
            print(f"Failed to save tensor to {sample_path}: {e}")
            continue  # Skip saving this frame due to error

def process_all_replays():
    os.makedirs(PLANNING_CACHE_DIR, exist_ok=True)
    os.makedirs(BATTLE_CACHE_DIR, exist_ok=True)
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
