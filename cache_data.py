# cache_data.py
import os
import json
import glob
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import h5py  # Ensure HDF5 is imported
from utils import get_root_dir, position_to_grid  # Ensure position_to_grid is imported

# Paths
TRAINING_DATA_DIR = os.path.join(get_root_dir(), 'training_data')
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')
PLANNING_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'planning')
BATTLE_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'battle')
WINDOW_SIZE = 600  # Number of frames to look ahead for rewards/punishments

def process_replay(replay_dir, planning_output_dir=PLANNING_CACHE_DIR, battle_output_dir=BATTLE_CACHE_DIR, rewards_only=True):
    replay_name = os.path.basename(replay_dir)
    planning_h5_path = os.path.join(planning_output_dir, f'{replay_name}.h5')
    battle_h5_path = os.path.join(battle_output_dir, f'{replay_name}.h5')

    if os.path.exists(planning_h5_path) and os.path.exists(battle_h5_path):
        print(f"HDF5 files already exist for {replay_name}, skipping this replay.")
        return

    os.makedirs(planning_output_dir, exist_ok=True)
    os.makedirs(battle_output_dir, exist_ok=True)

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
            timestamp = int(time.time() * 100)  # Use current time if extraction fails

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
        if not frame['inside_window']:
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

    if not all_planning_frames and not all_battle_frames:
        print(f"No valid frames to save for {replay_name}, skipping.")
        return

    # ----- Save Planning Frames in Bulk -----
    if all_planning_frames:
        # Initialize empty lists to collect data
        images = []
        inputs = []
        player_healths = []
        enemy_healths = []
        player_grids = []
        enemy_grids = []
        inside_windows = []
        net_rewards = []

        # Collect data with filtering before HDF5 allocation
        for frame in tqdm(all_planning_frames, desc=f'Collecting Planning Frames for {replay_name}'):
            input_str = frame['input_str']

            if input_str is None:
                continue  # Skip this frame

            # Skip planning frames with input '0000000000000000'
            if input_str == '0000000000000000':
                continue  # Skip frames with all-zero inputs

            # Check for invalid input strings
            if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
                continue  # Skip invalid input strings

            # Load image and convert to numpy array
            try:
                image = Image.open(frame['image_path']).convert('RGB')
                if image.mode != 'RGB':
                    image = image.convert('RGB')  # Ensures image is in RGB format
                image_np = np.array(image)
                if image_np.shape != (160, 240, 3):
                    continue  # Skip frames with unexpected image dimensions
                image_tensor = image_np.astype(np.float32) / 255.0  # Normalize
            except Exception as e:
                print(f"Error loading image {frame['image_path']}: {e}")
                continue  # Skip frames with image loading issues

            # Convert input string to binary array
            try:
                input_tensor = np.array([int(bit) for bit in input_str], dtype=np.float32)
            except ValueError:
                continue  # Skip frames with invalid input strings

            # Get assigned reward
            net_reward = frame.get('assigned_reward', 0.0)

            # Append data to lists
            images.append(image_tensor.transpose(2, 0, 1))  # Shape: (3, 160, 240)
            inputs.append(input_tensor)
            player_healths.append(frame['player_health'])
            enemy_healths.append(frame['enemy_health'])
            player_grids.append(frame['player_grid'])
            enemy_grids.append(frame['enemy_grid'])
            inside_windows.append(float(frame['inside_window']))
            net_rewards.append(net_reward)

        # After collecting and filtering, determine the number of valid samples
        num_samples = len(images)

        if num_samples == 0:
            print(f"No valid planning frames to save for {replay_name}, skipping planning cache.")
        else:
            with h5py.File(planning_h5_path, 'w') as h5f:
                # Define dynamic chunk sizes
                chunk_images = get_chunk_size(100, num_samples, 3, 160, 240)
                chunk_inputs = get_chunk_size(100, num_samples, 16)
                chunk_player_healths = get_chunk_size(100, num_samples)
                chunk_enemy_healths = get_chunk_size(100, num_samples)
                chunk_player_grids = get_chunk_size(100, num_samples, 6, 3)
                chunk_enemy_grids = get_chunk_size(100, num_samples, 6, 3)
                chunk_inside_windows = get_chunk_size(100, num_samples)
                chunk_net_rewards = get_chunk_size(100, num_samples)

                # Create datasets with dynamic chunk sizes
                h5f.create_dataset(
                    'images',
                    shape=(num_samples, 3, 160, 240),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_images
                )
                h5f.create_dataset(
                    'inputs',
                    shape=(num_samples, 16),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_inputs
                )
                h5f.create_dataset(
                    'player_healths',
                    shape=(num_samples,),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_player_healths
                )
                h5f.create_dataset(
                    'enemy_healths',
                    shape=(num_samples,),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_enemy_healths
                )
                h5f.create_dataset(
                    'player_grids',
                    shape=(num_samples, 6, 3),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_player_grids
                )
                h5f.create_dataset(
                    'enemy_grids',
                    shape=(num_samples, 6, 3),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_enemy_grids
                )
                h5f.create_dataset(
                    'inside_windows',
                    shape=(num_samples,),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_inside_windows
                )
                h5f.create_dataset(
                    'net_rewards',
                    shape=(num_samples,),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=chunk_net_rewards
                )

                # Convert lists to numpy arrays
                images = np.stack(images, axis=0)
                inputs = np.stack(inputs, axis=0)
                player_healths = np.array(player_healths, dtype=np.float32)
                enemy_healths = np.array(enemy_healths, dtype=np.float32)
                player_grids = np.stack(player_grids, axis=0)
                enemy_grids = np.stack(enemy_grids, axis=0)
                inside_windows = np.array(inside_windows, dtype=np.float32)
                net_rewards = np.array(net_rewards, dtype=np.float32)

                # Assign data to HDF5 datasets in bulk
                h5f['images'][:] = images
                h5f['inputs'][:] = inputs
                h5f['player_healths'][:] = player_healths
                h5f['enemy_healths'][:] = enemy_healths
                h5f['player_grids'][:] = player_grids
                h5f['enemy_grids'][:] = enemy_grids
                h5f['inside_windows'][:] = inside_windows
                h5f['net_rewards'][:] = net_rewards

                # Store min and max as attributes
                h5f.attrs['net_reward_min'] = float(net_rewards.min())
                h5f.attrs['net_reward_max'] = float(net_rewards.max())

                print(f"Saved {num_samples} planning frames to {planning_h5_path}")


    # ----- Save Battle Frames in Bulk -----
    if all_battle_frames:
        with h5py.File(battle_h5_path, 'w') as h5f:
            num_samples = len(all_battle_frames)
            
            # Define dynamic chunk sizes
            chunk_images = get_chunk_size(100, num_samples, 3, 160, 240)
            chunk_inputs = get_chunk_size(100, num_samples, 16)
            chunk_player_healths = get_chunk_size(100, num_samples)
            chunk_enemy_healths = get_chunk_size(100, num_samples)
            chunk_player_grids = get_chunk_size(100, num_samples, 6, 3)
            chunk_enemy_grids = get_chunk_size(100, num_samples, 6, 3)
            chunk_inside_windows = get_chunk_size(100, num_samples)
            chunk_net_rewards = get_chunk_size(100, num_samples)
            
            # Create datasets with dynamic chunk sizes
            h5f.create_dataset(
                'images',
                shape=(num_samples, 3, 160, 240),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_images
            )
            h5f.create_dataset(
                'inputs',
                shape=(num_samples, 16),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_inputs
            )
            h5f.create_dataset(
                'player_healths',
                shape=(num_samples,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_player_healths
            )
            h5f.create_dataset(
                'enemy_healths',
                shape=(num_samples,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_enemy_healths
            )
            h5f.create_dataset(
                'player_grids',
                shape=(num_samples, 6, 3),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_player_grids
            )
            h5f.create_dataset(
                'enemy_grids',
                shape=(num_samples, 6, 3),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_enemy_grids
            )
            h5f.create_dataset(
                'inside_windows',
                shape=(num_samples,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_inside_windows
            )
            h5f.create_dataset(
                'net_rewards',
                shape=(num_samples,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
                chunks=chunk_net_rewards
            )

            # Initialize empty lists to collect data
            images = []
            inputs = []
            player_healths = []
            enemy_healths = []
            player_grids = []
            enemy_grids = []
            inside_windows = []
            net_rewards = []

            # Collect data
            for frame in tqdm(all_battle_frames, desc=f'Collecting Battle Frames for {replay_name}'):
                input_str = frame['input_str']

                if input_str is None:
                    continue  # Skip this frame

                # Check for invalid input strings
                if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
                    continue  # Skip invalid input strings

                # Load image and convert to numpy array
                try:
                    # In cache_data.py, within process_replay function
                    image = Image.open(frame['image_path']).convert('RGB')
                    if image.mode != 'RGB':
                        image = image.convert('RGB')  # Redundant but ensures correctness
                    image_np = np.array(image)
                    if image_np.shape != (160, 240, 3):
                        continue  # Skip frames with unexpected image dimensions
                    image_tensor = image_np.astype(np.float32) / 255.0  # Normalize
                except:
                    continue  # Skip frames with image loading issues

                # Convert input string to binary array
                try:
                    input_tensor = np.array([int(bit) for bit in input_str], dtype=np.float32)
                except:
                    continue  # Skip frames with invalid input strings

                # Get assigned reward
                net_reward = frame.get('assigned_reward', 0.0)

                # Append data to lists
                images.append(image_tensor.transpose(2, 0, 1))  # Shape: (3, 160, 240)
                inputs.append(input_tensor)
                player_healths.append(frame['player_health'])
                enemy_healths.append(frame['enemy_health'])
                player_grids.append(frame['player_grid'])
                enemy_grids.append(frame['enemy_grid'])
                inside_windows.append(float(frame['inside_window']))
                net_rewards.append(net_reward)

            # Convert lists to numpy arrays
            images = np.stack(images, axis=0)
            inputs = np.stack(inputs, axis=0)
            player_healths = np.array(player_healths, dtype=np.float32)
            enemy_healths = np.array(enemy_healths, dtype=np.float32)
            player_grids = np.stack(player_grids, axis=0)
            enemy_grids = np.stack(enemy_grids, axis=0)
            inside_windows = np.array(inside_windows, dtype=np.float32)
            net_rewards = np.array(net_rewards, dtype=np.float32)

            # Assign data to HDF5 datasets in bulk
            h5f['images'][:] = images
            h5f['inputs'][:] = inputs
            h5f['player_healths'][:] = player_healths
            h5f['enemy_healths'][:] = enemy_healths
            h5f['player_grids'][:] = player_grids
            h5f['enemy_grids'][:] = enemy_grids
            h5f['inside_windows'][:] = inside_windows
            h5f['net_rewards'][:] = net_rewards

            # Store min and max as attributes
            h5f.attrs['net_reward_min'] = float(net_rewards.min())
            h5f.attrs['net_reward_max'] = float(net_rewards.max())

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

def get_chunk_size(default_chunk, num_samples, *dims):
    """Return a tuple for chunk size where the first dimension is the min of default_chunk and num_samples."""
    first_dim = min(default_chunk, num_samples)
    return (first_dim,) + dims


if __name__ == '__main__':
    process_all_replays()
