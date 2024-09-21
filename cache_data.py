# cache_data.py

import os
import json
import glob
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import h5py
import yaml  # For configuration
from utils import get_root_dir, position_to_grid, get_image_memory # Ensure these functions are defined in utils.py
from strategies import DefaultStrategy, DodgeStrategy, DamageStrategy, AggressiveStrategy, RewardEverythingStrategy # Import strategies
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc

# Paths
TRAINING_DATA_DIR = os.path.join(get_root_dir(), 'training_data')
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')
PLANNING_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'planning')
BATTLE_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'battle')

# Load configuration
def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Initialize strategy based on config
def initialize_strategy(config):
    strategy_config = config.get('strategy', {})
    strategy_name = strategy_config.get('name', 'default')
    parameters = strategy_config.get('parameters', {})

    if strategy_name == 'default':
        window_size = parameters.get('window_size', 100)
        return DefaultStrategy(window_size=window_size)
    elif strategy_name == 'dodge':
        punishment_value = parameters.get('punishment_value', -0.5)
        window_size = parameters.get('window_size', 100)
        return DodgeStrategy(punishment_value=punishment_value, window_size=window_size)
    elif strategy_name == 'damage':
        reward_value = parameters.get('reward_value', 1.0)
        window_size = parameters.get('window_size', 100)
        return DamageStrategy(reward_value=reward_value, window_size=window_size)
    elif strategy_name == 'aggressive':
        reward_value = parameters.get('reward_value', 1.0)
        punishment_value = parameters.get('punishment_value', -0.5)
        window_size = parameters.get('window_size', 100)
        return AggressiveStrategy(reward_value=reward_value, punishment_value=punishment_value, window_size=window_size)
    elif strategy_name == 'reward_everything':
        return RewardEverythingStrategy()
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

# Get dynamic chunk size for HDF5 datasets
def get_chunk_size(default_chunk, num_samples, *dims):
    """Return a tuple for chunk size where the first dimension is the min of default_chunk and num_samples."""
    first_dim = min(default_chunk, num_samples)
    return (first_dim,) + dims

# Process a single replay directory with a given strategy and configuration
def process_replay(replay_dir, strategy, planning_output_dir=PLANNING_CACHE_DIR, 
                  battle_output_dir=BATTLE_CACHE_DIR, rewards_only=False, config=None):
    # Collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    replay_name = os.path.basename(replay_dir)
    planning_h5_path = os.path.join(planning_output_dir, f'{replay_name}.h5')
    battle_h5_path = os.path.join(battle_output_dir, f'{replay_name}.h5')

    if os.path.exists(planning_h5_path) and os.path.exists(battle_h5_path):
        print(f"HDF5 files already exist for {replay_name}, skipping this replay.")
        return f"{replay_name}: Skipped (Already Exists)"

    os.makedirs(planning_output_dir, exist_ok=True)
    os.makedirs(battle_output_dir, exist_ok=True)

    # Collect all JSON files (exclude winner.json)
    json_files = sorted(glob.glob(os.path.join(replay_dir, '*.json')))
    json_files = [f for f in json_files if not os.path.basename(f) == 'winner.json']

    if not json_files:
        print(f"No JSON files found in {replay_dir}, skipping this folder.")
        return f"{replay_name}: Skipped (No JSON Files)"

    # First Pass: Determine max player_health and enemy_health
    max_player_health = 0
    max_enemy_health = 0
    temp_frames = []

    for json_file in json_files:
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
        return f"{replay_name}: Skipped (Zero Max Health)"

    # Extract configuration for input features
    input_features = config.get('input_features', {})
    include_image = input_features.get('include_image', True)
    include_position = input_features.get('include_position', True)
    position_type = input_features.get('position_type', 'grid')
    include_player_charge = input_features.get('include_player_charge', False)
    include_enemy_charge = input_features.get('include_enemy_charge', False)

    # Extract image_memory from config
    image_memory = config.get('image_memory', 1)
    if image_memory < 1:
        print(f"Invalid image_memory={image_memory} in config, setting to 1.")
        image_memory = 1

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

        # Conditionally process features
        processed_data = {
            'timestamp': timestamp,
            'input_str': input_str,
            'reward': reward,
            'punishment': punishment,
            'player_health': player_health,
            'enemy_health': enemy_health,
            'inside_window': inside_window
        }

        if include_image:
            processed_data['image_path'] = image_abs_path

        if include_position:
            if position_type == 'float':
                processed_data['player_position'] = player_position  # [x, y]
                processed_data['enemy_position'] = enemy_position
            elif position_type == 'grid':
                player_grid = position_to_grid(player_position[0], player_position[1])
                enemy_grid = position_to_grid(enemy_position[0], enemy_position[1])
                processed_data['player_grid'] = player_grid
                processed_data['enemy_grid'] = enemy_grid
            else:
                raise ValueError(f"Unknown position_type: {position_type}")

        if include_player_charge:
            processed_data['player_charge'] = data.get('player_charge', 0)

        if include_enemy_charge:
            processed_data['enemy_charge'] = data.get('enemy_charge', 0)

        frames.append(processed_data)

    if not frames:
        print(f"No valid frames found after processing in {replay_dir}, skipping this folder.")
        return f"{replay_name}: Skipped (No Valid Frames)"

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
        return f"{replay_name}: Skipped (No Complete Rounds)"

    # Initialize lists to collect all planning and battle frames
    all_planning_frames = []
    all_battle_frames = []

    # Collect frames per round
    for round_idx, round_data in enumerate(rounds):
        planning_frames = round_data['planning']
        battle_frames = round_data['battle']

        if not battle_frames:
            print(f"No battle frames in round {round_idx} of {replay_name}, skipping this round.")
            continue

        # Collect all planning frames
        all_planning_frames.extend(planning_frames)

        # Collect all battle frames
        all_battle_frames.extend(battle_frames)

    if not all_planning_frames and not all_battle_frames:
        print(f"No valid frames to save for {replay_name}, skipping.")
        return f"{replay_name}: Skipped (No Frames to Save)"

    print(f"Processing replay {replay_name} with {len(all_planning_frames)} planning frames and {len(all_battle_frames)} battle frames.")

    # Assign rewards using the selected strategy
    try:
        all_planning_frames, all_battle_frames = strategy.assign_rewards(
            all_planning_frames,
            all_battle_frames,
            max_player_health,
            max_enemy_health
        )
    except Exception as e:
        print(f"Error assigning rewards for {replay_name}: {e}")
        return f"{replay_name}: Skipped (Reward Assignment Error)"

    # ----- Save Planning Frames in Bulk -----
    if (all_planning_frames and (include_image or include_position or include_player_charge or include_enemy_charge)):
        # Initialize empty lists to collect data
        images = []
        positions = []
        player_charges = []
        enemy_charges = []
        net_rewards = []

        frames_to_remove = []

        # Collect data with filtering before HDF5 allocation
        for frame in all_planning_frames:
            # Process based on included features
            if include_image:
                input_str = frame.get('input_str')
                if input_str is None:
                    frames_to_remove.append(frame)
                    continue  # Skip this frame

                # Skip planning frames with input '0000000000000000'
                if input_str == '0000000000000000':
                    frames_to_remove.append(frame)
                    continue  # Skip frames with all-zero inputs

                # Check for invalid input strings
                if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
                    frames_to_remove.append(frame)
                    continue  # Skip invalid input strings

                # Load image and convert to numpy array
                try:
                    image = Image.open(frame['image_path']).convert('RGB')
                    if image.mode != 'RGB':
                        image = image.convert('RGB')  # Ensures image is in RGB format
                    image_np = np.array(image)
                    if image_np.shape != (160, 240, 3):
                        frames_to_remove.append(frame)
                        continue  # Skip frames with unexpected image dimensions
                    image_tensor = image_np.astype(np.float32) / 255.0  # Normalize
                except Exception as e:
                    print(f"Error loading image {frame['image_path']}: {e}")
                    frames_to_remove.append(frame)
                    continue  # Skip frames with image loading issues

                images.append(image_tensor.transpose(2, 0, 1))  # Shape: (3, 160, 240)

            if include_position:
                if position_type == 'float':
                    player_position = frame.get('player_position', [0.0, 0.0])
                    enemy_position = frame.get('enemy_position', [0.0, 0.0])
                    # Normalize positions max is 240x 160y
                    player_position[0] /= 240
                    player_position[1] /= 160
                    enemy_position[0] /= 240
                    enemy_position[1] /= 160
                    positions.append(player_position + enemy_position)  # Concatenate x and y
                elif position_type == 'grid':
                    player_grid = frame.get('player_grid', [[0]*3 for _ in range(6)])
                    enemy_grid = frame.get('enemy_grid', [[0]*3 for _ in range(6)])
                    flattened_player_grid = np.concatenate(player_grid).astype(np.float32)
                    flattened_enemy_grid = np.concatenate(enemy_grid).astype(np.float32)
                    positions.append(np.concatenate([flattened_player_grid, flattened_enemy_grid]))  # Shape: (36,)
                else:
                    raise ValueError(f"Unknown position_type: {position_type}")

            if include_player_charge:
                player_charge = frame.get('player_charge', 0)
                player_charge /= 2 #max is 2 so normalize
                player_charges.append(player_charge)

            if include_enemy_charge:
                enemy_charge = frame.get('enemy_charge', 0)
                enemy_charge /= 2 #max is 2 so normalize
                enemy_charges.append(enemy_charge)

            # Collect net_rewards
            net_reward = frame.get('assigned_reward', 0.0)
            net_rewards.append(net_reward)

        # Remove frames that failed filtering
        for frame in frames_to_remove:
            if frame in all_planning_frames:
                all_planning_frames.remove(frame)

        print(f"Number of planning frames: {len(images) if include_image else 0} "
              f"{len(positions) if include_position else 0} "
              f"{len(player_charges) if include_player_charge else 0} "
              f"{len(enemy_charges) if include_enemy_charge else 0}")
        print(f"Number of removed planning frames: {len(frames_to_remove)}")

        # Determine the number of valid samples
        num_samples = max(
            len(images) if include_image else 0,
            len(positions) if include_position else 0,
            len(player_charges) if include_player_charge else 0,
            len(enemy_charges) if include_enemy_charge else 0,
            len(net_rewards)
        )

        if num_samples == 0:
            print(f"No valid planning frames to save for {replay_name}, skipping planning cache.")
        else:
            try:
                with h5py.File(planning_h5_path, 'w') as h5f:
                    # Define dynamic chunk sizes and create datasets based on included features
                    if include_image:
                        h5f.create_dataset(
                            'images',
                            shape=(num_samples, 3, 160, 240),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples, 3, 160, 240)
                        )
                        images_np = np.stack(images, axis=0)
                        h5f['images'][:] = images_np

                    if include_position:
                        if position_type == 'float':
                            h5f.create_dataset(
                                'positions',
                                shape=(num_samples, 4),  # player_x, player_y, enemy_x, enemy_y
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4,
                                chunks=get_chunk_size(100, num_samples, 4)
                            )
                            positions_np = np.array(positions, dtype=np.float32)
                            h5f['positions'][:] = positions_np
                        elif position_type == 'grid':
                            h5f.create_dataset(
                                'positions',
                                shape=(num_samples, 36),  # 6x3 grids concatenated
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4,
                                chunks=get_chunk_size(100, num_samples, 36)
                            )
                            positions_np = np.array(positions, dtype=np.float32)
                            h5f['positions'][:] = positions_np

                    if include_player_charge:
                        h5f.create_dataset(
                            'player_charges',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples)
                        )
                        player_charges_np = np.array(player_charges, dtype=np.float32)
                        h5f['player_charges'][:] = player_charges_np

                    if include_enemy_charge:
                        h5f.create_dataset(
                            'enemy_charges',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples)
                        )
                        enemy_charges_np = np.array(enemy_charges, dtype=np.float32)
                        h5f['enemy_charges'][:] = enemy_charges_np

                    # Always save net_rewards
                    h5f.create_dataset(
                        'net_rewards',
                        shape=(num_samples,),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        chunks=get_chunk_size(100, num_samples)
                    )
                    net_rewards_np = np.array(net_rewards, dtype=np.float32)
                    h5f['net_rewards'][:] = net_rewards_np

                    # Convert 'input_str' to binary arrays
                    inputs = [[int(c) for c in frame['input_str']] for frame in all_planning_frames]
                    h5f.create_dataset(
                        'input',
                        shape=(num_samples, 16),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        chunks=get_chunk_size(100, num_samples, 16)
                    )
                    input_np = np.array(inputs, dtype=np.float32)
                    h5f['input'][:] = input_np

                    # Store min and max as attributes
                    h5f.attrs['net_reward_min'] = float(np.min(net_rewards_np))
                    h5f.attrs['net_reward_max'] = float(np.max(net_rewards_np))

                    print(f"Saved planning frames to {planning_h5_path}")
            except Exception as e:
                print(f"Error saving planning frames for {replay_name}: {e}")

    # ----- Save Battle Frames in Bulk -----
    if (all_battle_frames and (include_image or include_position or include_player_charge or include_enemy_charge)):
        # Initialize empty lists to collect data
        images = []
        positions = []
        player_charges = []
        enemy_charges = []
        net_rewards = []

        frames_to_remove = []

        # Collect data
        for frame in all_battle_frames:
            # Process based on included features
            if include_image:
                input_str = frame.get('input_str')
                if input_str is None:
                    print(f"No 'input_str' found in frame with timestamp {frame['timestamp']}, skipping this frame.")
                    frames_to_remove.append(frame)
                    continue  # Skip this frame

                # Check for invalid input strings
                if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
                    print(f"Invalid 'input_str' in frame with timestamp {frame['timestamp']}, skipping this frame.")
                    frames_to_remove.append(frame)
                    continue  # Skip invalid input strings

                # Load image and convert to numpy array
                try:
                    image = Image.open(frame['image_path']).convert('RGB')
                    if image.mode != 'RGB':
                        image = image.convert('RGB')  # Ensures image is in RGB format
                    image_np = np.array(image)
                    if image_np.shape != (160, 240, 3):
                        frames_to_remove.append(frame)
                        continue  # Skip frames with unexpected image dimensions
                    image_tensor = image_np.astype(np.float32) / 255.0  # Normalize
                except Exception as e:
                    print(f"Error loading image {frame['image_path']}: {e}")
                    frames_to_remove.append(frame)
                    continue  # Skip frames with image loading issues

                images.append(image_tensor.transpose(2, 0, 1))  # Shape: (3, 160, 240)

            if include_position:
                if position_type == 'float':
                    player_position = frame.get('player_position', [0.0, 0.0])
                    enemy_position = frame.get('enemy_position', [0.0, 0.0])
                    positions.append(player_position + enemy_position)  # Concatenate x and y
                elif position_type == 'grid':
                    player_grid = frame.get('player_grid', [[0]*3 for _ in range(6)])
                    enemy_grid = frame.get('enemy_grid', [[0]*3 for _ in range(6)])
                    flattened_player_grid = np.concatenate(player_grid).astype(np.float32)
                    flattened_enemy_grid = np.concatenate(enemy_grid).astype(np.float32)
                    positions.append(np.concatenate([flattened_player_grid, flattened_enemy_grid]))  # Shape: (36,)
                else:
                    raise ValueError(f"Unknown position_type: {position_type}")

            if include_player_charge:
                player_charge = frame.get('player_charge', 0)
                player_charges.append(player_charge)

            if include_enemy_charge:
                enemy_charge = frame.get('enemy_charge', 0)
                enemy_charges.append(enemy_charge)

            # Collect net_rewards
            net_reward = frame.get('assigned_reward', 0.0)
            net_rewards.append(net_reward)

        # Remove frames that failed filtering
        for frame in frames_to_remove:
            if frame in all_battle_frames:
                all_battle_frames.remove(frame)

        print(f"Number of battle frames: {len(images) if include_image else 0} "
              f"{len(positions) if include_position else 0} "
              f"{len(player_charges) if include_player_charge else 0} "
              f"{len(enemy_charges) if include_enemy_charge else 0}")
        print(f"Number of removed battle frames: {len(frames_to_remove)}")

        # Determine the number of valid samples
        num_samples = max(
            len(images) if include_image else 0,
            len(positions) if include_position else 0,
            len(player_charges) if include_player_charge else 0,
            len(enemy_charges) if include_enemy_charge else 0,
            len(net_rewards)
        )

        if num_samples == 0:
            print(f"No valid battle frames to save for {replay_name}, skipping battle cache.")
        else:
            try:
                with h5py.File(battle_h5_path, 'w') as h5f:
                    # Define dynamic chunk sizes and create datasets based on included features
                    if include_image:
                        h5f.create_dataset(
                            'images',
                            shape=(num_samples, 3, 160, 240),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples, 3, 160, 240)
                        )
                        images_np = np.stack(images, axis=0)
                        h5f['images'][:] = images_np

                    if include_position:
                        if position_type == 'float':
                            h5f.create_dataset(
                                'positions',
                                shape=(num_samples, 4),  # player_x, player_y, enemy_x, enemy_y
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4,
                                chunks=get_chunk_size(100, num_samples, 4)
                            )
                            positions_np = np.array(positions, dtype=np.float32)
                            h5f['positions'][:] = positions_np
                        elif position_type == 'grid':
                            h5f.create_dataset(
                                'positions',
                                shape=(num_samples, 36),  # 6x3 grids concatenated
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4,
                                chunks=get_chunk_size(100, num_samples, 36)
                            )
                            positions_np = np.array(positions, dtype=np.float32)
                            h5f['positions'][:] = positions_np

                    if include_player_charge:
                        h5f.create_dataset(
                            'player_charges',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples)
                        )
                        player_charges_np = np.array(player_charges, dtype=np.float32)
                        h5f['player_charges'][:] = player_charges_np

                    if include_enemy_charge:
                        h5f.create_dataset(
                            'enemy_charges',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=get_chunk_size(100, num_samples)
                        )
                        enemy_charges_np = np.array(enemy_charges, dtype=np.float32)
                        h5f['enemy_charges'][:] = enemy_charges_np

                    # Always save net_rewards
                    h5f.create_dataset(
                        'net_rewards',
                        shape=(num_samples,),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        chunks=get_chunk_size(100, num_samples)
                    )
                    net_rewards_np = np.array(net_rewards, dtype=np.float32)
                    h5f['net_rewards'][:] = net_rewards_np

                    # Convert 'input_str' to binary arrays
                    inputs = [[int(c) for c in frame['input_str']] for frame in all_battle_frames]
                    print(inputs)
                    h5f.create_dataset(
                        'input',
                        shape=(num_samples, 16),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        chunks=get_chunk_size(100, num_samples, 16)
                    )
                    input_np = np.array(inputs, dtype=np.float32)
                    h5f['input'][:] = input_np

                    # Store min and max as attributes
                    h5f.attrs['net_reward_min'] = float(np.min(net_rewards_np))
                    h5f.attrs['net_reward_max'] = float(np.max(net_rewards_np))

                    print(f"Saved battle frames to {battle_h5_path}")
            except Exception as e:
                print(f"Error saving battle frames for {replay_name}: {e}")

    return_code = f"{replay_name}: Completed"

    return return_code

# Process all replays in parallel using ProcessPoolExecutor
def process_all_replays_parallel(max_workers=4):
    """
    Process all replays in parallel using ProcessPoolExecutor.
    :param max_workers: Number of worker processes to use. Defaults to number of CPU cores.
    """
    os.makedirs(PLANNING_CACHE_DIR, exist_ok=True)
    os.makedirs(BATTLE_CACHE_DIR, exist_ok=True)
    replay_dirs = glob.glob(os.path.join(TRAINING_DATA_DIR, '*'))
    replay_dirs = [d for d in replay_dirs if os.path.isdir(d)]

    if not replay_dirs:
        print(f"No replay folders found in {TRAINING_DATA_DIR}.")
        return

    # Load configuration and initialize strategy
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    try:
        strategy = initialize_strategy(config)
    except Exception as e:
        print(f"Error initializing strategy: {e}")
        return

    # Extract image_memory from config
    image_memory = get_image_memory()
    if image_memory < 1:
        print(f"Invalid image_memory={image_memory} in config, setting to 1.")
        image_memory = 1

    # Determine the number of workers
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"Starting parallel processing with {max_workers} workers using {strategy.__class__.__name__}...")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all replay directories to the executor
        future_to_replay = {executor.submit(process_replay, replay_dir, strategy, config=config): replay_dir for replay_dir in replay_dirs}

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_replay), total=len(future_to_replay), desc='Processing all replays'):
            replay_dir = future_to_replay[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                replay_name = os.path.basename(replay_dir)
                print(f"{replay_name}: Generated an exception: {exc}")

if __name__ == '__main__':
    process_all_replays_parallel()
