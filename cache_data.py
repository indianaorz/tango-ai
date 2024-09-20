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
from utils import get_root_dir, position_to_grid  # Ensure these functions are defined in utils.py
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc
import yaml  # For configuration
from strategies import DefaultStrategy, DodgeStrategy, DamageStrategy, AggressiveStrategy  # Import strategies

# Paths
TRAINING_DATA_DIR = os.path.join(get_root_dir(), 'training_data')
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')
PLANNING_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'planning')
BATTLE_CACHE_DIR = os.path.join(TRAINING_CACHE_DIR, 'battle')
WINDOW_SIZE = 100  # Number of frames to look ahead for rewards/punishments

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
        window_size = parameters.get('window_size', WINDOW_SIZE)
        return DefaultStrategy(window_size=window_size)
    elif strategy_name == 'dodge':
        punishment_value = parameters.get('punishment_value', -0.5)
        window_size = parameters.get('window_size', WINDOW_SIZE)
        return DodgeStrategy(punishment_value=punishment_value, window_size=window_size)
    elif strategy_name == 'damage':
        reward_value = parameters.get('reward_value', 1.0)
        window_size = parameters.get('window_size', WINDOW_SIZE)
        return DamageStrategy(reward_value=reward_value, window_size=window_size)
    elif strategy_name == 'aggressive':
        reward_value = parameters.get('reward_value', 1.0)
        punishment_value = parameters.get('punishment_value', -0.5)
        window_size = parameters.get('window_size', WINDOW_SIZE)
        return AggressiveStrategy(reward_value=reward_value, punishment_value=punishment_value, window_size=window_size)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

# Get dynamic chunk size for HDF5 datasets
def get_chunk_size(default_chunk, num_samples, *dims):
    """Return a tuple for chunk size where the first dimension is the min of default_chunk and num_samples."""
    first_dim = min(default_chunk, num_samples)
    return (first_dim,) + dims

# Process a single replay directory with a given strategy
def process_replay(replay_dir, strategy, planning_output_dir=PLANNING_CACHE_DIR, battle_output_dir=BATTLE_CACHE_DIR, rewards_only=False):
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
        # print(f"Reward: {reward}, Punishment: {punishment}")
        # if rewards_only:
        #     punishment = 0
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

        frames_to_remove = []

        # Collect data with filtering before HDF5 allocation
        for frame in all_planning_frames:
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

            # Convert input string to binary array
            try:
                input_tensor = np.array([int(bit) for bit in input_str], dtype=np.float32)
            except ValueError:
                frames_to_remove.append(frame)
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

        # Remove frames that failed filtering
        for frame in frames_to_remove:
            if frame in all_planning_frames:
                all_planning_frames.remove(frame)

        print(f"Number of planning frames: {len(images)}")
        print(f"Number of removed planning frames: {len(frames_to_remove)}")

        # After collecting and filtering, determine the number of valid samples
        num_samples = len(images)

        if num_samples == 0:
            print(f"No valid planning frames to save for {replay_name}, skipping planning cache.")
        else:
            try:
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
                    images_np = np.stack(images, axis=0)
                    inputs_np = np.stack(inputs, axis=0)
                    player_healths_np = np.array(player_healths, dtype=np.float32)
                    enemy_healths_np = np.array(enemy_healths, dtype=np.float32)
                    player_grids_np = np.stack(player_grids, axis=0)
                    enemy_grids_np = np.stack(enemy_grids, axis=0)
                    inside_windows_np = np.array(inside_windows, dtype=np.float32)
                    net_rewards_np = np.array(net_rewards, dtype=np.float32)

                    # Assign data to HDF5 datasets in bulk
                    h5f['images'][:] = images_np
                    h5f['inputs'][:] = inputs_np
                    h5f['player_healths'][:] = player_healths_np
                    h5f['enemy_healths'][:] = enemy_healths_np
                    h5f['player_grids'][:] = player_grids_np
                    h5f['enemy_grids'][:] = enemy_grids_np
                    h5f['inside_windows'][:] = inside_windows_np
                    h5f['net_rewards'][:] = net_rewards_np

                    # Store min and max as attributes
                    h5f.attrs['net_reward_min'] = float(net_rewards_np.min())
                    h5f.attrs['net_reward_max'] = float(net_rewards_np.max())

                    print(f"Saved {num_samples} planning frames to {planning_h5_path}")
            except Exception as e:
                print(f"Error saving planning frames for {replay_name}: {e}")

    # ----- Save Battle Frames in Bulk -----
    if all_battle_frames:
        # Initialize empty lists to collect data
        images = []
        inputs = []
        player_healths = []
        enemy_healths = []
        player_grids = []
        enemy_grids = []
        inside_windows = []
        net_rewards = []

        battle_frames_to_remove = []

        # Collect data
        for frame in all_battle_frames:
            #print assigned_reward
            # print(frame['assigned_reward'])
            input_str = frame.get('input_str')
            if input_str is None:
                print(f"No 'input_str' found in frame with timestamp {frame['timestamp']}, skipping this frame.")
                battle_frames_to_remove.append(frame)
                continue  # Skip this frame

            # Check for invalid input strings
            if not isinstance(input_str, str) or len(input_str) != 16 or not set(input_str).issubset({'0', '1'}):
                print(f"Invalid 'input_str' in frame with timestamp {frame['timestamp']}, skipping this frame.")
                battle_frames_to_remove.append(frame)
                continue  # Skip invalid input strings

            # Load image and convert to numpy array
            try:
                image = Image.open(frame['image_path']).convert('RGB')
                if image.mode != 'RGB':
                    image = image.convert('RGB')  # Ensures image is in RGB format
                image_np = np.array(image)
                if image_np.shape != (160, 240, 3):
                    battle_frames_to_remove.append(frame)
                    continue  # Skip frames with unexpected image dimensions
                image_tensor = image_np.astype(np.float32) / 255.0  # Normalize
            except Exception as e:
                print(f"Error loading image {frame['image_path']}: {e}")
                battle_frames_to_remove.append(frame)
                continue  # Skip frames with image loading issues

            # Convert input string to binary array
            try:
                input_tensor = np.array([int(bit) for bit in input_str], dtype=np.float32)
            except ValueError:
                print(f"Invalid 'input_str' in frame with timestamp {frame['timestamp']}, skipping this frame.")
                battle_frames_to_remove.append(frame)
                continue  # Skip frames with invalid input strings

            # Get assigned reward
            net_reward = frame.get('assigned_reward', 0.0)

            # Skip if net_reward is zero
            if net_reward == 0.0:
                print(f"Net reward is zero in frame with timestamp {frame['timestamp']}, skipping this frame.")
                battle_frames_to_remove.append(frame)
                continue

            # Append data to lists
            images.append(image_tensor.transpose(2, 0, 1))  # Shape: (3, 160, 240)
            inputs.append(input_tensor)
            player_healths.append(frame['player_health'])
            enemy_healths.append(frame['enemy_health'])
            player_grids.append(frame['player_grid'])
            enemy_grids.append(frame['enemy_grid'])
            inside_windows.append(float(frame['inside_window']))
            net_rewards.append(net_reward)

        # Remove frames that failed filtering
        for frame in battle_frames_to_remove:
            if frame in all_battle_frames:
                all_battle_frames.remove(frame)

        print(f"Number of battle frames: {len(images)}")
        print(f"Number of removed battle frames: {len(battle_frames_to_remove)}")

        # After collecting and filtering, determine the number of valid samples
        num_samples = len(images)

        if num_samples == 0:
            print(f"No valid battle frames to save for {replay_name}, skipping battle cache.")
        else:
            try:
                with h5py.File(battle_h5_path, 'w') as h5f:
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
                    images_np = np.stack(images, axis=0)
                    inputs_np = np.stack(inputs, axis=0)
                    player_healths_np = np.array(player_healths, dtype=np.float32)
                    enemy_healths_np = np.array(enemy_healths, dtype=np.float32)
                    player_grids_np = np.stack(player_grids, axis=0)
                    enemy_grids_np = np.stack(enemy_grids, axis=0)
                    inside_windows_np = np.array(inside_windows, dtype=np.float32)
                    net_rewards_np = np.array(net_rewards, dtype=np.float32)

                    # Assign data to HDF5 datasets in bulk
                    h5f['images'][:] = images_np
                    h5f['inputs'][:] = inputs_np
                    h5f['player_healths'][:] = player_healths_np
                    h5f['enemy_healths'][:] = enemy_healths_np
                    h5f['player_grids'][:] = player_grids_np
                    h5f['enemy_grids'][:] = enemy_grids_np
                    h5f['inside_windows'][:] = inside_windows_np
                    h5f['net_rewards'][:] = net_rewards_np

                    # Store min and max as attributes
                    h5f.attrs['net_reward_min'] = float(net_rewards_np.min())
                    h5f.attrs['net_reward_max'] = float(net_rewards_np.max())

                    print(f"Saved {num_samples} battle frames to {battle_h5_path}")
            except Exception as e:
                print(f"Error saving battle frames for {replay_name}: {e}")

    return f"{replay_name}: Completed"

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

    # Determine the number of workers
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"Starting parallel processing with {max_workers} workers using {strategy.__class__.__name__}...")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all replay directories to the executor
        future_to_replay = {executor.submit(process_replay, replay_dir, strategy): replay_dir for replay_dir in replay_dirs}

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
