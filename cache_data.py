import os
import json
import glob
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm  # For progress bars

# Paths
TRAINING_DATA_DIR = 'training_data'
TRAINING_CACHE_DIR = 'training_cache'
WINDOW_SIZE = 60  # Number of frames to look ahead for rewards/punishments

def process_replay(replay_dir):
    # Create the cache directory for this replay
    replay_name = os.path.basename(replay_dir)
    cache_dir = os.path.join(TRAINING_CACHE_DIR, replay_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Collect all JSON files (exclude winner.json)
    json_files = sorted(glob.glob(os.path.join(replay_dir, '*.json')))
    json_files = [f for f in json_files if not os.path.basename(f) == 'winner.json']

    frames = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract timestamp from filename or use current time if not available
            try:
                timestamp = int(os.path.splitext(os.path.basename(json_file))[0])
            except ValueError:
                timestamp = int(time.time() * 1000)
            image_path = data['image_path']
            input_str = data['input']
            reward = data.get('reward') or 0
            punishment = data.get('punishment') or 0
            frames.append({
                'timestamp': timestamp,
                'image_path': os.path.join(replay_dir, os.path.basename(image_path)),
                'input_str': input_str,
                'reward': reward,
                'punishment': punishment
            })
    
    # Sort frames by timestamp
    frames.sort(key=lambda x: x['timestamp'])
    
    num_frames = len(frames)
    # Precompute weights for the window
    weights = np.array([(WINDOW_SIZE - k) / WINDOW_SIZE for k in range(WINDOW_SIZE)])
    
    # Extract rewards and punishments for all frames
    rewards = np.array([frame['reward'] for frame in frames])
    punishments = np.array([frame['punishment'] for frame in frames])
    net_rewards = rewards - punishments

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
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # Shape: (3, 160, 240)
        except Exception as e:
            print(f"Failed to load image {frame['image_path']}: {e}")
            continue  # Skip this frame

        # Convert input string to tensor
        input_tensor = torch.tensor([int(bit) for bit in input_str], dtype=torch.float32)

        # Get net reward
        net_reward = cumulative_rewards[idx]

        # Create sample dictionary
        sample = {
            'image': image_tensor,
            'input': input_tensor,
            'net_reward': net_reward
        }

        # Save sample to cache
        sample_path = os.path.join(cache_dir, f'{idx:06d}.pt')
        torch.save(sample, sample_path)

def process_all_replays():
    os.makedirs(TRAINING_CACHE_DIR, exist_ok=True)
    replay_dirs = glob.glob(os.path.join(TRAINING_DATA_DIR, '*'))
    for replay_dir in replay_dirs:
        if os.path.isdir(replay_dir):
            process_replay(replay_dir)
            
if __name__ == '__main__':
    process_all_replays()
