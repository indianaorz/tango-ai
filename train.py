# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import numpy as np
from tqdm import tqdm
import random
from torch.cuda.amp import GradScaler, autocast
import gc  # Import garbage collector
from utils import (
    get_exponential_sample,
    get_image_memory,
    get_checkpoint_path,
    get_exponental_amount,
    get_root_dir
)  # Import the helper functions
from cache_data import process_replay

import h5py
import shutil
import wandb  # Import wandb

# Define constants
TEMP_DIR = os.path.join(get_root_dir(), 'temp')

# Create temp dir if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

from h5_game_dataset import H5GameDataset
from game_dataset import GameDataset
from game_input_predictor import GameInputPredictor
from preloaded_h5_game_dataset import PreloadedH5GameDataset

# Helper function for clearing memory, cache and forcing garbage collection
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Define a separate train_model function that can handle multiple models
def train_model(model, train_loader, optimizer, criterion, device,
               epoch, net_reward_min, net_reward_max, scaler, model_name):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    # Create the progress bar for the epoch
    epoch_progress = tqdm(train_loader,
                          desc=f"Epoch {epoch+1} [{model_name}]",
                          leave=False, position=2, ncols=100)

    for batch_idx, batch in enumerate(epoch_progress):
        # Extract fields from the batch dictionary
        images = batch['image'].to(device)                    # Shape: (batch_size, 3, D, 160, 240)
        inputs = batch['input'].to(device)                    # Shape: (batch_size, 16)
        player_health = batch['player_health'].unsqueeze(1).to(device)    # Shape: (batch_size, 1)
        enemy_health = batch['enemy_health'].unsqueeze(1).to(device)      # Shape: (batch_size, 1)
        player_grid = batch['player_grid'].to(device)          # Shape: (batch_size, 6, 3)
        enemy_grid = batch['enemy_grid'].to(device)            # Shape: (batch_size, 6, 3)
        inside_window = batch['inside_window'].unsqueeze(1).to(device)  # Shape: (batch_size, 1)
        net_rewards = batch['net_reward'].to(device)            # Shape: (batch_size,)

        # Normalize net_rewards to [0, 1] instead of [-1, 1]
        net_rewards_normalized = (net_rewards - net_reward_min) / (net_reward_max - net_reward_min + 1e-6)
        net_rewards_normalized = torch.clamp(net_rewards_normalized, min=0.0, max=1.0)

        optimizer.zero_grad()

        with autocast():
            # Forward pass with additional inputs
            outputs = model(images, player_grid, enemy_grid, inside_window, player_health, enemy_health)  # Shape: (batch_size, 16)

            # Compute loss per sample
            loss = criterion(outputs, inputs)  # BCEWithLogitsLoss with reduction='none'
            loss_per_sample = loss.mean(dim=1)  # Shape: (batch_size,)

            # Use normalized net_rewards as sample_weights
            sample_weights = net_rewards_normalized  # Ensure weights are in [0, 1]

            # Multiply loss per sample by sample weights
            weighted_loss = loss_per_sample * sample_weights  # Shape: (batch_size,)

            # Compute mean loss over batch
            loss = weighted_loss.mean()

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        avg_loss = epoch_loss / (batch_idx + 1)

        # Update the progress bar with the current and average loss
        epoch_progress.set_postfix({
            'Batch Loss': f'{loss.item():.6f}',
            'Avg Loss': f'{avg_loss:.6f}'
        })

        # Log metrics to wandb
        wandb.log({
            f'Train/{model_name}/Batch_Loss': loss.item(),
            f'Train/{model_name}/Avg_Batch_Loss': avg_loss,
            'Train/Epoch': epoch + 1,
            'Train/Batch': batch_idx + 1,
            f'Train/{model_name}/Sample_Weights_Mean': sample_weights.mean().item(),
            f'Train/{model_name}/Sample_Weights_Std': sample_weights.std().item(),
        })

        # Clear variables at the end of the loop to free memory
        clear_memory()

    epoch_progress.close()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} [{model_name}], Average Loss: {avg_epoch_loss:.6f}")

    # Log epoch-level metrics
    wandb.log({
        f'Train/{model_name}/Epoch_Avg_Loss': avg_epoch_loss,
        'Train/Epoch': epoch + 1,
    })

    return avg_epoch_loss

# Define cache directories
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')
# Create dir if not exist
os.makedirs(TRAINING_CACHE_DIR, exist_ok=True)

def main():
    # Initialize wandb
    wandb.init(
        project="sigma_5_mem_1_sub",  # Replace with your wandb project name
        name="experiment_name",       # (Optional) Name for this run
        config={
            "batch_size": 256,
            "learning_rate": 1e-4,
            "image_memory": get_image_memory(),
            "num_epochs": 100,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "load_data_into_gpu": True,  # New flag added
            "efficient_storage": True,   # New flag for efficient storage
            # Add other hyperparameters you want to track
        }
    )

    config = wandb.config  # Access the config object

    cache_dir = TRAINING_CACHE_DIR
    image_memory = config.image_memory  # Use wandb config
    checkpoint_dir = os.path.join(get_root_dir(), 'checkpoints')
    raw_dir = os.path.join(get_root_dir(), 'training_data')
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    max_checkpoints = 5
    subset_size = 2  # Number of directories per subset
    checkpoint_freq = 1  # Save a checkpoint every epoch
    num_full_epochs = config.num_epochs
    preprocess = True  # Set to False to skip preprocessing the dataset
    load_into_memory = True  # Set to True to load dataset into memory
    is_raw = True

    # Extract the new flags
    load_data_into_gpu = config.get("load_data_into_gpu", False)
    efficient_storage = config.get("efficient_storage", False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_data_into_gpu and device.type != 'cuda':
        print("Warning: 'load_data_into_gpu' is set to True but no CUDA device is available. Data will remain on CPU.")
        load_data_into_gpu = False

    # Initialize two models: Planning and Battle
    planning_model = GameInputPredictor(image_memory=image_memory).to(device)
    battle_model = GameInputPredictor(image_memory=image_memory).to(device)

    # Define separate criteria for both models
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Define separate optimizers for both models
    planning_optimizer = optim.Adam(planning_model.parameters(), lr=learning_rate)
    battle_optimizer = optim.Adam(battle_model.parameters(), lr=learning_rate)

    # Log the model architectures to wandb
    wandb.watch(planning_model, log="all", log_freq=100)
    wandb.watch(battle_model, log="all", log_freq=100)

    # Check for existing checkpoints for Planning Model
    latest_planning_checkpoint = get_checkpoint_path(os.path.join(checkpoint_dir, 'planning'), image_memory)
    start_epoch_planning = 0
    if latest_planning_checkpoint:
        print(f"Loading Planning Model checkpoint: {latest_planning_checkpoint}")

        checkpoint = torch.load(latest_planning_checkpoint, map_location=device)

        # Ensure 'model_state_dict' exists in the checkpoint
        if 'model_state_dict' in checkpoint:
            planning_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Planning checkpoint does not contain 'model_state_dict'")

        if 'optimizer_state_dict' in checkpoint:
            planning_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise KeyError("Planning checkpoint does not contain 'optimizer_state_dict'")

        if 'epoch' in checkpoint:
            start_epoch_planning = checkpoint['epoch']
            print(f"Resuming Planning Model from epoch {start_epoch_planning}")
        else:
            raise KeyError("Planning checkpoint does not contain 'epoch'")
    else:
        print("No Planning Model checkpoint found, starting from scratch.")

    # Check for existing checkpoints for Battle Model
    latest_battle_checkpoint = get_checkpoint_path(os.path.join(checkpoint_dir, 'battle'), image_memory)
    start_epoch_battle = 0
    if latest_battle_checkpoint:
        print(f"Loading Battle Model checkpoint: {latest_battle_checkpoint}")

        checkpoint = torch.load(latest_battle_checkpoint, map_location=device)

        # Ensure 'model_state_dict' exists in the checkpoint
        if 'model_state_dict' in checkpoint:
            battle_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Battle checkpoint does not contain 'model_state_dict'")

        if 'optimizer_state_dict' in checkpoint:
            battle_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise KeyError("Battle checkpoint does not contain 'optimizer_state_dict'")

        if 'epoch' in checkpoint:
            start_epoch_battle = checkpoint['epoch']
            print(f"Resuming Battle Model from epoch {start_epoch_battle}")
        else:
            raise KeyError("Battle checkpoint does not contain 'epoch'")
    else:
        print("No Battle Model checkpoint found, starting from scratch.")

    # List all replay directories
    # if raw, get all replay dirs from raw_dir
    if is_raw:
        all_replay_dirs = [os.path.join(raw_dir, d) for d in os.listdir(
            raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    else:
        all_replay_dirs = [os.path.join(cache_dir, d) for d in os.listdir(
            cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]

    all_replay_dirs.sort()
    num_total_dirs = len(all_replay_dirs)
    print(f"Total replay directories: {num_total_dirs}")

    if num_total_dirs == 0:
        print("No directories found in the cache. Exiting.")
        return

    # Calculate total subsets per full epoch based on subset_size
    subsets_per_full_epoch = (num_total_dirs + subset_size - 1) // subset_size  # Ceiling division

    # Create progress bar for overall training progress
    overall_progress = tqdm(total=num_full_epochs,
                            desc="Overall Training Progress",
                            unit="epoch", position=0, leave=True)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Prepare separate cache directories for Planning and Battle models
    planning_cache_dir = os.path.join(TRAINING_CACHE_DIR, 'planning')
    battle_cache_dir = os.path.join(TRAINING_CACHE_DIR, 'battle')
    os.makedirs(planning_cache_dir, exist_ok=True)
    os.makedirs(battle_cache_dir, exist_ok=True)

    # Determine the starting epoch based on both models
    start_epoch = max(start_epoch_planning, start_epoch_battle)

    # Handle multiple subsets
    for full_epoch in range(start_epoch, num_full_epochs):
        print(f"\nStarting full epoch {full_epoch + 1}/{num_full_epochs}")

        # Shuffle directories at the start of each full epoch if desired
        random.shuffle(all_replay_dirs)

        # Divide directories into subsets
        subsets = [
            all_replay_dirs[i:i + subset_size]
            for i in range(0, num_total_dirs, subset_size)
        ]

        for subset_idx, subset_dirs in enumerate(subsets):
            print(f"\nTraining on subset {subset_idx + 1}/{len(subsets)}: {', '.join([os.path.basename(d) for d in subset_dirs])}")

            # Create a separate progress bar for subsets
            subset_progress = tqdm(total=2, desc=f"Subset {subset_idx+1}/{len(subsets)}", position=1, leave=False)

            # Define HDF5 paths for both Planning and Battle
            planning_h5_path = os.path.join(planning_cache_dir, f'subset_{subset_idx}.h5')
            battle_h5_path = os.path.join(battle_cache_dir, f'subset_{subset_idx}.h5')

            if efficient_storage:
                # Process Planning Data
                if os.path.exists(planning_h5_path):
                    print(f"Loading Planning dataset from HDF5 file: {planning_h5_path}")
                    planning_dataset = PreloadedH5GameDataset(
                        h5_path=planning_h5_path,
                        device=device,
                        batch_size=1000,  # Adjust based on your system's memory
                        prefetch_batches=4,
                        image_memory=image_memory
                    )
                else:
                    print("HDF5 file for Planning not found. Processing and creating HDF5 file.")
                    planning_dataset = GameDataset(
                        replay_dirs=subset_dirs,
                        image_memory=image_memory,
                        preprocess=preprocess,
                        load_into_memory=load_into_memory,
                        raw_dir=raw_dir,
                        is_raw=is_raw,
                        load_data_into_gpu=False,
                        device=device,
                        data_type='planning'  # Specify data type to separate planning frames
                    )
                    # Save Planning data to HDF5
                    print(f"Saving Planning processed data to HDF5 file: {planning_h5_path}")
                    with h5py.File(planning_h5_path, 'w') as h5f_planning:
                        num_samples_planning = len(planning_dataset)
                        # Define chunk sizes based on new fields
                        image_chunk = (1, 3, image_memory, 160, 240)  # One image per chunk
                        input_chunk = (1, 16)  # One input per chunk
                        player_health_chunk = (1,)
                        enemy_health_chunk = (1,)
                        player_grid_chunk = (1, 6, 3)
                        enemy_grid_chunk = (1, 6, 3)
                        inside_window_chunk = (1,)
                        net_reward_chunk = (1,)

                        # Create datasets with compression
                        h5f_planning.create_dataset(
                            'images',
                            shape=(num_samples_planning, 3, image_memory, 160, 240),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=image_chunk
                        )
                        h5f_planning.create_dataset(
                            'inputs',
                            shape=(num_samples_planning, 16),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=input_chunk
                        )
                        h5f_planning.create_dataset(
                            'player_healths',
                            shape=(num_samples_planning,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_health_chunk
                        )
                        h5f_planning.create_dataset(
                            'enemy_healths',
                            shape=(num_samples_planning,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_health_chunk
                        )
                        h5f_planning.create_dataset(
                            'player_grids',
                            shape=(num_samples_planning, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_grid_chunk
                        )
                        h5f_planning.create_dataset(
                            'enemy_grids',
                            shape=(num_samples_planning, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_grid_chunk
                        )
                        h5f_planning.create_dataset(
                            'inside_windows',
                            shape=(num_samples_planning,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=inside_window_chunk
                        )
                        h5f_planning.create_dataset(
                            'net_rewards',
                            shape=(num_samples_planning,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=net_reward_chunk
                        )

                        for idx in tqdm(range(num_samples_planning), desc="Saving Planning to HDF5", position=2):
                            sample = planning_dataset[idx]
                            h5f_planning['images'][idx] = sample['image'].cpu().numpy()
                            h5f_planning['inputs'][idx] = sample['input'].cpu().numpy()
                            h5f_planning['player_healths'][idx] = sample['player_health'].item()
                            h5f_planning['enemy_healths'][idx] = sample['enemy_health'].item()
                            h5f_planning['player_grids'][idx] = sample['player_grid'].cpu().numpy()
                            h5f_planning['enemy_grids'][idx] = sample['enemy_grid'].cpu().numpy()
                            h5f_planning['inside_windows'][idx] = sample['inside_window'].item()
                            h5f_planning['net_rewards'][idx] = sample['net_reward'].item()

                        # Save min and max as attributes
                        h5f_planning.attrs['net_reward_min'] = float(planning_dataset.net_reward_min)
                        h5f_planning.attrs['net_reward_max'] = float(planning_dataset.net_reward_max)
                    # Reload dataset from HDF5 using the preloaded dataset
                    planning_dataset = PreloadedH5GameDataset(
                        h5_path=planning_h5_path,
                        device=device,
                        batch_size=1000,
                        prefetch_batches=4,
                        image_memory=image_memory
                    )

                # Process Battle Data
                if os.path.exists(battle_h5_path):
                    print(f"Loading Battle dataset from HDF5 file: {battle_h5_path}")
                    battle_dataset = PreloadedH5GameDataset(
                        h5_path=battle_h5_path,
                        device=device,
                        batch_size=1000,  # Adjust based on your system's memory
                        prefetch_batches=4,
                        image_memory=image_memory
                    )
                else:
                    print("HDF5 file for Battle not found. Processing and creating HDF5 file.")
                    battle_dataset = GameDataset(
                        replay_dirs=subset_dirs,
                        image_memory=image_memory,
                        preprocess=preprocess,
                        load_into_memory=load_into_memory,
                        raw_dir=raw_dir,
                        is_raw=is_raw,
                        load_data_into_gpu=False,
                        device=device,
                        data_type='battle'  # Specify data type to separate battle frames
                    )
                    # Save Battle data to HDF5
                    print(f"Saving Battle processed data to HDF5 file: {battle_h5_path}")
                    with h5py.File(battle_h5_path, 'w') as h5f_battle:
                        num_samples_battle = len(battle_dataset)
                        # Define chunk sizes based on new fields
                        image_chunk = (1, 3, image_memory, 160, 240)  # One image per chunk
                        input_chunk = (1, 16)  # One input per chunk
                        player_health_chunk = (1,)
                        enemy_health_chunk = (1,)
                        player_grid_chunk = (1, 6, 3)
                        enemy_grid_chunk = (1, 6, 3)
                        inside_window_chunk = (1,)
                        net_reward_chunk = (1,)

                        # Create datasets with compression
                        h5f_battle.create_dataset(
                            'images',
                            shape=(num_samples_battle, 3, image_memory, 160, 240),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=image_chunk
                        )
                        h5f_battle.create_dataset(
                            'inputs',
                            shape=(num_samples_battle, 16),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=input_chunk
                        )
                        h5f_battle.create_dataset(
                            'player_healths',
                            shape=(num_samples_battle,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_health_chunk
                        )
                        h5f_battle.create_dataset(
                            'enemy_healths',
                            shape=(num_samples_battle,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_health_chunk
                        )
                        h5f_battle.create_dataset(
                            'player_grids',
                            shape=(num_samples_battle, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_grid_chunk
                        )
                        h5f_battle.create_dataset(
                            'enemy_grids',
                            shape=(num_samples_battle, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_grid_chunk
                        )
                        h5f_battle.create_dataset(
                            'inside_windows',
                            shape=(num_samples_battle,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=inside_window_chunk
                        )
                        h5f_battle.create_dataset(
                            'net_rewards',
                            shape=(num_samples_battle,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=net_reward_chunk
                        )

                        for idx in tqdm(range(num_samples_battle), desc="Saving Battle to HDF5", position=2):
                            sample = battle_dataset[idx]
                            h5f_battle['images'][idx] = sample['image'].cpu().numpy()
                            h5f_battle['inputs'][idx] = sample['input'].cpu().numpy()
                            h5f_battle['player_healths'][idx] = sample['player_health'].item()
                            h5f_battle['enemy_healths'][idx] = sample['enemy_health'].item()
                            h5f_battle['player_grids'][idx] = sample['player_grid'].cpu().numpy()
                            h5f_battle['enemy_grids'][idx] = sample['enemy_grid'].cpu().numpy()
                            h5f_battle['inside_windows'][idx] = sample['inside_window'].item()
                            h5f_battle['net_rewards'][idx] = sample['net_reward'].item()

                        # Save min and max as attributes
                        h5f_battle.attrs['net_reward_min'] = float(battle_dataset.net_reward_min)
                        h5f_battle.attrs['net_reward_max'] = float(battle_dataset.net_reward_max)
                    # Reload dataset from HDF5 using the preloaded dataset
                    battle_dataset = PreloadedH5GameDataset(
                        h5_path=battle_h5_path,
                        device=device,
                        batch_size=1000,
                        prefetch_batches=4,
                        image_memory=image_memory
                    )

            else:
                # Non-efficient storage: handle planning and battle separately
                planning_dataset = GameDataset(
                    replay_dirs=subset_dirs,
                    image_memory=image_memory,
                    preprocess=preprocess,
                    load_into_memory=load_into_memory,
                    raw_dir=raw_dir,
                    is_raw=is_raw,
                    load_data_into_gpu=load_data_into_gpu,
                    data_type='planning'
                )
                battle_dataset = GameDataset(
                    replay_dirs=subset_dirs,
                    image_memory=image_memory,
                    preprocess=preprocess,
                    load_into_memory=load_into_memory,
                    raw_dir=raw_dir,
                    is_raw=is_raw,
                    load_data_into_gpu=load_data_into_gpu,
                    data_type='battle'
                )

            # Check if planning_dataset has samples
            if efficient_storage:
                planning_len = len(planning_dataset)
                battle_len = len(battle_dataset)
            else:
                planning_len = len(planning_dataset)
                battle_len = len(battle_dataset)

            if planning_len == 0:
                print(f"Warning: Planning dataset for subset {subset_idx + 1} has no samples. Skipping Planning.")
                subset_progress.update(1)
            else:
                # Get net_reward_min and net_reward_max
                planning_net_reward_min = planning_dataset.net_reward_min
                planning_net_reward_max = planning_dataset.net_reward_max

                # Convert to torch tensors
                planning_net_reward_min = torch.tensor(planning_net_reward_min, dtype=torch.float32).to(device)
                planning_net_reward_max = torch.tensor(planning_net_reward_max, dtype=torch.float32).to(device)
                if planning_net_reward_max == planning_net_reward_min:
                    print("Warning: planning_net_reward_max == planning_net_reward_min, setting sample_weights to 1")
                    planning_net_reward_max = planning_net_reward_min + 1e-6

                # Initialize DataLoader for Planning
                if efficient_storage:
                    planning_loader = DataLoader(
                        planning_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,  # No workers since data is already on GPU
                        pin_memory=False
                    )
                else:
                    planning_loader = DataLoader(
                        planning_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=16,  # Adjust based on your CPU cores
                        pin_memory=True
                    )

                # Check if DataLoader has samples
                if len(planning_loader) == 0:
                    print(f"Warning: Planning DataLoader for subset {subset_idx + 1} has no samples. Skipping Planning.")
                else:
                    # Train Planning Model
                    avg_epoch_loss_planning = train_model(
                        planning_model,
                        planning_loader,
                        planning_optimizer,
                        criterion,
                        device,
                        full_epoch,
                        planning_net_reward_min,
                        planning_net_reward_max,
                        scaler,
                        model_name="Planning_Model"
                    )
                    subset_progress.update(1)

                # Clear memory
                clear_memory()

            if battle_len == 0:
                print(f"Warning: Battle dataset for subset {subset_idx + 1} has no samples. Skipping Battle.")
                subset_progress.update(1)
            else:
                # Get net_reward_min and net_reward_max
                battle_net_reward_min = battle_dataset.net_reward_min
                battle_net_reward_max = battle_dataset.net_reward_max

                # Convert to torch tensors
                battle_net_reward_min = torch.tensor(battle_net_reward_min, dtype=torch.float32).to(device)
                battle_net_reward_max = torch.tensor(battle_net_reward_max, dtype=torch.float32).to(device)
                if battle_net_reward_max == battle_net_reward_min:
                    print("Warning: battle_net_reward_max == battle_net_reward_min, setting sample_weights to 1")
                    battle_net_reward_max = battle_net_reward_min + 1e-6

                # Initialize DataLoader for Battle
                if efficient_storage:
                    battle_loader = DataLoader(
                        battle_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,  # No workers since data is already on GPU
                        pin_memory=False
                    )
                else:
                    battle_loader = DataLoader(
                        battle_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=16,  # Adjust based on your CPU cores
                        pin_memory=True
                    )

                # Check if DataLoader has samples
                if len(battle_loader) == 0:
                    print(f"Warning: Battle DataLoader for subset {subset_idx + 1} has no samples. Skipping Battle.")
                else:
                    # Train Battle Model
                    avg_epoch_loss_battle = train_model(
                        battle_model,
                        battle_loader,
                        battle_optimizer,
                        criterion,
                        device,
                        full_epoch,
                        battle_net_reward_min,
                        battle_net_reward_max,
                        scaler,
                        model_name="Battle_Model"
                    )
                    subset_progress.update(1)

                # Clear memory
                clear_memory()

            # Update the subset progress bar
            subset_progress.close()

            # Checkpointing per subset or per epoch as desired
            if (full_epoch + 1) % checkpoint_freq == 0:
                # Save Planning Model checkpoint
                save_at_planning = os.path.join(checkpoint_dir, 'planning')
                os.makedirs(save_at_planning, exist_ok=True)
                checkpoint_path_planning = os.path.join(
                    save_at_planning, f"checkpoint_epoch_{full_epoch+1}.pt")
                torch.save({
                    'epoch': full_epoch + 1,
                    'model_state_dict': planning_model.state_dict(),
                    'optimizer_state_dict': planning_optimizer.state_dict(),
                }, checkpoint_path_planning)
                print(f"Saved Planning Model checkpoint: {checkpoint_path_planning}")

                # Optionally upload to wandb as an artifact
                # artifact_planning = wandb.Artifact('planning-model-checkpoint', type='checkpoint')
                # artifact_planning.add_file(checkpoint_path_planning)
                # wandb.log_artifact(artifact_planning)
                # print(f"Uploaded Planning Model checkpoint to wandb: {checkpoint_path_planning}")

                # Remove old Planning checkpoints
                existing_planning_checkpoints = sorted(
                    glob.glob(os.path.join(save_at_planning, 'checkpoint_epoch_*.pt')),
                    key=lambda x: int(x.split('_')[-1].split('.')[0])
                )
                if len(existing_planning_checkpoints) > max_checkpoints:
                    oldest_planning_checkpoint = existing_planning_checkpoints[0]
                    os.remove(oldest_planning_checkpoint)
                    print(f"Removed old Planning Model checkpoint: {oldest_planning_checkpoint}")

                # Save Battle Model checkpoint
                save_at_battle = os.path.join(checkpoint_dir, 'battle')
                os.makedirs(save_at_battle, exist_ok=True)
                checkpoint_path_battle = os.path.join(
                    save_at_battle, f"checkpoint_epoch_{full_epoch+1}.pt")
                torch.save({
                    'epoch': full_epoch + 1,
                    'model_state_dict': battle_model.state_dict(),
                    'optimizer_state_dict': battle_optimizer.state_dict(),
                }, checkpoint_path_battle)
                print(f"Saved Battle Model checkpoint: {checkpoint_path_battle}")

                # Optionally upload to wandb as an artifact
                # artifact_battle = wandb.Artifact('battle-model-checkpoint', type='checkpoint')
                # artifact_battle.add_file(checkpoint_path_battle)
                # wandb.log_artifact(artifact_battle)
                # print(f"Uploaded Battle Model checkpoint to wandb: {checkpoint_path_battle}")

                # Remove old Battle checkpoints
                existing_battle_checkpoints = sorted(
                    glob.glob(os.path.join(save_at_battle, 'checkpoint_epoch_*.pt')),
                    key=lambda x: int(x.split('_')[-1].split('.')[0])
                )
                if len(existing_battle_checkpoints) > max_checkpoints:
                    oldest_battle_checkpoint = existing_battle_checkpoints[0]
                    os.remove(oldest_battle_checkpoint)
                    print(f"Removed old Battle Model checkpoint: {oldest_battle_checkpoint}")

            # Update the overall progress bar
            overall_progress.update(1)

            # Cleanup
            if efficient_storage:
                del planning_loader
                del battle_loader
            else:
                del planning_dataset
                del battle_dataset
                del planning_loader
                del battle_loader
            clear_memory()

        # After each full epoch, re-scan for new directories in the training folder
        print("\nScanning for new training data...")
        updated_replay_dirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

        # Compare the new list with the current replay directories
        new_dirs = list(set(updated_replay_dirs) - set(all_replay_dirs))
        if new_dirs:
            print(f"Found {len(new_dirs)} new directories. Adding to training set.")
            all_replay_dirs.extend(new_dirs)  # Add the new directories to the training set
            num_total_dirs = len(all_replay_dirs)  # Update the total number of directories
        else:
            print("No new training data found.")

    # After the training loop
    overall_progress.close()
    print("Training completed successfully.")

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
