# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from tqdm import tqdm
import random
from torch.cuda.amp import GradScaler, autocast
import gc  # Import garbage collector
from utils import get_exponential_sample, get_image_memory, get_checkpoint_path, get_exponental_amount, get_root_dir  # Import the helper function
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

# Same train_model function as before but now clears memory at the end of each batch
def train_model(model, train_loader, optimizer, criterion, device,
                epoch, net_reward_min, net_reward_max, scaler):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    # Create the progress bar for the epoch
    epoch_progress = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}",
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
            'Train/Batch_Loss': loss.item(),
            'Train/Avg_Batch_Loss': avg_loss,
            'Train/Epoch': epoch + 1,
            'Train/Batch': batch_idx + 1,
            'Train/Sample_Weights_Mean': sample_weights.mean().item(),
            'Train/Sample_Weights_Std': sample_weights.std().item(),
        })

        # Clear variables at the end of the loop to free memory
        clear_memory()

    epoch_progress.close()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.6f}")

    # Log epoch-level metrics
    wandb.log({
        'Train/Epoch_Avg_Loss': avg_epoch_loss,
        'Train/Epoch': epoch + 1,
    })

    return avg_epoch_loss


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
    subset_size = 12  # Number of directories per subset
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

    # Initialize the model, criterion, optimizer outside the loop
    model = GameInputPredictor(image_memory=image_memory).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Log the model architecture
    wandb.watch(model, log="all", log_freq=100)

    # Check for existing checkpoints
    latest_checkpoint = get_checkpoint_path(checkpoint_dir, image_memory)
    start_epoch = 0
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        # Ensure 'model_state_dict' exists in the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'optimizer_state_dict'")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        else:
            raise KeyError("Checkpoint does not contain 'epoch'")
    else:
        print("No checkpoint found, starting from scratch.")

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
            subset_progress = tqdm(total=1, desc=f"Subset {subset_idx+1}/{len(subsets)}", position=1, leave=False)

            if efficient_storage:
                h5_path = os.path.join(cache_dir, f'subset_{subset_idx}.h5')
                if os.path.exists(h5_path):
                    print(f"Loading dataset from HDF5 file: {h5_path}")
                    dataset = PreloadedH5GameDataset(
                        h5_path=h5_path, 
                        device=device, 
                        batch_size=1000,  # Adjust based on your system's memory
                        prefetch_batches=4,  # Adjust based on desired prefetching
                        image_memory=image_memory
                    )
                else:
                    print("HDF5 file not found. Processing and creating HDF5 file.")
                    dataset = GameDataset(
                        replay_dirs=subset_dirs,
                        image_memory=image_memory,
                        preprocess=preprocess,
                        load_into_memory=load_into_memory,
                        raw_dir=raw_dir,
                        is_raw=is_raw,
                        load_data_into_gpu=False,  # Data will be loaded by PreloadedH5GameDataset
                        device=device  # Pass the device
                    )
                    # Save to HDF5
                    print(f"Saving processed data to HDF5 file: {h5_path}")
                    with h5py.File(h5_path, 'w') as h5f:
                        num_samples = len(dataset)
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
                        h5f.create_dataset(
                            'images',
                            shape=(num_samples, 3, image_memory, 160, 240),
                            dtype=np.float32,
                            compression='gzip',          # Choose 'gzip' or 'lzf'
                            compression_opts=4,          # Compression level (1-9 for gzip)
                            chunks=image_chunk
                        )
                        h5f.create_dataset(
                            'inputs',
                            shape=(num_samples, 16),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=input_chunk
                        )
                        h5f.create_dataset(
                            'player_healths',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_health_chunk
                        )
                        h5f.create_dataset(
                            'enemy_healths',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_health_chunk
                        )
                        h5f.create_dataset(
                            'player_grids',
                            shape=(num_samples, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=player_grid_chunk
                        )
                        h5f.create_dataset(
                            'enemy_grids',
                            shape=(num_samples, 6, 3),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=enemy_grid_chunk
                        )
                        h5f.create_dataset(
                            'inside_windows',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=inside_window_chunk
                        )
                        h5f.create_dataset(
                            'net_rewards',
                            shape=(num_samples,),
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4,
                            chunks=net_reward_chunk
                        )
                        
                        for idx in tqdm(range(num_samples), desc="Saving to HDF5", position=2):
                            sample = dataset[idx]
                            h5f['images'][idx] = sample['image'].cpu().numpy()
                            h5f['inputs'][idx] = sample['input'].cpu().numpy()
                            h5f['player_healths'][idx] = sample['player_health'].item()
                            h5f['enemy_healths'][idx] = sample['enemy_health'].item()
                            h5f['player_grids'][idx] = sample['player_grid'].cpu().numpy()
                            h5f['enemy_grids'][idx] = sample['enemy_grid'].cpu().numpy()
                            h5f['inside_windows'][idx] = sample['inside_window'].item()
                            h5f['net_rewards'][idx] = sample['net_reward'].item()
                        
                        # Save min and max as attributes
                        h5f.attrs['net_reward_min'] = float(dataset.net_reward_min)
                        h5f.attrs['net_reward_max'] = float(dataset.net_reward_max)
                    # Reload dataset from HDF5 using the preloaded dataset
                    dataset = PreloadedH5GameDataset(
                        h5_path=h5_path, 
                        device=device, 
                        batch_size=1000,  # Adjust based on your system's memory
                        prefetch_batches=4,  # Adjust based on desired prefetching
                        image_memory=image_memory
                    )
            else:
                dataset = GameDataset(
                    replay_dirs=subset_dirs,
                    image_memory=image_memory,
                    preprocess=preprocess,
                    load_into_memory=load_into_memory,
                    raw_dir=raw_dir,
                    is_raw=is_raw,
                    load_data_into_gpu=load_data_into_gpu,  # Pass the flag
                    device=device  # Pass the device
                )
            
            # Check if dataset has samples
            if len(dataset) == 0:
                print(f"Warning: No samples found in subset {subset_idx + 1}. Skipping.")
                subset_progress.update(1)
                subset_progress.close()
                continue
            
            net_reward_min = dataset.net_reward_min
            net_reward_max = dataset.net_reward_max

            # Convert to torch tensors
            net_reward_min = torch.tensor(net_reward_min, dtype=torch.float32).to(device)
            net_reward_max = torch.tensor(net_reward_max, dtype=torch.float32).to(device)
            if net_reward_max == net_reward_min:
                print("Warning: net_reward_max == net_reward_min, setting sample_weights to 1")
                net_reward_max = net_reward_min + 1e-6
            
            # Initialize DataLoader
            if efficient_storage:
                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,  # Shuffle is handled by DataLoader
                    num_workers=0,  # No workers since data is already on GPU
                    pin_memory=False  # Not needed as data is already on GPU
                )
            else:
                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,  # Adjust based on your CPU cores
                    pin_memory=True  # Speeds up data transfer to GPU
                )


            
            # Check if DataLoader has samples
            if len(train_loader) == 0:
                print(f"Warning: Train loader for subset {subset_idx + 1} has no samples. Skipping.")
                subset_progress.update(1)
                subset_progress.close()
                continue

            # Train for one epoch on this subset
            avg_epoch_loss = train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                full_epoch,
                net_reward_min,
                net_reward_max,
                scaler
            )

            #clear memory
            clear_memory()

            # Update subset progress
            subset_progress.update(1)
            subset_progress.close()

            # Checkpointing per subset or per epoch as desired
            if (full_epoch + 1) % checkpoint_freq == 0:
                save_at = os.path.join(checkpoint_dir, str(image_memory))
                os.makedirs(save_at, exist_ok=True)
                checkpoint_path = os.path.join(
                    save_at, f"checkpoint_epoch_{full_epoch+1}.pt")
                torch.save({
                    'epoch': full_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

                # Optionally upload to wandb as an artifact
                # artifact = wandb.Artifact('model-checkpoint', type='checkpoint')
                # artifact.add_file(checkpoint_path)
                # wandb.log_artifact(artifact)
                # print(f"Uploaded checkpoint to wandb: {checkpoint_path}")

                # Remove old checkpoints
                existing_checkpoints = sorted(
                    glob.glob(os.path.join(save_at, 'checkpoint_epoch_*.pt')),
                    key=lambda x: int(x.split('_')[-1].split('.')[0])
                )
                if len(existing_checkpoints) > max_checkpoints:
                    oldest_checkpoint = existing_checkpoints[0]
                    os.remove(oldest_checkpoint)
                    print(f"Removed old checkpoint: {oldest_checkpoint}")

            # Update the overall progress bar
            overall_progress.update(1)

            # Cleanup
            del train_loader
            del dataset
            clear_memory()

    # After the training loop
    overall_progress.close()
    print("Training completed successfully.")

    # Finish the wandb run
    wandb.finish()

# Ensure that the main function is called when the script is run directly
if __name__ == '__main__':
    main()
