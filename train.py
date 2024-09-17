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
from utils import get_exponential_sample, get_image_memory, get_checkpoint_path, get_exponental_amount  # Import the helper function
from cache_data import process_replay

import shutil
import wandb  # Import wandb

# Define constants
TEMP_DIR = '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO/temp'

# Create temp dir if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

class GameDataset(Dataset):
    def __init__(self, replay_dirs, image_memory=1,
                 preprocess=True, load_into_memory=False,
                 raw_dir=None, is_raw=False):
        self.image_memory = image_memory
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        self.preprocess = preprocess
        self.load_into_memory = load_into_memory

        # Clear tmp dir, remove all subdirectories
        for f in os.listdir(TEMP_DIR):
            subdir_path = os.path.join(TEMP_DIR, f)
            if os.path.isdir(subdir_path):
                shutil.rmtree(subdir_path)

        if is_raw:
            for replay_dir in replay_dirs:
                process_replay(replay_dir, output_dir=TEMP_DIR)
            replay_dirs = [os.path.join(TEMP_DIR, d) for d in os.listdir(TEMP_DIR) if os.path.isdir(os.path.join(TEMP_DIR, d))]

        if not replay_dirs:
            print("No directories provided to GameDataset.")
            return

        # Top-level progress bar for replay directories
        outer_pbar = tqdm(replay_dirs, desc="Initializing Dataset",
                          unit="dir", position=0, leave=True)

        for replay_dir in outer_pbar:
            # Get sorted list of .pt files
            pt_files = sorted(glob.glob(os.path.join(replay_dir, '*.pt')))
            num_samples = len(pt_files)

            # If the number of samples is > 18000, reduce the number of samples
            if num_samples > 18000:
                pt_files = pt_files[::2]
                num_samples = len(pt_files)

            # Display progress for each .pt file within the directory
            if self.preprocess or self.load_into_memory:
                pt_progress = tqdm(range(num_samples),
                                   desc=f"Processing {os.path.basename(replay_dir)}",
                                   leave=False, unit="file", position=1)

            # Preprocess if flag is True
            if self.preprocess:
                for idx in pt_progress:
                    # Calculate exponentially spaced indices
                    sample_pt_files = self.get_exponential_sample(pt_files, idx)
                    if not sample_pt_files:
                        continue  # Skip if insufficient frames

                    try:
                        # Load net_reward from the last file only
                        sample = torch.load(sample_pt_files[-1],
                                            map_location='cpu')
                        net_reward = sample['net_reward']
                        self.net_reward_min = min(self.net_reward_min,
                                                  net_reward)
                        self.net_reward_max = max(self.net_reward_max,
                                                  net_reward)
                        sample_info = {
                            'pt_files': sample_pt_files,
                            'net_reward': net_reward
                        }
                        self.sample_paths.append(sample_info)

                        # If loading into memory, load all samples now
                        if self.load_into_memory:
                            loaded_samples = [torch.load(file,
                                                      map_location='cpu')
                                              for file in sample_pt_files]
                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                    except Exception as e:
                        print(f"Error loading {sample_pt_files[-1]}: {e}")
                pt_progress.close()
            else:
                for idx in range(self.image_memory - 1, num_samples):
                    sample_pt_files = self.get_exponential_sample(pt_files, idx)
                    if not sample_pt_files:
                        continue  # Skip if insufficient frames

                    sample_info = {'pt_files': sample_pt_files}
                    self.sample_paths.append(sample_info)
                    # Load into memory if specified
                    if self.load_into_memory:
                        try:
                            loaded_samples = [torch.load(file, map_location='cpu')
                                              for file in sample_pt_files]
                            net_reward = loaded_samples[-1]['net_reward']
                            self.net_reward_min = min(self.net_reward_min,
                                                      net_reward)
                            self.net_reward_max = max(self.net_reward_max,
                                                      net_reward)
                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                        except Exception as e:
                            print(f"Error loading files in subset: {e}")

        outer_pbar.close()
        print(f"Total samples: {len(self.sample_paths)}")
        print(f"net_reward_min: {self.net_reward_min}, "
              f"net_reward_max: {self.net_reward_max}")

    def get_exponential_sample(self, pt_files, current_idx):
        """
        Returns a list of .pt file paths sampled exponentially from the current index.
        Ensures exactly `image_memory` frames by allowing duplicates when necessary.
        """
        indices = [current_idx]
        step = 1
        while len(indices) < self.image_memory and current_idx - step >= 0:
            indices.append(current_idx - step)
            step *= get_exponental_amount()

        # If not enough frames, pad with the earliest frame (index 0)
        while len(indices) < self.image_memory:
            indices.append(0)

        # If more frames than needed, truncate the list
        if len(indices) > self.image_memory:
            indices = indices[:self.image_memory]

        # Sort the indices to have chronological order (oldest to newest)
        indices_sorted = sorted(indices)

        # Fetch the corresponding file paths
        sample_pt_files = [pt_files[i] for i in indices_sorted]

        # Debugging statement
        # print(f"Selected indices for current_idx {current_idx}: {indices_sorted}")
        return sample_pt_files


    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        try:
            if self.load_into_memory and idx < len(self.data_in_memory):
                # Access data directly from memory
                loaded_samples, net_reward = self.data_in_memory[idx]
                image_tensors = [sample['image'] for sample in loaded_samples]
                input_tensors = [sample['input'] for sample in loaded_samples]
            else:
                # Load data from disk
                sample_info = self.sample_paths[idx]
                image_tensors = []
                input_tensors = []
                for pt_file in sample_info['pt_files']:
                    try:
                        sample = torch.load(pt_file)
                        image_tensors.append(sample['image'])
                        input_tensors.append(sample['input'])
                    except Exception as e:
                        print(f"Error loading file {pt_file}: {e}")
                        raise

                net_reward = sample_info.get('net_reward', 0.0)

            # Ensure the number of images matches image_memory
            if len(image_tensors) != self.image_memory:
                print(f"Expected {self.image_memory} images but got {len(image_tensors)} for index {idx}")
                raise ValueError(f"Image count mismatch for index {idx}")

            # Stack images along the temporal dimension
            if self.image_memory == 1:
                image_tensor = image_tensors[0].unsqueeze(1)  # Add depth dimension at dim=1
                input_tensor = input_tensors[0]
            else:
                # Stack images along the depth dimension
                image_tensor = torch.stack(image_tensors, dim=1)  # (channels, depth, height, width)

            # Optionally, print the shape for debugging
            # print(f"Image tensor shape: {image_tensor.shape}")

            input_tensor = input_tensors[-1]
            net_reward_tensor = torch.tensor(net_reward, dtype=torch.float32)

            return image_tensor, input_tensor, net_reward_tensor

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise


class GameInputPredictor(nn.Module):
    def __init__(self, image_memory=1):
        super(GameInputPredictor, self).__init__()
        self.image_memory = image_memory
        # Updated to use 3D convolutions to capture spatiotemporal features
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self._to_linear = None
        self._get_conv_output_shape((3, image_memory, 160, 240))  # Input shape based on (channels, depth, height, width)
        
        # LSTM layer to capture temporal dependencies in the flattened output
        self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=512, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 16),  # Assuming 16 possible actions
        )

    def _get_conv_output_shape(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.conv_layers(input)
        self._to_linear = int(np.prod(output.shape[1:]))
        print(f"Convolution output size: {output.shape}")
        print(f"Flattened size: {self._to_linear}")

    def forward(self, x):
        # Expected input shape: (batch_size, channels=3, depth=image_memory, height=160, width=240)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1, self._to_linear)  # Flatten while keeping batch and sequence dimensions intact
        x, _ = self.lstm(x)  # Process through LSTM to capture temporal relationships
        x = x[:, -1, :]  # Use the last output from the LSTM as it represents the processed sequence
        x = self.fc_layers(x)
        return x


def train_model(model, train_loader, optimizer, criterion, device,
                num_epochs, checkpoint_dir, checkpoint_freq,
                max_checkpoints, start_epoch=0, net_reward_min=0.0,
                net_reward_max=1.0, overall_progress_position=0, image_memory=1):
    scaler = GradScaler()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        # Create the progress bar for the epoch
        epoch_progress = tqdm(train_loader,
                              desc=f"Epoch {epoch+1}/{num_epochs}",
                              leave=False, position=overall_progress_position+1)

        for batch_idx, (images, inputs, net_rewards) in enumerate(epoch_progress):
            images = images.to(device)
            inputs = inputs.to(device).clamp(0, 1)
            net_rewards = net_rewards.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)

                # Compute loss per sample
                loss = criterion(outputs, inputs)
                loss_per_sample = loss.mean(dim=1)

                # Compute sample weights
                sample_weights = (net_rewards - net_reward_min) / (
                    net_reward_max - net_reward_min + 1e-6)

                # Multiply loss per sample by sample weights
                weighted_loss = loss_per_sample * sample_weights

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
                'Train/Loss': loss.item(),
                'Train/Avg_Loss': avg_loss,
                'Train/Epoch': epoch + 1,
                'Train/Batch': batch_idx + 1,
            })

        epoch_progress.close()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss}")
        
        # Optionally, log epoch-level metrics
        wandb.log({
            'Train/Epoch_Avg_Loss': avg_epoch_loss,
            'Train/Epoch': epoch + 1,
        })

        save_at = os.path.join(checkpoint_dir, str(image_memory))  # Corrected path join
        if not os.path.exists(save_at):
            os.makedirs(save_at)

        # Save checkpoint if it's the right frequency
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                save_at, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save checkpoint as a wandb artifact
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


TRAINING_CACHE_DIR = '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO/training_cache'

def main():
    # Initialize wandb
    wandb.init(
        project="sigma_5_mem_1_sub",  # Replace with your wandb project name
        name="experiment_name",       # (Optional) Name for this run
        config={
            "batch_size": 64,
            "learning_rate": 1e-4,
            "image_memory": get_image_memory(),
            "num_epochs": 100,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            # Add other hyperparameters you want to track
        }
    )
    
    config = wandb.config  # Access the config object

    cache_dir = TRAINING_CACHE_DIR
    image_memory = config.image_memory  # Use wandb config
    checkpoint_dir = '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO/checkpoints'
    raw_dir = '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO/training_data'
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    max_checkpoints = 5
    subset_size = 1  # Number of directories per subset
    num_epochs_per_subset = 1  # Number of epochs per subset
    checkpoint_freq = 50  # Save a checkpoint every 50 epochs
    num_full_epochs = config.num_epochs
    preprocess = True  # Set to False to skip preprocessing the dataset
    load_into_memory = True  # Set to True to load dataset into memory
    is_raw = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Calculate total epochs based on full passes
    total_epochs = num_full_epochs * subsets_per_full_epoch * num_epochs_per_subset

    # Keep track of total epochs trained
    total_epochs_trained = start_epoch

    # Create progress bar for overall training progress
    overall_progress = tqdm(total=total_epochs,
                            desc="Overall Training Progress",
                            unit="epoch", position=0, leave=True)

    for full_epoch in range(num_full_epochs):
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

            # Initialize dataset with the current subset of directories
            dataset = GameDataset(
                replay_dirs=subset_dirs,
                image_memory=image_memory,
                preprocess=preprocess,
                load_into_memory=load_into_memory,
                raw_dir=raw_dir,
                is_raw=is_raw
            )

            # Check if dataset has samples
            if len(dataset) == 0:
                print(f"Warning: No samples found in subset {subset_idx + 1}. Skipping.")
                continue

            net_reward_min = dataset.net_reward_min
            net_reward_max = dataset.net_reward_max

            # Convert to torch tensors
            net_reward_min = torch.tensor(net_reward_min,
                                          dtype=torch.float32).to(device)
            net_reward_max = torch.tensor(net_reward_max,
                                          dtype=torch.float32).to(device)
            if net_reward_max == net_reward_min:
                print("Warning: net_reward_max == net_reward_min, "
                      "setting sample_weights to 1")
                net_reward_max = net_reward_min + 1e-6

            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True)

            # Calculate the epoch range for this subset
            end_epoch = total_epochs_trained + num_epochs_per_subset

            # Train on the current subset
            train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_epochs=end_epoch,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=checkpoint_freq,
                max_checkpoints=max_checkpoints,
                start_epoch=total_epochs_trained,
                net_reward_min=net_reward_min,
                net_reward_max=net_reward_max,
                overall_progress_position=0,  # Position for progress bars
                image_memory=image_memory
            )

            # Update total epochs trained
            total_epochs_trained += num_epochs_per_subset

            # Update the overall progress bar
            overall_progress.update(num_epochs_per_subset)

            # Free memory
            del dataset
            del train_loader
            torch.cuda.empty_cache()
            gc.collect()

    overall_progress.close()
    print("Training completed successfully.")

    # Finish the wandb run
    wandb.finish()


# Ensure that the main function is called when the script is run directly
if __name__ == '__main__':
    main()
