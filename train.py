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


class GameDataset(Dataset):
    def __init__(self, replay_dirs, image_memory=1,
                 preprocess=True, load_into_memory=False,
                 raw_dir=None, is_raw=False, 
                 load_data_into_gpu=False, device='cpu'):
        self.image_memory = image_memory
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        self.preprocess = preprocess
        self.load_into_memory = load_into_memory
        self.load_data_into_gpu = load_data_into_gpu
        self.device = device  # Device to load data onto if flag is True

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
                    # Generate list of indices corresponding to pt_files
                    indices_list = list(range(len(pt_files)))
                    # Use the common get_exponential_sample function from utils.py
                    sampled_indices = get_exponential_sample(indices_list, idx, self.image_memory)
                    if not sampled_indices:
                        continue  # Skip if insufficient frames

                    # Map sampled indices to file paths
                    sample_pt_files = [pt_files[i] for i in sampled_indices]

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
                            
                            if self.load_data_into_gpu:
                                # Move data to GPU
                                loaded_samples = [
                                    {
                                        'image': sample['image'].to(self.device),
                                        'input': sample['input'].to(self.device)
                                    }
                                    for sample in loaded_samples
                                ]

                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                    except Exception as e:
                        print(f"Error loading {sample_pt_files[-1]}: {e}")
                pt_progress.close()
            else:
                for idx in range(self.image_memory - 1, num_samples):
                    # Generate list of indices corresponding to pt_files
                    indices_list = list(range(len(pt_files)))
                    # Use the common get_exponential_sample function from utils.py
                    sampled_indices = get_exponential_sample(indices_list, idx, self.image_memory)
                    if not sampled_indices:
                        continue  # Skip if insufficient frames

                    # Map sampled indices to file paths
                    sample_pt_files = [pt_files[i] for i in sampled_indices]

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
                            
                            if self.load_data_into_gpu:
                                # Move data to GPU
                                loaded_samples = [
                                    {
                                        'image': sample['image'].to(self.device),
                                        'input': sample['input'].to(self.device)
                                    }
                                    for sample in loaded_samples
                                ]

                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                        except Exception as e:
                            print(f"Error loading files in subset: {e}")

        outer_pbar.close()
        print(f"Total samples: {len(self.sample_paths)}")
        print(f"net_reward_min: {self.net_reward_min}, "
              f"net_reward_max: {self.net_reward_max}")


    def clear_memory(self):
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        gc.collect()

    def __len__(self):
        return len(self.sample_paths)

    def __del__(self):
        print("GameDataset __del__ called")
        self.clear_memory()


    def __getitem__(self, idx):
        try:
            if self.load_into_memory and idx < len(self.data_in_memory):
                # Access data directly from memory (already on GPU if flag is True)
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
                        sample = torch.load(pt_file, map_location='cpu')  # Load to CPU first
                        if self.load_data_into_gpu:
                            # Move to GPU
                            sample['image'] = sample['image'].to(self.device)
                            sample['input'] = sample['input'].to(self.device)
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

            input_tensor = input_tensors[-1]
            net_reward_tensor = torch.tensor(net_reward, dtype=torch.float32)

            return image_tensor, input_tensor, net_reward_tensor

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise


class H5GameDataset(Dataset):
    def __init__(self, h5_path, device='cpu'):
        super(H5GameDataset, self).__init__()
        self.h5_path = h5_path
        self.device = device
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.images = self.h5_file['images']
        self.inputs = self.h5_file['inputs']
        self.net_rewards = self.h5_file['net_rewards']
        
        # Read net_reward_min and net_reward_max from file attributes
        self.net_reward_min = self.h5_file.attrs['net_reward_min']
        self.net_reward_max = self.h5_file.attrs['net_reward_max']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Directly load the image without permutation
        image = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0  # Shape: (3, D, 160, 240)
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        net_reward = torch.tensor(self.net_rewards[idx], dtype=torch.float32)
        
        # Move to device if necessary
        if self.device != 'cpu':
            image = image.to(self.device)
            input_tensor = input_tensor.to(self.device)
            net_reward = net_reward.to(self.device)
        
        return image, input_tensor, net_reward

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

# Add this class to your train.py
# Add this class below the H5GameDataset class in your train.py
import threading
import queue
import threading
import queue
import gc
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class PreloadedH5GameDataset(H5GameDataset):
    def __init__(self, h5_path, device='cpu', batch_size=1000, prefetch_batches=2, image_memory=1):
        """
        Initializes the PreloadedH5GameDataset.

        Args:
            h5_path (str): Path to the HDF5 file.
            device (torch.device): Device to load data onto ('cpu' or 'cuda').
            batch_size (int): Number of samples per batch during preloading.
            prefetch_batches (int): Number of batches to prefetch.
            image_memory (int): Depth dimension for image tensors.
        """
        super(PreloadedH5GameDataset, self).__init__(h5_path, device)
        self.image_memory = image_memory  # Define the missing attribute
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.preload_data()

    def preload_data(self):
        print("Preloading data into GPU...")
        num_samples = len(self)

        # Preallocate tensors on GPU
        try:
            images = torch.empty((num_samples, 3, self.image_memory, 160, 240), dtype=torch.float32, device=self.device)
            inputs = torch.empty((num_samples, 16), dtype=torch.float32, device=self.device)
            net_rewards = torch.empty(num_samples, dtype=torch.float32, device=self.device)
        except RuntimeError as e:
            print(f"Error allocating GPU memory: {e}")
            raise

        # Create a thread-safe queue
        prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)

        def loader():
            try:
                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)

                    # Load data from HDF5
                    images_np = self.h5_file['images'][start_idx:end_idx]
                    inputs_np = self.h5_file['inputs'][start_idx:end_idx]
                    net_rewards_np = self.h5_file['net_rewards'][start_idx:end_idx]

                    # Convert to torch tensors
                    images_cpu = torch.from_numpy(images_np).float() / 255.0  # Normalize
                    inputs_cpu = torch.from_numpy(inputs_np).float()
                    net_rewards_cpu = torch.from_numpy(net_rewards_np).float()

                    # Put the batch into the queue
                    prefetch_queue.put((start_idx, images_cpu, inputs_cpu, net_rewards_cpu))
            finally:
                # Signal that loading is done
                prefetch_queue.put(None)

        # Start the loader thread
        loader_thread = threading.Thread(target=loader, daemon=True)
        loader_thread.start()

        # Process the batches
        while True:
            batch = prefetch_queue.get()
            if batch is None:
                break  # No more data

            start_idx, images_cpu, inputs_cpu, net_rewards_cpu = batch
            batch_size_actual = images_cpu.size(0)

            # Asynchronously copy to GPU
            try:
                images[start_idx:start_idx+batch_size_actual].copy_(images_cpu, non_blocking=True)
                inputs[start_idx:start_idx+batch_size_actual].copy_(inputs_cpu, non_blocking=True)
                net_rewards[start_idx:start_idx+batch_size_actual].copy_(net_rewards_cpu, non_blocking=True)
            except RuntimeError as e:
                print(f"Error copying data to GPU: {e}")
                raise

        # Assign preloaded tensors to class attributes
        self.images = images
        self.inputs = inputs
        self.net_rewards = net_rewards

        # Delete the HDF5 file reference to free up CPU memory
        del self.h5_file
        gc.collect()
        print("Data preloading completed.")

    def __getitem__(self, idx):
        # Directly return the preloaded data
        image = self.images[idx]
        input_tensor = self.inputs[idx]
        net_reward = self.net_rewards[idx]
        return image, input_tensor, net_reward



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
               epoch, net_reward_min, net_reward_max, scaler):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    # Create the progress bar for the epoch
    epoch_progress = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}",
                          leave=False, position=2, ncols=100)

    for batch_idx, (images, inputs, net_rewards) in enumerate(epoch_progress):
        images = images.to(device)
        inputs = inputs.to(device).clamp(0, 1)
        net_rewards = net_rewards.to(device)

        net_rewards = (net_rewards - net_rewards.mean()) / (net_rewards.std() + 1e-6)
        net_rewards = torch.clamp(net_rewards, min=-1.0, max=1.0)

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
            'Train/Batch_Loss': loss.item(),
            'Train/Avg_Batch_Loss': avg_loss,
            'Train/Epoch': epoch + 1,
            'Train/Batch': batch_idx + 1,
        })

        # Clear variables at the end of the loop
        images = None
        inputs = None
        net_rewards = None
        loss = None
        outputs = None
        torch.cuda.empty_cache()
        gc.collect()

    epoch_progress.close()

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss}")

    # Optionally, log epoch-level metrics
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
    subset_size = 8  # Number of directories per subset
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

    if subsets_per_full_epoch == 1:
        # Handle the single subset case
        subset_dirs = all_replay_dirs
        print(f"\nTraining on single subset: {', '.join([os.path.basename(d) for d in subset_dirs])}")
        
        # Create HDF5 path
        if efficient_storage:
            h5_path = os.path.join(cache_dir, f'subset_0.h5')
            if os.path.exists(h5_path):
                print(f"Loading dataset from HDF5 file: {h5_path}")
                dataset = PreloadedH5GameDataset(
                    h5_path=h5_path, 
                    device=device, 
                    batch_size=1000,  # Adjust based on your system's memory
                    prefetch_batches=2  # Adjust based on desired prefetching
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
                    # Define chunk sizes
                    image_chunk = (1, 3, image_memory, 160, 240)  # One image per chunk
                    input_chunk = (1, 16)  # One input per chunk
                    reward_chunk = (1,)  # One reward per chunk

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
                        'net_rewards',
                        shape=(num_samples,),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        chunks=reward_chunk
                    )
                    
                    for idx in tqdm(range(num_samples), desc="Saving to HDF5", position=1):
                        image, input_tensor, net_reward = dataset[idx]
                        h5f['images'][idx] = image.cpu().numpy()
                        h5f['inputs'][idx] = input_tensor.cpu().numpy()
                        h5f['net_rewards'][idx] = net_reward.cpu().numpy()
                    
                    # Save min and max as attributes
                    h5f.attrs['net_reward_min'] = float(dataset.net_reward_min)
                    h5f.attrs['net_reward_max'] = float(dataset.net_reward_max)
                # Reload dataset from HDF5 using the preloaded dataset
                dataset = PreloadedH5GameDataset(
                    h5_path=h5_path, 
                    device=device, 
                    batch_size=1000,  # Adjust based on your system's memory
                    prefetch_batches=2  # Adjust based on desired prefetching
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
            print("Warning: No samples found. Exiting.")
            return
        
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
            print("Warning: Train loader has no samples. Exiting.")
            return

        # Loop over epochs
        for epoch in range(start_epoch, num_full_epochs):
            print(f"\nStarting epoch {epoch + 1}/{num_full_epochs}")

            # Train for one epoch
            avg_epoch_loss = train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                net_reward_min,
                net_reward_max,
                scaler
            )

            # Checkpointing per epoch
            if (epoch + 1) % checkpoint_freq == 0:
                save_at = os.path.join(checkpoint_dir, str(image_memory))
                os.makedirs(save_at, exist_ok=True)
                checkpoint_path = os.path.join(
                    save_at, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
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
            torch.cuda.empty_cache()
            gc.collect()

    else:
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
                            prefetch_batches=4  # Adjust based on desired prefetching
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
                            # Define chunk sizes
                            image_chunk = (1, 3, image_memory, 160, 240)  # One image per chunk
                            input_chunk = (1, 16)  # One input per chunk
                            reward_chunk = (1,)  # One reward per chunk

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
                                'net_rewards',
                                shape=(num_samples,),
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4,
                                chunks=reward_chunk
                            )
                            
                            for idx in tqdm(range(num_samples), desc="Saving to HDF5", position=2):
                                image, input_tensor, net_reward = dataset[idx]
                                h5f['images'][idx] = image.cpu().numpy()
                                h5f['inputs'][idx] = input_tensor.cpu().numpy()
                                h5f['net_rewards'][idx] = net_reward.cpu().numpy()
                            
                            # Save min and max as attributes
                            h5f.attrs['net_reward_min'] = float(dataset.net_reward_min)
                            h5f.attrs['net_reward_max'] = float(dataset.net_reward_max)
                        # Reload dataset from HDF5 using the preloaded dataset
                        dataset = PreloadedH5GameDataset(
                            h5_path=h5_path, 
                            device=device, 
                            batch_size=1000,  # Adjust based on your system's memory
                            prefetch_batches=4  # Adjust based on desired prefetching
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

                # Update subset progress
                subset_progress.update(1)
                subset_progress.close()

                # Cleanup
                del train_loader
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            # After completing all subsets for the current epoch, checkpoint
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
            torch.cuda.empty_cache()
            gc.collect()

    # After the training loop
    overall_progress.close()
    print("Training completed successfully.")

    # Finish the wandb run
    wandb.finish()

# Ensure that the main function is called when the script is run directly
if __name__ == '__main__':
    main()
