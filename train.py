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

class GameDataset(Dataset):
    def __init__(self, cache_dir, image_memory=1, dir_indices=None,
                 preprocess=True, load_into_memory=False):
        self.cache_dir = cache_dir
        self.image_memory = image_memory
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        self.preprocess = preprocess
        self.load_into_memory = load_into_memory

        all_replay_dirs = [os.path.join(cache_dir, d) for d in os.listdir(
            cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
        all_replay_dirs.sort()

        # Use only the specified subset of directories
        if dir_indices is not None:
            replay_dirs = [all_replay_dirs[i] for i in dir_indices
                           if i < len(all_replay_dirs)]
        else:
            replay_dirs = all_replay_dirs

        # Top-level progress bar for replay directories
        outer_pbar = tqdm(replay_dirs, desc="Initializing Dataset",
                          unit="dir", position=0, leave=True)

        for replay_dir in outer_pbar:
            # Get sorted list of .pt files
            pt_files = sorted(glob.glob(os.path.join(replay_dir, '*.pt')))
            num_samples = len(pt_files)

            # Display progress for each .pt file within the directory
            if self.preprocess or self.load_into_memory:
                pt_progress = tqdm(range(num_samples),
                                   desc=f"Processing {os.path.basename(replay_dir)}",
                                   leave=False, unit="file", position=1)

            # Preprocess if flag is True
            if self.preprocess:
                for idx in pt_progress:
                    if idx - image_memory + 1 >= 0:
                        sample_pt_files = pt_files[idx - image_memory + 1: idx + 1]
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
                for idx in range(image_memory - 1, num_samples):
                    sample_info = {'pt_files': pt_files[idx - image_memory + 1:
                                                        idx + 1]}
                    self.sample_paths.append(sample_info)
                    # Load into memory if specified
                    if self.load_into_memory:
                        loaded_samples = [torch.load(file, map_location='cpu')
                                          for file in sample_info['pt_files']]
                        net_reward = loaded_samples[-1]['net_reward']
                        self.net_reward_min = min(self.net_reward_min,
                                                  net_reward)
                        self.net_reward_max = max(self.net_reward_max,
                                                  net_reward)
                        self.data_in_memory.append((loaded_samples,
                                                    net_reward))

        outer_pbar.close()
        print(f"Total samples: {len(self.sample_paths)}")
        print(f"net_reward_min: {self.net_reward_min}, "
              f"net_reward_max: {self.net_reward_max}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if self.load_into_memory:
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
                sample = torch.load(pt_file)
                image_tensors.append(sample['image'])
                input_tensors.append(sample['input'])
            net_reward = sample_info.get('net_reward', 0.0)

        # Stack images along the channel dimension
        if self.image_memory == 1:
            image_tensor = image_tensors[0]
            input_tensor = input_tensors[0]
        else:
            # Concatenate images along the channel dimension
            image_tensor = torch.cat(image_tensors, dim=0)
            input_tensor = input_tensors[-1]

        net_reward_tensor = torch.tensor(net_reward, dtype=torch.float32)

        return image_tensor, input_tensor, net_reward_tensor

class GameInputPredictor(nn.Module):
    def __init__(self, image_memory=1):
        super(GameInputPredictor, self).__init__()
        in_channels = 3 * image_memory
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self._to_linear = None
        self._get_conv_output_shape((in_channels, 160, 240))
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )

    def _get_conv_output_shape(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.conv_layers(input)
        self._to_linear = int(np.prod(output.shape[1:]))
        print(f"Convolution output size: {output.shape}")
        print(f"Flattened size: {self._to_linear}")

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self._to_linear)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device,
                num_epochs, checkpoint_dir, checkpoint_freq,
                max_checkpoints, start_epoch=0, net_reward_min=0.0,
                net_reward_max=1.0, overall_progress_position=0):
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
                              leave=True, position=overall_progress_position+1)

        for batch_idx, (images, inputs, net_rewards) in enumerate(
                epoch_progress):
            images = images.to(device)
            inputs = inputs.to(device).clamp(0, 1)
            net_rewards = net_rewards.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)

                # Compute loss per sample
                loss = criterion(outputs, inputs)

                loss_per_sample = loss.mean(dim=1)

                sample_weights = (net_rewards - net_reward_min) / (
                    net_reward_max - net_reward_min + 1e-6)

                weighted_loss = loss_per_sample * sample_weights

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

        epoch_progress.close()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss}")

        # Save checkpoint if it's the right frequency
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            # Remove old checkpoints
            existing_checkpoints = sorted(
                glob.glob(os.path.join(checkpoint_dir,
                                       'checkpoint_epoch_*.pt')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            if len(existing_checkpoints) > max_checkpoints:
                oldest_checkpoint = existing_checkpoints[0]
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")

def main():
    cache_dir = 'training_cache'
    checkpoint_dir = 'checkpoints'
    image_memory = 1
    batch_size = 64
    learning_rate = 1e-4
    max_checkpoints = 5
    max_dirs = 10000  # Adjust based on your dataset
    subset_size = 10
    num_epochs_per_subset = 25
    checkpoint_freq = 5
    preprocess = False
    load_into_memory = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, criterion, optimizer outside the loop
    model = GameInputPredictor(image_memory=image_memory).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for existing checkpoints
    existing_checkpoints = sorted(glob.glob(os.path.join(
        checkpoint_dir, 'checkpoint_epoch_*.pt')))
    start_epoch = 0
    if existing_checkpoints:
        latest_checkpoint = existing_checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Calculate total iterations needed to cover all directories
    total_subsets = (max_dirs + subset_size - 1) // subset_size  # Ceiling div
    total_epochs = total_subsets * num_epochs_per_subset

    # Keep track of total epochs trained
    total_epochs_trained = start_epoch

    # Create progress bar for overall training progress
    overall_progress = tqdm(total=total_epochs,
                            desc="Overall Training Progress",
                            unit="epoch", position=0, leave=True)

    for subset_idx in range(total_subsets):
        start_index = subset_idx * subset_size
        end_index = min(start_index + subset_size, max_dirs)
        dir_indices = list(range(start_index, end_index))

        # Initialize dataset with the subset of directories
        dataset = GameDataset(
            cache_dir,
            image_memory=image_memory,
            dir_indices=dir_indices,
            preprocess=preprocess,
            load_into_memory=load_into_memory
        )
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

        # Calculate the epochs to train on this subset
        num_epochs = total_epochs_trained + num_epochs_per_subset

        train_model(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            max_checkpoints=max_checkpoints,
            start_epoch=total_epochs_trained,
            net_reward_min=net_reward_min,
            net_reward_max=net_reward_max,
            overall_progress_position=overall_progress.pos
        )

        # Update total epochs trained
        total_epochs_trained += num_epochs_per_subset

        # Update the overall progress
        overall_progress.update(num_epochs_per_subset)

        # After training on current subset, free the dataset and loader
        del dataset
        del train_loader
        torch.cuda.empty_cache()
        gc.collect()

    overall_progress.close()

if __name__ == '__main__':
    main()
