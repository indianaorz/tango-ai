# training_utils.py
import os
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import math

# training_utils.py

def final_training_epoch(model, optimizer, training_data_dir, model_type='Battle_Model', batch_size=64, num_workers=4, device='cuda', max_batches=100):
    """
    Performs one epoch of training on a limited number of batches with a progress bar.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        training_data_dir (str): Directory containing the training HDF5 files.
        model_type (str): Type of model ('Battle_Model' or 'Planning_Model').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        device (str): Device to run the training on ('cuda' or 'cpu').
        max_batches (int): Maximum number of batches to process.
    """
    model.train()  # Ensure the model is in training mode

    # Initialize the updated Dataset
    dataset = HDF5Dataset(directory=training_data_dir, model_type=model_type)

    # Initialize DataLoader with the updated Dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Now effective with the standard Dataset
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensures consistent batch sizes
    )

    # Initialize variables to track loss
    total_loss = 0.0
    batch_count = 0

    # Calculate total number of batches for tqdm
    total_possible_batches = math.ceil(len(dataset) / batch_size)
    length = min(total_possible_batches, max_batches)

    # Initialize tqdm progress bar with total number of batches
    progress_bar = tqdm(dataloader, desc=f"Final Training Epoch [{model_type}]", unit="batch", total=length)

    for batch in progress_bar:
        if batch_count >= max_batches:
            break

        try:
            # Filter out any None data points
            if batch is None or any(v is None for v in batch.values()):
                print(f"Skipping invalid batch {batch_count}")
                continue

            # Unpack batch data
            frames = batch['frames'].to(device)  # Shape: (batch_size, 3, D, H, W)
            position = batch['position'].to(device) if 'position' in batch and batch['position'] is not None else None
            player_charge_seq = batch['player_charge_seq'].to(device) if 'player_charge_seq' in batch else None
            enemy_charge_seq = batch['enemy_charge_seq'].to(device) if 'enemy_charge_seq' in batch else None
            player_charge = batch['player_charge'].to(device) if 'player_charge' in batch else None
            enemy_charge = batch['enemy_charge'].to(device) if 'enemy_charge' in batch else None
            previous_inputs = batch['previous_inputs'].to(device) if 'previous_inputs' in batch else None
            health_memory = batch['health_memory'].to(device) if 'health_memory' in batch else None
            actions = batch['actions'].to(device)  # Shape: (batch_size, num_actions)
            rewards = batch['rewards'].to(device)  # Shape: (batch_size,)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                frames,
                position=position,
                player_charge=player_charge,
                enemy_charge=enemy_charge,
                player_charge_temporal=player_charge_seq,
                enemy_charge_temporal=enemy_charge_seq,
                previous_inputs=previous_inputs,
                health_memory=health_memory
            )  # Shape: (batch_size, num_actions)

              # Compute log probabilities directly
            probs = torch.sigmoid(outputs)
            epsilon = 1e-6  # For numerical stability
            probs = torch.clamp(probs, epsilon, 1 - epsilon)
            log_probs = actions * torch.log(probs) + (1 - actions) * torch.log(1 - probs)

            # Compute the policy loss with the correct sign
            policy_loss = - (log_probs.sum(dim=1) * rewards).mean()

            # Compute entropy
            entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).sum(dim=1)

            entropy_coefficient = 0.01
            total_loss = policy_loss - entropy_coefficient * entropy.mean()

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Update loss and batch count
            #total_loss += total_epoch_loss.item()
            batch_count += 1

            # Update progress bar with current loss
            progress_bar.set_postfix({"Batch Loss": total_loss.item()})
            # Optionally, remove the print statement to reduce console clutter
            # print(f"{model_type} Training - Loss: {total_epoch_loss.item():.4f}")

        except Exception as e:
            print(f"Error during training on batch {batch_count}: {e}")
            continue  # Skip this batch and continue

    if batch_count > 0:
        average_loss = total_loss / batch_count
    else:
        average_loss = 0.0
    print(f"{model_type} Final Training Epoch - Average Loss: {average_loss:.4f}")



def save_models(training_planning_model, training_battle_model, optimizer_planning, optimizer_battle, get_checkpoint_dir, get_new_checkpoint_path, MAX_CHECKPOINTS=5, IMAGE_MEMORY=1):
    """
    Saves the Training Planning and Training Battle models to their respective checkpoint directories.
    Utilizes unique checkpoint paths to prevent overwriting and maintains a maximum of MAX_CHECKPOINTS.
    """
    def manage_checkpoints(checkpoint_dir):
        """
        Ensures that only the latest MAX_CHECKPOINTS are retained in the checkpoint directory.
        Older checkpoints are deleted.
        """
        try:
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                reverse=True
            )
            for ckpt in checkpoints[MAX_CHECKPOINTS:]:
                os.remove(os.path.join(checkpoint_dir, ckpt))
                print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Failed to manage checkpoints in {checkpoint_dir}: {e}")

    # Save Training Planning Model
    if training_planning_model is not None:
        planning_checkpoint_path = get_new_checkpoint_path(model_type='planning', image_memory=IMAGE_MEMORY)
        torch.save({'model_state_dict': training_planning_model.state_dict()}, planning_checkpoint_path)
        print(f"Training Planning Model saved to {planning_checkpoint_path}")
        # Manage checkpoints
        planning_checkpoint_dir = get_checkpoint_dir(model_type='planning', image_memory=IMAGE_MEMORY)
        manage_checkpoints(planning_checkpoint_dir)
    else:
        print("Training Planning Model is not loaded. Skipping save.")

    # Save Training Battle Model
    if training_battle_model is not None:
        battle_checkpoint_path = get_new_checkpoint_path(model_type='battle', image_memory=IMAGE_MEMORY)
        torch.save({'model_state_dict': training_battle_model.state_dict()}, battle_checkpoint_path)
        print(f"Training Battle Model saved to {battle_checkpoint_path}")
        # Manage checkpoints
        battle_checkpoint_dir = get_checkpoint_dir(model_type='battle', image_memory=IMAGE_MEMORY)
        manage_checkpoints(battle_checkpoint_dir)
    else:
        print("Training Battle Model is not loaded. Skipping save.")


# training_utils.py

import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random

class HDF5Dataset(Dataset):
    def __init__(self, directory, model_type='Battle_Model'):
        """
        Initializes the dataset by listing all relevant HDF5 files.

        Args:
            directory (str): Directory containing HDF5 files.
            model_type (str): Type of model ('Battle_Model' or 'Planning_Model').
        """
        self.directory = directory
        self.model_type = model_type
        self.files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if file.endswith('.h5') and file.startswith(model_type)
        ]
        if not self.files:
            raise FileNotFoundError(f"No HDF5 files found for model type '{model_type}' in directory '{directory}'.")

        # Precompute the total number of samples
        self.index_mapping = []
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    num_samples = hf['actions'].shape[0]
                    for i in range(num_samples):
                        self.index_mapping.append((file_path, i))
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

        self.total_samples = len(self.index_mapping)
        if self.total_samples == 0:
            raise ValueError(f"No samples found in the HDF5 files for model type '{model_type}'.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: A dictionary containing data tensors.
        """
        file_path, sample_idx = self.index_mapping[idx]
        try:
            with h5py.File(file_path, 'r') as hf:
                data_point = {}
                for key in hf.keys():
                    if key == 'rewards':
                        data_point['rewards'] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
                    else:
                        data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
            return data_point
        except Exception as e:
            print(f"Failed to read sample {sample_idx} from {file_path}: {e}")
            # Return a dummy data point or handle as needed
            return None
