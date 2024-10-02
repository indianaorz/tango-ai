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

import torch
import torch.nn as nn

import traceback

# training_utils.py
dataset = None
dataloader = None

def clear_dataset_and_loader():
    global dataset, dataloader
    dataset = None
    dataloader = None

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
    
    global dataset, dataloader
    
    model.train()  # Ensure the model is in training mode

    # Initialize the updated Dataset
    if dataset is None:
        dataset = HDF5Dataset(directory=training_data_dir, model_type=model_type)
    if dataloader is None:
        if model_type == "Battle_Model":
            # Initialize DataLoader with the updated Dataset
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,  # Now effective with the standard Dataset
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True  # Ensures consistent batch sizes
            )
        elif model_type == "Planning_Model":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=custom_collate_fn  # Use the custom collate function
            )



    # Initialize variables to track loss
    total_loss = 0.0
    batch_count = 0

    # Calculate total number of batches for tqdm
    total_possible_batches = math.ceil(len(dataset) / batch_size)
    length = min(total_possible_batches, max_batches)

    # Initialize tqdm progress bar with total number of batches
    progress_bar = tqdm(dataloader, desc=f"Final Training Epoch [{model_type}]", unit="batch", total=length)

    # Define the loss function
    criterion_cross = nn.CrossEntropyLoss()
    criterion_chip = nn.CrossEntropyLoss()
    for batch in progress_bar:
        if batch_count >= max_batches:
            break

        try:
            # Skip invalid batches
            if batch is None or any(v is None for v in batch.values()):
                print(f"Skipping invalid batch {batch_count}")
                continue

            if model_type == "Battle_Model":
                # Unpack batch data
                frames = batch['frames'].to(device)  # Shape: (batch_size, 3, D, H, W)
                position = batch['position'].to(device) if 'position' in batch and batch['position'] is not None else None
                player_charge_seq = batch['player_charge_seq'].to(device) if 'player_charge_seq' in batch else None
                enemy_charge_seq = batch['enemy_charge_seq'].to(device) if 'enemy_charge_seq' in batch else None
                player_charge = batch['player_charge'].to(device) if 'player_charge' in batch else None
                enemy_charge = batch['enemy_charge'].to(device) if 'enemy_charge' in batch else None
                previous_inputs = batch['previous_inputs'].to(device) if 'previous_inputs' in batch else None
                health_memory = batch['health_memory'].to(device) if 'health_memory' in batch else None
                player_chip = batch['player_chip'].to(device) if 'player_chip' in batch else None
                enemy_chip = batch['enemy_chip'].to(device) if 'enemy_chip' in batch else None
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
                    health_memory=health_memory,
                    player_chip=player_chip,
                    enemy_chip=enemy_chip
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

                entropy_coefficient = 0.05
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

            elif model_type == "Planning_Model":
                # Unpack batch data
                inputs = batch['inputs']
                cross_target = batch['cross_target'].to(device)    # Shape: (batch_size, 6)
                target_list = batch['target_list'].to(device)      # Shape: (batch_size, 5, 12)
                reward = batch['reward'].to(device)                # Shape: (batch_size,)
                #assign 1 reward to everything
                # reward = torch.ones_like(reward)
                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], dict):
                        for subkey in inputs[key]:
                            inputs[key][subkey] = inputs[key][subkey].to(device)
                    else:
                        inputs[key] = inputs[key].to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass through the PlanningModel
                cross_logits, chip_logits_list = model(inputs)  # cross_logits: (batch_size, 6), chip_logits_list: list of 5 tensors each (batch_size, 12)

                # Compute cross loss
                cross_loss = criterion_cross(cross_logits, cross_target)  # Shape: ()

                # Compute chip losses
                chip_losses = 0
                for i, chip_logit in enumerate(chip_logits_list):
                    chip_target = target_list[:, i, :]  # Shape: (batch_size, 12)
                    chip_loss = criterion_chip(chip_logit, chip_target)  # Shape: ()
                    chip_losses += chip_loss

                # Total policy loss
                policy_loss = cross_loss + chip_losses  # Scalar

                # Optionally weight the loss with reward
                weighted_loss = policy_loss * reward.mean()  # Adjust based on your requirements

                # Backward pass
                weighted_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                # Update loss and batch count
                total_loss += weighted_loss.item()
                batch_count += 1

                # Update progress bar with current loss
                progress_bar.set_postfix({"Batch Loss": weighted_loss.item()})

        except Exception as e:
            print(f"Error during training on batch {batch_count}: {e}")
            #traceback
            print(traceback.format_exc())
            continue  # Skip this batch and continue

    if batch_count > 0:
        average_loss = total_loss / batch_count
    else:
        average_loss = 0.0
    print(f"{model_type} Final Training Epoch - Average Loss: {average_loss:.4f}")
    return average_loss

def save_models(training_planning_model, training_battle_model, optimizer_planning, optimizer_battle, get_checkpoint_dir, get_new_checkpoint_path, MAX_CHECKPOINTS=5, IMAGE_MEMORY=1, append=0):
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

# Add this custom collate function
# training_utils.py

# training_utils.py

def custom_collate_fn(batch):
    """
    Custom collate function to handle nested dictionaries in batch data.
    """
    # Initialize batched data structures
    batched_inputs = {}
    batched_cross_target = []
    batched_target_list = []
    batched_reward = []

    # Iterate over the batch and collect data
    for data_point in batch:
        inputs = data_point['inputs']
        cross_target = data_point['cross_target']
        target_list = data_point['target_list']
        reward = data_point['reward']

        # For inputs, collect and batch the tensors
        for key in inputs:
            if key not in batched_inputs:
                if isinstance(inputs[key], dict):
                    batched_inputs[key] = {}
                else:
                    batched_inputs[key] = []
            if isinstance(inputs[key], dict):
                for subkey in inputs[key]:
                    if subkey not in batched_inputs[key]:
                        batched_inputs[key][subkey] = []
                    batched_inputs[key][subkey].append(inputs[key][subkey])
            else:
                batched_inputs[key].append(inputs[key])

        batched_cross_target.append(cross_target)
        batched_target_list.append(target_list)
        batched_reward.append(reward)

    # Now, stack the tensors
    for key in batched_inputs:
        if isinstance(batched_inputs[key], dict):
            for subkey in batched_inputs[key]:
                batched_inputs[key][subkey] = torch.stack(batched_inputs[key][subkey], dim=0)
        else:
            batched_inputs[key] = torch.stack(batched_inputs[key], dim=0)

    batched_cross_target = torch.stack(batched_cross_target, dim=0).float()  # Shape: (batch_size, 6)
    batched_target_list = torch.stack(batched_target_list, dim=0).float()    # Shape: (batch_size, 5, 12)
    batched_reward = torch.tensor(batched_reward, dtype=torch.float32)      # Shape: (batch_size,)

    # Return the batched data
    return {
        'inputs': batched_inputs,
        'cross_target': batched_cross_target,
        'target_list': batched_target_list,
        'reward': batched_reward
    }

# training_utils.py

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

def unflatten_dict(d, sep='_'):
    """
    Unflattens a flattened dictionary back into a nested dictionary.

    Args:
        d (dict): The flattened dictionary.
        sep (str): Separator used in keys.

    Returns:
        dict: A nested dictionary.
    """
    result_dict = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        d_ref = result_dict
        for key in keys[:-1]:
            if key not in d_ref:
                d_ref[key] = {}
            d_ref = d_ref[key]
        d_ref[keys[-1]] = value
    return result_dict
import os
import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, directory, model_type='Battle_Model', device='cuda'):
        """
        Initializes the dataset by listing all relevant HDF5 files and 
        creating an index mapping for quick access.

        Args:
            directory (str): Directory containing HDF5 files.
            model_type (str): Type of model ('Battle_Model' or 'Planning_Model').
        """
        self.directory = directory
        self.model_type = model_type
        self.device = device
        self.files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if file.endswith('.h5') and file.startswith(model_type)
        ]
        if not self.files:
            raise FileNotFoundError(
                f"No HDF5 files found for model type '{model_type}' in directory '{directory}'."
            )

        # Precompute the total number of samples by creating an index mapping
        self.index_mapping = []
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    if self.model_type == 'Battle_Model':
                        # For Battle_Model, use 'actions' dataset to determine number of samples
                        if 'actions' not in hf:
                            raise KeyError(f"'actions' dataset not found in {file_path} for Battle_Model.")
                        num_samples = hf['actions'].shape[0]
                    elif self.model_type == 'Planning_Model':
                        # For Planning_Model, use 'cross_target' to determine number of samples
                        if 'cross_target' not in hf:
                            raise KeyError(f"'cross_target' dataset not found in {file_path} for Planning_Model.")
                        if 'target_list' not in hf:
                            raise KeyError(f"'target_list' dataset not found in {file_path} for Planning_Model.")
                        num_samples = hf['cross_target'].shape[0]
                    else:
                        raise ValueError(f"Unsupported model_type: {self.model_type}")

                    for i in range(num_samples):
                        self.index_mapping.append((file_path, i))
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

        self.total_samples = len(self.index_mapping)
        if self.total_samples == 0:
            raise ValueError(
                f"No samples found in the HDF5 files for model type '{model_type}'."
            )
            
         # Preload all data into GPU memory
        # self.data = []
        # print(f"Preloading {self.total_samples} samples to GPU...")
        # for idx, (file_path, sample_idx) in enumerate(tqdm(self.index_mapping, desc="Loading Data")):
        #     try:
        #         with h5py.File(file_path, 'r') as hf:
        #             if self.model_type == "Battle_Model":
        #                 data_point = self._get_battle_model_data(hf, sample_idx)
        #             elif self.model_type == "Planning_Model":
        #                 data_point = self._get_planning_model_data(hf, sample_idx)
                    
        #             # Move tensors to GPU
        #             for key in data_point:
        #                 if isinstance(data_point[key], dict):
        #                     for subkey in data_point[key]:
        #                         data_point[key][subkey] = data_point[key][subkey].to(self.device, non_blocking=True)
        #                 elif isinstance(data_point[key], torch.Tensor):
        #                     data_point[key] = data_point[key].to(self.device, non_blocking=True)
        #             self.data.append(data_point)
        #     except Exception as e:
        #         print(f"Failed to load sample {sample_idx} from {file_path}: {e}")
        #         self.data.append(None)  # Placeholder for failed samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single data point based on the index.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: A dictionary containing data tensors.
        """
        file_path, sample_idx = self.index_mapping[idx]
        try:
            with h5py.File(file_path, 'r') as hf:
                if self.model_type == "Battle_Model":
                    return self._get_battle_model_data(hf, sample_idx)
                elif self.model_type == "Planning_Model":
                    return self._get_planning_model_data(hf, sample_idx)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")
        except Exception as e:
            print(f"Failed to read sample {sample_idx} from {file_path}: {e}")
            # Depending on your use case, you can either raise the exception
            # or return a dummy data point. Here, we'll raise the exception.
            raise e

    def _get_battle_model_data(self, hf, sample_idx):
        """
        Extracts data for the Battle_Model.

        Args:
            hf (h5py.File): Opened HDF5 file.
            sample_idx (int): Index of the sample within the file.

        Returns:
            dict: A dictionary containing Battle_Model data tensors.
        """
        data_point = {}
        # Define all the keys you want to extract for Battle_Model
        battle_keys = [
            'actions', 'rewards', 'frames', 'position', 'player_charge_seq',
            'enemy_charge_seq', 'player_charge', 'enemy_charge', 'previous_inputs',
            'health_memory', 'player_chip', 'enemy_chip'
        ]

        for key in battle_keys:
            if key in hf:
                # Handle different data types if necessary
                if key in ['frames', 'previous_inputs', 'position', 'player_charge_seq', 'enemy_charge_seq', 'player_charge', 'enemy_charge', 'health_memory', 'player_chip', 'enemy_chip']:
                    # Assuming these are numerical tensors
                    data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
                elif key == 'actions':
                    # If 'actions' are categorical or require a different dtype
                    data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.long)
                elif key == 'rewards':
                    data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
                else:
                    # Default handling
                    data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
            else:
                raise KeyError(f"Missing required key '{key}' in Battle_Model data.")

        return data_point


    def _get_planning_model_data(self, hf, sample_idx):
        """
        Extracts data for the Planning_Model.

        Args:
            hf (h5py.File): Opened HDF5 file.
            sample_idx (int): Index of the sample within the file.

        Returns:
            dict: A dictionary containing Planning_Model data tensors.
        """
        # Initialize nested inputs
        inputs = {
            'player_folder': {
                'chips_onehot': torch.tensor(
                    hf['inputs_player_folder_chips_onehot'][sample_idx], dtype=torch.float32
                ),
                'codes_onehot': torch.tensor(
                    hf['inputs_player_folder_codes_onehot'][sample_idx], dtype=torch.float32
                ),
                'flags': torch.tensor(
                    hf['inputs_player_folder_flags'][sample_idx], dtype=torch.float32
                )
            },
            'enemy_folder': {
                'chips_onehot': torch.tensor(
                    hf['inputs_enemy_folder_chips_onehot'][sample_idx], dtype=torch.float32
                ),
                'codes_onehot': torch.tensor(
                    hf['inputs_enemy_folder_codes_onehot'][sample_idx], dtype=torch.float32
                ),
                'flags': torch.tensor(
                    hf['inputs_enemy_folder_flags'][sample_idx], dtype=torch.float32
                )
            },
            'visible_chips': {
                'chips_onehot': torch.tensor(
                    hf['inputs_visible_chips_chips_onehot'][sample_idx], dtype=torch.float32
                ),
                'codes_onehot': torch.tensor(
                    hf['inputs_visible_chips_codes_onehot'][sample_idx], dtype=torch.float32
                )
            },
            'health': torch.tensor(hf['health'][sample_idx], dtype=torch.float32),
            'current_crosses': torch.tensor(hf['current_crosses'][sample_idx], dtype=torch.float32),
            'available_crosses': torch.tensor(hf['available_crosses'][sample_idx], dtype=torch.float32),
            'beast_flags': torch.tensor(hf['beast_flags'][sample_idx], dtype=torch.float32)
        }

        # Extract and process targets
        cross_target = int(hf['cross_target'][sample_idx])
        target_list = hf['target_list'][sample_idx].tolist()

        # Convert cross_target into one-hot encoded tensor (6 possible values)
        cross_target_onehot = torch.tensor(
            [1.0 if i == cross_target else 0.0 for i in range(6)], dtype=torch.float32
        )

        # Convert target_list into a 2D one-hot encoded tensor (5 targets, 12 possible values each)
        target_list_onehot = torch.tensor(
            [
                [1.0 if i == target else 0.0 for i in range(12)]
                for target in target_list
            ],
            dtype=torch.float32
        )

        # Extract reward
        reward = float(hf['reward'][sample_idx])

        # Structure the data_point
        data_point = {
            'inputs': inputs,
            'cross_target': cross_target_onehot,
            'target_list': target_list_onehot,
            'reward': reward
        }

        return data_point

# cross_target: 0
# target_list: [5, 7, 1, 2, 0]
# lh cross_target: tensor([1., 0., 0., 0., 0., 0.])
# lh target_list: 
# tensor([
    #     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
