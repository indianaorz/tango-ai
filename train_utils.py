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
                total_loss_batch = policy_loss - entropy_coefficient * entropy.mean()

                # Backward pass
                total_loss_batch.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                # Update loss and batch count
                total_loss += total_loss_batch.item()
                batch_count += 1

                # Update progress bar with current loss
                progress_bar.set_postfix({"Batch Loss": total_loss_batch.item()})

            elif model_type == "Planning_Model":
                # Unpack batch data
                # Since Planning_Model batch size is 1, handle accordingly
                # If batch_size >1 in future, this can be vectorized
                inputs = batch['inputs']
                cross_target = batch['cross_target']
                target_list = batch['target_list']
                reward = batch['reward']

                # Move all tensors to device
                for key in inputs:
                    if isinstance(inputs[key], dict):
                        for subkey in inputs[key]:
                            inputs[key][subkey] = inputs[key][subkey].to(device)
                    else:
                        inputs[key] = inputs[key].to(device)

                cross_target_tensor = torch.tensor([cross_target], dtype=torch.long, device=device)  # Shape: (1,)
                target_list_tensor = torch.tensor([target_list], dtype=torch.long, device=device)    # Shape: (1, 5)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)          # Shape: (1,)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass through the PlanningModel
                cross_logits, chip_logits_list = model(inputs)  # cross_logits: (1, 6), chip_logits_list: list of 5 tensors each (1, 10)

                # Compute log probabilities for cross_selection
                cross_log_probs = F.log_softmax(cross_logits, dim=-1)  # Shape: (1, 6)
                cross_selected_log_probs = cross_log_probs[torch.arange(1), cross_target_tensor]  # Shape: (1,)

                # Compute log probabilities for chip_selections
                chip_log_probs = [F.log_softmax(chip_logits, dim=-1) for chip_logits in chip_logits_list]  # List of 5 tensors each (1, 10)
                chip_selected_log_probs = torch.stack([
                    chip_log_probs[i][torch.arange(1), target_list_tensor[:, i]]
                    for i in range(5)
                ], dim=1)  # Shape: (1, 5)

                # Sum log_probs across all actions
                total_log_probs = cross_selected_log_probs + chip_selected_log_probs.sum(dim=1)  # Shape: (1,)

                # Compute policy loss
                policy_loss = - (total_log_probs * reward_tensor).mean()

                # Compute entropy for regularization
                cross_entropy = -(F.softmax(cross_logits, dim=-1) * F.log_softmax(cross_logits, dim=-1)).sum(dim=1)  # Shape: (1,)
                chip_entropy = [-(F.softmax(chip_logits, dim=-1) * F.log_softmax(chip_logits, dim=-1)).sum(dim=-1) for chip_logits in chip_logits_list]  # List of 5 tensors each (1,)
                chip_entropy = torch.stack(chip_entropy, dim=1).sum(dim=1)  # Shape: (1,)
                total_entropy = cross_entropy + chip_entropy  # Shape: (1,)

                entropy_coefficient = 0.05
                total_loss_batch = policy_loss - entropy_coefficient * total_entropy.mean()

                # Backward pass
                total_loss_batch.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                # Update loss and batch count
                total_loss += total_loss_batch.item()
                batch_count += 1

                # Update progress bar with current loss
                progress_bar.set_postfix({"Batch Loss": total_loss_batch.item()})

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
                    if self.model_type == 'Battle_Model':
                        # For Battle_Model, assume 'input' dataset exists
                        num_samples = hf['input'].shape[0]
                    elif self.model_type == 'Planning_Model':
                        # For Planning_Model, use 'cross_target' to determine number of samples
                        if 'cross_target' in hf:
                            num_samples = hf['cross_target'].shape[0]
                        else:
                            raise KeyError(f"'cross_target' dataset not found in {file_path} for Planning_Model.")
                        if 'target_list' not in hf:
                            raise KeyError(f"'target_list' dataset not found in {file_path} for Planning_Model.")
                    else:
                        raise ValueError(f"Unsupported model_type: {self.model_type}")
                    
                    for i in range(num_samples):
                        print(f"\n\n\nReading {file_path} sample {i}")
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
                if self.model_type == "Battle_Model":
                    # Initialize dictionary to hold data
                    for key in hf.keys():
                        if key in ['input', 'images', 'positions', 'player_charges', 'enemy_charges', 'net_rewards']:
                            data_point[key] = torch.tensor(hf[key][sample_idx], dtype=torch.float32)
                    # Ensure all required keys are present
                    required_keys = ['input', 'net_rewards']
                    if 'images' in data_point:
                        required_keys.append('images')
                    if 'positions' in data_point:
                        required_keys.append('positions')
                    if 'player_charges' in data_point:
                        required_keys.append('player_charges')
                    if 'enemy_charges' in data_point:
                        required_keys.append('enemy_charges')
                    for rk in required_keys:
                        assert rk in data_point, f"Missing required key '{rk}' in Battle_Model data."
                elif self.model_type == "Planning_Model":
                    # Initialize nested inputs
                    inputs = {
                        'player_folder': {
                            'chips_onehot': torch.tensor(hf['inputs_player_folder_chips_onehot'][sample_idx], dtype=torch.float32),
                            'codes_onehot': torch.tensor(hf['inputs_player_folder_codes_onehot'][sample_idx], dtype=torch.float32),
                            'flags': torch.tensor(hf['inputs_player_folder_flags'][sample_idx], dtype=torch.float32)
                        },
                        'enemy_folder': {
                            'chips_onehot': torch.tensor(hf['inputs_enemy_folder_chips_onehot'][sample_idx], dtype=torch.float32),
                            'codes_onehot': torch.tensor(hf['inputs_enemy_folder_codes_onehot'][sample_idx], dtype=torch.float32),
                            'flags': torch.tensor(hf['inputs_enemy_folder_flags'][sample_idx], dtype=torch.float32)
                        },
                        'visible_chips': {
                            'chips_onehot': torch.tensor(hf['inputs_visible_chips_chips_onehot'][sample_idx], dtype=torch.float32),
                            'codes_onehot': torch.tensor(hf['inputs_visible_chips_codes_onehot'][sample_idx], dtype=torch.float32)
                        },
                        'health': torch.tensor(hf['health'][sample_idx], dtype=torch.float32),
                        'current_crosses': torch.tensor(hf['current_crosses'][sample_idx], dtype=torch.float32),
                        'available_crosses': torch.tensor(hf['available_crosses'][sample_idx], dtype=torch.float32),
                        'beast_flags': torch.tensor(hf['beast_flags'][sample_idx], dtype=torch.float32)
                    }
                    # Extract targets and reward
                    cross_target = int(hf['cross_target'][sample_idx])
                    target_list = hf['target_list'][sample_idx].tolist()
                    reward = float(hf['reward'][sample_idx])
                    
                    # Structure the data_point
                    data_point = {
                        'inputs': inputs,
                        'cross_target': cross_target,
                        'target_list': target_list,
                        'reward': reward
                    }
                return data_point
        except Exception as e:
            print(f"Failed to read sample {sample_idx} from {file_path}: {e}")
            # Optionally, you can raise the exception or handle it as needed
            raise e
