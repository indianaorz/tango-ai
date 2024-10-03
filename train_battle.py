# train_battle.py

import os
import glob
import h5py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from battle_network_model import BattleNetworkModel
from utils import get_root_dir, get_image_memory

class BattleDataset(Dataset):
    def __init__(self, sequences, actions, rewards):
        """
        Initializes the dataset with gamestate sequences and corresponding targets.

        Args:
            sequences (list of list of dict): List where each element is a list of gamestates (dicts).
            actions (torch.Tensor): Tensor of shape (num_samples, 16) containing action targets.
            rewards (torch.Tensor): Tensor of shape (num_samples,) containing reward targets.
        """
        assert len(sequences) == len(actions) == len(rewards), "Sequences, actions, and rewards must have the same length."
        self.sequences = sequences
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        """
        Retrieves the gamestate sequence and target for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (list of gamestates dicts, action tensor, reward tensor)
        """
        return self.sequences[idx], self.actions[idx], self.rewards[idx]

def load_h5_file(file_path):
    """
    Loads data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: Dictionary containing all datasets as tensors.
    """
    data = {}
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            if key in ['action', 'reward']:
                data[key] = torch.tensor(hf[key][:], dtype=torch.float32)
            else:
                data[key] = torch.tensor(hf[key][:], dtype=torch.float32)
    return data

def create_sequences(data, memory):
    """
    Creates sequences of gamestates with corresponding targets.

    Args:
        data (dict): Dictionary containing all datasets as tensors.
        memory (int): Number of past gamestates to consider.

    Returns:
        tuple: (sequences, actions, rewards)
            sequences: List of sequences, each sequence is a list of gamestate dicts.
            actions: Tensor of shape (num_sequences, 16)
            rewards: Tensor of shape (num_sequences,)
    """
    num_samples = data['cust_gage'].shape[0]
    sequences = []
    actions = []
    rewards = []

    for idx in range(num_samples):
        sequence = []
        for m in range(memory):
            seq_idx = idx - (memory - 1 - m)
            seq_idx = max(seq_idx, 0)  # Pad with the first gamestate if out of bounds
            gamestate = {}
            for key in data.keys():
                if key not in ['action', 'reward']:
                    gamestate[key] = data[key][seq_idx]
            sequence.append(gamestate)
        sequences.append(sequence)
        actions.append(data['action'][idx])
        rewards.append(data['reward'][idx])

    actions = torch.stack(actions)  # Shape: (num_sequences, 16)
    rewards = torch.stack(rewards)  # Shape: (num_sequences,)
    return sequences, actions, rewards

def collate_fn(batch):
    """
    Custom collate function to handle sequences of dicts.

    Args:
        batch (list of tuples): Each tuple is (sequence, action, reward)

    Returns:
        sequences_batched (list of dicts): List of dicts for each memory step, keys -> batched tensors
        actions (torch.Tensor): Batched actions
        rewards (torch.Tensor): Batched rewards
    """
    sequences, actions, rewards = zip(*batch)  # sequences is list of sequences, each sequence is list of dicts

    memory = len(sequences[0])  # assuming all sequences have same memory

    sequences_batched = []
    for m in range(memory):
        batch_dict = {}
        for key in sequences[0][m].keys():
            if key == 'player_chip_hand':
                # Stack the 'player_chip_hand' tensors
                stacked = torch.stack([sequence[m][key] for sequence in sequences], dim=0)  # Shape: (batch_size, 2005)

                # Split into 5 tensors of shape (batch_size, 401)
                split_tensors = torch.split(stacked, 401, dim=1)  # Returns a tuple of 5 tensors

                # Convert tuple to list for consistency with the model's expectation
                split_tensors = list(split_tensors)  # List of 5 tensors each of shape (batch_size, 401)

                batch_dict[key] = split_tensors  # List of 5 tensors
            else:
                # Stack normally
                batch_dict[key] = torch.stack([sequence[m][key] for sequence in sequences], dim=0)
        sequences_batched.append(batch_dict)

    actions = torch.stack(actions, dim=0)
    rewards = torch.stack(rewards, dim=0)

    return sequences_batched, actions, rewards

def main():
    # Configuration
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data', 'battle_data')
    h5_files = glob.glob(os.path.join(data_dir, '*.h5'))
    random.shuffle(h5_files)  # Shuffle the order of HDF5 files

    memory = get_image_memory()  # e.g., memory = 5

    all_sequences = []
    all_actions = []
    all_rewards = []

    print(f"Total HDF5 files found: {len(h5_files)}")

    for file_path in h5_files:
        print(f"Loading data from {file_path}...")
        data = load_h5_file(file_path)
        sequences, actions, rewards = create_sequences(data, memory)
        all_sequences.extend(sequences)
        all_actions.append(actions)
        all_rewards.append(rewards)

    # Concatenate all actions and rewards
    all_actions_tensor = torch.cat(all_actions, dim=0)  # Shape: (total_samples, 16)
    all_rewards_tensor = torch.cat(all_rewards, dim=0)  # Shape: (total_samples,)

    print(f"Total sequences created: {len(all_sequences)}")

    # Create Dataset and DataLoader
    dataset = BattleDataset(all_sequences, all_actions_tensor, all_rewards_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Initialize model
    model = BattleNetworkModel(image_option='None', memory=memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss(reduction='none')  # We'll handle reduction manually to incorporate rewards
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch_idx, (sequences_batched, targets, rewards) in enumerate(dataloader):
            batch_size = targets.size(0)

            # Move data to device
            for m in range(memory):
                for key in sequences_batched[m].keys():
                    if key == 'player_chip_hand':
                        # Move each tensor in the list to the device
                        sequences_batched[m][key] = [tensor.to(device) for tensor in sequences_batched[m][key]]
                    else:
                        # Move the tensor to the device
                        sequences_batched[m][key] = sequences_batched[m][key].to(device)
            targets = targets.to(device)
            rewards = rewards.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sequences_batched)  # Shape: (batch_size, 16)

            # Compute loss
            loss_per_sample = criterion(outputs, targets)  # Shape: (batch_size, 16)
            loss_per_sample = loss_per_sample.mean(dim=1)  # Shape: (batch_size,)
            weighted_loss = loss_per_sample * rewards  # Incorporate reward as a weight
            loss = weighted_loss.mean()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            total_batches += 1

            # Print statistics every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / total_batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {avg_loss:.4f}')
                running_loss = 0.0
                total_batches = 0

        # Optionally, save the model checkpoint after each epoch
        checkpoint_dir = os.path.join(root_dir, 'models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'battle_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

    print('Training completed.')

if __name__ == '__main__':
    main()
