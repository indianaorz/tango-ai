# train_battle.py

import os
import glob
import h5py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import traceback
from battle_network_model import BattleNetworkModel
from utils import get_root_dir, get_image_memory

from tqdm import tqdm  # For progress bars


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


class BattleDataset(Dataset):
    def __init__(self, gamestate_sequences, actions, rewards, memory=10):
        """
        Initializes the dataset with gamestate sequences and corresponding targets.

        Args:
            gamestate_sequences (list of list of dict): List where each element is a list of gamestates (dicts).
            actions (torch.Tensor): Tensor of shape (num_samples, 16) containing target actions.
            rewards (torch.Tensor): Tensor of shape (num_samples,) containing target rewards.
            memory (int): Number of past gamestates to consider.
        """
        assert len(gamestate_sequences) == len(actions) == len(rewards), "Number of samples, actions, and rewards must match."
        self.gamestate_sequences = gamestate_sequences
        self.actions = actions
        self.rewards = rewards
        self.memory = memory

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        """
        Retrieves the gamestate sequence and target for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (list of gamestates dicts, target action tensor, target reward tensor)
        """
        return self.gamestate_sequences[idx], self.actions[idx], self.rewards[idx]


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of sequences containing dictionaries with list fields.

    Args:
        batch (list of tuples): Each tuple contains (sequence, action, reward).

    Returns:
        tuple: (batched_sequences, actions, rewards)
    """
    sequences, actions, rewards = zip(*batch)  # sequences is a tuple of sequences (list of dicts)

    memory = len(sequences[0])  # Number of memory steps
    batch_size = len(sequences)  # Number of sequences in the batch

    batched_sequences = []

    for t in range(memory):
        batched_gamestates = {}
        for key in sequences[0][t].keys():
            if isinstance(sequences[0][t][key], list):
                # Handle list fields like 'player_chip_hand', 'player_folder', 'enemy_folder'
                # Each element in the list corresponds to a separate tensor
                batched_gamestates[key] = [torch.cat([sequence[t][key][i] for sequence in sequences], dim=0) 
                                           for i in range(len(sequences[0][t][key]))]
            else:
                # Handle regular tensor fields
                batched_gamestates[key] = torch.cat([sequence[t][key] for sequence in sequences], dim=0)
        batched_sequences.append(batched_gamestates)

    actions = torch.stack(actions, dim=0)  # Shape: (batch_size, 16)
    rewards = torch.stack(rewards, dim=0)  # Shape: (batch_size,)

    return batched_sequences, actions, rewards


def prepare_sequences(data, memory=10):
    """
    Prepares sequences of gamestates with the specified memory.

    Args:
        data (dict): Dictionary containing all datasets as tensors.
        memory (int): Number of past gamestates to include in each sequence.

    Returns:
        list of list of dict: List of sequences, each sequence is a list of gamestates dicts.
        torch.Tensor: Tensor of target actions.
        torch.Tensor: Tensor of target rewards.
    """
    num_gamestates = data['action'].shape[0]
    all_sequences = []
    all_actions = []
    all_rewards = []

    # Iterate through each gamestate index
    for i in range(num_gamestates):
        sequence = []
        for m in range(memory):
            idx = i - memory + 1 + m
            if idx < 0:
                # If index is negative, pad with default gamestates
                gamestate = {
                    'cust_gage': torch.tensor([0.0], dtype=torch.float32),
                    'grid': torch.zeros(1, 6, 3, 16, dtype=torch.float32),
                    'player_health': torch.tensor([1.0], dtype=torch.float32),
                    'enemy_health': torch.tensor([1.0], dtype=torch.float32),
                    'player_chip': F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float().unsqueeze(0),
                    'enemy_chip': F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float().unsqueeze(0),
                    'player_charge': torch.tensor([0.0], dtype=torch.float32),
                    'enemy_charge': torch.tensor([0.0], dtype=torch.float32),
                    'player_chip_hand': [F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float().unsqueeze(0) for _ in range(5)],
                    'player_folder': [torch.zeros(1, 431, dtype=torch.float32) for _ in range(30)],
                    'enemy_folder': [torch.zeros(1, 431, dtype=torch.float32) for _ in range(30)],
                    'player_custom': torch.zeros(1, 200, dtype=torch.float32),
                    'enemy_custom': torch.zeros(1, 200, dtype=torch.float32),
                    'player_emotion_state': torch.zeros(1, 27, dtype=torch.float32),
                    'enemy_emotion_state': torch.zeros(1, 27, dtype=torch.float32),
                    'player_used_crosses': torch.zeros(1, 10, dtype=torch.float32),
                    'enemy_used_crosses': torch.zeros(1, 10, dtype=torch.float32),
                    'player_beasted_out': torch.tensor([0.0], dtype=torch.float32),
                    'enemy_beasted_out': torch.tensor([0.0], dtype=torch.float32),
                    'player_beasted_over': torch.tensor([0.0], dtype=torch.float32),
                    'enemy_beasted_over': torch.tensor([0.0], dtype=torch.float32)
                }
            else:
                # Load gamestate from data
                gamestate = {
                    'cust_gage': data['cust_gage'][idx].unsqueeze(0),
                    'grid': data['grid'][idx].unsqueeze(0),
                    'player_health': data['player_health'][idx].unsqueeze(0),
                    'enemy_health': data['enemy_health'][idx].unsqueeze(0),
                    'player_chip': data['player_chip'][idx].unsqueeze(0),
                    'enemy_chip': data['enemy_chip'][idx].unsqueeze(0),
                    'player_charge': data['player_charge'][idx].unsqueeze(0),
                    'enemy_charge': data['enemy_charge'][idx].unsqueeze(0),
                    'player_chip_hand': [data['player_chip_hand'][idx].view(-1, 401)[j].unsqueeze(0) for j in range(5)],
                    'player_folder': [data['player_folder'][idx].view(-1, 431)[j].unsqueeze(0) for j in range(30)],
                    'enemy_folder': [data['enemy_folder'][idx].view(-1, 431)[j].unsqueeze(0) for j in range(30)],
                    'player_custom': data['player_custom'][idx].unsqueeze(0),
                    'enemy_custom': data['enemy_custom'][idx].unsqueeze(0),
                    'player_emotion_state': data['player_emotion_state'][idx].unsqueeze(0),
                    'enemy_emotion_state': data['enemy_emotion_state'][idx].unsqueeze(0),
                    'player_used_crosses': data['player_used_crosses'][idx].unsqueeze(0),
                    'enemy_used_crosses': data['enemy_used_crosses'][idx].unsqueeze(0),
                    'player_beasted_out': data['player_beasted_out'][idx].unsqueeze(0),
                    'enemy_beasted_out': data['enemy_beasted_out'][idx].unsqueeze(0),
                    'player_beasted_over': data['player_beasted_over'][idx].unsqueeze(0),
                    'enemy_beasted_over': data['enemy_beasted_over'][idx].unsqueeze(0)
                }
            sequence.append(gamestate)
        all_sequences.append(sequence)
        all_actions.append(data['action'][i])
        all_rewards.append(data['reward'][i])

    all_actions_tensor = torch.stack(all_actions, dim=0)  # Shape: (num_samples, 16)
    all_rewards_tensor = torch.stack(all_rewards, dim=0)  # Shape: (num_samples,)

    return all_sequences, all_actions_tensor, all_rewards_tensor


def main():
    # Configuration
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data', 'battle_data')
    memory = get_image_memory()  # e.g., memory = 10
    checkpoint_dir = os.path.join(root_dir, f"checkpoints/battle/{memory}")
    h5_files = glob.glob(os.path.join(data_dir, '*.h5'))
    random.shuffle(h5_files)  # Shuffle the order of HDF5 files


    all_sequences = []
    all_actions = []
    all_rewards = []

    print(f"Total HDF5 files found: {len(h5_files)}")

    for file_path in h5_files:
        print(f"Loading data from {file_path}...")
        data = load_h5_file(file_path)
        sequences, actions, rewards = prepare_sequences(data, memory=memory)
        all_sequences.extend(sequences)
        all_actions.append(actions)
        all_rewards.append(rewards)

    # Concatenate all actions and rewards from all files
    all_actions_tensor = torch.cat(all_actions, dim=0)  # Shape: (total_samples, 16)
    all_rewards_tensor = torch.cat(all_rewards, dim=0)  # Shape: (total_samples,)

    print(f"Total training samples: {len(all_sequences)}")

    # Initialize Dataset and DataLoader
    dataset = BattleDataset(all_sequences, all_actions_tensor, all_rewards_tensor, memory=memory)
    batch_size = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Reduce to 0 to prevent multiple workers
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #load the latest checkpoint if it exists and note which epoch it was on
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"*.pt"))
    checkpoint_files.sort(key=os.path.getmtime)
    checkpoint_epoch = 0
    if len(checkpoint_files) > 0:
        checkpoint_path = checkpoint_files[-1]
        print(f"Loading checkpoint from {checkpoint_path}")
        model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5)
        model.load_state_dict(torch.load(checkpoint_path))
        checkpoint_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
    else:
        model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5)
    model.to(device)
    model.train()  # Set the model to training mode

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 20

    # Training loop with progress bar
    for epoch in range(num_epochs):
        print(f"\nEpoch {checkpoint_epoch + epoch + 1}/{checkpoint_epoch + num_epochs}")
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", unit="batch")

        for batch_idx, (batched_sequences, targets, rewards) in enumerate(progress_bar):
            # Move data to device
            for t in range(memory):
                for key in batched_sequences[t].keys():
                    if isinstance(batched_sequences[t][key], list):
                        # Move each tensor in the list to the device
                        batched_sequences[t][key] = [tensor.to(device) for tensor in batched_sequences[t][key]]
                    else:
                        batched_sequences[t][key] = batched_sequences[t][key].to(device)

            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            try:
                #skip if the batch size is 1
                if batched_sequences[0]['grid'].shape[0] == 1:
                    continue
                # Forward pass
                outputs = model(batched_sequences)  # Shape: (batch_size, 16)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))
            except Exception as e:
                print(f"\nError during forward pass: {e}")
                traceback.print_exc()
                return  # Exit training loop on error

        average_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}")

        # Save the model after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_epoch + epoch + 1}.pt")
        #create dir if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
        #ensure there's only 5 models in the save folder
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"*.pt"))
        checkpoint_files.sort(key=os.path.getmtime)
        if len(checkpoint_files) > 5:
            os.remove(checkpoint_files[0])

    print('Training completed.')


if __name__ == '__main__':
    main()
