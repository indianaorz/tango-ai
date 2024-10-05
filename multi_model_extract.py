# train_unified.py

import os
import glob
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from battle_network_model import BattleNetworkModel
from utils import get_root_dir, get_image_memory
from tqdm import tqdm  # For progress bars
import traceback

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

def unpack_and_save_data(file_path, output_dir):
    """
    Unpacks the h5 file and saves each data tensor as a separate file for later use.
    """
    try:
        # Load data from the h5 file
        data = load_h5_file(file_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each tensor to a separate file
        for key, tensor in data.items():
            output_file_path = os.path.join(output_dir, f"{key}.pt")
            torch.save(tensor, output_file_path)
            print(f"Saved {key} to {output_file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

class BattleDataset(Dataset):
    def __init__(self, gamestate_sequences, actions, rewards, memory=10):
        assert len(gamestate_sequences) == len(actions) == len(rewards), "Number of samples, actions, and rewards must match."
        self.gamestate_sequences = gamestate_sequences
        self.actions = actions
        self.rewards = rewards
        self.memory = memory

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.gamestate_sequences[idx], self.actions[idx], self.rewards[idx]

def custom_collate_fn(batch):
    sequences, actions, rewards = zip(*batch)  # sequences is a tuple of sequences (list of dicts)

    memory = len(sequences[0])  # Number of memory steps
    batch_size = len(sequences)  # Number of sequences in the batch

    batched_sequences = []

    for t in range(memory):
        batched_gamestates = {}
        for key in sequences[0][t].keys():
            if isinstance(sequences[0][t][key], list):
                # Handle list fields like 'player_chip_hand', 'player_folder', 'enemy_folder'
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
    required_keys = [
        'cust_gage', 'grid', 'player_health', 'enemy_health', 'player_chip', 'enemy_chip',
        'player_charge', 'enemy_charge', 'player_chip_hand', 'player_folder', 'enemy_folder',
        'player_custom', 'enemy_custom', 'player_emotion_state', 'enemy_emotion_state',
        'player_used_crosses', 'enemy_used_crosses', 'player_beasted_out', 'enemy_beasted_out',
        'player_beasted_over', 'enemy_beasted_over', 'action', 'reward'
    ]

    # Check if all required keys are present
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in the data.")

    num_gamestates = data['action'].shape[0]
    all_sequences = []
    all_actions = []
    all_rewards = []

    for i in range(num_gamestates):
        sequence = []
        for m in range(memory):
            idx = i - memory + 1 + m
            if idx < 0:
                # Pad with default gamestates
                gamestate = get_default_gamestate()
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

def get_default_gamestate():
    """
    Returns a default gamestate dictionary for padding purposes.
    """
    return {
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

def train_with_unpacked_data(output_root_dir):
    """
    Loads unpacked data and trains on it using a unified model.
    """
    # Configuration
    memory = get_image_memory()  # e.g., memory = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data', 'battle_data')
    memory = get_image_memory()  # e.g., memory = 10
    checkpoint_dir = os.path.join(root_dir, f"checkpoints/battle/{memory}")
    
    # Define button mapping: (button_name, bit_position)
    buttons = [
        ('MENU2', 15-9),    # 0000001000000000
        ('MENU', 15-8),     # 0000000100000000
        ('DOWN', 15-7),     # 0000000010000000
        ('UP', 15-6),       # 0000000001000000
        ('LEFT', 15-5),     # 0000000000100000
        ('RIGHT', 15-4),    # 0000000000010000
        ('SHOOT', 15-1),    # 0000000000000010
        ('CHIP', 15-0)      # 0000000000000001
    ]
    
    # Initialize the unified model, loss function, and optimizer
    unified_model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5, output_size=8)
    unified_model.to(device)
    
    # Load the latest checkpoint if available
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"*.pt"))
    checkpoint_files.sort(key=os.path.getmtime)
    if len(checkpoint_files) > 0:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading unified model checkpoint from {latest_checkpoint}")
        try:
            unified_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
            start_epoch = int(os.path.splitext(os.path.basename(latest_checkpoint))[0].split('_')[-1]) + 1
        except Exception as e:
            print(f"Error loading unified model: {e}")
            start_epoch = 0
    else:
        print("No existing checkpoints found. Starting training from scratch.")
        start_epoch = 0
    
    unified_model.train()  # Set the model to training mode
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    optimizer = optim.Adam(unified_model.parameters(), lr=0.001)
    
    # Define button names in order corresponding to output indices
    button_names = [btn[0] for btn in buttons]
    
    batch_size = 512
    num_epochs = 200

    # Gather all unpacked data directories
    unpacked_dirs = glob.glob(os.path.join(output_root_dir, '*'))
    unpacked_dirs = [d for d in unpacked_dirs if os.path.isdir(d)]
    if not unpacked_dirs:
        print(f"No unpacked data directories found in {output_root_dir}.")
        return

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        total_batches = 0
        
        for unpacked_dir in unpacked_dirs:
            if not os.path.isdir(unpacked_dir):
                continue  # Skip if it's not a directory

            print(f"Training with data from {unpacked_dir}...")
            # Load tensors from unpacked directory
            data = {}
            for tensor_file in glob.glob(os.path.join(unpacked_dir, '*.pt')):
                key = os.path.splitext(os.path.basename(tensor_file))[0]
                data[key] = torch.load(tensor_file)
            
            # Check if all required keys are present
            required_keys = [
                'cust_gage', 'grid', 'player_health', 'enemy_health', 'player_chip', 'enemy_chip',
                'player_charge', 'enemy_charge', 'player_chip_hand', 'player_folder', 'enemy_folder',
                'player_custom', 'enemy_custom', 'player_emotion_state', 'enemy_emotion_state',
                'player_used_crosses', 'enemy_used_crosses', 'player_beasted_out', 'enemy_beasted_out',
                'player_beasted_over', 'enemy_beasted_over', 'action', 'reward'
            ]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"Skipping {unpacked_dir} due to missing keys: {missing_keys}")
                continue
            
            # Prepare sequences, actions, and rewards
            sequences, actions, rewards = prepare_sequences(data, memory=memory)
            dataset = BattleDataset(sequences, actions, rewards, memory=memory)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Adjust based on your system
                collate_fn=custom_collate_fn
            )
            
            progress_bar = tqdm(dataloader, desc=f"Training on {os.path.basename(unpacked_dir)}", unit="batch")
            
            for batch_idx, (batched_sequences, actions_batch, rewards_batch) in enumerate(progress_bar):
                total_batches += 1
                # Move data to device
                for t in range(memory):
                    for key in batched_sequences[t].keys():
                        if isinstance(batched_sequences[t][key], list):
                            batched_sequences[t][key] = [tensor.to(device) for tensor in batched_sequences[t][key]]
                        else:
                            batched_sequences[t][key] = batched_sequences[t][key].to(device)
                
                actions_batch = actions_batch.to(device)  # Shape: (batch_size, 16)
                rewards_batch = rewards_batch.to(device)  # Shape: (batch_size,)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    outputs = unified_model(batched_sequences)  # Shape: (batch_size, 8)
                    
                    # Prepare target labels
                    # Map the 16-bit action to 8 outputs based on your button mapping
                    # Ensure that button_names correspond to the output indices
                    target_labels = torch.zeros(outputs.shape, dtype=torch.float32).to(device)  # Shape: (batch_size, 8)
                    for idx, (button_name, bit_pos) in enumerate(buttons):
                        target_labels[:, idx] = actions_batch[:, bit_pos]
                    
                    # Compute loss
                    loss = criterion(outputs, target_labels)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    # Update running loss
                    running_loss += loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print(f"\nError during forward/backward pass: {e}")
                    traceback.print_exc()
                    continue  # Skip to next batch

            # Calculate average loss for the epoch
            avg_loss = running_loss / total_batches if total_batches > 0 else 0
            print(f"\nEpoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
            
            # Save the model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pt")
            torch.save(unified_model.state_dict(), checkpoint_path)
            print(f"Saved unified model checkpoint at {checkpoint_path}")
            
            # Optionally, keep only the latest N checkpoints to save space
            N = 5
            checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")), key=os.path.getmtime)
            if len(checkpoint_files) > N:
                for ckpt in checkpoint_files[:-N]:
                    os.remove(ckpt)
                    print(f"Removed old checkpoint: {ckpt}")

    print('Training completed.')

def main():
    # Configuration
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data', 'battle_data')
    output_root_dir = os.path.join(root_dir, 'data', 'unpacked_data')
    
    
    # Mode selection
    unpack_data = False  # Set to True to unpack data, False to train
    
    if unpack_data:
        # Get list of h5 files
        h5_files = glob.glob(os.path.join(data_dir, '*.h5'))
        
        # Iterate over each file and unpack the data if not already unpacked
        for file_path in h5_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(output_root_dir, file_name)
            if not os.path.exists(output_dir):
                print(f"Unpacking data from {file_path}...")
                unpack_and_save_data(file_path, output_dir)
            else:
                print(f"Data from {file_path} is already unpacked. Skipping.")
        
        print("All files unpacked.")
    else:
        # Train with unpacked data
        train_with_unpacked_data(output_root_dir)

if __name__ == '__main__':
    main()
