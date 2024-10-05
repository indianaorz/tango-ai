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

    # Flag to determine whether to load all data into memory or load sequentially
    load_all_data = True

    # Define button mapping: (button_name, bit_position)
    buttons = [
        ('MENU2', 15-9),    # 0000001000000000
        ('MENU', 15-8),    # 0000000100000000
        ('DOWN', 15-7),    # 0000000010000000
        ('UP', 15-6),      # 0000000001000000
        ('LEFT', 15-5),    # 0000000000100000
        ('RIGHT', 15-4),   # 0000000000010000
        ('SHOOT', 15-1),   # 0000000000000010
        ('CHIP', 15-0)     # 0000000000000001
    ]

    # Initialize devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models, loss functions, and optimizers
    models = {}
    optimizers = {}
    criteria = {}
    checkpoint_paths = {}

    for button_name, bit_pos in buttons:
        model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5)
        model.to(device)
        model.train()  # Set the model to training mode

        # Load the latest checkpoint if it exists
        button_checkpoint_dir = os.path.join(checkpoint_dir, button_name)
        os.makedirs(button_checkpoint_dir, exist_ok=True)
        checkpoint_files = glob.glob(os.path.join(button_checkpoint_dir, f"*.pt"))
        checkpoint_files.sort(key=os.path.getmtime)
        checkpoint_epoch = 0
        if len(checkpoint_files) > 0:
            checkpoint_path = checkpoint_files[-1]
            print(f"Loading checkpoint for {button_name} from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
            checkpoint_epoch = int(os.path.splitext(os.path.basename(checkpoint_path))[0].split('_')[-1])
        else:
            checkpoint_path = os.path.join(button_checkpoint_dir, f"checkpoint_0.pt")
        checkpoint_paths[button_name] = checkpoint_path

        # Define loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Store in dictionaries
        models[button_name] = model
        criteria[button_name] = criterion
        optimizers[button_name] = optimizer
        
    batch_size = 512

    # Training parameters
    num_epochs = 20
    if load_all_data:
        all_sequences = []
        all_actions = []
        all_rewards = []

        for file_path in h5_files:
            print(f"Loading data from {file_path}...")
            data = load_h5_file(file_path)
            sequences, actions, rewards = prepare_sequences(data, memory=memory)
            all_sequences.extend(sequences)
            all_actions.append(actions)
            all_rewards.append(rewards)

        # Concatenate all actions and rewards
        all_actions_tensor = torch.cat(all_actions, dim=0)
        all_rewards_tensor = torch.cat(all_rewards, dim=0)

        # Initialize Dataset and DataLoader for all data
        dataset = BattleDataset(all_sequences, all_actions_tensor, all_rewards_tensor, memory=memory)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle the data for better training
            num_workers=0,  # Adjust based on your system
            collate_fn=custom_collate_fn  # Use the custom collate function
        )


        


    # Training loop with progress bar
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        if load_all_data:
            total_batches = 0
            running_losses = {button_name: 0.0 for button_name in models.keys()}
            progress_bar = tqdm(dataloader, desc="Training on all data", unit="batch")
            for batch_idx, (batched_sequences, actions, rewards) in enumerate(progress_bar):
                total_batches += 1
                # Move data to device
                for t in range(memory):
                    for key in batched_sequences[t].keys():
                        if isinstance(batched_sequences[t][key], list):
                            batched_sequences[t][key] = [tensor.to(device) for tensor in batched_sequences[t][key]]
                        else:
                            batched_sequences[t][key] = batched_sequences[t][key].to(device)

                actions = actions.to(device)
                rewards = rewards.to(device)

                # Iterate over each model
                for button_name, bit_pos in buttons:
                    model = models[button_name]
                    criterion = criteria[button_name]
                    optimizer = optimizers[button_name]

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    try:
                        # Skip if the batch size is 1
                        if batched_sequences[0]['grid'].shape[0] == 1:
                            continue

                        # Forward pass
                        outputs = model(batched_sequences)  # Shape: (batch_size, 1)

                        # Extract target for the specific button
                        targets = actions[:, bit_pos].unsqueeze(1)  # Shape: (batch_size, 1)

                        # Compute loss
                        loss = criterion(outputs, targets)

                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()

                        # Update running loss
                        running_losses[button_name] += loss.item()
                    except Exception as e:
                        print(f"\nError during forward pass for {button_name}: {e}")
                        traceback.print_exc()
                        return  # Exit training loop on error

                # Calculate average loss across all buttons
                avg_loss = sum(running_losses.values()) / (len(running_losses) * total_batches)
                progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
        if not load_all_data:
            for file_path in h5_files:
                print(f"Loading data from {file_path}...")
                data = load_h5_file(file_path)
                sequences, actions, rewards = prepare_sequences(data, memory=memory)

                # Initialize Dataset and DataLoader for the current file
                dataset = BattleDataset(sequences, actions, rewards, memory=memory)
                batch_size = 512
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,  # Shuffle the data for better training
                    num_workers=0,  # Adjust based on your system
                    collate_fn=custom_collate_fn  # Use the custom collate function
                )

                #initialize running losses
                running_losses = {button_name: 0.0 for button_name in models.keys()}
                total_batches = 0
                progress_bar = tqdm(dataloader, desc=f"Training on {os.path.basename(file_path)}", unit="batch")

                for batch_idx, (batched_sequences, actions, rewards) in enumerate(progress_bar):
                    total_batches += 1
                    # Move data to device
                    for t in range(memory):
                        for key in batched_sequences[t].keys():
                            if isinstance(batched_sequences[t][key], list):
                                batched_sequences[t][key] = [tensor.to(device) for tensor in batched_sequences[t][key]]
                            else:
                                batched_sequences[t][key] = batched_sequences[t][key].to(device)

                    actions = actions.to(device)
                    rewards = rewards.to(device)

                    # Iterate over each model
                    for button_name, bit_pos in buttons:
                        model = models[button_name]
                        criterion = criteria[button_name]
                        optimizer = optimizers[button_name]

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        try:
                            # Skip if the batch size is 1
                            if batched_sequences[0]['grid'].shape[0] == 1:
                                continue

                            # Forward pass
                            outputs = model(batched_sequences)  # Shape: (batch_size, 1)

                            # Extract target for the specific button
                            targets = actions[:, bit_pos].unsqueeze(1)  # Shape: (batch_size, 1)

                            # Compute loss
                            loss = criterion(outputs, targets)

                            # Backward pass and optimization
                            loss.backward()
                            optimizer.step()

                            # Update running loss
                            running_losses[button_name] += loss.item()
                        except Exception as e:
                            print(f"\nError during forward pass for {button_name}: {e}")
                            traceback.print_exc()
                            return  # Exit training loop on error

                    # Calculate average loss across all buttons
                    avg_loss = sum(running_losses.values()) / (len(running_losses) * total_batches)
                    progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        # After each epoch, save the models
        for button_name, model in models.items():
            button_checkpoint_dir = os.path.join(checkpoint_dir, button_name)
            os.makedirs(button_checkpoint_dir, exist_ok=True)
            checkpoint_epoch = epoch + 1
            checkpoint_path = os.path.join(button_checkpoint_dir, f"checkpoint_{checkpoint_epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint for {button_name} saved at {checkpoint_path}")

            # Ensure only the latest 5 checkpoints are kept
            checkpoint_files = glob.glob(os.path.join(button_checkpoint_dir, f"*.pt"))
            checkpoint_files.sort(key=os.path.getmtime)
            if len(checkpoint_files) > 5:
                os.remove(checkpoint_files[0])

    print('Training completed.')

if __name__ == '__main__':
    main()