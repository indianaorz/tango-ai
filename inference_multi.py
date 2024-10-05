# inference_multi.py

import os
import glob
import h5py
import random
import torch
import torch.nn.functional as F
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

def prepare_inference_sequence(data, start_idx, memory=10):
    """
    Prepares a sequence of gamestates for inference.

    Args:
        data (dict): Dictionary containing all datasets as tensors.
        start_idx (int): Starting index for the sequence.
        memory (int): Number of past gamestates to include.

    Returns:
        list of dict: List of gamestates for the model.
        torch.Tensor: Actual action tensor for comparison.
    """
    sequence = []
    for m in range(memory):
        idx = start_idx - memory + 1 + m
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
    # Retrieve the actual action at the start_idx
    actual_action = data['action'][start_idx]  # Shape: (16,)
    return sequence, actual_action

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

def batch_gamestates(gamestates_list):
    """
    Batches multiple gamestates into a single gamestate dictionary with batch dimension.

    Args:
        gamestates_list (list of dict): List of gamestates to batch.

    Returns:
        dict: Batched gamestate with tensors having batch_size dimension.
    """
    batched_gamestate = {}
    for key in gamestates_list[0].keys():
        if isinstance(gamestates_list[0][key], list):
            # For list-based fields like player_chip_hand, player_folder, enemy_folder
            batched_gamestate[key] = [torch.cat([g[key][i] for g in gamestates_list], dim=0) for i in range(len(gamestates_list[0][key]))]
        else:
            # For tensor fields, use torch.cat directly
            batched_gamestate[key] = torch.cat([g[key] for g in gamestates_list], dim=0)  # Shape: (batch_size, ...)
    return batched_gamestate

def main():
    # Configuration
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
    
    # Select a random HDF5 file for testing
    h5_files = glob.glob(os.path.join(data_dir, '*.h5'))
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}.")
        return
    test_file = random.choice(h5_files)
    print(f"Selected test file: {test_file}")
    
    # Load data from the selected HDF5 file
    data = load_h5_file(test_file)
    num_samples = data['action'].shape[0]
    if num_samples < memory:
        print(f"Not enough samples in the dataset to test with memory={memory}.")
        return
    
    # Select a random starting index ensuring enough gamestates
    max_start_idx = num_samples - 1
    start_idx = random.randint(memory - 1, max_start_idx)
    
    # Find a start index where the action is not all 0s
    while torch.sum(data['action'][start_idx]) == 0 and start_idx < max_start_idx:
        start_idx = random.randint(memory - 1, max_start_idx)
    if torch.sum(data['action'][start_idx]) == 0:
        print("Could not find a non-zero action in the dataset.")
        return
    
    # Log the action
    print(f"Action at start index: {data['action'][start_idx]}")
    print(f"Selected starting index for inference: {start_idx}")
    
    # Prepare the sequence and retrieve actual action
    sequence, actual_action = prepare_inference_sequence(data, start_idx, memory=memory)
    
    # Ensure the sequence has exactly 'memory' gamestates
    if len(sequence) != memory:
        print(f"Error: Prepared sequence length {len(sequence)} does not match memory {memory}.")
        return
    
    # No need to batch since we're dealing with a single sample (batch_size=1)
    # Each gamestate in the sequence already has batch_size=1
    batched_gamestates_list = sequence  # List of 'memory' gamestates with batch_size=1
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move the batched gamestates to the device
    for i in range(len(batched_gamestates_list)):
        for key in batched_gamestates_list[i].keys():
            if isinstance(batched_gamestates_list[i][key], list):
                for j in range(len(batched_gamestates_list[i][key])):
                    batched_gamestates_list[i][key][j] = batched_gamestates_list[i][key][j].to(device)
            else:
                batched_gamestates_list[i][key] = batched_gamestates_list[i][key].to(device)
    
    # Load the unified model
    unified_model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5, output_size=8)
    unified_model.to(device)
    unified_model.eval()  # Set the model to evaluation mode
    
    # Find the latest checkpoint for the unified model
    unified_checkpoint_dir = os.path.join(checkpoint_dir)  # Example directory
    os.makedirs(unified_checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(unified_checkpoint_dir, f"*.pt"))
    if not checkpoint_files:
        print(f"No checkpoints found for the unified model in {unified_checkpoint_dir}.")
        return
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading unified model checkpoint from {latest_checkpoint}")
    try:
        unified_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    except Exception as e:
        print(f"Error loading unified model: {e}")
        return
    
    # Define button names in the order of the output
    button_names = [btn[0] for btn in buttons]
    
    # Perform inference
    with torch.no_grad():
        print("\n--- Inference Results ---")
        try:
            outputs = unified_model(batched_gamestates_list)  # Shape: (batch_size=1, 8)
            predicted_probs = outputs.squeeze(0)  # Shape: (8,)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return
    
    # Process and display results
    print("\nPredicted Button Press Probabilities:")
    for idx, btn in enumerate(button_names):
        print(f"{btn}: {predicted_probs[idx].item():.4f}")
    
    # Compare with actual action
    actual_action_bits = [int(bit) for bit in data['action'][start_idx].tolist()]
    print("\nActual Button Presses:")
    for idx, btn in enumerate(button_names):
        print(f"{btn}: {actual_action_bits[(buttons[idx][1])]}")  # Adjust bit position as needed

if __name__ == '__main__':
    main()
