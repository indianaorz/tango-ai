#battle_network_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import traceback

import numpy as np

import h5py
import os
from datetime import datetime, timezone

# Define your default gamestate for padding
def get_default_gamestate():
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

def safe_cat(tensors, dim=0):
    """
    Safely concatenates a list or tuple of tensors along the specified dimension.
    
    Args:
        tensors (list or tuple of torch.Tensor or torch.Tensor): 
            The tensors to concatenate. Can be a single tensor or a list/tuple of tensors.
        dim (int): The dimension along which to concatenate.
    
    Returns:
        torch.Tensor: The concatenated tensor.
    
    Raises:
        ValueError: If tensors is neither a list/tuple nor a single tensor.
        TypeError: If elements within tensors are not torch.Tensor instances.
    """
    if isinstance(tensors, torch.Tensor):
        # Single tensor, return as is or add a new dimension if needed
        return tensors
    elif isinstance(tensors, (list, tuple)):
        if len(tensors) == 0:
            raise ValueError("safe_cat received an empty list/tuple of tensors.")
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("All elements in tensors must be torch.Tensor instances.")
        return torch.cat(tensors, dim=dim)
    else:
        raise ValueError("safe_cat expects a torch.Tensor or a list/tuple of torch.Tensor instances.")
  
# Function to prepare the sequence of gamestates
def prepare_inference_sequence(data_buffer, current_data_point, memory=10):
    """
    Prepares a sequence of gamestates for model inference.
    
    Args:
        data_buffer (deque or list): Historical gamestates.
        current_data_point (dict): Current gamestate.
        memory (int): Number of past gamestates to include.
    
    Returns:
        list of dict: Sequence of gamestates of length 'memory'.
    """
    sequence = list(data_buffer)  # Convert deque to list if necessary
    sequence.append(current_data_point)
    
    
    # If sequence is shorter than memory, pad with default gamestates at the beginning
    while len(sequence) < memory:
        sequence.insert(0, get_default_gamestate())
    
    # If sequence is longer than memory, trim the oldest gamestates
    if len(sequence) > memory:
        sequence = sequence[-memory:]
    
    return sequence
    
class BattleNetworkModel(nn.Module):
    def __init__(self, image_option='None', memory=1, scale=1.0, dropout_p=0.5, output_size=8):
        """
        Initializes the BattleNetworkModel.

        Args:
            image_option (str): Options are 'None', 'Greyscale', 'Full'.
                                Determines how the image input is processed.
            memory (int): Number of past gamestates to consider for the model.
            scale (float): Scaling factor for the hidden layers to adjust model complexity.
            dropout_p (float): Dropout probability.
        """
        super(BattleNetworkModel, self).__init__()
        
        self.image_option = image_option
        self.memory = memory
        self.scale = scale
        self.dropout_p = dropout_p

        # Image Processing Layers
        if self.image_option != 'None':
            if self.image_option == 'Greyscale':
                in_channels = 1
            elif self.image_option == 'Full':
                in_channels = 3
            else:
                raise ValueError("image_option must be 'None', 'Greyscale', or 'Full'")
                
            # Convolutional Neural Network for Image Processing
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels, int(32 * scale), kernel_size=8, stride=4),
                nn.ReLU(),
                nn.BatchNorm2d(int(32 * scale)),
                nn.Conv2d(int(32 * scale), int(64 * scale), kernel_size=4, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(int(64 * scale)),
                nn.Conv2d(int(64 * scale), int(64 * scale), kernel_size=3, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(int(64 * scale)),
                nn.Flatten()
            )
            # Determine the size of the image feature vector
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, 240, 260)
                conv_output_size = self.image_conv(dummy_input).shape[1]
        else:
            conv_output_size = 0  # No image input

        # Grid Data Processing
        # Each grid object has 16 features, grid size is 6x3 = 18 objects
        self.grid_input_size = 6 * 3 * 16  # 288

        # Other Features Processing
        # Detailed calculation based on the input description
        # cust_gage: 1
        # player_health, enemy_health: 2
        # player_chip, enemy_chip: 2 * 401
        # player_charge, enemy_charge: 2
        # player_chip_hand: 5 * 401
        # player_folder, enemy_folder: 2 * 30 * 431
        # player_custom, enemy_custom: 2 * 200
        # player_emotion_state, enemy_emotion_state: 2 * 27
        # player_used_crosses, enemy_used_crosses: 2 * 10
        # player_beasted_out, enemy_beasted_out: 2
        # player_beasted_over, enemy_beasted_over: 2
        # Total: 1 + 2 + 802 + 2 + 2005 + 25860 + 400 + 54 + 20 + 4 = 28250

        self.other_input_size = (
            1 +  # cust_gage
            2 +  # player_health, enemy_health
            2 * 401 +  # player_chip, enemy_chip
            2 +  # player_charge, enemy_charge
            5 * 401 +  # player_chip_hand
            2 * 30 * 431 +  # player_folder, enemy_folder
            2 * 200 +  # player_custom, enemy_custom
            2 * 27 +  # player_emotion_state, enemy_emotion_state
            2 * 10 +  # player_used_crosses, enemy_used_crosses
            2 +  # player_beasted_out, enemy_beasted_out
            2    # player_beasted_over, enemy_beasted_over
        )  # Total: 28250

        # Total input size per gamestate
        self.gamestate_feature_size = conv_output_size + self.grid_input_size + self.other_input_size

        # LSTM Layer for Sequence Modeling
        self.lstm_hidden_size = int(512 * scale)
        self.lstm_num_layers = 2
        self.lstm = nn.LSTM(
            input_size=self.gamestate_feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=dropout_p if self.lstm_num_layers > 1 else 0
        )

        # Fully Connected Layers for Output
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_hidden_size),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_hidden_size // 2),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.lstm_hidden_size // 2, output_size),
            nn.Sigmoid()  # Output probabilities between 0 and 1 for each button
        )
        
    def forward(self, gamestates):
            """
            Forward pass of the model.

            Args:
                gamestates (list of dict): List of gamestates, length equal to self.memory.
                                        Each gamestate is a dictionary containing all the required inputs.

            Returns:
                Tensor: Output probabilities for the 16 buttons. Shape: (batch_size, 16)
            """
            batch_size = gamestates[0]['cust_gage'].size(0)
            gamestate_features = []

            for i in range(self.memory):
                gamestate = gamestates[i]
                features = []

                # Process image if included
                if self.image_option != 'None':
                    image = gamestate['screen_image']  # Shape: (batch_size, C, H, W)
                    image_features = self.image_conv(image)  # Shape: (batch_size, conv_output_size)
                    features.append(image_features)

                # Process grid data
                grid_data = gamestate['grid']  # Shape: (batch_size, 6, 3, 16)
                grid_data = grid_data.view(batch_size, -1)  # Flatten to (batch_size, 6*3*16)
                features.append(grid_data)

                # Process other inputs
                other_features = []

                # cust_gage: (batch_size,)
                other_features.append(gamestate['cust_gage'].unsqueeze(1))  # Shape: (batch_size, 1)

                # player_health, enemy_health: (batch_size,), (batch_size,)
                other_features.append(gamestate['player_health'].unsqueeze(1))
                other_features.append(gamestate['enemy_health'].unsqueeze(1))

                # player_chip, enemy_chip: (batch_size, 401), (batch_size, 401)
                other_features.append(gamestate['player_chip'])  # Already one-hot encoded
                other_features.append(gamestate['enemy_chip'])

                # player_charge, enemy_charge: (batch_size,), (batch_size,)
                other_features.append(gamestate['player_charge'].unsqueeze(1))
                other_features.append(gamestate['enemy_charge'].unsqueeze(1))

                # player_chip_hand: list of 5 one-hot encoded tensors of size (batch_size, 401)
                # Use safe_cat to concatenate along the feature dimension
                player_chip_hand = safe_cat(gamestate['player_chip_hand'], dim=1)  # Shape: (batch_size, 5*401)
                other_features.append(player_chip_hand)

                # player_folder, enemy_folder: list of 30 folder_chips each
                # Use safe_cat to concatenate along the feature dimension
                #shuffle the order of player_folder and enemy_folder
                # np.random.shuffle(gamestate['player_folder'])
                # np.random.shuffle(gamestate['enemy_folder'])
                          
                player_folder = safe_cat(gamestate['player_folder'], dim=1)  # Shape: (batch_size, 30*431)
                enemy_folder = safe_cat(gamestate['enemy_folder'], dim=1)    # Shape: (batch_size, 30*431)
                other_features.append(player_folder)
                other_features.append(enemy_folder)

                # player_custom, enemy_custom: (batch_size, 200), (batch_size, 200)
                other_features.append(gamestate['player_custom'])
                other_features.append(gamestate['enemy_custom'])

                # player_emotion_state, enemy_emotion_state: (batch_size, 27), (batch_size, 27)
                other_features.append(gamestate['player_emotion_state'])
                other_features.append(gamestate['enemy_emotion_state'])

                # player_used_crosses, enemy_used_crosses: (batch_size, 10), (batch_size, 10)
                other_features.append(gamestate['player_used_crosses'])
                other_features.append(gamestate['enemy_used_crosses'])

                # player_beasted_out, enemy_beasted_out: (batch_size,), (batch_size,)
                other_features.append(gamestate['player_beasted_out'].unsqueeze(1))
                other_features.append(gamestate['enemy_beasted_out'].unsqueeze(1))

                # player_beasted_over, enemy_beasted_over: (batch_size,), (batch_size,)
                other_features.append(gamestate['player_beasted_over'].unsqueeze(1))
                other_features.append(gamestate['enemy_beasted_over'].unsqueeze(1))

                # Concatenate all other features using safe_cat
                other_features = safe_cat(other_features, dim=1)  # Shape: (batch_size, other_input_size)
                features.append(other_features)

                # Concatenate all features for this gamestate using safe_cat
                gamestate_feature = safe_cat(features, dim=1)  # Shape: (batch_size, gamestate_feature_size)
                gamestate_features.append(gamestate_feature)

            # Stack gamestate features to form a sequence using safe_cat
            # Shape: (batch_size, memory, gamestate_feature_size)
            gamestate_sequence = torch.stack(gamestate_features, dim=1)

            # Pass the sequence through the LSTM
            lstm_out, _ = self.lstm(gamestate_sequence)  # lstm_out shape: (batch_size, memory, lstm_hidden_size)

            # Use the output from the last timestep
            lstm_last_output = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size)

            # Pass through fully connected layers to get the final output
            output = self.fc(lstm_last_output)  # Shape: (batch_size, output_size)

            return output

def get_gamestate_tensor(
    tensor_params
    # screen_image=None,  # (C, H, W) tensor or None
    # cust_gage=None,     # float (0-64)
    # grid_tiles=None,    # list of 18 integers (0-12)
    # grid_owner=None,    # list of 18 floats (0-1)
    # player_grid_position=None,  # list of 2 integers [x, y]
    # enemy_grid_position=None,   # list of 2 integers [x, y]
    # player_health=None,         # float (0-1)
    # enemy_health=None,          # float (0-1)
    # player_chip=None,           # integer (0-400)
    # enemy_chip=None,            # integer (0-400)
    # player_charge=None,         # float (0-1)
    # enemy_charge=None,          # float (0-1)
    # player_chip_hand=None,      # list of 5 integers (0-400) or None
    # player_folder=None,         # list of 30 dicts with keys 'chip', 'code', 'used', 'regged', 'tagged' or None
    # enemy_folder=None,          # list of 30 dicts with keys 'chip', 'code', 'used', 'regged', 'tagged' or None
    # player_custom=None,         # list of 200 floats (0 or 1) or None
    # enemy_custom=None,          # list of 200 floats (0 or 1) or None
    # player_emotion_state=None,  # integer (0-26)
    # enemy_emotion_state=None,   # integer (0-26)
    # player_used_crosses=None,  # list of integers (1-10)
    # enemy_used_crosses=None,   # list of integers (1-10)
    # player_beasted_out=None,    # bool
    # enemy_beasted_out=None,     # bool
    # player_beasted_over=None,   # bool
    # enemy_beasted_over=None,    # bool
):
    #set values from param object instead
    screen_image = tensor_params['screen_image'] if 'screen_image' in tensor_params else None
    cust_gage = tensor_params['cust_gage']
    grid_tiles = tensor_params['grid_state']
    grid_owner = tensor_params['grid_owner_state']
    player_grid_position = tensor_params['player_grid_position']
    enemy_grid_position = tensor_params['enemy_grid_position']
    player_health = tensor_params['player_health']
    enemy_health = tensor_params['enemy_health']
    player_chip = tensor_params['player_chip']
    enemy_chip = tensor_params['enemy_chip']
    player_charge = tensor_params['player_charge']
    enemy_charge = tensor_params['enemy_charge']
    player_chip_hand = tensor_params['player_chip_hand']
    player_folder = tensor_params['player_folder']
    enemy_folder = tensor_params['enemy_folder']
    player_custom = tensor_params['player_custom']
    enemy_custom = tensor_params['enemy_custom']
    player_emotion_state = tensor_params['player_emotion_state']
    enemy_emotion_state = tensor_params['enemy_emotion_state']
    player_used_crosses = tensor_params['player_used_crosses']
    enemy_used_crosses = tensor_params['enemy_used_crosses']
    player_beasted_out = tensor_params['player_beasted_out']
    enemy_beasted_out = tensor_params['enemy_beasted_out']
    player_beasted_over = tensor_params['player_beasted_over']
    enemy_beasted_over = tensor_params['enemy_beasted_over']
    
    """
    Converts a single game state into the tensor format expected by the model.
    Logs the shapes and types of all inputs for verification.

    Args:
        All parameters as described above.

    Returns:
        dict: gamestate dictionary with tensors
    """
    # print("Processing single gamestate...")

    gamestate = {}

    # 1. screen_image
    # if screen_image is not None:
    #     # Expecting a tensor of shape (C, H, W)
    #     if not isinstance(screen_image, torch.Tensor):
    #         screen_image = torch.tensor(screen_image, dtype=torch.float32)
    #     else:
    #         screen_image = screen_image.float()
    #     # Add batch dimension
    #     gamestate['screen_image'] = screen_image.unsqueeze(0)  # Shape: (1, C, H, W)
    # else:
    #     gamestate['screen_image'] = None

    # 2. cust_gage: Normalize between 0 and 1
    if cust_gage is not None:
        cust_gage = torch.tensor([cust_gage / 64.0], dtype=torch.float32)  # Shape: (1,)
    else:
        cust_gage = torch.tensor([0.0], dtype=torch.float32)
    gamestate['cust_gage'] = cust_gage

    # 3. grid
    # Each grid object has 16 features: 13 (grid_type one-hot) + 3 (grid_owner, player, enemy)
    grid = torch.zeros(1, 6, 3, 16, dtype=torch.float32)  # Shape: (1, 6, 3, 16)
    if grid_tiles is not None and grid_owner is not None:
        for tile_idx in range(18):
            x = tile_idx % 6  # columns from 0 to 5
            y = tile_idx // 6  # rows from 0 to 2

            # grid_type one-hot encoding
            grid_type_idx = grid_tiles[tile_idx]
            grid_type_onehot = torch.zeros(13, dtype=torch.float32)
            if 0 <= grid_type_idx < 13:
                grid_type_onehot[grid_type_idx] = 1.0
            else:
                print(f"Warning: grid_type_idx {grid_type_idx} out of range. Setting all to 0.")

            # grid_owner
            grid_owner_value = float(grid_owner[tile_idx]) if grid_owner[tile_idx] in [0, 1] else 0.0

            # player presence
            player_present = 0.0
            if player_grid_position is not None and [x + 1, y + 1] == player_grid_position:
                player_present = 1.0

            # enemy presence
            enemy_present = 0.0
            if enemy_grid_position is not None and [x + 1, y + 1] == enemy_grid_position:
                enemy_present = 1.0

            # Combine features
            grid_tile_features = torch.cat([
                grid_type_onehot,  # length 13
                torch.tensor([grid_owner_value, player_present, enemy_present], dtype=torch.float32)  # length 3
            ], dim=0)  # total length 16

            grid[0, x, y, :] = grid_tile_features
    else:
        print("Warning: grid_tiles or grid_owner is None. Setting grid to zeros.")

    gamestate['grid'] = grid  # Shape: (1, 6, 3, 16)

    # 4. player_health and enemy_health
    player_health_tensor = torch.tensor([player_health], dtype=torch.float32) if player_health is not None else torch.tensor([1.0], dtype=torch.float32)
    enemy_health_tensor = torch.tensor([enemy_health], dtype=torch.float32) if enemy_health is not None else torch.tensor([1.0], dtype=torch.float32)
    gamestate['player_health'] = player_health_tensor  # Shape: (1,)
    gamestate['enemy_health'] = enemy_health_tensor    # Shape: (1,)

    # 5. player_chip and enemy_chip
    if player_chip is not None and 0 <= player_chip <= 400:
        player_chip_tensor = F.one_hot(torch.tensor(player_chip, dtype=torch.long), num_classes=401).float().unsqueeze(0)  # Shape: (1, 401)
    else:
        # print(f"Warning: Invalid player_chip {player_chip}. Setting to 400 to 1.")
        player_chip_tensor = F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float().unsqueeze(0)  # Shape: (1, 401)
    if enemy_chip is not None and 0 <= enemy_chip <= 400:
        enemy_chip_tensor = F.one_hot(torch.tensor(enemy_chip, dtype=torch.long), num_classes=401).float().unsqueeze(0)  # Shape: (1, 401)
    else:
        # print(f"Warning: Invalid enemy_chip {enemy_chip}. Setting to 400 to 1.")
        enemy_chip_tensor = F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float().unsqueeze(0)  # Shape: (1, 401)
        
    gamestate['player_chip'] = player_chip_tensor  # Shape: (1, 401)
    gamestate['enemy_chip'] = enemy_chip_tensor    # Shape: (1, 401)

    # 6. player_charge and enemy_charge
    #player and enemy charge come in as values from 0-2, we need to make it a single float from 0-1
    player_charge_tensor = torch.tensor([player_charge / 2.0], dtype=torch.float32) if player_charge is not None else torch.tensor([0.0], dtype=torch.float32)
    enemy_charge_tensor = torch.tensor([enemy_charge / 2.0], dtype=torch.float32) if enemy_charge is not None else torch.tensor([0.0], dtype=torch.float32)
    gamestate['player_charge'] = player_charge_tensor  # Shape: (1,)
    gamestate['enemy_charge'] = enemy_charge_tensor    # Shape: (1,)

    # 7. player_chip_hand
    def process_chip_hand(chip_hand, name='player_chip_hand'):
        """
        Processes the chip hand list by ensuring it has exactly 5 elements.
        Invalid chips are set to 400. Missing chips are filled with 400.

        Args:
            chip_hand (list of int): List of chip indices.
            name (str): Name of the chip hand for logging.

        Returns:
            list of torch.Tensor: List of 5 one-hot encoded tensors, each of shape (1, 401).
        """
        if chip_hand is not None:
            # Take the first 5 chips, fill the rest with 400
            chips = chip_hand[:5]
            if len(chips) < 5:
                chips += [400] * (5 - len(chips))
                # print(f"Warning: {name} has less than 5 elements. Filling missing slots with 400.")
            elif len(chips) > 5:
                # print(f"Warning: {name} has more than 5 elements. Truncating to first 5 elements.")
                chips = chips[:5]
        else:
            # If None, set all 5 chips to 400
            chips = [400] * 5
            # print(f"Warning: {name} is None. Setting all chips to 400.")

        chip_hand_tensors = []
        for chip in chips:
            if 0 <= chip <= 400:
                chip_onehot = F.one_hot(torch.tensor(chip, dtype=torch.long), num_classes=401).float()
            else:
                # print(f"Warning: Invalid {name} value {chip}. Setting to 400.")
                chip_onehot = F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float()
            chip_hand_tensors.append(chip_onehot.unsqueeze(0))  # Shape: (1, 401)
        
        return chip_hand_tensors  # List of 5 tensors, each (1, 401)

    # Process player_chip_hand
    gamestate['player_chip_hand'] = process_chip_hand(player_chip_hand, name='player_chip_hand')


    # 8. player_folder and enemy_folder
    def process_folder(folder_list, folder_name):
        folder_tensor = torch.zeros(1, 30 * 431, dtype=torch.float32)  # Each folder_chip has 431 features
        if folder_list is not None and len(folder_list) == 30:
            folder_features = []
            for chip in folder_list:
                # Process 'id'
                chip_id = chip.get('chip', 0)
                if 0 <= chip_id <= 400:
                    id_onehot = F.one_hot(torch.tensor(chip_id, dtype=torch.long), num_classes=401).float()  # Shape: (401,)
                else:
                    # print(f"Warning: Invalid {folder_name} chip id {chip_id}. Setting to 400 to 1.")
                    id_onehot = F.one_hot(torch.tensor(400, dtype=torch.long), num_classes=401).float()  # Shape: (401,)
                # Process 'code'
                chip_code = chip.get('code', 0)
                if 0 <= chip_code <= 26:
                    code_onehot = F.one_hot(torch.tensor(chip_code, dtype=torch.long), num_classes=27).float()  # Shape: (27,)
                else:
                    code_onehot = torch.zeros(27, dtype=torch.float32)
                    print(f"Warning: Invalid {folder_name} chip code {chip_code}. Setting to all zeros.")

                # Process 'used', 'regged', 'tagged'
                used = torch.tensor([float(chip.get('used', False))], dtype=torch.float32)      # Shape: (1,)
                regged = torch.tensor([float(chip.get('regged', False))], dtype=torch.float32)  # Shape: (1,)
                tagged = torch.tensor([float(chip.get('tagged', False))], dtype=torch.float32)  # Shape: (1,)

                # Combine all features
                folder_chip_features = torch.cat([id_onehot, code_onehot, used, regged, tagged], dim=0)  # Shape: (401 + 27 + 1 + 1 + 1 = 431,)
                folder_features.append(folder_chip_features)
            # Concatenate all 30 folder_chips
            folder_features = torch.cat(folder_features, dim=0)  # Shape: (30 * 431,)
            folder_tensor[0] = folder_features
        else:
            if folder_list is not None:
                print(f"Warning: {folder_name} does not have 30 folder_chips. Setting to all zeros.")
        return folder_tensor  # Shape: (1, 30 * 431)

    gamestate['player_folder'] = process_folder(player_folder, 'player_folder')  # Shape: (1, 30*431)
    gamestate['enemy_folder'] = process_folder(enemy_folder, 'enemy_folder')    # Shape: (1, 30*431)

    # 9. player_custom and enemy_custom
    def indices_to_one_hot(indices, size=200, name=''):
        """
        Converts a list of indices to a one-hot encoded tensor.

        Args:
            indices (list of int): List of indices to set to 1.
            size (int): Size of the one-hot tensor.
            name (str): Name for logging purposes.

        Returns:
            torch.Tensor: One-hot encoded tensor of shape (1, size).
        """
        tensor = torch.zeros(1, size, dtype=torch.float32)
        if indices is not None:
            for idx in indices:
                if 0 <= idx < size:
                    tensor[0, idx] = 1.0
                else:
                    print(f"Warning: {name} index {idx} out of range (0-{size-1})")
        return tensor

    # Convert player_custom and enemy_custom indices to one-hot tensors
    gamestate['player_custom'] = indices_to_one_hot(
        player_custom, size=200, name='player_custom'
    )  # Shape: (1, 200)
    gamestate['enemy_custom'] = indices_to_one_hot(
        enemy_custom, size=200, name='enemy_custom'
    )    # Shape: (1, 200)

    # 10. player_emotion_state and enemy_emotion_state
    if player_emotion_state is not None and 0 <= player_emotion_state <= 26:
        player_emotion_state_tensor = F.one_hot(torch.tensor(player_emotion_state, dtype=torch.long), num_classes=27).float().unsqueeze(0)  # Shape: (1, 27)
    else:
        player_emotion_state_tensor = torch.zeros(1, 27, dtype=torch.float32)
        if player_emotion_state is not None:
            print(f"Warning: Invalid player_emotion_state {player_emotion_state}. Setting to all zeros.")
    if enemy_emotion_state is not None and 0 <= enemy_emotion_state <= 26:
        enemy_emotion_state_tensor = F.one_hot(torch.tensor(enemy_emotion_state, dtype=torch.long), num_classes=27).float().unsqueeze(0)  # Shape: (1, 27)
    else:
        enemy_emotion_state_tensor = torch.zeros(1, 27, dtype=torch.float32)
        if enemy_emotion_state is not None:
            print(f"Warning: Invalid enemy_emotion_state {enemy_emotion_state}. Setting to all zeros.")

    gamestate['player_emotion_state'] = player_emotion_state_tensor  # Shape: (1, 27)
    gamestate['enemy_emotion_state'] = enemy_emotion_state_tensor    # Shape: (1, 27)

    # 11. player_used_crosses and enemy_used_crosses
    def process_crosses(crosses_list, crosses_name):
        crosses_tensor = torch.zeros(1, 10, dtype=torch.float32)  # Positions 1-10 mapped to indices 0-9
        if crosses_list is not None:
            for cross in crosses_list:
                if 1 <= cross <= 10:
                    crosses_tensor[0, cross - 1] = 1.0
                else:
                    print(f"Warning: Invalid {crosses_name} value {cross}. Must be between 1 and 10.")
        return crosses_tensor  # Shape: (1, 10)

    gamestate['player_used_crosses'] = process_crosses(player_used_crosses, 'player_used_crosses')  # Shape: (1, 10)
    gamestate['enemy_used_crosses'] = process_crosses(enemy_used_crosses, 'enemy_used_crosses')  # Shape: (1, 10)
    
    
    # 12. player_beasted_out and enemy_beasted_out
    player_beasted_out_tensor = torch.tensor([float(player_beasted_out)], dtype=torch.float32) if player_beasted_out is not None else torch.tensor([0.0], dtype=torch.float32)
    enemy_beasted_out_tensor = torch.tensor([float(enemy_beasted_out)], dtype=torch.float32) if enemy_beasted_out is not None else torch.tensor([0.0], dtype=torch.float32)
    gamestate['player_beasted_out'] = player_beasted_out_tensor  # Shape: (1,)
    gamestate['enemy_beasted_out'] = enemy_beasted_out_tensor    # Shape: (1,)

    # 13. player_beasted_over and enemy_beasted_over
    player_beasted_over_tensor = torch.tensor([float(player_beasted_over)], dtype=torch.float32) if player_beasted_over is not None else torch.tensor([0.0], dtype=torch.float32)
    enemy_beasted_over_tensor = torch.tensor([float(enemy_beasted_over)], dtype=torch.float32) if enemy_beasted_over is not None else torch.tensor([0.0], dtype=torch.float32)
    gamestate['player_beasted_over'] = player_beasted_over_tensor  # Shape: (1,)
    gamestate['enemy_beasted_over'] = enemy_beasted_over_tensor    # Shape: (1,)

    # Log shapes and types
    # for key, value in gamestate.items():
    #     if isinstance(value, list):
    #         print(f"{key}: List of {len(value)} tensors with shapes {[v.shape for v in value]}")
    #     elif value is not None:
    #         print(f"{key}: Tensor of shape {value.shape} and dtype {value.dtype}")
    #     else:
    #         print(f"{key}: None")

    return gamestate





class BattleDataset(Dataset):
    def __init__(self, gamestate_sequences, targets, memory=1):
        """
        Initializes the dataset with gamestate sequences and corresponding targets.

        Args:
            gamestate_sequences (list of list of dict): List where each element is a list of gamestates (dicts).
            targets (torch.Tensor): Tensor of shape (num_samples, 16) containing target labels.
            memory (int): Number of past gamestates to consider.
        """
        assert len(gamestate_sequences) == len(targets), "Number of samples and targets must match."
        self.gamestate_sequences = gamestate_sequences
        self.targets = targets
        self.memory = memory

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves the gamestate sequence and target for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (list of gamestates dicts, target tensor)
        """
        return self.gamestate_sequences[idx], self.targets[idx]


def test_battle_network_model():
    import h5py
    import torch
    import random

    # Define the file path
    file_path = '../TANGO/data/battle_data/Battle_Model_port_12345_20241003223718928205.h5'

    # Load the h5 file
    with h5py.File(file_path, 'r') as hf:
        # Extract datasets
        datasets = {}
        for key in hf.keys():
            datasets[key] = torch.tensor(hf[key][()], dtype=torch.float32)  # Convert to torch tensors

    # Initialize the model with memory=10
    memory = 10
    batch_size = 5
    model = BattleNetworkModel(image_option='None', memory=memory, scale=1.0, dropout_p=0.5)
    model.eval()  # Set the model to evaluation mode

    # Select random starting indices ensuring there are enough gamestates
    num_samples = datasets['cust_gage'].shape[0]
    if num_samples < memory:
        print(f"Not enough samples in the dataset to test with memory={memory}.")
        return

    # To ensure that each sequence has enough gamestates, select starting indices such that start_idx + memory <= num_samples
    max_start_idx = num_samples - memory
    if max_start_idx < batch_size:
        print(f"Not enough samples to create a batch of size {batch_size} with memory={memory}.")
        return

    # Select 5 unique starting indices
    # To allow overlapping sequences, you can allow duplicate starting indices by using random.randint multiple times
    # Here, we select unique starting indices
    start_indices = random.sample(range(0, max_start_idx), batch_size)
    print(f"Selected starting indices for {batch_size} sequences: {start_indices}")

    # Helper function to split flattened tensors
    def split_flattened_tensor(flat_tensor, num_chunks, chunk_size):
        """
        Splits a flattened tensor into a tuple of tensors.

        Args:
            flat_tensor (torch.Tensor): The flattened tensor of shape (batch_size, num_chunks * chunk_size).
            num_chunks (int): Number of chunks to split into.
            chunk_size (int): Size of each chunk.

        Returns:
            tuple of torch.Tensor: Tuple containing the split tensors.
        """
        return torch.chunk(flat_tensor, chunks=num_chunks, dim=1)

    # Helper function to process a single gamestate
    def load_gamestate(datasets, idx):
        gamestate = {
            'cust_gage': datasets['cust_gage'][idx].unsqueeze(0),  # Shape: (1,)
            'grid': datasets['grid'][idx].unsqueeze(0),  # Shape: (1, 6, 3, 16)
            'player_health': datasets['player_health'][idx].unsqueeze(0),  # Shape: (1,)
            'enemy_health': datasets['enemy_health'][idx].unsqueeze(0),  # Shape: (1,)
            'player_chip': datasets['player_chip'][idx].unsqueeze(0),  # Shape: (1, 401)
            'enemy_chip': datasets['enemy_chip'][idx].unsqueeze(0),  # Shape: (1, 401)
            'player_charge': datasets['player_charge'][idx].unsqueeze(0),  # Shape: (1,)
            'enemy_charge': datasets['enemy_charge'][idx].unsqueeze(0),  # Shape: (1,)
            # 'player_chip_hand' is a flattened tensor (1, 2005) → split into 5 tensors of (1, 401)
            'player_chip_hand': split_flattened_tensor(
                datasets['player_chip_hand'][idx].unsqueeze(0), num_chunks=5, chunk_size=401
            ),
            # 'player_folder' is a flattened tensor (1, 12930) → split into 30 tensors of (1, 431)
            'player_folder': split_flattened_tensor(
                datasets['player_folder'][idx].unsqueeze(0), num_chunks=30, chunk_size=431
            ),
            # 'enemy_folder' is a flattened tensor (1, 12930) → split into 30 tensors of (1, 431)
            'enemy_folder': split_flattened_tensor(
                datasets['enemy_folder'][idx].unsqueeze(0), num_chunks=30, chunk_size=431
            ),
            'player_custom': datasets['player_custom'][idx].unsqueeze(0),  # Shape: (1, 200)
            'enemy_custom': datasets['enemy_custom'][idx].unsqueeze(0),  # Shape: (1, 200)
            'player_emotion_state': datasets['player_emotion_state'][idx].unsqueeze(0),  # Shape: (1, 27)
            'enemy_emotion_state': datasets['enemy_emotion_state'][idx].unsqueeze(0),  # Shape: (1, 27)
            'player_used_crosses': datasets['player_used_crosses'][idx].unsqueeze(0),  # Shape: (1, 10)
            'enemy_used_crosses': datasets['enemy_used_crosses'][idx].unsqueeze(0),  # Shape: (1, 10)
            'player_beasted_out': datasets['player_beasted_out'][idx].unsqueeze(0),  # Shape: (1,)
            'enemy_beasted_out': datasets['enemy_beasted_out'][idx].unsqueeze(0),  # Shape: (1,)
            'player_beasted_over': datasets['player_beasted_over'][idx].unsqueeze(0),  # Shape: (1,)
            'enemy_beasted_over': datasets['enemy_beasted_over'][idx].unsqueeze(0)  # Shape: (1,)
        }
        return gamestate

    # Helper function to batch multiple gamestates
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
            if isinstance(gamestates_list[0][key], (list, tuple)):
                # For list-based fields like player_chip_hand, player_folder, enemy_folder
                # Use safe_cat to concatenate along the batch dimension
                batched_gamestate[key] = [safe_cat([g[key][i] for g in gamestates_list], dim=0) for i in range(len(gamestates_list[0][key]))]
            else:
                # For tensor fields, use safe_cat directly
                batched_gamestate[key] = safe_cat([g[key] for g in gamestates_list], dim=0)  # Shape: (batch_size, ...)
        return batched_gamestate

    # Load all sequences and batch them
    batch_gamestates_list = []  # List of gamestates per memory step (length=memory)
    for t in range(memory):
        current_gamestates = []
        for s in range(batch_size):
            current_idx = start_indices[s] + t
            gamestate = load_gamestate(datasets, current_idx)
            current_gamestates.append(gamestate)
        # Batch the current gamestates across the batch dimension
        batched_gamestate = batch_gamestates(current_gamestates)
        batch_gamestates_list.append(batched_gamestate)

    # **Debugging: Inspect Each Batched Gamestate in the Sequence**
    for t, batched_gamestate in enumerate(batch_gamestates_list):
        print(f"\nMemory Step {t + 1}:")
        print("Type of 'player_chip_hand':", type(batched_gamestate['player_chip_hand']))
        print("After splitting 'player_chip_hand':")
        for j, chip in enumerate(batched_gamestate['player_chip_hand']):
            print(f"  player_chip_hand[{j}].shape: {chip.shape}")  # Should be torch.Size([batch_size, 401])

        print("After splitting 'player_folder':")
        for j, folder in enumerate(batched_gamestate['player_folder']):
            print(f"  player_folder[{j}].shape: {folder.shape}")  # Should be torch.Size([batch_size, 431])
            if j < 3:  # Limit to first 3 for brevity
                print(f"  player_folder[{j}].shape: {folder.shape}")
        # Similarly, limit enemy_folder prints
        print("After splitting 'enemy_folder':")
        for j, folder in enumerate(batched_gamestate['enemy_folder']):
            print(f"  enemy_folder[{j}].shape: {folder.shape}")  # Should be torch.Size([batch_size, 431])
            if j < 3:  # Limit to first 3 for brevity
                print(f"  enemy_folder[{j}].shape: {folder.shape}")

    # **Validate Each Batched Gamestate in the Sequence**
    def validate_batched_gamestate(batched_gamestate, memory_step, batch_size):
        # Validate 'player_chip_hand'
        if not isinstance(batched_gamestate['player_chip_hand'], (list, tuple)):
            raise TypeError(f"Memory Step {memory_step}: player_chip_hand must be a list or tuple.")
        if len(batched_gamestate['player_chip_hand']) != 5:
            raise ValueError(f"Memory Step {memory_step}: player_chip_hand must contain 5 tensors.")
        for i, chip in enumerate(batched_gamestate['player_chip_hand']):
            if chip.dim() != 2 or chip.shape[0] != batch_size or chip.shape[1] != 401:
                raise ValueError(f"Memory Step {memory_step}: player_chip_hand[{i}] must have shape (batch_size, 401). Got {chip.shape}")

        # Validate 'player_folder' and 'enemy_folder'
        for folder_key in ['player_folder', 'enemy_folder']:
            folder = batched_gamestate[folder_key]
            if not isinstance(folder, (list, tuple)):
                raise TypeError(f"Memory Step {memory_step}: {folder_key} must be a list or tuple.")
            if len(folder) != 30:
                raise ValueError(f"Memory Step {memory_step}: {folder_key} must contain 30 tensors.")
            for i, chip in enumerate(folder):
                if chip.dim() != 2 or chip.shape[0] != batch_size or chip.shape[1] != 431:
                    raise ValueError(f"Memory Step {memory_step}: {folder_key}[{i}] must have shape (batch_size, 431). Got {chip.shape}")

        # Validate other tensor shapes as needed
        expected_shapes = {
            'cust_gage': (batch_size,),
            'grid': (batch_size, 6, 3, 16),
            'player_health': (batch_size,),
            'enemy_health': (batch_size,),
            'player_chip': (batch_size, 401),
            'enemy_chip': (batch_size, 401),
            'player_charge': (batch_size,),
            'enemy_charge': (batch_size,),
            'player_custom': (batch_size, 200),
            'enemy_custom': (batch_size, 200),
            'player_emotion_state': (batch_size, 27),
            'enemy_emotion_state': (batch_size, 27),
            'player_used_crosses': (batch_size, 10),
            'enemy_used_crosses': (batch_size, 10),
            'player_beasted_out': (batch_size,),
            'enemy_beasted_out': (batch_size,),
            'player_beasted_over': (batch_size,),
            'enemy_beasted_over': (batch_size,)
        }

        for key, shape in expected_shapes.items():
            tensor = batched_gamestate.get(key)
            if tensor is None:
                raise ValueError(f"Memory Step {memory_step}: Missing key: {key}")
            if tensor.shape != shape:
                raise ValueError(f"Memory Step {memory_step}: {key} has shape {tensor.shape}, expected {shape}.")

    # Validate all batched gamestates
    for t, batched_gamestate in enumerate(batch_gamestates_list):
        try:
            validate_batched_gamestate(batched_gamestate, memory_step=t + 1, batch_size=batch_size)
            print(f"\nMemory Step {t + 1} validation passed.")
        except (TypeError, ValueError) as e:
            print(f"\nValidation Error in Memory Step {t + 1}: {e}")
            return

    # **Prepare the Sequence for the Model**
    # The model expects a list of 'memory' gamestates, each containing batched data with batch_size=5
    # So, batch_gamestates_list is already a list of 10 batched gamestates
    print("\nBatch gamestates prepared.")

    # **Forward Pass Through the Model**
    print("\nPerforming model inference...")
    with torch.no_grad():
        output = model(batch_gamestates_list)

    print("\nModel output:", output)

# If this script is run directly, execute the test function
if __name__ == "__main__":
    test_battle_network_model()