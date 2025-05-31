# planning_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated list of crosses in the specified order
CROSS_CLASSES = [
    "None",
    "Fire",
    "Elec",
    "Slash",
    "Erase",
    "Charge",
    "Aqua",
    "Thawk",
    "Tengu",
    "Grnd",
    "Dust",
    # Additional classes can be added here if needed
]  # Total 11 classes, but FORM_MAPPING includes up to 22 forms
class PlanningModel(nn.Module):
    def __init__(self, hidden_size=64, dropout_rate=0.3, scale=10):
        """
        Initializes the PlanningModel with the ability to scale its hidden layers.

        Args:
            hidden_size (int): Base size for hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            scale (float): Scaling factor to increase/decrease model size.
        """
        super(PlanningModel, self).__init__()
        
        # Apply scaling to hidden_size and other layer dimensions
        scaled_hidden_size = int(hidden_size * scale)
        scaled_health_size = int(16 * scale)
        scaled_cross_size = int(32 * scale)
        scaled_used_crosses_size = int(16 * scale)
        scaled_beast_flags_size = int(16 * scale)
        scaled_fc1_size = int(128 * scale)
        scaled_fc2_size = int(128 * scale)
        scaled_fc3_size = int(64 * scale)
        scaled_fc4_size = int(64 * scale)
        scaled_output_size = int(64 * scale)  # For cross_output and chip_outputs

        self.hidden_size = scaled_hidden_size
        
        # Folder chip encoder
        self.folder_chip_fc = nn.Linear(430, scaled_hidden_size)  # From one-hot encoded chip and code + flags
        
        # Visible chip encoder
        self.visible_chip_fc = nn.Linear(427, scaled_hidden_size)  # From one-hot encoded chip and code
        
        # Other feature encoders
        self.health_fc = nn.Linear(2, scaled_health_size)  # Player and enemy health
        self.cross_fc = nn.Linear(52, scaled_cross_size)  # Player and enemy current cross (each 26 one-hot)
        self.used_crosses_fc = nn.Linear(20, scaled_used_crosses_size)  # Player and enemy available crosses (each 10 bits, total 20)
        self.beast_flags_fc = nn.Linear(4, scaled_beast_flags_size)  # Beast out flags for player and enemy
        
        # Fully connected layers
        total_features = (
            scaled_hidden_size * 3 + 
            scaled_health_size + 
            scaled_cross_size + 
            scaled_used_crosses_size + 
            scaled_beast_flags_size
        )  # From processed inputs
        self.fc1 = nn.Linear(total_features, scaled_fc1_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(scaled_fc1_size, scaled_fc2_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(scaled_fc2_size, scaled_fc3_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(scaled_fc3_size, scaled_fc4_size)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output layers
        self.cross_output = nn.Linear(scaled_fc4_size, 6)  # 6 possible crosses including 'None' and 5 available crosses
        self.chip_outputs = nn.ModuleList([nn.Linear(scaled_fc4_size, 12) for _ in range(5)])  # Five chip selections (12 classes each)
    
    def forward(self, inputs):
        # Process player and enemy folders
        player_folder_embeds = self.process_folder(inputs['player_folder'])  # (batch_size, hidden_size)
        enemy_folder_embeds = self.process_folder(inputs['enemy_folder'])    # (batch_size, hidden_size)
        
        # Process visible chips
        visible_chips_embeds = self.process_visible_chips(inputs['visible_chips'])  # (batch_size, hidden_size)
        
        # Process other features
        health_feat = F.relu(self.health_fc(inputs['health']))  # (batch_size, scaled_health_size)
        cross_feat = F.relu(self.cross_fc(inputs['current_crosses']))  # (batch_size, scaled_cross_size)
        used_crosses_feat = F.relu(self.used_crosses_fc(inputs['used_crosses']))  # (batch_size, scaled_used_crosses_size)
        beast_flags_feat = F.relu(self.beast_flags_fc(inputs['beast_flags']))  # (batch_size, scaled_beast_flags_size)
        
        # Concatenate all features
        x = torch.cat([
            player_folder_embeds, 
            enemy_folder_embeds, 
            visible_chips_embeds, 
            health_feat, 
            cross_feat, 
            used_crosses_feat, 
            beast_flags_feat
        ], dim=-1)
        
        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        # Outputs
        cross_selection = self.cross_output(x)  # Shape: (batch_size, 6)
        chip_selections = [chip_output(x) for chip_output in self.chip_outputs]  # List of 5 tensors, each (batch_size, 12)
        
        return cross_selection, chip_selections
    
    def process_folder(self, folder):
        """
        Processes the folder information for either player or enemy.
        
        Args:
            folder (dict): Contains 'chips_onehot', 'codes_onehot', and 'flags'.
        
        Returns:
            torch.Tensor: Embedded representation of the folder, shape (batch_size, hidden_size)
        """
        chips_onehot = folder['chips_onehot']  # (batch_size, 30, 400)
        codes_onehot = folder['codes_onehot']  # (batch_size, 30, 27)
        flags = folder['flags']                # (batch_size, 30, 3)
        
        x = torch.cat([chips_onehot, codes_onehot, flags], dim=-1)  # (batch_size, 30, 430)
        x = F.relu(self.folder_chip_fc(x))  # (batch_size, 30, hidden_size)
        x = x.mean(dim=1)  # Aggregate over the sequence dimension -> (batch_size, hidden_size)
        return x
    
    def process_visible_chips(self, visible_chips):
        """
        Processes the visible chips information.
        
        Args:
            visible_chips (dict): Contains 'chips_onehot' and 'codes_onehot'.
        
        Returns:
            torch.Tensor: Embedded representation of visible chips, shape (batch_size, hidden_size)
        """
        chips_onehot = visible_chips['chips_onehot']  # (batch_size, 10, 400)
        codes_onehot = visible_chips['codes_onehot']  # (batch_size, 10, 27)
        
        x = torch.cat([chips_onehot, codes_onehot], dim=-1)  # (batch_size, 10, 427)
        x = F.relu(self.visible_chip_fc(x))  # (batch_size, 10, hidden_size)
        x = x.mean(dim=1)  # Aggregate over the sequence dimension -> (batch_size, hidden_size)
        return x

def validate_chip_ids(chip_ids, max_chip_id=399):
    """
    Clamps chip IDs to be within [0, max_chip_id].

    Args:
        chip_ids (list): List of chip IDs.
        max_chip_id (int): Maximum allowed chip ID.

    Returns:
        list: Validated chip IDs.
    """
    chip_ids = [min(max(chip_id, 0), max_chip_id) for chip_id in chip_ids]
    return chip_ids

def validate_code_ids(code_ids, max_code_id=26):
    """
    Clamps code IDs to be within [0, max_code_id].

    Args:
        code_ids (list): List of code IDs.
        max_code_id (int): Maximum allowed code ID.

    Returns:
        list: Validated code IDs.
    """
    code_ids = [min(max(code_id, 0), max_code_id) for code_id in code_ids]
    return code_ids

def encode_folder(folder):
    """
    Encodes the player or enemy folder into tensors.

    Args:
        folder (list): List of dictionaries containing chip information.

    Returns:
        dict: Encoded tensors for chips_onehot, codes_onehot, and flags.
    """
    chip_ids = []
    codes = []
    flags = []
    for chip in folder:
        chip_ids.append(chip['chip'])  # Integer ID (0-399)
        codes.append(chip['code'])     # Integer ID (0-26)
        flags.append([int(chip['used']), int(chip['tagged']), int(chip['regged'])])
    
    # Pad if fewer than 30 chips
    while len(chip_ids) < 30:
        chip_ids.append(0)
        codes.append(0)
        flags.append([0, 0, 0])
    
    # Validate chip_ids and codes
    chip_ids = validate_chip_ids(chip_ids)
    codes = validate_code_ids(codes)
    
    # Convert to tensors
    chip_ids_tensor = torch.tensor([chip_ids], dtype=torch.long, device=device)  # Shape: (1, 30)
    codes_tensor = torch.tensor([codes], dtype=torch.long, device=device)        # Shape: (1, 30)
    flags_tensor = torch.tensor([flags], dtype=torch.float32, device=device)     # Shape: (1, 30, 3)
    
    # One-hot encode chip_ids and codes
    chips_onehot = F.one_hot(chip_ids_tensor, num_classes=400).float()  # Shape: (1, 30, 400)
    codes_onehot = F.one_hot(codes_tensor, num_classes=27).float()      # Shape: (1, 30, 27)
    
    return {'chips_onehot': chips_onehot, 'codes_onehot': codes_onehot, 'flags': flags_tensor}

def encode_visible_chips(chips, visible_count):
    """
    Encodes the visible chips into tensors.

    Args:
        chips (list): List of tuples containing (chip_id, code_id).
        visible_count (int): Number of visible chips (up to 10).

    Returns:
        dict: Encoded tensors for chips_onehot and codes_onehot.
    """
    chip_ids = []
    codes = []
    for chip in chips[:visible_count]:
        chip_ids.append(chip[0])  # chip_id
        codes.append(chip[1])     # code_id
    
    # Pad if fewer than 10 visible chips
    while len(chip_ids) < 10:
        chip_ids.append(0)
        codes.append(0)
    
    # Validate chip_ids and codes
    chip_ids = validate_chip_ids(chip_ids)
    codes = validate_code_ids(codes)
    
    chip_ids_tensor = torch.tensor([chip_ids], dtype=torch.long, device=device)  # Shape: (1, 10)
    codes_tensor = torch.tensor([codes], dtype=torch.long, device=device)        # Shape: (1, 10)
    
    # One-hot encode chip_ids and codes
    chips_onehot = F.one_hot(chip_ids_tensor, num_classes=400).float()  # Shape: (1, 10, 400)
    codes_onehot = F.one_hot(codes_tensor, num_classes=27).float()      # Shape: (1, 10, 27)
    
    return {'chips_onehot': chips_onehot, 'codes_onehot': codes_onehot}

def encode_used_crosses(used_crosses):
    """
    Encodes available crosses into a 10-bit binary tensor.

    Args:
        used_crosses (list): List of available cross indices (integers from 0 to 9).

    Returns:
        torch.Tensor: Binary tensor of shape (1,10), where each bit represents the availability of a cross.
    """
    num_crosses = 10
    binary_vector = [0] * num_crosses
    for cross_index in used_crosses:
        if 0 <= cross_index < num_crosses:
            binary_vector[cross_index] = 1
        else:
            print(f"Warning: Cross index {cross_index} is out of bounds [0, {num_crosses-1}]. Ignored.")
    
    binary_tensor = torch.tensor([binary_vector], dtype=torch.float32, device=device)  # Shape: (1,10)
    return binary_tensor  # Shape: (1,10)


def encode_current_cross(cross):
    """
    Encodes the current cross into a one-hot tensor.

    Args:
        cross (int): Current cross index (0-25).

    Returns:
        torch.Tensor: One-hot encoded tensor for the current cross, shape (1,26)
    """
    num_classes = 26  # 0-25
    # Validate cross index
    cross = min(max(cross, 0), num_classes - 1)
    
    cross_tensor = torch.tensor([[cross]], dtype=torch.long, device=device)  # Shape: (1,1)
    cross_onehot = F.one_hot(cross_tensor, num_classes=num_classes).float()  # Shape: (1,1,26)
    cross_onehot = cross_onehot.squeeze(1)  # Shape: (1,26)
    return cross_onehot  # Shape: (1,26)

def encode_beast_flags(player_beasted_out, enemy_beasted_out, player_beasted_over, enemy_beasted_over):
    """
    Encodes beast flags into a tensor.

    Args:
        player_beasted_out (bool/int): Beast out flag for the player.
        enemy_beasted_out (bool/int): Beast out flag for the enemy.
        player_beasted_over (bool/int): Beast over flag for the player.
        enemy_beasted_over (bool/int): Beast over flag for the enemy.

    Returns:
        torch.Tensor: Tensor of shape (1,4)
    """
    flags = [
        int(player_beasted_out),
        int(enemy_beasted_out),
        int(player_beasted_over),
        int(enemy_beasted_over)
    ]
    flags_tensor = torch.tensor([flags], dtype=torch.float32, device=device)  # Shape: (1,4)
    return flags_tensor

def get_planning_input_from_instance(inference_planning_model, instance, GAMMA, device):
    """
    Prepares the input for the planning model based on the game instance,
    performs inference to select cross_target and target_list,
    and assigns them to the instance.

    Args:
        inference_planning_model (nn.Module): The trained PlanningModel.
        instance (dict): The game instance containing all necessary state information.
        GAMMA (float): Probability for the epsilon-greedy strategy.
        device (torch.device): The device to perform computations on.

    Modifies:
        instance (dict): Updates 'cross_target' and 'target_list' based on model inference.

    Returns:
        tuple: (inputs, cross_target, target_list)
    """
    # Extract data from instance
    player_folder = instance['player_folder']
    enemy_folder = instance['enemy_folder']
    player_health = instance['player_health']
    enemy_health = instance['enemy_health']
    player_chips = instance['player_chips']
    chip_visible_count = instance['chip_visible_count']
    player_used_crosses = instance['player_used_crosses']  # List of up to 5 indices [0-4]
    enemy_used_crosses = instance['enemy_used_crosses']  # List of up to 5 indices [0-4]
    player_cross = instance['player_game_emotion']
    opponent_cross = instance['enemy_game_emotion']
    player_beasted_out = instance['player_beasted_out']
    enemy_beasted_out = instance['enemy_beasted_out']
    player_beasted_over = instance['player_beasted_over']
    enemy_beasted_over = instance['enemy_beasted_over']
    
    instance['cross_target'] = -1

    # Encode player and enemy folders
    player_folder_encoded = encode_folder(player_folder)
    enemy_folder_encoded = encode_folder(enemy_folder)

    # Encode visible chips
    visible_chips_encoded = encode_visible_chips(player_chips, chip_visible_count)

    # Encode health
    health_tensor = torch.tensor([[player_health, enemy_health]], dtype=torch.float32, device=device)  # Shape: (1, 2)

    # Encode available crosses for player and enemy
    player_used_crosses_encoded = encode_used_crosses(player_used_crosses)  # Shape: (1,10)
    enemy_used_crosses_encoded = encode_used_crosses(enemy_used_crosses)    # Shape: (1,10)

    # Concatenate player and enemy available crosses to form a 20-bit vector
    used_crosses_tensor = torch.cat([player_used_crosses_encoded, enemy_used_crosses_encoded], dim=1)  # Shape: (1,20)

    # Encode current crosses for player and enemy
    player_current_cross_encoded = encode_current_cross(player_cross)      # Shape: (1,26)
    enemy_current_cross_encoded = encode_current_cross(opponent_cross)   # Shape: (1,26)
    
    # Concatenate player and enemy current crosses to form a 52-bit vector
    current_crosses_tensor = torch.cat([player_current_cross_encoded, enemy_current_cross_encoded], dim=1)  # Shape: (1,52)

    # Encode beast flags
    beast_flags_encoded = encode_beast_flags(player_beasted_out, enemy_beasted_out, player_beasted_over, enemy_beasted_over)  # Shape: (1,4)

    # Assemble all inputs into a dictionary as expected by the model
    inputs = {
        'player_folder': {
            'chips_onehot': player_folder_encoded['chips_onehot'],  # (1,30,400)
            'codes_onehot': player_folder_encoded['codes_onehot'],  # (1,30,27)
            'flags': player_folder_encoded['flags']                 # (1,30,3)
        },
        'enemy_folder': {
            'chips_onehot': enemy_folder_encoded['chips_onehot'],    # (1,30,400)
            'codes_onehot': enemy_folder_encoded['codes_onehot'],    # (1,30,27)
            'flags': enemy_folder_encoded['flags']                   # (1,30,3)
        },
        'visible_chips': {
            'chips_onehot': visible_chips_encoded['chips_onehot'],   # (1,10,400)
            'codes_onehot': visible_chips_encoded['codes_onehot']    # (1,10,27)
        },
        'health': health_tensor,                                     # (1,2)
        'current_crosses': current_crosses_tensor,                   # (1,52)
        'used_crosses': used_crosses_tensor,               # (1,10)
        'beast_flags': beast_flags_encoded                           # (1,4)
    }
    
    # Perform inference with the planning model
    inference_planning_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], dict):
                for subkey in inputs[key]:
                    inputs[key][subkey] = inputs[key][subkey].to(device)
            else:
                inputs[key] = inputs[key].to(device)
        # Forward pass through the model
        cross_selection_logits, chip_selection_logits = inference_planning_model(inputs)
        # cross_selection_logits: (1,6)
        # chip_selection_logits: list of 5 tensors, each (1,12)
        
        # Apply softmax to cross_selection_logits to get probabilities
        cross_probabilities = F.softmax(cross_selection_logits, dim=-1)
        cross_selection = cross_probabilities.argmax(dim=-1).item()  # Value from 0 to 5
        
        #log cross_probabilities
        # print(f"Port {instance.get('port', 'N/A')}: Planning Output - cross_probabilities: {cross_probabilities}")
        
        
        # Assign cross_target
        cross_target = cross_selection  # Value from 0 to 5
        
        # For chip selections, apply softmax and argmax for each selection
        target_list = []
        for chip_logits in chip_selection_logits:
            chip_prob = F.softmax(chip_logits, dim=-1)
            chip_selection = chip_prob.argmax(dim=-1).item()  # Value from 0 to 11
            target_list.append(chip_selection)
            
            #log chip selection
            # print(f"Port {instance.get('port', 'N/A')}: Planning Output - chip_selection_logits: {chip_prob}")
        
        # Ensure the selected cross is within the range of available crosses
        # 'None' is considered as the first cross (index 0), so used_crosses_count = len - 1
        used_crosses = len(instance['player_used_crosses'])  # Number of available crosses
        print(f"-------------------{len(instance['player_used_crosses'])}:")
        cross_target = min(cross_target, used_crosses)  # Ensures cross_target does not exceed available crosses
        
        # Ensure the selected chips are within the range of visible chips
        chip_visible = instance['chip_visible_count']
        for i in range(len(target_list)):
            target_list[i] = min(target_list[i], chip_visible - 1)
        
        # Assign the selections to the instance
        instance['cross_target'] = cross_target - 1  # Adjusting as per original logic
        instance['target_list'] = target_list
    
    # print(f"Port {instance.get('port', 'N/A')}: Planning Output - cross_target: {cross_target}, target_list: {target_list}")
    
    # Epsilon-Greedy Strategy: With probability GAMMA, select a random action
    if random.random() < GAMMA:
        # Determine cross_target
        used_crosses_count = len(instance['player_used_crosses'])
        if used_crosses_count > 0:
            cross_target = random.randint(0, used_crosses_count - 1)
        else:
            cross_target = 0  # Default to 'None'
        
        # Determine target_list: sample up to 5 unique indices from visible chips
        chip_visible = instance['chip_visible_count']
        if chip_visible >= 5:
            target_list = random.sample(range(chip_visible), 5)
        else:
            target_list = random.sample(range(chip_visible), chip_visible)
        
        # Ensure the selected cross is within the range of available crosses
        cross_target = min(cross_target, used_crosses_count - 1)
        
        # Ensure the selected chips are within the range of visible chips
        for i in range(len(target_list)):
            target_list[i] = min(target_list[i], chip_visible - 1)
        
        # Assign the random selections to the instance
        instance['cross_target'] = cross_target - 1  # Adjusting as per original logic
        instance['target_list'] = target_list
        
        print(f"Port {instance.get('port', 'N/A')}: Gamma applied. Random cross_target: {cross_target}, target_list: {target_list}")
    
    return inputs, instance['cross_target'], instance['target_list']

def get_planning_input_from_replay(instance, GAMMA, device):
    """
    Prepares the input for the planning model based on the game instance,
    processes selected actions from replay data, and assigns them to the instance.

    Args:
        instance (dict): The game instance containing all necessary state information.
        GAMMA (float): Probability for the epsilon-greedy strategy.
        device (torch.device): The device to perform computations on.

    Modifies:
        instance (dict): Updates 'cross_target' and 'target_list' based on replay data.

    Returns:
        tuple: (inputs, selected_cross_index, selected_chip_indices)
    """
    # Extract data from instance
    player_folder = instance['player_folder']
    enemy_folder = instance['enemy_folder']
    player_health = instance['player_health']
    enemy_health = instance['enemy_health']
    player_chips = instance['player_chips']
    chip_visible_count = instance['chip_visible_count']
    player_used_crosses = instance['player_used_crosses']  # List of up to 5 indices [0-4]
    enemy_used_crosses = instance['enemy_used_crosses']  # List of up to 5 indices [0-4]
    player_cross = instance['player_game_emotion']
    opponent_cross = instance['enemy_game_emotion']
    player_beasted_out = instance['player_beasted_out']
    enemy_beasted_out = instance['enemy_beasted_out']
    player_beasted_over = instance['player_beasted_over']
    enemy_beasted_over = instance['enemy_beasted_over']
    
    selected_cross_index = instance['selected_cross_index']
    chip_selected_count = instance['chip_selected_count']
    selected_chip_indices = instance['selected_chip_indices']

    # Encode player and enemy folders
    player_folder_encoded = encode_folder(player_folder)
    enemy_folder_encoded = encode_folder(enemy_folder)

    # Encode visible chips
    visible_chips_encoded = encode_visible_chips(player_chips, chip_visible_count)

    # Encode health
    health_tensor = torch.tensor([[player_health, enemy_health]], dtype=torch.float32, device=device)  # Shape: (1, 2)

    # Encode available crosses for player and enemy
    player_used_crosses_encoded = encode_used_crosses(player_used_crosses)  # Shape: (1,5)
    enemy_used_crosses_encoded = encode_used_crosses(enemy_used_crosses)    # Shape: (1,5)
    
    # Concatenate player and enemy available crosses to form a 10-bit vector
    used_crosses_tensor = torch.cat([player_used_crosses_encoded, enemy_used_crosses_encoded], dim=1)  # Shape: (1,10)

    # Encode current crosses for player and enemy
    player_current_cross_encoded = encode_current_cross(player_cross)      # Shape: (1,26)
    enemy_current_cross_encoded = encode_current_cross(opponent_cross)   # Shape: (1,26)
    
    # Concatenate player and enemy current crosses to form a 52-bit vector
    current_crosses_tensor = torch.cat([player_current_cross_encoded, enemy_current_cross_encoded], dim=1)  # Shape: (1,52)

    # Encode beast flags
    beast_flags_encoded = encode_beast_flags(player_beasted_out, enemy_beasted_out, player_beasted_over, enemy_beasted_over)  # Shape: (1,4)

    # Assemble all inputs into a dictionary as expected by the model
    inputs = {
        'player_folder': {
            'chips_onehot': player_folder_encoded['chips_onehot'],  # (1,30,400)
            'codes_onehot': player_folder_encoded['codes_onehot'],  # (1,30,27)
            'flags': player_folder_encoded['flags']                 # (1,30,3)
        },
        'enemy_folder': {
            'chips_onehot': enemy_folder_encoded['chips_onehot'],    # (1,30,400)
            'codes_onehot': enemy_folder_encoded['codes_onehot'],    # (1,30,27)
            'flags': enemy_folder_encoded['flags']                   # (1,30,3)
        },
        'visible_chips': {
            'chips_onehot': visible_chips_encoded['chips_onehot'],   # (1,10,400)
            'codes_onehot': visible_chips_encoded['codes_onehot']    # (1,10,27)
        },
        'health': health_tensor,                                     # (1,2)
        'current_crosses': current_crosses_tensor,                   # (1,52)
        'used_crosses': used_crosses_tensor,               # (1,10)
        'beast_flags': beast_flags_encoded                           # (1,4)
    }
    
    # Add 1 to all selected_chip_indices
    selected_chip_indices = [x + 1 for x in selected_chip_indices]
    
    # Making sure that 0 equals to no chip selected
    # Set all values greater than chip_selected_count to 0
    for i in range(len(selected_chip_indices)):
        if i >= chip_selected_count:
            selected_chip_indices[i] = 0
    
    current_player_emotion = instance['player_emotion']
    beast_out_selectable = instance['beast_out_selectable']
    selected_cross_index = -1  # Initialize
    
    if beast_out_selectable == 1:
        # If this is 1 then a cross was selected, so we should return the index of the selected cross
        selected_cross_index = instance['selected_cross_index']
    
    print(f"Port {instance.get('port', 'N/A')}: Replay Input - selected_cross_index: {selected_cross_index}, chip_selected_count: {chip_selected_count}, selected_chip_indices: {selected_chip_indices}, current_player_emotion: {current_player_emotion}")

    selected_cross_index += 1  # Adjust as per original logic

    # Ensure selected_cross_index has 6 classes
    selected_cross_index = min(selected_cross_index, 5)  # 0 to 5

    # Ensure selected_chip_indices are within 0 to 11
    for i in range(len(selected_chip_indices)):
        selected_chip_indices[i] = min(selected_chip_indices[i], 11)
    
    # Assign selected_cross_index and selected_chip_indices to the instance
    instance['cross_target'] = selected_cross_index - 1  # Adjusting as per original logic
    if instance['cross_target'] < 0:
        instance['cross_target'] = 0
    instance['target_list'] = selected_chip_indices
    
    # # Epsilon-Greedy Strategy: With probability GAMMA, select a random action
    # if random.random() < GAMMA:
    #     # Determine cross_target
    #     used_crosses_count = len(instance['player_used_crosses'])
    #     if used_crosses_count > 0:
    #         cross_target = random.randint(0, used_crosses_count - 1)
    #     else:
    #         cross_target = 0  # Default to 'None'
        
    #     # Determine target_list: sample up to 5 unique indices from visible chips
    #     chip_visible = instance['chip_visible_count']
    #     if chip_visible >= 5:
    #         target_list = random.sample(range(chip_visible), 5)
    #     else:
    #         target_list = random.sample(range(chip_visible), chip_visible)
        
    #     # Ensure the selected cross is within the range of available crosses
    #     cross_target = min(cross_target, used_crosses_count - 1)
        
    #     # Ensure the selected chips are within the range of visible chips
    #     for i in range(len(target_list)):
    #         target_list[i] = min(target_list[i], chip_visible - 1)
        
    #     # Assign the random selections to the instance
    #     instance['cross_target'] = cross_target - 1  # Adjusting as per original logic
    #     if instance['cross_target'] < 0:
    #         instance['cross_target'] = 0
    #     instance['target_list'] = target_list
        
    #     print(f"Port {instance.get('port', 'N/A')}: Gamma applied. Random cross_target: {cross_target}, target_list: {target_list}")
    
    return inputs, instance['cross_target'], instance['target_list']


