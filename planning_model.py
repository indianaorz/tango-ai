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
]  # Total 11 classes


class PlanningModel(nn.Module):
    def __init__(self, hidden_size=64, dropout_rate=0.3):
        super(PlanningModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Folder chip encoder
        self.folder_chip_fc = nn.Linear(430, hidden_size)  # From one-hot encoded chip and code + flags
        
        # Visible chip encoder
        self.visible_chip_fc = nn.Linear(427, hidden_size)  # From one-hot encoded chip and code
        
        # Other feature encoders
        self.health_fc = nn.Linear(2, 16)  # Player and enemy health
        self.cross_fc = nn.Linear(52, 32)  # Player and enemy current cross (each 26 one-hot)
        self.available_crosses_fc = nn.Linear(60, 16)  # Player and enemy available crosses (each 5x6=30 flattened)
        self.beast_flags_fc = nn.Linear(4, 16)  # Beast out flags for player and enemy
        
        # Fully connected layers
        total_features = hidden_size * 3 + 16 + 32 + 16 + 16  # From processed inputs
        self.fc1 = nn.Linear(total_features, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64, 64)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output layers
        self.cross_output = nn.Linear(64, 6)  # 6 possible crosses including none
        self.chip_outputs = nn.ModuleList([nn.Linear(64, 11) for _ in range(5)])  # Five chip selections
    
    def forward(self, inputs):
        # Process player and enemy folders
        player_folder_embeds = self.process_folder(inputs['player_folder'])  # (batch_size, hidden_size)
        enemy_folder_embeds = self.process_folder(inputs['enemy_folder'])    # (batch_size, hidden_size)
        
        # Process visible chips
        visible_chips_embeds = self.process_visible_chips(inputs['visible_chips'])  # (batch_size, hidden_size)
        
        # Process other features
        health_feat = F.relu(self.health_fc(inputs['health']))  # (batch_size, 16)
        cross_feat = F.relu(self.cross_fc(inputs['current_crosses']))  # (batch_size, 32)
        available_crosses_feat = F.relu(self.available_crosses_fc(inputs['available_crosses']))  # (batch_size, 16)
        beast_flags_feat = F.relu(self.beast_flags_fc(inputs['beast_flags']))  # (batch_size, 16)
        
        # Concatenate all features
        x = torch.cat([
            player_folder_embeds, 
            enemy_folder_embeds, 
            visible_chips_embeds, 
            health_feat, 
            cross_feat, 
            available_crosses_feat, 
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
        cross_selection = self.cross_output(x)  # Shape: (batch_size, 26)
        chip_selections = [chip_output(x) for chip_output in self.chip_outputs]  # List of 5 tensors, each (batch_size, 10)
        
        return cross_selection, chip_selections
    
    def process_folder(self, folder):
        chips_onehot = folder['chips_onehot']  # (batch_size, 30, 400)
        codes_onehot = folder['codes_onehot']  # (batch_size, 30, 27)
        flags = folder['flags']                # (batch_size, 30, 3)
        
        x = torch.cat([chips_onehot, codes_onehot, flags], dim=-1)  # (batch_size, 30, 430)
        x = F.relu(self.folder_chip_fc(x))  # (batch_size, 30, hidden_size)
        x = x.mean(dim=1)  # Aggregate over the sequence dimension -> (batch_size, hidden_size)
        return x
    
    def process_visible_chips(self, visible_chips):
        chips_onehot = visible_chips['chips_onehot']  # (batch_size, 10, 400)
        codes_onehot = visible_chips['codes_onehot']  # (batch_size, 10, 27)
        
        x = torch.cat([chips_onehot, codes_onehot], dim=-1)  # (batch_size, 10, 427)
        x = F.relu(self.visible_chip_fc(x))  # (batch_size, 10, hidden_size)
        x = x.mean(dim=1)  # Aggregate over the sequence dimension -> (batch_size, hidden_size)
        return x


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
    """
    # Extract data from instance
    player_folder = instance['player_folder']
    enemy_folder = instance['enemy_folder']
    player_health = instance['player_health']
    enemy_health = instance['enemy_health']
    player_chips = instance['player_chips']
    chip_visible_count = instance['chip_visible_count']
    player_available_crosses = instance['player_available_crosses']
    enemy_available_crosses = instance['enemy_available_crosses']
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
    player_available_crosses_encoded = encode_available_crosses(player_available_crosses)  # Shape: (1,5,6)
    enemy_available_crosses_encoded = encode_available_crosses(enemy_available_crosses)    # Shape: (1,5,6)
    
   
    
    # Encode current crosses for player and enemy
    player_current_cross_encoded = encode_current_cross(player_cross)      # Shape: (1,26)
    enemy_current_cross_encoded = encode_current_cross(opponent_cross)     # Shape: (1,26)
    

    
    beast_flags_encoded = encode_beast_flags(player_beasted_out,enemy_beasted_out, player_beasted_over,enemy_beasted_over)  # Shape: (1,4)
    
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
        'current_crosses': torch.cat([player_current_cross_encoded, enemy_current_cross_encoded], dim=1),  # (1,52)
        'available_crosses': torch.cat([player_available_crosses_encoded.view(1, -1), enemy_available_crosses_encoded.view(1, -1)], dim=1),  # (1,60)
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
        # cross_selection_logits: (1,26)
        # chip_selection_logits: list of 5 tensors, each (1,10)
        
        # Apply softmaxbeast_out_selectable to cross_selection_logits to get probabilities
        cross_probabilities = F.softmax(cross_selection_logits, dim=-1)
        cross_selection = cross_probabilities.argmax(dim=-1).item()
        
        # Assign cross_target
        cross_target = cross_selection  # Value from 0 to 25
        
        # For chip selections, apply softmax and argmax for each selection
        target_list = []
        for chip_logits in chip_selection_logits:
            chip_prob = F.softmax(chip_logits, dim=-1)
            chip_selection = chip_prob.argmax(dim=-1).item()
            target_list.append(chip_selection)
        
        # Ensure the selected cross is within the range of available crosses
        available_crosses = len(instance['player_available_crosses']) - 1
        cross_target = min(cross_target, available_crosses)
        
        # Ensure the selected chips are within the range of visible chips
        chip_visible = instance['chip_visible_count']
        for i in range(len(target_list)):
            target_list[i] = min(target_list[i], chip_visible - 1)
        
        # Assign the selections to the instance
        instance['cross_target'] = cross_target -1
        instance['target_list'] = target_list
    
    print(f"Port {instance.get('port', 'N/A')}: Planning Output - cross_target: {cross_target}, target_list: {target_list}")
    
    # Epsilon-Greedy Strategy: With probability GAMMA, select a random action
    if random.random() < GAMMA:
        # Determine cross_target
        available_crosses_count = len(instance['player_available_crosses'])
        if available_crosses_count > 0:
            cross_target = random.randint(0, available_crosses_count - 1)
        else:
            cross_target = 0  # Default to 'None'
        
        # Determine target_list: sample up to 5 unique indices from visible chips
        chip_visible = instance['chip_visible_count']
        if chip_visible >= 5:
            target_list = random.sample(range(chip_visible), 5)
        else:
            target_list = random.sample(range(chip_visible), chip_visible)
        
        # Ensure the selected cross is within the range of available crosses
        cross_target = min(cross_target, available_crosses_count - 1)
        
        # Ensure the selected chips are within the range of visible chips
        for i in range(len(target_list)):
            target_list[i] = min(target_list[i], chip_visible - 1)
        
        # Assign the random selections to the instance
        instance['cross_target'] = cross_target - 1
        instance['target_list'] = target_list
        
        print(f"Port {instance.get('port', 'N/A')}: Gamma applied. Random cross_target: {cross_target}, target_list: {target_list}")
    
    return inputs, instance['cross_target'], instance['target_list']




def get_planning_input_from_replay(instance, GAMMA, device):
    """
    Prepares the input for the planning model based on the game instance,
    performs inference to select cross_target and target_list,
    and assigns them to the instance.
    
    Args:
        instance (dict): The game instance containing all necessary state information.
        GAMMA (float): Probability for the epsilon-greedy strategy.
        device (torch.device): The device to perform computations on.
    
    Modifies:
        instance (dict): Updates 'cross_target' and 'target_list' based on model inference.
    """
    # Extract data from instance
    player_folder = instance['player_folder']
    enemy_folder = instance['enemy_folder']
    player_health = instance['player_health']
    enemy_health = instance['enemy_health']
    player_chips = instance['player_chips']
    chip_visible_count = instance['chip_visible_count']
    player_available_crosses = instance['player_available_crosses']
    enemy_available_crosses = instance['enemy_available_crosses']
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
    player_available_crosses_encoded = encode_available_crosses(player_available_crosses)  # Shape: (1,5,6)
    enemy_available_crosses_encoded = encode_available_crosses(enemy_available_crosses)    # Shape: (1,5,6)
    
   
    
    # Encode current crosses for player and enemy
    player_current_cross_encoded = encode_current_cross(player_cross)      # Shape: (1,26)
    enemy_current_cross_encoded = encode_current_cross(opponent_cross)     # Shape: (1,26)
    

    
    beast_flags_encoded = encode_beast_flags(player_beasted_out,enemy_beasted_out, player_beasted_over,enemy_beasted_over)  # Shape: (1,4)
    
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
        'current_crosses': torch.cat([player_current_cross_encoded, enemy_current_cross_encoded], dim=1),  # (1,52)
        'available_crosses': torch.cat([player_available_crosses_encoded.view(1, -1), enemy_available_crosses_encoded.view(1, -1)], dim=1),  # (1,60)
        'beast_flags': beast_flags_encoded                           # (1,4)
    }
    
    # Perform inference with the planning model
    # inference_planning_model.eval()  # Ensure the model is in evaluation mode
    # with torch.no_grad():
    #     # Move inputs to device
    #     for key in inputs:
    #         if isinstance(inputs[key], dict):
    #             for subkey in inputs[key]:
    #                 inputs[key][subkey] = inputs[key][subkey].to(device)
    #         else:
    #             inputs[key] = inputs[key].to(device)
    #     # Forward pass through the model
    #     cross_selection_logits, chip_selection_logits = inference_planning_model(inputs)
    #     # cross_selection_logits: (1,26)
    #     # chip_selection_logits: list of 5 tensors, each (1,10)
        
    #     # Apply softmax to cross_selection_logits to get probabilities
    #     cross_probabilities = F.softmax(cross_selection_logits, dim=-1)
    #     cross_selection = cross_probabilities.argmax(dim=-1).item()
        
    #     # Assign cross_target
    #     cross_target = cross_selection  # Value from 0 to 25
        
    #     # For chip selections, apply softmax and argmax for each selection
    #     target_list = []
    #     for chip_logits in chip_selection_logits:
    #         chip_prob = F.softmax(chip_logits, dim=-1)
    #         chip_selection = chip_prob.argmax(dim=-1).item()
    #         target_list.append(chip_selection)
        
    #     # Ensure the selected cross is within the range of available crosses
    #     available_crosses = len(instance['player_available_crosses']) - 1
    #     cross_target = min(cross_target, available_crosses)
        
    #     # Ensure the selected chips are within the range of visible chips
    #     chip_visible = instance['chip_visible_count']
    #     for i in range(len(target_list)):
    #         target_list[i] = min(target_list[i], chip_visible - 1)
        
    #     # Assign the selections to the instance
    #     instance['cross_target'] = cross_target
    #     instance['target_list'] = target_list
    
    # print(f"Port {instance.get('port', 'N/A')}: Planning Output - cross_target: {cross_target}, target_list: {target_list}")
    
    # # Epsilon-Greedy Strategy: With probability GAMMA, select a random action
    # if random.random() < GAMMA:
    #     # Determine cross_target
    #     available_crosses_count = len(instance['player_available_crosses'])
    #     if available_crosses_count > 0:
    #         cross_target = random.randint(0, available_crosses_count - 1)
    #     else:
    #         cross_target = 0  # Default to 'None'
        
    #     # Determine target_list: sample up to 5 unique indices from visible chips
    #     chip_visible = instance['chip_visible_count']
    #     if chip_visible >= 5:
    #         target_list = random.sample(range(chip_visible), 5)
    #     else:
    #         target_list = random.sample(range(chip_visible), chip_visible)
        
    #     # Ensure the selected cross is within the range of available crosses
    #     cross_target = min(cross_target, available_crosses_count - 1)
        
    #     # Ensure the selected chips are within the range of visible chips
    #     for i in range(len(target_list)):
    #         target_list[i] = min(target_list[i], chip_visible - 1)
        
    #     # Assign the random selections to the instance
    #     instance['cross_target'] = cross_target
    #     instance['target_list'] = target_list
        
    #     print(f"Port {instance.get('port', 'N/A')}: Gamma applied. Random cross_target: {cross_target}, target_list: {target_list}")
    #add 1 to all selected_chip_indices
    selected_chip_indices = [x+1 for x in selected_chip_indices]
    
    #making sure that 0 equals to no chip selected
    
    
    #set all values greater than chip_selected_count to 0
    for i in range(len(selected_chip_indices)):
        if i >= chip_selected_count:
            selected_chip_indices[i] = 0
    
    current_player_emotion = instance['player_emotion']
    beast_out_selectable = instance['beast_out_selectable']
    selected_cross_index = -1#instance['selected_cross_index']
    if beast_out_selectable == 1:
        #if this is 1 then a cross was selected, so we should return the index of the selected cross
        selected_cross_index = instance['selected_cross_index']
        
    
    
    print(f"Port {instance.get('port', 'N/A')}: Replay Input - selected_cross_index: {selected_cross_index}, chip_selected_count: {chip_selected_count}, selected_chip_indices: {selected_chip_indices}, current_player_emotion: {current_player_emotion}")

    selected_cross_index += 1

    return inputs, selected_cross_index, selected_chip_indices



# Helper function to encode folders
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
    
    # Convert to tensors
    chip_ids_tensor = torch.tensor([chip_ids], dtype=torch.long, device=device)  # Shape: (1, 30)
    codes_tensor = torch.tensor([codes], dtype=torch.long, device=device)        # Shape: (1, 30)
    flags_tensor = torch.tensor([flags], dtype=torch.float32, device=device)     # Shape: (1, 30, 3)
    
    # One-hot encode chip_ids and codes
    chips_onehot = F.one_hot(chip_ids_tensor, num_classes=400).float()  # Shape: (1, 30, 400)
    codes_onehot = F.one_hot(codes_tensor, num_classes=27).float()      # Shape: (1, 30, 27)
    
    return {'chips_onehot': chips_onehot, 'codes_onehot': codes_onehot, 'flags': flags_tensor}
    # Helper function to encode visible chips
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
    
    chip_ids_tensor = torch.tensor([chip_ids], dtype=torch.long, device=device)  # Shape: (1, 10)
    codes_tensor = torch.tensor([codes], dtype=torch.long, device=device)        # Shape: (1, 10)
    
    # One-hot encode chip_ids and codes
    chips_onehot = F.one_hot(chip_ids_tensor, num_classes=400).float()  # Shape: (1, 10, 400)
    codes_onehot = F.one_hot(codes_tensor, num_classes=27).float()      # Shape: (1, 10, 27)
    
    return {'chips_onehot': chips_onehot, 'codes_onehot': codes_onehot}
# Helper function to encode available crosses
def encode_available_crosses(available_crosses):
    """
    Encodes available crosses into a tensor.
    
    Args:
        available_crosses (list): List of available cross types (strings).
    
    Returns:
        torch.Tensor: Tensor of shape (1,5,6), where each cross is one-hot encoded over 6 classes.
    """
    classes = ['None', 'Fire', 'Elec', 'Slash', 'Erase', 'Charge']  # 6 classes
    # Map the available crosses to indices
    cross_indices = []
    for cross in available_crosses:
        if cross in classes:
            index = classes.index(cross)
            cross_indices.append(index)
        else:
            cross_indices.append(0)  # If not found, default to 'None'
    
    # Pad if fewer than 5 crosses
    while len(cross_indices) < 5:
        cross_indices.append(0)  # 'None'
    
    cross_indices_tensor = torch.tensor([cross_indices], dtype=torch.long, device=device)  # Shape: (1,5)
    crosses_onehot = F.one_hot(cross_indices_tensor, num_classes=6).float()  # Shape: (1,5,6)
    
    return crosses_onehot  # Shape: (1,5,6)
# Helper function to encode current crosses
def encode_current_cross(cross):
    """
    Encodes the current cross into a one-hot tensor.
    
    Args:
        cross (int): Current cross index (0-25).
    
    Returns:
        torch.Tensor: One-hot encoded tensor for the current cross, shape (1,26)
    """
    cross_tensor = torch.tensor([[cross]], dtype=torch.long, device=device)  # Shape: (1,1)
    cross_onehot = F.one_hot(cross_tensor, num_classes=26).float()  # Shape: (1,1,26)
    cross_onehot = cross_onehot.squeeze(1)  # Shape: (1,26)
    return cross_onehot  # Shape: (1,26)
# Encode beast flags for player and enemy
def encode_beast_flags(player_beasted_out, enemy_beasted_out, player_beasted_over, enemy_beasted_over):
    """
    Encodes beast flags into a tensor.
    
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
