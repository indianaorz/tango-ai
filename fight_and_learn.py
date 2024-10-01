# online_learn_battle.py
import traceback

import subprocess
import os
import time
import asyncio
import json
import base64
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import deque  # For frame buffering and data storage
import yaml  # To load config
import random
from threading import Lock
import pyautogui
import subprocess
import glob
from train import GameInputPredictor  # Import the model class
from utils import (
    get_image_memory, get_exponential_sample,
    get_exponental_amount, get_threshold, get_root_dir, position_to_grid, get_threshold_plan, inference_fps,get_latest_checkpoint , get_checkpoint_dir,extract_number_from_checkpoint
)

import threading
import queue

print("Saving data points to disk.")
planning_data_dir = get_root_dir() + "/data/planning_data"
os.makedirs(planning_data_dir, exist_ok=True)
battle_data_dir = get_root_dir() + "/data/battle_data"
os.makedirs(battle_data_dir, exist_ok=True)

# Import necessary modules at the top
import torch.optim as optim
import torch.nn.functional as F


# import wandb

import time
from collections import defaultdict

# Initialize global dictionaries for tracking per-instance state
window_entry_time = {}
previous_sent_dict = defaultdict(int)
previous_inside_window_dict = defaultdict(float)

# Global dictionaries for Planning Model

current_round_damage = defaultdict(float)  # Tracks damage dealt in the current round per port
max_round_damage = defaultdict(float)  # Tracks max damage dealt in the current round

final_health = defaultdict(lambda: (0.0, 0.0))  # {port: (player_health, enemy_health)}


# Initialize locks for thread safety
# Removed existing planning_model_lock and battle_model_lock as we'll have separate training locks
training_planning_lock = Lock()
training_battle_lock = Lock()

inference_planning_lock = Lock()
inference_battle_lock = Lock()

latest_checkpoint_number = {'planning': 0, 'battle': 0}

training_queue = queue.Queue()

# Timer duration in seconds (can be set via environment variable or config)
CHIP_WINDOW_TIMER = float(os.getenv("CHIP_WINDOW_TIMER", 10.0))  # Default is 5 seconds
REPLAYS_DIR = '/home/lee/Documents/Tango/replaysOrig'



# Initialize locks for thread safety
# planning_model_lock = Lock()
# battle_model_lock = Lock()
MAX_CHECKPOINTS = 5
force_skip_chip_window = True
# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "valuesearch"
env_common["AI_MODEL_PATH"] = "ai_model"
GAMMA = float(os.getenv("GAMMA", 0.1))  # Default gamma is 0.05
learning_rate = 1e-5

# Initialize maximum health values
max_player_health = 1.0  # Start with a default value to avoid division by zero
max_enemy_health = 1.0

replay_count = 0
battle_count = 0
include_orig = True
do_replays = False

INSTANCES = []
# Define the server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
        'name': 'Instance 1',
        # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20230929001213-ummm-bn6-vs-DthKrdMnSP-round1-p1.tangoreplay',
        # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20230929001213-ummm-bn6-vs-IndianaOrz-round1-p2.tangoreplay',
        # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20231006015542-lunazoe-bn6-vs-IndianaOrz-round3-p2.tangoreplay',#player 2 cross change emotion state check fix needed
        # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20231006020253-lunazoe-bn6-vs-DthKrdMnSP-round1-p1.tangoreplay',
        'init_link_code': 'arena1',
        'is_player': True  # Set to True if you don't want this instance to send inputs
    },
    {
        'address': '127.0.0.1',
        'port': 12345,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
        'name': 'Instance 2',
        'init_link_code': 'arena1',
        'is_player': True  # Set to False if you want this instance to send inputs
    },
    # Additional instances can be added here
]

# Paths
SAVES_DIR = '/home/lee/Documents/Tango/saves'


# Define a function to filter save files based on the ROM type
def get_random_save(rom_path):
    # Filter based on the rom_path: 'bn6,0' for Gregar, 'bn6,1' for Falzar
    if rom_path == 'bn6,0':
        saves = [f for f in os.listdir(SAVES_DIR) if 'Gregar' in f]
    elif rom_path == 'bn6,1':
        saves = [f for f in os.listdir(SAVES_DIR) if 'Falzar' in f]
    else:
        raise ValueError("Unknown ROM path.")
    
    if not saves:
        raise FileNotFoundError(f"No save files found for ROM: {rom_path}")
    
    return os.path.join(SAVES_DIR, random.choice(saves))

# Function to create an instance configuration
def create_instance(port, rom_path, init_link_code, is_player=False):
    return {
        'address': '127.0.0.1',
        'port': port,
        'rom_path': rom_path,
        'save_path': get_random_save(rom_path),
        'name': f'Instance {port}',
        'init_link_code': init_link_code,
        'is_player': is_player
    }

# Function to generate battles
def generate_battles(num_matches):
    battles = []
    if num_matches  != 0:
        if not include_orig:
            INSTANCES.clear()
        port_base = 12346  # Starting port number

        for match in range(num_matches):
            # Generate a unique init_link_code for this match
            init_link_code = f"arena{match}"

            # Randomly choose the ROM path for this match
            rom_path = random.choice(['bn6,0', 'bn6,1'])

            # Generate two instances for each match
            instance1 = create_instance(port_base, rom_path, init_link_code)
            instance2 = create_instance(port_base + 1, rom_path, init_link_code)
            INSTANCES.append(instance1)
            INSTANCES.append(instance2)
            port_base += 2  # Increment the port numbers

    return battles

generate_battles(battle_count)


if do_replays:
    # Get all replay files with 'bn6' in their name
    replay_files = glob.glob(os.path.join(REPLAYS_DIR, '*bn6*.tangoreplay'))
    print(f"Found {len(replay_files)} replay files.")
    processed_replays = []
    port = 12844
    #shuffle the replay_files/home/lee/Documents/Tango/replaysOrig/
    random.shuffle(replay_files)
    #create an instance for battle_count * 2 replays
    for replay_file in replay_files:
        if replay_file not in processed_replays:
            replay_path = os.path.join(REPLAYS_DIR, replay_file)
            instance = {
            'address': '127.0.0.1',
            'port': port,
            'rom_path': 'bn6,0',
            'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
            'name': 'Instance 1',
            'replay_path': replay_path,
            'init_link_code': 'arena1',
            'is_player': False  # Set to True if you don't want this instance to send inputs
            }
            INSTANCES.append(instance)
            processed_replays.append(replay_file)
            port += 1
            if port >= 12844 + replay_count:
                break

# Key mappings based on model output indices
KEY_MAPPINGS = {
    0: 'A',        # 0000000100000000
    1: 'DOWN',     # 0000000010000000
    2: 'UP',       # 0000000001000000
    3: 'LEFT',     # 0000000000100000
    4: 'RIGHT',    # 0000000000010000
    5: 'X',        # 0000000000000010
    6: 'Z',        # 0000000000000001
    7: 'S',        # 0000000000000000??
}

RANDOM_MAPPINGS = {
    0: 'A',
    1: 'DOWN',
    2: 'UP',
    3: 'LEFT',
    4: 'RIGHT',
    5: 'X',
    6: 'Z',
    7: 'S'
}

# Define bit positions for each key
KEY_BIT_POSITIONS = {
    'A': 8,
    'DOWN': 7,
    'UP': 6,
    'LEFT': 5,
    'RIGHT': 4,
    'X': 1,
    'Z': 0,
    'S': 9,
    'RETURN': 3  # 0000000000001000
}


# Initialize the input tally dictionary with all keys set to 0
input_tally = {key: 0 for key in KEY_BIT_POSITIONS.keys()}
input_tally_lock = Lock()  # Lock for thread-safe updates

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Function to get the training directory based on the replay file name
def get_training_data_dir(replay_path):
    if replay_path:
        replay_name = os.path.basename(replay_path).split('.')[0]  # Extract the file name without extension
        training_data_dir = os.path.join("training_data", replay_name)
    else:
        # Generate a generic directory name based on the instance's port or name
        training_data_dir = os.path.join("training_data", "generic_instance")
    os.makedirs(training_data_dir, exist_ok=True)
    return training_data_dir

# Function to load the configuration from config.yaml
def load_config():
    config_path = 'config.yaml'  # Adjust path as needed
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config()
# Function to load the AI model

# Initialize input buffers for previous inputs
input_memory_size = config['input_features'].get('input_memory', 0)
if input_memory_size > 0:
    input_buffers = {instance['port']: deque(maxlen=input_memory_size) for instance in INSTANCES}
else:
    input_buffers = {}

# Initialize health memory buffers for previous health states
health_memory_size = config['input_features'].get('health_memory', 0)
if health_memory_size > 0:
    health_buffers = {instance['port']: deque(maxlen=health_memory_size) for instance in INSTANCES}
else:
    health_buffers = {}




# Function to load the AI models
# Initialize Separate Models for Inference and Training
def load_models(image_memory=1):
    """
    Loads separate Inference and Training models for both Planning and Battle.
    If checkpoints do not exist, initializes new models.
    Utilizes the configuration parameters.
    """
    global inference_planning_model, training_planning_model
    global inference_battle_model, training_battle_model
    global optimizer_planning, optimizer_battle
    global latest_checkpoint_number  # Access the global variable
    
    # Define the root directory
    root_dir = get_root_dir()

    # Load Training Planning Model
    training_planning_checkpoint_path = get_latest_checkpoint(model_type='planning', image_memory=image_memory)

    if training_planning_checkpoint_path:
        training_planning_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        checkpoint_training_planning = torch.load(training_planning_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_training_planning:
            training_planning_model.load_state_dict(checkpoint_training_planning['model_state_dict'])
            print(f"Training Planning Model loaded from {training_planning_checkpoint_path}")
            # Extract the checkpoint number
            latest_number = extract_number_from_checkpoint(training_planning_checkpoint_path)
            latest_checkpoint_number['planning'] = latest_number
        else:
            raise KeyError("Training Planning checkpoint does not contain 'model_state_dict'")
    else:
        # Initialize new Training Planning Model
        training_planning_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        print("No Training Planning Model checkpoint found. Initialized a new Training Planning Model.")

    training_planning_model.train()  # Set to train mode


    # Load Training Battle Model
    training_battle_checkpoint_path = get_latest_checkpoint(model_type='battle', image_memory=image_memory)

    if training_battle_checkpoint_path:
        training_battle_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        checkpoint_training_battle = torch.load(training_battle_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_training_battle:
            training_battle_model.load_state_dict(checkpoint_training_battle['model_state_dict'])
            print(f"Training Battle Model loaded from {training_battle_checkpoint_path}")
            # Extract the checkpoint number
            latest_number = extract_number_from_checkpoint(training_battle_checkpoint_path)
            latest_checkpoint_number['battle'] = latest_number
        else:
            raise KeyError("Training Battle checkpoint does not contain 'model_state_dict'")
    else:
        # Initialize new Training Battle Model
        training_battle_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        print("No Training Battle Model checkpoint found. Initialized a new Training Battle Model.")

    training_battle_model.train()  # Set to train mode

    # Initialize Inference Battle Model as a copy of Training Battle Model
    # inference_battle_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
    # inference_battle_model.load_state_dict(training_battle_model.state_dict())
    # inference_battle_model.eval()  # Set to eval mode
    # # Initialize Inference Planning Model as a copy of Training Planning Model
    # inference_planning_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
    # inference_planning_model.load_state_dict(training_planning_model.state_dict())
    # inference_planning_model.eval()  # Set to eval mode

    #just set the inference model to the train model
    inference_battle_model = training_battle_model
    inference_planning_model = training_planning_model

    inference_battle_model.eval()
    inference_planning_model.eval()
    inference_battle_model.half()
    inference_planning_model.half()

    # Initialize separate optimizers for Training Models
    optimizer_planning = optim.Adam(training_planning_model.parameters(), lr=learning_rate)
    optimizer_battle = optim.Adam(training_battle_model.parameters(), lr=learning_rate)

def get_new_checkpoint_path(model_type='battle', image_memory=1, battle_count=4, replay_count = 0):
    """
    Generates a new checkpoint path by incrementing the latest checkpoint number
    by battle_count * 2.
    """
    global latest_checkpoint_number

    # Current latest number for the model_type
    current_number = latest_checkpoint_number.get(model_type, 0)

    # Compute new number
    new_number = current_number + len(INSTANCES)#(battle_count * 2)

    # Define the checkpoint directory
    checkpoint_dir = get_checkpoint_dir(model_type=model_type, image_memory=image_memory)

    # Define the new checkpoint filename
    checkpoint_filename = f'checkpoint_{new_number}.pt'

    # Full checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Update the latest_checkpoint_number
    latest_checkpoint_number[model_type] = new_number

    return checkpoint_path


# Transform to preprocess images before inference
transform = transforms.Compose([
    transforms.Resize((160, 240)),
    transforms.ToTensor()
])

# Define image_memory and temporal_charge
IMAGE_MEMORY = get_image_memory()  # Default to 1 if not set
TEMPORAL_CHARGE = config['input_features'].get('temporal_charge', 0)

# Load the model checkpoint
load_models(IMAGE_MEMORY)


# Define optimizer and loss function
optimizer = optim.Adam(training_battle_model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(reduction='none')
scaler = GradScaler()

# Initialize frame buffers and frame counters
frame_buffers = {instance['port']: deque(maxlen=get_exponental_amount()**IMAGE_MEMORY) for instance in INSTANCES}
frame_counters = {instance['port']: -1 for instance in INSTANCES}

# Initialize sliding windows for temporal_charge if enabled
if TEMPORAL_CHARGE > 0:
    player_charge_sliding_windows = {instance['port']: deque(maxlen=TEMPORAL_CHARGE) for instance in INSTANCES}
    enemy_charge_sliding_windows = {instance['port']: deque(maxlen=TEMPORAL_CHARGE) for instance in INSTANCES}
else:
    player_charge_sliding_windows = {}
    enemy_charge_sliding_windows = {}

# Initialize data buffers for online learning
window_size = config['strategy']['parameters']['window_size']
# Initialize data buffers for online learning without window_size limit
data_buffers = {instance['port']: [] for instance in INSTANCES}

# Global dictionaries for Planning Model
# Keep all data points as long as their actions aren't all zeros
planning_data_buffers = defaultdict(list)  # Stores data points during planning phase per port


# Function to convert integer to a 16-bit binary string
def int_to_binary_string(value):
    return format(value, '016b')

# Function to generate a random action
def generate_random_action():
    # Randomly select one or more keys to press
    num_keys = random.randint(1, 3)  # Choose between 1 to 3 keys
    selected_keys = random.sample(list(RANDOM_MAPPINGS.values()), num_keys)

    # Convert selected keys to binary string
    binary_command = 0
    for key in selected_keys:
        bit_pos = KEY_BIT_POSITIONS[key]
        binary_command |= (1 << bit_pos)

    binary_string = int_to_binary_string(binary_command)
    # print(f"Generated random binary command: {binary_string} from keys {selected_keys}")
    return binary_string

# Update max health if the current health exceeds it
def update_max_health(player_health, enemy_health):
    global max_player_health, max_enemy_health
    if player_health > max_player_health:
        max_player_health = player_health
    if enemy_health > max_enemy_health:
        max_enemy_health = enemy_health

# Normalize health values
def normalize_health(player_health, enemy_health):
    normalized_player_health = player_health / max_player_health
    normalized_enemy_health = enemy_health / max_enemy_health
    return normalized_player_health, normalized_enemy_health

def process_chip(chip_value):
    """
    Converts chip_value into a one-hot tensor of size 400.
    If chip_value is greater than or equal to 400, returns a tensor of zeros.
    """
    chip_size = 400
    chip_tensor = torch.zeros(chip_size, dtype=torch.float32, device=device)
    if 0 <= chip_value < chip_size:
        chip_tensor[chip_value] = 1.0
    return chip_tensor.unsqueeze(0)  # Shape: (1, 400)


# Function to perform inference with the AI model
# previous_sent = 0

def predict(port, frames, position_tensor, inside_window, player_health, enemy_health,
            player_charge_seq, enemy_charge_seq, current_player_charge, current_enemy_charge,
            previous_inputs_tensor, health_memory_tensor, current_player_chip, current_enemy_chip):
    """
    Predict the next action based on a sequence of frames and additional game state information.
    Chooses between Planning and Battle models based on `inside_window`.

    Args:
        frames (list of PIL.Image): List of image frames.
        position_tensor (torch.Tensor or None): Processed position tensor based on config.
        inside_window (torch.Tensor): Tensor indicating if inside window.
        player_health (torch.Tensor): Normalized player health tensor.
        enemy_health (torch.Tensor): Normalized enemy health tensor.
        player_charge_seq (torch.Tensor or None): Tensor of past player charges, shape (1, temporal_charge)
        enemy_charge_seq (torch.Tensor or None): Tensor of past enemy charges, shape (1, temporal_charge)

    Returns:
        dict: Command dictionary to send to the game instance.
    """
    global inference_planning_model, inference_battle_model, input_tally, input_tally_lock
    global window_entry_time, previous_sent_dict, previous_inside_window_dict

    current_time = time.time()

    # Detect entering or exiting the window
    if inside_window.item() == 1.0 and previous_inside_window_dict[port] == 0.0:
        
        #check if planning_data_buffers has content
        if planning_data_buffers[port]:
            # print(f"Port {port}: Entering planning window. Training Planning Model.")
            max_round_damage[port] = 500#max(max_round_damage[port], current_round_damage[port])
            # Normalize the damage
            if max_round_damage[port] > 0:
                normalized_damage = current_round_damage[port] / max_round_damage[port]
            else:
                normalized_damage = 0.0
            # print(f"Port {port}: Normalized Damage = {normalized_damage}")
            
            # Assign the normalized damage as the reward to all collected data points
            # with training_planning_lock:
            for data_point in planning_data_buffers[port]:
                if not data_point.get('assigned_reward', False):
                    data_point['reward'] = normalized_damage  # Assign positive reward
                    # Train the Planning Model with the collected data points
                    # if planning_data_buffers[port]:
                        # asyncio.create_task(asyncio.to_thread(
                        #     train_model_online,
                        #     port,
                        #     planning_data_buffers[port],
                        #     model_type="Planning_Model"
                        # ))
                        # training_queue.put((port, list(data_buffers[port]), "Planning_Model", True))
                        # print(f"Port {port}: Submitted {len(planning_data_buffers[port])} data points for Planning Model training.")
                    # Clear the buffer after training
                    # planning_data_buffers[port].clear()
                    data_point['assigned_reward']=True
                    current_round_damage[port] = 0.0

        window_entry_time[port] = current_time
        previous_inside_window_dict[port] = 1.0
        current_round_damage[port] = 0.0
        # Ensure the planning data buffer is empty
        # planning_data_buffers[port].clear()
        # data_point['assigned_reward']=True
        # print(f"Port {port}: Entered window. Timer started.")

    # If inside the window, check if the timer has expired
    elif inside_window.item() == 1.0:
        elapsed_time = current_time - window_entry_time.get(port, current_time)
        if elapsed_time >= CHIP_WINDOW_TIMER:
            if force_skip_chip_window:
                if previous_sent_dict[port] == 0:
                    previous_sent_dict[port] = 1
                    # print(f"Port {port}: Sending first key in sequence.")
                    return {'type': 'key_press', 'key': '0000000001000000'}
                elif previous_sent_dict[port] == 1:
                    previous_sent_dict[port] = 2
                    # print(f"Port {port}: Sending second key in sequence.")
                    return {'type': 'key_press', 'key': '0000000000001000'}
                elif previous_sent_dict[port] == 2:
                    previous_sent_dict[port] = 0
                    # print(f"Port {port}: Sending third key in sequence.")
                    return {'type': 'key_press', 'key': '0000000000000001'}
                    

    # If not inside the window, reset the tracking variables
    else:
        if port in window_entry_time:
            del window_entry_time[port]
        previous_sent_dict[port] = 0
        previous_inside_window_dict[port] = 0.0

    
    

    # Preprocess and stack frames
    preprocessed_frames = []
    for img in frames:
        if img is None:
            print("Encountered None in frames. Skipping inference.")
            return {'type': 'key_press', 'key': '0000000000000000'}  # No action
        img = img.convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        preprocessed_frames.append(img)

    try:
        stacked_frames = torch.stack(preprocessed_frames, dim=2).to(device)  # Shape: (1, 3, D, 160, 240)
    except Exception as e:
        print(f"Error stacking frames: {e}")
        return {'type': 'key_press', 'key': '0000000000000000'}

    predicted_input_str = None
    model_type = "Battle_Model"  # Default model

     # Check additional inputs
    if position_tensor is not None:
        assert not torch.isnan(position_tensor).any(), "Position tensor contains NaN"
    if player_charge_seq is not None:
        assert not torch.isnan(player_charge_seq).any(), "Player charge sequence contains NaN"
    if enemy_charge_seq is not None:
        assert not torch.isnan(enemy_charge_seq).any(), "Enemy charge sequence contains NaN"

    # Select model based on inside_window
    if inside_window.item() == 1.0:
        selected_model = inference_planning_model
        selected_lock = inference_planning_lock
        model_type = "Planning_Model"
    else:
        selected_model = inference_battle_model
        selected_lock = inference_battle_lock
        model_type = "Battle_Model"

    # If there's no model, perform the random action
    if selected_model is None:
        print(f"Gamma condition met (gamma={GAMMA}). Taking a random action.")
        random_command = generate_random_action()
        # If inside_window ignore inputs for A, S, or X
        if inside_window.item() == 1.0:
            while (random_command[15 - KEY_BIT_POSITIONS['A']] == '1' or
                   random_command[15 - KEY_BIT_POSITIONS['S']] == '1' or
                   random_command[15 - KEY_BIT_POSITIONS['X']] == '1'):
                random_command = generate_random_action()
        #no pausing while not inside window
        if inside_window.item() == 0.0 and random_command[15 - KEY_BIT_POSITIONS['RETURN']] == '1':
            random_command = '0000000000000000'
        return {'type': 'key_press', 'key': random_command}
    
    max_prob = 0

    # Acquire the appropriate lock before performing model inference
    with selected_lock:
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Prepare additional inputs based on configuration
                    additional_inputs = {}

                    if input_memory_size > 0 and previous_inputs_tensor is not None:
                        additional_inputs['previous_inputs'] = previous_inputs_tensor  # Shape: (1, input_memory_size * 16)

                    # Include health_memory if health_memory_size is set
                    if health_memory_size > 0 and health_memory_tensor is not None:
                        additional_inputs['health_memory'] = health_memory_tensor  # Shape: (1, 2 * health_memory_size)

                    if config['input_features'].get('include_position', False):
                        additional_inputs['position'] = position_tensor  # Shape based on config
                    if config['input_features'].get('temporal_charge', 0) > 0:
                        additional_inputs['player_charge_temporal'] = player_charge_seq  # Shape: (1, temporal_charge)
                        additional_inputs['enemy_charge_temporal'] = enemy_charge_seq    # Shape: (1, temporal_charge)
                        
                        # Convert scalar charges to tensors
                        additional_inputs['player_charge'] = torch.tensor(
                            [current_player_charge], dtype=torch.float32, device=device
                        )  # Shape: (1,)
                        additional_inputs['enemy_charge'] = torch.tensor(
                            [current_enemy_charge], dtype=torch.float32, device=device
                        )  # Shape: (1,)
                    else:
                        if config['input_features'].get('include_player_charge', False):
                            additional_inputs['player_charge'] = torch.tensor(
                                [current_player_charge], dtype=torch.float32, device=device
                            )  # Shape: (1,)
                        if config['input_features'].get('include_enemy_charge', False):
                            additional_inputs['enemy_charge'] = torch.tensor(
                                [current_enemy_charge], dtype=torch.float32, device=device
                            )  # Shape: (1,)

                    # Process player_chip
                    if config['input_features'].get('include_player_chip', False):
                        player_chip_tensor = process_chip(current_player_chip)
                        additional_inputs['player_chip'] = player_chip_tensor  # Shape: (1, 400)

                    # Process enemy_chip
                    if config['input_features'].get('include_enemy_chip', False):
                        enemy_chip_tensor = process_chip(current_enemy_chip)
                        additional_inputs['enemy_chip'] = enemy_chip_tensor  # Shape: (1, 400)
                        
                    # Perform inference
                    outputs = selected_model(
                        stacked_frames,
                        **additional_inputs
                    )
                    probs = torch.sigmoid(outputs)
                    probs = probs.cpu().numpy()[0]
                    # Get threshold based on model type
                    if model_type == "Planning_Model":
                        command_threshold = get_threshold_plan()
                    else:
                        command_threshold = get_threshold()

                    # Convert probabilities to binary input string based on threshold
                    predicted_inputs = (probs >= command_threshold).astype(int)
                    predicted_input_str = ''.join(map(str, predicted_inputs))
                    # Convert probabilities to binary input string based on threshold
                    # Determine the number of actions to select
                    # This could be a fixed number or dynamically determined
                    # For example, selecting at least one action
                    # k = max(1, int(np.sum(probs > command_threshold)))  # Adjust based on your needs

                    # Select top K probabilities
                    # predicted_inputs = select_top_k(probs, k=2)
                    # predicted_input_str = ''.join(map(str, predicted_inputs))
                    # Usage within your code
                    max_prob = np.max(probs)
                    # threshold = adaptive_threshold(probs, percentile=85)
                    # predicted_inputs = (probs >= threshold).astype(int)
                    # predicted_input_str = ''.join(map(str, predicted_inputs))

                    # if max_prob < command_threshold:
                    #     predicted_input_str = '0000000000000000'
                    
        except Exception as e:
            print(f"Error during inference with {model_type}: {e}")
            return {'type': 'key_press', 'key': '0000000000000000'}  # Return no key press on failure

    print(f"{predicted_input_str} from {model_type} | {max_prob:.3f}")

    # --- Start of Tally Update ---
    with input_tally_lock:
        for key, bit_pos in KEY_BIT_POSITIONS.items():
            if predicted_inputs[15 - bit_pos] == 1:
                input_tally[key] += 1

    #never allow sending the pause command when out of the window
    if inside_window.item() == 0.0 and predicted_input_str[15 - KEY_BIT_POSITIONS['RETURN']] == '1':
        predicted_input_str = '0000000000000000'
    # Decide whether to take a random action based on GAMMA
    if random.random() < GAMMA:
        # print(f"Gamma condition met (gamma={GAMMA}). Taking a random action.")
        random_command = generate_random_action()
        # If inside_window ignore inputs for A, S, or X
        if inside_window.item() == 1.0:
            while (random_command[15 - KEY_BIT_POSITIONS['A']] == '1' or
                   random_command[15 - KEY_BIT_POSITIONS['S']] == '1' or
                   random_command[15 - KEY_BIT_POSITIONS['X']] == '1'):
                random_command = generate_random_action()
        #no pausing while not inside window
        if inside_window.item() == 0.0 and random_command[15 - KEY_BIT_POSITIONS['RETURN']] == '1':
            random_command = '0000000000000000'
        return {'type': 'key_press', 'key': random_command}
    else:
        #no pausing 0000000000001000 while not inside window
        return {'type': 'key_press', 'key': predicted_input_str}
    

def adaptive_threshold(probs, percentile=75):
    """
    Determine a threshold based on the given percentile of probabilities.

    Args:
        probs (np.ndarray): Array of probabilities.
        percentile (float): Percentile to determine the threshold.

    Returns:
        float: The threshold value.
    """
    return np.percentile(probs, percentile)


# Define a function to select top K probabilities
def select_top_k(probs, k=2):
    """
    Select the top K probabilities and set them to 1, others to 0.

    Args:
        probs (np.ndarray): Array of probabilities.
        k (int): Number of top probabilities to select.

    Returns:
        np.ndarray: Binary array with top K probabilities set to 1.
    """
    # Initialize a binary array of zeros
    binary_inputs = np.zeros_like(probs, dtype=int)
    
    if k > 0:
        # Get the indices of the top K probabilities
        top_k_indices = probs.argsort()[-k:]
        # Set the top K indices to 1
        binary_inputs[top_k_indices] = 1
    
    return binary_inputs

max_reward = 1
max_punishment = 1
# Function to receive messages from the game and process them
async def receive_messages(reader, writer, port, training_data_dir, config):
    global max_reward, max_punishment
    buffer = ""
    current_input = None
    current_reward = 0
    current_punishment = 0
    # Initialize additional fields
    current_player_health = None
    current_enemy_health = None
    current_player_position = None
    current_enemy_position = None
    current_inside_window = None
    current_player_charge = None  # New field
    current_enemy_charge = None  # New field

    game_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
    is_replay = game_instance.get('replay_path', None) is not None

    # Define the minimum required frames for exponential sampling
    minimum_required_frames = get_exponental_amount()**(IMAGE_MEMORY - 1) if IMAGE_MEMORY > 1 else 1

    try:
        while True:
            # Read data from the connection
            data = await reader.read(8192)
            if not data:
                print(f"Port {port}: Connection closed by peer.")
                break

            buffer += data.decode()

            # Split the buffer by newlines to process each message
            while "\n" in buffer:
                try:
                    message, buffer = buffer.split("\n", 1)  # Split off the first complete message
                    message = message.strip()  # Clean up any whitespace

                    if not message:
                        continue  # Ignore empty messages

                    parsed_message = json.loads(message)
                except json.JSONDecodeError:
                    print(f"Port {port}: Failed to parse JSON message: {message}")
                    continue

                event = parsed_message.get("event", "Unknown")
                details = parsed_message.get("details", "No details provided")

                if event == "local_input":
                    try:
                        current_input = int_to_binary_string(int(details))
                        # print(f"Port {port}: Updated current_input to {current_input}")
                    except ValueError:
                        print(f"Port {port}: Failed to parse local input: {details}")

                elif event == "screen_image":
                    # Parse the details JSON to extract additional data
                    try:
                        screen_data = json.loads(details)
                        #log screendata
                        # print(f"Port {port}: Screen data: {screen_data}")
                        encoded_image = screen_data.get("image", "")
                        current_player_health = float(screen_data.get("player_health", 0))
                        current_enemy_health = float(screen_data.get("enemy_health", 0))
                        current_player_position = screen_data.get("player_position", None)
                        current_enemy_position = screen_data.get("enemy_position", None)
                        current_inside_window = float(screen_data.get("inside_window", 0))
                        current_player_charge = float(screen_data.get("player_charge", 0))  # Extract player_charge
                        current_enemy_charge = float(screen_data.get("enemy_charge", 0))  # Extract enemy_charge
                        current_reward += float(screen_data.get("reward", 0))  # Accumulate reward
                        current_punishment += float(screen_data.get("punishment", 0))  # Accumulate punishment
                        current_input = screen_data.get("current_input", None)  # Update current_input if available
                        if current_input is not None:
                            current_input = int_to_binary_string(current_input)
                        current_player_chip = screen_data.get("player_chip", 0)#value 0-400
                        current_enemy_chip = screen_data.get("enemy_chip", 0)#value 0-400
                    except json.JSONDecodeError:
                        print(f"Port {port}: Failed to parse screen_image details: {details}")
                        continue
                    
                    #print current player chip and enemy chip
                    # print(f"Port {port}: Current player chip: {current_player_chip}, Current enemy chip: {current_enemy_chip}")

                    current_round_damage[port] += float(screen_data.get("reward", 0))
                    current_round_damage[port] -= float(screen_data.get("punishment", 0))  

                    if current_input is not None and input_memory_size > 0:
                        input_buffers[port].append(current_input)


                    if current_player_health is not None and current_enemy_health is not None:
                        final_health[port] = (current_player_health, current_enemy_health)
                        # print(f"Port {port}: Final Health - Player: {current_player_health}, Enemy: {current_enemy_health}")
                    #input
                    # print(f"Port {port}: Current input: {current_input}")
                    #print rewards
                    # print(f"Port {port}: Current reward: {current_reward}, Current punishment: {current_punishment}")

                    # Update sliding windows if temporal_charge is included
                    if TEMPORAL_CHARGE > 0:
                        player_charge_sliding_windows[port].append(current_player_charge)
                        enemy_charge_sliding_windows[port].append(current_enemy_charge)
                    
                    # Process position based on configuration
                    position_tensor = process_position(current_player_position, current_enemy_position, config)

                    # Compute additional tensors
                    inside_window_tensor = torch.tensor([current_inside_window], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    # Before passing health to the model
                    update_max_health(current_player_health, current_enemy_health)
                    normalized_player_health, normalized_enemy_health = normalize_health(current_player_health, current_enemy_health)

                    player_health_tensor = torch.tensor([normalized_player_health], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    enemy_health_tensor = torch.tensor([normalized_enemy_health], dtype=torch.float32, device=device).unsqueeze(0)    # Shape: (1, 1)

                    # Update health memory buffers
                    if health_memory_size > 0:
                        health_buffers[port].append((normalized_player_health, normalized_enemy_health))

                    # Prepare health_memory_tensor
                    if health_memory_size > 0:
                        health_memory_list = list(health_buffers[port])
                        health_memory_flat = []
                        for hp, ep in health_memory_list:
                            health_memory_flat.extend([hp, ep])
                        # Pad if necessary
                        if len(health_memory_flat) < 2 * health_memory_size:
                            padding = [0.0] * (2 * health_memory_size - len(health_memory_flat))
                            health_memory_flat = padding + health_memory_flat
                        health_memory_tensor = torch.tensor([health_memory_flat], dtype=torch.float32, device=device)  # Shape: (1, 2 * health_memory_size)
                    else:
                        health_memory_tensor = None



                    # Prepare previous inputs tensor
                    if input_memory_size > 0:
                        previous_inputs = list(input_buffers[port])
                        # Convert to tensor
                        previous_inputs_tensor = inputs_to_tensor(previous_inputs, input_memory_size)
                        # print(f"Port {port}: Previous inputs tensor shape: {previous_inputs_tensor.shape}")
                    else:
                        previous_inputs_tensor = None

                    # Save the image and retrieve it
                    save_result = (None, Image.open(BytesIO(base64.b64decode(encoded_image))))
                    if save_result is None:
                        print(f"Port {port}: No screen image available.")
                        continue
                    image_path, image = save_result

                    if image is None:
                        print(f"Port {port}: Failed to decode image.")
                        continue

                    # Increment the frame counter (0-based indexing)
                    frame_counters[port] += 1
                    current_frame_idx = frame_counters[port]

                    # Add the new frame with its index to the buffer
                    frame_buffers[port].append((current_frame_idx, image))

                    # Retrieve all frame indices currently in the buffer
                    available_indices = [frame_idx for frame_idx, _ in frame_buffers[port]]

                    # Current index is the latest frame's index
                    current_idx = available_indices[-1]

                    # Check if buffer has enough frames for exponential sampling
                    if len(frame_buffers[port]) >= minimum_required_frames:
                        # Use get_exponential_sample to get the required frame indices
                        sampled_indices = get_exponential_sample(available_indices, current_idx, IMAGE_MEMORY)

                        if not sampled_indices:
                            continue

                        #clamp sampled indices to the available indices
                        # sampled_indices = [min(max(0, idx), len(frame_buffers[port]) - 1) for idx in sampled_indices]
                        # Fetch the corresponding frames based on sampled indices
                        sampled_frames = []
                        index_set = set(sampled_indices)  # For faster lookup
                        for frame_idx, img in frame_buffers[port]:
                            if frame_idx in index_set:
                                sampled_frames.append(img)
                                if len(sampled_frames) == IMAGE_MEMORY:
                                    break

                        # If not enough frames were sampled, pad with the earliest frame
                        if len(sampled_frames) < IMAGE_MEMORY:
                            if sampled_frames:
                                earliest_frame = sampled_frames[0]
                                while len(sampled_frames) < IMAGE_MEMORY:
                                    sampled_frames.insert(0, earliest_frame)
                            else:
                                sampled_frames = [None] * IMAGE_MEMORY  # Placeholder for no action

                        # Validate sampled_frames before inference
                        if any(frame is None for frame in sampled_frames):
                            print(f"Port {port}: Insufficient frames for inference. Skipping command sending.")
                            continue

                        # Get temporal_charge sequences if included
                        if TEMPORAL_CHARGE > 0:
                            # Get sliding window data as tensors
                            player_charge_seq = torch.tensor(list(player_charge_sliding_windows[port]), dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, temporal_charge)
                            enemy_charge_seq = torch.tensor(list(enemy_charge_sliding_windows[port]), dtype=torch.float32, device=device).unsqueeze(0)    # Shape: (1, temporal_charge)
                        else:
                            player_charge_seq = None
                            enemy_charge_seq = None

                        # Prepare health_memory_tensor
                        if health_memory_size > 0:
                            health_memory_tensor = torch.tensor([health_memory_flat], dtype=torch.float32, device=device)  # Shape: (1, 2 * health_memory_size)
                        else:
                            health_memory_tensor = None

                        # Only send commands if the instance is not a player
                        game_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
                        if game_instance:# and not game_instance.get('is_player', False):

                            # Prepare data point
                            data_point = {
                                'frames': sampled_frames,
                                'position_tensor': position_tensor,
                                'inside_window': inside_window_tensor,
                                'player_health': player_health_tensor,
                                'enemy_health': enemy_health_tensor,
                                'player_charge_seq': player_charge_seq,
                                'enemy_charge_seq': enemy_charge_seq,
                                'current_player_charge': current_player_charge,
                                'current_enemy_charge': current_enemy_charge,
                                'previous_inputs': previous_inputs_tensor,
                                'health_memory': health_memory_tensor,  # Add health_memory
                                'current_player_chip': current_player_chip,
                                'current_enemy_chip': current_enemy_chip,
                                'action': None  # Will be updated after sending the command
                            }
                            if not game_instance.get('is_player', False) and not is_replay:
                                command = predict(
                                    port,
                                    sampled_frames,
                                    position_tensor,
                                    inside_window_tensor,
                                    player_health_tensor,
                                    enemy_health_tensor,
                                    player_charge_seq,
                                    enemy_charge_seq,
                                    current_player_charge,
                                    current_enemy_charge,
                                    previous_inputs_tensor,
                                    health_memory_tensor,
                                    current_player_chip,
                                    current_enemy_chip
                                )
                                #final check to make sure the action doesn't contain the pause button while outside the window
                                if current_inside_window == 0 and command['key'][15 - KEY_BIT_POSITIONS['RETURN']] == '1': 
                                    command['key'][15 - KEY_BIT_POSITIONS['RETURN']] = '0'
                                data_point['action'] = command['key']  # Store the action
                                # print(f"Port {port}: Sending command: {command['key']}")
                                await send_input_command(writer, command, port)
                            else:
                                data_point['action'] = current_input  # Store the current input
                                # print(f"Port {port}: Sending command: {current_input}")
                            # print(command['key'])

                            # Append data_point to buffer
                            data_buffers[port].append(data_point)
                           
                            max_reward = 200#max(max_reward, current_reward)
                            max_punishment = 200#max(max_punishment, current_punishment)
                            # If we received a reward or punishment, perform online training
                            # Assign reward to the current data_point
                            if current_reward != 0 or current_punishment != 0:
                                # Compute reward for the data_point
                                reward_value = (current_reward / max_reward) - (current_punishment / max_punishment)
                                # print(f"Port {port}: Assigning reward: {reward_value}")

                                #assign datapoint if not assigned
                                if 'reward' not in data_buffers[port][-1]:
                                    data_buffers[port][-1]['reward'] = reward_value
                                else:
                                    data_point['reward'] += reward_value    

                                # Apply discounted rewards to past data points
                                gamma = 0.99  # Discount factor
                                for i, past_data_point in enumerate(reversed(data_buffers[port])):
                                    discounted_reward = reward_value * (gamma ** i)
                                    if 'reward' in past_data_point:
                                        past_data_point['reward'] += discounted_reward
                                    else:
                                        past_data_point['reward'] = discounted_reward

                                # Train on the updated buffer
                                # await train_model_online(
                                #     port,
                                #     list(data_buffers[port]),
                                #     model_type="Battle_Model"
                                # )
                                # asyncio.create_task(asyncio.to_thread(
                                #     train_model_online,
                                #     port,
                                #     list(data_buffers[port]),
                                #     model_type="Battle_Model"
                                # ))

                                # training_queue.put((port, list(data_buffers[port]), "Battle_Model", True))
                                # Reset rewards
                                current_reward = 0
                                current_punishment = 0
                            # else:
                            #     if current_player_charge != 0:# or current_punishment == 0:
                            #         # print(f"Port {port}: Assigning reward: {0.1}")
                            #         # Train on the current frame (data_point)
                            #         if current_player_charge != 0:
                            #             data_point['reward'] = 0.1

                            # Pruning Logic Starts Here
                            # Remove data points older than window_size that have no reward assigned
                            if len(data_buffers[port]) > window_size:
                                # Calculate how many data points to prune
                                num_to_prune = len(data_buffers[port]) - window_size
                                pruned_count = 0
                                i = 0
                                while pruned_count < num_to_prune and i < len(data_buffers[port]):
                                    if 'reward' not in data_buffers[port][i]:
                                        del data_buffers[port][i]
                                        pruned_count += 1
                                    else:
                                        i += 1

                            

                                # if pruned_count > 0:
                                #     print(f"Port {port}: Pruned {pruned_count} data points from data_buffers.")
                            # else:
                            #     # Assign a small positive reward for frames with no damage taken
                            #     # data_point['reward'] = 0.1  # Adjust the value as appropriate

                            #     # Append data_point to buffer
                            #     data_buffers[port].append(data_point)

                                
                                    # else:
                                    #     data_point['reward'] = 0.001

                            #     asyncio.create_task(asyncio.to_thread(
                            #         train_model_online,
                            #         port,
                            #         [data_point],  # Pass a list with only the current data_point
                            #         model_type="Battle_Model",
                            #         log=False
                            #     ))
                            
                            #if not moving, give a small punishment
                            # elif current_input == '0000000000000000':
                            #     print(f"Port {port}: Assigning punishment: {-1}")
                            #     data_point['reward'] = -1
                            #     await train_model_online(
                            #         port,
                            #         [data_point],  # Pass a list with only the current data_point
                            #         model_type="Battle_Model",  # Assuming you're updating the Battle Model
                            #         log=False
                            #     )
                            
                            # Reset rewards/punishments
                            current_reward = 0
                            current_punishment = 0

                        # Save the game state only if training_data_dir is set
                        # if training_data_dir and current_input is not None:
                        #     input_binary = current_input
                            # save_game_state(
                            #     image_path=image_path,
                            #     input_binary=input_binary,
                            #     reward=current_reward,
                            #     punishment=current_punishment,
                            #     training_data_dir=training_data_dir,
                            #     player_health=current_player_health,
                            #     enemy_health=current_enemy_health,
                            #     player_position=current_player_position,
                            #     enemy_position=current_enemy_position,
                            #     inside_window=current_inside_window
                            # )
                            # print(f"Port {port}: Saved game state.")

                        # Reset additional fields after processing
                        if current_inside_window == 1.0:
                            # Append data_point to planning_data_buffers
                            #only append if the action is not all 0s, unless the current is all 0s and the previous is not (to signify release)
                            if data_point['action'] != '0000000000000000' or (data_point['action'] == '0000000000000000' and planning_data_buffers[port] and planning_data_buffers[port][-1]['action'] != '0000000000000000'):
                                planning_data_buffers[port].append(data_point)
                            # print(f"Port {port}: Appended data_point to planning_data_buffers. {current_input}. {len(planning_data_buffers[port])}")
                        current_player_health = None
                        current_enemy_health = None
                        current_player_position = None
                        current_enemy_position = None
                        current_inside_window = None
                        current_player_charge = None  # Reset charge
                        current_enemy_charge = None

                        

                    # else:
                    #     print(f"Port {port}: Not enough frames for exponential sampling. Required: {minimum_required_frames}, Available: {len(frame_buffers[port])}")



    except (ConnectionResetError, BrokenPipeError):
        print(f"Port {port}: Connection was reset by peer, closing receiver.")
    except Exception as e:
        print(f"Port {port}: Failed to receive message: {e}")    
        print("Detailed error line:")
        print(traceback.format_exc())
    finally:

        if writer:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                print(f"Error closing connection to {game_instance['name']} on port {port}: {e}")
        print(f"Connection to {game_instance['name']} on port {port} closed.")
        
def inputs_to_tensor(inputs_list, input_memory_size):
    """
    Converts a list of input bitstrings to a tensor.
    Each input is a 16-bit binary string.

    Args:
        inputs_list (list): List of input bitstrings.
        input_memory_size (int): Expected size of the input memory.

    Returns:
        torch.Tensor: Tensor of shape (1, input_memory_size * 16)
    """
    # Pad the inputs_list if it's shorter than input_memory_size
    if len(inputs_list) < input_memory_size:
        padding = ['0' * 16] * (input_memory_size - len(inputs_list))
        inputs_list = padding + inputs_list

    # Flatten the inputs into a single list of bits
    flattened_bits = []
    for input_str in inputs_list:
        bits = [int(bit) for bit in input_str]
        flattened_bits.extend(bits)

    # Convert to tensor
    inputs_tensor = torch.tensor([flattened_bits], dtype=torch.float32, device=device)  # Shape: (1, input_memory_size * 16)
    return inputs_tensor


from tqdm import tqdm
import traceback
data_buffers_locks = {instance['port']: Lock() for instance in INSTANCES}

def training_thread_function(batch_size=48, max_wait_time=1.0):
    """
    Training thread that processes training data in batches from each port's data buffer.
    
    Args:
        batch_size (int): Number of training data items per batch.
        max_wait_time (float): Maximum time (in seconds) to wait before processing a batch.
    """
    global training_battle_model, training_planning_model, optimizer_battle, optimizer_planning
    
    # Initialize the progress bar
    progress_bar = tqdm(desc="Training Progress", unit="batch")
    
    while True:
        start_time = time.time()
        any_data_processed = False
        
        #train battle model
        for port, buffer in data_buffers.items():
            lock = data_buffers_locks[port]
            with lock:
                if len(buffer) == 0:
                    continue  # No data to process for this port
                
                # Determine the number of data points to process
                current_batch_size = min(batch_size, len(buffer))
                batch_data = buffer[:current_batch_size]
                
                # Remove the processed data points from the buffer
                del buffer[:current_batch_size]
            
            if batch_data:
                any_data_processed = True
                # Perform training on the batch_data
                train_model_online(port, batch_data, model_type="Battle_Model", log=True)
                progress_bar.update(1)
            
        #remove all battle data from memory
        for port, buffer in data_buffers.items():
            lock = data_buffers_locks[port]
            with lock:
                buffer.clear()

        #train planning model
        for port, buffer in planning_data_buffers.items():
            lock = data_buffers_locks[port]
            with lock:
                if len(buffer) == 0:
                    continue
                
                # Determine the number of data points to process
                current_batch_size = min(batch_size, len(buffer))
                batch_data = buffer[:current_batch_size]

                # Remove the processed data points from the buffer
                del buffer[:current_batch_size]

            if batch_data:
                any_data_processed = True
                # Perform training on the batch_data
                train_model_online(port, batch_data, model_type="Planning_Model", log=True)
                progress_bar.update(1)
        
        if not any_data_processed:
            # If no data was processed, wait for a short duration
            elapsed = time.time() - start_time
            sleep_time = max_wait_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Stop once all the data finished being processed
        if all(len(buffer) == 0 for buffer in data_buffers.values()):
            break



def train_model_online(port, data_buffer, model_type="Battle_Model", log=True):
    global training_battle_model, training_planning_model, optimizer_battle, optimizer_planning

    # Select the appropriate training model and lock
    if model_type == "Battle_Model":
        selected_model = training_battle_model
        selected_lock = training_battle_lock
        selected_optimizer = optimizer_battle
    elif model_type == "Planning_Model":
        selected_model = training_planning_model
        selected_lock = training_planning_lock
        selected_optimizer = optimizer_planning
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with selected_lock:
        selected_model = selected_model.float()
        selected_model.train()  # Switch to train mode

        batch_size = len(data_buffer)
        # if batch_size == 0:
        #     print(f"Port {port}: No data to train on.")
        #     return

        # Prepare batches
        frames_batch = []
        position_batch = []
        player_charge_seq_batch = []
        enemy_charge_seq_batch = []
        player_charge_batch = []
        enemy_charge_batch = []
        targets_batch = []
        rewards_batch = []
        previous_inputs_batch = []
        health_memory_batch = []
        player_chip_batch = []
        enemy_chip_batch = []



        for data_point in data_buffer:
            # Process frames
            preprocessed_frames = []
            for img in data_point['frames']:
                img = img.convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                preprocessed_frames.append(img)
            stacked_frames = torch.stack(preprocessed_frames, dim=2).squeeze(0)
            frames_batch.append(stacked_frames)

            # Process player_chip
            if data_point.get('player_chip') is not None:
                player_chip_tensor = process_chip(data_point['player_chip'])
                player_chip_batch.append(player_chip_tensor.squeeze(0))  # Remove batch dimension
            else:
                player_chip_batch.append(torch.zeros(400, device=device))

            # Process enemy_chip
            if data_point.get('enemy_chip') is not None:
                enemy_chip_tensor = process_chip(data_point['enemy_chip'])
                enemy_chip_batch.append(enemy_chip_tensor.squeeze(0))  # Remove batch dimension
            else:
                enemy_chip_batch.append(torch.zeros(400, device=device))


            # Positions
            if data_point['position_tensor'] is not None:
                position_batch.append(data_point['position_tensor'].squeeze(0))

            # Temporal charges
            if data_point['player_charge_seq'] is not None:
                player_charge_seq = data_point['player_charge_seq'].squeeze(0)
                # Ensure the straining completedequence is of length TEMPORAL_CHARGE by padding if necessary
                if player_charge_seq.size(0) < TEMPORAL_CHARGE:
                    padding = torch.zeros(TEMPORAL_CHARGE - player_charge_seq.size(0), device=device)
                    player_charge_seq = torch.cat((padding, player_charge_seq))
                player_charge_seq_batch.append(player_charge_seq)
            else:
                # If not present, pad with zeros
                player_charge_seq_batch.append(torch.zeros(TEMPORAL_CHARGE, device=device))

            # Previous inputs
            if data_point.get('previous_inputs') is not None:
                previous_inputs_batch.append(data_point['previous_inputs'].squeeze(0))
            else:
                previous_inputs_batch.append(torch.zeros(input_memory_size * 16, device=device))

            # Health memory
            if data_point.get('health_memory') is not None:
                health_memory_batch.append(data_point['health_memory'].squeeze(0))
            else:
                health_memory_batch.append(torch.zeros(2 * health_memory_size, device=device))


            if data_point['enemy_charge_seq'] is not None:
                enemy_charge_seq = data_point['enemy_charge_seq'].squeeze(0)
                # Ensure the sequence is of length TEMPORAL_CHARGE by padding if necessary
                if enemy_charge_seq.size(0) < TEMPORAL_CHARGE:
                    padding = torch.zeros(TEMPORAL_CHARGE - enemy_charge_seq.size(0), device=device)
                    enemy_charge_seq = torch.cat((padding, enemy_charge_seq))
                enemy_charge_seq_batch.append(enemy_charge_seq)
            else:
                # If not present, pad with zeros
                enemy_charge_seq_batch.append(torch.zeros(TEMPORAL_CHARGE, device=device))

            # Current charges
            if data_point['current_player_charge'] is not None:
                player_charge_batch.append(
                    torch.tensor(data_point['current_player_charge'], dtype=torch.float32, device=device)
                )
            else:
                player_charge_batch.append(torch.tensor(0.0, dtype=torch.float32, device=device))

            if data_point['current_enemy_charge'] is not None:
                enemy_charge_batch.append(
                    torch.tensor(data_point['current_enemy_charge'], dtype=torch.float32, device=device)
                )
            else:
                enemy_charge_batch.append(torch.tensor(0.0, dtype=torch.float32, device=device))

            # Actions (targets)
            action_binary = data_point['action']  # Binary string
            action_tensor = torch.tensor([int(bit) for bit in action_binary], dtype=torch.float32, device=device)
            targets_batch.append(action_tensor)

            # Rewards
            rewards_batch.append(data_point.get('reward', 0.0))

        # Stack batches
        frames_batch = torch.stack(frames_batch)  # Shape: (batch_size, 3, D, H, W)

        if player_chip_batch:
            player_chip_batch = torch.stack(player_chip_batch)  # Shape: (batch_size, 400)
        else:
            player_chip_batch = None

        if enemy_chip_batch:
            enemy_chip_batch = torch.stack(enemy_chip_batch)  # Shape: (batch_size, 400)
        else:
            enemy_chip_batch = None

        if position_batch:
            position_batch = torch.stack(position_batch)  # Shape: (batch_size, position_dim)
        else:
            position_batch = None
            
        if previous_inputs_batch:
                previous_inputs_batch = torch.stack(previous_inputs_batch)  # Shape: (batch_size, input_memory_size * 16)
        else:
            previous_inputs_batch = None

        if player_charge_seq_batch:
            player_charge_seq_batch = torch.stack(player_charge_seq_batch)  # Shape: (batch_size, TEMPORAL_CHARGE)
        else:
            player_charge_seq_batch = torch.zeros(batch_size, TEMPORAL_CHARGE, device=device)

        if enemy_charge_seq_batch:
            enemy_charge_seq_batch = torch.stack(enemy_charge_seq_batch)  # Shape: (batch_size, TEMPORAL_CHARGE)
        else:
            enemy_charge_seq_batch = torch.zeros(batch_size, TEMPORAL_CHARGE, device=device)

        if player_charge_batch:
            player_charge_batch = torch.stack(player_charge_batch)  # Shape: (batch_size,)
        else:
            player_charge_batch = torch.zeros(batch_size, device=device)

        if enemy_charge_batch:
            enemy_charge_batch = torch.stack(enemy_charge_batch)  # Shape: (batch_size,)
        else:
            enemy_charge_batch = torch.zeros(batch_size, device=device)

        if health_memory_batch:
            health_memory_batch = torch.stack(health_memory_batch)  # Shape: (batch_size, 2 * health_memory_size)
        else:
            health_memory_batch = None

        targets_batch = torch.stack(targets_batch)  # Shape: (batch_size, num_actions)

        # Convert rewards to tensor
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)  # Shape: (batch_size,)

         # Zero gradients
        selected_optimizer.zero_grad()

        assert not torch.isnan(frames_batch).any(), "frames_batch contains NaN"
        assert not torch.isinf(frames_batch).any(), "frames_batch contains Inf"

        if position_batch is not None:
            assert not torch.isnan(position_batch).any(), "position_batch contains NaN"
            assert not torch.isinf(position_batch).any(), "position_batch contains Inf"

        assert not torch.isnan(player_charge_batch).any(), "player_charge_batch contains NaN"
        assert not torch.isinf(player_charge_batch).any(), "player_charge_batch contains Inf"

        assert not torch.isnan(enemy_charge_batch).any(), "enemy_charge_batch contains NaN"
        assert not torch.isinf(enemy_charge_batch).any(), "enemy_charge_batch contains Inf"

        assert not torch.isnan(player_charge_seq_batch).any(), "player_charge_seq_batch contains NaN"
        assert not torch.isinf(player_charge_seq_batch).any(), "player_charge_seq_batch contains Inf"

        assert not torch.isnan(enemy_charge_seq_batch).any(), "enemy_charge_seq_batch contains NaN"
        assert not torch.isinf(enemy_charge_seq_batch).any(), "enemy_charge_seq_batch contains Inf"

        # Save the batch here
        batch_data = {
            'frames': frames_batch,  # Shape: (batch_size, 3, D, H, W)
            'position': position_batch,  # Shape: (batch_size, position_dim) or None
            'player_charge_seq': player_charge_seq_batch,  # Shape: (batch_size, TEMPORAL_CHARGE)
            'enemy_charge_seq': enemy_charge_seq_batch,  # Shape: (batch_size, TEMPORAL_CHARGE)
            'player_charge': player_charge_batch,  # Shape: (batch_size,)
            'enemy_charge': enemy_charge_batch,  # Shape: (batch_size,)
            'previous_inputs': previous_inputs_batch,  # Shape: (batch_size, input_memory_size * 16) or None
            'health_memory': health_memory_batch,  # Shape: (batch_size, 2 * health_memory_size) or None
            'player_chip': player_chip_batch,  # Shape: (batch_size, 400) or None
            'enemy_chip': enemy_chip_batch,  # Shape: (batch_size, 400) or None
            'actions': targets_batch,  # Shape: (batch_size, num_actions)
            'rewards': rewards_batch   # Shape: (batch_size,)
        }

        # Determine the training_data_dir based on model_type and port
        # Assuming you have access to `instance` or `replay_path`. Modify as necessary.
        # If `replay_path` is part of your data_buffer or accessible globally, adjust accordingly.
        # Here, we'll assume it's `None`. Adjust this based on your actual implementation.
        
        # Save the batch data using HDF5
        if model_type == "Battle_Model":
            save_batch_to_hdf5(batch_data, battle_data_dir, port, model_type)
        elif model_type == "Planning_Model":
            save_batch_to_hdf5(batch_data, planning_data_dir, port, model_type)


        # Forward pass
        outputs = selected_model(
            frames_batch,
            position=position_batch,
            player_charge=player_charge_batch,
            enemy_charge=enemy_charge_batch,
            player_charge_temporal=player_charge_seq_batch,
            enemy_charge_temporal=enemy_charge_seq_batch,
            previous_inputs=previous_inputs_batch,
            health_memory=health_memory_batch,
            player_chip=player_chip_batch,
            enemy_chip=enemy_chip_batch
        )



        # Compute log probabilities directly
        probs = torch.sigmoid(outputs)
        epsilon = 1e-6  # For numerical stability
        probs = torch.clamp(probs, epsilon, 1 - epsilon)
        log_probs = targets_batch * torch.log(probs) + (1 - targets_batch) * torch.log(1 - probs)

        # Compute the policy loss with the correct sign
        policy_loss = - (log_probs.sum(dim=1) * rewards_batch).mean()

        # Compute entropy
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).sum(dim=1)

        entropy_coefficient = 0.05
        total_loss = policy_loss - entropy_coefficient * entropy.mean()

        # Backward pass
        total_loss.backward()

        # After computing outputs
        assert not torch.isnan(outputs).any(), "Outputs contain NaN"
        assert not torch.isinf(outputs).any(), "Outputs contain Inf"

        # After computing log_probs
        assert not torch.isnan(log_probs).any(), "log_probs contain NaN"
        assert not torch.isinf(log_probs).any(), "log_probs contain Inf"

        # After computing policy_loss
        assert not torch.isnan(policy_loss).any(), "policy_loss contains NaN"
        assert not torch.isinf(policy_loss).any(), "policy_loss contain Inf"

        # After computing entropy
        assert not torch.isnan(entropy).any(), "entropy contains NaN"
        assert not torch.isinf(entropy).any(), "entropy contains Inf"

        # After computing total_loss
        assert not torch.isnan(total_loss).any(), "total_loss contains NaN"
        assert not torch.isinf(total_loss).any(), "total_loss contain Inf"


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(selected_model.parameters(), max_norm=1.0)
        optimizer.step()

        # if log:
        #     print(f"{model_type} learning from Port {port}:\nLoss: {total_loss.item()}")
        #     #log remaining items to train on

        # Log metrics to W&B
        # After computing outputs and losses
        total_loss_value = total_loss.item()
        policy_loss_value = policy_loss.item()
        entropy_value = entropy.mean().item()
        print(f"{model_type} learning from Port {port}:\nLoss: {total_loss_value}")
        # wandb.log({
        #     f"{model_type}/total_loss": total_loss_value,
        #     f"{model_type}/policy_loss": policy_loss_value,
        #     f"{model_type}/entropy": entropy_value,
        #     f"{model_type}/batch_size": batch_size,
        #     f"{model_type}/port": port
        # })

        # selected_model.eval()  # Switch back to eval mode


# Function to request the current screen image
async def request_screen_image(writer):
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command)
    except Exception as e:
        print(f"Failed to request screen image: {e}")
        raise
# Function to save the winner status in the replay folder
def save_winner_status(training_data_dir, player_won):
    winner_status = {
        "is_winner": player_won
    }
    file_path = os.path.join(training_data_dir, "winner.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(winner_status, f)
        print(f"Saved winner status to {file_path}")
    except Exception as e:
        print(f"Failed to save winner status to {file_path}: {e}")

# Function to send input command to a specific instance
async def send_input_command(writer, command, port=0):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
        # print(f"Port {port}: Sent command: {command}")
    except (ConnectionResetError, BrokenPipeError):
        # Connection has been closed; handle gracefully
        raise
    except Exception as e:
        print(f"Failed to send command on port {port}: {e}")
        raise
# Other functions remain the same (handle_connection, process_position, train_model_online, etc.)
async def handle_connection(instance, config):
    writer = None
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        training_data_dir = get_training_data_dir(instance.get('replay_path'))

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, writer, instance['port'], training_data_dir, config))

        # Set inference interval for higher frequency (e.g., 60 times per second)
        inference_interval = 1 / inference_fps()  # seconds
        # If the instance is doing a replay, adjust the interval
        if instance.get('replay_path'):
            inference_interval = inference_interval / 4.0

        while not reader.at_eof():
            try:
                # Request the current screen image
                # print(f"Requesting screen image from {instance['name']} on port {instance['port']}")
                await request_screen_image(writer)
                # print(f"Requested screen image from {instance['name']} on port {instance['port']}")
                await asyncio.sleep(inference_interval)  # Run inferences at the desired rate

            except (ConnectionResetError, BrokenPipeError):
                print(f"Connection to {instance['name']} was reset. Stopping send loop.")
                break  # Exit the loop
            except Exception as e:
                print(f"An error occurred in connection to {instance['name']} on port {instance['port']}: {e}")
                break  # Exit the loop on other exceptions

        # Wait for the receive_messages task to finish
        await receive_task

    except ConnectionRefusedError:
        print(f"Failed to connect to {instance['name']} on port {instance['port']}. Is the application running?")
    except Exception as e:
        print(f"An error occurred with {instance['name']} on port {instance['port']}: {e}")
    finally:
        if writer:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                print(f"Error closing connection to {instance['name']} on port {instance['port']}: {e}")
        print(f"Connection to {instance['name']} on port {instance['port']} closed.")

def process_position(player_position, enemy_position, config):
    """
    Processes player and enemy positions based on the position_type in the config.

    Args:
        player_position (list or tuple): [x, y] coordinates of the player.
        enemy_position (list or tuple): [x, y] coordinates of the enemy.
        config (dict): Configuration dictionary loaded from config.yaml.

    Returns:
        torch.Tensor or None: Processed position tensor or None if not included.
    """
    if not config['input_features'].get('include_position', False):
        return None

    position_type = config['input_features'].get('position_type', 'grid')

    if position_type == 'float':
        # Expecting [player_x, player_y, enemy_x, enemy_y]
        if (player_position and enemy_position and 
            isinstance(player_position, (list, tuple)) and 
            isinstance(enemy_position, (list, tuple)) and 
            len(player_position) == 2 and 
            len(enemy_position) == 2):
            position = [
                float(player_position[0]),
                float(player_position[1]),
                float(enemy_position[0]),
                float(enemy_position[1])
            ]
        else:
            # print(f"Invalid positions: player_position={player_position}, enemy_position={enemy_position}. Using zeros.")
            position = [0.0, 0.0, 0.0, 0.0]

        position_tensor = torch.tensor([position], dtype=torch.float32, device=device)  # Shape: (1, 4)
        return position_tensor

    elif position_type == 'grid':
        # Assuming compute_grid returns a (6, 3) grid
        player_grid = compute_grid(player_position)  # (6,3)
        enemy_grid = compute_grid(enemy_position)    # (6,3)
        # Flatten and concatenate to form a (1, 36) tensor
        position = torch.cat((player_grid, enemy_grid), dim=1).view(1, -1)  # Shape: (1, 36)
        return position

    else:
        raise ValueError(f"Unknown position_type: {position_type}")

import asyncio

async def monitor_mouse_over_instances(instances, check_interval=0.5):
    """
    Monitors the mouse position and updates the 'is_player' flag for each instance.

    Args:
        instances (list): List of instance dictionaries.
        check_interval (float): Time interval between checks in seconds.
    """
    while True:
        mouse_x, mouse_y = get_mouse_position()
        for instance in instances:
            window_title = str(instance['port'])
            geometry = await get_window_geometry_async(window_title)
            if geometry:
                x, y, width, height = geometry
                if (x <= mouse_x <= x + width) and (y <= mouse_y <= y + height):
                    if not instance['is_player']:
                        print(f"Mouse is over '{window_title}'. Setting is_player to True.")
                        instance['is_player'] = True
                else:
                    if instance['is_player']:
                        print(f"Mouse is not over '{window_title}'. Setting is_player to False.")
                        instance['is_player'] = False
            # else:
            #     print(f"Window '{window_title}' not found.")
        await asyncio.sleep(check_interval)


import asyncio
import subprocess

async def get_window_geometry_async(window_title):
    """
    Asynchronously retrieves the geometry of a window based on its title using wmctrl.

    Args:
        window_title (str): The exact title of the window.

    Returns:
        tuple: (x, y, width, height) if window is found, else None.
    """
    loop = asyncio.get_running_loop()
    try:
        output = await loop.run_in_executor(
            None, subprocess.check_output, ['wmctrl', '-lG']
        )
        output = output.decode()
        for line in output.splitlines():
            parts = line.split(None, 7)
            if len(parts) < 7:
                continue
            win_title = parts[7]
            if win_title == window_title:
                x = int(parts[2])
                y = int(parts[3])
                width = int(parts[4])
                height = int(parts[5])
                return (x, y, width, height)
    except subprocess.CalledProcessError as e:
        print(f"wmctrl command failed: {e}")
    return None



def get_mouse_position():
    """
    Retrieves the current position of the mouse cursor.

    Returns:
        tuple: (x, y) coordinates of the mouse.
    """
    return pyautogui.position()


# Function to run the AppImage with specific ROM, SAVE paths, and PORT
def run_instance(rom_path, save_path, port, replay_path, init_link_code, name):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["INIT_LINK_CODE"] = init_link_code
    env["PORT"] = str(port)
    if replay_path:
        env["REPLAY_PATH"] = replay_path
    env["INSTANCE_NAME"] = str(port)  # Pass the instance name as an environment variable

    print(f"Running instance '{name}' with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    try:
        subprocess.Popen([APP_PATH], env=env)
    except Exception as e:
        print(f"Failed to start instance '{name}' on port {port}: {e}")



# Function to start all instances
def start_instances():
    for instance in INSTANCES:
        #ensure each instance has a different name
        # instance['name'] = f"{instance['name']}_{instance['port']}"
        run_instance(
            instance['rom_path'],
            instance['save_path'],
            instance['port'],
            instance.get('replay_path', None),
            instance['init_link_code'],
            instance['name']
        )
        time.sleep(0.5)  # Adjust sleep time based on app's boot time

def compute_grid(position, grid_size=(6, 3)):
    """
    Computes a grid tensor based on the given position using utils.position_to_grid.

    Args:
        position (tuple or list): The (x, y) position of the player/enemy.
        grid_size (tuple): The size of the grid (rows, cols).

    Returns:
        torch.Tensor: A tensor of shape (6, 3) representing the grid on the specified device.
    """
    if position and isinstance(position, (list, tuple)) and len(position) == 2:
        x, y = position
        grid_list = position_to_grid(x, y)  # Correctly pass x and y
        grid_tensor = torch.tensor(grid_list, dtype=torch.float32, device=device)
    else:
        print(f"Invalid position format: {position}. Expected a tuple/list of (x, y).")
        grid_tensor = torch.zeros(grid_size, dtype=torch.float32, device=device)
    return grid_tensor  # Shape: (6, 3)

# Main function to start instances and handle inputs
async def main():


    # Initialize W&B
    # wandb.init(
    #     project="BattleAI",  # Replace with your project name
    #     name=f"Run-{int(time.time())}",  # Unique run name
    #     config={
    #         "learning_rate": learning_rate,
    #         "gamma": GAMMA,
    #         "image_memory": IMAGE_MEMORY,
    #         "temporal_charge": TEMPORAL_CHARGE,
    #         "batch_size": config['strategy']['parameters']['window_size'],
    #         # Add other hyperparameters as needed
    #     }
    # )
    
    # # Optionally, log the entire config.yaml
    # wandb.config.update(config)

    # Start game instances
    start_instances()
    print("Instances are running.")
    await asyncio.sleep(0.5)  # Allow some time for instances to initialize

    # Start handling connections
    connection_tasks = [asyncio.create_task(handle_connection(instance, config)) for instance in INSTANCES]

    # Start mouse monitoring task
    # monitor_task = asyncio.create_task(monitor_mouse_over_instances(INSTANCES))



    # Wait for all connection tasks to complete
    await asyncio.gather(*connection_tasks)
    print("All instances have completed. Exiting program.")
    WIN_REWARD = 0.5
    REPLAY_REWARD = 0.5
    # Determine winners and losers based on final_health
    print("Processing final health to determine winners and losers.")
    for port, (player_health, enemy_health) in final_health.items():
        if player_health > enemy_health:
            # Winner: Add +1 to all rewards
            with data_buffers_locks[port]:
                for data_point in data_buffers[port]:
                    if 'reward' in data_point:
                        data_point['reward'] += WIN_REWARD
                    else:
                        data_point['reward'] = WIN_REWARD
                for data_point in planning_data_buffers[port]:
                    if 'reward' in data_point:
                        data_point['reward'] += WIN_REWARD
                    else:
                        data_point['reward'] = WIN_REWARD
            print(f"Port {port}: Winner. Added +{WIN_REWARD} reward to all data points.")
        else:
            # Only punish non-replay instances
            # Get instance
            instance = [instance for instance in INSTANCES if instance['port'] == port][0]
            if not instance.get('replay_path'):
                # Loser: Add -1 to all rewards
                with data_buffers_locks[port]:
                    for data_point in data_buffers[port]:
                        # Make sure the reward exists
                        if 'reward' in data_point:
                            data_point['reward'] -= WIN_REWARD
                        else:
                            data_point['reward'] = -WIN_REWARD
                    for data_point in planning_data_buffers[port]:
                        if 'reward' in data_point:
                            data_point['reward'] -= WIN_REWARD
                        else:
                            data_point['reward'] = -WIN_REWARD
                print(f"Port {port}: Loser. Added -{WIN_REWARD} reward to all data points.")
    
    # Iterate over all instances
    for instance in INSTANCES:
        port = instance['port']
        if instance.get('replay_path'):
            with data_buffers_locks[port]:
                for data_point in data_buffers[port]:
                    if 'reward' in data_point:
                        data_point['reward'] += REPLAY_REWARD
                    else:
                        data_point['reward'] = REPLAY_REWARD
                for data_point in planning_data_buffers[port]:
                    if 'reward' in data_point:
                        data_point['reward'] += REPLAY_REWARD
                    else:
                        data_point['reward'] = REPLAY_REWARD
            print(f"Port {port}: Replay. Added +{REPLAY_REWARD} reward to all data points.")

    
    # print("Saving data points to disk.")
    # planning_data_dir = get_root_dir() + "/data/planning_data"
    # os.makedirs(planning_data_dir, exist_ok=True)
    # battle_data_dir = get_root_dir() + "/data/battle_data"
    # os.makedirs(battle_data_dir, exist_ok=True)
    
    # # Save all data points after assigning final rewards
    # for port, buffer in tqdm(data_buffers.items(), desc="Saving battle data"):
    #     for data_point in buffer:
    #         save_data_point(data_point, battle_data_dir, port)
    
    # for port, buffer in tqdm(planning_data_buffers.items(), desc="Saving planning data"):
    #     for data_point in buffer:
    #         save_data_point(data_point, planning_data_dir, port)
    
    # print("Data points saved to disk.")

     # Start the training thread... testing to see if we can do this after the instances have been closed
    
    # training_thread = threading.Thread(target=training_thread_function, daemon=True)
    # training_thread.start()

    # # # At the end, before exiting, signal the training thread to stop
    # training_queue.put(None)  # Sentinel value to stop the thread
    # training_thread.join() # Wait for the thread to finish

    # # --- Start of Tally Display ---
    # print("\n--- Input Tally ---")
    # with input_tally_lock:
    #     for key, count in input_tally.items():
    #         print(f"{key} - {count}")
    # print("-------------------\n")
    # # --- End of Tally Display ---

    # # Cancel the monitor task gracefully
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("Mouse monitoring task cancelled.")


    # # Save the models
    # save_models()

import h5py
import numpy as np
import torch

def save_batch_to_hdf5(batch_data, training_data_dir, port, model_type):
    """
    Saves a batch of training data to an HDF5 file with gzip compression.

    Args:
        batch_data (dict): Dictionary containing batch data tensors.
        training_data_dir (str): Directory where training data is saved.
        port (int): Port number for the instance.
        model_type (str): 'Battle_Model' or 'Planning_Model'.
    """
    # Ensure the training_data_dir exists
    os.makedirs(training_data_dir, exist_ok=True)
    
    # Create a timestamp for unique file naming
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    filename = f"{model_type}_port_{port}_{timestamp}.h5"
    file_path = os.path.join(training_data_dir, filename)
    
    # Prepare data for saving
    # Convert all torch tensors to numpy arrays
    numpy_data = {}
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            numpy_data[key] = value.cpu().numpy()
        elif isinstance(value, list):
            # Handle lists of tensors or other data types as needed
            # Example: list of previous inputs
            numpy_data[key] = np.array(value)
        elif value is None:
            # Handle None by skipping or assigning default values
            continue  # Skip saving if not essential
        else:
            numpy_data[key] = value  # Handle other data types appropriately
    
    # Additional metadata
    metadata = {
        'timestamp': timestamp,
        'port': port,
        'model_type': model_type
    }
    
    # Save to HDF5 with gzip compression
    try:
        with h5py.File(file_path, 'w') as hf:
            for key, data in numpy_data.items():
                # Determine appropriate compression level (0-9)
                hf.create_dataset(key, data=data, compression="gzip", compression_opts=4)
            # Save metadata as attributes
            for meta_key, meta_value in metadata.items():
                hf.attrs[meta_key] = meta_value
        print(f"Saved batch to {file_path} with metadata.")
        
        # Optionally verify the saved data
        with h5py.File(file_path, 'r') as hf:
            for key in numpy_data.keys():
                if key not in hf:
                    raise ValueError(f"Dataset '{key}' not found in the saved file.")
                if not np.array_equal(hf[key][:], numpy_data[key]):
                    raise ValueError(f"Data mismatch in dataset '{key}'.")
        print(f"Verified the integrity of {file_path}")
    except Exception as e:
        print(f"Failed to save or verify batch to {file_path}: {e}")

import os
import json
from datetime import datetime

def save_data_point(data_point, training_data_dir, port):
    """
    Saves a single data_point to the training_data_dir.
    
    Args:
        data_point (dict): The data point to save.
        training_data_dir (str): Directory where training data is stored.
        port (int): Port number of the instance (used to organize data).
    """
    # Create a subdirectory for the port
    port_dir = os.path.join(training_data_dir, f"port_{port}")
    os.makedirs(port_dir, exist_ok=True)
    
    # Create a unique identifier for the data point
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    data_point_id = f"data_{timestamp}"
    
    # Create a subdirectory for this data point
    data_point_dir = os.path.join(port_dir, data_point_id)
    os.makedirs(data_point_dir, exist_ok=True)
    
    # Save frames as image files
    frame_paths = []
    for i, frame in enumerate(data_point['frames']):
        frame_filename = f"frame_{i:03d}.png"
        frame_path = os.path.join(data_point_dir, frame_filename)
        frame.save(frame_path)
        frame_paths.append(frame_filename)
    
    # Prepare metadata
    metadata = {
        'frames': frame_paths,
        'position_tensor': data_point['position_tensor'].cpu().numpy().tolist() if data_point['position_tensor'] is not None else None,
        'inside_window': data_point['inside_window'].cpu().numpy().tolist() if data_point['inside_window'] is not None else None,
        'player_health': data_point['player_health'].cpu().numpy().tolist() if data_point['player_health'] is not None else None,
        'enemy_health': data_point['enemy_health'].cpu().numpy().tolist() if data_point['enemy_health'] is not None else None,
        'player_charge_seq': data_point['player_charge_seq'].cpu().numpy().tolist() if data_point['player_charge_seq'] is not None else None,
        'enemy_charge_seq': data_point['enemy_charge_seq'].cpu().numpy().tolist() if data_point['enemy_charge_seq'] is not None else None,
        'current_player_charge': data_point['current_player_charge'],
        'current_enemy_charge': data_point['current_enemy_charge'],
        'previous_inputs': data_point['previous_inputs'].cpu().numpy().tolist() if data_point['previous_inputs'] is not None else None,
        'health_memory': data_point['health_memory'].cpu().numpy().tolist() if data_point['health_memory'] is not None else None,
        'player_chip': data_point['player_chip'].cpu().numpy().tolist() if data_point['player_chip'] is not None else None,
        'enemy_chip': data_point['enemy_chip'].cpu().numpy().tolist() if data_point['enemy_chip'] is not None else None,
        'action': data_point['action'],
        'reward': data_point.get('reward', 0.0)
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(data_point_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Port {port}: Saved data_point {data_point_id} to {data_point_dir}")


def save_models():
    """
    Saves the Training Planning and Training Battle models to their respective checkpoint directories.
    Utilizes unique checkpoint paths to prevent overwriting and maintains a maximum of MAX_CHECKPOINTS.
    """
    # Define the battle_count
    global battle_count, replay_count

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
    if 'training_planning_model' in globals() and training_planning_model is not None:
        planning_checkpoint_path = get_new_checkpoint_path(model_type='planning', image_memory=IMAGE_MEMORY, battle_count=battle_count)
        torch.save({'model_state_dict': training_planning_model.state_dict()}, planning_checkpoint_path)
        print(f"Training Planning Model saved to {planning_checkpoint_path}")
        # Manage checkpoints
        planning_checkpoint_dir = get_checkpoint_dir(model_type='planning', image_memory=IMAGE_MEMORY)
        manage_checkpoints(planning_checkpoint_dir)
    else:
        print("Training Planning Model is not loaded. Skipping save.")

    # Save Training Battle Model
    if 'training_battle_model' in globals() and training_battle_model is not None:
        battle_checkpoint_path = get_new_checkpoint_path(model_type='battle', image_memory=IMAGE_MEMORY, battle_count=battle_count)
        torch.save({'model_state_dict': training_battle_model.state_dict()}, battle_checkpoint_path)
        print(f"Training Battle Model saved to {battle_checkpoint_path}")
        # Manage checkpoints
        battle_checkpoint_dir = get_checkpoint_dir(model_type='battle', image_memory=IMAGE_MEMORY)
        manage_checkpoints(battle_checkpoint_dir)
    else:
        print("Training Battle Model is not loaded. Skipping save.")


import gc

# After your main training loops and before final_training_epoch()
def clear_memory():
    global data_buffers, planning_data_buffers, final_health, frame_buffers, frame_counters
    global player_charge_sliding_windows, enemy_charge_sliding_windows
    global training_battle_model, training_planning_model
    global inference_battle_model, inference_planning_model

    # Delete buffers and data structures
    del data_buffers
    del planning_data_buffers
    del final_health
    del frame_buffers
    del frame_counters
    del player_charge_sliding_windows
    del enemy_charge_sliding_windows

    # Delete models if they are not needed anymore or to reset their memory
    # If you need them later, consider keeping them; otherwise, delete to free memory
    # del training_battle_model
    # del training_planning_model
    # del inference_battle_model
    # del inference_planning_model

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    print("Cleared memory and CUDA cache.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
