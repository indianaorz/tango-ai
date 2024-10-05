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
# from train import GameInputPredictor  # Import the model class
from utils import (
    get_image_memory, get_exponential_sample,
    get_exponental_amount, get_threshold, get_root_dir, position_to_grid, get_threshold_plan, inference_fps,get_latest_checkpoint , get_checkpoint_dir,extract_number_from_checkpoint
)

import threading
import queue

from battle_network_model import get_gamestate_tensor, BattleNetworkModel,prepare_inference_sequence

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
GAMMA = 0.0#5#float(os.getenv("GAMMA", 0.1))  # Default gamma is 0.05
learning_rate = 1e-4

# Initialize maximum health values
max_player_health = 1.0  # Start with a default value to avoid division by zero
max_enemy_health = 1.0

replay_count = 0
battle_count = 1
include_orig = False
do_replays = False
save_data = False
prune = False

# replay_count = 8
# battle_count = 0
# include_orig = False
# do_replays = True
# save_data = True
# prune = False

INSTANCES = []
# Define the server addresses and ports for each instance
INSTANCES = [
    # {
    #     'address': '127.0.0.1',
    #     'port': 12344,
    #     'rom_path': 'bn6,0',
    #     # 'rom_path': 'bn6,1',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
    #     # 'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
    #     'name': 'Instance 1',
    #     # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20230929001213-ummm-bn6-vs-DthKrdMnSP-round1-p1.tangoreplay',
    #     # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20230929001213-ummm-bn6-vs-IndianaOrz-round1-p2.tangoreplay',
    #     # 'replay_path':'/home/lee/Documents/Tango/replaysOrig/20231006015542-lunazoe-bn6-vs-IndianaOrz-round3-p2.tangoreplay',#player 2 cross change emotion state check fix needed
    #     'replay_path':'/home/lee/Documents/Tango/replaysOrig/20231006020253-lunazoe-bn6-vs-DthKrdMnSP-round1-p1.tangoreplay',
    #     'init_link_code': 'arena1',
    #     'is_player': False  # Set to True if you don't want this instance to send inputs
    # },
    # {
    #     'address': '127.0.0.1',
    #     'port': 12345,
    #     # 'rom_path': 'bn6,0',
    #     'rom_path': 'bn6,1',
    #     # 'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
    #     'name': 'Instance 2',
    #     'init_link_code': 'arena1',
    #     'is_player': False  # Set to False if you want this instance to send inputs
    # },
    # Additional instances can be added here
]

# Paths
SAVES_DIR = '/home/lee/Documents/Tango/saves'

GREGAR_CROSSES = [
    "Fire",
    "Elec",
    "Slash",
    "Erase",
    "Charge",
]

FALZAR_CROSSES = [
    "Aqua",
    "Thawk",
    "Tengu",
    "Grnd",
    "Dust",
]

FORM_MAPPING = [
    {
        "type": 'Normal',
        "normal":0,
        "beast":11,
    },
    {
        "type": 'Fire',
        "normal":1,
        "beast":13,
    },  
    {
        "type": 'Elec',
        "normal":2,
        "beast":14,
    },
    {
        "type": 'Slash',
        "normal":3,
        "beast":15,
    },
    {
        "type": 'Erase',
        "normal":4,
        "beast":16,
    },
    {
        "type": 'Charge',
        "normal":5,
        "beast":17,
    },
    {
        "type": 'Aqua',
        "normal":6,
        "beast":18,
    },
    {
        "type": 'Thawk',
        "normal":7,
        "beast":19,
    },
    {
        "type": 'Tengu',
        "normal":8,
        "beast":20,
    },
    {
        "type": 'Grnd',
        "normal":9,
        "beast":21,
    },
    {
        "type": 'Dust',
        "normal":10,
        "beast":22,
    },
]



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
    #read from processed replay file
    if os.path.exists('processed_replays.txt'):
        with open('processed_replays.txt', 'r') as f:
            processed_replays = f.read().splitlines()
            
    #remove processed replays from replay_files
    replay_files = [replay for replay in replay_files if os.path.basename(replay) not in processed_replays]
    if not replay_files:
        print("No replay files found.")
        exit()
        
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
    #write a file with all processed replays
    with open('processed_replays.txt', 'w') as f:
        for replay in processed_replays:
            f.write(replay + '\n')

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


# Define button mapping
BUTTONS = [
    ('MENU2', 15-9),    # 0000001000000000
    ('MENU', 15-8),     # 0000000100000000
    ('DOWN', 15-7),     # 0000000010000000
    ('UP', 15-6),       # 0000000001000000
    ('LEFT', 15-5),     # 0000000000100000
    ('RIGHT', 15-4),    # 0000000000010000
    ('SHOOT', 15-1),    # 0000000000000010
    ('CHIP', 15-0)      # 0000000000000001
]

def convert_button_16_bit_string(button_name):
    """
    Converts the button name to a 16-bit binary string.
    """
    bit_pos = next((pos for name, pos in BUTTONS if name == button_name), None)
    if bit_pos is not None:
        return format(1 << bit_pos, '016b')
    else:
        raise ValueError(f"Button '{button_name}' not found in BUTTONS.")



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


from planning_model import PlanningModel, get_planning_input_from_replay

inference_battle_model = None

# Function to load the AI models
# Initialize Separate Models for Inference and Training

def load_models(image_memory=1, learning_rate=1e-3):
    """
    Loads separate Inference and Training models for both Planning and Battle.
    If checkpoints do not exist, initializes new models.
    Utilizes the configuration parameters.
    """
    global inference_planning_model, training_planning_model
    global inference_battle_model
    global optimizer_planning, optimizer_battle
    global latest_checkpoint_number  # Access the global variable
    
    # Define the root directory
    root_dir = get_root_dir()
    
    # Load Training Planning Model
    training_planning_checkpoint_path = get_latest_checkpoint(model_type='planning', image_memory=image_memory)
    
    if training_planning_checkpoint_path:
        training_planning_model = PlanningModel().to(device)
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
        training_planning_model = PlanningModel().to(device)
        print("No Training Planning Model checkpoint found. Initialized a new Training Planning Model.")
    
    training_planning_model.train()  # Set to train mode
    
    # Load Training Battle Models for Each Button
    # training_battle_models = {}
    # for button_name, bit_pos in BUTTONS:
    #     training_battle_checkpoint_path = get_latest_checkpoint(model_type='battle', image_memory=image_memory, button=button_name)
        
    #     if training_battle_checkpoint_path:
    #         model = BattleNetworkModel(memory=image_memory).to(device)
    #         checkpoint_training_battle = torch.load(training_battle_checkpoint_path, map_location=device)
    #         if 'model_state_dict' in checkpoint_training_battle:
    #             model.load_state_dict(checkpoint_training_battle['model_state_dict'])
    #             print(f"Training Battle Model for {button_name} loaded from {training_battle_checkpoint_path}")
    #             # Extract the checkpoint number
    #             latest_number = extract_number_from_checkpoint(training_battle_checkpoint_path)
    #             latest_checkpoint_number['battle'] = latest_number
    #         else:
    #             # Load directly if 'model_state_dict' is not present
    #             model.load_state_dict(torch.load(training_battle_checkpoint_path))
    #             latest_number = extract_number_from_checkpoint(training_battle_checkpoint_path)
    #             latest_checkpoint_number['battle'] = latest_number
    #     else:
    #         # Initialize new Training Battle Model
    #         model = BattleNetworkModel(memory=image_memory).to(device)
    #         print(f"No Training Battle Model checkpoint found for {button_name}. Initialized a new Training Battle Model.")
        
    #     model.train()  # Set to train mode
    #     training_battle_models[button_name] = model
    
    # Initialize Inference Battle Models as copies of Training Battle Models
    # inference_battle_models = {}
    # for button_name, model in training_battle_models.items():
    #     inference_model = BattleNetworkModel(memory=image_memory).to(device)
    #     inference_model.load_state_dict(model.state_dict())
    #     inference_model.eval()
    #     # inference_model.half()  # Use half precision if desired
    #     inference_battle_models[button_name] = inference_model
    
    # Load the unified model
    unified_model = BattleNetworkModel(image_option='None', memory=image_memory, scale=1.0, dropout_p=0.5, output_size=8)


    checkpoint_dir = os.path.join(root_dir, f"checkpoints/battle/{image_memory}")
    unified_checkpoint_dir = os.path.join(checkpoint_dir)  # Example directory


    # Find the latest checkpoint for the unified model
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

    unified_model.to(device)
    unified_model.eval()
    
    #load the unified single model
    training_battle_checkpoint_path = get_latest_checkpoint(model_type='battle', image_memory=image_memory)
    
    
    # Assign to global inference models
    inference_battle_model = unified_model
    print(f"Unified Model loaded from {latest_checkpoint}")
    
    # Initialize Inference Planning Model as a copy of Training Planning Model
    inference_planning_model = PlanningModel().to(device)
    inference_planning_model.load_state_dict(training_planning_model.state_dict())
    inference_planning_model.eval()
    # inference_planning_model.half()
    
    # Initialize separate optimizers for Training Models
    optimizer_planning = optim.Adam(training_planning_model.parameters(), lr=learning_rate)
    # optimizer_battle = {button: optim.Adam(model.parameters(), lr=learning_rate) for button, model in training_battle_models.items()}
    
    print("All models loaded and initialized successfully.")



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
# optimizer = optim.Adam(training_battle_model.parameters(), lr=learning_rate)
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


from planning_model import get_planning_input_from_instance, encode_used_crosses, encode_beast_flags,encode_current_cross,encode_folder,encode_visible_chips
# Function to perform inference with the AI model
# previous_sent = 0

def predict(port, current_data_point, inside_window):
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
    global window_entry_time, previous_sent_dict, previous_inside_window_dict, INSTANCES

    current_time = time.time()
    
    
    
    # Detect entering or exiting the window
    if inside_window.item() == 1.0 and previous_inside_window_dict[port] == 0.0:
        #just entered window, set plan and reset values
        current_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)

        current_instance['set_cross'] = False
        current_instance['target_index'] = 0
        current_instance['press_count'] = 0
        current_instance['last_pressed_time'] = current_time +0.1

        #todo: get from model
        # current_instance['cross_target'] = 4
        # current_instance['target_list'] = [5,6,0,11,3, 10]

        chip_slots = current_instance['chip_slots']
        chip_codes = current_instance['chip_codes']
        
        chips_visible = current_instance['chip_visible_count']

        chip_data = []
        #convert codes and slots to the appropriate format. 
        for i in range(len(chip_slots)):
            if i >= chips_visible:
                break
            #if the code is event, add to the chip_data
            if chip_codes[i] % 2 == 0:
                chip_data.append((chip_slots[i], chip_codes[i] / 2))
            else:
                #if the code is odd, add to the chip_data
                chip_data.append((chip_slots[i] + 256, (chip_codes[i] - 1) / 2))
        print(chip_data)
        
        #set game instance chip data
        current_instance['player_chips'] = chip_data
        
        inputs, cross_target, target_list = get_planning_input_from_instance(inference_planning_model, current_instance, GAMMA, device)
        target_list_input = []
        for i in range(len(target_list)):
            target_list_input.append(target_list[i])
            
        
        #log the inputs and outputs to the port for training
        data_point = {
            'inputs': inputs,
            'cross_target': cross_target + 1,
            'target_list': target_list_input
        }
        planning_data_buffers[port].append(data_point)
        
        #set values higher than chip visible to 0
        for i in range(len(target_list)):
            if target_list[i] >= chips_visible:
                target_list[i] = 0
                
                
        #get the min index of any chip_data with a slot value > 360
        min_null = 1000
        for i in range(len(chip_data)):
            if chip_data[i][0] > 360:
                min_null = i
                print(f"detected null at [{i}]")
                break
            
        #set values higher than min_null to 0 as well
        for i in range(len(target_list)):
            if target_list[i] >= min_null:
                target_list[i] = 0
        
        #remove all 0 values from the target_list
        target_list = list(filter(lambda a: a != 0, target_list))
        #subtract 1 from all values from the target list
        target_list = list(map(lambda a: a - 1, target_list))
        
        
        current_instance['target_list'] = target_list
        print(f"Port {port}: Cross Target: {cross_target}")
        print(f"Port {port}: Target List: {target_list}")
        


        #randomly select value for cross_target
        #get count of available crosses
        # used_crosses = len(current_instance['player_used_crosses']) - 1
        # print(used_crosses)
        # current_instance['cross_target'] = random.randint(-1, used_crosses)

        # #randomly select target_list
        # current_instance['target_list'] = random.sample(range(chips_visible), 5)

        # #prune the same nubmers
        # current_instance['target_list'] = list(set(current_instance['target_list']))

        # #shuffle
        # random.shuffle(current_instance['target_list'])

        # #if the list contains less than 5 items, 25 percent chance to add 11 to the list
        # if len(current_instance['target_list']) < 5:
        #     if random.random() > 0.75:
        #         current_instance['target_list'].append(11)
        #         print("Added beast out to the list")

        # #print the random targets and cross
        # print(f"Port {port}: Cross Target: {current_instance['cross_target']}")
        # print(f"Port {port}: Target List: {current_instance['target_list']}")

        #add 10 to the end of the list
        current_instance['target_list'].append(10)


        
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
                        #     save_data_to_hdf5,
                        #     port,
                        #     planning_data_buffers[port],
                        #     model_type="Planning_Model"
                        # ))
                        # training_queue.put((port, list(data_buffers[port]), "Planning_Model", True))
                        # print(f"Port {port}: Submitted {len(planning_data_buffers[port])} data points for Planning Model training.")
                    # Clear the buffer after trainingbeast_flags_encoded
                    # planning_data_buffers[port].clear()
                    data_point['assigned_reward']=True
                    current_round_damage[port] = 0.0

        window_entry_time[port] = current_time
        
        current_round_damage[port] = 0.0
        # Ensure the planning data buffer is empty
        # planning_data_buffers[port].clear()
        # data_point['assigned_reward']=True
        # print(f"Port {port}: Entered window. Timer started.")

    # If inside the window, check if the timer has expired
    if inside_window.item() == 1.0:
        elapsed_time = current_time - window_entry_time.get(port, current_time)
        if elapsed_time >= CHIP_WINDOW_TIMER:
            if force_skip_chip_window:
                if previous_sent_dict[port] == 0:
                    previous_sent_dict[port] = 1
                    # print(f"Port {port}: Sending first key in sequence.")
                    return {'type': 'key_press', 'key': '0000000000001000'}
                elif previous_sent_dict[port] == 1:
                    previous_sent_dict[port] = 2
                    # print(f"Port {port}: Sending second key in sequence.")
                    return {'type': 'key_press', 'key': '0000000000001000'}
                elif previous_sent_dict[port] == 2:
                    previous_sent_dict[port] = 0
                    # print(f"Port {port}: Sending third key in sequence.")
                    return {'type': 'key_press', 'key': '0000000000000001'}
                    

    # # If not inside the window, reset the tracking variables
    # elif inside_window.item() == 0.0:
    #     if port in window_entry_time:
    #         del window_entry_time[port]
    #     previous_sent_dict[port] = 0
    #     previous_inside_window_dict[port] = 0.0
    
    #make sure cross_target is set
    current_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
    if inside_window.item() == 1.0 and 'cross_target' in current_instance:
            #automate the window
            # print(f"Port {port}: Inside window. Navigating the menu.")

            cross_target = current_instance['cross_target']
            target_list = current_instance['target_list']

            #navigate the menu based on the selected index
            current_index = current_instance['selected_menu_index']
            current_cross_index = current_instance['selected_cross_index']
            current_inside_cross_window = current_instance['inside_cross_window']
            # print(f"Port {port}: Inside window. Navigating the menu.")
            # print(f"Port {port}: Current Index: {current_index}")
            # print(f"Port {port}: Current Cross Index: {current_cross_index}")
            # print(f"Port {port}: Inside Cross Window: {current_inside_cross_window}")

            
            if 'set_cross' not in current_instance:
                current_instance['set_cross'] = False

            #set instance's current target index if not set
            if 'target_index' not in current_instance:
                current_instance['target_index'] = 0
            
            #get the target index if in range
            if current_instance['target_index'] < len(target_list):
                target_index = target_list[current_instance['target_index']]
            else:
                #done selecting
                # print(f"Port {port}: Done selecting. Sending key press.")
                return {'type': 'key_press', 'key': '0000000000000000'}

            #set last pressed time if it isn't set
            if 'last_pressed_time' not in current_instance:
                current_instance['last_pressed_time'] = 0
            # do nothing if it hasn't been 1 second since last time
            # print(f"Port {port}: Target Index: {target_index}. current_index: {current_index}")
            time_difference = current_time - current_instance['last_pressed_time']
            if  time_difference < 0.25 and time_difference > 0.1:
                # print(f"Port {port}: Skipping key press. Not enough time elapsed.")
                return {'type': 'key_press', 'key': '0000000000000000'}
            


            if cross_target != -1 and current_instance['set_cross'] == False:
                if current_inside_cross_window == 0:
                    #press up to get in the menu
                    # print(f"Port {port}: Moving up. Target index: {cross_target}. Current index: {current_cross_index}")
                    current_instance['last_pressed_time'] = current_time    
                    return {'type': 'key_press', 'key': '0000000001000000'}
                else:
                    if cross_target < current_cross_index:
                        current_instance['press_count'] = 0
                        # print(f"Port {port}: Moving up. Target index: {cross_target}. Current index: {current_cross_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000001000000'}
                    elif cross_target > current_cross_index:
                        current_instance['press_count'] = 0
                        # print(f"Port {port}: Moving down. Target index: {cross_target}. Current index: {current_cross_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000010000000'}
                    elif cross_target == current_cross_index:
                        # print(f"Port {port}: Target index reached. Sending key press.")
                        #increment target index for the instance
                        if 'press_count' not in current_instance:
                            current_instance['press_count'] = 0
                        
                        if current_instance['press_count'] < 25:
                            current_instance['press_count'] += 1
                            return {'type': 'key_press', 'key': '0000000000000000'}  
                        else:
                            current_instance['press_count'] += 1
                            if current_instance['press_count'] > 30:
                                current_instance['set_cross'] = True
                                current_instance['press_count'] = 0
                            if current_instance['press_count'] % 2 == 0:
                                return {'type': 'key_press', 'key': '0000000000000001'}
                            else:
                                return {'type': 'key_press', 'key': '0000000000000000'}  

            else:
                if current_index == target_index:
                    # print(f"Port {port}: Target index reached. Sending key press.")
                    #increment target index for the instance
                    if 'press_count' not in current_instance:
                        current_instance['press_count'] = 0
                    if current_instance['press_count'] > 50:
                        current_instance['press_count'] = 0
                        current_instance['target_index'] = (current_instance['target_index'] + 1)
                        # print(f"Port {port}: Incremented target index to {current_instance['target_index']}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000000000'}
                    
                    elif current_instance['press_count'] % 2 == 0:
                        current_instance['press_count'] += 1
                        return {'type': 'key_press', 'key': '0000000000000000'}  
                    else:
                        current_instance['press_count'] += 1
                        return {'type': 'key_press', 'key': '0000000000000001'}
                    
                    # current_instance['target_index'] = (current_instance['target_index'] + 1)
                    # print(f"Port {port}: Incremented target index to {current_instance['target_index']}")
                    # current_instance['last_pressed_time'] = current_time  
                else:
                    #on top row (0-4), move right
                    #when 10 is the target index, move right
                    if current_inside_cross_window != 0:
                        # print(f"Port {port}: Backing out of cross window. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000000000010'}
                    elif target_index == 10 and current_index != 10:
                        # print(f"Port {port}: Moving to okay. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000000001000'}
                    elif target_index != 10 and current_index == 10 and target_index != 11:
                        # print(f"Port {port}: Moving left. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000000100000'}
                    elif target_index == 11 and current_index != 10 and current_index != 11:
                        # print(f"Port {port}: Moving right. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000000010000'}
                    elif target_index != 11 and current_index == 11:
                        # print(f"Port {port}: Moving up. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000001000000'}
                    elif target_index == 11 and current_index == 10 and current_index != 11:
                        # print(f"Port {port}: Moving down. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time    
                        return {'type': 'key_press', 'key': '0000000010000000'}
                    elif current_index < target_index and target_index < 5 and current_index < 5:
                        # print(f"Port {port}: Moving right. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000010000'}
                    #on top row (0-4), move left
                    elif current_index > target_index and target_index < 5 and current_index < 5:
                        # print(f"Port {port}: Moving left. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000100000'}
                    #on bottom row (5-9), move right
                    elif current_index < target_index and target_index >= 5 and current_index >= 5:
                        # print(f"Port {port}: Moving right. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000010000'}
                    #on bottom row (5-9), move left
                    elif current_index > target_index and target_index >= 5 and current_index >= 5:
                        # print(f"Port {port}: Moving left. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000100000'}
                    #on top row (0-2), move down
                    elif current_index < target_index and target_index >= 5 and current_index < 5 and current_index <=0:
                        # print(f"Port {port}: Moving down. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000010000000'}
                    #on top row >0, move left
                    elif current_index < target_index and target_index >= 5 and current_index < 5 and current_index >0:
                        # print(f"Port {port}: Moving left. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000000100000'}
                    #on bottom row(5-9), move up
                    elif current_index > target_index and target_index < 5 and current_index >= 5:
                        # print(f"Port {port}: Moving up. Target index: {target_index}. Current index: {current_index}")
                        current_instance['last_pressed_time'] = current_time
                        return {'type': 'key_press', 'key': '0000000001000000'}

    

    #return at this point if planning model
    if inside_window.item() == 1.0:
        return {'type': 'key_press', 'key': '0000000000000000'}
    
    
    predicted_input_str = None
    model_type = "Battle_Model"  # Default model


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
    if model_type != "Battle_Model" and selected_model is None:
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
    memory= get_image_memory() 
    
    if model_type == "Battle_Model":
        # Acquire the appropriate lock before performing model inference
        
         # Prepare the sequence
        sequence = prepare_inference_sequence(data_buffers[port], current_data_point, memory=memory)
        
        # Ensure the sequence has exactly 'memory' gamestates
        assert len(sequence) == memory, f"Sequence length {len(sequence)} != memory {memory}"
        
        # Move sequence data to device
        # Each gamestate in the sequence is a dict; models expect a list of gamestates
        sequence_device = []
        for gamestate in sequence:
            gamestate_device = {}
            for key, value in gamestate.items():
                if key in ['action', 'reward']:
                    continue  # Skip action and reward if not needed for inference
                if isinstance(value, list):
                    # If the value is a list of tensors, move each tensor to device
                    gamestate_device[key] = [tensor.to(device) for tensor in value]
                else:
                    # Move tensor to device
                    gamestate_device[key] = value.to(device)
            sequence_device.append(gamestate_device)
        
        # Initialize a dictionary to hold predicted probabilities for each button
        button_predictions = {}
        
        # with torch.no_grad():
        #     predictions = {}
            # for button_name, bit_pos in BUTTONS:
            #     model = inference_battle_models.get(button_name)
            #     if not model:
            #         print(f"No inference model loaded for button: {button_name}. Skipping.")
            #         continue
            #     try:
            #         # Forward pass
            #         output = model(sequence_device)  # Adjust based on your model's forward method
            #         # predictions[button_name] = output
            #         predicted_prob = output.item()  # Get the prediction for the last gamestate
            #         button_predictions[button_name] = predicted_prob
            #         max_prob = max(max_prob, predicted_prob)
            #     except Exception as e:
            #         print(f"Error during forward pass for {button_name}: {e}")
            #         button_predictions[button_name] = 0.0  # Default to no press on error
        # Perform inference
        with torch.no_grad():
            # print("\n--- Inference Results ---")
            try:
                outputs = inference_battle_model(sequence_device)  # Shape: (batch_size=1, 8)
                predicted_probs = outputs.squeeze(0)  # Shape: (8,)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                return

            # Define a threshold or use adaptive thresholding
            # Example using a fixed threshold
            # Define a threshold for button presses
            command_threshold = 0.01  # Adjust based on your requirements

            # Initialize the binary command string with all '0's
            predicted_input_list = list("0000000000000000")

            # Track the maximum probability for logging
            max_prob = torch.max(predicted_probs).item()
            # Define button names in the order of the output
            button_names = [btn[0] for btn in BUTTONS]
            # Map each predicted probability to its corresponding button
            for idx, btn in enumerate(button_names):
                prob = predicted_probs[idx].item()
                predicted = '1' if prob >= command_threshold else '0'
                bit_pos = BUTTONS[idx][1]
                predicted_input_list[bit_pos] = predicted

            # Convert the list back to a string
            predicted_input_str = ''.join(predicted_input_list)

            # Display the results
            # print(f"\nPredicted Button Presses: {predicted_input_str}")
            # print(f"Predicted Probabilities: {predicted_probs.cpu().numpy()}")
            # print(f"Maximum Probability in this Command: {max_prob:.3f}")
            
            print(f"Port {port}: Predicted Input: {predicted_input_str} | {max_prob:.3f}")

        # --- Start of Tally Update ---
        # with input_tally_lock:
        #     for key, bit_pos in KEY_BIT_POSITIONS.items():
        #         if predicted_inputs[15 - bit_pos] == 1:
        #             input_tally[key] += 1

        #never allow sending the pause command when out of the window
        # if inside_window.item() == 0.0 and predicted_input_str[15 - KEY_BIT_POSITIONS['RETURN']] == '1':
        #     predicted_input_str = '0000000000000000'
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
        # else:
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
    global max_reward, max_punishment, prune
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

                        #attach to game instance
                        current_player_emotion = screen_data.get("player_emotion", 0)
                        current_enemy_emotion = screen_data.get("enemy_emotion", 0)
                        current_player_game_emotion = screen_data.get("player_game_emotion", 0)
                        current_enemy_game_emotion = screen_data.get("enemy_game_emotion", 0)
                        selected_menu_index = screen_data.get("selected_menu_index", 0)
                        selected_cross_index = screen_data.get("selected_cross_index", 0)
                        chip_selected_count = screen_data.get("chip_selected_count", 0)
                        chip_visible_count = screen_data.get("chip_visible_count", 0)
                        chip_slots = screen_data.get("chip_slots", 0)
                        chip_codes = screen_data.get("chip_codes", 0)
                        selected_chip_indices = screen_data.get("selected_chip_indices", 0)
                        beast_out_selectable = screen_data.get("beast_out_selectable", 0)
                        inside_cross_window = screen_data.get("inside_cross_window", 0)
                        player_chip_folder = screen_data.get("player_chip_folder", 0)
                        enemy_chip_folder = screen_data.get("enemy_chip_folder", 0)
                        player_code_folder = screen_data.get("player_code_folder", 0)
                        enemy_code_folder = screen_data.get("enemy_code_folder", 0)
                        player_tag_chips = screen_data.get("player_tag_chips", 0)
                        enemy_tag_chips = screen_data.get("enemy_tag_chips", 0)
                        player_reg_chip = screen_data.get("player_reg_chip", 0)
                        enemy_reg_chip = screen_data.get("enemy_reg_chip", 0)
                        grid_state = screen_data.get("grid_state", 0)
                        grid_owner_state = screen_data.get("grid_owner_state", 0)
                        player_grid_position = screen_data.get("player_grid_position", 0)
                        enemy_grid_position = screen_data.get("enemy_grid_position", 0)
                        is_offerer = screen_data.get("is_offerer", 0)
                        cust_gage = screen_data.get("cust_gage", 0)
                        own_navi_cust = screen_data.get("own_navi_cust", 0)
                        enemy_navi_cust = screen_data.get("enemy_navi_cust", 0)

                    except json.JSONDecodeError:
                        print(f"Port {port}: Failed to parse screen_image details: {details}")
                        continue


                    #attach fields to the game instance
                    game_instance['player_health'] = current_player_health
                    game_instance['enemy_health'] = current_enemy_health
                    game_instance['player_position'] = current_player_position
                    game_instance['enemy_position'] = current_enemy_position
                    game_instance['inside_window'] = current_inside_window
                    game_instance['player_charge'] = current_player_charge
                    game_instance['enemy_charge'] = current_enemy_charge
                    game_instance['player_chip'] = current_player_chip
                    game_instance['enemy_chip'] = current_enemy_chip
                    game_instance['player_emotion'] = current_player_emotion
                    game_instance['enemy_emotion'] = current_enemy_emotion
                    game_instance['player_game_emotion'] = current_player_game_emotion
                    game_instance['enemy_game_emotion'] = current_enemy_game_emotion
                    game_instance['selected_menu_index'] = selected_menu_index
                    game_instance['selected_cross_index'] = selected_cross_index
                    game_instance['chip_selected_count'] = chip_selected_count
                    game_instance['chip_visible_count'] = chip_visible_count
                    game_instance['chip_slots'] = chip_slots
                    game_instance['chip_codes'] = chip_codes
                    game_instance['selected_chip_indices'] = selected_chip_indices
                    game_instance['beast_out_selectable'] = beast_out_selectable
                    game_instance['inside_cross_window'] = inside_cross_window
                    game_instance['grid_state'] = grid_state
                    game_instance['grid_owner_state'] = grid_owner_state
                    game_instance['player_grid_position'] = player_grid_position
                    game_instance['enemy_grid_position'] = enemy_grid_position
                    game_instance['own_navi_cust'] = own_navi_cust
                    game_instance['enemy_navi_cust'] = enemy_navi_cust
                    
                    
                    #gridx positionx mapping
                    # 1-20 2-60 3-100 4-140 5-180 6-220
                    # Define the x mappings
                    x_mapping = {0: 0, 1: 20, 2: 60, 3: 100, 4: 140, 5: 180, 6: 220}
                    x_mapping_inverted = {v: k for k, v in x_mapping.items()}

                    def get_closest_x(x, mapping_values):
                        """
                        Find the closest value in mapping_values to the given x.
                        """
                        return min(mapping_values, key=lambda k: abs(k - x))
                    
                    
                    # Find the closest x values from the mapping
                    
                    if(current_player_position is None):
                        current_player_position = [0,0]
                    if(current_enemy_position is None):
                        current_enemy_position = [0,0]
                    closest_player_x = get_closest_x(current_player_position[0], x_mapping.values())
                    closest_enemy_x = get_closest_x(current_enemy_position[0], x_mapping.values())
                    
                    # Map the closest x values to grid indices with error handling
                    try:
                        player_grid_x = x_mapping_inverted[closest_player_x]
                    except KeyError:
                        print(f"Error: Closest Player X position {closest_player_x} not found in x_mapping_inverted.")
                        player_grid_x = 0  # Assign a default or handle as needed

                    try:
                        enemy_grid_x = x_mapping_inverted[closest_enemy_x]
                    except KeyError:
                        print(f"Error: Closest Enemy X position {closest_enemy_x} not found in x_mapping_inverted.")
                        enemy_grid_x = 0  # Assign a default or handle as needed
                   
                    #get the closest x value to the mapping from the current position (so 57 should map to 60)
                    player_grid_position[0] = player_grid_x#x_mapping_inverted[current_player_position[0]]
                    #y1 is about 260, y2 is about 515, y3 is about 770
                    player_grid_position[1] = 1 if current_player_position[1] < 300 else 2 if current_player_position[1] < 600 else 3
                    enemy_grid_position[0] = enemy_grid_x#x_mapping_inverted[current_enemy_position[0]]
                    enemy_grid_position[1] = 1 if current_enemy_position[1] < 300 else 2 if current_enemy_position[1] < 600 else 3
                    
                    game_instance['player_grid_position'] = player_grid_position
                    game_instance['enemy_grid_position'] = enemy_grid_position
                    game_instance['cust_gage'] = cust_gage
                    #print cust gage
                    # print(f"Port {port}: Cust Gage: {cust_gage}")

                    
                    #print grid state in a 6x3 grid
                    # for i in range(0, 6):
                    #     print(f"Port {port}: {grid_state[i*3:i*3+3]}")
                    # for i in range(0, 6):
                    #     print(f"Port {port}: {grid_owner_state[i*3:i*3+3]}")
                    # print(f"Port {port}: Player GP: {game_instance['player_grid_position']} {is_offerer}")
                    # print(f"Port {port}: Enemy GP: {game_instance['enemy_grid_position']} {is_offerer}")
                    # # #print player and enemy position
                    # print(f"Port {port}: Player Position: {current_player_position}")
                    # print(f"Port {port}: Enemy Position: {current_enemy_position}")
                    if 'player_beasted_out' not in game_instance:
                        game_instance['player_beasted_out'] = False
                    if 'enemy_beasted_out' not in game_instance:
                        game_instance['enemy_beasted_out'] = False
                    if 'player_beasted_over' not in game_instance:
                        game_instance['player_beasted_over'] = False
                    if 'enemy_beasted_over' not in game_instance:
                        game_instance['enemy_beasted_over'] = False
                        
                    game_instance['player_beasted_out'] = game_instance['player_beasted_out'] or game_instance['player_game_emotion'] >= 11
                    game_instance['enemy_beasted_out'] = game_instance['enemy_beasted_out'] or game_instance['enemy_game_emotion'] >= 11
                    game_instance['player_beasted_over'] = game_instance['player_beasted_over'] or game_instance['player_game_emotion'] >= 23
                    game_instance['enemy_beasted_over'] = game_instance['enemy_beasted_over'] or game_instance['enemy_beasted_over'] >= 23
                    #only get this if player_chip_folder hasn't been set before
                    if 'player_chip_folder' not in game_instance:
                        game_instance['player_chip_folder'] = player_chip_folder
                        game_instance['enemy_chip_folder'] = enemy_chip_folder
                        game_instance['player_code_folder'] = player_code_folder
                        game_instance['enemy_code_folder'] = enemy_code_folder

                        #combine player chip and code into player_folder objectarray with chip and code fields
                        game_instance['player_folder'] = []
                        game_instance['enemy_folder'] = []

                        for i in range(0, len(player_chip_folder)):
                            player_tagged = i in player_tag_chips
                            enemy_tagged = i in enemy_tag_chips
                            player_regged = i == player_reg_chip
                            enemy_regged = i == enemy_reg_chip
                            game_instance['player_folder'].append({'chip': player_chip_folder[i], 'code': player_code_folder[i], 'used': False, 'tagged': player_tagged, 'regged': player_regged})
                            game_instance['enemy_folder'].append({'chip': enemy_chip_folder[i], 'code': enemy_code_folder[i], 'used': False, 'tagged': enemy_tagged, 'regged': enemy_regged})
                        # print player chip folder and enemy code folder from instance
                        # print(f"Port {port}: Player Chip Folder: {game_instance['player_folder']}")
                        # print(f"Port {port}: Enemy Chip Folder: {game_instance['enemy_folder']}")
                        


                    if(current_reward !=0):
                        print(f"Port {port}: Reward: {current_reward}")
                    if(current_punishment !=0):
                        print(f"Port {port}: Punishment: {current_punishment}")



                    #selected_chip_indices
                    # print(f"Port {port}: {selected_chip_indices}")
                    #set available crosses based on which crosses have been used
                    if 'player_used_crosses' not in game_instance:
                        #initialize available crosses to include all
                        game_instance['player_used_crosses'] = []
                    
                    selected_cross_form = next((cross for cross in FORM_MAPPING if cross['normal'] == current_player_game_emotion or cross['beast'] == current_player_game_emotion), None)
                    if selected_cross_form is not None:
                        #get the index of the form
                        selected_cross_index = FORM_MAPPING.index(selected_cross_form)
                        #add to used crosses list if it's not already there
                        if selected_cross_index >= 1 and selected_cross_index not in game_instance['player_used_crosses']:
                            game_instance['player_used_crosses'].append(selected_cross_index)
                            print(f"Port {port}: Adding to used crosses: {selected_cross_index}")
                            
                    if 'enemy_used_crosses' not in game_instance:
                        #initialize available crosses to include all
                        game_instance['enemy_used_crosses'] = []
                    
                    selected_cross_form = next((cross for cross in FORM_MAPPING if cross['normal'] == current_enemy_game_emotion or cross['beast'] == current_enemy_game_emotion), None)
                    if selected_cross_form is not None:
                        #get the index of the form
                        selected_cross_index = FORM_MAPPING.index(selected_cross_form)
                        #add to used crosses list if it's not already there
                        if selected_cross_index >= 1 and selected_cross_index not in game_instance['enemy_used_crosses']:
                            game_instance['enemy_used_crosses'].append(selected_cross_index)
                            print(f"Port {port}: Adding to used crosses: {selected_cross_index}")
                    
                    # print(f"Port {port}: Enemy Available Crosses: {game_instance['enemy_used_crosses']}")
                    
                    #print emotions and indicies
                    # print(f"Port {port}: Player Emotion: {current_player_emotion}, Enemy Emotion: {current_enemy_emotion}")
                    # print(f"Port {port}: Player Game Emotion: {current_player_game_emotion}, Enemy Game Emotion: {current_enemy_game_emotion}")
                    # print(f"Port {port}: Selected Menu Index: {selected_menu_index}, Selected Cross Index: {selected_cross_index}")
                    # print(f"Port {port}: Chip Selected Count: {chip_selected_count}, Chip Visible Count: {chip_visible_count}")
                    # print(f"Port {port}: Chip Slots: {chip_slots}, Chip Codes: {chip_codes}")
                    # print(f"Port {port}: Selected Chip Indices: {selected_chip_indices}")
                    # print(f"Port {port}: Beast Out Selectable: {beast_out_selectable}")


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
                    

                    # Compute additional tensors
                    inside_window_tensor = torch.tensor([current_inside_window], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    # Before passing health to the model
                    update_max_health(current_player_health, current_enemy_health)
                    normalized_player_health, normalized_enemy_health = normalize_health(current_player_health, current_enemy_health)
                    
                    #update the instance's player health to the normalized value
                    game_instance['player_health'] = normalized_player_health
                    game_instance['enemy_health'] = normalized_enemy_health

                    player_health_tensor = torch.tensor([normalized_player_health], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    enemy_health_tensor = torch.tensor([normalized_enemy_health], dtype=torch.float32, device=device).unsqueeze(0)    # Shape: (1, 1)




                    gamestate_tensor = get_gamestate_tensor(
                        screen_data,
                        cust_gage,#validated 2
                        grid_state,#validated 2
                        grid_owner_state,#validated 2
                        player_grid_position,#validated 2
                        enemy_grid_position,#validated 2
                        normalized_player_health,#validated2
                        normalized_enemy_health,#validated2
                        current_player_chip, #validated2
                        current_enemy_chip, #validated2
                        current_player_charge, #validated2
                        current_enemy_charge, #validated2
                        game_instance['current_hand'] if 'current_hand' in game_instance else None,#validated2
                        game_instance['player_folder'],#validated2
                        game_instance['enemy_folder'],#validated2
                        own_navi_cust,#validated2
                        enemy_navi_cust,#validated2
                        current_player_game_emotion,#validated2
                        current_enemy_game_emotion,#validated2
                        game_instance['player_used_crosses'],#validated2
                        game_instance['enemy_used_crosses'],#validated 2
                        game_instance['player_beasted_out'],#validated
                        game_instance['enemy_beasted_out'],#validated
                        game_instance['player_beasted_over'],
                        game_instance['enemy_beasted_over'],
                    )
                    
                    #print out count of one tensors in player folder
                    # print(f"Port {port}: PLAYER: {gamestate_tensor['player_chip_hand']}")
                    
                    #print out argmax of each tensor in player_chip_hand
                    # for i in range(0, len(gamestate_tensor['player_chip_hand'])):
                    #     print(f"Port {port}: {i}: {torch.argmax(gamestate_tensor['player_chip_hand'][i])}")
                    
                    


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
                            data_point = gamestate_tensor
                            
                            current_time = time.time()
                            # Detect entering or exiting the window
                            if inside_window_tensor.item() == 1.0 and previous_inside_window_dict[port] == 0.0:
                                #just entered window, set plan and reset values
                                current_instance = game_instance#next((inst for inst in INSTANCES if inst['port'] == port), None)

                                current_instance['set_cross'] = False
                                current_instance['target_index'] = 0
                                current_instance['press_count'] = 0
                                current_instance['last_pressed_time'] = current_time +0.1
                                chip_slots = current_instance['chip_slots']
                                
                                
                                chip_codes = current_instance['chip_codes']
                                chips_visible = current_instance['chip_visible_count']

                                chip_data = []
                                #convert codes and slots to the appropriate format. 
                                for i in range(len(chip_slots)):
                                    if i >= chips_visible:
                                        break
                                    #if the code is event, add to the chip_data
                                    if chip_codes[i] % 2 == 0:
                                        chip_data.append((chip_slots[i], chip_codes[i] / 2))
                                    else:
                                        #if the code is odd, add to the chip_data
                                        chip_data.append((chip_slots[i] + 256, (chip_codes[i] - 1) / 2))
                                print(chip_data)
                                
                                
                                
                                #set game instance chip data
                                current_instance['player_chips'] = chip_data
                                if is_replay:
                                    previous_inside_window_dict[port] = 1.0
                                
                                                
                                                
                            
                            if not game_instance.get('is_player', False) and not is_replay:
                                command = predict(
                                    port,
                                    data_point,
                                    inside_window_tensor,
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
                            
                            if inside_window_tensor.item() == 0.0:
                                # Append data_point to buffer
                                max_reward = 200#max(max_reward, current_reward)
                                max_punishment = 200#max(max_punishment, current_punishment)
                                # If we received a reward or punishment, perform online training
                                # Assign reward to the current data_point
                                data_buffers[port].append(data_point)
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
                                    for i, past_data_point in enumerate(reversed(data_buffers[port][-window_size:])):
                                        discounted_reward = reward_value * (gamma ** i)
                                        if 'reward' in past_data_point:
                                            past_data_point['reward'] += discounted_reward
                                        else:
                                            past_data_point['reward'] = discounted_reward

                                    # Reset rewards
                                    current_reward = 0
                                    current_punishment = 0

                                # Pruning Logic Starts Here
                                # Remove data points older than window_size that have no reward assigned
                                # print( len(data_buffers[port]))
                                # if prune:
                                #     if len(data_buffers[port]) > window_size:
                                #         # Calculate how many data points to prune
                                #         num_to_prune = len(data_buffers[port]) - window_size
                                #         pruned_count = 0
                                #         i = 0
                                #         while pruned_count < num_to_prune and i < len(data_buffers[port]):
                                #             if 'reward' not in data_buffers[port][i]:
                                #                 del data_buffers[port][i]
                                #                 pruned_count += 1
                                #             else:
                                #                 i += 1

                                current_reward = 0
                                current_punishment = 0
                                
                                if port in window_entry_time:
                                    del window_entry_time[port]
                                previous_sent_dict[port] = 0
                                if previous_inside_window_dict[port] == 1.0:
                                    #use the selected indicies to set the current hand's chip data
                                    #create a list of the data from chip slots at each selected indicies
                                    #game_instance['chip_slots'] game_instance['selected_chip_indices']
                                    #ensure there are any selected chips
                                    selected_chips = []
                                    for i in game_instance['selected_chip_indices']:
                                        if 0 <= i < len(game_instance['player_chips']):
                                            selected_chips.append(game_instance['player_chips'][i][0])
                                        else:
                                            # Handle the error, e.g., log a warning or skip the index
                                            print(f"Warning: Index {i} is out of range for player_chips.")                                    
                                            #remove selected chips which are higher than the amount selected count
                                    selected_chips = selected_chips[:game_instance['chip_selected_count']]
                                    print(f"Port {port}: Selected Chips: {selected_chips}")
                                    game_instance['current_hand'] = selected_chips
                                    #if exiting the window and the game is a replay
                                    if is_replay or game_instance.get('is_player', True):
                                        inputs, cross_target, target_list_input = get_planning_input_from_replay(game_instance, GAMMA, device)
                                        #append the selected data to the planning data buffer
                                        planning_data_point = {
                                            'inputs': inputs,
                                            'cross_target': cross_target,
                                            'target_list': target_list_input
                                        }
                                        print(f"Port {port}: Appending planning data to buffer.")
                                        planning_data_buffers[port].append(planning_data_point)
                                        
                                previous_inside_window_dict[port] = 0.0

                            previous_inside_window_dict[port] = inside_window_tensor.item()
                           
                           


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

def saving_thread_function(batch_size=100000000, max_wait_time=1.0):
    """
    Training thread that processes training data in batches from each port's data buffer.
    
    Args:
        batch_size (int): Number of training data items per batch.
        max_wait_time (float): Maximum time (in seconds) to wait before processing a batch.
    """
    global training_planning_model, optimizer_battle, optimizer_planning
    
    # Initialize the progress bar
    progress_bar = tqdm(desc="Training Progress", unit="batch")
    
    
    while True:
        start_time = time.time()
        any_data_processed = False
        
        # train battle model                            # else:
                            #     if current_player_charge != 0:# or current_punishment == 0:
                            #         # print(f"Port {port}: Assigning reward: {0.1}")
                            #         # Train on the current frame (data_point)
                            #         if current_player_charge != 0:
                            #             data_point['reward'] = 0.1
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
                save_data_to_hdf5(port, batch_data, model_type="Battle_Model", log=True)
                progress_bar.update(1)
            
        #remove all battle data from memory
        for port, buffer in data_buffers.items():
            lock = data_buffers_locks[port]
            with lock:
                buffer.clear()

        #train planning model
        batch_size = 1
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
                save_data_to_hdf5(port, batch_data, model_type="Planning_Model", log=True)
                progress_bar.update(1)
        
        if not any_data_processed:
            # If no data was processed, wait for a short duration
            elapsed = time.time() - start_time
            sleep_time = max_wait_time - elapsed
            if sleep_time > 0:
                break
                time.sleep(sleep_time)
        
        # Stop once all the data finished being processed
        if all(len(buffer) == 0 for buffer in data_buffers.values()):
            break

import random

def save_data_to_hdf5(port, data_buffer, model_type="Battle_Model", log=True):
    global training_planning_model, optimizer_battle, optimizer_planning

    # Select the appropriate training model and lock
    if model_type == "Battle_Model":
        # selected_model = training_battle_model
        selected_lock = training_battle_lock
        selected_optimizer = optimizer_battle
    elif model_type == "Planning_Model":
        selected_model = training_planning_model
        selected_lock = training_planning_lock
        selected_optimizer = optimizer_planning
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with selected_lock:
        # selected_model = selected_model.float()
        # selected_model.train()  # Switch to train mode

        batch_size = len(data_buffer)
        if batch_size == 0:
            print(f"Port {port}: No data to train on.")
            return

        if model_type == "Battle_Model":
            print(f"Port {port}: Preparing Battle_Model data with {batch_size} data points.")

            # Initialize dictionaries to collect batched inputs
            batched_features = {}  # To collect feature tensors, keys are feature names
            batched_actions = []   # To collect action tensors
            batched_rewards = []   # To collect rewards

            # For each data_point in data_buffer:
            for data_point in data_buffer:
                # data_point contains the features (keys and tensors), 'action', and 'reward' (optional)

                # For the features:
                for key, value in data_point.items():
                    if key == 'action' or key == 'reward':
                        continue  # Skip these keys
                    if key not in batched_features:
                        batched_features[key] = []

                    if isinstance(value, list):
                        # For lists (e.g., 'player_chip_hand')
                        # Concatenate tensors in the list along the feature dimension
                        concatenated_value = torch.cat(value, dim=1)  # Shape: (1, total_feature_size)
                        batched_features[key].append(concatenated_value)
                    else:
                        # Value is a tensor
                        batched_features[key].append(value)

                # For the action:
                action_bitstring = data_point['action']  # Should be a 16-character string of '0's and '1's
                # Convert to a tensor of shape (1, 16)
                action_tensor = torch.tensor([[int(bit) for bit in action_bitstring]], dtype=torch.float32)
                batched_actions.append(action_tensor)

                # For the reward:
                reward = data_point.get('reward', 0.0)
                batched_rewards.append(reward)

            # Now, convert batched_features to tensors
            for key in batched_features:
                #make sure everything is on the same device
                for i in range(len(batched_features[key])):
                    batched_features[key][i] = batched_features[key][i].to(device)
                batched_features[key] = torch.cat(batched_features[key], dim=0)  # Concatenate along batch dimension

            # Convert batched_actions to tensor
            batched_actions_tensor = torch.cat(batched_actions, dim=0)  # Shape: (batch_size, 16)

            # Convert batched_rewards to tensor
            batched_rewards_tensor = torch.tensor(batched_rewards, dtype=torch.float32)  # Shape: (batch_size,)

            # Prepare batch_data dictionary
            batch_data = batched_features  # Contains the feature tensors
            batch_data['action'] = batched_actions_tensor
            batch_data['reward'] = batched_rewards_tensor

            # Now, save batch_data to HDF5
            battle_data_dir = get_root_dir() + "/data/battle_data"
            os.makedirs(battle_data_dir, exist_ok=True)
            save_batch_to_hdf5(batch_data, battle_data_dir, port, model_type)
            
        elif model_type == "Planning_Model":
            # Updated Training Logic for Planning_Model
            # Initialize lists to collect batched inputs and targets
            batched_inputs = {
                'player_folder': {
                    'chips_onehot': [],
                    'codes_onehot': [],
                    'flags': []
                },
                'enemy_folder': {
                    'chips_onehot': [],
                    'codes_onehot': [],
                    'flags': []
                },
                'visible_chips': {
                    'chips_onehot': [],
                    'codes_onehot': []
                },
                'health': [],
                'current_crosses': [],
                'used_crosses': [],
                'beast_flags': []
            }
            cross_targets = []
            target_lists = []
            rewards_batch = []

            for data_point in data_buffer:
                # Extract 'inputs', 'cross_target', 'target_list', and 'reward'
                inputs = data_point['inputs']
                cross_target = data_point['cross_target']
                target_list = data_point['target_list']
                reward = data_point.get('reward', 0.0)

                # Append inputs
                for key in batched_inputs['player_folder']:
                    batched_inputs['player_folder'][key].append(inputs['player_folder'][key])
                for key in batched_inputs['enemy_folder']:
                    batched_inputs['enemy_folder'][key].append(inputs['enemy_folder'][key])
                for key in batched_inputs['visible_chips']:
                    batched_inputs['visible_chips'][key].append(inputs['visible_chips'][key])

                batched_inputs['health'].append(inputs['health'])
                batched_inputs['current_crosses'].append(inputs['current_crosses'])
                batched_inputs['used_crosses'].append(inputs['used_crosses'])
                batched_inputs['beast_flags'].append(inputs['beast_flags'])

                # Append targets and rewards
                cross_targets.append(cross_target)
                target_lists.append(target_list)
                rewards_batch.append(reward)

            # Convert batched_inputs lists to tensors
            for key in batched_inputs['player_folder']:
                batched_inputs['player_folder'][key] = torch.cat(batched_inputs['player_folder'][key], dim=0)  # Shape: (batch_size, 30, 400) etc.
            for key in batched_inputs['enemy_folder']:
                batched_inputs['enemy_folder'][key] = torch.cat(batched_inputs['enemy_folder'][key], dim=0)
            for key in batched_inputs['visible_chips']:
                batched_inputs['visible_chips'][key] = torch.cat(batched_inputs['visible_chips'][key], dim=0)
            batched_inputs['health'] = torch.cat(batched_inputs['health'], dim=0)  # Shape: (batch_size, 2)
            batched_inputs['current_crosses'] = torch.cat(batched_inputs['current_crosses'], dim=0)  # Shape: (batch_size, 52)
            batched_inputs['used_crosses'] = torch.cat(batched_inputs['used_crosses'], dim=0)  # Shape: (batch_size, 60)
            batched_inputs['beast_flags'] = torch.cat(batched_inputs['beast_flags'], dim=0)  # Shape: (batch_size, 4)

            # Convert targets and rewards to tensors
            cross_targets = torch.tensor(cross_targets, dtype=torch.long, device=device)  # Shape: (batch_size,)
            target_lists = torch.tensor(target_lists, dtype=torch.long, device=device)    # Shape: (batch_size, 6)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)  # Shape: (batch_size,)
            
            cross_targets = cross_targets.clamp(max=6)
            target_lists = target_lists.clamp(max=9)
            
            # Construct the complete batch_data dictionary
            batch_data = {
                'inputs_player_folder_chips_onehot': batched_inputs['player_folder']['chips_onehot'],
                'inputs_player_folder_codes_onehot': batched_inputs['player_folder']['codes_onehot'],
                'inputs_player_folder_flags': batched_inputs['player_folder']['flags'],
                'inputs_enemy_folder_chips_onehot': batched_inputs['enemy_folder']['chips_onehot'],
                'inputs_enemy_folder_codes_onehot': batched_inputs['enemy_folder']['codes_onehot'],
                'inputs_enemy_folder_flags': batched_inputs['enemy_folder']['flags'],
                'inputs_visible_chips_chips_onehot': batched_inputs['visible_chips']['chips_onehot'],
                'inputs_visible_chips_codes_onehot': batched_inputs['visible_chips']['codes_onehot'],
                'health': batched_inputs['health'],                # Top-level key without prefix
                'current_crosses': batched_inputs['current_crosses'],
                'used_crosses': batched_inputs['used_crosses'],
                'beast_flags': batched_inputs['beast_flags'],
                'cross_target': cross_targets,
                'target_list': target_lists,
                'reward': rewards_batch
            }


            save_batch_to_hdf5(batch_data, planning_data_dir, port, model_type)

    return  # Function does not return anything by default


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
# Other functions remain the same (handle_connection, process_position, save_data_to_hdf5, etc.)
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
    #prune all datapoints which have not been rewarded
    prune_count = 0
    if prune:
        #loop through instances ports
        for port, buffer in data_buffers.items():
            for data_point in buffer:
                if 'reward' not in data_point:
                    del data_point
                    prune_count += 1
                #or if the abs of the reward is < 0.1
                elif abs(data_point['reward']) < 0.1:
                    del data_point
                    prune_count += 1
                    
    print(f"Pruned {prune_count} data points.")
    
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
    if save_data:
        training_thread = threading.Thread(target=saving_thread_function, daemon=True)
        training_thread.start()

        # # # At the end, before exiting, signal the training thread to stop
        training_queue.put(None)  # Sentinel value to stop the thread
        training_thread.join() # Wait for the thread to finish

    # # --- Start of Tally Display ---
    # print("\n--- Input Tally ---")
    # with input_tally_lock:
    #     for key, count in input_tally.items():
    #         print(f"{key} - {count}")
    # print("-------------------\n")
    # # --- End of Tally Display ---

    # # Cancel the monitor task gracefully
    # monitor_task.cancel()
    # try:
    #     await monitor_task
    # except asyncio.CancelledError:
    #     print("Mouse monitoring task cancelled.")

    #log all the data from the planning buffers
    # for port, buffer in planning_data_buffers.items():
    #     for data_point in buffer:
    #         print(f"Port {port}: \n{data_point} ")
    # # Save the models
    # save_models()

import h5py
import numpy as np
import torch

import os
import h5py
import torch
import numpy as np
from datetime import datetime, timezone

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
    
    # Create a timestamp for unique file naming using timezone-aware UTC datetime
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    filename = f"{model_type}_port_{port}_{timestamp}.h5"
    file_path = os.path.join(training_data_dir, filename)
    
    # Function to recursively flatten nested dictionaries
    def flatten_dict(d, parent_key='', sep='_'):
        """
        Recursively flattens a nested dictionary.

        Args:
            d (dict): The dictionary to flatten.
            parent_key (str): The base key string.
            sep (str): Separator between parent and child keys.

        Returns:
            dict: A flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Flatten the batch_data to handle nested dictionaries
    flattened_data = flatten_dict(batch_data)
    
    # Convert all data to NumPy arrays with compatible dtypes
    numpy_data = {}
    for key, value in flattened_data.items():
        if isinstance(value, torch.Tensor):
            numpy_data[key] = value.detach().cpu().numpy()
        elif isinstance(value, list):
            # Handle lists of tensors or other data types as needed
            # Example: list of previous inputs or target lists
            # Ensure all elements are of the same type and size
            try:
                if all(isinstance(elem, (int, float, np.integer, np.floating)) for elem in value):
                    numpy_data[key] = np.array(value, dtype=np.int64)
                elif all(isinstance(elem, list) for elem in value):
                    # Ensure all sub-lists have the same length
                    sub_lengths = [len(sub) for sub in value]
                    if len(set(sub_lengths)) == 1:
                        numpy_data[key] = np.array(value, dtype=np.int64)
                    else:
                        raise ValueError(f"List elements for key '{key}' have varying lengths.")
                else:
                    raise TypeError(f"Unsupported list elements for key '{key}'.")
            except Exception as e:
                print(f"Error processing list for key '{key}': {e}")
                raise
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numpy_data[key] = np.array(value)
        elif isinstance(value, np.ndarray):
            numpy_data[key] = value
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
        
        # Verify that the data is not of object dtype
        if numpy_data[key].dtype == object:
            raise TypeError(f"Data for key '{key}' has dtype 'object', which is not supported by HDF5.")
    #print the keys of the data
    print(f"Keys of the data: {numpy_data.keys()}")
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
                # Compare data
                saved_data = hf[key][:]
                if not np.array_equal(saved_data, numpy_data[key]):
                    raise ValueError(f"Data mismatch in dataset '{key}'.")
        print(f"Verified the integrity of {file_path}")
    except Exception as e:
        print(f"Failed to save or verify batch to {file_path}: {e}")

import os
import json
from datetime import datetime
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
        torch.save(training_battle_model.state_dict(), battle_checkpoint_path)
        print(f"Training Battle Model saved to {battle_checkpoint_path}")
        # Manage checkpoints
        battle_checkpoint_dir = get_checkpoint_dir(model_type='battle', image_memory=IMAGE_MEMORY)
        manage_checkpoints(battle_checkpoint_dir)
    else:
        print("Training Battle Model is not loaded. Skipping save.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
