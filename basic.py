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
CHIP_WINDOW_TIMER = float(os.getenv("CHIP_WINDOW_TIMER", 0.0))  # Default is 5 seconds
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
battle_count = 0
include_orig =True
do_replays = False
save_data = False
prune = False

#grid experience for ports
grid_experiences = defaultdict(list)

shoot_experiences = defaultdict(list)


# replay_count = 0
# battle_count = 1
# include_orig = False
# do_replays = False
# save_data = False
# prune = False

# replay_count = 8
# battle_count = 0
# include_orig = False
# do_replays = True
# save_data = True
# prune = False

INSTANCES = []
# Define the server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0',
        # 'rom_path': 'bn6,1',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
        # 'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
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
        # 'rom_path': 'bn6,1',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav',
        # 'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar.sav',
        'name': 'Instance 2',
        'init_link_code': 'arena1',
        'is_player': False  # Set to False if you want this instance to send inputs
    },
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


from planning_model import PlanningModel, get_planning_input_from_replay


# Function to load the AI models
# Initialize Separate Models for Inference and Training

gridstate_model = None
from dodgemodel import GridStateEvaluator

shoot_model = None
from dodgemodel import GridStateEvaluator

def load_models(image_memory=1, learning_rate=1e-3):
    global gridstate_model
    global latest_checkpoint_number  # Access the global variable
    
    # Define the root directory
    root_dir = get_root_dir()
    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model
    gridstate_model = GridStateEvaluator().to(device)

    # Path to the model checkpoint
    checkpoint_path = 'grideval.pth'

    # Load the model weights if the checkpoint exists
    if os.path.exists(checkpoint_path):
        gridstate_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        gridstate_model.eval()
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Initializing a new model.")
        
    #load shootmodel
    global shoot_model
    shoot_model = GridStateEvaluator().to(device)
    checkpoint_path = 'shooteval.pth'
    if os.path.exists(checkpoint_path):
        shoot_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        shoot_model.eval()
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Initializing a new model.")
        

    


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


from planning_model import get_planning_input_from_instance, encode_used_crosses, encode_beast_flags,encode_current_cross,encode_folder,encode_visible_chips
# Function to perform inference with the AI model
# previous_sent = 0

def predict(port, current_data_point, inside_window, tensor_params):
    global window_entry_time, previous_sent_dict, previous_inside_window_dict, INSTANCES, gridstate_model, shoot_model
    current_time = time.time()

    #skip chip window
    if inside_window.item() == 1.0:
        print(f"Port {port}: Inside chip window.")
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
                    

    else:
        # Get confidence score using the model's method
        confidence = gridstate_model.get_confidence_score(current_data_point)
        
        shoot_confidence = shoot_model.get_confidence_score(current_data_point)
        
        shoot_bit = '0' if not tensor_params['player_shoot_button'] else '1'
        
        input_string='0000000000000000'
        press_confidence = 0
        release_confidence = 0
        
        release_params = tensor_params.copy()
        release_params['player_shoot_button'] = False
        release_confidence = shoot_model.get_confidence_score(get_gamestate_tensor(release_params))
        
        press_params = tensor_params.copy()
        press_params['player_shoot_button'] = True
        press_confidence = shoot_model.get_confidence_score(get_gamestate_tensor(press_params))
        
        if press_confidence > shoot_confidence:
            shoot_bit = '1'
            # print('shoot')
        elif release_confidence > shoot_confidence:
            shoot_bit = '0'
            # print('release')
        
        print(press_confidence - release_confidence)
        
        #set 
        input_string = input_string[:15 -KEY_BIT_POSITIONS['X']] + shoot_bit + input_string[15 - KEY_BIT_POSITIONS['X']+1:]
        # print(input_string)

        if confidence is None:
            print(f"Port {port}: Unable to compute confidence score.")
            return {'type': 'key_press', 'key': '0000000000000000'}

        # print(f"Port {port}: Confidence score for current position: {confidence}")
        
        #generate new grid states based on the tensor params for the player moving up, down, left, right, and staying still
        
        possible_states = []
        
        # print(tensor_params['player_grid_position'])
        
        left_confidence = get_offset_confidence([-1,0], tensor_params)
        right_confidence = get_offset_confidence([1,0], tensor_params)
        up_confidence = get_offset_confidence([0,-1], tensor_params)
        down_confidence = get_offset_confidence([0,1], tensor_params)
        
        possible_states.append([[0,0], confidence])
        possible_states.append([[-1,0], left_confidence])
        possible_states.append([[1,0], right_confidence])
        possible_states.append([[0,-1], up_confidence])
        possible_states.append([[0,1], down_confidence])
        possible_states = sorted(possible_states, key=lambda x: x[1], reverse=True)
        
        
        #sort the possible states by confidence
        # print(possible_states)
        
        #print confidence of stay, up down left right to 2 decimals
        # print(f"Position: [{tensor_params['player_grid_position']}] Conidences: Stay[{confidence:.6f}] Left[{left_confidence:.6f}] Right[{right_confidence:.6f}] Up[{up_confidence:.6f}] Down[{down_confidence:.6f}] Shoot[{press_confidence:.6f}] Release[{release_confidence:.6f}]")
        
        #perform the most confident action
        offset = possible_states[0][0]
        # if offset[0] == 0 and offset[1] == 0:
            #stay
            # print('stay')
        if offset[0] == -1 and offset[1] == 0:
            #move left
            # print('left')
            input_string = input_string[:15 -KEY_BIT_POSITIONS['LEFT']] + '1' + input_string[15 - KEY_BIT_POSITIONS['LEFT']+1:]
        elif offset[0] == 1 and offset[1] == 0:
            #move right
            # print('right')
            input_string = input_string[:15 -KEY_BIT_POSITIONS['RIGHT']] + '1' + input_string[15 - KEY_BIT_POSITIONS['RIGHT']+1:]
        elif offset[0] == 0 and offset[1] == -1:
            #move up
            # print('up')
            input_string = input_string[:15 -KEY_BIT_POSITIONS['UP']] + '1' + input_string[15 - KEY_BIT_POSITIONS['UP']+1:]
        elif offset[0] == 0 and offset[1] == 1:
            #move down
            # print('down')
            input_string = input_string[:15 -KEY_BIT_POSITIONS['DOWN']] + '1' + input_string[15 - KEY_BIT_POSITIONS['DOWN']+1:]

        return {'type': 'key_press', 'key': input_string}


    return {'type': 'key_press', 'key': '0000000000000000'}

def get_offset_confidence(offset, tensor_params):
    global gridstate_model
    offset_position = [tensor_params['player_grid_position'][0] + offset[0], tensor_params['player_grid_position'][1] + offset[1]]
    #check if offset position x is between 1-6 and y is between 1-3
    if offset_position[0] < 1 or offset_position[0] > 6 or offset_position[1] < 1 or offset_position[1] > 3:
        return 0
    offset_tensor = tensor_params.copy()
    offset_tensor['player_grid_position'] = offset_position
    gamestate = get_gamestate_tensor(offset_tensor)
    #get left press confidence
    confidence = gridstate_model.get_confidence_score(gamestate)
    return confidence

max_reward = 1
max_punishment = 1
# Function to receive messages from the game and process them
async def receive_messages(reader, writer, port, training_data_dir, config):
    global max_reward, max_punishment, prune, gridstate_model, grid_experiences, shoot_experiences
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


                    tensor_params = {
                        "screen_data": screen_data,
                        "cust_gage": cust_gage,  # validated 2
                        "grid_state": grid_state,  # validated 2
                        "grid_owner_state": grid_owner_state,  # validated 2
                        "player_grid_position": player_grid_position,  # validated 2
                        "enemy_grid_position": enemy_grid_position,  # validated 2
                        "player_health": normalized_player_health,  # validated 2
                        "enemy_health": normalized_enemy_health,  # validated 2
                        "player_chip": current_player_chip,  # validated 2
                        "enemy_chip": current_enemy_chip,  # validated 2
                        "player_charge": current_player_charge,  # validated 2
                        "enemy_charge": current_enemy_charge,  # validated 2
                        "player_shoot_button": current_input[14] == '1',
                        "player_chip_button": current_input[15] == '1',
                        "player_chip_hand": game_instance['current_hand'] if 'current_hand' in game_instance else None,  # validated 2
                        "player_folder": game_instance['player_folder'],  # validated 2
                        "enemy_folder": game_instance['enemy_folder'],  # validated 2
                        "player_custom": own_navi_cust,  # validated 2
                        "enemy_custom": enemy_navi_cust,  # validated 2
                        "player_emotion_state": current_player_game_emotion,  # validated 2
                        "enemy_emotion_state": current_enemy_game_emotion,  # validated 2
                        "player_used_crosses": game_instance['player_used_crosses'],  # validated 2
                        "enemy_used_crosses": game_instance['enemy_used_crosses'],  # validated 2
                        "player_beasted_out": game_instance['player_beasted_out'],  # validated
                        "enemy_beasted_out": game_instance['enemy_beasted_out'],  # validated
                        "player_beasted_over": game_instance['player_beasted_over'],
                        "enemy_beasted_over": game_instance['enemy_beasted_over'],
                    }

                    gamestate_tensor = get_gamestate_tensor(
                        tensor_params
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
                        # print(f"Port {port}: No screen image available.")
                        continue
                    image_path, image = save_result

                    if image is None:
                        print(f"Port {port}: Failed to decode image.")
                        continue

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
                                tensor_params
                            )
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
                                # Compute reward
                                reward_value = (current_reward / max_reward) - (current_punishment / max_punishment)
                                grid_experiences[port].append([data_point, reward_value])
                                
                                # Apply discounted rewards to all experiences
                                gamma = 0.99
                                grid_experiences[port] = apply_discounted_rewards(grid_experiences[port], gamma)
                                shoot_experiences[port] = apply_discounted_rewards(shoot_experiences[port], gamma)
                                
                                # Train on the last batch of window_size
                                if len(grid_experiences[port]) > window_size:
                                    batch = grid_experiences[port][-window_size:]
                                else:
                                    batch = grid_experiences[port]
                                
                                average_loss, num_trained = gridstate_model.train_batch(batch)
                                
                                if reward_value > 0:
                                    shoot_experiences[port].append([data_point, reward_value])
                                    #train on shoot experiences
                                    if len(shoot_experiences[port]) > window_size:
                                        batch = shoot_experiences[port][-window_size:]
                                    else:
                                        batch = shoot_experiences[port]
                                    
                                    average_loss, num_trained = shoot_model.train_batch(batch)


                                # Reset rewards
                                current_reward = 0
                                current_punishment = 0
                            else:
                                grid_experiences[port].append([data_point, 0.001])
                                #train on the current experience
                                gridstate_model.train_batch(grid_experiences[port][-1:])
                                #train on random batch of 100
                                # if len(grid_experiences[port]) > 100:
                                #     batch = grid_experiences[port][-100:]
                                #     average_loss, num_trained = gridstate_model.train_batch(batch)
                                # else:
                                #     average_loss, num_trained = gridstate_model.train_batch(grid_experiences[port])
                                # print(f"Port {port}: Average Loss: {average_loss}, Num Trained: {num_trained}")

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
                            #reward shoot model if charging
                            # if current_player_charge >= 0.5:
                            #     shoot_experiences[port].append([data_point, 0.1])
                            #     shoot_model.train_batch(shoot_experiences[port][-1:])
                                
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
def apply_discounted_rewards(experiences, gamma=0.99):
    """
    Applies discounted rewards to a list of experiences.

    Args:
        experiences (list): List of [data_point, reward] pairs.
        gamma (float): Discount factor.

    Returns:
        list: List of [data_point, discounted_return] pairs.
    """
    discounted_returns = []
    G = 0
    for data_point, reward in reversed(experiences):
        G = reward + gamma * G
        discounted_returns.insert(0, [data_point, G])
    return discounted_returns


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
    
    # WIN_REWARD = 0.5
    # REPLAY_REWARD = 0.5
    # # Determine winners and losers based on final_health
    # print("Processing final health to determine winners and losers.")
    # for port, (player_health, enemy_health) in final_health.items():
    #     if player_health > enemy_health:
    #         # Winner: Add +1 to all rewards
    #         with data_buffers_locks[port]:
    #             for data_point in data_buffers[port]:
    #                 if 'reward' in data_point:
    #                     data_point['reward'] += WIN_REWARD
    #                 else:
    #                     data_point['reward'] = WIN_REWARD
    #             for data_point in planning_data_buffers[port]:
    #                 if 'reward' in data_point:
    #                     data_point['reward'] += WIN_REWARD
    #                 else:
    #                     data_point['reward'] = WIN_REWARD
    #         print(f"Port {port}: Winner. Added +{WIN_REWARD} reward to all data points.")
    #     else:
    #         # Only punish non-replay instances
    #         # Get instance
    #         instance = [instance for instance in INSTANCES if instance['port'] == port][0]
    #         if not instance.get('replay_path'):
    #             # Loser: Add -1 to all rewards
    #             with data_buffers_locks[port]:
    #                 for data_point in data_buffers[port]:
    #                     # Make sure the reward exists
    #                     if 'reward' in data_point:
    #                         data_point['reward'] -= WIN_REWARD
    #                     else:
    #                         data_point['reward'] = -WIN_REWARD
    #                 for data_point in planning_data_buffers[port]:
    #                     if 'reward' in data_point:
    #                         data_point['reward'] -= WIN_REWARD
    #                     else:
    #                         data_point['reward'] = -WIN_REWARD
    #             print(f"Port {port}: Loser. Added -{WIN_REWARD} reward to all data points.")
    
    # # Iterate over all instances
    # for instance in INSTANCES:
    #     port = instance['port']
    #     if instance.get('replay_path'):
    #         with data_buffers_locks[port]:
    #             for data_point in data_buffers[port]:
    #                 if 'reward' in data_point:
    #                     data_point['reward'] += REPLAY_REWARD
    #                 else:
    #                     data_point['reward'] = REPLAY_REWARD
    #             for data_point in planning_data_buffers[port]:
    #                 if 'reward' in data_point:
    #                     data_point['reward'] += REPLAY_REWARD
    #                 else:
    #                     data_point['reward'] = REPLAY_REWARD
    #         print(f"Port {port}: Replay. Added +{REPLAY_REWARD} reward to all data points.")

    
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

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
