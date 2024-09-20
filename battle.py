# battle.py
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
from torch.utils.data import DataLoader
import numpy as np  # Import numpy for type conversion if needed
from train import GameInputPredictor  # Import the model class
from utils import (
    get_checkpoint_path, get_image_memory, get_exponential_sample,
    get_exponental_amount, get_threshold, get_root_dir, position_to_grid
)
import random
from collections import deque  # Import deque for frame buffering

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "valuesearch"
env_common["AI_MODEL_PATH"] = "ai_model"

command_threshold = get_threshold()

GAMMA = float(os.getenv("GAMMA", 0.00))  # Default gamma is 0.01 (1% chance of random action)

from threading import Lock

# Initialize locks for thread safety
planning_model_lock = Lock()
battle_model_lock = Lock()




# Initialize maximum health values
max_player_health = 1.0  # Start with a default value to avoid division by zero
max_enemy_health = 1.0
INSTANCES = []
battle_count = 1
# Define the server addresses and ports for each instance
# INSTANCES = [
#     {
#         'address': '127.0.0.1',
#         'port': 12344,
#         'rom_path': 'bn6,0',
#         'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
#         'name': 'Instance 1',
#         'init_link_code': 'areana1',
#         # 'replay_path':'/home/lee/Documents/Tango/replays/20240917185150-gregarbattleset1-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
#         # 'replay_path': '/home/lee/Documents/Tango/replays/20230929014832-ummm-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
#         'is_player': False  # Set to True if you don't want this instance to send inputs
#     },
#     {
#         'address': '127.0.0.1',
#         'port': 12345,
#         'rom_path': 'bn6,0',
#         'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
#         'name': 'Instance 1',
#         'init_link_code': 'areana1',
#         # 'replay_path':'/home/lee/Documents/Tango/replays/20240917185150-gregarbattleset1-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
#         # 'replay_path': '/home/lee/Documents/Tango/replays/20230929014832-ummm-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
#         'is_player': False  # Set to True if you don't want this instance to send inputs
#     },
#     # Additional instances can be added here
# ]

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
    port_base = 12344  # Starting port number

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
    7: 'S'         # 0000000000000000??
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
    'RETURN':3 #0000000000001000
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the AI model
def load_models(image_memory=1):
    """
    Loads both Planning and Battle models from their respective checkpoints.
    """
    global planning_model, battle_model

    # Load Planning Model
    planning_checkpoint_dir = os.path.join(get_root_dir(), 'checkpoints', 'planning')
    planning_checkpoint_path = get_checkpoint_path(planning_checkpoint_dir, image_memory)
    if planning_checkpoint_path:
        planning_model = GameInputPredictor(image_memory=image_memory).to(device)
        checkpoint_planning = torch.load(planning_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_planning:
            planning_model.load_state_dict(checkpoint_planning['model_state_dict'])
        else:
            raise KeyError("Planning checkpoint does not contain 'model_state_dict'")
        planning_model.eval()
        print(f"Planning Model loaded from {planning_checkpoint_path}")
    else:
        print("No Planning Model checkpoint found. Exiting.")
        exit(1)

    # Load Battle Model
    battle_checkpoint_dir = os.path.join(get_root_dir(), 'checkpoints', 'battle')
    battle_checkpoint_path = get_checkpoint_path(battle_checkpoint_dir, image_memory)
    if battle_checkpoint_path:
        battle_model = GameInputPredictor(image_memory=image_memory).to(device)
        checkpoint_battle = torch.load(battle_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_battle:
            battle_model.load_state_dict(checkpoint_battle['model_state_dict'])
        else:
            raise KeyError("Battle checkpoint does not contain 'model_state_dict'")
        battle_model.eval()
        print(f"Battle Model loaded from {battle_checkpoint_path}")
    else:
        print("No Battle Model checkpoint found. Exiting.")
        exit(1)



# Transform to preprocess images before inference
transform = transforms.Compose([
    transforms.Resize((160, 240)),
    transforms.ToTensor()
])

# Define image_memory
IMAGE_MEMORY = get_image_memory()  # Default to 1 if not set

# Load the model checkpoint
load_models(IMAGE_MEMORY)

# Initialize frame buffers and frame counters
frame_buffers = {instance['port']: deque(maxlen=get_exponental_amount()**IMAGE_MEMORY) for instance in INSTANCES}
frame_counters = {instance['port']: -1 for instance in INSTANCES}

# Function to convert integer to a 16-bit binary string
def int_to_binary_string(value):
    return format(value, '016b')

# Function to map the model output to a key based on the highest activation value
def model_output_to_key(output):
    # Define key to bit position mapping
    KEY_BIT_POSITIONS_LOCAL = {
        'A': 8,
        'DOWN': 7,
        'UP': 6,
        'LEFT': 5,
        'RIGHT': 4,
        'X': 1,
        'Z': 0,
        'S': 9,
        'RETURN':3
    }

    # Define a dynamic threshold to determine which key activations are significant
    threshold = 0.0001  # Adjust this value as needed based on model performance

    # Collect keys that have activation values above the threshold
    keys_above_threshold = []
    for key, bit_pos in KEY_BIT_POSITIONS_LOCAL.items():
        if bit_pos < len(output) and output[bit_pos] > threshold:
            keys_above_threshold.append((key, output[bit_pos]))

    # If no key is above the threshold, return 'NO_KEY'
    # if not keys_above_threshold:
    #     print(f"No valid key detected. Max value: {max(output)} below threshold: {threshold}")
    #     return 'NO_KEY'

    # Sort keys by activation value in descending order and select the one with the highest activation
    keys_above_threshold.sort(key=lambda x: x[1], reverse=True)
    selected_key, selected_activation = keys_above_threshold[0]

    # Log the selected key and its activation value
    # print(f"Selected key: {selected_key}, Activation: {selected_activation}, Threshold: {threshold}")
    return selected_key

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
    print(f"Generated random binary command: {binary_string} from keys {selected_keys}")
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



# Function to perform inference with the AI model
# Function to perform inference with the AI model
def predict(frames, player_grid, enemy_grid, inside_window, player_health, enemy_health):
    """
    Predict the next action based on a sequence of frames and additional game state information.
    Chooses between Planning and Battle models based on `inside_window`.
    """
    global planning_model, battle_model

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
        return {'type': 'key_press', 'key': '0000000000000000'}

    predicted_input_str = None
    model_type = "Battle_Model"  # Default model

    # Select model based on inside_window
    if inside_window.item() == 1.0:
        selected_model = planning_model
        selected_lock = planning_model_lock
        model_type = "Planning_Model"
    else:
        selected_model = battle_model
        selected_lock = battle_model_lock
        model_type = "Battle_Model"

    # Acquire the appropriate lock before performing model inference
    with selected_lock:
        try:
            with torch.no_grad():
                outputs = selected_model(
                    stacked_frames,
                    player_grid,       # Shape: (1, 6, 3)
                    enemy_grid,        # Shape: (1, 6, 3)
                    inside_window,     # Shape: (1, 1)
                    player_health,     # Shape: (1, 1)
                    enemy_health       # Shape: (1, 1)
                )
                probs = torch.sigmoid(outputs)
                probs = probs.cpu().numpy()[0]
                predicted_inputs = (probs >= command_threshold).astype(int)
                predicted_input_str = ''.join(map(str, predicted_inputs))
        except Exception as e:
            print(f"Error during inference with {model_type}: {e}")
            return {'type': 'key_press', 'key': '0000000000000000'}  # Return no key press on failure

    print(f"{model_type} predicted binary command: {predicted_input_str}")

    # Decide whether to take a random action based on GAMMA
    if random.random() < GAMMA:
        print(f"Gamma condition met (gamma={GAMMA}). Taking a random action.")
        random_command = generate_random_action()
        return {'type': 'key_press', 'key': random_command}
    else:
        return {'type': 'key_press', 'key': predicted_input_str}


# Function to convert model output to a 16-bit binary string
def model_output_to_binary(output):
    binary = 0
    for idx, value in enumerate(output):
        # Use the binarized output directly; value will be 0 or 1
        if value == 1:  # Check if the bit is set after thresholding in the predict function
            key = KEY_MAPPINGS.get(idx)
            if key:
                bit_pos = KEY_BIT_POSITIONS[key]
                binary |= (1 << bit_pos)
                print(f"Setting bit position {bit_pos} for key {key}")

    # Convert the integer to a 16-bit binary string
    binary_string = int_to_binary_string(binary)
    print(f"Constructed binary string: {binary_string}")
    return binary_string

# Function to save images received from the app and return the path and image
def save_image_from_base64(encoded_image, port, training_data_dir):
    try:
        decoded_image = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(decoded_image)).convert("RGB")
        filename = f"{port}_{int(time.time() * 1000)}.png"
        image_path = os.path.join(training_data_dir, filename)
        image.save(image_path)
        # print(f"Saved image from port {port} to {image_path}")
        return image_path, image
    except Exception as e:
        print(f"Failed to save image from port {port}: {e}")
        return None

# Function to save a game state as JSON with additional data
def save_game_state(
    image_path,
    input_binary,
    reward=None,
    punishment=None,
    training_data_dir=None,
    player_health=None,
    enemy_health=None,
    player_position=None,
    enemy_position=None,
    inside_window=None
):
    game_state = {
        "image_path": image_path,
        "input": input_binary,
        "reward": reward,
        "punishment": punishment,
        "player_health": player_health,
        "enemy_health": enemy_health,
        "player_position": player_position,
        "enemy_position": enemy_position,
        "inside_window": inside_window
    }
    filename = f"{int(time.time() * 1000)}.json"
    file_path = os.path.join(training_data_dir, filename)

    try:
        with open(file_path, 'w') as f:
            json.dump(game_state, f)
        # print(f"Saved game state to {file_path}")
    except Exception as e:
        print(f"Failed to save game state to {file_path}: {e}")

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

# Function to request the current screen image
async def request_screen_image(writer):
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command)
    except Exception as e:
        print(f"Failed to request screen image: {e}")
        raise

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

# Function to run the AppImage with specific ROM, SAVE paths, and PORT
def run_instance(rom_path, save_path, port, replay_path, init_link_code):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["INIT_LINK_CODE"] = init_link_code
    env["PORT"] = str(port)
    if replay_path:
        env["REPLAY_PATH"] = replay_path

    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    try:
        subprocess.Popen([APP_PATH], env=env)
    except Exception as e:
        print(f"Failed to start instance on port {port}: {e}")

# Function to start all instances
def start_instances():
    for instance in INSTANCES:
        run_instance(
            instance['rom_path'],
            instance['save_path'],
            instance['port'],
            instance.get('replay_path', None),
            instance['init_link_code']
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



# Function to receive messages from the game and process them
async def receive_messages(reader, writer, port, training_data_dir):
    buffer = ""
    current_input = None
    current_reward = None
    current_punishment = None
    # Initialize additional fields
    current_player_health = None
    current_enemy_health = None
    current_player_position = None
    current_enemy_position = None
    current_inside_window = None

    game_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
    is_replay = game_instance.get('replay_path', None) is not None

    # Define the minimum required frames for exponential sampling
    minimum_required_frames = 2**(IMAGE_MEMORY - 1) if IMAGE_MEMORY > 1 else 1

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
                    print(f"Port {port}: Failed to parse JSON message: {json_message}")
                    continue


                event = parsed_message.get("event", "Unknown")
                details = parsed_message.get("details", "No details provided")

                if event == "local_input":
                    try:
                        current_input = int_to_binary_string(int(details))
                        print(f"Port {port}: Updated current_input to {current_input}")
                    except ValueError:
                        print(f"Port {port}: Failed to parse local input: {details}")

                elif event == "screen_image":
                    # Parse the details JSON to extract additional data
                    try:
                        screen_data = json.loads(details)
                        encoded_image = screen_data.get("image", "")
                        current_player_health = float(screen_data.get("player_health", 0))
                        current_enemy_health = float(screen_data.get("enemy_health", 0))
                        current_player_position = screen_data.get("player_position", None)
                        current_enemy_position = screen_data.get("enemy_position", None)
                        current_inside_window = float(screen_data.get("inside_window", 0))

                        #log
                        # print(f"Port {port}: Received screen image with player health: {current_player_health}, enemy health: {current_enemy_health}, player position: {current_player_position}, enemy position: {current_enemy_position}, inside window: {current_inside_window}")
                        
                    except json.JSONDecodeError:
                        print(f"Port {port}: Failed to parse screen_image details: {details}")
                        continue

                    # Compute grids based on positions
                    player_grid = compute_grid(current_player_position).to(device)  # Shape: (6, 3)
                    enemy_grid = compute_grid(current_enemy_position).to(device)    # Shape: (6, 3)
                    inside_window_tensor = torch.tensor([current_inside_window], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    # Before passing health to the model
                    update_max_health(current_player_health, current_enemy_health)
                    normalized_player_health, normalized_enemy_health = normalize_health(current_player_health, current_enemy_health)

                    player_health_tensor = torch.tensor([normalized_player_health], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 1)
                    enemy_health_tensor = torch.tensor([normalized_enemy_health], dtype=torch.float32, device=device).unsqueeze(0)    # Shape: (1, 1)

                    # Save the image and retrieve it
                    save_result = save_image_from_base64(encoded_image, port, training_data_dir) if is_replay else (None, Image.open(BytesIO(base64.b64decode(encoded_image))))
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
                    # print(f"Port {port}: Received frame {current_frame_idx}")

                    # Add the new frame with its index to the buffer
                    frame_buffers[port].append((current_frame_idx, image))
                    # print(f"Port {port}: Added frame {current_frame_idx} to buffer. Buffer size: {len(frame_buffers[port])}/{frame_buffers[port].maxlen}")

                    # Retrieve all frame indices currently in the buffer
                    available_indices = [frame_idx for frame_idx, _ in frame_buffers[port]]
                    # print(f"Port {port}: Available indices in buffer: {available_indices}")

                    # Current index is the latest frame's index
                    current_idx = available_indices[-1]

                    # Check if buffer has enough frames for exponential sampling
                    if len(frame_buffers[port]) >= minimum_required_frames:
                        # Use get_exponential_sample to get the required frame indices
                        sampled_indices = get_exponential_sample(available_indices, current_idx, IMAGE_MEMORY)
                        # print(f"Port {port}: Sampled indices for inference: {sampled_indices}")

                        if not sampled_indices:
                            # print(f"Port {port}: No sampled indices returned. Skipping inference.")
                            continue

                        # Fetch the corresponding frames based on sampled indices
                        sampled_frames = []
                        index_set = set(sampled_indices)  # For faster lookup
                        for frame_idx, img in frame_buffers[port]:
                            if frame_idx in index_set:
                                sampled_frames.append(img)
                                # print(f"Port {port}: Selected frame {frame_idx} for inference.")
                                if len(sampled_frames) == IMAGE_MEMORY:
                                    break

                        # print(f"Port {port}: Number of sampled frames before padding: {len(sampled_frames)}")

                        # If not enough frames were sampled, pad with the earliest frame
                        if len(sampled_frames) < IMAGE_MEMORY:
                            if sampled_frames:
                                earliest_frame = sampled_frames[0]
                                while len(sampled_frames) < IMAGE_MEMORY:
                                    sampled_frames.insert(0, earliest_frame)
                                    # print(f"Port {port}: Padded sampled_frames with earliest frame to meet IMAGE_MEMORY.")
                            else:
                                # print(f"Port {port}: No frames available to pad. Using no action.")
                                sampled_frames = [None] * IMAGE_MEMORY  # Placeholder for no action

                        # print(f"Port {port}: Number of sampled frames after padding: {len(sampled_frames)}")

                        # Validate sampled_frames before inference
                        if any(frame is None for frame in sampled_frames):
                            print(f"Port {port}: Insufficient frames for inference. Skipping command sending.")
                            continue

                        # Only send commands if the instance is not a player
                        game_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
                        if game_instance and not game_instance.get('is_player', False):
                            command = predict(
                                sampled_frames,
                                player_grid.unsqueeze(0),       # Shape: (1, 6, 3)
                                enemy_grid.unsqueeze(0),        # Shape: (1, 6, 3)
                                inside_window_tensor,     # Shape: (1, 1)
                                player_health_tensor,     # Shape: (1, 1)
                                enemy_health_tensor       # Shape: (1, 1)
                            )
                            #don't allow sending the return command unless inside_window is true
                            # Ensure the command for the RETURN key is ignored unless inside_window == 1
                            if command['key'][15 - KEY_BIT_POSITIONS['RETURN']] == '1' and current_inside_window == 0:
                                # Replace the '1' with '0' at the correct bit position for 'RETURN'
                                command['key'] = (
                                    command['key'][:15 - KEY_BIT_POSITIONS['RETURN']] + '0' + command['key'][15 - KEY_BIT_POSITIONS['RETURN'] + 1:]
                                )
                                print(f"Port {port}: Return command attempted while not inside window. Ignoring.")
                            #else print that return was used when it is used
                            elif command['key'][15 - KEY_BIT_POSITIONS['RETURN']] == '1':
                                print(f"Port {port}: Return command used.")
                            await send_input_command(writer, command, port)
                            # print(f"Port {port}: Sent command: {command}")

                        # Save the game state only if training_data_dir is set
                        if training_data_dir and current_input is not None:
                            input_binary = current_input
                            save_game_state(
                                image_path=image_path,
                                input_binary=input_binary,
                                reward=current_reward,
                                punishment=current_punishment,
                                training_data_dir=training_data_dir,
                                player_health=current_player_health,
                                enemy_health=current_enemy_health,
                                player_position=current_player_position,
                                enemy_position=current_enemy_position,
                                inside_window=current_inside_window
                            )
                            print(f"Port {port}: Saved game state.")

                        # Reset rewards/punishments and additional fields after processing
                        current_reward = None
                        current_punishment = None
                        current_player_health = None
                        current_enemy_health = None
                        current_player_position = None
                        current_enemy_position = None
                        current_inside_window = None
                    else:
                        print(f"Port {port}: Not enough frames for exponential sampling. Required: {minimum_required_frames}, Available: {len(frame_buffers[port])}")

                elif event == "reward":
                    try:
                        current_reward = int(details.split(":")[1].strip())
                        print(f"Port {port}: Received reward: {current_reward}")
                    except (IndexError, ValueError):
                        print(f"Port {port}: Failed to parse reward message: {details}")

                elif event == "punishment":
                    try:
                        current_punishment = int(details.split(":")[1].strip())
                        print(f"Port {port}: Received punishment: {current_punishment}")
                    except (IndexError, ValueError):
                        print(f"Port {port}: Failed to parse punishment message: {details}")

                elif event == "winner":
                    player_won = details.lower() == "true"
                    print(f"Port {port}: Received winner message: Player won = {player_won}")
                    save_winner_status(training_data_dir, player_won)

                # else:
                #     print(f"Port {port}: Received unknown event: {event}, Details: {details}")

    except (ConnectionResetError, BrokenPipeError):
        print(f"Port {port}: Connection was reset by peer, closing receiver.")
    except Exception as e:
        print(f"Port {port}: Failed to receive message: {e}")

# Function to handle connection to a specific instance
async def handle_connection(instance):
    writer = None
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        training_data_dir = get_training_data_dir(instance.get('replay_path'))

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, writer, instance['port'], training_data_dir))

        # Set inference interval for higher frequency (e.g., 60 times per second)
        inference_interval = 1 / 60.0  # seconds
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

# Main function to start instances and handle inputs
async def main():
    start_instances()
    print("Instances are running.")
    await asyncio.sleep(0.5)  # Allow some time for instances to initialize

    tasks = [asyncio.create_task(handle_connection(instance)) for instance in INSTANCES]

    # Wait for all handle_connection tasks to complete
    await asyncio.gather(*tasks)
    print("All instances have completed. Exiting program.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
