# battle_input_test.py
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
import numpy as np
from train import GameInputPredictor  # This can be removed if not used
from utils import get_checkpoint_path, get_image_memory, get_exponential_sample, get_exponental_amount, get_threshold, get_root_dir
import random
from collections import deque
import sys

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "inputtest"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "inputtest"  # Replace with the actual matchmaking ID

command_threshold = get_threshold()

GAMMA = float(os.getenv("GAMMA", 0.05))  # Default gamma is 0.05 (5% chance of random action)

# Define the server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
        'name': 'Instance 1',
        'is_player': False  # Set to True if you don't want this instance to send inputs
    },
    # Add more instances if needed
]

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
    'S': 9
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The model loading section can be removed or commented out if not using the model
# def load_model(checkpoint_path, image_memory=1):
#     global model
#     model = GameInputPredictor(image_memory=image_memory).to(device)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         raise KeyError("Checkpoint does not contain 'model_state_dict'")
#     model.eval()
#     print(f"Model loaded from {checkpoint_path}")

# Remove or comment out model loading if not using the model
# transform = transforms.Compose([
#     transforms.Resize((160, 240)),
#     transforms.ToTensor()
# ])

# Define image_memory
# IMAGE_MEMORY = get_image_memory() # Default to 1 if not set

# path = get_checkpoint_path(get_root_dir() + "/checkpoints", IMAGE_MEMORY)
# print(path)

# load_model(path, IMAGE_MEMORY)

# Initialize frame buffers and frame counters
frame_buffers = {instance['port']: deque(maxlen=get_exponental_amount()**1) for instance in INSTANCES}  # IMAGE_MEMORY set to 1
frame_counters = {instance['port']: -1 for instance in INSTANCES}

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
def run_instance(rom_path, save_path, port, replay_path):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["PORT"] = str(port)
    if replay_path:
        env["REPLAY_PATH"] = replay_path

    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    subprocess.Popen([APP_PATH], env=env)

# Function to start all instances
def start_instances():
    for instance in INSTANCES:
        run_instance(
            instance['rom_path'],
            instance['save_path'],
            instance['port'],
            instance.get('replay_path', None)
        )
        time.sleep(0.5)  # Adjust sleep time based on app's boot time

# Function to convert integer to a 16-bit binary string
def int_to_binary_string(value):
    return format(value, '016b')

# Function to map user input to a binary command
def user_input_to_binary(command):
    # Split the command by spaces to allow multiple keys (e.g., "up left")
    keys = command.upper().split()
    binary_command = 0
    for key in keys:
        if key in KEY_BIT_POSITIONS:
            bit_pos = KEY_BIT_POSITIONS[key]
            binary_command |= (1 << bit_pos)
        else:
            print(f"Unknown key: {key}")
    binary_string = int_to_binary_string(binary_command)
    print(f"User input '{command}' mapped to binary command: {binary_string}")
    return binary_string

# Function to save images received from the app and return the path and image
def save_image_from_base64(encoded_image, port, training_data_dir):
    try:
        decoded_image = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(decoded_image)).convert("RGB")
        filename = f"{port}_{int(time.time() * 1000)}.png"
        image_path = os.path.join(training_data_dir, filename)
        image.save(image_path)
        return image_path, image
    except Exception as e:
        print(f"Failed to save image from base64: {e}")
        return None, None

# Function to save a game state as JSON
def save_game_state(image_path, input_binary, reward=None, punishment=None, training_data_dir=None):
    game_state = {
        "image_path": image_path,
        "input": input_binary,
        "reward": reward,
        "punishment": punishment
    }
    filename = f"{int(time.time() * 1000)}.json"
    file_path = os.path.join(training_data_dir, filename)

    with open(file_path, 'w') as f:
        json.dump(game_state, f)
    # print(f"Saved game state to {file_path}")

# Function to save the winner status in the replay folder
def save_winner_status(training_data_dir, player_won):
    winner_status = {
        "is_winner": player_won
    }
    file_path = os.path.join(training_data_dir, "winner.json")
    with open(file_path, 'w') as f:
        json.dump(winner_status, f)
    print(f"Saved winner status to {file_path}")

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
        print(f"Failed to send command: {e}")
        raise

# Function to request the current screen image
async def request_screen_image(writer):
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command)
    except Exception as e:
        print(f"Failed to request screen image: {e}")
        raise

async def receive_messages(reader, writer, port, training_data_dir):
    buffer = ""
    current_input = None
    current_reward = None
    current_punishment = None

    try:
        while True:
            data = await reader.read(4096)
            if not data:
                print(f"Port {port}: Connection closed by peer.")
                break
            buffer += data.decode()

            while True:
                json_end_index = buffer.find('}\n')
                if json_end_index == -1:
                    json_end_index = buffer.find('}')
                if json_end_index == -1:
                    break

                json_message = buffer[:json_end_index + 1].strip()
                buffer = buffer[json_end_index + 1:]

                try:
                    parsed_message = json.loads(json_message)
                except json.JSONDecodeError:
                    # print(f"Port {port}: Failed to parse JSON message: {json_message}")
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
                    # Save the image and retrieve it regardless of training_data_dir status
                    save_result = save_image_from_base64(details, port, training_data_dir)
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

                    # Since we're manually sending inputs, we don't need to perform inference
                    # However, you can still save the game state if needed
                    # Only send commands if the instance is not a player
                    game_instance = next((inst for inst in INSTANCES if inst['port'] == port), None)
                    if game_instance and not game_instance.get('is_player', False):
                        if current_input:
                            command = {'type': 'key_press', 'key': current_input}
                            await send_input_command(writer, command, port)
                            print(f"Port {port}: Sent command: {command}")
                        # Save the game state only if training_data_dir is set
                        if training_data_dir and current_input is not None:
                            input_binary = current_input
                            save_game_state(image_path, input_binary, current_reward, current_punishment, training_data_dir)
                            print(f"Port {port}: Saved game state.")

                    # Reset rewards/punishments after processing
                    current_reward = None
                    current_punishment = None

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

async def handle_connection(instance):
    writer = None
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        training_data_dir = get_training_data_dir(instance.get('replay_path'))

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, writer, instance['port'], training_data_dir))

        # Set inference interval for requesting screen images
        inference_interval = 1 / 60.0  # seconds
        # When the instance is doing a replay, adjust the interval
        if instance.get('replay_path'):
            inference_interval = inference_interval / 4.0

        while not reader.at_eof():
            try:
                await request_screen_image(writer)
                await asyncio.sleep(inference_interval)  # Run requests at the specified interval

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

# Function to handle user input asynchronously
async def handle_user_input(connections):
    print("Enter commands in the format: <port> <command>")
    print("Available commands: up, down, left, right, a, s, z, x")
    print("Type 'exit' to quit.")
    loop = asyncio.get_event_loop()
    while True:
        # Read user input in a non-blocking way
        user_input = await loop.run_in_executor(None, sys.stdin.readline)
        user_input = user_input.strip()
        if user_input.lower() == 'exit':
            print("Exiting input handler.")
            for writer in connections.values():
                if writer:
                    writer.close()
            break
        if not user_input:
            continue
        try:
            port_str, *command_parts = user_input.split()
            port = int(port_str)
            command = ' '.join(command_parts)
            if port not in connections:
                print(f"No connection found for port {port}.")
                continue
            writer = connections[port]
            if not writer:
                print(f"No writer available for port {port}.")
                continue
            binary_command = user_input_to_binary(command)
            command_dict = {'type': 'key_press', 'key': binary_command}
            await send_input_command(writer, command_dict, port)
            print(f"Sent command to port {port}: {command_dict}")
        except ValueError:
            print("Invalid input format. Use: <port> <command>")
        except Exception as e:
            print(f"Error processing input: {e}")

# Main function to start instances and handle inputs
async def main():
    start_instances()
    print("Instances are running.")
    await asyncio.sleep(0.5)  # Allow some time for instances to initialize

    # Establish connections to all instances
    connection_tasks = [handle_connection(instance) for instance in INSTANCES]
    connections = {instance['port']: None for instance in INSTANCES}  # To store writer objects

    async def connect_instance(instance):
        nonlocal connections
        try:
            reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
            connections[instance['port']] = writer
            print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")
        except Exception as e:
            print(f"Failed to connect to {instance['name']} on port {instance['port']}: {e}")

    # Start connection tasks
    connect_tasks = [connect_instance(instance) for instance in INSTANCES]
    await asyncio.gather(*connect_tasks)

    # Start handling user input
    input_task = asyncio.create_task(handle_user_input(connections))

    # Wait for all connection tasks to complete
    await asyncio.gather(*connection_tasks, input_task)

    print("All instances have completed. Exiting program.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
