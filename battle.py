#battle.py
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
from PIL import Image  # Import PIL for image loading
from train import GameInputPredictor  # Import the model class
from utils import get_checkpoint_path, get_image_memory, get_exponential_sample,get_exponental_amount, get_threshold, get_root_dir
import random
from collections import deque  # Import deque for frame buffering


# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "your_link_code"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "your_matchmaking_id"  # Replace with the actual matchmaking ID

command_threshold = get_threshold()

GAMMA = float(os.getenv("GAMMA", 0.01))  # Default gamma is 0.1 (10% chance of random action)

# Define the server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
        'name': 'Instance 1',
        'replay_path':'/home/lee/Documents/Tango/replays/lrtest.tangoreplay',
        # 'replay_path': '/home/lee/Documents/Tango/replays/20230929014832-ummm-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
        'is_player': False  # Set to True if you don't want this instance to send inputs
    },
    # {
    #     'address': '127.0.0.1',
    #     'port': 12345,
    #     'rom_path': 'bn6,0',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
    #     'name': 'Instance 1',
    #     # 'replay_path': '/home/lee/Documents/Tango/replays/20230929014832-ummm-bn6-vs-IndianaOrz-round1-p1.tangoreplay',
    #     'is_player': False  # Set to True if you don't want this instance to send inputs
    # },
    # {
    #     'address': '127.0.0.1',
    #     'port': 12346,
    #     'rom_path': 'bn6,0',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
    #     'name': 'Instance 2',
    #     # 'replay_path': '/home/lee/Documents/Tango/replays/20231013013155-annular-bn6-vs-DthKrdMnSP-round3-p2.tangoreplay',
    #     'is_player': False  # Set to True if you don't want this instance to send inputs
    # },
]

# Key mappings based on model output indices
KEY_MAPPINGS = {
    0: 'A',        # 0000000100000000
    1: 'DOWN',     # 0000000010000000
    2: 'UP',       # 0000000001000000
    3: 'LEFT',     # 0000000000100000
    4: 'RIGHT',    # 0000000000010000
    5: 'X',        # 0000000000000010
    6: 'Z',         # 0000000000000001
    7: 'S'        # 0000000000000000??
}
RANDOM_MAPPINGS = {
    0: 'A',        # 0000000100000000
    1: 'DOWN',     # 0000000010000000
    2: 'UP',       # 0000000001000000
    3: 'LEFT',     # 0000000000100000
    4: 'RIGHT',    # 0000000000010000
    5: 'X',        # 0000000000000010
    6: 'Z',         # 0000000000000001
    7: 'S'        # 0000000000000000??
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

def load_model(checkpoint_path, image_memory=1):
    global model
    model = GameInputPredictor(image_memory=image_memory).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    


    # Ensure 'model_state_dict' exists in the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")
    
    model.eval()
    print(f"Model loaded from {checkpoint_path}")




# Transform to preprocess images before inference
transform = transforms.Compose([
    transforms.Resize((160, 240)),
    transforms.ToTensor()
])

# Define image_memory
IMAGE_MEMORY = get_image_memory() # Default to 1 if not set

path = get_checkpoint_path(get_root_dir() + "/checkpoints",IMAGE_MEMORY)
print(path)

# load_model(path, IMAGE_MEMORY)

# Initialize frame buffers and frame counters
frame_buffers = {instance['port']: deque(maxlen=get_exponental_amount()**IMAGE_MEMORY) for instance in INSTANCES}
frame_counters = {instance['port']: -1 for instance in INSTANCES}

# Add this function below your existing functions
def generate_random_action():
    # Randomly select one or more keys to press
    # You can customize this to suit your game's control scheme
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

# Function to map the model output to a key based on the highest activation value
def model_output_to_key(output):
    # Define key to bit position mapping
    KEY_BIT_POSITIONS = {
        'A': 8,
        'DOWN': 7,
        'UP': 6,
        'LEFT': 5,
        'RIGHT': 4,
        'X': 1,
        'Z': 0
    }

    # Define a dynamic threshold to determine which key activations are significant
    threshold = 0.0001  # Adjust this value as needed based on model performance

    # Collect keys that have activation values above the threshold
    keys_above_threshold = []
    for key, bit_pos in KEY_BIT_POSITIONS.items():
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

# Function to perform inference with the AI model
# def predict(image):
#     # print("Performing inference...")
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.sigmoid(output).squeeze().cpu().numpy()

#     # Map model output to a key based on the highest activation
#     # print(f"Model output: {prediction}")
#     key = model_output_to_key(prediction)
#     # print(f"Predicted key: {key}")

#     # Return the key to be sent directly to the game
#     return {'type': 'key_press', 'key': key}
def predict(frames):
    """
    Predict the next action based on a sequence of frames.

    Args:
        frames (list of PIL.Image): List of PIL.Image objects, length == IMAGE_MEMORY

    Returns:
        dict: Command to be sent to the game
    """
    if len(frames) < IMAGE_MEMORY:
        print(f"Insufficient frames for prediction. Needed: {IMAGE_MEMORY}, Available: {len(frames)}")
        return {'type': 'key_press', 'key': '0000000000000000'}  # No action

    # Preprocess and stack frames
    preprocessed_frames = []
    for img in frames:
        if img is None:
            print("Encountered None in frames. Skipping inference.")
            return {'type': 'key_press', 'key': '0000000000000000'}
        img = img.convert('RGB')
        img = transform(img)  # Apply the transform (Resize and ToTensor)
        preprocessed_frames.append(img)

    # Stack frames along the temporal (depth) dimension
    # Resulting shape: (channels, depth, height, width)
    try:
        stacked_frames = torch.stack(preprocessed_frames, dim=1).unsqueeze(0).to(device)  # Shape: (1, 3, depth, H, W)
    except Exception as e:
        print(f"Error stacking frames: {e}")
        return {'type': 'key_press', 'key': '0000000000000000'}

    predicted_input_str = None
    try:
        with torch.no_grad():
            outputs = model(stacked_frames)
            probs = torch.sigmoid(outputs)
            probs = probs.cpu().numpy()[0]
            predicted_inputs = (probs >= command_threshold).astype(int)
            predicted_input_str = ''.join(map(str, predicted_inputs))
    except Exception as e:
        print(f"Error during inference: {e}")
        return {'type': 'key_press', 'key': '0000000000000000'}  # Return no key press on failure

    print(f"Model predicted binary command: {predicted_input_str}")

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
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_image)).convert("RGB")
    filename = f"{port}_{int(time.time() * 1000)}.png"
    image_path = os.path.join(training_data_dir, filename)
    image.save(image_path)
    return image_path, image

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
async def send_input_command(writer, command, port = 0):
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

    # Define the minimum required frames for exponential sampling
    minimum_required_frames = 2**(IMAGE_MEMORY - 1) if IMAGE_MEMORY > 1 else 1

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
                            command = predict(sampled_frames)
                            await send_input_command(writer, command, port)
                            # print(f"Port {port}: Sent command: {command}")

                        # Save the game state only if training_data_dir is set
                        if training_data_dir and current_input is not None:
                            input_binary = current_input
                            save_game_state(image_path, input_binary, current_reward, current_punishment, training_data_dir)
                            print(f"Port {port}: Saved game state.")

                        # Reset rewards/punishments after processing
                        current_reward = None
                        current_punishment = None
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

async def handle_connection(instance):
    writer = None
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        training_data_dir = get_training_data_dir(instance.get('replay_path'))

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, writer, instance['port'], training_data_dir))

        # Set inference interval for 10 times per second
        inference_interval = 1 / 60.0 # seconds
        #when the instance is doing a replay, divide by 4 because the replay is 4 times faster
        if instance.get('replay_path'):
            inference_interval = inference_interval / 4.0

        while not reader.at_eof():
            try:
                # print(f"Requesting screen image for {instance['name']} on port {instance['port']}")
                await request_screen_image(writer)
                await asyncio.sleep(inference_interval)  # Run inferences at 10 times per second

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