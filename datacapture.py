import subprocess
import os
import time
import asyncio
import json
import random
import base64
from PIL import Image
from io import BytesIO
import sys
import glob  # For file pattern matching
import argparse  # For parsing command-line arguments

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Paths
REPLAYS_DIR = '/home/lee/Documents/Tango/replays'
# TRAINING_DATA_DIR = 'training_data'
TRAINING_DATA_DIR = '/media/lee/A416C57D16C5514A/Users/Lee/FFCO/ai/TANGO'
# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "your_link_code"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "your_matchmaking_id"  # Replace with the actual matchmaking ID

# Function to get the training directory based on the replay file name
def get_training_data_dir(replay_path):
    replay_name = os.path.basename(replay_path).split('.')[0]  # Extract the file name without extension
    training_data_dir = os.path.join(TRAINING_DATA_DIR, replay_name)
    os.makedirs(training_data_dir, exist_ok=True)
    return training_data_dir

# Function to get a list of unprocessed replays with an optional maximum limit
def get_unprocessed_replays(max_replays=None):
    """
    Retrieves a list of unprocessed replay files. Optionally limits the number
    of replays returned based on the max_replays parameter.
    
    A replay is considered unprocessed if its corresponding training data directory
    does not exist or if the winner status in winner.json is undecided.

    Args:
        max_replays (int, optional): Maximum number of replays to return. Defaults to None.
        
    Returns:
        list: List of unprocessed replay file paths.
    """
    # Get all replay files with 'bn6' in their name
    replay_files = glob.glob(os.path.join(REPLAYS_DIR, '*bn6*.tangoreplay'))
    
    unprocessed_replays = []
    for replay_file in replay_files:
        replay_name = os.path.basename(replay_file).split('.')[0]
        training_data_dir = os.path.join(TRAINING_DATA_DIR, replay_name)
        
        # Check if the training data directory exists
        if not os.path.exists(training_data_dir):
            # Directory doesn't exist, so it's unprocessed
            unprocessed_replays.append(replay_file)
        else:
            # Check if winner.json exists and if the winner status is undecided
            winner_file = os.path.join(training_data_dir, 'winner.json')
            if os.path.exists(winner_file):
                try:
                    with open(winner_file, 'r') as f:
                        winner_data = json.load(f)
                        is_winner = winner_data.get('is_winner', None)
                        # Check if the winner status is undecided
                        if is_winner is None:
                            unprocessed_replays.append(replay_file)
                except json.JSONDecodeError:
                    # If winner.json is corrupted, consider the replay as unprocessed
                    unprocessed_replays.append(replay_file)
                    print(f"Invalid JSON format in {winner_file}, considering {replay_name} unprocessed.")
            else:
                # If winner.json does not exist, consider it processed (or handle differently if needed)
                pass
        
        # Stop adding more replays if max_replays is reached
        if max_replays is not None and len(unprocessed_replays) >= max_replays:
            break

    return unprocessed_replays


# Function to run the AppImage with specific ROM, SAVE paths, and PORT
def run_instance(rom_path, save_path, port, replay_path):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["PORT"] = str(port)
    env["REPLAY_PATH"] = replay_path

    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    subprocess.Popen([APP_PATH], env=env)

# Function to start instances in batches
def start_instances_in_batches(instances, batch_size=10):
    """
    Starts instances of the Tango AppImage in specified batch sizes.
    
    Args:
        instances (list): List of instance configurations to run.
        batch_size (int, optional): Number of instances to start per batch. Defaults to 10.
    """
    total_instances = len(instances)
    for i in range(0, total_instances, batch_size):
        batch = instances[i:i+batch_size]
        # Start instances in the current batch
        for instance in batch:
            run_instance(instance['rom_path'], instance['save_path'], instance['port'], instance['replay_path'])
            time.sleep(0.5)  # Adjust sleep time based on app's boot time
        # Wait for the batch to complete
        print(f"Processing batch {i // batch_size + 1} of {((total_instances - 1) // batch_size) + 1}")
        asyncio.run(process_batch(batch))
        # Optionally, you can add a delay or cleanup here if needed

# Function to process a batch of instances
async def process_batch(instances):
    tasks = [asyncio.create_task(handle_connection(instance)) for instance in instances]
    # Wait for all handle_connection tasks to complete
    await asyncio.gather(*tasks)
    print("Batch processing completed.")

# Placeholder predict method for AI inference (currently random actions)
def predict(image):
    possible_keys = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'A']
    action = random.choice(['key_press', 'key_release'])
    key = random.choice(possible_keys)
    return {'type': action, 'key': key}

# Function to save images received from the app and return the path
def save_image_from_base64(encoded_image, port, training_data_dir):
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_image))
    filename = f"{port}_{int(time.time() * 1000)}.png"
    image_path = os.path.join(training_data_dir, filename)
    image.save(image_path)
    return image_path

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

# Function to send input command to a specific instance
async def send_input_command(writer, command):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
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

def int_to_binary_string(value):
    return format(value, '016b')

# Function to receive messages from the game and process them
async def receive_messages(reader, port, training_data_dir):
    buffer = ""
    current_input = None
    current_reward = None
    current_punishment = None

    try:
        while True:
            data = await reader.read(4096)
            if not data:
                print(f"Connection closed by peer on port {port}.")
                break
            buffer += data.decode()

            while True:
                try:
                    # Look for JSON object termination
                    json_end_index = buffer.find('}\n')
                    if json_end_index == -1:
                        json_end_index = buffer.find('}')
                    if json_end_index == -1:
                        break

                    json_message = buffer[:json_end_index + 1].strip()
                    buffer = buffer[json_end_index + 1:]

                    parsed_message = json.loads(json_message)
                    event = parsed_message.get("event", "Unknown")
                    details = parsed_message.get("details", "No details provided")

                    if event == "local_input":
                        current_input = int_to_binary_string(int(details))

                    elif event == "screen_image":
                        image_path = save_image_from_base64(details, port, training_data_dir)
                        
                        # Save game state when an image is received
                        save_game_state(image_path, current_input, current_reward, current_punishment, training_data_dir)
                        
                        # Reset rewards/punishments after saving
                        current_reward = None
                        current_punishment = None

                    elif event == "reward":
                        try:
                            current_reward = int(details.split(":")[1].strip())
                        except ValueError:
                            print(f"Failed to parse reward message: {details}")

                    elif event == "punishment":
                        try:
                            current_punishment = int(details.split(":")[1].strip())
                        except ValueError:
                            print(f"Failed to parse punishment message: {details}")
                    elif event == "winner":
                        player_won = details.lower() == "true"
                        save_winner_status(training_data_dir, player_won)

                    else:
                        print(f"Received message: Event - {event}, Details - {details}")

                except json.JSONDecodeError:
                    break

    except (ConnectionResetError, BrokenPipeError):
        print(f"Connection was reset by peer on port {port}, closing receiver.")
    except Exception as e:
        print(f"Failed to receive message on port {port}: {e}")

# Function to save the winner status in the replay folder
def save_winner_status(training_data_dir, player_won):
    winner_status = {
        "is_winner": player_won
    }
    file_path = os.path.join(training_data_dir, "winner.json")
    with open(file_path, 'w') as f:
        json.dump(winner_status, f)

# Function to handle connection to a specific instance and predict actions based on screen capture
async def handle_connection(instance):
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        # Get the training data directory based on the replay path
        training_data_dir = get_training_data_dir(instance['replay_path'])

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, instance['port'], training_data_dir))

        # Set a delay to request images
        image_request_interval = 1 / 60.0 / 4.0  # seconds

        while not reader.at_eof():
            try:
                await request_screen_image(writer)
                await asyncio.sleep(image_request_interval)  # Control the request rate

                # Predict and send inputs if not a player instance
                if not instance.get('is_player', False):
                    command = predict(None)
                    await send_input_command(writer, command)

            except (ConnectionResetError, BrokenPipeError):
                print(f"Connection to {instance['name']} was reset. Stopping send loop.")
                break  # Exit the loop
            except Exception as e:
                print(f"An error occurred in connection to {instance['name']}: {e}")
                break  # Exit the loop on other exceptions

        # Wait for the receive_messages task to finish
        await receive_task

    except ConnectionRefusedError:
        print(f"Failed to connect to {instance['name']}. Is the application running?")
    except Exception as e:
        print(f"An error occurred with {instance['name']}: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"Error closing connection to {instance['name']}: {e}")
        print(f"Connection to {instance['name']} closed.")

# Main function to start instances and handle inputs
def main():
    parser = argparse.ArgumentParser(description="Start Tango AppImage instances with a maximum number of replays.")
    parser.add_argument('--max_replays', type=int, default=10,
                        help='Maximum number of replays to process. If not set, all unprocessed replays will be processed.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of instances to start per batch. Default is 10.')
    parser.add_argument('--start_port', type=int, default=12345,
                        help='Starting port number for instances. Default is 12345.')

    args = parser.parse_args()

    max_replays = args.max_replays
    batch_size = args.batch_size
    starting_port = args.start_port

    # Get the list of unprocessed replays
    unprocessed_replays = get_unprocessed_replays(max_replays)

    if not unprocessed_replays:
        print("No unprocessed replays found.")
        return

    print(f"Found {len(unprocessed_replays)} unprocessed replays.")

    # Prepare instances
    instances = []
    port = starting_port  # Starting port number; adjust if needed

    for replay_path in unprocessed_replays:
        replay_name = os.path.basename(replay_path).split('.')[0]

        # Check if the training data directory exists
        training_data_dir = os.path.join(TRAINING_DATA_DIR, replay_name)
        if os.path.exists(training_data_dir):
            print(f"Training data for {replay_name} already exists. Skipping.")
            continue

        # Create the instance configuration
        instance = {
            'address': '127.0.0.1',
            'port': port,
            'rom_path': 'bn6,0',
            'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
            'name': f'Instance {port}',
            'replay_path': replay_path,
            'is_player': True  # Set to True if you don't want this instance to send inputs
        }
        instances.append(instance)
        port += 1  # Increment port for the next instance

    if not instances:
        print("No new instances to process.")
        return

    print(f"Starting {len(instances)} instances.")

    # Start instances in batches
    start_instances_in_batches(instances, batch_size=batch_size)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
