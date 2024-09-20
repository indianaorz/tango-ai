# data_capture.py
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
from utils import get_root_dir
from tqdm import tqdm  # Import tqdm for progress bar
import logging  # Import logging module

# Configure logging
logging.basicConfig(
    filename='data_capture.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Paths
REPLAYS_DIR = '/home/lee/Documents/Tango/replays'
# TRAINING_DATA_DIR = 'training_data'
TRAINING_DATA_DIR = get_root_dir() + '/training_data'
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
    Retrieves a list of unprocessed replay files. Optionally limits the number of replays returned based on the max_replays parameter.
    
    A replay is considered unprocessed if its corresponding training data directory does not exist or if the winner status in winner.json is undecided.

    Args:
        max_replays (int, optional): Maximum number of replays to return. Defaults to None.
        
    Returns:
        list: List of unprocessed replay file paths.
    """
    # Get all replay files with 'bn6' in their name
    replay_files = glob.glob(os.path.join(REPLAYS_DIR, '*bn6*.tangoreplay'))
    print(f"Found {len(replay_files)} replay files.")
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
                    logging.warning(f"Invalid JSON format in {winner_file}, considering {replay_name} unprocessed.")
            else:
                # If winner.json does not exist, consider it processed (or handle differently if needed)
                pass
        
        # Stop adding more replays if max_replays is reached
        if max_replays is not None and len(unprocessed_replays) >= max_replays:
            break
    print(f"Found {len(unprocessed_replays)} unprocessed replays.")
    return unprocessed_replays


# Function to run the AppImage with specific ROM, SAVE paths, and PORT
def run_instance(rom_path, save_path, port, replay_path):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["PORT"] = str(port)
    env["REPLAY_PATH"] = replay_path

    logging.info(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    try:
        subprocess.Popen([APP_PATH], env=env)
    except Exception as e:
        logging.error(f"Failed to start instance on port {port}: {e}")
        print(f"Failed to start instance on port {port}: {e}")

# Function to check if a port is open
async def is_port_open(host, port, timeout=5):
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.close()
        await writer.wait_closed()
        return True
    except:
        return False
        
# Function to track net rewards within a 10-second window and apply punishment if needed
async def track_rewards_and_apply_punishment(instance, training_data_dir, window_duration=10):
    """
    Tracks the rewards over a specified time window and applies punishment if no net rewards are gained.

    Args:
        instance (dict): Instance data containing instance name and port.
        training_data_dir (str): Path to the training data directory.
        window_duration (int): Duration of the reward window in seconds.
    """
    net_rewards = 0
    while True:
        await asyncio.sleep(window_duration)
        
        # Check if there are no net rewards within the time window
        if net_rewards <= 0:
            punishment = -10  # Apply a negative punishment
            current_time = int(time.time() * 1000)
            punishment_filename = f"{current_time}_punishment.json"
            punishment_data = {
                "time": current_time,
                "punishment": punishment,
                "reason": "No net rewards within the time window."
            }
            punishment_file_path = os.path.join(training_data_dir, punishment_filename)
            
            with open(punishment_file_path, 'w') as f:
                json.dump(punishment_data, f)
            
            logging.info(f"Applied punishment to instance {instance['name']} for lack of net rewards.")
            print(f"Applied punishment to instance {instance['name']} for lack of net rewards.")
        
        # Reset net rewards for the next window
        net_rewards = 0

# Function to start instances in batches
def start_instances_in_batches(instances, batch_size=10):
    """
    Starts instances of the Tango AppImage in specified batch sizes.
    
    Args:
        instances (list): List of instance configurations to run.
        batch_size (int, optional): Number of instances to start per batch. Defaults to 10.
    """
    total_instances = len(instances)
    num_batches = (total_instances + batch_size - 1) // batch_size  # Calculate total number of batches

    with tqdm(total=num_batches, desc="Processing Batches", unit="batch") as pbar:
        for i in range(0, total_instances, batch_size):
            batch = instances[i:i+batch_size]
            batch_number = (i // batch_size) + 1
            logging.info(f"Starting batch {batch_number} with {len(batch)} instances.")
            print(f"Starting batch {batch_number} with {len(batch)} instances.")

            # Start instances in the current batch
            for instance in batch:
                run_instance(instance['rom_path'], instance['save_path'], instance['port'], instance['replay_path'])
                logging.info(f"Started instance {instance['name']} on port {instance['port']}.")
                print(f"Started instance {instance['name']} on port {instance['port']}.")
                time.sleep(0.5)  # Adjust sleep time based on app's boot time

            # Wait for the batch to complete
            logging.info(f"Processing batch {batch_number} of {num_batches}")
            print(f"Processing batch {batch_number} of {num_batches}")
            try:
                asyncio.run(process_batch(batch))
                logging.info(f"Completed batch {batch_number} of {num_batches}")
                print(f"Completed batch {batch_number} of {num_batches}")
            except Exception as e:
                logging.error(f"Error processing batch {batch_number}: {e}")
                print(f"Error processing batch {batch_number}: {e}")
            pbar.update(1)
            # Optionally, you can add a delay or cleanup here if needed

# Function to process a batch of instances
async def process_batch(instances):
    tasks = [asyncio.create_task(handle_connection(instance)) for instance in instances]
    # Wait for all handle_connection tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Batch processing completed.")
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
def save_game_state(image_path, input_binary, player_health, enemy_health, player_position, enemy_position, inside_window, reward=None, punishment=None, training_data_dir=None):
    return
    # game_state = {
    #     "image_path": image_path,
    #     "input": input_binary,
    #     "player_health": player_health,
    #     "enemy_health": enemy_health,
    #     "player_position": player_position,
    #     "enemy_position": enemy_position,
    #     "inside_window": inside_window,
    #     "reward": reward,
    #     "punishment": punishment
    # }
    # filename = f"{int(time.time() * 1000)}.json"
    # file_path = os.path.join(training_data_dir, filename)

    # with open(file_path, 'w') as f:
    #     json.dump(game_state, f)

# Function to send input command to a specific instance
async def send_input_command(writer, command):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        raise
    except Exception as e:
        logging.error(f"Failed to send command: {e}")
        print(f"Failed to send command: {e}")
        raise

# Function to request the current screen image
async def request_screen_image(writer):
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command)
    except Exception as e:
        logging.error(f"Failed to request screen image: {e}")
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
                    current_input = int_to_binary_string(int(details))
                    # print(f"Received local_input: {current_input}")

                elif event == "screen_image":
                    # Parse the details JSON
                    try:
                        screen_data = json.loads(details)
                        encoded_image = screen_data.get("image", "")
                        player_health = screen_data.get("player_health", 0)
                        enemy_health = screen_data.get("enemy_health", 0)
                        player_position = screen_data.get("player_position", None)
                        enemy_position = screen_data.get("enemy_position", None)
                        inside_window = screen_data.get("inside_window", False)
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse screen_image details: {details}")
                        print(f"Failed to parse screen_image details: {details}")
                        continue

                    # image_path = save_image_from_base64(encoded_image, port, training_data_dir)
                    
                    # # Save game state with additional data
                    # save_game_state(
                    #     image_path=image_path,
                    #     input_binary=current_input,
                    #     player_health=player_health,
                    #     enemy_health=enemy_health,
                    #     player_position=player_position,
                    #     enemy_position=enemy_position,
                    #     inside_window=inside_window,
                    #     reward=current_reward,
                    #     punishment=current_punishment,
                    #     training_data_dir=training_data_dir
                    # )
                    
                    # Reset rewards/punishments after saving
                    current_reward = None
                    current_punishment = None

                elif event == "reward":
                    try:
                        current_reward = int(details.split(":")[1].strip())
                    except ValueError:
                        logging.warning(f"Failed to parse reward message: {details}")
                        print(f"Failed to parse reward message: {details}")

                elif event == "punishment":
                    try:
                        current_punishment = int(details.split(":")[1].strip())
                    except ValueError:
                        logging.warning(f"Failed to parse punishment message: {details}")
                        print(f"Failed to parse punishment message: {details}")
                elif event == "winner":
                    player_won = details.lower() == "true"
                    save_winner_status(training_data_dir, player_won)

                else:
                    logging.info(f"Received message: Event - {event}, Details - {details}")
                    print(f"Received message: Event - {event}, Details - {details}")

    except (ConnectionResetError, BrokenPipeError):
        logging.warning(f"Connection was reset by peer on port {port}, closing receiver.")
        print(f"Connection was reset by peer on port {port}, closing receiver.")
    except Exception as e:
        logging.error(f"Failed to receive message on port {port}: {e}")
        print(f"Failed to receive message on port {port}: {e}")

# Function to save the winner status in the replay folder
def save_winner_status(training_data_dir, player_won):
    winner_status = {
        "is_winner": player_won
    }
    file_path = os.path.join(training_data_dir, "winner.json")
    with open(file_path, 'w') as f:
        json.dump(winner_status, f)
    logging.info(f"Saved winner status: {winner_status} in {file_path}")
    print(f"Saved winner status: {winner_status} in {file_path}")

# Function to handle connection to a specific instance and predict actions based on screen capture
async def handle_connection(instance, connection_timeout=10):
    writer = None  # Initialize writer to ensure it's accessible in finally block
    try:
        # Attempt to establish a connection with a timeout
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(instance['address'], instance['port']),
                timeout=connection_timeout
            )
            logging.info(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")
            print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")
        except asyncio.TimeoutError:
            logging.error(f"Connection to {instance['name']} timed out after {connection_timeout} seconds.")
            print(f"Connection to {instance['name']} timed out after {connection_timeout} seconds.")
            return  # Exit the function if connection times out
        except Exception as e:
            logging.error(f"Failed to connect to {instance['name']}: {e}")
            print(f"Failed to connect to {instance['name']}: {e}")
            return  # Exit the function on other connection errors

        # Get the training data directory based on the replay path
        training_data_dir = get_training_data_dir(instance['replay_path'])

        # Start receiving messages
        receive_task = asyncio.create_task(receive_messages(reader, instance['port'], training_data_dir))

        # Set a delay to request images
        image_request_interval = 1 / 60.0  # seconds

        while not reader.at_eof():
            try:
                await request_screen_image(writer)
                await asyncio.sleep(image_request_interval)  # Control the request rate

                # Predict and send inputs if not a player instance
                if not instance.get('is_player', False):
                    command = predict(None)
                    await send_input_command(writer, command)

            except (ConnectionResetError, BrokenPipeError):
                logging.warning(f"Connection to {instance['name']} was reset. Stopping send loop.")
                print(f"Connection to {instance['name']} was reset. Stopping send loop.")
                break  # Exit the loop
            except Exception as e:
                logging.error(f"An error occurred in connection to {instance['name']}: {e}")
                print(f"An error occurred in connection to {instance['name']}: {e}")
                break  # Exit the loop on other exceptions

        # Wait for the receive_messages task to finish
        await receive_task

    except Exception as e:
        logging.error(f"Unhandled exception with {instance['name']}: {e}")
        print(f"Unhandled exception with {instance['name']}: {e}")
    finally:
        if writer:
            try:
                writer.close()
                await writer.wait_closed()
                logging.info(f"Connection to {instance['name']} closed.")
                print(f"Connection to {instance['name']} closed.")
            except Exception as e:
                logging.error(f"Error closing connection to {instance['name']}: {e}")
                print(f"Error closing connection to {instance['name']}: {e}")

# Main function to start instances and handle inputs
def main():
    parser = argparse.ArgumentParser(description="Start Tango AppImage instances with a maximum number of replays.")
    parser.add_argument('--max_replays', type=int, default=None,
                        help='Maximum number of replays to process. If not set, all unprocessed replays will be processed.')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='Number of instances to start per batch. Default is 20.')
    parser.add_argument('--start_port', type=int, default=12345,
                        help='Starting port number for instances. Default is 12345.')

    args = parser.parse_args()

    max_replays = args.max_replays
    batch_size = args.batch_size
    starting_port = args.start_port

    print(f"Starting instances with max_replays={max_replays}, batch_size={batch_size}, starting_port={starting_port}")

    # Get the list of unprocessed replays
    unprocessed_replays = get_unprocessed_replays(max_replays)

    if not unprocessed_replays:
        logging.info("No unprocessed replays found.")
        print("No unprocessed replays found.")
        return

    # Log and print the list of unprocessed replays
    logging.info(f"Found {len(unprocessed_replays)} unprocessed replays:")
    for replay in unprocessed_replays:
        logging.info(f" - {replay}")
    print("List of unprocessed replays:")
    for replay in unprocessed_replays:
        print(f" - {replay}")

    print(f"Found {len(unprocessed_replays)} unprocessed replays.")

    # Prepare instances
    instances = []
    port = starting_port  # Starting port number; adjust if needed

    for replay_path in unprocessed_replays:
        replay_name = os.path.basename(replay_path).split('.')[0]

        # Check if the training data directory exists
        training_data_dir = os.path.join(TRAINING_DATA_DIR, replay_name)
        if os.path.exists(training_data_dir):
            logging.info(f"Training data for {replay_name} already exists. Skipping.")
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
        logging.info("No new instances to process.")
        print("No new instances to process.")
        return

    logging.info(f"Starting {len(instances)} instances.")
    print(f"Starting {len(instances)} instances.")

    # Start instances in batches with a progress bar
    start_instances_in_batches(instances, batch_size=batch_size)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
        print("\nExiting...")
