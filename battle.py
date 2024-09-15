import subprocess
import os
import time
import asyncio
import json
import random
import base64
from PIL import Image
from io import BytesIO

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "your_link_code"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "your_matchmaking_id"  # Replace with the actual matchmaking ID

# Define the server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12345,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
        'name': 'Instance 1',
        'is_player': True  # Set to True if you don't want this instance to send inputs
    },
    # {
    #     'address': '127.0.0.1',
    #     'port': 12346,
    #     'rom_path': 'bn6,1',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar.sav',
    #     'name': 'Instance 1',
    #     'is_player': True  # Set to True if you don't want this instance to send inputs
    # },
]

# Function to run the AppImage with specific ROM, SAVE paths, and PORT
def run_instance(rom_path, save_path, port):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["PORT"] = str(port)

    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    subprocess.Popen([APP_PATH], env=env)

# Function to start all instances
def start_instances():
    for instance in INSTANCES:
        run_instance(instance['rom_path'], instance['save_path'], instance['port'])
        time.sleep(0.5)  # Adjust sleep time based on app's boot time

# Placeholder predict method for AI inference (currently random actions)
def predict(image):
    possible_keys = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'A']
    action = random.choice(['key_press', 'key_release'])
    key = random.choice(possible_keys)
    return {'type': action, 'key': key}

# Function to save images received from the app
def save_image_from_base64(encoded_image, port):
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_image))
    filename = f"{port}.png"
    image.save(filename)
    print(f"Saved image as {filename}")

# Function to send input command to a specific instance
async def send_input_command(writer, command):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
        print(f"Sent command: {command}")
    except Exception as e:
        print(f"Failed to send command: {e}")

# Function to request the current screen image
async def request_screen_image(writer):
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command)
    except Exception as e:
        print(f"Failed to request screen image: {e}")



# Function to receive messages from the game and process them
async def receive_messages(reader, port):
    buffer = ""
    while True:
        try:
            # Read larger chunks to handle potentially large data
            data = await reader.read(4096)
            if not data:
                break

            # Accumulate data in buffer
            buffer += data.decode()

            # Try parsing JSON from the buffer
            while True:
                try:
                    # Attempt to parse the first complete JSON object
                    json_end_index = buffer.find('}\n')
                    if json_end_index == -1:
                        json_end_index = buffer.find('}')
                    
                    if json_end_index == -1:
                        # No complete JSON object found yet
                        break

                    # Extract the complete JSON message
                    json_message = buffer[:json_end_index + 1].strip()
                    buffer = buffer[json_end_index + 1:]  # Remove processed message from buffer

                    # Parse and handle the JSON message
                    parsed_message = json.loads(json_message)
                    event = parsed_message.get("event", "Unknown")
                    details = parsed_message.get("details", "No details provided")

                    # Handle specific events
                    if event == "screen_image":
                        print(f"Received screen_image event for port {port}.")
                        save_image_from_base64(details, port)
                    elif event == "reward":
                        print(f"Received reward message: {details}")
                    elif event == "punishment":
                        print(f"Received punishment message: {details}")
                    else:
                        print(f"Received message: Event - {event}, Details - {details}")

                except json.JSONDecodeError:
                    # If JSON parsing fails, continue accumulating
                    break

        except Exception as e:
            print(f"Failed to receive message: {e}")
            break


# Function to handle connection to a specific instance and predict actions based on screen capture
async def handle_connection(instance):
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        # Start receiving messages from the game
        asyncio.create_task(receive_messages(reader, instance['port']))

        # Initialize a counter to manage the screen request interval
        screen_request_interval = 50  # Every 5 seconds if loop sleeps for 0.1s
        counter = 0

        while True:
            # Request the screen image periodically
            if counter % screen_request_interval == 0:
                await request_screen_image(writer)

            # Predict and send inputs if not a player instance
            if not instance.get('is_player', False):
                # Capture and predict inputs based on an internal timer, not actual screen capture
                command = predict(None)
                await send_input_command(writer, command)

            counter += 1
            await asyncio.sleep(0.1)

    except ConnectionRefusedError:
        print(f"Failed to connect to {instance['name']}. Is the application running?")
    except Exception as e:
        print(f"An error occurred with {instance['name']}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Connection to {instance['name']} closed.")

# Main function to start instances and handle inputs
async def main():
    start_instances()
    print("Both instances are running.")
    await asyncio.sleep(2)

    tasks = [handle_connection(instance) for instance in INSTANCES]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
