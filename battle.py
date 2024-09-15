import subprocess
import os
import time
import asyncio
import json
import random
import mss
import re
from PIL import Image

# Path to the Tango AppImage
APP_PATH = "./dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "your_link_code"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "your_matchmaking_id"  # Replace with the actual matchmaking ID

# Define the server addresses and ports for each instance
INSTANCES = [
    # {
    #     'address': '127.0.0.1',
    #     'port': 12345,
    #     'rom_path': 'bn6,0',
    #     'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
    #     'name': 'Instance 1',
    #     'is_player': True
    # },
    {
        'address': '127.0.0.1',
        'port': 12346,
        'rom_path': 'bn6,1',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
        'name': 'Instance 2',
        'is_player': True
    },
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

# Function to find the game window by title (which is the port number)
def find_game_window(port):
    result = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if str(port) in line:
            window_id = line.split()[0]
            return window_id
    raise Exception(f"No window found with title containing '{port}'")

# Function to get window geometry using xwininfo
def get_window_geometry(window_id):
    result = subprocess.run(['xwininfo', '-id', window_id], stdout=subprocess.PIPE, text=True)
    x = y = width = height = 0
    for line in result.stdout.splitlines():
        if "Absolute upper-left X" in line:
            x = int(re.search(r'\d+', line).group())
        elif "Absolute upper-left Y" in line:
            y = int(re.search(r'\d+', line).group())
        elif "Width" in line:
            width = int(re.search(r'\d+', line).group())
        elif "Height" in line:
            height = int(re.search(r'\d+', line).group())
    return {"left": x, "top": y, "width": width, "height": height}

# Function to capture the game window using mss
def capture_window(geometry):
    with mss.mss() as sct:
        sct_img = sct.grab(geometry)
        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return image

# Placeholder predict method for AI inference (currently random actions)
def predict(image):
    possible_keys = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'A']
    action = random.choice(['key_press', 'key_release'])
    key = random.choice(possible_keys)
    return {'type': action, 'key': key}

# Function to save captured images for testing
def save_image(image, port):
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

# Function to receive messages from the game and print them
async def receive_messages(reader):
    while True:
        try:
            data = await reader.read(1024)
            if not data:
                break
            message = data.decode().strip()
            
            # Parse the message as JSON if possible
            try:
                parsed_message = json.loads(message)
                event = parsed_message.get("event", "Unknown")
                details = parsed_message.get("details", "No details provided")

                # Print the event type and details
                print(f"Received message: Event - {event}, Details - {details}")
            except json.JSONDecodeError:
                # If message is not JSON, just print the raw message
                print(f"Received raw message: {message}")

        except Exception as e:
            print(f"Failed to receive message: {e}")
            break


# Function to handle connection to a specific instance and predict actions based on screen capture
async def handle_connection(instance):
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        # Start receiving messages from the game
        asyncio.create_task(receive_messages(reader))

        # Find the corresponding game window based on the port number
        window_id = find_game_window(instance['port'])
        print(f"Found window for {instance['name']} with ID {window_id}")

        # Initialize a counter to manage the saving interval
        save_interval = 20
        counter = 0

        while True:
            geometry = get_window_geometry(window_id)
            image = capture_window(geometry)

            # Save the captured image every 2 seconds
            if counter % save_interval == 0:
                save_image(image, instance['port'])

            if not instance.get('is_player', False):
                command = predict(image)
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
