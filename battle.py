import subprocess
import os
import time
import asyncio
import json
import random

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
    },
    {
        'address': '127.0.0.1',
        'port': 12346,
        'rom_path': 'bn6,1',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
        'name': 'Instance 2',
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

# Function to send input command to a specific instance
async def send_input_command(writer, command):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
        print(f"Sent command: {command}")
    except Exception as e:
        print(f"Failed to send command: {e}")
# Function to handle connection to a specific instance
async def handle_connection(instance):
    try:
        reader, writer = await asyncio.open_connection(instance['address'], instance['port'])
        print(f"Connected to {instance['name']} at {instance['address']}:{instance['port']}")

        # List of possible keys that map correctly to Rust `Key` enum values
        possible_keys = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'A']

        # Continuously send random input commands to the instance every second
        while True:
            key = random.choice(possible_keys)
            action = random.choice(['key_press', 'key_release'])

            command = {'type': action, 'key': key}
            await send_input_command(writer, command)
            await asyncio.sleep(0.1)  # Send an input every second

    except ConnectionRefusedError:
        print(f"Failed to connect to {instance['name']}. Is the application running?")
    except Exception as e:
        print(f"An error occurred with {instance['name']}: {e}")
    finally:
        # Optional: cleanly close the connection if needed
        writer.close()
        await writer.wait_closed()
        print(f"Connection to {instance['name']} closed.")


# Main function to start instances and handle inputs
async def main():
    # Start the application instances
    start_instances()
    print("Both instances are running.")

    # Wait briefly to allow instances to fully boot up and listen
    await asyncio.sleep(2)

    # Create tasks to handle input connections for each instance
    tasks = [handle_connection(instance) for instance in INSTANCES]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
