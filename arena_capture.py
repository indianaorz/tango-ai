import os
import time
import shutil
import subprocess
from datetime import datetime
import asyncio

# Paths
REPLAYS_DIR = '/home/lee/Documents/Tango/replays'
NEW_REPLAY_FOLDER_BASE = '/home/lee/Documents/Tango/replays_'

# Function to move the replay folder to a new directory with the current date and time
def move_replays():
    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the new folder path
    new_replay_folder = f"{NEW_REPLAY_FOLDER_BASE}{timestamp}"

    # Move the replays folder
    try:
        shutil.move(REPLAYS_DIR, new_replay_folder)
        print(f"Moved replays to {new_replay_folder}")
    except Exception as e:
        print(f"Failed to move replays: {e}")

# Function to run battle instances
def run_battle_instances():
    try:
        # Assuming battle.py is set up to handle everything automatically
        subprocess.run(["python3", "battle.py"], check=True)
    except Exception as e:
        print(f"Failed to run battle instances: {e}")

# Function to run data capture on saved replays
def run_data_capture():
    try:
        # Run the data capture script
        subprocess.run(["python3", "datacapture.py"], check=True)
    except Exception as e:
        print(f"Failed to run data capture: {e}")

# Main function to run the entire process
async def main():
    while True:
        # Step 1: Run the battle instances
        print("Starting battle instances...")
        run_battle_instances()

        # Step 2: Run data capture on completed battles
        print("Running data capture on replays...")
        run_data_capture()

        # Step 3: Move old replays to a new folder
        print("Moving replays to a new folder...")
        move_replays()

        # Step 4: Wait a bit before restarting the process (if needed)
        print("Process completed. Waiting for the next cycle...")
        await asyncio.sleep(5)  # Adjust the delay time if needed

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
