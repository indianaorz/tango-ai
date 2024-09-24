# online_arena.py
import os
import time
import shutil
import subprocess
from datetime import datetime
import asyncio
import math
import yaml

# Paths
REPLAYS_DIR = '/home/lee/Documents/Tango/replays'
NEW_REPLAY_FOLDER_BASE = '/home/lee/Documents/Tango/replays_'
CONFIG_PATH = 'config.yaml'  # Assuming a config file exists for parameters

# Load configuration
def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

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

# Function to run battle instances with a specific GAMMA
def run_battle_instances(gamma):
    try:
        # Prepare environment variables, including GAMMA
        env = os.environ.copy()
        env["GAMMA"] = str(gamma)
        print(f"Launching battle instances with GAMMA={gamma:.4f}")
        
        # Run online_learn_battle.py as a subprocess
        # You can adjust the command as needed, e.g., run multiple instances if required
        subprocess.run(["python3", "fight_and_learn.py"], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Battle instances terminated with an error: {e}")
    except Exception as e:
        print(f"Failed to run battle instances: {e}")

# Function to run data capture on saved replays
def run_data_capture():
    try:
        # Run the data capture script
        subprocess.run(["python3", "datacapture.py"], check=True)
    except Exception as e:
        print(f"Failed to run data capture: {e}")

# Cosine annealing function
def cosine_annealing(current_step, total_steps, initial_gamma=0.1, min_gamma=0.01):
    """
    Compute GAMMA using cosine annealing.
    
    Args:
        current_step (int): The current training step or cycle.
        total_steps (int): The total number of steps over which to anneal GAMMA.
        initial_gamma (float): The starting value of GAMMA.
        min_gamma (float): The minimum value of GAMMA.
        
    Returns:
        float: The updated GAMMA value.
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
    return min_gamma + (initial_gamma - min_gamma) * cosine_decay

# Main function to run the entire process with cosine annealing for GAMMA
async def main():
    # Load or set initial GAMMA parameters
    initial_gamma = config.get('gamma', {}).get('initial', 0.25)
    min_gamma = config.get('gamma', {}).get('min', 0.01)
    total_steps = config.get('gamma', {}).get('total_steps', 10)  # Define total steps or cycles
    
    current_step = 0

    while True:
        if current_step >= total_steps:
            # Reset or stop annealing if desired
            print("Completed all annealing steps. Resetting current_step.")
            current_step = 0  # Or break if you want to stop
            # Optionally, you can set GAMMA back to initial_gamma or keep it at min_gamma
            # break

        # Compute GAMMA using cosine annealing
        gamma = cosine_annealing(current_step, total_steps, initial_gamma, min_gamma)
        print(f"Cycle {current_step + 1}/{total_steps}: GAMMA={gamma:.4f}")
        
        # Step 1: Run the battle instances with the current GAMMA
        run_battle_instances(gamma)

        # Step 2: Optionally, run data capture on completed battles
        # print("Running data capture on replays...")
        # run_data_capture()

        # Step 3: Move old replays to a new folder
        print("Moving replays to a new folder...")
        move_replays()

        # Step 4: Increment the step and wait before the next cycle
        current_step += 1
        print("Cycle completed. Waiting for the next cycle...")
        await asyncio.sleep(5)  # Adjust the delay time as needed

# Optional: Update configuration file with current GAMMA (for monitoring)
# def update_config_with_gamma(current_step, gamma, config_path=CONFIG_PATH):
#     config = load_config(config_path)
#     config['gamma']['current_step'] = current_step
#     config['gamma']['current_gamma'] = gamma
#     with open(config_path, 'w') as f:
#         yaml.dump(config, f)

# Updated main function with config updates
async def main_with_config_update():
    # Load or set initial GAMMA parameters
    config = load_config()
    initial_gamma = config.get('gamma', {}).get('initial', 0.5)
    min_gamma = config.get('gamma', {}).get('min', 0.01)
    total_steps = config.get('gamma', {}).get('total_steps', 100)  # Define total steps or cycles
    
    current_step = 0

    while True:
        if current_step >= total_steps:
            # Reset or stop annealing if desired
            print("Completed all annealing steps. Resetting current_step.")
            current_step = 0  # Or break if you want to stop
            # Optionally, you can set GAMMA back to initial_gamma or keep it at min_gamma
            # break

        # Compute GAMMA using cosine annealing
        gamma = cosine_annealing(current_step, total_steps, initial_gamma, min_gamma)
        print(f"Cycle {current_step + 1}/{total_steps}: GAMMA={gamma:.4f}")
        
        # Optionally, update the config with the current GAMMA
        # update_config_with_gamma(current_step, gamma, CONFIG_PATH)
        
        # Step 1: Run the battle instances with the current GAMMA
        run_battle_instances(gamma)

        # Step 2: Optionally, run data capture on completed battles
        print("Running data capture on replays...")
        run_data_capture()

        # Step 3: Move old replays to a new folder
        print("Moving replays to a new folder...")
        move_replays()

        # Step 4: Increment the step and wait before the next cycle
        current_step += 1
        print("Cycle completed. Waiting for the next cycle...")
        await asyncio.sleep(5)  # Adjust the delay time as needed

# Uncomment and use the appropriate main function
if __name__ == '__main__':
    try:
        asyncio.run(main())  # Use main() if not updating config
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
