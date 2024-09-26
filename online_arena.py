# online_arena.py

import os
import time
import shutil
import subprocess
from datetime import datetime
import asyncio
import math
import yaml
import torch
import traceback
import gc

# Import necessary functions from training_utils.py
from train_utils import final_training_epoch, save_models
from utils import get_checkpoint_dir, get_latest_checkpoint, get_new_checkpoint_path, get_root_dir, get_image_memory  # Ensure these are accessible

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
        
        # Run fight_and_learn.py as a subprocess
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

# Function to load trained models
def load_trained_models(get_checkpoint_dir, get_latest_checkpoint, image_memory=1, device='cuda'):
    """
    Loads the latest trained Planning and Battle models.

    Args:
        get_checkpoint_dir (function): Function to get checkpoint directory.
        get_latest_checkpoint (function): Function to get the latest checkpoint file.
        image_memory (int): Image memory parameter.
        device (str): Device to load the models on.

    Returns:
        tuple: (training_planning_model, training_battle_model, optimizer_planning, optimizer_battle)
    """
    from train import GameInputPredictor  # Import the model class
    import torch.optim as optim

    # Load Training Planning Model
    training_planning_checkpoint_path = get_latest_checkpoint(model_type='planning', image_memory=image_memory)

    if training_planning_checkpoint_path:
        training_planning_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        checkpoint_training_planning = torch.load(training_planning_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_training_planning:
            training_planning_model.load_state_dict(checkpoint_training_planning['model_state_dict'])
            print(f"Training Planning Model loaded from {training_planning_checkpoint_path}")
        else:
            raise KeyError("Training Planning checkpoint does not contain 'model_state_dict'")
    else:
        # Initialize new Training Planning Model
        training_planning_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        print("No Training Planning Model checkpoint found. Initialized a new Training Planning Model.")

    training_planning_model.train()  # Set to train mode

    # Load Training Battle Model
    training_battle_checkpoint_path = get_latest_checkpoint(model_type='battle', image_memory=image_memory)

    if training_battle_checkpoint_path:
        training_battle_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        checkpoint_training_battle = torch.load(training_battle_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_training_battle:
            training_battle_model.load_state_dict(checkpoint_training_battle['model_state_dict'])
            print(f"Training Battle Model loaded from {training_battle_checkpoint_path}")
        else:
            raise KeyError("Training Battle checkpoint does not contain 'model_state_dict'")
    else:
        # Initialize new Training Battle Model
        training_battle_model = GameInputPredictor(image_memory=image_memory, config=config).to(device)
        print("No Training Battle Model checkpoint found. Initialized a new Training Battle Model.")

    training_battle_model.train()  # Set to train mode

    # Initialize separate optimizers for Training Models
    learning_rate = config.get('learning_rate', 5e-5)
    optimizer_planning = optim.Adam(training_planning_model.parameters(), lr=learning_rate)
    optimizer_battle = optim.Adam(training_battle_model.parameters(), lr=learning_rate)

    return training_planning_model, training_battle_model, optimizer_planning, optimizer_battle

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

        # # Step 2: Optionally, run data capture on completed battles
        # print("Running data capture on replays...")
        # run_data_capture()

        # # Step 3: Move old replays to a new folder
        # print("Moving replays to a new folder...")
        move_replays()

        # # Step 4: Load trained models for final training
        try:
            training_planning_model, training_battle_model, optimizer_planning, optimizer_battle = load_trained_models(
                get_checkpoint_dir, 
                get_latest_checkpoint, 
                image_memory=get_image_memory(), 
                device='cuda'
            )
        except Exception as e:
            print(f"Failed to load trained models: {e}")
            traceback.print_exc()
            # Optionally, handle the error (e.g., skip training, retry, etc.)
            continue

        # Step 5: Perform Final Training Epoch
        print("Starting final training epoch on 100 batches of accumulated data.")
        battle_data_dir = os.path.join(get_root_dir(), "data", "battle_data")
        planning_data_dir = os.path.join(get_root_dir(), "data", "planning_data")

        # Train on Battle Data
        if os.path.exists(battle_data_dir) and os.listdir(battle_data_dir):
            try:
                final_training_epoch(
                    model=training_battle_model,
                    optimizer=optimizer_battle,
                    training_data_dir=battle_data_dir,
                    model_type='Battle_Model',
                    batch_size=64,  # Adjust based on your system
                    num_workers=4,  # Increased workers for faster data loading
                    device='cuda',
                    max_batches=500
                )
            except Exception as e:
                print(f"Failed to train Battle_Model: {e}")
                traceback.print_exc()
        else:
            print("No Battle data available for final training.")


        # Step 6: Save the models
        save_models(
            None, 
            training_battle_model, 
            None, 
            optimizer_battle, 
            get_checkpoint_dir, 
            get_latest_checkpoint, 
            MAX_CHECKPOINTS=config.get('MAX_CHECKPOINTS', 5), 
            IMAGE_MEMORY=get_image_memory()
        )


        #clear
        del training_battle_model
        del optimizer_battle
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()


        # Train on Planning Data
        if os.path.exists(planning_data_dir) and os.listdir(planning_data_dir):
            try:
                final_training_epoch(
                    model=training_planning_model,
                    optimizer=optimizer_planning,
                    training_data_dir=planning_data_dir,
                    model_type='Planning_Model',
                    batch_size=64,  # Adjust based on your system
                    num_workers=4,  # Increased workers for faster data loading
                    device='cuda',
                    max_batches=500
                )
            except Exception as e:
                print(f"Failed to train Planning_Model: {e}")
                traceback.print_exc()
        else:
            print("No Planning data available for final training.")

        # Step 6: Save the models
        save_models(
            training_planning_model, 
            None, 
            optimizer_planning, 
            None, 
            get_checkpoint_dir, 
            get_latest_checkpoint, 
            MAX_CHECKPOINTS=config.get('MAX_CHECKPOINTS', 5), 
            IMAGE_MEMORY=get_image_memory()
        )

        # Clear memory
        del training_planning_model
        del optimizer_planning
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

        # Step 7: Increment the step and wait before the next cycle
        current_step += 1
        print("Cycle completed. Waiting for the next cycle...")
        await asyncio.sleep(5)  # Adjust the delay time as needed

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
