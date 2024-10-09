
import os
import pickle
from utils import get_root_dir, get_latest_checkpoint, extract_number_from_checkpoint
from planning_model import PlanningModel
import torch
from tqdm import tqdm
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

#import defaultdict
from collections import defaultdict

planning_model = None
planning_data_buffers = defaultdict(list)  
optimizer_planning = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latest_checkpoint_number = {'planning': 0, 'battle': 0}

CHECKPOINTS_DIR = os.path.join(get_root_dir(), "checkpoints")
TRAINING_DATA_DIR = os.path.join(CHECKPOINTS_DIR, "training_data")
def load_data():
    global planning_model, planning_data_buffers, optimizer_planning
    planning_data_path = os.path.join(TRAINING_DATA_DIR, "planning_data.pkl")
    print(planning_data_path)
    if os.path.exists(planning_data_path):
        with open(planning_data_path, 'rb') as f:
            planning_data_buffers = pickle.load(f)
        print(f"[LOAD] planning_data loaded from {planning_data_path}")
        
    # Load Training Planning Model
    training_planning_checkpoint_path = get_latest_checkpoint(model_type='planning', image_memory=1)
    if training_planning_checkpoint_path:
        planning_model = PlanningModel().to(device)
        checkpoint_training_planning = torch.load(training_planning_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_training_planning:
            planning_model.load_state_dict(checkpoint_training_planning['model_state_dict'])
            print(f"Training Planning Model loaded from {training_planning_checkpoint_path}")
            # Extract the checkpoint number
            latest_number = extract_number_from_checkpoint(training_planning_checkpoint_path)
            latest_checkpoint_number['planning'] = latest_number
        else:
            raise KeyError("Training Planning checkpoint does not contain 'model_state_dict'")
        
        if 'optimizer_state_dict' in checkpoint_training_planning:
            optimizer_planning = optim.Adam(planning_model.parameters(), lr=1e-4)
            optimizer_planning.load_state_dict(checkpoint_training_planning['optimizer_state_dict'])
            print("Optimizer state loaded.")
        else:
            optimizer_planning = optim.Adam(planning_model.parameters(), lr=1e-4)
            raise KeyError("Training Planning checkpoint does not contain 'optimizer_state_dict'")
    else:
        # Initialize new Training Planning Model
        planning_model = PlanningModel().to(device)
        optimizer_planning = optim.Adam(planning_model.parameters(), lr=1e-4)
        print("No Training Planning Model checkpoint found. Initialized a new Training Planning Model.")
        
          
def train():
    
    print(len(planning_data_buffers))
    pbar = tqdm(range(1000), desc="Training Planning Model")
    for epoch in pbar:
        # Iterate over each port and its buffer
        for port, buffer in planning_data_buffers.items():
            if not buffer:
                # print(f"[INFO] No data in buffer for port {port}. Skipping.")
                continue
            
            # print(f"[INFO] Training on port {port} with {len(buffer)} data points.")
            
            # Prepare batch_data as a list of dicts
            batch_data = []
            for data_point in buffer:
                try:
                    inputs = data_point['inputs']
                    
                    # Shuffle player_folder and enemy_folder
                    inputs['player_folder'] = planning_model.shuffle_folder_encoded(inputs['player_folder'])
                    inputs['enemy_folder'] = planning_model.shuffle_folder_encoded(inputs['enemy_folder'])
    
                    cross_target = data_point['cross_target']
                    target_list = data_point['target_list']
                    reward = 1#data_point['reward'] 
                    
                    # Restructure data_point to match train_batch expectations
                    batch_dict = {
                        'inputs': inputs,
                        'cross_target': cross_target,
                        'target_list': target_list,
                        'reward': reward
                    }
                    
                    batch_data.append(batch_dict)
                except KeyError as e:
                    print(f"[ERROR] Missing key {e} in data_point for port {port}. Skipping this data_point.")
                    continue
            
            if not batch_data:
                print(f"[WARN] No valid data points found for port {port}. Skipping training for this port.")
                continue
            
            # Call the train_batch method
            loss, count = planning_model.train_batch(
                batch_data=batch_data,
                optimizer=optimizer_planning,
                max_grad_norm=1.0
            )
            # Update the progress bar description with the loss
            pbar.set_description(f"Loss: {loss:.4f}")
            # Update the bar with the loss
            # tqdm.write(f"Epoch {epoch}, Port {port}: Loss: {loss}, Num Trained: {count}")
            
    #save the model
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"planning_{latest_checkpoint_number['planning'] + 1}.pth")
    torch.save({
        'model_state_dict': planning_model.state_dict(),
        'optimizer_state_dict': optimizer_planning.state_dict(),
    }, checkpoint_path)
    print(f"Planning Model saved to {checkpoint_path}")
           
load_data()
train()