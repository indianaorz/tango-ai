# game_dataset.py
import os
import glob
import shutil
import gc
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import get_exponential_sample, get_image_memory, get_checkpoint_path, get_exponental_amount, get_root_dir  # Import the helper function
from cache_data import process_replay

class GameDataset(Dataset):
    def __init__(self, replay_dirs, image_memory=1,
                 preprocess=True, load_into_memory=False,
                 raw_dir=None, is_raw=False, 
                 load_data_into_gpu=False, device='cpu'):
        self.image_memory = image_memory
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        self.preprocess = preprocess
        self.load_into_memory = load_into_memory
        self.load_data_into_gpu = load_data_into_gpu
        self.device = device  # Device to load data onto if flag is True

        # Define TEMP_DIR if needed
        TEMP_DIR = os.path.join(get_root_dir(), 'temp_cache')
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Clear tmp dir, remove all subdirectories
        for f in os.listdir(TEMP_DIR):
            subdir_path = os.path.join(TEMP_DIR, f)
            if os.path.isdir(subdir_path):
                shutil.rmtree(subdir_path)

        if is_raw:
            for replay_dir in replay_dirs:
                # Update the argument names here
                process_replay(
                    replay_dir, 
                    planning_output_dir=os.path.join(TEMP_DIR, 'planning'), 
                    battle_output_dir=os.path.join(TEMP_DIR, 'battle')
                )
            # Update replay_dirs to include both planning and battle directories
            planning_dirs = [os.path.join(TEMP_DIR, 'planning', d) for d in os.listdir(os.path.join(TEMP_DIR, 'planning')) if os.path.isdir(os.path.join(TEMP_DIR, 'planning', d))]
            battle_dirs = [os.path.join(TEMP_DIR, 'battle', d) for d in os.listdir(os.path.join(TEMP_DIR, 'battle')) if os.path.isdir(os.path.join(TEMP_DIR, 'battle', d))]
            # Combine both directories for the dataset
            replay_dirs = planning_dirs + battle_dirs

        if not replay_dirs:
            print("No directories provided to GameDataset.")
            return

        # Top-level progress bar for replay directories
        outer_pbar = tqdm(replay_dirs, desc="Initializing Dataset",
                          unit="dir", position=0, leave=True)

        for replay_dir in outer_pbar:
            # Get sorted list of .pt files
            pt_files = sorted(glob.glob(os.path.join(replay_dir, '*.pt')))
            num_samples = len(pt_files)

            # If the number of samples is > 18000, reduce the number of samples
            if num_samples > 18000:
                pt_files = pt_files[::2]
                num_samples = len(pt_files)

            # Display progress for each .pt file within the directory
            if self.preprocess or self.load_into_memory:
                pt_progress = tqdm(range(num_samples),
                                   desc=f"Processing {os.path.basename(replay_dir)}",
                                   leave=False, unit="file", position=1)

            # Preprocess if flag is True
            if self.preprocess:
                for idx in pt_progress:
                    # Generate list of indices corresponding to pt_files
                    indices_list = list(range(len(pt_files)))
                    # Use the common get_exponential_sample function from utils.py
                    sampled_indices = get_exponential_sample(indices_list, idx, self.image_memory)
                    if not sampled_indices:
                        continue  # Skip if insufficient frames

                    # Map sampled indices to file paths
                    sample_pt_files = [pt_files[i] for i in sampled_indices]

                    try:
                        # Load net_reward from the last file only
                        sample = torch.load(sample_pt_files[-1],
                                            map_location='cpu')
                        net_reward = sample['net_reward']
                        self.net_reward_min = min(self.net_reward_min,
                                                  net_reward)
                        self.net_reward_max = max(self.net_reward_max,
                                                  net_reward)
                        sample_info = {
                            'pt_files': sample_pt_files,
                            'net_reward': net_reward
                        }
                        self.sample_paths.append(sample_info)


                        # If loading into memory, load all samples now
                        if self.load_into_memory:
                            loaded_samples = [torch.load(file,
                                                      map_location='cpu')
                                              for file in sample_pt_files]
                            
                            if self.load_data_into_gpu:
                                # Move data to GPU
                                loaded_samples = [
                                    {
                                        'image': sample['image'].to(self.device),
                                        'input': sample['input'].to(self.device),
                                        'player_health': sample['player_health'].to(self.device),
                                        'enemy_health': sample['enemy_health'].to(self.device),
                                        'player_grid': sample['player_grid'].to(self.device),
                                        'enemy_grid': sample['enemy_grid'].to(self.device),
                                        'inside_window': sample['inside_window'].to(self.device)
                                    }
                                    for sample in loaded_samples
                                ]

                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                    except Exception as e:
                        print(f"Error loading {sample_pt_files[-1]}: {e}")
                pt_progress.close()
            else:
                for idx in range(self.image_memory - 1, num_samples):
                    # Generate list of indices corresponding to pt_files
                    indices_list = list(range(len(pt_files)))
                    # Use the common get_exponential_sample function from utils.py
                    sampled_indices = get_exponential_sample(indices_list, idx, self.image_memory)
                    if not sampled_indices:
                        continue  # Skip if insufficient frames

                    # Map sampled indices to file paths
                    sample_pt_files = [pt_files[i] for i in sampled_indices]

                    sample_info = {'pt_files': sample_pt_files}
                    self.sample_paths.append(sample_info)
                    # Load into memory if specified
                    if self.load_into_memory:
                        try:
                            loaded_samples = [torch.load(file, map_location='cpu')
                                              for file in sample_pt_files]
                            net_reward = loaded_samples[-1]['net_reward']
                            self.net_reward_min = min(self.net_reward_min,
                                                      net_reward)
                            self.net_reward_max = max(self.net_reward_max,
                                                      net_reward)
                            
                            if self.load_data_into_gpu:
                                # Move data to GPU
                                loaded_samples = [
                                    {
                                        'image': sample['image'].to(self.device),
                                        'input': sample['input'].to(self.device),
                                        'player_health': sample['player_health'].to(self.device),
                                        'enemy_health': sample['enemy_health'].to(self.device),
                                        'player_grid': sample['player_grid'].to(self.device),
                                        'enemy_grid': sample['enemy_grid'].to(self.device),
                                        'inside_window': sample['inside_window'].to(self.device)
                                    }
                                    for sample in loaded_samples
                                ]

                            self.data_in_memory.append((loaded_samples,
                                                        net_reward))
                        except Exception as e:
                            print(f"Error loading files in subset: {e}")

        outer_pbar.close()
        print(f"Total samples: {len(self.sample_paths)}")
        print(f"net_reward_min: {self.net_reward_min}, "
              f"net_reward_max: {self.net_reward_max}")


    def clear_memory(self):
        self.sample_paths = []
        self.data_in_memory = []
        self.net_reward_min = float('inf')
        self.net_reward_max = float('-inf')
        gc.collect()

    def __len__(self):
        return len(self.sample_paths)

    def __del__(self):
        print("GameDataset __del__ called")
        self.clear_memory()



    def __getitem__(self, idx):
        try:
            if self.load_into_memory and idx < len(self.data_in_memory):
                # Access data directly from memory (already on GPU if flag is True)
                loaded_samples, net_reward = self.data_in_memory[idx]
                image_tensors = [sample['image'] for sample in loaded_samples]
                input_tensors = [sample['input'] for sample in loaded_samples]
                player_healths = [sample['player_health'] for sample in loaded_samples]
                enemy_healths = [sample['enemy_health'] for sample in loaded_samples]
                player_grids = [sample['player_grid'] for sample in loaded_samples]
                enemy_grids = [sample['enemy_grid'] for sample in loaded_samples]
                inside_windows = [sample['inside_window'] for sample in loaded_samples]
            else:
                # Load data from disk
                sample_info = self.sample_paths[idx]
                image_tensors = []
                input_tensors = []
                player_healths = []
                enemy_healths = []
                player_grids = []
                enemy_grids = []
                inside_windows = []
                for pt_file in sample_info['pt_files']:
                    try:
                        sample = torch.load(pt_file, map_location='cpu')  # Load to CPU first
                        if self.load_data_into_gpu:
                            # Move to GPU
                            sample['image'] = sample['image'].to(self.device)
                            sample['input'] = sample['input'].to(self.device)
                            sample['player_health'] = sample['player_health'].to(self.device)
                            sample['enemy_health'] = sample['enemy_health'].to(self.device)
                            sample['player_grid'] = sample['player_grid'].to(self.device)
                            sample['enemy_grid'] = sample['enemy_grid'].to(self.device)
                            sample['inside_window'] = sample['inside_window'].to(self.device)
                        image_tensors.append(sample['image'])
                        input_tensors.append(sample['input'])
                        player_healths.append(sample['player_health'])
                        enemy_healths.append(sample['enemy_health'])
                        player_grid = sample['player_grid']
                        enemy_grid = sample['enemy_grid']
                        # Debug: Print shapes
                        print(f"player_grid shape: {player_grid.shape}")
                        enemy_grids.append(sample['enemy_grid'])
                        inside_windows.append(sample['inside_window'])
                    except Exception as e:
                        print(f"Error loading file {pt_file}: {e}")
                        raise

                net_reward = sample_info.get('net_reward', 0.0)

            # Ensure the number of images matches image_memory
            if len(image_tensors) != self.image_memory:
                print(f"Expected {self.image_memory} images but got {len(image_tensors)} for index {idx}")
                raise ValueError(f"Image count mismatch for index {idx}")

            # Stack images along the temporal dimension
            if self.image_memory == 1:
                image_tensor = image_tensors[0].unsqueeze(1)  # Add depth dimension at dim=1
                input_tensor = input_tensors[0]
                player_health = player_healths[0]
                enemy_health = enemy_healths[0]
                player_grid = player_grids[0]  # Remove unsqueeze
                enemy_grid = enemy_grids[0]    # Remove unsqueeze
                inside_window = inside_windows[0]
            else:
                image_tensor = torch.stack(image_tensors, dim=1)  # (channels, depth, height, width)
                input_tensor = input_tensors[-1]
                player_health = player_healths[-1]
                enemy_health = enemy_healths[-1]
                player_grid = player_grids[-1]  # Remove unsqueeze
                enemy_grid = enemy_grids[-1]    # Remove unsqueeze
                inside_window = inside_windows[-1]

            net_reward_tensor = torch.tensor(net_reward, dtype=torch.float32)

            return {
                'image': image_tensor,
                'input': input_tensor,
                'player_health': player_health,
                'enemy_health': enemy_health,
                'player_grid': player_grid,
                'enemy_grid': enemy_grid,
                'inside_window': inside_window,
                'net_reward': net_reward_tensor
            }


        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise
