# h5_game_dataset.py
import h5py
import torch
from torch.utils.data import Dataset
from utils import get_exponential_sample, get_image_memory, get_checkpoint_path, get_exponental_amount, get_root_dir  # Import the helper function

class H5GameDataset(Dataset):
    def __init__(self, h5_path, device='cpu'):
        super(H5GameDataset, self).__init__()
        self.h5_path = h5_path
        self.device = device
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.images = self.h5_file['images']
        self.inputs = self.h5_file['inputs']
        self.player_healths = self.h5_file['player_healths']
        self.enemy_healths = self.h5_file['enemy_healths']
        self.player_grids = self.h5_file['player_grids']
        self.enemy_grids = self.h5_file['enemy_grids']
        self.inside_windows = self.h5_file['inside_windows']
        self.net_rewards = self.h5_file['net_rewards']
        
        # Read net_reward_min and net_reward_max from file attributes
        self.net_reward_min = self.h5_file.attrs['net_reward_min']
        self.net_reward_max = self.h5_file.attrs['net_reward_max']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Directly load the image without permutation
        image = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0  # Shape: (3, D, 160, 240)
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        player_health = torch.tensor(self.player_healths[idx], dtype=torch.float32)
        enemy_health = torch.tensor(self.enemy_healths[idx], dtype=torch.float32)
        player_grid = torch.tensor(self.player_grids[idx], dtype=torch.float32)
        enemy_grid = torch.tensor(self.enemy_grids[idx], dtype=torch.float32)
        inside_window = torch.tensor(self.inside_windows[idx], dtype=torch.float32)
        net_reward = torch.tensor(self.net_rewards[idx], dtype=torch.float32)
        
        # Move to device if necessary
        if self.device != 'cpu':
            image = image.to(self.device)
            input_tensor = input_tensor.to(self.device)
            player_health = player_health.to(self.device)
            enemy_health = enemy_health.to(self.device)
            player_grid = player_grid.to(self.device)
            enemy_grid = enemy_grid.to(self.device)
            inside_window = inside_window.to(self.device)
            net_reward = net_reward.to(self.device)
        
        return {
            'image': image,
            'input': input_tensor,
            'player_health': player_health,
            'enemy_health': enemy_health,
            'player_grid': player_grid,
            'enemy_grid': enemy_grid,
            'inside_window': inside_window,
            'net_reward': net_reward
        }

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
