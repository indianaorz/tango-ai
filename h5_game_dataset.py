import h5py
import torch
from torch.utils.data import Dataset

class H5GameDataset(Dataset):
    def __init__(self, h5_path, image_memory=1, device='cuda'):
        """
        Initializes the dataset by loading all data into GPU memory.

        Args:
            h5_path (str): Path to the HDF5 file.
            image_memory (int, optional): Number of image frames to include in a sequence. Defaults to 1.
            device (str, optional): Device to load the tensors onto ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(H5GameDataset, self).__init__()
        self.h5_path = h5_path
        self.device = device
        self.image_memory = image_memory

        # Open the HDF5 file and load data
        with h5py.File(self.h5_path, 'r') as h5_file:
            # Convert datasets to PyTorch tensors and move to device
            self.images = torch.from_numpy(h5_file['images'][:]).float().to(self.device)             # Shape: (N, 3, 160, 240)
            self.inputs = torch.from_numpy(h5_file['inputs'][:]).float().to(self.device)             # Shape: (N, 16)
            self.player_healths = torch.from_numpy(h5_file['player_healths'][:]).float().to(self.device)  # Shape: (N, 1)
            self.enemy_healths = torch.from_numpy(h5_file['enemy_healths'][:]).float().to(self.device)    # Shape: (N, 1)
            self.player_grids = torch.from_numpy(h5_file['player_grids'][:]).float().to(self.device)      # Shape: (N, 6, 3)
            self.enemy_grids = torch.from_numpy(h5_file['enemy_grids'][:]).float().to(self.device)        # Shape: (N, 6, 3)
            self.inside_windows = torch.from_numpy(h5_file['inside_windows'][:]).float().to(self.device)  # Shape: (N, 1)
            self.net_rewards = torch.from_numpy(h5_file['net_rewards'][:]).float().to(self.device)        # Shape: (N,)

            # Read net_reward_min and net_reward_max from file attributes
            self.net_reward_min = h5_file.attrs['net_reward_min']
            self.net_reward_max = h5_file.attrs['net_reward_max']
        
        # Calculate the total number of samples considering image memory
        self.total_samples = len(self.images) - self.image_memory + 1  # Number of possible sequences

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing all relevant tensors for the sample.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")
        
        # Extract a sequence of images based on image_memory
        image_sequence = self.images[idx : idx + self.image_memory]  # Shape: (image_memory, 3, 160, 240)
        
        if self.image_memory == 1:
            # Return a single image without the image_memory dimension
            image = image_sequence[0]  # Shape: (3, 160, 240)
        else:
            # Return a sequence of images
            image = image_sequence  # Shape: (image_memory, 3, 160, 240)
        
        # Retrieve corresponding inputs and features for the last frame in the sequence
        input_tensor = self.inputs[idx + self.image_memory - 1]             # Shape: (16,)
        player_health = self.player_healths[idx + self.image_memory - 1]   # Shape: (1,)
        enemy_health = self.enemy_healths[idx + self.image_memory - 1]     # Shape: (1,)
        player_grid = self.player_grids[idx + self.image_memory - 1]       # Shape: (6, 3)
        enemy_grid = self.enemy_grids[idx + self.image_memory - 1]         # Shape: (6, 3)
        inside_window = self.inside_windows[idx + self.image_memory - 1]   # Shape: (1,)
        net_reward = self.net_rewards[idx + self.image_memory - 1]         # Shape: ()

        return {
            'image': image,                     # (3, 160, 240) or (image_memory, 3, 160, 240)
            'input': input_tensor,              # (16,)
            'player_health': player_health,     # (1,)
            'enemy_health': enemy_health,       # (1,)
            'player_grid': player_grid,         # (6, 3)
            'enemy_grid': enemy_grid,           # (6, 3)
            'inside_window': inside_window,     # (1,)
            'net_reward': net_reward            # ()
        }

    def __del__(self):
        """Ensures that the HDF5 file is closed when the dataset is deleted."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
