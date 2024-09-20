# game_dataset.py
import torch
from torch.utils.data import Dataset, ConcatDataset
from h5_game_dataset import H5GameDataset  # Use the updated H5GameDataset

class GameDataset(Dataset):
    def __init__(self, h5_paths, image_memory=1):
        """
        h5_paths: List of paths to HDF5 files
        """
        self.datasets = [H5GameDataset(h5_path, image_memory=image_memory) for h5_path in h5_paths]
        self.combined_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]
