# h5_game_dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class H5GameDataset(Dataset):
    def __init__(self, h5_path, image_memory=1, config=None):
        self.h5_path = h5_path
        self.image_memory = image_memory

        # Extract configuration for input features
        input_features = config.get('input_features', {})
        self.include_image = input_features.get('include_image', True)
        self.include_position = input_features.get('include_position', True)
        self.position_type = input_features.get('position_type', 'grid')
        self.include_player_charge = input_features.get('include_player_charge', False)
        self.include_enemy_charge = input_features.get('include_enemy_charge', False)
        self.temporal_charge = input_features.get('temporal_charge', 0)
        self.include_temporal_charge = self.temporal_charge > 0

        # Open the HDF5 file and load selected data
        with h5py.File(self.h5_path, 'r') as h5_file:
            if self.include_image:
                self.images = h5_file['images'][:].astype(np.float32)  # Shape: (num_samples, 3, 160, 240)
                print(f"Loaded images with shape: {self.images.shape}")  # Debugging
            if self.include_position:
                self.positions = h5_file['positions'][:].astype(np.float32)  # Shape: (num_samples, 4) or (num_samples, 36)
                print(f"Loaded positions with shape: {self.positions.shape}")  # Debugging
            if self.include_player_charge:
                self.player_charges = h5_file['player_charges'][:].astype(np.float32)  # Shape: (num_samples,)
                print(f"Loaded player_charges with shape: {self.player_charges.shape}")  # Debugging
            if self.include_enemy_charge:
                self.enemy_charges = h5_file['enemy_charges'][:].astype(np.float32)  # Shape: (num_samples,)
                print(f"Loaded enemy_charges with shape: {self.enemy_charges.shape}")  # Debugging
            
            # Always load net_rewards
            self.net_rewards = h5_file['net_rewards'][:].astype(np.float32)  # Shape: (num_samples,)
            print(f"Loaded net_rewards with shape: {self.net_rewards.shape}")  # Debugging
            
            # Load 'input' dataset
            if 'input' in h5_file:
                self.inputs = h5_file['input'][:].astype(np.float32)  # Shape: (num_samples, 16)
                print(f"Loaded inputs with shape: {self.inputs.shape}")  # Debugging
            else:
                raise KeyError(f"'input' dataset not found in {h5_path}")

            # Read net_reward_min and net_reward_max from file attributes
            self.net_reward_min = h5_file.attrs['net_reward_min']
            self.net_reward_max = h5_file.attrs['net_reward_max']

        # Calculate the total number of samples considering image memory and temporal_charge
        feature_lengths = []
        if self.include_image:
            feature_lengths.append(len(self.images))
        if self.include_position:
            feature_lengths.append(len(self.positions))
        if self.include_player_charge:
            feature_lengths.append(len(self.player_charges))
        if self.include_enemy_charge:
            feature_lengths.append(len(self.enemy_charges))
        feature_lengths.append(len(self.net_rewards))
        feature_lengths.append(len(self.inputs))
        
        min_length = min(feature_lengths)
        # total_samples = min_length - image_memory + 1
        total_reduction = self.image_memory - 1
        if self.include_temporal_charge:
            total_reduction += self.temporal_charge -1
        self.total_samples = min_length - total_reduction

        # Move data to GPU if available
        if torch.cuda.is_available():
            if self.include_image:
                self.images = torch.tensor(self.images).cuda()
            if self.include_position:
                self.positions = torch.tensor(self.positions).cuda()
            if self.include_player_charge:
                self.player_charges = torch.tensor(self.player_charges).cuda()
            if self.include_enemy_charge:
                self.enemy_charges = torch.tensor(self.enemy_charges).cuda()
            self.net_rewards = torch.tensor(self.net_rewards).cuda()
            self.inputs = torch.tensor(self.inputs).cuda()

        if self.total_samples < 0:
            print(f"Warning: Calculated total_samples={self.total_samples} for {h5_path} is negative. Setting to 0.")
            self.total_samples = 0

        print(f"Total samples in dataset: {self.total_samples}")  # Debugging

        # Assertion to ensure __len__() is non-negative
        assert self.total_samples >= 0, f"Dataset {h5_path} has negative total_samples={self.total_samples}"

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")
        
        sample = {}
        base_idx = idx + self.image_memory -1 + (self.temporal_charge -1 if self.include_temporal_charge else 0)
        
        if self.include_image:
            # Extract image sequence based on image_memory
            image_sequence = self.images[base_idx - (self.image_memory -1) : base_idx +1]  # Shape: (image_memory, 3, 160, 240)
            if self.image_memory == 1:
                sample['image'] = image_sequence[0].unsqueeze(1)  # Shape: (3, 1, 160, 240)
            else:
                # Shape: (3, image_memory, 160, 240)
                sample['image'] = image_sequence.permute(1, 0, 2, 3)  # Reorder to (channels, image_memory, height, width)

            # print(f"Sample image shape: {sample['image'].shape}")  # Debugging

            # Assertion to ensure image shape is correct
            expected_image_shape = (3, self.image_memory, 160, 240) if self.image_memory > 1 else (3, 1, 160, 240)
            assert sample['image'].shape == expected_image_shape, f"Unexpected image shape: {sample['image'].shape}, expected: {expected_image_shape}"
        
        if self.include_position:
            position = self.positions[base_idx]  # Shape: (4,) or (36,)
            sample['position'] = position
            # print(f"Sample position shape: {sample['position'].shape}")  # Debugging
            if self.position_type == 'float':
                assert sample['position'].shape == torch.Size([4]), f"Unexpected position shape: {sample['position'].shape}"
            elif self.position_type == 'grid':
                assert sample['position'].shape == torch.Size([36]), f"Unexpected position shape: {sample['position'].shape}"
        
        if self.include_player_charge:
            player_charge = self.player_charges[base_idx]  # Scalar
            sample['player_charge'] = player_charge
            # print(f"Sample player_charge shape: {sample['player_charge'].shape}")  # Debugging
            assert sample['player_charge'].shape == torch.Size([]), f"Unexpected player_charge shape: {sample['player_charge'].shape}"
        
        if self.include_enemy_charge:
            enemy_charge = self.enemy_charges[base_idx]  # Scalar
            sample['enemy_charge'] = enemy_charge
            # print(f"Sample enemy_charge shape: {sample['enemy_charge'].shape}")  # Debugging
            assert sample['enemy_charge'].shape == torch.Size([]), f"Unexpected enemy_charge shape: {sample['enemy_charge'].shape}"
        
        if self.include_temporal_charge:
            # Extract temporal charge sequences
            player_charge_seq = self.player_charges[base_idx - (self.temporal_charge -1) : base_idx +1]  # Shape: (temporal_charge,)
            enemy_charge_seq = self.enemy_charges[base_idx - (self.temporal_charge -1) : base_idx +1]  # Shape: (temporal_charge,)
            sample['player_charge_temporal'] = player_charge_seq  # Shape: (temporal_charge,)
            sample['enemy_charge_temporal'] = enemy_charge_seq    # Shape: (temporal_charge,)
            # print(f"Sample temporal charge shapes: player={sample['player_charge_temporal'].shape}, enemy={sample['enemy_charge_temporal'].shape}")
            assert sample['player_charge_temporal'].shape == torch.Size([self.temporal_charge]), f"Unexpected player_charge_temporal shape: {sample['player_charge_temporal'].shape}"
            assert sample['enemy_charge_temporal'].shape == torch.Size([self.temporal_charge]), f"Unexpected enemy_charge_temporal shape: {sample['enemy_charge_temporal'].shape}"
        
        # Always include net_reward
        net_reward = self.net_rewards[base_idx]  # Scalar
        sample['net_reward'] = net_reward
        # print(f"Sample net_reward shape: {sample['net_reward'].shape}")  # Debugging
        assert sample['net_reward'].shape == torch.Size([]), f"Unexpected net_reward shape: {sample['net_reward'].shape}"
        
        # Always include input
        input_tensor = self.inputs[base_idx]  # Shape: (16,)
        sample['input'] = input_tensor
        # print(f"Sample input shape: {sample['input'].shape}")  # Debugging
        assert sample['input'].shape == torch.Size([16]), f"Unexpected input shape: {sample['input'].shape}"
        
        return sample

    def __del__(self):
        pass  # Handled by context manager
