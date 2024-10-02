# h5_game_dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string.
        sep (str): Separator between parent and child keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='_'):
    """
    Unflattens a flattened dictionary back into a nested dictionary.

    Args:
        d (dict): The flattened dictionary.
        sep (str): Separator used in keys.

    Returns:
        dict: A nested dictionary.
    """
    result_dict = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        d_ref = result_dict
        for key in keys[:-1]:
            if key not in d_ref:
                d_ref[key] = {}
            d_ref = d_ref[key]
        d_ref[keys[-1]] = value
    return result_dict

class H5GameDataset(Dataset):
    def __init__(self, h5_path, model_type='Battle_Model', image_memory=1, config=None):
        """
        Initializes the dataset by loading relevant data from an HDF5 file based on the model type.

        Args:
            h5_path (str): Path to the HDF5 file.
            model_type (str): Type of model ('Battle_Model' or 'Planning_Model').
            image_memory (int): Number of sequential images to include.
            config (dict, optional): Configuration dictionary for input features.
        """
        self.h5_path = h5_path
        self.model_type = model_type
        self.image_memory = image_memory

        # Validate model_type
        if self.model_type not in ['Battle_Model', 'Planning_Model']:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported types: 'Battle_Model', 'Planning_Model'.")

        # Extract configuration for input features
        input_features = config.get('input_features', {}) if config else {}
        self.include_image = input_features.get('include_image', True) if self.model_type == 'Battle_Model' else False
        self.include_position = input_features.get('include_position', True) if self.model_type == 'Battle_Model' else False
        self.position_type = input_features.get('position_type', 'grid') if self.model_type == 'Battle_Model' else None
        self.include_player_charge = input_features.get('include_player_charge', False) if self.model_type == 'Battle_Model' else False
        self.include_enemy_charge = input_features.get('include_enemy_charge', False) if self.model_type == 'Battle_Model' else False
        self.temporal_charge = input_features.get('temporal_charge', 0) if self.model_type == 'Battle_Model' else 0
        self.include_temporal_charge = self.temporal_charge > 0 if self.model_type == 'Battle_Model' else False

        # Initialize storage for datasets
        self.data = {}

        # Open the HDF5 file and load selected data based on model_type
        with h5py.File(self.h5_path, 'r') as h5_file:
            if self.model_type == 'Battle_Model':
                if self.include_image:
                    self.data['images'] = h5_file['images'][:].astype(np.float32)  # Shape: (num_samples, 3, 160, 240)
                    print(f"Loaded images with shape: {self.data['images'].shape}")  # Debugging
                if self.include_position:
                    self.data['positions'] = h5_file['positions'][:].astype(np.float32)  # Shape: (num_samples, 4) or (num_samples, 36)
                    print(f"Loaded positions with shape: {self.data['positions'].shape}")  # Debugging
                if self.include_player_charge:
                    self.data['player_charges'] = h5_file['player_charges'][:].astype(np.float32)  # Shape: (num_samples,)
                    print(f"Loaded player_charges with shape: {self.data['player_charges'].shape}")  # Debugging
                if self.include_enemy_charge:
                    self.data['enemy_charges'] = h5_file['enemy_charges'][:].astype(np.float32)  # Shape: (num_samples,)
                    print(f"Loaded enemy_charges with shape: {self.data['enemy_charges'].shape}")  # Debugging

                # Always load net_rewards
                self.data['net_rewards'] = h5_file['net_rewards'][:].astype(np.float32)  # Shape: (num_samples,)
                print(f"Loaded net_rewards with shape: {self.data['net_rewards'].shape}")  # Debugging

                # Load 'input' dataset
                if 'input' in h5_file:
                    self.data['input'] = h5_file['input'][:].astype(np.float32)  # Shape: (num_samples, 16)
                    print(f"Loaded input with shape: {self.data['input'].shape}")  # Debugging
                else:
                    raise KeyError(f"'input' dataset not found in {h5_path}")

            elif self.model_type == 'Planning_Model':
                # Load Planning_Model specific datasets
                required_datasets = [
                    'inputs_player_folder_chips_onehot',
                    'inputs_player_folder_codes_onehot',
                    'inputs_player_folder_flags',
                    'inputs_enemy_folder_chips_onehot',
                    'inputs_enemy_folder_codes_onehot',
                    'inputs_enemy_folder_flags',
                    'inputs_visible_chips_chips_onehot',
                    'inputs_visible_chips_codes_onehot',
                    'health',
                    'current_crosses',
                    'available_crosses',
                    'beast_flags',
                    'cross_target',
                    'target_list',
                    'reward'
                ]

                for ds in required_datasets:
                    if ds not in h5_file:
                        raise KeyError(f"'{ds}' dataset not found in {h5_path} for Planning_Model.")

                self.data['inputs_player_folder_chips_onehot'] = h5_file['inputs_player_folder_chips_onehot'][:].astype(np.float32)  # Shape: (num_samples, 30, 400)
                self.data['inputs_player_folder_codes_onehot'] = h5_file['inputs_player_folder_codes_onehot'][:].astype(np.float32)  # Shape: (num_samples, 30, 27)
                self.data['inputs_player_folder_flags'] = h5_file['inputs_player_folder_flags'][:].astype(np.float32)              # Shape: (num_samples, 30, 3)

                self.data['inputs_enemy_folder_chips_onehot'] = h5_file['inputs_enemy_folder_chips_onehot'][:].astype(np.float32)    # Shape: (num_samples, 30, 400)
                self.data['inputs_enemy_folder_codes_onehot'] = h5_file['inputs_enemy_folder_codes_onehot'][:].astype(np.float32)    # Shape: (num_samples, 30, 27)
                self.data['inputs_enemy_folder_flags'] = h5_file['inputs_enemy_folder_flags'][:].astype(np.float32)                # Shape: (num_samples, 30, 3)

                self.data['inputs_visible_chips_chips_onehot'] = h5_file['inputs_visible_chips_chips_onehot'][:].astype(np.float32)  # Shape: (num_samples, 10, 400)
                self.data['inputs_visible_chips_codes_onehot'] = h5_file['inputs_visible_chips_codes_onehot'][:].astype(np.float32)  # Shape: (num_samples, 10, 27)

                self.data['health'] = h5_file['health'][:].astype(np.float32)                        # Shape: (num_samples, 2)
                self.data['current_crosses'] = h5_file['current_crosses'][:].astype(np.float32)      # Shape: (num_samples, 52)
                self.data['available_crosses'] = h5_file['available_crosses'][:].astype(np.float32)  # Shape: (num_samples, 60)
                self.data['beast_flags'] = h5_file['beast_flags'][:].astype(np.float32)              # Shape: (num_samples, 6)

                self.data['cross_target'] = h5_file['cross_target'][:].astype(np.int64)            # Shape: (num_samples,)
                self.data['target_list'] = h5_file['target_list'][:].astype(np.int64)              # Shape: (num_samples, 5)
                self.data['reward'] = h5_file['reward'][:].astype(np.float32)                      # Shape: (num_samples,)

                print("Loaded Planning_Model specific datasets.")  # Debugging

            # Calculate the total number of samples considering image_memory and temporal_charge
            if self.model_type == 'Battle_Model':
                feature_lengths = []
                if self.include_image:
                    feature_lengths.append(len(self.data['images']))
                if self.include_position:
                    feature_lengths.append(len(self.data['positions']))
                if self.include_player_charge:
                    feature_lengths.append(len(self.data['player_charges']))
                if self.include_enemy_charge:
                    feature_lengths.append(len(self.data['enemy_charges']))
                feature_lengths.append(len(self.data['net_rewards']))
                feature_lengths.append(len(self.data['input']))

                total_reduction = self.image_memory - 1
                if self.include_temporal_charge:
                    total_reduction += self.temporal_charge - 1
                self.total_samples = min(feature_lengths) - total_reduction

                print(f"Total samples in Battle_Model dataset: {self.total_samples}")  # Debugging

            elif self.model_type == 'Planning_Model':
                # For Planning_Model, all required datasets should have the same length
                num_samples = len(self.data['cross_target'])
                print(f"Total samples in Planning_Model dataset: {num_samples}")  # Debugging
                self.total_samples = num_samples  # Assuming each sample is independent

            # Move data to GPU if available (Deferred to __getitem__ for efficiency)
            self.cuda_available = torch.cuda.is_available()

            # Ensure total_samples is non-negative
            if self.total_samples < 0:
                print(f"Warning: Calculated total_samples={self.total_samples} for {h5_path} is negative. Setting to 0.")
                self.total_samples = 0
            # Assertion to ensure __len__() is non-negative
            assert self.total_samples >= 0, f"Dataset {h5_path} has negative total_samples={self.total_samples}"

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single data point based on the index.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: A dictionary containing data tensors tailored to the model type.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")

        sample = {}

        if self.model_type == 'Battle_Model':
            base_idx = idx + self.image_memory - 1 + (self.temporal_charge -1 if self.include_temporal_charge else 0)

            if self.include_image:
                # Extract image sequence based on image_memory
                image_sequence = self.data['images'][base_idx - (self.image_memory -1) : base_idx +1]  # Shape: (image_memory, 3, 160, 240)
                if self.image_memory == 1:
                    sample['image'] = image_sequence[0].unsqueeze(0)  # Shape: (3, 1, 160, 240)
                else:
                    # Shape: (3, image_memory, 160, 240)
                    sample['image'] = image_sequence.transpose(0,1)  # From (image_memory, 3, H, W) to (3, image_memory, H, W)
                # Convert to tensor if not already
                if not isinstance(sample['image'], torch.Tensor):
                    sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
                # Move to GPU if available
                if self.cuda_available:
                    sample['image'] = sample['image'].cuda()

                # Assertion to ensure image shape is correct
                expected_image_shape = (3, self.image_memory, 160, 240) if self.image_memory > 1 else (3, 1, 160, 240)
                assert sample['image'].shape == torch.Size(expected_image_shape), f"Unexpected image shape: {sample['image'].shape}, expected: {expected_image_shape}"

            if self.include_position:
                position = self.data['positions'][base_idx]  # Shape: (4,) or (36,)
                sample['position'] = torch.tensor(position, dtype=torch.float32)
                if self.cuda_available:
                    sample['position'] = sample['position'].cuda()
                if self.position_type == 'float':
                    assert sample['position'].shape == torch.Size([4]), f"Unexpected position shape: {sample['position'].shape}"
                elif self.position_type == 'grid':
                    assert sample['position'].shape == torch.Size([36]), f"Unexpected position shape: {sample['position'].shape}"

            if self.include_player_charge:
                player_charge = self.data['player_charges'][base_idx]  # Scalar
                sample['player_charge'] = torch.tensor(player_charge, dtype=torch.float32)
                if self.cuda_available:
                    sample['player_charge'] = sample['player_charge'].cuda()
                assert sample['player_charge'].shape == torch.Size([]), f"Unexpected player_charge shape: {sample['player_charge'].shape}"

            if self.include_enemy_charge:
                enemy_charge = self.data['enemy_charges'][base_idx]  # Scalar
                sample['enemy_charge'] = torch.tensor(enemy_charge, dtype=torch.float32)
                if self.cuda_available:
                    sample['enemy_charge'] = sample['enemy_charge'].cuda()
                assert sample['enemy_charge'].shape == torch.Size([]), f"Unexpected enemy_charge shape: {sample['enemy_charge'].shape}"

            if self.include_temporal_charge:
                # Extract temporal charge sequences
                player_charge_seq = self.data['player_charges'][base_idx - (self.temporal_charge -1) : base_idx +1]  # Shape: (temporal_charge,)
                enemy_charge_seq = self.data['enemy_charges'][base_idx - (self.temporal_charge -1) : base_idx +1]  # Shape: (temporal_charge,)
                sample['player_charge_temporal'] = torch.tensor(player_charge_seq, dtype=torch.float32)
                sample['enemy_charge_temporal'] = torch.tensor(enemy_charge_seq, dtype=torch.float32)
                if self.cuda_available:
                    sample['player_charge_temporal'] = sample['player_charge_temporal'].cuda()
                    sample['enemy_charge_temporal'] = sample['enemy_charge_temporal'].cuda()
                assert sample['player_charge_temporal'].shape == torch.Size([self.temporal_charge]), f"Unexpected player_charge_temporal shape: {sample['player_charge_temporal'].shape}"
                assert sample['enemy_charge_temporal'].shape == torch.Size([self.temporal_charge]), f"Unexpected enemy_charge_temporal shape: {sample['enemy_charge_temporal'].shape}"

            # Always include net_reward
            net_reward = self.data['net_rewards'][base_idx]  # Scalar
            sample['net_reward'] = torch.tensor(net_reward, dtype=torch.float32)
            if self.cuda_available:
                sample['net_reward'] = sample['net_reward'].cuda()
            assert sample['net_reward'].shape == torch.Size([]), f"Unexpected net_reward shape: {sample['net_reward'].shape}"

            # Always include input
            input_tensor = self.data['input'][base_idx]  # Shape: (16,)
            sample['input'] = torch.tensor(input_tensor, dtype=torch.float32)
            if self.cuda_available:
                sample['input'] = sample['input'].cuda()
            assert sample['input'].shape == torch.Size([16]), f"Unexpected input shape: {sample['input'].shape}"

        elif self.model_type == 'Planning_Model':
            # For Planning_Model, each sample is independent
            base_idx = idx  # Since each sample is independent

            # Prepare the nested inputs dictionary
            inputs_nested = {
                'player_folder': {
                    'chips_onehot': torch.tensor(self.data['inputs_player_folder_chips_onehot'][base_idx], dtype=torch.float32),
                    'codes_onehot': torch.tensor(self.data['inputs_player_folder_codes_onehot'][base_idx], dtype=torch.float32),
                    'flags': torch.tensor(self.data['inputs_player_folder_flags'][base_idx], dtype=torch.float32)
                },
                'enemy_folder': {
                    'chips_onehot': torch.tensor(self.data['inputs_enemy_folder_chips_onehot'][base_idx], dtype=torch.float32),
                    'codes_onehot': torch.tensor(self.data['inputs_enemy_folder_codes_onehot'][base_idx], dtype=torch.float32),
                    'flags': torch.tensor(self.data['inputs_enemy_folder_flags'][base_idx], dtype=torch.float32)
                },
                'visible_chips': {
                    'chips_onehot': torch.tensor(self.data['inputs_visible_chips_chips_onehot'][base_idx], dtype=torch.float32),
                    'codes_onehot': torch.tensor(self.data['inputs_visible_chips_codes_onehot'][base_idx], dtype=torch.float32)
                },
                'health': torch.tensor(self.data['health'][base_idx], dtype=torch.float32),                         # Shape: (2,)
                'current_crosses': torch.tensor(self.data['current_crosses'][base_idx], dtype=torch.float32),       # Shape: (52,)
                'available_crosses': torch.tensor(self.data['available_crosses'][base_idx], dtype=torch.float32),   # Shape: (60,)
                'beast_flags': torch.tensor(self.data['beast_flags'][base_idx], dtype=torch.float32)                # Shape: (6,)
            }

            # Move to GPU if available
            if self.cuda_available:
                for folder in ['player_folder', 'enemy_folder', 'visible_chips']:
                    for key in inputs_nested[folder]:
                        inputs_nested[folder][key] = inputs_nested[folder][key].cuda()
                inputs_nested['health'] = inputs_nested['health'].cuda()
                inputs_nested['current_crosses'] = inputs_nested['current_crosses'].cuda()
                inputs_nested['available_crosses'] = inputs_nested['available_crosses'].cuda()
                inputs_nested['beast_flags'] = inputs_nested['beast_flags'].cuda()

            # Include targets and rewards
            cross_target = self.data['cross_target'][base_idx].item()    # Scalar
            target_list = self.data['target_list'][base_idx].tolist()    # List of 5 ints
            reward = self.data['reward'][base_idx].item()                # Scalar

            sample['inputs'] = inputs_nested
            sample['cross_target'] = cross_target
            sample['target_list'] = target_list
            sample['reward'] = torch.tensor(reward, dtype=torch.float32)
            if self.cuda_available:
                sample['reward'] = sample['reward'].cuda()

            # Assertions to ensure shapes are correct
            assert sample['inputs']['player_folder']['chips_onehot'].shape == torch.Size([30, 400]), f"Unexpected player_folder_chips_onehot shape: {sample['inputs']['player_folder']['chips_onehot'].shape}"
            assert sample['inputs']['player_folder']['codes_onehot'].shape == torch.Size([30, 27]), f"Unexpected player_folder_codes_onehot shape: {sample['inputs']['player_folder']['codes_onehot'].shape}"
            assert sample['inputs']['player_folder']['flags'].shape == torch.Size([30, 3]), f"Unexpected player_folder_flags shape: {sample['inputs']['player_folder']['flags'].shape}"

            assert sample['inputs']['enemy_folder']['chips_onehot'].shape == torch.Size([30, 400]), f"Unexpected enemy_folder_chips_onehot shape: {sample['inputs']['enemy_folder']['chips_onehot'].shape}"
            assert sample['inputs']['enemy_folder']['codes_onehot'].shape == torch.Size([30, 27]), f"Unexpected enemy_folder_codes_onehot shape: {sample['inputs']['enemy_folder']['codes_onehot'].shape}"
            assert sample['inputs']['enemy_folder']['flags'].shape == torch.Size([30, 3]), f"Unexpected enemy_folder_flags shape: {sample['inputs']['enemy_folder']['flags'].shape}"

            assert sample['inputs']['visible_chips']['chips_onehot'].shape == torch.Size([10, 400]), f"Unexpected visible_chips_chips_onehot shape: {sample['inputs']['visible_chips']['chips_onehot'].shape}"
            assert sample['inputs']['visible_chips']['codes_onehot'].shape == torch.Size([10, 27]), f"Unexpected visible_chips_codes_onehot shape: {sample['inputs']['visible_chips']['codes_onehot'].shape}"

            assert sample['inputs']['health'].shape == torch.Size([2]), f"Unexpected health shape: {sample['inputs']['health'].shape}"
            assert sample['inputs']['current_crosses'].shape == torch.Size([52]), f"Unexpected current_crosses shape: {sample['inputs']['current_crosses'].shape}"
            assert sample['inputs']['available_crosses'].shape == torch.Size([60]), f"Unexpected available_crosses shape: {sample['inputs']['available_crosses'].shape}"
            assert sample['inputs']['beast_flags'].shape == torch.Size([6]), f"Unexpected beast_flags shape: {sample['inputs']['beast_flags'].shape}"

            assert isinstance(sample['cross_target'], int), f"cross_target should be int, got {type(sample['cross_target'])}"
            assert isinstance(sample['target_list'], list) and len(sample['target_list']) == 5, f"target_list should be a list of 5 ints, got {sample['target_list']}"
            assert sample['reward'].shape == torch.Size([]), f"reward should be scalar tensor, got {sample['reward'].shape}"
            assert sample['reward'].dtype == torch.float32, f"reward should be float32 tensor, got {sample['reward'].dtype}"

        return sample

    def __del__(self):
        pass  # Handled by context manager
