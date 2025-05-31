# preloaded_h5_game_dataset.py
import h5py
import torch
import threading
import queue
import gc
from tqdm import tqdm
from torch.utils.data import Dataset

class PreloadedH5GameDataset(Dataset):
    def __init__(self, h5_path, device='cpu', batch_size=1000, prefetch_batches=2, image_memory=1):
        """
        Initializes the PreloadedH5GameDataset.

        Args:
            h5_path (str): Path to the HDF5 file.
            device (torch.device or str): Device to load data onto ('cpu' or 'cuda').
            batch_size (int): Number of samples per batch during preloading.
            prefetch_batches (int): Number of batches to prefetch.
            image_memory (int): Depth dimension for image tensors.
        """
        super(PreloadedH5GameDataset, self).__init__()
        self.h5_path = h5_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.image_memory = image_memory  # Depth dimension for images

        # Open the HDF5 file
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.images = self.h5_file['images']  # Shape: (num_samples, 3, D, 160, 240)
        self.inputs = self.h5_file['inputs']  # Shape: (num_samples, 16)
        self.player_healths = self.h5_file['player_healths']  # Shape: (num_samples,)
        self.enemy_healths = self.h5_file['enemy_healths']    # Shape: (num_samples,)
        self.player_grids = self.h5_file['player_grids']      # Shape: (num_samples, 6, 3)
        self.enemy_grids = self.h5_file['enemy_grids']        # Shape: (num_samples, 6, 3)
        self.inside_windows = self.h5_file['inside_windows']  # Shape: (num_samples,)
        self.net_rewards = self.h5_file['net_rewards']        # Shape: (num_samples,)

        # Read min and max net_rewards from file attributes
        self.net_reward_min = self.h5_file.attrs.get('net_reward_min', float('inf'))
        self.net_reward_max = self.h5_file.attrs.get('net_reward_max', float('-inf'))

        # Preload data
        self.preload_data()

    def preload_data(self):
        print("Preloading data into GPU...")
        num_samples = self.images.shape[0]  # Directly access the number of samples

        # Preallocate tensors on GPU for all datasets
        try:
            images_gpu = torch.empty((num_samples, 3, self.image_memory, 160, 240), dtype=torch.float32, device=self.device)
            inputs_gpu = torch.empty((num_samples, 16), dtype=torch.float32, device=self.device)
            player_healths_gpu = torch.empty(num_samples, dtype=torch.float32, device=self.device)
            enemy_healths_gpu = torch.empty(num_samples, dtype=torch.float32, device=self.device)
            player_grids_gpu = torch.empty((num_samples, 6, 3), dtype=torch.float32, device=self.device)
            enemy_grids_gpu = torch.empty((num_samples, 6, 3), dtype=torch.float32, device=self.device)
            inside_windows_gpu = torch.empty(num_samples, dtype=torch.float32, device=self.device)
            net_rewards_gpu = torch.empty(num_samples, dtype=torch.float32, device=self.device)
        except RuntimeError as e:
            print(f"Error allocating GPU memory: {e}")
            raise

        # Create a thread-safe queue with a maximum size based on prefetch_batches
        prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)

        def loader():
            try:
                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)

                    # Load data from HDF5
                    images_np = self.images[start_idx:end_idx]  # (batch_size, 3, D, 160, 240)
                    inputs_np = self.inputs[start_idx:end_idx]  # (batch_size, 16)
                    player_healths_np = self.player_healths[start_idx:end_idx]  # (batch_size,)
                    enemy_healths_np = self.enemy_healths[start_idx:end_idx]    # (batch_size,)
                    player_grids_np = self.player_grids[start_idx:end_idx]      # (batch_size, 6, 3)
                    enemy_grids_np = self.enemy_grids[start_idx:end_idx]        # (batch_size, 6, 3)
                    inside_windows_np = self.inside_windows[start_idx:end_idx]  # (batch_size,)
                    net_rewards_np = self.net_rewards[start_idx:end_idx]        # (batch_size,)

                    # Convert to torch tensors
                    images_cpu = torch.from_numpy(images_np).float() / 255.0  # Normalize images
                    inputs_cpu = torch.from_numpy(inputs_np).float()
                    player_healths_cpu = torch.from_numpy(player_healths_np).float()
                    enemy_healths_cpu = torch.from_numpy(enemy_healths_np).float()
                    player_grids_cpu = torch.from_numpy(player_grids_np).float()
                    enemy_grids_cpu = torch.from_numpy(enemy_grids_np).float()
                    inside_windows_cpu = torch.from_numpy(inside_windows_np).float()
                    net_rewards_cpu = torch.from_numpy(net_rewards_np).float()

                    # Put the batch into the queue
                    prefetch_queue.put((start_idx, images_cpu, inputs_cpu,
                                        player_healths_cpu, enemy_healths_cpu,
                                        player_grids_cpu, enemy_grids_cpu,
                                        inside_windows_cpu, net_rewards_cpu))
            finally:
                # Signal that loading is done
                prefetch_queue.put(None)

        # Start the loader thread
        loader_thread = threading.Thread(target=loader, daemon=True)
        loader_thread.start()

        # Process the batches
        with tqdm(total=num_samples, desc="Preloading Batches") as pbar:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break  # No more data

                (start_idx, images_cpu, inputs_cpu,
                player_healths_cpu, enemy_healths_cpu,
                player_grids_cpu, enemy_grids_cpu,
                inside_windows_cpu, net_rewards_cpu) = batch

                batch_size_actual = images_cpu.size(0)

                # Asynchronously copy to GPU
                try:
                    images_gpu[start_idx:start_idx+batch_size_actual].copy_(images_cpu, non_blocking=True)
                    inputs_gpu[start_idx:start_idx+batch_size_actual].copy_(inputs_cpu, non_blocking=True)
                    player_healths_gpu[start_idx:start_idx+batch_size_actual].copy_(player_healths_cpu, non_blocking=True)
                    enemy_healths_gpu[start_idx:start_idx+batch_size_actual].copy_(enemy_healths_cpu, non_blocking=True)
                    player_grids_gpu[start_idx:start_idx+batch_size_actual].copy_(player_grids_cpu, non_blocking=True)
                    enemy_grids_gpu[start_idx:start_idx+batch_size_actual].copy_(enemy_grids_cpu, non_blocking=True)
                    inside_windows_gpu[start_idx:start_idx+batch_size_actual].copy_(inside_windows_cpu, non_blocking=True)
                    net_rewards_gpu[start_idx:start_idx+batch_size_actual].copy_(net_rewards_cpu, non_blocking=True)
                except RuntimeError as e:
                    print(f"Error copying data to GPU: {e}")
                    raise

                pbar.update(batch_size_actual)

        # Assign preloaded tensors to class attributes
        self.images_gpu = images_gpu
        self.inputs_gpu = inputs_gpu
        self.player_healths_gpu = player_healths_gpu
        self.enemy_healths_gpu = enemy_healths_gpu
        self.player_grids_gpu = player_grids_gpu
        self.enemy_grids_gpu = enemy_grids_gpu
        self.inside_windows_gpu = inside_windows_gpu
        self.net_rewards_gpu = net_rewards_gpu

        # Delete the HDF5 file reference to free up CPU memory
        del self.h5_file
        gc.collect()
        print("Data preloading completed.")


    def __len__(self):
        return self.images_gpu.size(0)

    def __getitem__(self, idx):
        try:
            # Retrieve preloaded data
            image = self.images_gpu[idx]  # Shape: (3, D, 160, 240)
            input_tensor = self.inputs_gpu[idx]  # Shape: (16,)
            player_health = self.player_healths_gpu[idx]  # Scalar
            enemy_health = self.enemy_healths_gpu[idx]    # Scalar
            player_grid = self.player_grids_gpu[idx]      # Shape: (6, 3)
            enemy_grid = self.enemy_grids_gpu[idx]        # Shape: (6, 3)
            inside_window = self.inside_windows_gpu[idx]  # Scalar
            net_reward = self.net_rewards_gpu[idx]        # Scalar

            # If image_memory >1, ensure image shape is correct
            # Currently, image_memory is handled during preloading by adjusting the depth dimension

            return {
                'image': image,  # Shape: (3, D, 160, 240)
                'input': input_tensor,  # Shape: (16,)
                'player_health': player_health,  # Scalar
                'enemy_health': enemy_health,    # Scalar
                'player_grid': player_grid,      # Shape: (6, 3)
                'enemy_grid': enemy_grid,        # Shape: (6, 3)
                'inside_window': inside_window,  # Scalar
                'net_reward': net_reward         # Scalar
            }

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
