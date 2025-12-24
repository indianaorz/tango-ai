import torch
from torch.utils.data import Dataset
import json
import os

class NitrogenDataset(Dataset):
    def __init__(self, cache_dir, rl_weights_path="data/nitrogen_rl/frame_weights.jsonl", action_horizon=18):
        self.cache_dir = cache_dir
        self.action_horizon = action_horizon
        
        # 1. Load Cache Files
        self.pt_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        
        # 2. Load RL Weights (JSONL Format)
        self.weights = {}
        if os.path.exists(rl_weights_path):
            print(f"⚖️ Loading RL Weights from {rl_weights_path}...")
            try:
                with open(rl_weights_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # data = {"key": "replay/frame", "val": 3.0}
                            self.weights[data['key']] = data['val']
                print(f"   ✅ Loaded weights for {len(self.weights)} frames.")
            except Exception as e:
                print(f"   ❌ Error loading weights: {e}")
        else:
            print("ℹ️ No RL weights found. Defaulting to pure Behavior Cloning (1.0).")

        # 3. Build Index
        self.indices = []
        print("   Indexing cache files...")
        for fname in self.pt_files:
            # Quick-load to get length (cpu map avoids VRAM spikes)
            path = os.path.join(cache_dir, fname)
            try:
                # We assume metadata is cheaper to load or we just read the shape
                # If loading is slow, you can cache this index too.
                data = torch.load(path, map_location='cpu')
                num_frames = data['actions'].shape[0]
                
                # We need enough future frames for the horizon
                valid_count = num_frames - self.action_horizon
                replay_name = fname.replace(".pt", "")
                
                for i in range(valid_count):
                    self.indices.append((fname, replay_name, i))
            except:
                print(f"   ⚠️ Skipping corrupt file: {fname}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fname, replay_name, i = self.indices[idx]
        path = os.path.join(self.cache_dir, fname)
        
        # Load File
        data = torch.load(path, map_location='cpu')
        
        # A. Vision (Normalize uint8 [0-255] -> float [-1, 1])
        frame = data['frames'][i]
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0 * 2.0 - 1.0
            
        # B. Action Targets (Sequence)
        actions = data['actions'][i : i + self.action_horizon]
        
        # C. RL Weight
        # Look up: "replay_name/frame_index"
        key = f"{replay_name}/{i}"
        w = self.weights.get(key, 1.0) # Default to 1.0
        
        return {
            "pixel_values": frame,
            "labels": actions,
            "weight": torch.tensor(w, dtype=torch.float32)
        }