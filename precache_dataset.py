import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from decord import VideoReader, cpu

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOURCE_DIR = "data/dataset"
TARGET_DIR = "data/dataset_cached"
RESOLUTION = (256, 256)
BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE', 
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER', 
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST', 
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP' 
]
CHUNK_SIZE = 64  # How many frames to decode at once (higher = faster, more RAM)

def main():
    src_path = Path(SOURCE_DIR)
    tgt_path = Path(TARGET_DIR)
    tgt_path.mkdir(parents=True, exist_ok=True)

    # 1. Gather all replays
    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])
    print(f"📦 Found {len(replays)} replays in {src_path}")
    print(f"🚀 converting to {tgt_path}...")

    total_frames = 0

    for replay_idx, folder in enumerate(tqdm(replays)):
        vid_path = folder / "video.mp4"
        act_path = folder / "actions.jsonl"

        if not (vid_path.exists() and act_path.exists()):
            continue

        try:
            # --- Step A: Load Actions (Dense) ---
            actions_list = []
            with open(act_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    act = json.loads(line)
                    
                    # Helpers for safe extraction
                    def get_axis(k): 
                        v = act.get(k, 0.0)
                        return float(v[0]) if isinstance(v, list) else float(v)
                    
                    def get_btn(k):
                        v = act.get(k, 0.0)
                        return float(v[0]) if isinstance(v, list) else float(v)

                    # Build 25-dim vector
                    vec = [
                        get_axis("AXIS_LEFTX"), get_axis("AXIS_LEFTY"),
                        get_axis("AXIS_RIGHTX"), get_axis("AXIS_RIGHTY")
                    ]
                    for btn in BUTTON_TOKENS:
                        vec.append(get_btn(btn))
                    
                    actions_list.append(vec)
            
            if not actions_list:
                continue

            num_actions = len(actions_list)
            actions_tensor = torch.tensor(actions_list, dtype=torch.float32) # [T, 25]

            # --- Step B: Decode Video (Batched) ---
            vr = VideoReader(str(vid_path), ctx=cpu(0), width=RESOLUTION[0], height=RESOLUTION[1])
            real_len = len(vr)
            
            # We assume actions align with frames. If mismatch, clamp to shorter.
            final_len = min(num_actions, real_len)
            
            # Pre-allocate uint8 tensor for frames [T, 3, H, W]
            # uint8 saves 4x disk space compared to float32
            frames_tensor = torch.empty((final_len, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)

            indices = list(range(final_len))
            
            # Process in chunks to save RAM
            for i in range(0, final_len, CHUNK_SIZE):
                batch_indices = indices[i : i + CHUNK_SIZE]
                
                # Decord returns [B, H, W, C] in uint8
                batch_frames = vr.get_batch(batch_indices).asnumpy()
                
                # Convert to Tensor -> Permute -> Store
                batch_torch = torch.from_numpy(batch_frames) # [B, H, W, 3]
                batch_torch = batch_torch.permute(0, 3, 1, 2) # [B, 3, H, W]
                
                frames_tensor[i : i + len(batch_indices)] = batch_torch

            # --- Step C: Slice Actions to match Video ---
            actions_tensor = actions_tensor[:final_len]

            # --- Step D: Save ---
            save_name = tgt_path / f"{folder.name}.pt"
            torch.save({
                "frames": frames_tensor,  # uint8, [T, 3, 256, 256]
                "actions": actions_tensor # float32, [T, 25]
            }, save_name)

            total_frames += final_len

        except Exception as e:
            print(f"❌ Error processing {folder.name}: {e}")

    print(f"✅ Pre-caching complete!")
    print(f"   Saved {len(replays)} files.")
    print(f"   Total Frames: {total_frames:,}")
    print(f"   Output Location: {tgt_path}")

if __name__ == "__main__":
    main()