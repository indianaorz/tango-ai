import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from decord import VideoReader, cpu

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOURCE_DIR = "data/dataset"
TARGET_DIR = "data/dataset_cached"
RESOLUTION = (256, 256) # Target Model Input Size (Square)
NATIVE_RES = (160, 240) # GBA Native Size (H, W)

BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE', 
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER', 
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST', 
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP' 
]
CHUNK_SIZE = 64 

def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    1. Downscales video to Native GBA (240x160) using Nearest Neighbor.
    2. Centers the 240x160 image on a black canvas of target_h x target_w.
    """
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES
    
    # 1. Force Resize to Native GBA Resolution
    # This handles 480x320 -> 240x160 perfectly (2x -> 1x)
    # It also handles 240x160 -> 240x160 (No-op)
    frames_native = F.interpolate(
        frames.float(), 
        size=(native_h, native_w), 
        mode='nearest'
    )
    
    # 2. Create Black Canvas
    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames.dtype)
    
    # 3. Paste Native Image in Center
    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2
    
    # Ensure we don't crash if target < native (unlikely for 256 vs 240)
    paste_h = min(native_h, target_h)
    paste_w = min(native_w, target_w)
    
    canvas[:, :, y_off:y_off+paste_h, x_off:x_off+paste_w] = frames_native[:, :, :paste_h, :paste_w]
    
    return canvas

def main():
    src_path = Path(SOURCE_DIR)
    tgt_path = Path(TARGET_DIR)
    tgt_path.mkdir(parents=True, exist_ok=True)

    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])
    print(f"📦 Found {len(replays)} replays")
    print(f"🚀 Processing: Input(Any) -> Native(240x160) -> Padded({RESOLUTION})")

    total_frames = 0

    for replay_idx, folder in enumerate(tqdm(replays)):
        vid_path = folder / "video.mp4"
        act_path = folder / "actions.jsonl"

        if not (vid_path.exists() and act_path.exists()):
            continue

        try:
            # --- Step A: Load Actions (Digitalized) ---
            actions_list = []
            with open(act_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    act = json.loads(line)
                    
                    def get_axis(k): 
                        v = act.get(k, 0.0)
                        return float(v[0]) if isinstance(v, list) else float(v)
                    
                    def get_btn(k):
                        v = act.get(k, 0.0)
                        val = float(v[0]) if isinstance(v, list) else float(v)
                        return 1.0 if val > 0.5 else 0.0

                    vec = [
                        get_axis("AXIS_LEFTX"), get_axis("AXIS_LEFTY"),
                        get_axis("AXIS_RIGHTX"), get_axis("AXIS_RIGHTY")
                    ]
                    for btn in BUTTON_TOKENS:
                        vec.append(get_btn(btn))
                    
                    actions_list.append(vec)
            
            if not actions_list: continue

            num_actions = len(actions_list)
            actions_tensor = torch.tensor(actions_list, dtype=torch.float32)

            # --- Step B: Decode Video ---
            vr = VideoReader(str(vid_path), ctx=cpu(0))
            real_len = len(vr)
            final_len = min(num_actions, real_len)
            
            frames_tensor = torch.empty((final_len, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)
            indices = list(range(final_len))
            
            for i in range(0, final_len, CHUNK_SIZE):
                batch_indices = indices[i : i + CHUNK_SIZE]
                batch_frames = vr.get_batch(batch_indices).asnumpy()
                
                # [B, H, W, 3] -> [B, 3, H, W]
                batch_torch = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)
                
                # --- APPLY GBA NORMALIZATION ---
                batch_padded = process_batch_gba(batch_torch, RESOLUTION[1], RESOLUTION[0])
                
                frames_tensor[i : i + len(batch_indices)] = batch_padded.to(torch.uint8)

            actions_tensor = actions_tensor[:final_len]

            save_name = tgt_path / f"{folder.name}.pt"
            torch.save({
                "frames": frames_tensor,
                "actions": actions_tensor
            }, save_name)

            total_frames += final_len

        except Exception as e:
            print(f"❌ Error processing {folder.name}: {e}")

    print(f"✅ Pre-caching complete! Total Frames: {total_frames:,}")

if __name__ == "__main__":
    main()