# precache_dataset.py
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

RESOLUTION = (256, 256)   # Target Model Input Size (W, H) usage below keeps consistency
NATIVE_RES = (160, 240)   # GBA Native Size (H, W)

CHUNK_SIZE = 64

# -----------------------------------------------------------------------------
# IMPORTANT: Button order must match Nitrogen's token order used at inference time
# -----------------------------------------------------------------------------
try:
    # This is the canonical order used by Nitrogen inference (TOKEN_SET = BUTTON_ACTION_TOKENS)
    from nitrogen.shared import BUTTON_ACTION_TOKENS as BUTTON_TOKENS
except Exception:
    # Fallback: keeps script usable standalone, but you should prefer the import above.
    # If you hit this fallback, you're at risk of order mismatches.
    BUTTON_TOKENS = [
        'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
        'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER',
        'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST',
        'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP'
    ]
    print("⚠️ WARNING: Could not import nitrogen.shared.BUTTON_ACTION_TOKENS. "
          "Using fallback BUTTON_TOKENS list; verify ordering matches your model.")

# -----------------------------------------------------------------------------
# Helpers: robust scalar extraction + normalization
# -----------------------------------------------------------------------------
def _scalar(v, default: float = 0.0) -> float:
    """
    Pull a scalar float from values that might be:
      - float/int
      - list/tuple with one element (common in JSONL)
      - numpy scalar/array
    """
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        if not v:
            return float(default)
        v = v[0]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return float(default)
        v = v.reshape(-1)[0]
    try:
        return float(v)
    except Exception:
        return float(default)

def _norm_axis(v: float) -> float:
    """
    Normalize axis values to [-1, 1].

    Supports:
      - already-normalized floats in [-1,1]
      - Nitrogen debug/inference style int range [-32767, 32767]
    """
    fv = float(v)
    if abs(fv) > 1.5:  # heuristic: treat as int-range joystick
        fv = fv / 32767.0
    # clamp for safety
    if fv > 1.0:
        fv = 1.0
    elif fv < -1.0:
        fv = -1.0
    return fv

def _norm_trigger(v: float) -> float:
    """
    Normalize triggers to [0, 1].

    Supports:
      - already-normalized floats in [0,1]
      - Nitrogen inference style 0..255
    """
    fv = float(v)
    if fv > 1.5:  # heuristic: treat as 0..255
        fv = fv / 255.0
    # clamp
    if fv < 0.0:
        fv = 0.0
    elif fv > 1.0:
        fv = 1.0
    return fv

def _btn01(v: float) -> float:
    return 1.0 if float(v) > 0.5 else 0.0

# -----------------------------------------------------------------------------
# Video frame normalization for GBA
# -----------------------------------------------------------------------------
def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    1) Resizes video frames to native GBA (160x240) using nearest neighbor.
    2) Centers the native image on a black canvas of target_h x target_w.
    """
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES

    # 1) Force Resize to Native GBA Resolution
    frames_native = F.interpolate(frames.float(), size=(native_h, native_w), mode="nearest")

    # 2) Create Black Canvas
    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames.dtype)

    # 3) Paste Native Image in Center
    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2

    paste_h = min(native_h, target_h)
    paste_w = min(native_w, target_w)

    canvas[:, :, y_off:y_off + paste_h, x_off:x_off + paste_w] = frames_native[:, :, :paste_h, :paste_w]
    return canvas

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    src_path = Path(SOURCE_DIR)
    tgt_path = Path(TARGET_DIR)
    tgt_path.mkdir(parents=True, exist_ok=True)

    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])
    print(f"📦 Found {len(replays)} replays")
    print(f"🚀 Processing: Input(Any) -> Native(240x160) -> Padded({RESOLUTION})")
    print(f"🎛️ Button token order (len={len(BUTTON_TOKENS)}): {BUTTON_TOKENS}")

    total_frames = 0

    for folder in tqdm(replays):
        vid_path = folder / "video.mp4"
        act_path = folder / "actions.jsonl"

        if not (vid_path.exists() and act_path.exists()):
            continue

        try:
            # --- Step A: Load Actions ---
            actions_list = []
            with open(act_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    act = json.loads(line)

                    # Axes first (match your existing layout: 4 floats)
                    ax_lx = _norm_axis(_scalar(act.get("AXIS_LEFTX", 0.0)))
                    ax_ly = _norm_axis(_scalar(act.get("AXIS_LEFTY", 0.0)))
                    ax_rx = _norm_axis(_scalar(act.get("AXIS_RIGHTX", 0.0)))
                    ax_ry = _norm_axis(_scalar(act.get("AXIS_RIGHTY", 0.0)))

                    vec = [ax_lx, ax_ly, ax_rx, ax_ry]

                    # Buttons next, in canonical Nitrogen order
                    for btn in BUTTON_TOKENS:
                        raw = _scalar(act.get(btn, 0.0))
                        if "TRIGGER" in btn:
                            vec.append(_norm_trigger(raw))
                        else:
                            vec.append(_btn01(raw))

                    actions_list.append(vec)

            if not actions_list:
                continue

            actions_tensor = torch.tensor(actions_list, dtype=torch.float32)

            # --- Step B: Decode Video ---
            vr = VideoReader(str(vid_path), ctx=cpu(0))
            real_len = len(vr)
            final_len = min(len(actions_list), real_len)

            # frames tensor layout: [T, 3, H, W]
            frames_tensor = torch.empty((final_len, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)

            indices = list(range(final_len))
            for i in range(0, final_len, CHUNK_SIZE):
                batch_indices = indices[i:i + CHUNK_SIZE]
                batch_frames = vr.get_batch(batch_indices).asnumpy()  # [B, H, W, 3]

                batch_torch = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)  # [B, 3, H, W]
                batch_padded = process_batch_gba(batch_torch, RESOLUTION[1], RESOLUTION[0])

                frames_tensor[i:i + len(batch_indices)] = batch_padded.to(torch.uint8)

            actions_tensor = actions_tensor[:final_len]

            save_name = tgt_path / f"{folder.name}.pt"
            torch.save({"frames": frames_tensor, "actions": actions_tensor}, save_name)

            total_frames += final_len

        except Exception as e:
            print(f"❌ Error processing {folder.name}: {e}")

    print(f"✅ Pre-caching complete! Total Frames: {total_frames:,}")

if __name__ == "__main__":
    main()
