# precache_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

from action_schema import BUTTON_TOKENS, ACTION_DIM


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOURCE_DIR = "data/dataset"
TARGET_DIR = "data/dataset_cached"

RESOLUTION = (256, 256)   # (W,H)
NATIVE_RES = (160, 240)   # (H,W) GBA native (your pipeline convention)
CHUNK_SIZE = 64


# -----------------------------------------------------------------------------
# Helpers: robust scalar extraction + normalization
# -----------------------------------------------------------------------------
def _scalar(v, default: float = 0.0) -> float:
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
    fv = float(v)
    # Some pipelines emit int16-ish joystick units; normalize if so.
    if abs(fv) > 1.5:
        fv = fv / 32767.0
    if fv > 1.0:
        fv = 1.0
    elif fv < -1.0:
        fv = -1.0
    return fv


def _btn01(v: float) -> float:
    return 1.0 if float(v) > 0.5 else 0.0


# -----------------------------------------------------------------------------
# Video frame normalization for GBA
# -----------------------------------------------------------------------------
def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    frames: [B,3,H,W] uint8/float
    Returns: [B,3,target_h,target_w] uint8/float
    """
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES

    # Nearest keeps pixel edges stable for GBA-style frames
    frames_native = F.interpolate(frames.float(), size=(native_h, native_w), mode="nearest")

    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames.dtype)

    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2

    paste_h = min(native_h, target_h)
    paste_w = min(native_w, target_w)

    canvas[:, :, y_off:y_off + paste_h, x_off:x_off + paste_w] = frames_native[:, :, :paste_h, :paste_w]
    return canvas


def _validate_vec_len(vec: list[float]) -> None:
    if len(vec) != ACTION_DIM:
        raise ValueError(
            f"Action vector length mismatch: got {len(vec)}, expected {ACTION_DIM} "
            f"(4 axes + {len(BUTTON_TOKENS)} buttons). Check BUTTON_TOKENS order/source."
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    src_path = Path(SOURCE_DIR)
    tgt_path = Path(TARGET_DIR)
    tgt_path.mkdir(parents=True, exist_ok=True)

    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])
    print(f"📦 Found {len(replays)} replays")
    print(f"🚀 Processing: Input(Any) -> Native(240x160) -> Padded({RESOLUTION})")
    print(f"🎛️ Button token order (len={len(BUTTON_TOKENS)}): {BUTTON_TOKENS}")
    print(f"🧱 Expected cache action dim: {ACTION_DIM}")

    total_frames = 0

    for folder in tqdm(replays, desc="Precache"):
        vid_path = folder / "video.mp4"
        act_path = folder / "actions.jsonl"

        if not (vid_path.exists() and act_path.exists()):
            continue

        try:
            # --- Step A: Load Actions (cache-space: [axes(4), buttons(len=21)]) ---
            actions_list: list[list[float]] = []
            with open(act_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    act = json.loads(line)

                    ax_lx = _norm_axis(_scalar(act.get("AXIS_LEFTX", 0.0)))
                    ax_ly = _norm_axis(_scalar(act.get("AXIS_LEFTY", 0.0)))
                    ax_rx = _norm_axis(_scalar(act.get("AXIS_RIGHTX", 0.0)))
                    ax_ry = _norm_axis(_scalar(act.get("AXIS_RIGHTY", 0.0)))

                    vec = [ax_lx, ax_ly, ax_rx, ax_ry]

                    # BN6/GBA is digital for our purposes: keep ALL buttons strictly 0/1.
                    for btn in BUTTON_TOKENS:
                        vec.append(_btn01(_scalar(act.get(btn, 0.0))))

                    _validate_vec_len(vec)
                    actions_list.append(vec)

            if not actions_list:
                continue

            actions_tensor = torch.tensor(actions_list, dtype=torch.float32)

            # --- Step B: Decode Video ---
            vr = VideoReader(str(vid_path), ctx=cpu(0))
            real_len = len(vr)
            final_len = min(len(actions_list), real_len)
            if final_len <= 0:
                continue

            frames_tensor = torch.empty((final_len, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)

            indices = list(range(final_len))
            for i in range(0, final_len, CHUNK_SIZE):
                batch_indices = indices[i:i + CHUNK_SIZE]
                batch_frames = vr.get_batch(batch_indices).asnumpy()  # [B,H,W,3]

                batch_torch = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)  # [B,3,H,W]
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
