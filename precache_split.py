# precache_split.py
from __future__ import annotations

import argparse
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

# Two separate cache targets
PLANNING_DIR = "data/planning_cache"
BATTLE_DIR = "data/battle_cache"

RESOLUTION = (256, 256)
NATIVE_RES = (160, 240)
CHUNK_SIZE = 64

# Future horizon to "bake" into the tensor
ACTION_HORIZON = 18

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _scalar(v, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        return float(v[0]) if v else float(default)
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size > 0 else float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _norm_axis(v: float) -> float:
    fv = float(v)
    if abs(fv) > 1.5:
        fv = fv / 32767.0
    return max(-1.0, min(1.0, fv))


def _btn01(v: float) -> float:
    return 1.0 if float(v) > 0.5 else 0.0


def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES
    frames_native = F.interpolate(frames.float(), size=(native_h, native_w), mode="nearest")
    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames.dtype)
    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2
    paste_h, paste_w = min(native_h, target_h), min(native_w, target_w)
    canvas[:, :, y_off : y_off + paste_h, x_off : x_off + paste_w] = frames_native[:, :, :paste_h, :paste_w]
    return canvas


def save_subset(
    indices: list[int],
    video_frame_indices: list[int],
    raw_actions_tensor: torch.Tensor,
    vr: VideoReader,
    save_path: Path,
    desc: str,
) -> int:
    """
    Saves a subset of frames defined by 'indices' to 'save_path'.
    Bakes the horizon from the global raw_actions_tensor.
    """
    if not indices:
        return 0

    total_time = raw_actions_tensor.shape[0]

    # Filter out indices that go beyond video length
    real_vid_len = len(vr)

    valid_pairs: list[tuple[int, int]] = []
    for i, vid_idx in enumerate(video_frame_indices):
        if vid_idx < real_vid_len:
            valid_pairs.append((indices[i], vid_idx))

    if not valid_pairs:
        return 0

    final_action_indices = [p[0] for p in valid_pairs]
    final_vid_indices = [p[1] for p in valid_pairs]
    final_len = len(valid_pairs)

    # 1) Build baked action windows
    kept_action_windows: list[torch.Tensor] = []
    for t in final_action_indices:
        end_idx = min(t + ACTION_HORIZON, total_time)
        window = raw_actions_tensor[t:end_idx]

        # Pad if needed
        if window.shape[0] < ACTION_HORIZON:
            pad_amt = ACTION_HORIZON - window.shape[0]
            padding = torch.zeros((pad_amt, ACTION_DIM), dtype=torch.float32)
            window = torch.cat([window, padding], dim=0)

        kept_action_windows.append(window)

    final_actions = torch.stack(kept_action_windows)

    # 2) Extract video frames
    frames_tensor = torch.empty((final_len, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)

    for i in range(0, final_len, CHUNK_SIZE):
        chunk_range = range(i, min(i + CHUNK_SIZE, final_len))
        batch_vid_indices = [final_vid_indices[k] for k in chunk_range]

        try:
            batch_frames = vr.get_batch(batch_vid_indices).asnumpy()
            batch_torch = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)
            batch_padded = process_batch_gba(batch_torch, RESOLUTION[1], RESOLUTION[0])
            frames_tensor[i : i + len(chunk_range)] = batch_padded.to(torch.uint8)
        except Exception as e:
            print(f"⚠️ Read error at chunk {i} ({desc}): {e}")
            # Leave zeros for this chunk; continue
            continue

    torch.save({"frames": frames_tensor, "actions": final_actions}, save_path)
    return final_len


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Split and cache dataset into planning/battle tensors.")
    parser.add_argument(
        "--mode",
        choices=["planning", "battle", "both"],
        default="both",
        help="Which cache to write: planning, battle, or both (default: both).",
    )
    args = parser.parse_args()

    plan_enabled = args.mode in ("planning", "both")
    battle_enabled = args.mode in ("battle", "both")

    src_path = Path(SOURCE_DIR)

    plan_path = Path(PLANNING_DIR)
    battle_path = Path(BATTLE_DIR)

    if plan_enabled:
        plan_path.mkdir(parents=True, exist_ok=True)
    if battle_enabled:
        battle_path.mkdir(parents=True, exist_ok=True)

    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])
    print(f"📦 Splitting Dataset (Strict Window Logic) | mode={args.mode}")
    if plan_enabled:
        print("   Planning: inside_window=True AND cust_gauge=0")
    if battle_enabled:
        print("   Battle:   cust_gauge > 0")

    stats = {"plan": 0, "battle": 0, "skipped": 0}

    for folder in tqdm(replays, desc="Splitting"):
        vid_path, act_path = folder / "video.mp4", folder / "actions.jsonl"
        if not (vid_path.exists() and act_path.exists()):
            continue

        name = folder.name
        plan_save = plan_path / f"{name}.pt"
        battle_save = battle_path / f"{name}.pt"

        # Determine which targets we need to generate
        need_plan = plan_enabled and (not plan_save.exists())
        need_battle = battle_enabled and (not battle_save.exists())

        # If nothing to do for this replay, skip
        if not need_plan and not need_battle:
            stats["skipped"] += 1
            continue

        try:
            raw_actions: list[list[float]] = []

            # (action_idx, video_frame_idx) tuples
            plan_map: list[tuple[int, int]] | None = [] if need_plan else None
            battle_map: list[tuple[int, int]] | None = [] if need_battle else None

            action_idx = 0

            with open(act_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    # 1) Parse Action
                    vec = [
                        _norm_axis(_scalar(row.get("AXIS_LEFTX"))),
                        _norm_axis(_scalar(row.get("AXIS_LEFTY"))),
                        _norm_axis(_scalar(row.get("AXIS_RIGHTX"))),
                        _norm_axis(_scalar(row.get("AXIS_RIGHTY"))),
                    ]
                    for btn in BUTTON_TOKENS:
                        vec.append(_btn01(_scalar(row.get(btn))))
                    raw_actions.append(vec)

                    # 2) Sync
                    vid_idx = int(row.get("frame_idx", action_idx))

                    # 3) Classify (STRICT LOGIC)
                    inside_window = bool(row.get("inside_window", False))
                    cust_gauge = int(row.get("cust_gauge", 0))

                    if plan_map is not None and inside_window and cust_gauge == 0:
                        plan_map.append((action_idx, vid_idx))
                    elif battle_map is not None and cust_gauge > 0:
                        # Note: We implicitly drop frames where
                        # inside_window=False AND cust_gauge=0 (Idle/Wait)
                        battle_map.append((action_idx, vid_idx))

                    action_idx += 1

            if not raw_actions:
                continue

            raw_actions_tensor = torch.tensor(raw_actions, dtype=torch.float32)
            vr = VideoReader(str(vid_path), ctx=cpu(0))

            if plan_map:
                p_act = [x[0] for x in plan_map]
                p_vid = [x[1] for x in plan_map]
                count = save_subset(p_act, p_vid, raw_actions_tensor, vr, plan_save, "Plan")
                stats["plan"] += count

            if battle_map:
                b_act = [x[0] for x in battle_map]
                b_vid = [x[1] for x in battle_map]
                count = save_subset(b_act, b_vid, raw_actions_tensor, vr, battle_save, "Battle")
                stats["battle"] += count

        except Exception as e:
            print(f"❌ Error {folder.name}: {e}")

    print("✅ Complete.")
    if plan_enabled:
        print(f"   Planning Frames: {stats['plan']:,}")
    if battle_enabled:
        print(f"   Battle Frames:   {stats['battle']:,}")
    print(f"   Skipped Replays:  {stats['skipped']:,}")


if __name__ == "__main__":
    main()
