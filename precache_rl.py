# precache_split_rl.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Input: produced by scripts/create_rl_dataset.py
RL_DIR = "data/nitrogen_rl"
RL_EVENTS_FILE = "rl_events.jsonl"

# Output: cached RL segments (one file per event)
OUTPUT_DIR = "data/nitrogen_rl_cache"

RESOLUTION = (256, 256)
NATIVE_RES = (160, 240)

# Bake future horizon into each timestep
ACTION_HORIZON = 18


# -----------------------------------------------------------------------------
# Helpers: action parsing (same semantics as precache_split.py)
# -----------------------------------------------------------------------------
def _scalar(v: Any, default: float = 0.0) -> float:
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
    # Some logs store raw int16-like values; normalize if it looks too large
    if abs(fv) > 1.5:
        fv = fv / 32767.0
    return max(-1.0, min(1.0, fv))


def _btn01(v: float) -> float:
    return 1.0 if float(v) > 0.5 else 0.0


def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize to GBA-native, then center-pad into (target_h, target_w).
    Input: (B, 3, H, W) uint8/float
    Output: (B, 3, target_h, target_w) float (caller can cast)
    """
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES
    frames_native = F.interpolate(frames.float(), size=(native_h, native_w), mode="nearest")
    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames_native.dtype)
    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2
    paste_h, paste_w = min(native_h, target_h), min(native_w, target_w)
    canvas[:, :, y_off : y_off + paste_h, x_off : x_off + paste_w] = frames_native[:, :, :paste_h, :paste_w]
    return canvas


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    if not s:
        s = "event"
    return s[:max_len]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _build_action_and_vid_maps(actions_rows: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[int]]:
    """
    Returns:
      - raw_actions_tensor: (T, ACTION_DIM) float32
      - vid_idx_map: length T list of int (video frame index per action timestep)
    """
    raw_actions: List[List[float]] = []
    vid_idx_map: List[int] = []

    for action_idx, row in enumerate(actions_rows):
        vec = [
            _norm_axis(_scalar(row.get("AXIS_LEFTX"))),
            _norm_axis(_scalar(row.get("AXIS_LEFTY"))),
            _norm_axis(_scalar(row.get("AXIS_RIGHTX"))),
            _norm_axis(_scalar(row.get("AXIS_RIGHTY"))),
        ]
        for btn in BUTTON_TOKENS:
            vec.append(_btn01(_scalar(row.get(btn))))

        if len(vec) != ACTION_DIM:
            raise ValueError(f"Action dim mismatch at idx={action_idx}: got {len(vec)} expected {ACTION_DIM}")

        raw_actions.append(vec)

        # Video frame index used for decoding (matches existing precache behavior)
        vid_idx = int(row.get("frame_idx", action_idx))
        vid_idx_map.append(vid_idx)

    raw_actions_tensor = torch.tensor(raw_actions, dtype=torch.float32)
    return raw_actions_tensor, vid_idx_map


def _bake_action_windows(indices: List[int], raw_actions_tensor: torch.Tensor) -> torch.Tensor:
    """
    For each t in indices, returns a window raw_actions[t : t+ACTION_HORIZON], padded with zeros.
    Output: (len(indices), ACTION_HORIZON, ACTION_DIM) float32
    """
    total_time = int(raw_actions_tensor.shape[0])
    windows: List[torch.Tensor] = []

    for t in indices:
        t = int(t)
        if t < 0 or t >= total_time:
            windows.append(torch.zeros((ACTION_HORIZON, ACTION_DIM), dtype=torch.float32))
            continue

        end_idx = min(t + ACTION_HORIZON, total_time)
        w = raw_actions_tensor[t:end_idx]  # (k, ACTION_DIM)

        if w.shape[0] < ACTION_HORIZON:
            pad_amt = ACTION_HORIZON - int(w.shape[0])
            w = torch.cat([w, torch.zeros((pad_amt, ACTION_DIM), dtype=torch.float32)], dim=0)

        windows.append(w)

    return torch.stack(windows, dim=0)


def _decode_unique_video_frames(
    vr: VideoReader,
    vid_list_sorted_unique: List[int],
    *,
    decode_chunk: int,
) -> torch.Tensor:
    """
    Decode all needed frames ONCE for this replay.

    Args:
      vid_list_sorted_unique: sorted unique video frame indices
      decode_chunk: batch size for vr.get_batch()

    Returns:
      decoded: (M, 3, RESOLUTION[1], RESOLUTION[0]) uint8 aligned to vid_list_sorted_unique
    """
    M = len(vid_list_sorted_unique)
    decoded = torch.empty((M, 3, RESOLUTION[1], RESOLUTION[0]), dtype=torch.uint8)

    # Decode in large batches to reduce per-call overhead.
    for i in range(0, M, decode_chunk):
        batch_vid = vid_list_sorted_unique[i : i + decode_chunk]
        try:
            batch_frames = vr.get_batch(batch_vid).asnumpy()  # (B, H, W, 3)
            batch_torch = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)  # (B, 3, H, W)
            batch_padded = process_batch_gba(batch_torch, RESOLUTION[1], RESOLUTION[0]).to(torch.uint8)
            decoded[i : i + len(batch_vid)] = batch_padded
        except Exception as e:
            # Leave zeros for failed chunk; continue.
            print(f"⚠️ Video read error at decode chunk starting {i}: {e}")
            continue

    return decoded


# -----------------------------------------------------------------------------
# RL event loading/grouping
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RLEvent:
    replay: str
    actor: str
    kind: str
    label: str
    outcome: str
    reason: str
    start_frame: int
    event_frame: int
    end_frame: int
    weight: float
    damage_dealt: int
    damage_taken: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RLEvent":
        return RLEvent(
            replay=str(d.get("replay", "")),
            actor=str(d.get("actor", "")),
            kind=str(d.get("kind", "")),
            label=str(d.get("label", "")),
            outcome=str(d.get("outcome", "")),
            reason=str(d.get("reason", "")),
            start_frame=int(d.get("start_frame", 0)),
            event_frame=int(d.get("event_frame", 0)),
            end_frame=int(d.get("end_frame", 0)),
            weight=float(d.get("weight", 1.0)),
            damage_dealt=int(d.get("damage_dealt", 0)),
            damage_taken=int(d.get("damage_taken", 0)),
        )


def _load_rl_events_grouped(rl_events_path: Path) -> Dict[str, List[RLEvent]]:
    grouped: Dict[str, List[RLEvent]] = {}
    rows = _read_jsonl(rl_events_path)
    for r in rows:
        try:
            evt = RLEvent.from_dict(r)
        except Exception:
            continue
        if not evt.replay:
            continue
        grouped.setdefault(evt.replay, []).append(evt)

    for k in list(grouped.keys()):
        grouped[k].sort(key=lambda e: (e.start_frame, e.event_frame, e.end_frame, e.actor, e.kind, e.label))
    return grouped


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precache RL event segments: frames [start_frame..end_frame] with baked action windows (union video decode per replay)."
    )
    parser.add_argument("--source-dir", default=SOURCE_DIR, help="Dataset root containing replay folders.")
    parser.add_argument("--rl-dir", default=RL_DIR, help="Directory containing rl_events.jsonl (from create_rl_dataset.py).")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Where to write cached RL segment .pt files.")
    parser.add_argument(
        "--actor",
        choices=["player", "enemy", "both"],
        default="both",
        help="Filter RL events by actor (default: both).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached segment files (default: skip existing).",
    )
    parser.add_argument(
        "--max-events-per-replay",
        type=int,
        default=0,
        help="Optional cap per replay (0 = no cap). Uses sorted event order.",
    )
    parser.add_argument(
        "--decode-chunk",
        type=int,
        default=1024,
        help="Batch size for vr.get_batch() when decoding union frames (bigger is usually faster).",
    )
    args = parser.parse_args()

    src_path = Path(args.source_dir)
    rl_path = Path(args.rl_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rl_events_path = rl_path / RL_EVENTS_FILE
    if not rl_events_path.exists():
        raise FileNotFoundError(f"Missing {rl_events_path}. Run scripts/create_rl_dataset.py first.")

    events_by_replay = _load_rl_events_grouped(rl_events_path)

    replays = sorted([d for d in src_path.iterdir() if d.is_dir()])

    print("📦 Precaching RL segments (Union Decode)")
    print(f"   Source: {src_path}")
    print(f"   RL events: {rl_events_path} (replays with events: {len(events_by_replay)})")
    print(f"   Output: {out_path}")
    print(f"   Actor filter: {args.actor}")
    print(f"   Overwrite: {bool(args.overwrite)}")
    print(f"   decode_chunk: {int(args.decode_chunk)}")

    stats = {
        "segments_written": 0,
        "segments_skipped_exists": 0,
        "segments_skipped_no_events": 0,
        "segments_skipped_empty": 0,
        "replays_missing_files": 0,
        "replays_processed": 0,
        "unique_video_frames_decoded": 0,
    }

    for folder in tqdm(replays, desc="Replays"):
        name = folder.name
        events = events_by_replay.get(name, [])

        if args.actor != "both":
            events = [e for e in events if e.actor == args.actor]

        if not events:
            stats["segments_skipped_no_events"] += 1
            continue

        if args.max_events_per_replay and args.max_events_per_replay > 0:
            events = events[: int(args.max_events_per_replay)]

        vid_path = folder / "video.mp4"
        act_path = folder / "actions.jsonl"
        if not (vid_path.exists() and act_path.exists()):
            stats["replays_missing_files"] += 1
            continue

        try:
            actions_rows = _read_jsonl(act_path)
            if not actions_rows:
                stats["segments_skipped_empty"] += 1
                continue

            raw_actions_tensor, vid_idx_map = _build_action_and_vid_maps(actions_rows)

            vr = VideoReader(str(vid_path), ctx=cpu(0))
            vid_len = len(vr)
            T = len(vid_idx_map)

            # -----------------------------------------------------------------
            # Prepare all events for this replay:
            #   - compute kept action indices and corresponding video indices
            #   - collect union of ALL required video indices across events
            # -----------------------------------------------------------------
            prepared: List[Tuple[int, RLEvent, int, int, List[int], List[int], Path]] = []
            needed_vid_set: set[int] = set()

            for j, evt in enumerate(events):
                # Clamp to action timeline length
                sf = max(0, int(evt.start_frame))
                ef = max(0, int(evt.end_frame))
                if ef < sf:
                    ef = sf
                if sf >= T:
                    continue
                ef = min(ef, T - 1)

                seg_action_indices = range(sf, ef + 1)

                seg_vid_indices: List[int] = []
                seg_action_indices_kept: List[int] = []

                for t in seg_action_indices:
                    vid_idx = int(vid_idx_map[t])
                    if 0 <= vid_idx < vid_len:
                        seg_vid_indices.append(vid_idx)
                        seg_action_indices_kept.append(int(t))

                if not seg_action_indices_kept:
                    continue

                tag = _sanitize_filename(f"{evt.actor}_{evt.kind}_{evt.label}")
                out_file = out_path / f"{name}__{tag}__sf{sf}_ef{int(evt.event_frame)}_en{ef}__{j:04d}.pt"

                if out_file.exists() and not args.overwrite:
                    stats["segments_skipped_exists"] += 1
                    continue

                prepared.append((j, evt, sf, ef, seg_action_indices_kept, seg_vid_indices, out_file))
                for v in seg_vid_indices:
                    needed_vid_set.add(int(v))

            if not prepared:
                # All events skipped due to existing files (or empty), but replay counted as processed.
                stats["replays_processed"] += 1
                continue

            # -----------------------------------------------------------------
            # UNION DECODE: decode each unique needed video frame once
            # -----------------------------------------------------------------
            needed_vid_list = sorted(needed_vid_set)
            decoded_all = _decode_unique_video_frames(
                vr,
                needed_vid_list,
                decode_chunk=max(1, int(args.decode_chunk)),
            )
            stats["unique_video_frames_decoded"] += int(decoded_all.shape[0])

            # Map vid_idx -> row in decoded_all
            vid_pos: Dict[int, int] = {int(v): i for i, v in enumerate(needed_vid_list)}

            # -----------------------------------------------------------------
            # Write each event segment by slicing decoded frames + baking actions
            # -----------------------------------------------------------------
            for j, evt, sf, ef, seg_action_indices_kept, seg_vid_indices, out_file in prepared:
                # Translate segment's vid indices into decoded_all indices
                idxs = [vid_pos[int(v)] for v in seg_vid_indices]
                idxs_t = torch.tensor(idxs, dtype=torch.long)

                # frames: (N, 3, H, W)
                frames_tensor = decoded_all.index_select(0, idxs_t)

                # actions: (N, ACTION_HORIZON, ACTION_DIM)
                baked_actions = _bake_action_windows(seg_action_indices_kept, raw_actions_tensor)

                payload = {
                    "frames": frames_tensor,  # (N, 3, H, W) uint8
                    "actions": baked_actions,  # (N, ACTION_HORIZON, ACTION_DIM) float32
                    "meta": {
                        "replay": name,
                        "actor": evt.actor,
                        "kind": evt.kind,
                        "label": evt.label,
                        "outcome": evt.outcome,
                        "reason": evt.reason,
                        "weight": float(evt.weight),
                        "damage_dealt": int(evt.damage_dealt),
                        "damage_taken": int(evt.damage_taken),
                        "start_frame": int(sf),
                        "event_frame": int(evt.event_frame),
                        "end_frame": int(ef),
                        "action_indices": [int(x) for x in seg_action_indices_kept],
                        "video_indices": [int(x) for x in seg_vid_indices],
                        "action_horizon": int(ACTION_HORIZON),
                        "resolution": [int(RESOLUTION[0]), int(RESOLUTION[1])],
                        "union_decode": True,
                    },
                }

                torch.save(payload, out_file)
                stats["segments_written"] += 1

            stats["replays_processed"] += 1

        except Exception as e:
            print(f"❌ Error {name}: {e}")

    print("✅ Done.")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]:,}")


if __name__ == "__main__":
    main()
