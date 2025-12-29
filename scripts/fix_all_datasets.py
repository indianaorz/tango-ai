# scripts/fix_all_datasets.py
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm  # NEW: For progress bars

# Ensure we can import sibling scripts
sys.path.append(str(Path(__file__).parent))

try:
    # 1. Grid Flip
    import batch_fix_grid_flip as bf_grid
    # 2. Cross Owner
    import batch_fix_cross_owner as bf_cross
    # 3. Tile Owner 0/1
    import fix_tile_owner_left_is_zero as bf_tile
    # 4. Health/Pos Flip
    import detect_health_flip as bf_health
except ImportError as e:
    print(f"Error importing helper scripts: {e}")
    print("Ensure batch_fix_grid_flip.py, batch_fix_cross_owner.py, "
          "fix_tile_owner_left_is_zero.py, and detect_health_flip.py are in scripts/")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Configuration & Re-implementations for In-Memory Processing
# -----------------------------------------------------------------------------

def _get_pos(frame: Dict[str, Any], key: str) -> Optional[List[float]]:
    val = frame.get(key)
    if isinstance(val, list) and len(val) >= 2:
        return val
    return None

def _get_input(frame: Dict[str, Any], key: str) -> float:
    val = frame.get(key, 0)
    try:
        return float(val)
    except:
        return 0.0

def analyze_health_flip_in_memory(frames: List[Dict[str, Any]]) -> bf_health.Score:
    """
    Adapted from detect_health_flip.py to work on in-memory list instead of file path.
    """
    score_normal = 0.0
    score_flipped = 0.0
    valid_samples = 0

    prev_p_pos = None
    prev_e_pos = None

    # Limit scan to first 50k frames to match original script default
    scan_frames = frames[:50000]

    for frame in scan_frames:
        p_pos = _get_pos(frame, "player_pos")
        e_pos = _get_pos(frame, "enemy_pos")

        if prev_p_pos is not None and prev_e_pos is not None and p_pos and e_pos:
            dp = [p_pos[0] - prev_p_pos[0], p_pos[1] - prev_p_pos[1]]
            de = [e_pos[0] - prev_e_pos[0], e_pos[1] - prev_e_pos[1]]

            for btn, axis, direction in bf_health.MOVEMENT_RULES:
                if _get_input(frame, btn) > 0.5:
                    # Normal Hypothesis
                    if (dp[axis] * direction) > 0.01: 
                        score_normal += 1.0
                    elif (dp[axis] * direction) < -0.01:
                        score_normal -= 0.5

                    # Flipped Hypothesis
                    if (de[axis] * direction) > 0.01:
                        score_flipped += 1.0
                    elif (de[axis] * direction) < -0.01:
                        score_flipped -= 0.5
                    
                    valid_samples += 1

        prev_p_pos = p_pos
        prev_e_pos = e_pos

    return bf_health.Score(score_normal, score_flipped, valid_samples)


# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------

@dataclass
class FileResult:
    path: str
    frames: int
    modified: bool
    ops: List[str]  # ["GridFlip", "CrossSwap", "TileInvert", "HealthSwap"]

def process_file_pipeline(args_pack) -> FileResult:
    path_str, apply_changes = args_pack
    path = Path(path_str)
    
    # 1. Read File
    frames = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        frames.append(json.loads(line))
                    except:
                        pass
    except Exception as e:
        return FileResult(path.name, 0, False, [f"Error: {e}"])

    if not frames:
        return FileResult(path.name, 0, False, ["Empty"])

    ops_performed = []
    is_modified = False

    # ---------------------------------------------------------
    # Step 1: Batch Fix Grid Flip
    # ---------------------------------------------------------
    try:
        _, _, _, best_flipped, _, decision, _ = bf_grid.analyze_flip_owner_based(
            frames, margin=0.03, min_obs=1000, center_method="auto"
        )
        
        if decision == "flip":
            ops_performed.append("GridFlip")
            if apply_changes:
                plan = bf_grid.FixPlan(
                    mirror=best_flipped.mirror,
                    swap_owner=best_flipped.swap_owner,
                    swap_entities=best_flipped.swap_entities
                )
                for fr in frames:
                    bf_grid.apply_fix_in_place(fr, plan)
                is_modified = True
    except Exception as e:
        ops_performed.append(f"GridErr:{e}")

    # ---------------------------------------------------------
    # Step 2: Batch Fix Cross Owner
    # ---------------------------------------------------------
    try:
        ev_cross = bf_cross.detect.analyze_actions_frames(
            frames,
            pre_mode_frames=12,
            pre_battle_scan=180,
            lead_frames=2,
            early_exit_margin=2,
            min_votes=2
        )
        
        ev_dict = {
            "decision": ev_cross.decision,
            "reason": ev_cross.reason
        }
        
        verdict, _ = bf_cross.decide_cross_owner_swap(ev_dict)
        
        if verdict == "swap":
            ops_performed.append("CrossSwap")
            if apply_changes:
                for fr in frames:
                    bf_cross.apply_swap_game_emotion(fr)
                is_modified = True
    except Exception as e:
        ops_performed.append(f"CrossErr:{e}")

    # ---------------------------------------------------------
    # Step 3: Fix Tile Owner (Left is Zero)
    # ---------------------------------------------------------
    try:
        dec_tile = bf_tile.decide_invert_for_replay(
            frames,
            max_scan_frames=400,
            min_votes=10,
            ratio=1.25
        )
        
        if dec_tile.verdict == "invert":
            ops_performed.append("TileInvert")
            if apply_changes:
                for fr in frames:
                    grids = bf_tile._walk_find_owner_grids(fr)
                    for gref in grids:
                        g = gref.get()
                        bf_tile._invert_owner_grid_inplace(g, gref.shape)
                is_modified = True
    except Exception as e:
        ops_performed.append(f"TileErr:{e}")

    # ---------------------------------------------------------
    # Step 4: Detect Health/Pos Flip
    # ---------------------------------------------------------
    try:
        score_health = analyze_health_flip_in_memory(frames)
        
        health_verdict = "UNCLEAR"
        # Using margin 5.0 same as script default
        if score_health.flipped > (score_health.normal + 5.0):
            health_verdict = "FLIP"
        
        if health_verdict == "FLIP":
            ops_performed.append("HealthSwap")
            if apply_changes:
                for fr in frames:
                    bf_health.swap_keys_inplace(fr)
                is_modified = True
    except Exception as e:
        ops_performed.append(f"HealthErr:{e}")

    # ---------------------------------------------------------
    # Save (Atomic)
    # ---------------------------------------------------------
    if apply_changes and is_modified:
        bf_health.write_jsonl_atomic(path, frames)

    return FileResult(path.parent.name, len(frames), is_modified, ops_performed)


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run complete data correction pipeline (4 passes) in parallel.")
    parser.add_argument("--root", type=str, default="data/dataset", help="Dataset root directory")
    parser.add_argument("--apply", action="store_true", help="Actually modify files. Default is DRY RUN.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    # Find files
    print(f"Scanning {root} for actions.jsonl...")
    files = sorted([str(p) for p in root.glob("**/actions.jsonl")])
    print(f"Found {len(files)} files.")
    
    if not args.apply:
        print("\n[DRY RUN MODE] No files will be modified. Use --apply to execute fixes.\n")
    else:
        print("\n[APPLY MODE] Files WILL be modified in place.\n")

    # Prepare args for map
    task_args = [(f, args.apply) for f in files]

    t0 = time.time()
    
    print(f"Starting processing with {args.workers} workers...")
    
    stats = {
        "GridFlip": 0,
        "CrossSwap": 0,
        "TileInvert": 0,
        "HealthSwap": 0,
        "Modified": 0,
        "Total": 0
    }

    with multiprocessing.Pool(processes=args.workers) as pool:
        # Use imap_unordered for responsive progress bar
        # Chunksize=1 usually fine, but slightly larger helps reduce IPC overhead for many small files
        iterator = pool.imap_unordered(process_file_pipeline, task_args, chunksize=1)
        
        for res in tqdm(iterator, total=len(files), desc="Processing", unit="file"):
            stats["Total"] += 1
            if res.modified:
                stats["Modified"] += 1
            
            for op in res.ops:
                # Remove specific error text to categorize just the op type if valid
                op_key = op.split(":")[0] if "Err" in op else op
                if op_key in stats:
                    stats[op_key] += 1
            
            # Log significant changes using tqdm.write so bar doesn't break
            if res.ops:
                ops_str = ", ".join(res.ops)
                status = "FIXED" if args.apply and res.modified else "NEEDS FIX"
                tqdm.write(f"[{status}] {res.path}: {ops_str}")

    dt = time.time() - t0
    print("-" * 60)
    print(f"Processing Complete in {dt:.2f}s")
    print(f"Files Scanned: {stats['Total']}")
    print(f"Files Modified: {stats['Modified']}")
    print("Operations Summary:")
    print(f"  Grid Flips:   {stats['GridFlip']}")
    print(f"  Cross Swaps:  {stats['CrossSwap']}")
    print(f"  Tile Inverts: {stats['TileInvert']}")
    print(f"  Health Swaps: {stats['HealthSwap']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()