#scripts/detect_health_flip.py
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Keys to check for movement correlation (Detection Logic)
# (Input Key,  Axis Index,  Direction (+1 or -1))
# X axis is index 0, Y axis is index 1.
# Assuming standard grid: Right increases X, Down increases Y.
MOVEMENT_RULES = [
    ("DPAD_RIGHT", 0,  1),
    ("DPAD_LEFT",  0, -1),
    ("DPAD_UP",    1, -1),
    ("DPAD_DOWN",  1,  1),
]

# STRICT SWAP LIST: Only these specific pairs will be flipped.
# We do not touch chips, emotions, or grid states as those are handled elsewhere.
TARGET_PAIRS = [
    ("player_pos", "enemy_pos"),
    ("player_health", "enemy_health"),
    ("player_charge", "enemy_charge"),
    ("player_chip", "enemy_chip"),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass

def write_jsonl_atomic(path: Path, rows: Iterator[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for obj in rows:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)

def _get_pos(frame: Dict[str, Any], key: str) -> Optional[List[float]]:
    val = frame.get(key)
    if isinstance(val, list) and len(val) >= 2:
        return val
    # Handle CSV-string style "1,2" if present in raw data
    if isinstance(val, str) and "," in val:
        try:
            return [float(x) for x in val.split(",")]
        except:
            pass
    return None

def _get_input(frame: Dict[str, Any], key: str) -> float:
    # Handles 0/1 ints, floats, or booleans
    val = frame.get(key, 0)
    try:
        return float(val)
    except:
        return 0.0

# -----------------------------------------------------------------------------
# Detection Logic (Unchanged)
# -----------------------------------------------------------------------------

@dataclass
class Score:
    normal: float
    flipped: float
    samples: int

def analyze_file(path: Path, max_frames: int = 50000) -> Score:
    """
    Scans the file and calculates a score for Normal vs Flipped based on
    Input -> Movement correlation.
    """
    score_normal = 0.0
    score_flipped = 0.0
    valid_samples = 0

    prev_p_pos = None
    prev_e_pos = None

    for i, frame in enumerate(iter_jsonl(path)):
        if i >= max_frames:
            break

        # Get current positions
        p_pos = _get_pos(frame, "player_pos")
        e_pos = _get_pos(frame, "enemy_pos")

        # We need previous frame to calculate velocity
        if prev_p_pos is not None and prev_e_pos is not None and p_pos and e_pos:
            
            # Calculate deltas
            dp = [p_pos[0] - prev_p_pos[0], p_pos[1] - prev_p_pos[1]]
            de = [e_pos[0] - prev_e_pos[0], e_pos[1] - prev_e_pos[1]]

            # Check inputs
            for btn, axis, direction in MOVEMENT_RULES:
                if _get_input(frame, btn) > 0.5:
                    # Expect movement in 'direction' on 'axis'
                    
                    # Normal Hypothesis: Input moves Player
                    if (dp[axis] * direction) > 0.01: 
                        score_normal += 1.0
                    elif (dp[axis] * direction) < -0.01:
                        score_normal -= 0.5

                    # Flipped Hypothesis: Input moves Enemy
                    if (de[axis] * direction) > 0.01:
                        score_flipped += 1.0
                    elif (de[axis] * direction) < -0.01:
                        score_flipped -= 0.5
                    
                    # Note: We don't count frames where NO movement happens
                    valid_samples += 1

        prev_p_pos = p_pos
        prev_e_pos = e_pos

    return Score(score_normal, score_flipped, valid_samples)

# -----------------------------------------------------------------------------
# Fixer Logic (Restricted)
# -----------------------------------------------------------------------------

def swap_keys_inplace(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Swaps ONLY the keys defined in TARGET_PAIRS.
    """
    for p_key, e_key in TARGET_PAIRS:
        # Check if both exist (or at least one exists to swap)
        # Usually better to check if at least one exists.
        if p_key in frame or e_key in frame:
            val_p = frame.get(p_key)
            val_e = frame.get(e_key)
            
            # Swap values. If one key was missing, it becomes None (or we can pop it).
            # JSON serialization of None is null, which is fine.
            frame[p_key] = val_e
            frame[e_key] = val_p

    return frame

def apply_fix(path: Path) -> None:
    def process():
        for frame in iter_jsonl(path):
            yield swap_keys_inplace(frame)
    
    write_jsonl_atomic(path, process())

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect and fix swapped player/enemy data (Pos, Health, Charge ONLY).")
    parser.add_argument("--root", type=str, default="data/dataset", help="Dataset root directory")
    parser.add_argument("--apply", action="store_true", help="Actually modify the files. Default is dry-run.")
    parser.add_argument("--margin", type=float, default=5.0, help="Score difference required to make a decision.")
    parser.add_argument("--report", type=str, default="health_flip_report.csv", help="Output report CSV.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    files = sorted(root.glob("**/actions.jsonl"))
    print(f"Found {len(files)} files in {root}...")

    results = []
    n_flipped = 0
    n_normal = 0
    n_unclear = 0

    print(f"{'STATUS':<10} | {'NORMAL':<8} | {'FLIPPED':<8} | {'PATH'}")
    print("-" * 80)

    for p in files:
        score = analyze_file(p)
        
        decision = "UNCLEAR"
        
        # Heuristic: Flipped score must be significantly higher than Normal
        if score.flipped > (score.normal + args.margin):
            decision = "FLIP"
            n_flipped += 1
        elif score.normal > (score.flipped + args.margin):
            decision = "NORMAL"
            n_normal += 1
        else:
            n_unclear += 1

        print(f"{decision:<10} | {score.normal:<8.1f} | {score.flipped:<8.1f} | {p.parent.name}")

        if args.apply and decision == "FLIP":
            print(f"   -> Fixing {p.parent.name} (Swapping Pos/Health/Charge)...")
            apply_fix(p)

        results.append({
            "path": str(p),
            "folder": p.parent.name,
            "decision": decision,
            "score_normal": score.normal,
            "score_flipped": score.flipped,
            "samples": score.samples
        })

    # Summary
    print("-" * 80)
    print(f"Summary: Normal={n_normal}, Flipped={n_flipped}, Unclear={n_unclear}")
    
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "folder", "decision", "score_normal", "score_flipped", "samples"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Report written to {args.report}")

    if n_flipped > 0 and not args.apply:
        print("\n⚠️  Run with --apply to fix the FLIPPED files.")

if __name__ == "__main__":
    main()