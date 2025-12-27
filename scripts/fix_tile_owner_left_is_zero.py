#!/usr/bin/env python3
# scripts/fix_tile_owner_left_is_zero.py
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


# Battle Network panel grid is 6x3.
GRID_W = 6
GRID_H = 3
GRID_N = GRID_W * GRID_H


# -----------------------------
# JSONL helpers (tolerant)
# -----------------------------

def iter_jsonl_tolerant(path: Path, *, max_bad_lines: int = 50) -> Iterator[Dict[str, Any]]:
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] {path} bad json at line {line_no} (skipping)")
                if bad >= max_bad_lines:
                    raise RuntimeError(f"Too many bad json lines in {path} (>= {max_bad_lines}). Aborting.")
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl_atomic(path: Path, rows: Iterator[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for obj in rows:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


# -----------------------------
# Owner-grid detection
# -----------------------------

JsonObj = Union[Dict[str, Any], List[Any]]

@dataclass(frozen=True)
class GridRef:
    """
    Reference to a mutable list that represents a tile-owner grid.
    We store the container + key/index so we can edit in-place.
    """
    parent: Union[Dict[str, Any], List[Any]]
    key: Union[str, int]
    shape: str  # "flat" or "2d"

    def get(self) -> Any:
        if isinstance(self.parent, dict) and isinstance(self.key, str):
            return self.parent.get(self.key)
        if isinstance(self.parent, list) and isinstance(self.key, int):
            return self.parent[self.key]
        return None

    def set(self, value: Any) -> None:
        if isinstance(self.parent, dict) and isinstance(self.key, str):
            self.parent[self.key] = value
        elif isinstance(self.parent, list) and isinstance(self.key, int):
            self.parent[self.key] = value


def _is_int01(v: Any) -> bool:
    # Accept bools too, but treat them as ints.
    if isinstance(v, bool):
        v = int(v)
    if not isinstance(v, int):
        return False
    return v in (0, 1)


def _is_candidate_flat_owner_grid(x: Any) -> bool:
    if not isinstance(x, list):
        return False
    if len(x) != GRID_N:
        return False
    return all(_is_int01(v) for v in x)


def _is_candidate_2d_owner_grid(x: Any) -> bool:
    if not isinstance(x, list) or len(x) != GRID_H:
        return False
    for row in x:
        if not isinstance(row, list) or len(row) != GRID_W:
            return False
        if not all(_is_int01(v) for v in row):
            return False
    return True


def _walk_find_owner_grids(obj: JsonObj) -> List[GridRef]:
    """
    Find all lists that look like a BN tile owner grid (0/1 only):
      - flat list len 18
      - 2d list 3x6
    We do NOT rely on key names, but this still avoids false positives by shape+values.
    """
    found: List[GridRef] = []

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                # Check direct value
                if _is_candidate_flat_owner_grid(v):
                    found.append(GridRef(parent=node, key=k, shape="flat"))
                elif _is_candidate_2d_owner_grid(v):
                    found.append(GridRef(parent=node, key=k, shape="2d"))
                # Recurse
                if isinstance(v, (dict, list)):
                    rec(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if _is_candidate_flat_owner_grid(v):
                    found.append(GridRef(parent=node, key=i, shape="flat"))
                elif _is_candidate_2d_owner_grid(v):
                    found.append(GridRef(parent=node, key=i, shape="2d"))
                if isinstance(v, (dict, list)):
                    rec(v)

    rec(obj)
    return found


def _leftmost_column_values(grid: Any, shape: str) -> Optional[List[int]]:
    if shape == "flat":
        if not isinstance(grid, list) or len(grid) != GRID_N:
            return None
        # flat is row-major: rows 0..H-1, cols 0..W-1
        vals: List[int] = []
        for r in range(GRID_H):
            idx = r * GRID_W + 0
            vals.append(int(grid[idx]))
        return vals
    if shape == "2d":
        if not _is_candidate_2d_owner_grid(grid):
            return None
        return [int(grid[r][0]) for r in range(GRID_H)]
    return None


def _invert_owner_grid_inplace(grid: Any, shape: str) -> bool:
    """
    Invert 0<->1 in-place. Returns True if modified.
    """
    if shape == "flat":
        if not _is_candidate_flat_owner_grid(grid):
            return False
        for i, v in enumerate(grid):
            grid[i] = 1 - int(v)
        return True
    if shape == "2d":
        if not _is_candidate_2d_owner_grid(grid):
            return False
        for r in range(GRID_H):
            row = grid[r]
            for c in range(GRID_W):
                row[c] = 1 - int(row[c])
        return True
    return False


# -----------------------------
# Decision logic
# -----------------------------

@dataclass(frozen=True)
class ReplayDecision:
    verdict: str  # "invert" | "keep" | "unclear"
    reason: str
    frames_scanned: int
    left0_votes: int
    left1_votes: int
    grids_seen: int


def _is_battle_frame(fr: Dict[str, Any]) -> bool:
    # Conservative: if cust_gauge exists and >0, we're in battle flow.
    v = fr.get("cust_gauge", 0)
    try:
        return int(v) > 0
    except Exception:
        return False


def decide_invert_for_replay(
    frames: List[Dict[str, Any]],
    *,
    max_scan_frames: int,
    min_votes: int,
    ratio: float,
) -> ReplayDecision:
    """
    Sample battle frames. For each frame, for each detected owner-grid,
    vote based on leftmost column:
      - if majority of leftmost tiles are 0 => left0_votes++
      - if majority are 1 => left1_votes++
      - ties ignored
    Decide invert if left1 dominates left0 by `ratio`.
    """
    left0 = 0
    left1 = 0
    scanned = 0
    grids_seen_total = 0

    for fr in frames:
        if max_scan_frames > 0 and scanned >= max_scan_frames:
            break
        if not _is_battle_frame(fr):
            continue

        grids = _walk_find_owner_grids(fr)
        grids_seen_total += len(grids)
        if not grids:
            scanned += 1
            continue

        # We take a frame-level vote using ALL grids in that frame.
        # This makes it robust even if you store multiple owner grids (e.g., raw + derived).
        frame_left_vals: List[int] = []
        for gref in grids:
            g = gref.get()
            vals = _leftmost_column_values(g, gref.shape)
            if vals:
                frame_left_vals.extend(vals)

        if frame_left_vals:
            zeros = sum(1 for v in frame_left_vals if v == 0)
            ones = sum(1 for v in frame_left_vals if v == 1)
            if zeros > ones:
                left0 += 1
            elif ones > zeros:
                left1 += 1

        scanned += 1

    total_votes = left0 + left1
    if total_votes < min_votes:
        return ReplayDecision(
            verdict="unclear",
            reason=f"votes<{min_votes}",
            frames_scanned=scanned,
            left0_votes=left0,
            left1_votes=left1,
            grids_seen=grids_seen_total,
        )

    # If leftmost is mostly 1, invert so leftmost becomes 0.
    if left1 >= int(left0 * ratio + 0.5):
        return ReplayDecision(
            verdict="invert",
            reason=f"left1>=left0*{ratio}",
            frames_scanned=scanned,
            left0_votes=left0,
            left1_votes=left1,
            grids_seen=grids_seen_total,
        )

    if left0 >= int(left1 * ratio + 0.5):
        return ReplayDecision(
            verdict="keep",
            reason=f"left0>=left1*{ratio}",
            frames_scanned=scanned,
            left0_votes=left0,
            left1_votes=left1,
            grids_seen=grids_seen_total,
        )

    return ReplayDecision(
        verdict="unclear",
        reason="no_clear_dominance",
        frames_scanned=scanned,
        left0_votes=left0,
        left1_votes=left1,
        grids_seen=grids_seen_total,
    )


# -----------------------------
# Apply (per replay)
# -----------------------------

def apply_invert_to_actions_jsonl(actions_path: Path, *, max_bad_lines: int) -> None:
    def transformed() -> Iterator[Dict[str, Any]]:
        for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=max_bad_lines):
            grids = _walk_find_owner_grids(fr)
            for gref in grids:
                g = gref.get()
                _invert_owner_grid_inplace(g, gref.shape)
            yield fr

    write_jsonl_atomic(actions_path, transformed())


def find_actions_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if "actions.jsonl" in filenames:
            out.append(Path(dirpath) / "actions.jsonl")
    out.sort()
    return out


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/dataset")
    ap.add_argument("--apply", action="store_true", help="rewrite actions.jsonl for verdict=invert")
    ap.add_argument("--report", type=str, default="tile_owner_left_is_zero_report.csv")

    ap.add_argument("--max-bad-lines", type=int, default=50)
    ap.add_argument("--max-scan-frames", type=int, default=400, help="battle frames to sample per replay (0 = no limit)")
    ap.add_argument("--min-votes", type=int, default=10, help="min decisive votes needed to decide invert/keep")
    ap.add_argument("--ratio", type=float, default=1.25, help="dominance ratio needed to decide")

    args = ap.parse_args()

    root = Path(args.root)
    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    report_rows: List[Dict[str, Any]] = []
    n_invert = 0
    n_keep = 0
    n_unclear = 0

    for actions_path in files:
        replay = actions_path.parent.name

        # Load a sample of frames (tolerant); decision is per replay.
        frames: List[Dict[str, Any]] = []
        for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=args.max_bad_lines):
            frames.append(fr)
            # Don't hard-cap raw frames here; decide_invert_for_replay handles battle sampling.
            # Keeping this as full read keeps code simpler and robust for battle-frame filtering.

        dec = decide_invert_for_replay(
            frames,
            max_scan_frames=int(args.max_scan_frames),
            min_votes=int(args.min_votes),
            ratio=float(args.ratio),
        )

        if dec.verdict == "invert":
            n_invert += 1
        elif dec.verdict == "keep":
            n_keep += 1
        else:
            n_unclear += 1

        print(
            f"[{dec.verdict.upper()}] {replay}  "
            f"votes(L0={dec.left0_votes},L1={dec.left1_votes})  "
            f"scanned={dec.frames_scanned}  grids_seen={dec.grids_seen}  reason={dec.reason}"
        )

        report_rows.append(
            {
                "folder": replay,
                "path": str(actions_path).replace("\\", "/"),
                "verdict": dec.verdict,
                "reason": dec.reason,
                "left0_votes": dec.left0_votes,
                "left1_votes": dec.left1_votes,
                "frames_scanned": dec.frames_scanned,
                "grids_seen": dec.grids_seen,
            }
        )

        if args.apply and dec.verdict == "invert":
            apply_invert_to_actions_jsonl(actions_path, max_bad_lines=args.max_bad_lines)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    print("")
    print(f"Done. files={len(files)}  invert={n_invert}  keep={n_keep}  unclear={n_unclear}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
