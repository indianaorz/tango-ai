#!/usr/bin/env python3
# scripts/fix_tile_owner_left_is_zero.py
from __future__ import annotations

import argparse
import csv
import json
import os
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional


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
                if _is_candidate_flat_owner_grid(v):
                    found.append(GridRef(parent=node, key=k, shape="flat"))
                elif _is_candidate_2d_owner_grid(v):
                    found.append(GridRef(parent=node, key=k, shape="2d"))
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
    v = fr.get("cust_gauge", 0)
    try:
        return int(v) > 0
    except Exception:
        return False


def decide_invert_for_replay_streaming(
    actions_path: Path,
    *,
    max_bad_lines: int,
    max_scan_frames: int,
    min_votes: int,
    ratio: float,
) -> ReplayDecision:
    """
    Streaming version of decide_invert_for_replay:
    - Iterates JSONL once
    - Samples only battle frames
    - Stops early when:
        - scanned battle frames hits max_scan_frames (if >0)
        - OR we already have >= min_votes and dominance is clear
    """
    left0 = 0
    left1 = 0
    scanned = 0
    grids_seen_total = 0

    # helper to check if either side already dominates enough to stop early
    def dominance_verdict() -> Optional[str]:
        total_votes = left0 + left1
        if total_votes < min_votes:
            return None
        if left1 >= int(left0 * ratio + 0.5):
            return "invert"
        if left0 >= int(left1 * ratio + 0.5):
            return "keep"
        return None

    for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=max_bad_lines):
        if not _is_battle_frame(fr):
            continue

        if max_scan_frames > 0 and scanned >= max_scan_frames:
            break

        grids = _walk_find_owner_grids(fr)
        grids_seen_total += len(grids)

        if grids:
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

        dv = dominance_verdict()
        if dv is not None:
            # Early-exit once decisive (saves time on long replays)
            if dv == "invert":
                return ReplayDecision(
                    verdict="invert",
                    reason=f"early:left1>=left0*{ratio}",
                    frames_scanned=scanned,
                    left0_votes=left0,
                    left1_votes=left1,
                    grids_seen=grids_seen_total,
                )
            return ReplayDecision(
                verdict="keep",
                reason=f"early:left0>=left1*{ratio}",
                frames_scanned=scanned,
                left0_votes=left0,
                left1_votes=left1,
                grids_seen=grids_seen_total,
            )

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
    # recursive: **/actions.jsonl
    return sorted(root.glob("**/actions.jsonl"))


# -----------------------------
# Parallel worker
# -----------------------------

@dataclass(frozen=True)
class WorkerResult:
    folder: str
    path: str
    verdict: str
    reason: str
    left0_votes: int
    left1_votes: int
    frames_scanned: int
    grids_seen: int
    applied: bool
    error: str


def _process_one(args: Tuple[str, bool, int, int, int, float]) -> WorkerResult:
    path_str, do_apply, max_bad_lines, max_scan_frames, min_votes, ratio = args
    actions_path = Path(path_str)
    folder = actions_path.parent.name
    path_out = str(actions_path).replace("\\", "/")

    try:
        dec = decide_invert_for_replay_streaming(
            actions_path,
            max_bad_lines=max_bad_lines,
            max_scan_frames=max_scan_frames,
            min_votes=min_votes,
            ratio=ratio,
        )

        applied = False
        if do_apply and dec.verdict == "invert":
            apply_invert_to_actions_jsonl(actions_path, max_bad_lines=max_bad_lines)
            applied = True

        return WorkerResult(
            folder=folder,
            path=path_out,
            verdict=dec.verdict,
            reason=dec.reason,
            left0_votes=dec.left0_votes,
            left1_votes=dec.left1_votes,
            frames_scanned=dec.frames_scanned,
            grids_seen=dec.grids_seen,
            applied=applied,
            error="",
        )
    except Exception as e:
        return WorkerResult(
            folder=folder,
            path=path_out,
            verdict="error",
            reason="",
            left0_votes=0,
            left1_votes=0,
            frames_scanned=0,
            grids_seen=0,
            applied=False,
            error=str(e),
        )


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
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel workers")

    args = ap.parse_args()

    root = Path(args.root)
    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    tasks: List[Tuple[str, bool, int, int, int, float]] = [
        (str(p), bool(args.apply), int(args.max_bad_lines), int(args.max_scan_frames), int(args.min_votes), float(args.ratio))
        for p in files
    ]

    results: List[WorkerResult] = []
    use_parallel = (args.workers or 1) > 1 and len(tasks) > 1

    def log_line(res: WorkerResult) -> None:
        prefix = "[APPLY]" if args.apply else "[DRY]"
        if res.verdict == "error":
            msg = f"{prefix} {res.folder}: ERROR {res.error}"
        else:
            msg = (
                f"{prefix} {res.folder}: {res.verdict}  "
                f"votes(L0={res.left0_votes},L1={res.left1_votes})  "
                f"scanned={res.frames_scanned}  grids_seen={res.grids_seen}  reason={res.reason}"
            )
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)

    if use_parallel:
        with multiprocessing.Pool(processes=args.workers) as pool:
            it = pool.imap_unordered(_process_one, tasks, chunksize=1)
            if tqdm is not None:
                it = tqdm(it, total=len(tasks), desc="TileOwner", unit="file")
            for res in it:
                results.append(res)
                log_line(res)
    else:
        for t in tasks:
            res = _process_one(t)
            results.append(res)
            log_line(res)

    # Stats
    n_invert = sum(1 for r in results if r.verdict == "invert")
    n_keep = sum(1 for r in results if r.verdict == "keep")
    n_unclear = sum(1 for r in results if r.verdict == "unclear")
    n_error = sum(1 for r in results if r.verdict == "error")
    n_applied = sum(1 for r in results if r.applied)

    # Report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "folder",
        "path",
        "verdict",
        "reason",
        "left0_votes",
        "left1_votes",
        "frames_scanned",
        "grids_seen",
        "applied",
        "error",
    ]

    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "folder": r.folder,
                    "path": r.path,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "left0_votes": r.left0_votes,
                    "left1_votes": r.left1_votes,
                    "frames_scanned": r.frames_scanned,
                    "grids_seen": r.grids_seen,
                    "applied": str(r.applied),
                    "error": r.error,
                }
            )

    print("")
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Done ({mode}). files={len(files)}  invert={n_invert}  keep={n_keep}  unclear={n_unclear}  error={n_error}")
    if args.apply:
        print(f"Applied rewrites: {n_applied}")
    print(f"Workers: {args.workers}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
