#!/usr/bin/env python3
"""
analyze_jsonl.py (v4)

Filter (requested):
  Use ONLY frames where:
    inside_window == False AND cust_gauge > 0

Changes vs v3:
  - FIX grid labeling: choose which PCA axis is "columns" by spread/range.
    On a 3x6 BN grid, the 6-column direction must have the larger extent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROWS = 3
COLS = 6
N = ROWS * COLS

DPAD_KEYS = ["DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"]
DIRS = {
    "DPAD_UP": (-1, 0),
    "DPAD_DOWN": (1, 0),
    "DPAD_LEFT": (0, -1),
    "DPAD_RIGHT": (0, 1),
}


def parse_jsonl(path: Path, max_lines: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_pressed(v: Any, thresh: float = 0.5) -> bool:
    try:
        return float(v) > thresh
    except Exception:
        return False


def summarize_int(values: List[int]) -> str:
    if not values:
        return "n=0"
    vs = sorted(values)
    n = len(vs)

    def q(p: float) -> int:
        if n == 1:
            return vs[0]
        idx = int(round(p * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return vs[idx]

    return (
        f"n={n} mean={statistics.mean(vs):.2f} "
        f"p50={q(0.50)} p90={q(0.90)} p99={q(0.99)} "
        f"min={vs[0]} max={vs[-1]}"
    )


def first_valid_grid_snapshot(frames: List[Dict[str, Any]]) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[int]]:
    for i, fr in enumerate(frames):
        owners = fr.get("grid_owner_state")
        tiles = fr.get("grid_state")
        if isinstance(owners, list) and isinstance(tiles, list) and len(owners) == N and len(tiles) == N:
            try:
                owners_i = [int(x) for x in owners]
                tiles_i = [int(x) for x in tiles]
                return owners_i, tiles_i, i
            except Exception:
                continue
    return None, None, None


def keep_frame(fr: Dict[str, Any]) -> bool:
    if bool(fr.get("inside_window", False)):
        return False
    try:
        return int(fr.get("cust_gauge", 0)) > 0
    except Exception:
        return False


# -----------------------------
# K-means (deterministic, numpy)
# -----------------------------
def kmeans_pp_init(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)

    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[i]) ** 2, axis=1))

    return centers


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    centers = kmeans_pp_init(X, k=k, seed=seed)
    labels = np.zeros((X.shape[0],), dtype=np.int32)

    for _ in range(iters):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)

    return centers, labels


# -----------------------------
# PCA + grid labeling (fixed)
# -----------------------------
@dataclass
class PCAAxes:
    origin: np.ndarray
    a: np.ndarray  # principal axis (largest variance)
    b: np.ndarray  # secondary axis


def pca_axes(centers: np.ndarray) -> PCAAxes:
    mu = centers.mean(axis=0)
    C = centers - mu
    cov = (C.T @ C) / max(C.shape[0] - 1, 1)
    w, V = np.linalg.eigh(cov)  # ascending
    a = V[:, np.argmax(w)]
    b = V[:, np.argmin(w)]
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return PCAAxes(origin=mu, a=a, b=b)


def assign_centers_to_grid(centers: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Key idea:
      On a 3x6 grid, the 6-column direction should have larger extent.
    So we:
      - compute projections along both PCA axes
      - choose the axis with larger range as "col axis"
      - the other becomes "row axis"
    Then we orient:
      - col axis to point to increasing +x (right-ish)
      - row axis to point to increasing +y (down-ish)
    """
    axes = pca_axes(centers)
    C = centers - axes.origin[None, :]
    pa = C @ axes.a
    pb = C @ axes.b

    range_a = float(pa.max() - pa.min())
    range_b = float(pb.max() - pb.min())

    if range_a >= range_b:
        col_dir = axes.a
        row_dir = axes.b
        pcol = pa
        prow = pb
    else:
        col_dir = axes.b
        row_dir = axes.a
        pcol = pb
        prow = pa

    # Orient col_dir so right increases x
    if col_dir[0] < 0:
        col_dir = -col_dir
        pcol = -pcol

    # Orient row_dir so down increases y
    if row_dir[1] < 0:
        row_dir = -row_dir
        prow = -prow

    # Now bin by columns using pcol into 6 groups, and within each sort by prow into 3 rows.
    order = np.argsort(pcol)
    col_of = np.full((centers.shape[0],), -1, dtype=np.int32)
    for c in range(COLS):
        chunk = order[c * ROWS : (c + 1) * ROWS]
        for idx in chunk:
            col_of[idx] = c

    grid: Dict[Tuple[int, int], np.ndarray] = {}
    for c in range(COLS):
        ids = np.where(col_of == c)[0]
        ids_sorted = ids[np.argsort(prow[ids])]
        if ids_sorted.shape[0] != ROWS:
            ids_sorted = ids_sorted[:ROWS]
        for r in range(ROWS):
            grid[(r, c)] = centers[ids_sorted[r]]
    return grid


def nearest_cell(pos: np.ndarray, grid: Dict[Tuple[int, int], np.ndarray]) -> Tuple[Tuple[int, int], float]:
    best_rc = (0, 0)
    best_d2 = float("inf")
    for rc, ctr in grid.items():
        d2 = float(np.sum((pos - ctr) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_rc = rc
    return best_rc, math.sqrt(best_d2)


@dataclass
class MoveEvent:
    frame_idx: int
    tick: int
    key: str
    start_rc: Tuple[int, int]
    end_rc: Optional[Tuple[int, int]]
    dt_frames: Optional[int]
    matched_dir: Optional[bool]
    note: str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=str)
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--stable_eps", type=float, default=2.0)
    ap.add_argument("--assign_max_dist", type=float, default=60.0)
    ap.add_argument("--kmeans_seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    frames = list(parse_jsonl(path, args.max_lines))
    if len(frames) < 50:
        print("Not enough frames.")
        return 1

    owners, tiles, snap_i = first_valid_grid_snapshot(frames)

    # Collect stable points for clustering (ONLY kept frames)
    pts: List[Tuple[float, float]] = []
    prev: Optional[Tuple[float, float]] = None
    kept_count = 0

    for fr in frames:
        if not keep_frame(fr):
            prev = None
            continue
        kept_count += 1
        p = fr.get("player_pos")
        if not (isinstance(p, list) and len(p) == 2):
            prev = None
            continue
        x, y = float(p[0]), float(p[1])
        if prev is not None:
            dx = abs(x - prev[0])
            dy = abs(y - prev[1])
            if dx <= args.stable_eps and dy <= args.stable_eps:
                pts.append((x, y))
        prev = (x, y)

    if len(pts) < 500:
        print(f"Kept frames (inside_window=false & cust_gauge>0): {kept_count}")
        print(f"Not enough stable points for clustering (got {len(pts)}). Try raising --stable_eps or --max_lines.")
        return 1

    X = np.array(pts, dtype=np.float64)
    centers, _labels = kmeans(X, k=N, iters=80, seed=args.kmeans_seed)
    grid = assign_centers_to_grid(centers)

    # Mean neighbor deltas (using our labeled r/c)
    dcol = []
    drow = []
    for r in range(ROWS):
        for c in range(COLS - 1):
            dcol.append(grid[(r, c + 1)] - grid[(r, c)])
    for r in range(ROWS - 1):
        for c in range(COLS):
            drow.append(grid[(r + 1, c)] - grid[(r, c)])
    mean_dcol = np.mean(np.stack(dcol), axis=0)
    mean_drow = np.mean(np.stack(drow), axis=0)

    print("============================================================")
    print("INFERRED GRID (kept frames only: inside_window=false & cust_gauge>0)")
    print("============================================================")
    print(f"Kept frames: {kept_count}")
    print(f"Stable points used: {len(pts)} (stable_eps={args.stable_eps}px)")
    print(f"Mean +1 col delta (px): dx={mean_dcol[0]:.2f}, dy={mean_dcol[1]:.2f}")
    print(f"Mean +1 row delta (px): dx={mean_drow[0]:.2f}, dy={mean_drow[1]:.2f}")
    print("Interpretation:")
    print("  If +col has dx>0 => right increases x. If +row has dy>0 => down increases y.")
    print()

    # Print snapshot
    print("============================================================")
    print("GRID OWNER / TILE SNAPSHOT (first frame where len==18)")
    print("============================================================")
    if snap_i is None or owners is None or tiles is None:
        print("No frame found with grid_owner_state/grid_state of length 18.")
    else:
        print(f"Found at line/frame index: {snap_i}")
        print("Owners (r0..r2): 0=player side, 1=enemy side")
        for r in range(ROWS):
            rowv = owners[r * COLS : (r + 1) * COLS]
            print("  " + " ".join(str(x) for x in rowv))
        print("Tiles (r0..r2): tile type id")
        for r in range(ROWS):
            rowv = tiles[r * COLS : (r + 1) * COLS]
            print("  " + " ".join(str(x) for x in rowv))
    print()

    # Inferred cell per frame (None if skipped or too far)
    inferred: List[Optional[Tuple[int, int]]] = [None] * len(frames)
    for i, fr in enumerate(frames):
        if not keep_frame(fr):
            continue
        p = fr.get("player_pos")
        if not (isinstance(p, list) and len(p) == 2):
            continue
        pos = np.array([float(p[0]), float(p[1])], dtype=np.float64)
        rc, dist = nearest_cell(pos, grid)
        if dist <= args.assign_max_dist:
            inferred[i] = rc

    # Movement events (ONLY kept frames; reset edge state when skipping)
    events: List[MoveEvent] = []
    last_pressed = {k: False for k in DPAD_KEYS}

    def find_move(i0: int, start_rc: Tuple[int, int], key: str) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[bool], str]:
        expect = DIRS[key]
        for j in range(i0 + 1, min(len(frames), i0 + 1 + args.window)):
            if not keep_frame(frames[j]):
                continue
            rcj = inferred[j]
            if rcj is None:
                continue
            if rcj != start_rc:
                moved = (rcj[0] - start_rc[0], rcj[1] - start_rc[1])
                if abs(moved[0]) + abs(moved[1]) != 1:
                    continue
                matched = moved == expect
                return (j - i0, rcj, matched, "")
        return (None, None, None, "no_adjacent_move_in_window")

    for i, fr in enumerate(frames):
        if not keep_frame(fr):
            for k in DPAD_KEYS:
                last_pressed[k] = False
            continue

        rc0 = inferred[i]
        for k in DPAD_KEYS:
            now = is_pressed(fr.get(k, 0.0))
            if now and not last_pressed[k]:
                if rc0 is None:
                    events.append(
                        MoveEvent(
                            frame_idx=int(fr.get("frame_idx", i)),
                            tick=int(fr.get("tick", i)),
                            key=k,
                            start_rc=(-1, -1),
                            end_rc=None,
                            dt_frames=None,
                            matched_dir=None,
                            note="start_cell_unknown",
                        )
                    )
                else:
                    dt, rc1, matched, note = find_move(i, rc0, k)
                    events.append(
                        MoveEvent(
                            frame_idx=int(fr.get("frame_idx", i)),
                            tick=int(fr.get("tick", i)),
                            key=k,
                            start_rc=rc0,
                            end_rc=rc1,
                            dt_frames=dt,
                            matched_dir=matched,
                            note=note,
                        )
                    )
            last_pressed[k] = now

    # Summaries
    per_dir_lat: Dict[str, List[int]] = {k: [] for k in DPAD_KEYS}
    per_dir_mismatch: Dict[str, int] = {k: 0 for k in DPAD_KEYS}
    per_dir_nomove: Dict[str, int] = {k: 0 for k in DPAD_KEYS}
    per_dir_seen: Dict[str, int] = {k: 0 for k in DPAD_KEYS}

    mismatches: List[MoveEvent] = []
    nomoves: List[MoveEvent] = []

    for ev in events:
        if ev.start_rc == (-1, -1):
            continue
        per_dir_seen[ev.key] += 1
        if ev.dt_frames is None:
            per_dir_nomove[ev.key] += 1
            nomoves.append(ev)
        else:
            per_dir_lat[ev.key].append(ev.dt_frames)
            if ev.matched_dir is False:
                per_dir_mismatch[ev.key] += 1
                mismatches.append(ev)

    print("============================================================")
    print("MOVEMENT LATENCY (kept frames only)")
    print("============================================================")
    total_seen = sum(per_dir_seen.values())
    total_nomove = sum(per_dir_nomove.values())
    total_mismatch = sum(per_dir_mismatch.values())
    print(f"Press edges (with known start cell): {total_seen}")
    print(f"No adjacent move within window({args.window}): {total_nomove}")
    print(f"Adjacent move but wrong direction: {total_mismatch}")
    print()

    for k in DPAD_KEYS:
        print(f"{k}: seen={per_dir_seen[k]}  no_move={per_dir_nomove[k]}  mismatched={per_dir_mismatch[k]}")
        print(f"  latency: {summarize_int(per_dir_lat[k])}")
    print()

    def dump(title: str, lst: List[MoveEvent]) -> None:
        if not lst:
            return
        print("============================================================")
        print(title)
        print("============================================================")
        for ev in lst[: args.examples]:
            s = f"frame={ev.frame_idx} tick={ev.tick} {ev.key} start={ev.start_rc}"
            if ev.end_rc is None:
                s += f" -> end=? dt=? ({ev.note})"
            else:
                s += f" -> end={ev.end_rc} dt={ev.dt_frames} matched={ev.matched_dir} ({ev.note})"
            print(s)
        print()

    dump("EXAMPLES: direction mismatches (adjacent moves)", mismatches)
    dump("EXAMPLES: no adjacent move within window", nomoves)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
