# mmbn_sim/simcore/constants.py
from __future__ import annotations

ROWS = 3
COLS = 6
N = ROWS * COLS

OWNER_P1 = 0  # left side in canonical
OWNER_P2 = 1  # right side in canonical

# Tile indices (match your capture conventions)
TILE_HOLE_PERM = 0
TILE_BASIC = 2

DIR_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def opposite_dir(d: str) -> str:
    return {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(d, d)
