# mmbn_sim/simcore/coords.py
from __future__ import annotations

from typing import Tuple

from .constants import COLS, ROWS


def rc_to_idx(r: int, c: int) -> int:
    return r * COLS + c


def idx_to_rc(idx: int) -> Tuple[int, int]:
    return (idx // COLS, idx % COLS)


def mirror_idx(idx: int) -> int:
    """Flip columns: (r,c) -> (r, COLS-1-c)."""
    r, c = idx_to_rc(idx)
    return rc_to_idx(r, (COLS - 1) - c)


def mirror_dir(d: str) -> str:
    """Flip LEFT<->RIGHT for P2-local actions into canonical."""
    if d == "LEFT":
        return "RIGHT"
    if d == "RIGHT":
        return "LEFT"
    return d
