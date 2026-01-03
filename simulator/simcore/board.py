# mmbn_sim/simcore/board.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .constants import COLS, N, OWNER_P1, OWNER_P2, ROWS, TILE_BASIC
from .coords import rc_to_idx


def default_owners() -> List[int]:
    # Canonical: left 3 cols = P1-owned, right 3 cols = P2-owned
    owners: List[int] = []
    for r in range(ROWS):
        for c in range(COLS):
            owners.append(OWNER_P1 if c < 3 else OWNER_P2)
    return owners


def default_tiles() -> List[int]:
    return [TILE_BASIC] * N


def _mirror_rowwise(values: List[int]) -> List[int]:
    if len(values) != N:
        raise ValueError("expected len N")
    out = [0] * N
    for r in range(ROWS):
        for c in range(COLS):
            src = rc_to_idx(r, c)
            dst = rc_to_idx(r, (COLS - 1) - c)
            out[dst] = values[src]
    return out


def mirror_tiles(values: List[int]) -> List[int]:
    """Tiles flip horizontally; no semantic inversion."""
    return _mirror_rowwise(values)


def mirror_owners(values: List[int]) -> List[int]:
    """Owners flip horizontally AND invert (P1<->P2)."""
    flipped = _mirror_rowwise(values)
    out = [0] * N
    for i, v in enumerate(flipped):
        if v == OWNER_P1:
            out[i] = OWNER_P2
        elif v == OWNER_P2:
            out[i] = OWNER_P1
        else:
            out[i] = v
    return out


@dataclass(frozen=True)
class Board:
    owners: List[int]
    tiles: List[int]

    @staticmethod
    def new() -> "Board":
        return Board(owners=default_owners(), tiles=default_tiles())

    def validate(self) -> None:
        if len(self.owners) != N:
            raise ValueError(f"owners must be len {N}")
        if len(self.tiles) != N:
            raise ValueError(f"tiles must be len {N}")
