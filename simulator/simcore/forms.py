# mmbn_sim/simcore/forms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import COLS, ROWS
from .coords import idx_to_rc, rc_to_idx


GREGAR_CROSSES = [
    "Fire",
    "Elec",
    "Slash",
    "Erase",
    "Charge",
]

FALZAR_CROSSES = [
    "Aqua",
    "Thawk",
    "Tengu",
    "Grnd",
    "Dust",
]

FORM_MAPPING = [
    {"type": "Normal", "normal": 0, "beast": 11},
    {"type": "Fire", "normal": 1, "beast": 13},
    {"type": "Elec", "normal": 2, "beast": 14},
    {"type": "Slash", "normal": 3, "beast": 15},
    {"type": "Erase", "normal": 4, "beast": 16},
    {"type": "Charge", "normal": 5, "beast": 17},
    {"type": "Aqua", "normal": 6, "beast": 18},
    {"type": "Thawk", "normal": 7, "beast": 19},
    {"type": "Tengu", "normal": 8, "beast": 20},
    {"type": "Grnd", "normal": 9, "beast": 21},
    {"type": "Dust", "normal": 10, "beast": 22},
]


# Only "basic crosses" (normal forms) are used by default right now.
# We still keep the full dict for future expansion, but the simulator only uses:
#  - MatchY
#  - Pattern
_form_charges: Dict[int, Dict[str, object]] = {
    0: {"name": "Normal", "range": ["MatchY"]},
    1: {
        "name": "Fire",
        "range": ["Pattern"],
        "pattern": [
            ["P", "X", "X", "X"],
        ],
    },
    2: {"name": "Elec", "range": ["MatchY"]},
    3: {
        "name": "Slash",
        "range": ["Pattern"],
        "pattern": [
            ["0", "X", "X"],
            ["P", "X", "X"],
            ["0", "X", "X"],
        ],
    },
    4: {"name": "Erase", "range": ["MatchY"]},
    5: {
        "name": "Charge",
        "range": ["Pattern"],
        "pattern": [
            ["P", "X", "X", "X"],
        ],
    },
    6: {"name": "Aqua", "range": ["MatchY"]},
    7: {
        "name": "Thawk",
        "range": ["Pattern"],
        "pattern": [
            ["0", "X", "X"],
            ["P", "X", "X"],
            ["0", "X", "X"],
        ],
    },
    8: {
        "name": "Tengu",
        "range": ["Pattern"],
        "pattern": [
            ["0", "X"],
            ["P", "X"],
            ["0", "X"],
        ],
    },
    9: {"name": "Grnd", "range": ["MatchY"]},
    10: {"name": "Dust", "range": ["MatchY"]},
    # beasts retained for later (not used right now)
    11: {"name": "Beast", "range": ["Any"]},
    12: {"name": "Beast", "range": ["Any"]},
    13: {
        "name": "FireBeast",
        "range": ["Pattern"],
        "pattern": [
            ["0", "0", "X", "X"],
            ["P", "X", "X", "X"],
            ["0", "0", "X", "X"],
        ],
    },
    14: {
        "name": "ElecBeast",
        "range": ["Pattern"],
        "pattern": [
            ["0", "0", "X", "X"],
            ["P", "X", "X", "X"],
            ["0", "0", "X", "X"],
        ],
    },
    15: {
        "name": "SlashBeast",
        "range": ["Pattern"],
        "pattern": [
            ["0", "X", "0", "X"],
            ["P", "0", "X", "0"],
            ["0", "X", "0", "X"],
        ],
    },
    16: {"name": "EraseBeast", "range": ["Any"]},
    17: {"name": "ChargeBeast", "range": ["MatchY"]},
    19: {
        "name": "ThawkBeast",
        "range": ["Pattern"],
        "pattern": [
            ["X", "X", "X", "X", "X", "X"],
            ["0", "0", "0", "0", "0", "X"],
            ["X", "X", "X", "X", "X", "X"],
        ],
    },
    20: {
        "name": "TenguBeast",
        "range": ["Pattern"],
        "pattern": [
            ["0", "0", "0", "X"],
            ["P", "X", "X", "X"],
            ["0", "0", "0", "X"],
        ],
    },
    21: {"name": "GrndBeast", "range": ["MatchY"]},
    22: {"name": "DustBeast", "range": ["Any"]},
}


@dataclass(frozen=True)
class FormChargeSpec:
    form_id: int
    name: str
    ranges: Tuple[str, ...]
    pattern: Optional[Tuple[Tuple[str, ...], ...]] = None


def form_name(form_id: int) -> str:
    d = _form_charges.get(int(form_id))
    if not d:
        return f"Unknown({form_id})"
    return str(d.get("name", f"Unknown({form_id})"))


def form_charge_spec(form_id: int) -> FormChargeSpec:
    form_id = int(form_id)
    d = _form_charges.get(form_id)
    if not d:
        # Safe fallback: treat unknown as normal MatchY ray.
        return FormChargeSpec(form_id=form_id, name=f"Unknown({form_id})", ranges=("MatchY",), pattern=None)

    ranges_raw = d.get("range", ["MatchY"])
    ranges = tuple(str(x) for x in (ranges_raw if isinstance(ranges_raw, list) else [ranges_raw]))

    pat_raw = d.get("pattern", None)
    if pat_raw is None:
        return FormChargeSpec(form_id=form_id, name=str(d.get("name", "Unknown")), ranges=ranges, pattern=None)

    if not isinstance(pat_raw, list):
        raise ValueError("pattern must be a list of rows")

    rows: List[Tuple[str, ...]] = []
    for r in pat_raw:
        if not isinstance(r, list):
            raise ValueError("pattern rows must be lists")
        rows.append(tuple(str(x) for x in r))

    return FormChargeSpec(
        form_id=form_id,
        name=str(d.get("name", "Unknown")),
        ranges=ranges,
        pattern=tuple(rows),
    )


def pattern_offsets(pattern: Sequence[Sequence[str]]) -> List[Tuple[int, int]]:
    """
    Convert a local-space pattern into (dr, dc_forward) offsets for each 'X',
    with P treated as origin (0,0).

    - dr: row delta (down positive)
    - dc_forward: forward delta (right positive in local view)
    """
    pr = pc = None
    for r, row in enumerate(pattern):
        for c, cell in enumerate(row):
            if str(cell) == "P":
                pr, pc = r, c
                break
        if pr is not None:
            break

    if pr is None or pc is None:
        raise ValueError("pattern must contain a single 'P' origin")

    out: List[Tuple[int, int]] = []
    for r, row in enumerate(pattern):
        for c, cell in enumerate(row):
            if str(cell) == "X":
                out.append((r - pr, c - pc))

    return out


def pattern_target_indices(
    actor_idx_canon: int,
    actor_is_p1: bool,
    pattern: Sequence[Sequence[str]],
) -> List[int]:
    """
    Returns canonical indices affected by a Pattern charge shot.
    Pattern is defined in *local-player space* (forward = right).
    For P2, forward is canonical LEFT, so dc is mirrored by sign.
    """
    ar, ac = idx_to_rc(actor_idx_canon)
    fsign = +1 if actor_is_p1 else -1

    out: List[int] = []
    for dr, dc_fwd in pattern_offsets(pattern):
        rr = ar + int(dr)
        cc = ac + int(dc_fwd) * fsign
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            out.append(rc_to_idx(rr, cc))

    # unique + stable order
    seen = set()
    uniq: List[int] = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    return uniq
