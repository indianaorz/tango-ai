# mmbn_sim/simcore/actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Damage constants (phase 0)
DMG_BUSTER = 5
DMG_CHARGE = 50


ActionId = str


@dataclass(frozen=True)
class EventSpec:
    type: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ActionSpec:
    lock_cust: int
    timeline: List[Tuple[int, EventSpec]]


# Control actions (not in ACTIONS dict)
ACT_NOOP: ActionId = "NOOP"
ACT_HOLD_ON: ActionId = "HOLD_ON"
ACT_HOLD_OFF: ActionId = "HOLD_OFF"

# Gameplay actions are defined in *local-player space*.
# The engine transforms LEFT/RIGHT + "forward shot" based on which actor is acting.
ACTIONS: Dict[ActionId, ActionSpec] = {
    "MOVE_UP": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"dir": "UP"})),
            (1, EventSpec("MOVE_APPLY", {"dir": "UP"})),
        ],
    ),
    "MOVE_DOWN": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"dir": "DOWN"})),
            (1, EventSpec("MOVE_APPLY", {"dir": "DOWN"})),
        ],
    ),
    "MOVE_LEFT": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"dir": "LEFT"})),
            (1, EventSpec("MOVE_APPLY", {"dir": "LEFT"})),
        ],
    ),
    "MOVE_RIGHT": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"dir": "RIGHT"})),
            (1, EventSpec("MOVE_APPLY", {"dir": "RIGHT"})),
        ],
    ),
    "SHOOT": ActionSpec(
        lock_cust=2,
        timeline=[
            (
                0,
                EventSpec(
                    "RAY_SHOT",
                    {"dmg": DMG_BUSTER, "reset_charge": True, "shot_kind": "buster"},
                ),
            )
        ],
    ),
    "RELEASE_CHARGE": ActionSpec(
        lock_cust=5,
        timeline=[
            (
                0,
                EventSpec(
                    "RAY_SHOT",
                    {"dmg": DMG_CHARGE, "reset_charge": True, "shot_kind": "charge"},
                ),
            )
        ],
    ),
}
