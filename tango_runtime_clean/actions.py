from __future__ import annotations
from typing import Dict, List

KEY_BIT_POSITIONS: Dict[str, int] = {
    "A": 8, "DOWN": 7, "UP": 6, "LEFT": 5, "RIGHT": 4,
    "RETURN": 3, "X": 1, "Z": 0,
}

DISCRETE_ACTIONS: List[str] = [
    "NO_OP",
    "UP", "DOWN", "LEFT", "RIGHT",
    "X", "Z",
    "HOLD_X", "HOLD_Z",
    "RELEASE_X", "RELEASE_Z",
]

def int_to_binary_string(value: int) -> str:
    return format(value, "016b")

def map_discrete_to_mask(action_index: int,
                         discrete_actions: List[str] = DISCRETE_ACTIONS,
                         key_bits: Dict[str, int] = KEY_BIT_POSITIONS) -> int:
    if not (0 <= action_index < len(discrete_actions)):
        return 0
    name = discrete_actions[action_index]
    kb = key_bits
    mask = 0
    if name == "UP":        mask |= 1 << kb["UP"]
    elif name == "DOWN":    mask |= 1 << kb["DOWN"]
    elif name == "LEFT":    mask |= 1 << kb["LEFT"]
    elif name == "RIGHT":   mask |= 1 << kb["RIGHT"]
    elif name == "X":       mask |= 1 << kb["X"]
    elif name == "Z":       mask |= 1 << kb["Z"]
    elif name == "HOLD_X":  mask |= 1 << kb["X"]
    elif name == "HOLD_Z":  mask |= 1 << kb["Z"]
    return mask
