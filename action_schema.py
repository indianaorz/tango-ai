# action_schema.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

# -----------------------------------------------------------------------------
# Canonical Nitrogen button token order (single source of truth)
# -----------------------------------------------------------------------------
def get_button_tokens() -> List[str]:
    """
    The ONLY valid button ordering is whatever Nitrogen uses internally.
    We import it when available; otherwise we fall back to the known canonical list.
    """
    try:
        from nitrogen.shared import BUTTON_ACTION_TOKENS as _TOK
        return list(_TOK)
    except Exception:
        # Canonical fallback (keep identical everywhere)
        return [
            "BACK",
            "DPAD_DOWN",
            "DPAD_LEFT",
            "DPAD_RIGHT",
            "DPAD_UP",
            "EAST",
            "GUIDE",
            "LEFT_SHOULDER",
            "LEFT_THUMB",
            "LEFT_TRIGGER",
            "NORTH",
            "RIGHT_SHOULDER",
            "RIGHT_THUMB",
            "RIGHT_TRIGGER",
            "SOUTH",
            "START",
            "WEST",
            "RIGHT_BOTTOM",
            "RIGHT_LEFT",
            "RIGHT_RIGHT",
            "RIGHT_UP",
        ]


BUTTON_TOKENS: List[str] = get_button_tokens()

# NitroGen action vector layout is always:
#   [AXIS_LEFTX, AXIS_LEFTY, AXIS_RIGHTX, AXIS_RIGHTY, ...buttons in BUTTON_TOKENS order...]
AXES: List[str] = ["AXIS_LEFTX", "AXIS_LEFTY", "AXIS_RIGHTX", "AXIS_RIGHTY"]
ACTION_DIM: int = 4 + len(BUTTON_TOKENS)

# -----------------------------------------------------------------------------
# GBA subset that we *use* (others remain 0, but still exist in the vector)
# -----------------------------------------------------------------------------
GBA_UI_BUTTONS: List[str] = [
    "DPAD_UP",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "EAST",           # GBA A
    "SOUTH",          # GBA B
    "LEFT_SHOULDER",  # GBA L
    "RIGHT_SHOULDER", # GBA R
    "BACK",           # GBA SELECT
    "START",          # GBA START
]

# -----------------------------------------------------------------------------
# Conversion mappings
# -----------------------------------------------------------------------------
GBA_BITS: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "SELECT",
    3: "START",
    4: "RIGHT",
    5: "LEFT",
    6: "UP",
    7: "DOWN",
    8: "R",
    9: "L",
}

GBA_TO_NITROGEN: Dict[str, str] = {
    "A": "EAST",
    "B": "SOUTH",
    "L": "LEFT_SHOULDER",
    "R": "RIGHT_SHOULDER",
    "START": "START",
    "SELECT": "BACK",
    "UP": "DPAD_UP",
    "DOWN": "DPAD_DOWN",
    "LEFT": "DPAD_LEFT",
    "RIGHT": "DPAD_RIGHT",
}

# -----------------------------------------------------------------------------
# JSONL row template: keep same fields/types NitroGen tooling expects
# (axes are list-wrapped like Nitrogen debug dumps; buttons are float scalars)
# -----------------------------------------------------------------------------
NITROGEN_TEMPLATE = OrderedDict([
    ("WEST", 0.0), ("SOUTH", 0.0), ("BACK", 0.0),
    ("DPAD_DOWN", 0.0), ("DPAD_LEFT", 0.0), ("DPAD_RIGHT", 0.0), ("DPAD_UP", 0.0),
    ("GUIDE", 0.0), ("AXIS_LEFTX", [0.0]), ("AXIS_LEFTY", [0.0]),
    ("LEFT_SHOULDER", 0.0), ("LEFT_TRIGGER", [0.0]),
    ("AXIS_RIGHTX", [0.0]), ("AXIS_RIGHTY", [0.0]),
    ("LEFT_THUMB", 0.0), ("RIGHT_THUMB", 0.0),
    ("RIGHT_SHOULDER", 0.0), ("RIGHT_TRIGGER", [0.0]),
    ("START", 0.0), ("EAST", 0.0), ("NORTH", 0.0),
])
