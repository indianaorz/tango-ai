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
        from nitrogen.shared import BUTTON_ACTION_TOKENS as _TOK  # type: ignore
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
            "RIGHT_BOTTOM",
            "RIGHT_LEFT",
            "RIGHT_RIGHT",
            "RIGHT_SHOULDER",
            "RIGHT_THUMB",
            "RIGHT_TRIGGER",
            "RIGHT_UP",
            "SOUTH",
            "START",
            "WEST",
        ]


# Public, canonical button order
BUTTON_TOKENS: List[str] = get_button_tokens()

# Cache-space action vector layout (what we store in .pt):
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
#
# IMPORTANT:
# - This template is only for dataset JSONL/export tooling (convert_dataset.py).
# - It is NOT the cache tensor layout. Cache tensor layout is always:
#     [4 axes] + BUTTON_TOKENS
# -----------------------------------------------------------------------------
def _axis_list(v: float = 0.0) -> List[float]:
    return [float(v)]


def build_nitrogen_template() -> OrderedDict:
    """
    Build a row with:
      - all BUTTON_TOKENS as scalar floats (0.0)
      - the 4 axes as list-wrapped floats [0.0]
    """
    row: OrderedDict = OrderedDict()

    # Buttons (scalar floats)
    for b in BUTTON_TOKENS:
        row[b] = 0.0

    # Axes (list-wrapped floats)
    row["AXIS_LEFTX"] = _axis_list(0.0)
    row["AXIS_LEFTY"] = _axis_list(0.0)
    row["AXIS_RIGHTX"] = _axis_list(0.0)
    row["AXIS_RIGHTY"] = _axis_list(0.0)

    # Optional triggers if present in BUTTON_TOKENS: Nitrogen sometimes treats these as axes-like.
    # We keep them scalar in this JSON template unless you explicitly want list-wrapped.
    # (Your precache_dataset.py already normalizes triggers specially if they exist.)
    return row


NITROGEN_TEMPLATE: OrderedDict = build_nitrogen_template()
