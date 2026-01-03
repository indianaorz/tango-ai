# mmbn_sim/simcore/serialize.py
from __future__ import annotations

from typing import Any, Dict, List, Literal

from .actions import ActionId
from .board import mirror_owners, mirror_tiles
from .coords import idx_to_rc, mirror_idx
from .state import ActorId, GameState, other


ViewActorLabel = Literal["P", "E"]


def _view_map_idx(viewer: ActorId, idx_canon: int) -> int:
    return idx_canon if viewer == "P1" else mirror_idx(idx_canon)


def _view_map_dir_local(_viewer: ActorId, dir_local: str | None) -> str | None:
    # Stored as local dirs already, so no transform needed.
    return dir_local


def _label_for(viewer: ActorId, actor: ActorId) -> ViewActorLabel:
    # In each view, local actor is "P" and opponent is "E".
    return "P" if actor == viewer else "E"


def view_state(st: GameState, viewer: ActorId) -> Dict[str, Any]:
    # Board for viewer
    if viewer == "P1":
        owners = st.board.owners
        tiles = st.board.tiles
    else:
        owners = mirror_owners(st.board.owners)
        tiles = mirror_tiles(st.board.tiles)

    # Actor mapping
    local = viewer
    enemy = other(viewer)

    p = st.actors[local]
    e = st.actors[enemy]

    p_idx_v = _view_map_idx(viewer, p.idx)
    e_idx_v = _view_map_idx(viewer, e.idx)

    pr, pc = idx_to_rc(p_idx_v)
    er, ec = idx_to_rc(e_idx_v)

    is_locked = p.is_locked(st.cust)
    lock_remaining = max(0, p.locked_until - st.cust)

    # Shot lines for this viewer (multiple possible now)
    shot_lines: List[Dict[str, Any]] = []
    for sl in st.shot_lines:
        if not sl.alive(st.cust):
            continue
        shot_lines.append(
            {
                "actor": _label_for(viewer, sl.actor),  # "P" or "E"
                "kind": sl.kind,
                "from_idx": _view_map_idx(viewer, sl.from_idx),
                "to_idx": _view_map_idx(viewer, sl.to_idx),
                "expires_cust": sl.expires_cust,
            }
        )

    return {
        "viewer": viewer,
        "cust": st.cust,
        "grid_owner_state": owners,
        "grid_state": tiles,
        # Local/Enemy entity fields (in viewer space)
        "p_hp": p.hp,
        "e_hp": e.hp,
        "p_idx": p_idx_v,
        "e_idx": e_idx_v,
        "p_rc": [pr, pc],
        "e_rc": [er, ec],
        # Pending shown for local only (what that player is choosing)
        "pending_action": p.pending_action,
        "is_locked": is_locked,
        "lock_remaining": lock_remaining,
        # Charge shown for local + enemy (useful for debugging)
        "p_charge_level": p.charge.level,
        "p_charge_hold": p.charge.hold,
        "p_charge_progress": p.charge.progress,
        "e_charge_level": e.charge.level,
        "e_charge_hold": e.charge.hold,
        "e_charge_progress": e.charge.progress,
        "charge_full_at": p.charge.FULL_AT,
        # Visual (local dirs already)
        "p_lean_dir": _view_map_dir_local(viewer, p.lean_dir_local),
        "e_lean_dir": _view_map_dir_local(viewer, e.lean_dir_local),
        "p_entry_dir": _view_map_dir_local(viewer, p.entry_dir_local),
        "e_entry_dir": _view_map_dir_local(viewer, e.entry_dir_local),
        "p_is_entering": (p.entry_dir_local is not None) and (st.cust < p.entry_until),
        "e_is_entering": (e.entry_dir_local is not None) and (st.cust < e.entry_until),
        "shot_lines": shot_lines,
        # Debug
        "last_action_started": st.last_action_started,
        "last_events": st.last_events,
        # Canonical (minimal) for debugging
        "canon": {
            "p1_idx": st.actors["P1"].idx,
            "p2_idx": st.actors["P2"].idx,
        },
    }
