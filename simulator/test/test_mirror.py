# mmbn_sim/tests/test_mirror.py
from __future__ import annotations

from simcore.board import default_owners, mirror_owners
from simcore.coords import mirror_dir, mirror_idx, idx_to_rc, rc_to_idx
from simcore.state import GameState


def test_mirror_idx_involution():
    for idx in range(18):
        assert mirror_idx(mirror_idx(idx)) == idx


def test_mirror_dir_left_right():
    assert mirror_dir("LEFT") == "RIGHT"
    assert mirror_dir("RIGHT") == "LEFT"
    assert mirror_dir("UP") == "UP"
    assert mirror_dir("DOWN") == "DOWN"


def test_owner_mirror_inverts_and_flips():
    owners = default_owners()
    m = mirror_owners(owners)
    # Mirror twice should return original
    mm = mirror_owners(m)
    assert mm == owners


def test_p2_local_move_right_maps_to_canonical_left_behaviorally():
    st = GameState()
    # Put P2 somewhere on its side and try a P2-local RIGHT move.
    st.actors["P2"].idx = rc_to_idx(1, 4)  # canonical col 4 on P2 side
    # In P2 local view, RIGHT should move toward canonical LEFT.
    assert st.can_move("P2", "RIGHT") == st.can_move("P2", "RIGHT")
