# mmbn_sim/tests/test_charge_patterns.py
from __future__ import annotations

from simcore.forms import pattern_target_indices
from simcore.coords import rc_to_idx


def test_fire_pattern_p1_targets_forward_row():
    # P at (1,1) -> X at (1,2)(1,3)(1,4) in canonical for P1
    actor = rc_to_idx(1, 1)
    pattern = [["P", "X", "X", "X"]]
    out = pattern_target_indices(actor, actor_is_p1=True, pattern=pattern)
    assert out == [rc_to_idx(1, 2), rc_to_idx(1, 3), rc_to_idx(1, 4)]


def test_fire_pattern_p2_mirrors_forward_left():
    # P2 forward is canonical LEFT: (1,4) -> (1,3)(1,2)(1,1)
    actor = rc_to_idx(1, 4)
    pattern = [["P", "X", "X", "X"]]
    out = pattern_target_indices(actor, actor_is_p1=False, pattern=pattern)
    assert out == [rc_to_idx(1, 3), rc_to_idx(1, 2), rc_to_idx(1, 1)]
