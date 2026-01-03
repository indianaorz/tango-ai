# mmbn_sim/tests/test_legal_actions.py
from __future__ import annotations

from simcore.state import GameState


def test_legal_actions_locked_only_controls():
    st = GameState()
    p1 = st.actors["P1"]
    p1.locked_until = st.cust + 10

    acts = set(st.legal_action_ids("P1"))
    assert "NOOP" in acts
    assert "MOVE_UP" not in acts
    assert "SHOOT" not in acts


def test_legal_actions_disallow_shoot_at_full_charge():
    st = GameState()
    p1 = st.actors["P1"]
    p1.charge.progress = p1.charge.FULL_AT

    acts = set(st.legal_action_ids("P1"))
    assert "SHOOT" not in acts
    assert "RELEASE_CHARGE" in acts
