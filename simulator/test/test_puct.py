# mmbn_sim/tests/test_puct.py
from __future__ import annotations

from simcore.mcts import MCTSStatsStore, list_joint_actions, _select_joint_puct
from simcore.state import GameState


def test_puct_prefers_high_prior_at_start():
    st = GameState()

    # Make P1 full charge so RELEASE_CHARGE should get very high prior.
    p1 = st.actors["P1"]
    p1.charge.progress = p1.charge.FULL_AT

    stats = MCTSStatsStore()
    node_id = "root"

    # Seed parent visits so sqrt(N_parent) behaves like "early search"
    stats.node_stats(node_id).n = 1

    joints = list_joint_actions(st)
    chosen = _select_joint_puct(stats, node_id, st, joints, c_puct=1.25)

    # We don't require P2 side exact action, but P1 should be releasing.
    assert chosen.startswith("P1:RELEASE_CHARGE|")
