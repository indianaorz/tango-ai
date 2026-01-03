# mmbn_sim/tests/test_plan.py
from __future__ import annotations

from simcore.tree import TreeStore


def test_plan_creates_64_steps_and_restores_current():
    t = TreeStore()
    base = t.current_id

    plan = t.plan_to_cust(
        target_cust=64,
        iters_per_step=10,
        lookahead_depth=5,
        seed=0,
        time_penalty=0.0,
        jitter_eps=0.0,
    )

    assert plan.target_cust == 64
    assert t.plan is not None
    assert len(t.plan.steps) == 64

    # planning restores current id
    assert t.current_id == base


def test_replay_walks_forward_and_stops():
    t = TreeStore()
    t.plan_to_cust(
        target_cust=8,
        iters_per_step=5,
        lookahead_depth=4,
        seed=0,
        time_penalty=0.0,
        jitter_eps=0.0,
    )

    assert t.plan is not None
    assert len(t.plan.steps) == 8

    t.replay_reset()
    assert t.current_id == t.plan.base_node_id

    # Walk all the way forward
    for i in range(8):
        t.replay_step()
        assert t.plan_replay_index == i + 1

    # Now we're at the last step node
    assert t.current_id == t.plan.steps[-1].node_id

    # Further steps should no-op
    last = t.current_id
    t.replay_step()
    assert t.current_id == last
    assert t.plan_replay_index == 8
