# mmbn_sim/tests/test_plan.py
from __future__ import annotations

from simcore.tree import TreeStore


def test_plan_creates_64_steps_and_restores_current():
    t = TreeStore()
    base = t.current_id
    out = t.plan_run(horizon=64, iterations=10, max_depth=5, seed=0)
    assert out["steps"] == 64
    assert t.plan is not None
    assert len(t.plan.steps) == 64
    # plan_run restores current id
    assert t.current_id == base


def test_replay_walks_forward_and_stops():
    t = TreeStore()
    t.plan_run(horizon=8, iterations=5, max_depth=4, seed=0)
    t.replay_start()
    assert t.replay.active is True
    for _ in range(8):
        t.replay_step(1)
    # after finishing, replay becomes inactive
    assert t.replay.active is False
