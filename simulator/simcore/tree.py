# mmbn_sim/simcore/tree.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .actions import ACTIONS, ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP, ActionId
from .mcts import (
    MCTSConfig,
    MCTSStatsStore,
    fmt_joint,
    parse_joint,
    run_mcts,
    recommend_root_maximin,
)
from .state import ActorId, GameState

JointActionId = str


@dataclass
class TreeNode:
    node_id: str
    parent_id: Optional[str]
    joint_action_from_parent: Optional[JointActionId]
    depth: int
    state: GameState  # immutable snapshot
    children: Dict[JointActionId, str]

    def summary(self) -> Dict[str, Any]:
        p1 = self.state.actors["P1"]
        p2 = self.state.actors["P2"]
        return {
            "id": self.node_id,
            "parent": self.parent_id,
            "action": self.joint_action_from_parent,
            "depth": self.depth,
            "children_count": len(self.children),
            "s": {
                "cust": self.state.cust,
                "p1_hp": p1.hp,
                "p2_hp": p2.hp,
                "p1_locked": p1.is_locked(self.state.cust),
                "p2_locked": p2.is_locked(self.state.cust),
            },
        }


@dataclass
class PlanStep:
    node_id: str
    joint: JointActionId
    cust_after: int


@dataclass
class PlanResult:
    base_node_id: str
    target_cust: int
    steps: List[PlanStep]

    def to_json(self) -> Dict[str, Any]:
        return {
            "base_node_id": self.base_node_id,
            "target_cust": self.target_cust,
            "steps": [
                {"node_id": s.node_id, "joint": s.joint, "cust_after": s.cust_after}
                for s in self.steps
            ],
        }


class TreeStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: str = ""
        self.current_id: str = ""
        self._next_id: int = 1

        self.cursor_base_id: str = ""
        self.cursor: GameState = GameState()

        self.mcts = MCTSStatsStore()

        # Planning / replay
        self.plan: Optional[PlanResult] = None
        self.plan_replay_index: int = 0  # 0 means "at base"

        self.reset()

    def reset(self) -> None:
        self.nodes.clear()
        self._next_id = 1
        self.mcts.reset()
        self.plan = None
        self.plan_replay_index = 0

        root_state = GameState()
        root = TreeNode(
            node_id=self._alloc_id(),
            parent_id=None,
            joint_action_from_parent=None,
            depth=0,
            state=root_state,
            children={},
        )
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        self.current_id = root.node_id
        self._reset_cursor_from_current()

    def _alloc_id(self) -> str:
        nid = f"n{self._next_id}"
        self._next_id += 1
        return nid

    def _reset_cursor_from_current(self) -> None:
        self.cursor_base_id = self.current_id
        self.cursor = copy.deepcopy(self.nodes[self.current_id].state)

    def set_current(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError("unknown node_id")
        self.current_id = node_id
        self._reset_cursor_from_current()

    # -------------------------------
    # Cursor API (UI)
    # -------------------------------
    def cursor_set_pending(self, actor: ActorId, action: Optional[ActionId]) -> None:
        st = self.cursor.actors[actor]
        if st.is_locked(self.cursor.cust):
            st.pending_action = None
            self.cursor._log(f"{actor}: locked -> pending ignored")
            return
        st.pending_action = action

    def cursor_apply_control(self, actor: ActorId, action: ActionId) -> None:
        if action == ACT_NOOP:
            self.cursor_set_pending(actor, None)
            return
        if action == ACT_HOLD_ON:
            self.cursor.cursor_set_hold(actor, True)
            self.cursor._log(f"{actor}: cursor HOLD_ON")
            return
        if action == ACT_HOLD_OFF:
            self.cursor.cursor_set_hold(actor, False)
            self.cursor._log(f"{actor}: cursor HOLD_OFF")
            return
        if action in ACTIONS:
            self.cursor_set_pending(actor, action)
            return
        raise ValueError(f"unknown action: {action}")

    # -------------------------------
    # Commit / branching
    # -------------------------------
    def commit_cursor_step(self) -> None:
        if self.cursor_base_id != self.current_id:
            self._reset_cursor_from_current()

        base_node = self.nodes[self.current_id]
        joint = self._derive_committed_joint(self.cursor, base_node.state)

        child_id = self.ensure_child(base_node.node_id, joint)
        self.current_id = child_id
        self._reset_cursor_from_current()

    def _derive_committed_joint(self, cur: GameState, base: GameState) -> JointActionId:
        def derive_for(actor: ActorId) -> ActionId:
            c = cur.actors[actor]
            b = base.actors[actor]

            if c.charge.hold != b.charge.hold:
                if (
                    b.charge.hold
                    and (not c.charge.hold)
                    and b.charge.level == 2
                    and (not b.is_locked(base.cust))
                ):
                    return "RELEASE_CHARGE"
                return ACT_HOLD_ON if c.charge.hold else ACT_HOLD_OFF

            if b.is_locked(base.cust):
                return ACT_NOOP

            return c.pending_action if c.pending_action is not None else ACT_NOOP

        return fmt_joint(derive_for("P1"), derive_for("P2"))

    # -------------------------------
    # Public child creation (used by MCTS + commit)
    # -------------------------------
    def ensure_child(self, parent_id: str, joint: JointActionId) -> str:
        if parent_id not in self.nodes:
            raise ValueError("unknown parent_id")
        node = self.nodes[parent_id]
        if joint in node.children:
            return node.children[joint]

        st = copy.deepcopy(node.state)
        self.apply_joint_to_state(st, joint)
        child = TreeNode(
            node_id=self._alloc_id(),
            parent_id=node.node_id,
            joint_action_from_parent=joint,
            depth=node.depth + 1,
            state=st,
            children={},
        )
        self.nodes[child.node_id] = child
        node.children[joint] = child.node_id
        return child.node_id

    def apply_joint_to_state(self, st: GameState, joint: JointActionId) -> None:
        p1_act, p2_act = parse_joint(joint)

        def apply(actor: ActorId, act: ActionId) -> None:
            a = st.actors[actor]

            if act == ACT_NOOP:
                a.pending_action = None
                return
            if act == ACT_HOLD_ON:
                st.cursor_set_hold(actor, True)
                a.pending_action = None
                return
            if act == ACT_HOLD_OFF:
                st.cursor_set_hold(actor, False)
                a.pending_action = None
                return

            if act == "RELEASE_CHARGE":
                st.cursor_set_hold(actor, False)
                if a.is_locked(st.cust):
                    a.pending_action = None
                    return
                a.pending_action = "RELEASE_CHARGE"
                return

            if a.is_locked(st.cust):
                a.pending_action = None
                return

            if act in ACTIONS:
                a.pending_action = act
                return

            raise ValueError(f"unknown joint action element: {act}")

        apply("P1", p1_act)
        apply("P2", p2_act)
        st.advance_cust()

    # -------------------------------
    # MCTS API
    # -------------------------------
    def run_mcts_from_current(
        self,
        iterations: int,
        max_depth: int,
        seed: int = 0,
        target_cust: Optional[int] = None,
        time_penalty: float = 0.002,
        jitter_eps: float = 1e-4,
    ) -> Dict[str, Any]:
        root_id = self.current_id
        root_state = self.nodes[root_id].state

        cfg = MCTSConfig(
            iterations=int(iterations),
            max_depth=int(max_depth),
            seed=int(seed),
            target_cust=target_cust,
            time_penalty=float(time_penalty),
            jitter_eps=float(jitter_eps),
        )

        def ensure_child_fn(pid: str, j: JointActionId) -> str:
            return self.ensure_child(pid, j)

        def get_state_fn(nid: str) -> GameState:
            return self.nodes[nid].state

        def get_children_fn(nid: str) -> Dict[JointActionId, str]:
            return self.nodes[nid].children

        return run_mcts(
            stats=self.mcts,
            root_id=root_id,
            root_state=root_state,
            ensure_child_fn=ensure_child_fn,
            get_node_state_fn=get_state_fn,
            get_node_children_fn=get_children_fn,
            config=cfg,
        )

    def mcts_summary_current(self) -> Dict[str, Any]:
        root_id = self.current_id
        root_state = self.nodes[root_id].state
        return recommend_root_maximin(self.mcts, root_id, root_state)

    # -------------------------------
    # Planning: produce full plan to target_cust
    # -------------------------------
    def plan_to_cust(
        self,
        target_cust: int = 64,
        iters_per_step: int = 800,
        lookahead_depth: int = 16,
        seed: int = 0,
        time_penalty: float = 0.002,
        jitter_eps: float = 1e-4,
    ) -> PlanResult:
        """
        Generates an entire sequence of joint actions from current node until target_cust.

        This is *receding-horizon planning*:
          for each cust:
            - run MCTS with lookahead toward target_cust
            - pick recommended (P1 maximin, P2 minimax)
            - commit that joint and continue

        Result is deterministic given the seed and current tree state.
        """
        target_cust = int(target_cust)
        if target_cust <= 0:
            raise ValueError("target_cust must be > 0")

        iters_per_step = int(iters_per_step)
        lookahead_depth = int(lookahead_depth)
        if iters_per_step < 0:
            raise ValueError("iters_per_step must be >= 0")
        if lookahead_depth < 1:
            raise ValueError("lookahead_depth must be >= 1")

        base_id = self.current_id
        nid = base_id

        steps: List[PlanStep] = []
        step_i = 0

        while True:
            st = self.nodes[nid].state
            if st.cust >= target_cust:
                break

            remaining = target_cust - st.cust
            depth = min(lookahead_depth, remaining)

            # temporarily set current for MCTS callbacks
            self.current_id = nid

            summary = self.run_mcts_from_current(
                iterations=iters_per_step,
                max_depth=depth,
                seed=seed + step_i,
                target_cust=target_cust,
                time_penalty=time_penalty,
                jitter_eps=jitter_eps,
            )

            p1_best = summary["p1_maximin"]["best"]
            p2_best = summary["p2_minimax"]["best"]
            joint = fmt_joint(p1_best, p2_best)

            child_id = self.ensure_child(nid, joint)
            cust_after = self.nodes[child_id].state.cust
            steps.append(PlanStep(node_id=child_id, joint=joint, cust_after=cust_after))

            nid = child_id
            step_i += 1

            # safety
            if step_i > 512:
                raise RuntimeError("plan exceeded safety bound")

        # restore current to base (planning doesn't auto-move you)
        self.current_id = base_id
        self._reset_cursor_from_current()

        self.plan = PlanResult(base_node_id=base_id, target_cust=target_cust, steps=steps)
        self.plan_replay_index = 0
        return self.plan

    # -------------------------------
    # Replay controls
    # -------------------------------
    def replay_reset(self) -> None:
        if not self.plan:
            self.plan_replay_index = 0
            return
        self.plan_replay_index = 0
        self.set_current(self.plan.base_node_id)

    def replay_step(self) -> None:
        if not self.plan:
            return
        if self.plan_replay_index >= len(self.plan.steps):
            return
        step = self.plan.steps[self.plan_replay_index]
        self.plan_replay_index += 1
        self.set_current(step.node_id)

    def replay_set_index(self, idx: int) -> None:
        if not self.plan:
            return
        idx = max(0, min(int(idx), len(self.plan.steps)))
        self.plan_replay_index = idx
        if idx == 0:
            self.set_current(self.plan.base_node_id)
        else:
            self.set_current(self.plan.steps[idx - 1].node_id)

    def plan_json(self) -> Dict[str, Any]:
        if not self.plan:
            return {"has_plan": False}
        return {
            "has_plan": True,
            "replay_index": self.plan_replay_index,
            "replay_len": len(self.plan.steps),
            "plan": self.plan.to_json(),
        }

    # -------------------------------
    # Tree query (includes mcts stats)
    # -------------------------------
    def subtree(self, node_id: str, depth: int) -> Dict[str, Any]:
        if node_id not in self.nodes:
            raise ValueError("unknown node_id")
        depth = max(0, min(10, int(depth)))

        out_nodes: Dict[str, Any] = {}
        out_edges: Dict[str, Dict[str, str]] = {}
        out_edge_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        out_node_stats: Dict[str, Dict[str, Any]] = {}

        def rec(nid: str, d: int) -> None:
            n = self.nodes[nid]
            out_nodes[nid] = n.summary()
            out_edges[nid] = {a: cid for a, cid in n.children.items()}

            ns = self.mcts.node_stats(nid)
            out_node_stats[nid] = {"N": ns.n, "Q": ns.q}

            est: Dict[str, Dict[str, Any]] = {}
            for a, _cid in n.children.items():
                es = self.mcts.edge_stats(nid, a)
                est[a] = {"N": es.n, "Q": es.q}
            out_edge_stats[nid] = est

            if d <= 0:
                return
            for _, cid in n.children.items():
                rec(cid, d - 1)

        rec(node_id, depth)
        return {
            "root_id": self.root_id,
            "current_id": self.current_id,
            "node_id": node_id,
            "depth": depth,
            "nodes": out_nodes,
            "edges": out_edges,
            "mcts": {
                "node_stats": out_node_stats,
                "edge_stats": out_edge_stats,
                "root_summary": self.mcts_summary_current(),
            },
        }
