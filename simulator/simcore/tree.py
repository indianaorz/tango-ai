# mmbn_sim/simcore/tree.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .actions import ACTIONS, ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP, ActionId
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


def _fmt_joint(p1_act: ActionId, p2_act: ActionId) -> JointActionId:
    return f"P1:{p1_act}|P2:{p2_act}"


def _parse_joint(j: JointActionId) -> Tuple[ActionId, ActionId]:
    # "P1:X|P2:Y"
    parts = j.split("|")
    if len(parts) != 2:
        raise ValueError("bad joint action")
    a = parts[0].split(":", 1)[1]
    b = parts[1].split(":", 1)[1]
    return a, b


class TreeStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: str = ""
        self.current_id: str = ""
        self._next_id: int = 1

        self.cursor_base_id: str = ""
        self.cursor: GameState = GameState()

        self.reset()

    def reset(self) -> None:
        self.nodes.clear()
        self._next_id = 1
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
        # If locked: ignore anything except clearing pending (server-authoritative rule)
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

    def cursor_state_json(self) -> Dict[str, Any]:
        return {
            "tree": {
                "root_id": self.root_id,
                "current_id": self.current_id,
                "cursor_base_id": self.cursor_base_id,
            }
        }

    # -------------------------------
    # Commit / branching
    # -------------------------------
    def commit_cursor_step(self) -> None:
        if self.cursor_base_id != self.current_id:
            self._reset_cursor_from_current()

        base_node = self.nodes[self.current_id]
        joint = self._derive_committed_joint(self.cursor, base_node.state)

        child_id = self._ensure_child(base_node, joint)
        self.current_id = child_id
        self._reset_cursor_from_current()

    def _derive_committed_joint(self, cur: GameState, base: GameState) -> JointActionId:
        def derive_for(actor: ActorId) -> ActionId:
            c = cur.actors[actor]
            b = base.actors[actor]

            # Hold changes commit as control actions, but:
            # Releasing at full charge is not "just hold off" — it *means* charge shot.
            if c.charge.hold != b.charge.hold:
                # release (true->false) at full => represents charge shot
                if b.charge.hold and (not c.charge.hold) and b.charge.level == 2 and (not b.is_locked(base.cust)):
                    return "RELEASE_CHARGE"
                return ACT_HOLD_ON if c.charge.hold else ACT_HOLD_OFF


            # Locked => cannot commit any action (NOOP)
            if b.is_locked(base.cust):
                return ACT_NOOP

            # Pending or NOOP
            return c.pending_action if c.pending_action is not None else ACT_NOOP

        return _fmt_joint(derive_for("P1"), derive_for("P2"))

    def _ensure_child(self, node: TreeNode, joint: JointActionId) -> str:
        if joint in node.children:
            return node.children[joint]

        st = copy.deepcopy(node.state)
        self._apply_joint_to_state(st, joint)
        st.advance_cust()

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

    def _apply_joint_to_state(self, st: GameState, joint: JointActionId) -> None:
        p1_act, p2_act = _parse_joint(joint)

        def apply(actor: ActorId, act: ActionId) -> None:
            a = st.actors[actor]
            # Control actions apply immediately, do not advance time.
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
                # Releasing B at full charge implies hold becomes false.
                st.cursor_set_hold(actor, False)

                # If locked, the release queues and will fire when unlocked (advance_cust handles it).
                if a.is_locked(st.cust):
                    a.pending_action = None
                    return

                a.pending_action = "RELEASE_CHARGE"
                return



            # If actor is locked at this cust, ignore.
            if a.is_locked(st.cust):
                a.pending_action = None
                return

            # Otherwise set as pending action for advance_cust start.
            if act in ACTIONS:
                a.pending_action = act
                return

            raise ValueError(f"unknown joint action element: {act}")

        apply("P1", p1_act)
        apply("P2", p2_act)

    # -------------------------------
    # Tree query
    # -------------------------------
    def subtree(self, node_id: str, depth: int) -> Dict[str, Any]:
        if node_id not in self.nodes:
            raise ValueError("unknown node_id")
        depth = max(0, min(10, int(depth)))

        out_nodes: Dict[str, Any] = {}
        out_edges: Dict[str, Dict[str, str]] = {}

        def rec(nid: str, d: int) -> None:
            n = self.nodes[nid]
            out_nodes[nid] = n.summary()
            out_edges[nid] = {a: cid for a, cid in n.children.items()}
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
        }
