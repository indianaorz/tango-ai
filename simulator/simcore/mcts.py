# mmbn_sim/simcore/mcts.py
# (ONLY showing the full file because you asked for complete updates in this repo style.)
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .actions import ACTIONS, ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP, ActionId
from .state import ActorId, GameState

JointActionId = str


def fmt_joint(p1_act: ActionId, p2_act: ActionId) -> JointActionId:
    return f"P1:{p1_act}|P2:{p2_act}"


def parse_joint(j: JointActionId) -> Tuple[ActionId, ActionId]:
    parts = j.split("|")
    if len(parts) != 2:
        raise ValueError(f"bad joint action: {j!r}")
    a = parts[0].split(":", 1)[1]
    b = parts[1].split(":", 1)[1]
    return a, b


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 400
    max_depth: int = 10
    c_ucb: float = 1.25
    seed: int = 0
    target_cust: Optional[int] = None
    time_penalty: float = 0.002
    jitter_eps: float = 1e-4


@dataclass
class EdgeStats:
    n: int = 0
    w: float = 0.0

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


@dataclass
class NodeStats:
    n: int = 0
    w: float = 0.0

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class MCTSStatsStore:
    """
    Stores stats keyed by (node_id, joint_action) + per-node totals.

    Value is always from P1 perspective:
      +1.0 good for P1, -1.0 good for P2.
    """

    def __init__(self) -> None:
        self.node: Dict[str, NodeStats] = {}
        self.edge: Dict[Tuple[str, JointActionId], EdgeStats] = {}

    def reset(self) -> None:
        self.node.clear()
        self.edge.clear()

    def node_stats(self, node_id: str) -> NodeStats:
        ns = self.node.get(node_id)
        if ns is None:
            ns = NodeStats()
            self.node[node_id] = ns
        return ns

    def edge_stats(self, node_id: str, joint: JointActionId) -> EdgeStats:
        k = (node_id, joint)
        es = self.edge.get(k)
        if es is None:
            es = EdgeStats()
            self.edge[k] = es
        return es

    def edge_qn(self, node_id: str, joint: JointActionId) -> Tuple[float, int]:
        es = self.edge.get((node_id, joint))
        if es is None:
            return (0.0, 0)
        return (es.q, es.n)


def is_terminal(st: GameState) -> bool:
    p1 = st.actors["P1"]
    p2 = st.actors["P2"]
    return (p1.hp <= 0) or (p2.hp <= 0)


def _fnv1a32_init() -> int:
    return 2166136261


def _fnv1a32_mix(h: int, v: int) -> int:
    h ^= (v & 0xFFFFFFFF)
    h = (h * 16777619) & 0xFFFFFFFF
    return h


def _stable_state_hash32(st: GameState) -> int:
    h = _fnv1a32_init()
    h = _fnv1a32_mix(h, int(st.cust))

    for aid in ("P1", "P2"):
        a = st.actors[aid]
        h = _fnv1a32_mix(h, int(a.hp))
        h = _fnv1a32_mix(h, int(a.idx))
        h = _fnv1a32_mix(h, int(a.form))  # <-- IMPORTANT for cross matchups
        h = _fnv1a32_mix(h, int(a.locked_until))
        h = _fnv1a32_mix(h, 1 if a.charge.hold else 0)
        h = _fnv1a32_mix(h, int(a.charge.progress))
        h = _fnv1a32_mix(h, 1 if a.charge.queued_release else 0)

    h = _fnv1a32_mix(h, len(st.scheduled))
    for ev in st.scheduled[:16]:
        h = _fnv1a32_mix(h, int(ev.due_cust))
        h = _fnv1a32_mix(h, 1 if ev.actor == "P1" else 2)
        for ch in ev.spec.type.encode("utf-8", errors="ignore")[:16]:
            h = _fnv1a32_mix(h, int(ch))

    return h


def _jitter(st: GameState, eps: float) -> float:
    if eps <= 0:
        return 0.0
    h = _stable_state_hash32(st)
    x = (h % 1000003) / 500001.5 - 1.0
    return float(eps) * float(x)


def eval_p1_at_target(
    st: GameState,
    target_cust: Optional[int],
    time_penalty: float,
    jitter_eps: float,
) -> float:
    p1 = st.actors["P1"].hp
    p2 = st.actors["P2"].hp

    if p2 <= 0 and p1 > 0:
        v = 1.0
    elif p1 <= 0 and p2 > 0:
        v = -1.0
    else:
        diff = float(p1 - p2)
        v = math.tanh(diff / 350.0)

    if time_penalty != 0.0:
        if target_cust is None:
            v -= float(time_penalty) * (float(st.cust) / 64.0)
        else:
            if is_terminal(st) and st.cust < target_cust:
                v -= float(time_penalty) * (float(st.cust) / float(max(1, target_cust)))

    v += _jitter(st, jitter_eps)
    return max(-1.0, min(1.0, v))


def _canonical_commit_actions(st: GameState, actor: ActorId) -> List[ActionId]:
    legal = list(st.legal_action_ids(actor))
    a = st.actors[actor]

    if a.charge.queued_release and (not a.is_locked(st.cust)):
        return ["RELEASE_CHARGE"]

    if (
        (not a.is_locked(st.cust))
        and a.charge.hold
        and a.charge.level == 2
        and ("HOLD_OFF" in legal)
    ):
        legal = [x for x in legal if x != "HOLD_OFF"]
        if "RELEASE_CHARGE" not in legal:
            legal.append("RELEASE_CHARGE")

    allowed: Set[ActionId] = {ACT_NOOP, ACT_HOLD_ON, ACT_HOLD_OFF, "RELEASE_CHARGE"} | set(ACTIONS.keys())
    legal2 = [x for x in legal if x in allowed]

    if ACT_NOOP not in legal2:
        legal2.append(ACT_NOOP)

    return sorted(set(legal2))


def list_joint_actions(st: GameState) -> List[JointActionId]:
    p1 = _canonical_commit_actions(st, "P1")
    p2 = _canonical_commit_actions(st, "P2")
    out: List[JointActionId] = []
    for a in p1:
        for b in p2:
            out.append(fmt_joint(a, b))
    return out


def apply_joint_in_place(st: GameState, joint: JointActionId) -> None:
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

        raise ValueError(f"unknown joint action element: {act!r}")

    apply("P1", p1_act)
    apply("P2", p2_act)
    st.advance_cust()


def _ucb_score(parent_n: int, edge: EdgeStats, c_ucb: float) -> float:
    if parent_n < 1:
        parent_n = 1
    bonus = c_ucb * math.sqrt(math.log(parent_n + 1.0) / (edge.n + 1.0))
    return edge.q + bonus


def _select_joint_ucb(
    stats: MCTSStatsStore,
    node_id: str,
    joint_actions: List[JointActionId],
    c_ucb: float,
) -> JointActionId:
    ns = stats.node_stats(node_id)
    best_j = joint_actions[0]
    best_s = -1e18
    for j in joint_actions:
        es = stats.edge_stats(node_id, j)
        s = _ucb_score(ns.n, es, c_ucb)
        if s > best_s:
            best_s = s
            best_j = j
    return best_j


def _pick_unexpanded(
    tree_children: Dict[JointActionId, str],
    joint_actions: List[JointActionId],
    rng: random.Random,
) -> Optional[JointActionId]:
    unexp = [j for j in joint_actions if j not in tree_children]
    if not unexp:
        return None
    return rng.choice(unexp)


def _rollout_value(st: GameState, rng: random.Random, cfg: MCTSConfig) -> float:
    start_cust = st.cust
    for d in range(max(0, int(cfg.max_depth))):
        if is_terminal(st):
            break
        if cfg.target_cust is not None and st.cust >= cfg.target_cust:
            break
        joints = list_joint_actions(st)
        j = rng.choice(joints)
        apply_joint_in_place(st, j)

    return eval_p1_at_target(
        st,
        target_cust=cfg.target_cust,
        time_penalty=cfg.time_penalty,
        jitter_eps=cfg.jitter_eps,
    )


def recommend_root_maximin(
    stats: MCTSStatsStore,
    root_id: str,
    root_state: GameState,
) -> Dict[str, Any]:
    p1_acts = _canonical_commit_actions(root_state, "P1")
    p2_acts = _canonical_commit_actions(root_state, "P2")

    def q_or_pess(node_id: str, j: JointActionId, pess: float) -> float:
        q, n = stats.edge_qn(node_id, j)
        return q if n > 0 else pess

    p1_table: Dict[str, float] = {}
    best_a = p1_acts[0]
    best_v = -1e18
    for a in p1_acts:
        worst = 1e18
        for b in p2_acts:
            j = fmt_joint(a, b)
            v = q_or_pess(root_id, j, pess=-1.0)
            worst = min(worst, v)
        p1_table[a] = worst
        if worst > best_v:
            best_v = worst
            best_a = a

    p2_table: Dict[str, float] = {}
    best_b = p2_acts[0]
    best_b_v = 1e18
    for b in p2_acts:
        br = -1e18
        for a in p1_acts:
            j = fmt_joint(a, b)
            v = q_or_pess(root_id, j, pess=+1.0)
            br = max(br, v)
        p2_table[b] = br
        if br < best_b_v:
            best_b_v = br
            best_b = b

    return {
        "root_node_id": root_id,
        "root_visits": stats.node_stats(root_id).n,
        "p1_maximin": {"best": best_a, "value": best_v, "table": p1_table},
        "p2_minimax": {"best": best_b, "value": best_b_v, "table": p2_table},
    }


def run_mcts(
    stats: MCTSStatsStore,
    root_id: str,
    root_state: GameState,
    ensure_child_fn,
    get_node_state_fn,
    get_node_children_fn,
    config: MCTSConfig,
) -> Dict[str, Any]:
    if config.iterations <= 0:
        return recommend_root_maximin(stats, root_id, root_state)

    rng = random.Random(int(config.seed))

    for _ in range(int(config.iterations)):
        node_id = root_id
        path: List[Tuple[str, JointActionId]] = []
        depth = 0

        while True:
            st = get_node_state_fn(node_id)

            if is_terminal(st):
                break
            if config.target_cust is not None and st.cust >= config.target_cust:
                break
            if depth >= config.max_depth:
                break

            joint_actions = list_joint_actions(st)
            children = get_node_children_fn(node_id)

            unexp = _pick_unexpanded(children, joint_actions, rng)
            if unexp is not None:
                child_id = ensure_child_fn(node_id, unexp)
                path.append((node_id, unexp))
                node_id = child_id
                depth += 1
                break

            chosen = _select_joint_ucb(stats, node_id, joint_actions, config.c_ucb)
            child_id = ensure_child_fn(node_id, chosen)
            path.append((node_id, chosen))
            node_id = child_id
            depth += 1

        leaf_state = copy.deepcopy(get_node_state_fn(node_id))
        value = _rollout_value(leaf_state, rng, config)

        stats.node_stats(root_id).n += 1
        stats.node_stats(root_id).w += value

        for parent_id, joint in path:
            stats.node_stats(parent_id).n += 1
            stats.node_stats(parent_id).w += value
            es = stats.edge_stats(parent_id, joint)
            es.n += 1
            es.w += value

    return recommend_root_maximin(stats, root_id, root_state)
