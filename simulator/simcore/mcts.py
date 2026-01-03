# mmbn_sim/simcore/mcts.py
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .actions import ACTIONS, ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP, ActionId
from .coords import idx_to_rc
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


# -----------------------------------------------------------------------------
# Stable key + hash (used for jitter and TreeStore transpositions)
# -----------------------------------------------------------------------------
def stable_state_key(st: GameState) -> Tuple[Any, ...]:
    """
    A deterministic, hashable summary of state used for:
      - transposition caching
      - stable tie-breaks
      - stable hashing

    Must stay in sync with gameplay-relevant state.
    """
    k: List[Any] = []
    k.append(int(st.cust))

    for aid in ("P1", "P2"):
        a = st.actors[aid]
        k.extend(
            [
                aid,
                int(a.hp),
                int(a.idx),
                int(a.form),
                int(a.locked_until),
                1 if a.charge.hold else 0,
                int(a.charge.progress),
                1 if a.charge.queued_release else 0,
            ]
        )

    # Scheduled events: order matters for determinism.
    # Keep it bounded so key doesn't explode (matches hash mixing).
    k.append(int(len(st.scheduled)))
    for ev in st.scheduled[:16]:
        k.extend(
            [
                int(ev.due_cust),
                ev.actor,
                str(ev.spec.type),
                str(ev.source_action),
            ]
        )
        # payload may be large; we avoid it for key stability unless needed.
        # If payload becomes gameplay-relevant in future, add a compact digest.

    return tuple(k)


def _fnv1a32_init() -> int:
    return 2166136261


def _fnv1a32_mix(h: int, v: int) -> int:
    h ^= (v & 0xFFFFFFFF)
    h = (h * 16777619) & 0xFFFFFFFF
    return h


def stable_state_hash32(st: GameState) -> int:
    """
    Stable 32-bit hash derived from stable_state_key.
    Fast enough for per-step usage.
    """
    h = _fnv1a32_init()
    # We mix ints; for strings we mix bytes.
    for item in stable_state_key(st):
        if isinstance(item, int):
            h = _fnv1a32_mix(h, int(item))
        else:
            bs = str(item).encode("utf-8", errors="ignore")
            # bound bytes mixed per item
            for ch in bs[:32]:
                h = _fnv1a32_mix(h, int(ch))
    return h


def _jitter(st: GameState, eps: float) -> float:
    if eps <= 0:
        return 0.0
    h = stable_state_hash32(st)
    x = (h % 1000003) / 500001.5 - 1.0
    return float(eps) * float(x)


# -----------------------------------------------------------------------------
# Shaped evaluation (deterministic + bounded)
# -----------------------------------------------------------------------------
def _can_attack_now(st: GameState, actor: ActorId) -> bool:
    """
    Immediate "can deal damage this cust" approximation.
    We only model buster/charge. Requires:
      - unlocked
      - same row (ray hits only then)
      - has a legal attack action this cust
    """
    a = st.actors[actor]
    if a.is_locked(st.cust):
        return False

    e = st.actors["P2" if actor == "P1" else "P1"]
    ar, _ac = idx_to_rc(a.idx)
    er, _ec = idx_to_rc(e.idx)
    if ar != er:
        return False

    legal = set(st.legal_action_ids(actor))
    # At full charge, SHOOT is disallowed; RELEASE_CHARGE is the attack.
    if "RELEASE_CHARGE" in legal:
        return True
    if "SHOOT" in legal:
        return True
    return False


def _mobility_score(st: GameState, actor: ActorId) -> float:
    """
    Rough mobility score in [0,1].
    Counts directional moves available this cust.
    """
    legal = set(st.legal_action_ids(actor))
    moves = 0
    for a in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
        if a in legal:
            moves += 1
    return float(moves) / 4.0


def eval_p1_at_target(
    st: GameState,
    target_cust: Optional[int],
    time_penalty: float,
    jitter_eps: float,
) -> float:
    """
    Deterministic bounded value in [-1, +1], from P1 perspective.

    Components:
      - HP diff (dominant)
      - lock advantage
      - immediate threat advantage (can attack this cust)
      - charge advantage (0/1/2)
      - position/mobility (small)
      - time penalty + small deterministic jitter
    """
    p1 = st.actors["P1"]
    p2 = st.actors["P2"]

    # Terminal first (crisp)
    if p2.hp <= 0 and p1.hp > 0:
        v = 1.0
    elif p1.hp <= 0 and p2.hp > 0:
        v = -1.0
    else:
        hp_diff = float(p1.hp - p2.hp)
        hp_term = math.tanh(hp_diff / 350.0)  # [-1,1]

        p1_locked = 1.0 if p1.is_locked(st.cust) else 0.0
        p2_locked = 1.0 if p2.is_locked(st.cust) else 0.0
        lock_adv = (p2_locked - p1_locked)  # good for P1 if P2 locked

        p1_threat = 1.0 if _can_attack_now(st, "P1") else 0.0
        p2_threat = 1.0 if _can_attack_now(st, "P2") else 0.0
        threat_adv = p1_threat - p2_threat  # {-1,0,1}

        # Charge advantage: treat queued_release as full (2)
        p1_lvl = 2 if p1.charge.queued_release else int(p1.charge.level)
        p2_lvl = 2 if p2.charge.queued_release else int(p2.charge.level)
        charge_adv = float(p1_lvl - p2_lvl) / 2.0  # [-1,1]

        # Small positional bias: center row + mobility
        p1r, _ = idx_to_rc(p1.idx)
        p2r, _ = idx_to_rc(p2.idx)
        center_adv = (1.0 if p1r == 1 else 0.0) - (1.0 if p2r == 1 else 0.0)

        mob_adv = _mobility_score(st, "P1") - _mobility_score(st, "P2")

        # Weighted sum -> squash
        raw = (
            1.00 * hp_term
            + 0.35 * lock_adv
            + 0.25 * threat_adv
            + 0.20 * charge_adv
            + 0.10 * center_adv
            + 0.10 * mob_adv
        )
        v = math.tanh(raw)  # keep bounded and smooth

    # Time penalty: prefer winning sooner / avoid endless stalling.
    if time_penalty != 0.0:
        if target_cust is None:
            v -= float(time_penalty) * (float(st.cust) / 64.0)
        else:
            if is_terminal(st) and st.cust < target_cust:
                v -= float(time_penalty) * (float(st.cust) / float(max(1, target_cust)))

    v += _jitter(st, jitter_eps)
    return max(-1.0, min(1.0, float(v)))


# -----------------------------------------------------------------------------
# Action listing (unchanged) + deterministic rollout policy (NEW)
# -----------------------------------------------------------------------------
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


def _tie_break_key(st: GameState, actor: ActorId, act: ActionId) -> int:
    """
    Deterministic tie-breaker: no RNG, but stable per (state, actor, action).
    """
    h = stable_state_hash32(st)
    # Mix in actor/action strings
    for ch in (actor + ":" + act).encode("utf-8", errors="ignore")[:32]:
        h = _fnv1a32_mix(h, int(ch))
    return h


def _prefer_center_row_action(st: GameState, actor: ActorId, legal: Set[ActionId]) -> Optional[ActionId]:
    a = st.actors[actor]
    r, _c = idx_to_rc(a.idx)
    if r == 1:
        return None
    # Move toward row 1
    if r == 0 and "MOVE_DOWN" in legal:
        return "MOVE_DOWN"
    if r == 2 and "MOVE_UP" in legal:
        return "MOVE_UP"
    return None


def _prefer_match_enemy_row_action(st: GameState, actor: ActorId, legal: Set[ActionId]) -> Optional[ActionId]:
    a = st.actors[actor]
    e = st.actors["P2" if actor == "P1" else "P1"]
    ar, _ = idx_to_rc(a.idx)
    er, _ = idx_to_rc(e.idx)
    if ar == er:
        return None
    if er < ar and "MOVE_UP" in legal:
        return "MOVE_UP"
    if er > ar and "MOVE_DOWN" in legal:
        return "MOVE_DOWN"
    return None


def _prefer_forward_pressure_action(st: GameState, actor: ActorId, legal: Set[ActionId]) -> Optional[ActionId]:
    """
    In local view, "forward" is MOVE_RIGHT.
    This tends to move closer to center line for both actors (due to canonical mirroring).
    """
    if "MOVE_RIGHT" in legal:
        return "MOVE_RIGHT"
    if "MOVE_LEFT" in legal:
        return "MOVE_LEFT"
    return None


def _choose_rollout_action_for_actor(st: GameState, actor: ActorId) -> ActionId:
    """
    Deterministic heuristic rollout policy.
    No randomness unless tie-break (and tie-break is deterministic too).
    """
    legal_list = _canonical_commit_actions(st, actor)
    legal: Set[ActionId] = set(legal_list)

    a = st.actors[actor]

    # If queued release and unlocked, the sim forces it anyway.
    if (not a.is_locked(st.cust)) and a.charge.queued_release and ("RELEASE_CHARGE" in legal):
        return "RELEASE_CHARGE"

    # While locked: prefer turning hold on (so charge can build immediately when unlocked).
    if a.is_locked(st.cust):
        if (not a.charge.hold) and ("HOLD_ON" in legal):
            return "HOLD_ON"
        return ACT_NOOP

    # If full charge, prefer releasing (strong immediate value).
    if a.charge.level == 2 and ("RELEASE_CHARGE" in legal):
        return "RELEASE_CHARGE"

    # If we can attack now (same row), do it.
    if _can_attack_now(st, actor):
        # If full charge exists, we'd have returned above; otherwise SHOOT is the attack.
        if "SHOOT" in legal:
            return "SHOOT"
        if "RELEASE_CHARGE" in legal:
            return "RELEASE_CHARGE"

    # If not holding and not yet level 2, start holding to build charge.
    if (not a.charge.hold) and a.charge.level < 2 and ("HOLD_ON" in legal):
        return "HOLD_ON"

    # Movement heuristics (in priority order)
    cand: List[ActionId] = []
    x = _prefer_center_row_action(st, actor, legal)
    if x:
        cand.append(x)

    x = _prefer_match_enemy_row_action(st, actor, legal)
    if x:
        cand.append(x)

    x = _prefer_forward_pressure_action(st, actor, legal)
    if x:
        cand.append(x)

    # Fall back to any legal move if we couldn't pick a preferred one.
    for a_id in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
        if a_id in legal and a_id not in cand:
            cand.append(a_id)

    if cand:
        # Deterministic tie-break among candidates by stable hash.
        best = cand[0]
        best_k = _tie_break_key(st, actor, best)
        for act in cand[1:]:
            kk = _tie_break_key(st, actor, act)
            if kk < best_k:
                best = act
                best_k = kk
        return best

    return ACT_NOOP


def rollout_policy_joint(st: GameState) -> JointActionId:
    """
    Returns a deterministic joint action (P1 + P2) using heuristics.
    """
    a1 = _choose_rollout_action_for_actor(st, "P1")
    a2 = _choose_rollout_action_for_actor(st, "P2")
    return fmt_joint(a1, a2)


# -----------------------------------------------------------------------------
# UCB + selection (mostly unchanged)
# -----------------------------------------------------------------------------
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
    """
    Rollout no longer samples random joints.
    It uses deterministic rollout_policy_joint(st) at each step.
    (rng kept in signature so run_mcts callsites don't change.)
    """
    _ = rng  # intentionally unused now; kept for API stability

    for _d in range(max(0, int(cfg.max_depth))):
        if is_terminal(st):
            break
        if cfg.target_cust is not None and st.cust >= cfg.target_cust:
            break
        j = rollout_policy_joint(st)
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
