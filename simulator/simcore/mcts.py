# mmbn_sim/simcore/mcts.py
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .actions import ACTIONS, ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP, ACT_USE_CHIP, ActionId
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

    # PUCT:
    # score = Q + c_puct * P(a|s) * sqrt(parentN) / (1 + edgeN)
    c_puct: float = 1.25

    # legacy (unused when use_puct=True, kept for compatibility)
    c_ucb: float = 1.25
    use_puct: bool = True

    seed: int = 0
    target_cust: Optional[int] = None
    time_penalty: float = 0.002
    jitter_eps: float = 1e-4

    # Parallel rollout evaluation
    workers: int = 1
    inflight_per_worker: int = 4  # how many rollouts to keep queued per worker

    # Console progress
    progress: bool = False
    progress_every_sec: float = 0.35  # rate-limit prints
    progress_label: str = ""


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

        # Cached priors per node:
        #   node_id -> {joint_action: prior_prob}
        self.prior: Dict[str, Dict[JointActionId, float]] = {}

    def reset(self) -> None:
        self.node.clear()
        self.edge.clear()
        self.prior.clear()

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

    # IMPORTANT: board ownership/tiles are gameplay-relevant (AreaGrab etc.)
    k.append(("owners", tuple(int(x) for x in st.board.owners)))
    k.append(("tiles", tuple(int(x) for x in st.board.tiles)))

    # AreaGrab timers are gameplay-relevant (expiry behavior)
    if getattr(st, "area_cols", None):
        # deterministic ordering by col
        cols = sorted(st.area_cols.items(), key=lambda kv: int(kv[0]))
        k.append(("area_cols_len", len(cols)))
        for col, rec in cols:
            k.extend(
                [
                    int(col),
                    int(rec.stolen_owner),
                    int(rec.started_cust),
                    int(rec.expires_cust),
                ]
            )
    else:
        k.append(("area_cols_len", 0))

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
                ("hand", tuple(int(x) for x in a.chip_hand)),
            ]
        )

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
    for item in stable_state_key(st):
        if isinstance(item, int):
            h = _fnv1a32_mix(h, int(item))
        else:
            bs = str(item).encode("utf-8", errors="ignore")
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
    a = st.actors[actor]
    if a.is_locked(st.cust):
        return False

    e = st.actors["P2" if actor == "P1" else "P1"]
    ar, _ac = idx_to_rc(a.idx)
    er, _ec = idx_to_rc(e.idx)
    if ar != er:
        return False

    legal = set(st.legal_action_ids(actor))
    if "RELEASE_CHARGE" in legal:
        return True
    if "SHOOT" in legal:
        return True
    if ACT_USE_CHIP in legal:
        return True
    return False


def _mobility_score(st: GameState, actor: ActorId) -> float:
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
    p1 = st.actors["P1"]
    p2 = st.actors["P2"]

    if p2.hp <= 0 and p1.hp > 0:
        v = 1.0
    elif p1.hp <= 0 and p2.hp > 0:
        v = -1.0
    else:
        hp_diff = float(p1.hp - p2.hp)
        hp_term = math.tanh(hp_diff / 350.0)

        p1_locked = 1.0 if p1.is_locked(st.cust) else 0.0
        p2_locked = 1.0 if p2.is_locked(st.cust) else 0.0
        lock_adv = (p2_locked - p1_locked)

        p1_threat = 1.0 if _can_attack_now(st, "P1") else 0.0
        p2_threat = 1.0 if _can_attack_now(st, "P2") else 0.0
        threat_adv = p1_threat - p2_threat

        p1_lvl = 2 if p1.charge.queued_release else int(p1.charge.level)
        p2_lvl = 2 if p2.charge.queued_release else int(p2.charge.level)
        charge_adv = float(p1_lvl - p2_lvl) / 2.0

        p1r, _ = idx_to_rc(p1.idx)
        p2r, _ = idx_to_rc(p2.idx)
        center_adv = (1.0 if p1r == 1 else 0.0) - (1.0 if p2r == 1 else 0.0)

        mob_adv = _mobility_score(st, "P1") - _mobility_score(st, "P2")

        raw = (
            1.00 * hp_term
            + 0.35 * lock_adv
            + 0.25 * threat_adv
            + 0.20 * charge_adv
            + 0.10 * center_adv
            + 0.10 * mob_adv
        )
        v = math.tanh(raw)

    if time_penalty != 0.0:
        if target_cust is None:
            v -= float(time_penalty) * (float(st.cust) / 64.0)
        else:
            if is_terminal(st) and st.cust < target_cust:
                v -= float(time_penalty) * (float(st.cust) / float(max(1, target_cust)))

    v += _jitter(st, jitter_eps)
    return max(-1.0, min(1.0, float(v)))


# -----------------------------------------------------------------------------
# Action listing + deterministic rollout policy
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
    h = stable_state_hash32(st)
    for ch in (actor + ":" + act).encode("utf-8", errors="ignore")[:32]:
        h = _fnv1a32_mix(h, int(ch))
    return h


def _prefer_center_row_action(st: GameState, actor: ActorId, legal: Set[ActionId]) -> Optional[ActionId]:
    a = st.actors[actor]
    r, _c = idx_to_rc(a.idx)
    if r == 1:
        return None
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
    if "MOVE_RIGHT" in legal:
        return "MOVE_RIGHT"
    if "MOVE_LEFT" in legal:
        return "MOVE_LEFT"
    return None


def _choose_rollout_action_for_actor(st: GameState, actor: ActorId) -> ActionId:
    legal_list = _canonical_commit_actions(st, actor)
    legal: Set[ActionId] = set(legal_list)

    a = st.actors[actor]

    if (not a.is_locked(st.cust)) and a.charge.queued_release and ("RELEASE_CHARGE" in legal):
        return "RELEASE_CHARGE"

    if a.is_locked(st.cust):
        if (not a.charge.hold) and ("HOLD_ON" in legal):
            return "HOLD_ON"
        return ACT_NOOP

    if a.charge.level == 2 and ("RELEASE_CHARGE" in legal):
        return "RELEASE_CHARGE"

    # If we can hit now and have a chip, bias to chip first (phase 0 behavior).
    if _can_attack_now(st, actor) and (ACT_USE_CHIP in legal):
        return ACT_USE_CHIP

    if _can_attack_now(st, actor):
        if "SHOOT" in legal:
            return "SHOOT"
        if "RELEASE_CHARGE" in legal:
            return "RELEASE_CHARGE"

    if (not a.charge.hold) and a.charge.level < 2 and ("HOLD_ON" in legal):
        return "HOLD_ON"

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

    for a_id in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
        if a_id in legal and a_id not in cand:
            cand.append(a_id)

    if cand:
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
    a1 = _choose_rollout_action_for_actor(st, "P1")
    a2 = _choose_rollout_action_for_actor(st, "P2")
    return fmt_joint(a1, a2)


# (rest unchanged)
# -----------------------------------------------------------------------------
# Priors for PUCT (heuristic, policy-free)
# -----------------------------------------------------------------------------
def _softmaxish_normalize(weights: Dict[str, float]) -> Dict[str, float]:
    s = 0.0
    for v in weights.values():
        s += float(max(0.0, v))
    if s <= 0.0:
        n = float(max(1, len(weights)))
        return {k: 1.0 / n for k in weights.keys()}
    return {k: float(max(0.0, v)) / s for k, v in weights.items()}


def _actor_action_prior(st: GameState, actor: ActorId, act: ActionId) -> float:
    a = st.actors[actor]
    legal = set(st.legal_action_ids(actor))
    if act not in legal and act not in (ACT_NOOP,):
        return 0.0

    w = 1.0

    if act == ACT_NOOP:
        w = 0.25 if (not a.is_locked(st.cust)) else 0.75

    elif act == ACT_HOLD_ON:
        if (not a.charge.hold) and a.charge.level < 2:
            w = 1.8
        else:
            w = 0.6

    elif act == ACT_HOLD_OFF:
        w = 0.5

    elif act == "RELEASE_CHARGE":
        if a.charge.level == 2 or a.charge.queued_release:
            w = 3.5
        else:
            w = 0.05

    elif act == ACT_USE_CHIP:
        has_chip = len(a.chip_hand) > 0
        w = 2.6 if (has_chip and _can_attack_now(st, actor)) else (1.2 if has_chip else 0.0)

    elif act == "SHOOT":
        w = 2.2 if _can_attack_now(st, actor) else 1.1

    elif act.startswith("MOVE_"):
        legal_set = set(_canonical_commit_actions(st, actor))
        bias = 1.0

        x = _prefer_center_row_action(st, actor, legal_set)
        if x == act:
            bias *= 1.35

        x = _prefer_match_enemy_row_action(st, actor, legal_set)
        if x == act:
            bias *= 1.25

        x = _prefer_forward_pressure_action(st, actor, legal_set)
        if x == act:
            bias *= 1.10

        w = 1.0 * bias

    return float(max(0.0, w))


def _joint_priors_for_node(
    stats: "MCTSStatsStore",
    node_id: str,
    st: GameState,
    joint_actions: List[JointActionId],
) -> Dict[JointActionId, float]:
    cached = stats.prior.get(node_id)
    if cached is not None:
        return cached

    p1_legal = _canonical_commit_actions(st, "P1")
    p2_legal = _canonical_commit_actions(st, "P2")
    p1w = {a: _actor_action_prior(st, "P1", a) for a in p1_legal}
    p2w = {b: _actor_action_prior(st, "P2", b) for b in p2_legal}
    p1p = _softmaxish_normalize(p1w)
    p2p = _softmaxish_normalize(p2w)

    jw: Dict[JointActionId, float] = {}
    for j in joint_actions:
        a, b = parse_joint(j)
        jw[j] = float(p1p.get(a, 0.0)) * float(p2p.get(b, 0.0))

    jp = _softmaxish_normalize(jw)
    stats.prior[node_id] = jp
    return jp


def _puct_score(parent_n: int, edge: EdgeStats, prior_p: float, c_puct: float) -> float:
    if parent_n < 1:
        parent_n = 1
    bonus = float(c_puct) * float(prior_p) * math.sqrt(float(parent_n)) / (1.0 + float(edge.n))
    return edge.q + bonus


def _select_joint_puct(
    stats: MCTSStatsStore,
    node_id: str,
    st: GameState,
    joint_actions: List[JointActionId],
    c_puct: float,
) -> JointActionId:
    ns = stats.node_stats(node_id)
    priors = _joint_priors_for_node(stats, node_id, st, joint_actions)

    best_j = joint_actions[0]
    best_s = -1e18
    for j in joint_actions:
        es = stats.edge_stats(node_id, j)
        p = float(priors.get(j, 0.0))
        s = _puct_score(ns.n, es, p, c_puct)
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


def _rollout_value(st: GameState, cfg: MCTSConfig) -> float:
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


def _rollout_worker(payload: Tuple[GameState, MCTSConfig]) -> float:
    st, cfg = payload
    st2 = copy.deepcopy(st)
    return _rollout_value(st2, cfg)


def _clamp_workers(w: int) -> int:
    w = int(w)
    if w <= 1:
        return 1
    cpu = os.cpu_count() or 1
    return max(1, min(w, cpu))


def _progress_line(done: int, total: int, start_t: float, label: str) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    frac = float(done) / float(total)
    width = 28
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    dt = max(1e-6, time.monotonic() - start_t)
    it_s = float(done) / dt
    prefix = f"[MCTS{(' ' + label) if label else ''}]"
    return f"{prefix} {done}/{total} {frac*100:5.1f}% |{bar}| {it_s:6.1f} it/s"


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
    *,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Any]:
    iters = int(config.iterations)
    if iters <= 0:
        return recommend_root_maximin(stats, root_id, root_state)

    rng = random.Random(int(config.seed))

    workers = _clamp_workers(int(config.workers))
    inflight_limit = max(1, int(config.inflight_per_worker)) * workers

    start_t = time.monotonic()
    last_print = 0.0
    completed = 0

    last_print_done = -1

    def maybe_print(force: bool = False) -> None:
        nonlocal last_print, last_print_done
        if not config.progress:
            return
        now = time.monotonic()
        if completed == last_print_done and not force:
            return
        if force or (now - last_print) >= float(config.progress_every_sec) or completed >= iters:
            last_print = now
            last_print_done = completed
            print(_progress_line(completed, iters, start_t, config.progress_label), file=sys.stderr, flush=True)

    inflight: List[Tuple[Any, List[Tuple[str, JointActionId]], str]] = []

    def backprop(value: float, edge_path: List[Tuple[str, JointActionId]], leaf_id: str) -> None:
        visited_nodes: List[str] = [root_id]
        for parent_id, joint in edge_path:
            if parent_id != root_id:
                visited_nodes.append(parent_id)
            es = stats.edge_stats(parent_id, joint)
            es.n += 1
            es.w += float(value)

        if leaf_id not in visited_nodes:
            visited_nodes.append(leaf_id)

        for nid in visited_nodes:
            ns = stats.node_stats(nid)
            ns.n += 1
            ns.w += float(value)

    def select_and_expand_one() -> Tuple[GameState, List[Tuple[str, JointActionId]], str]:
        node_id = root_id
        edge_path: List[Tuple[str, JointActionId]] = []
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
                edge_path.append((node_id, unexp))
                node_id = child_id
                depth += 1
                break

            if config.use_puct:
                chosen = _select_joint_puct(stats, node_id, st, joint_actions, config.c_puct)
            else:
                ns = stats.node_stats(node_id)
                best_j = joint_actions[0]
                best_s = -1e18
                for j in joint_actions:
                    es = stats.edge_stats(node_id, j)
                    parent_n = max(1, ns.n)
                    bonus = config.c_ucb * math.sqrt(math.log(parent_n + 1.0) / (es.n + 1.0))
                    s = es.q + bonus
                    if s > best_s:
                        best_s = s
                        best_j = j
                chosen = best_j

            child_id = ensure_child_fn(node_id, chosen)
            edge_path.append((node_id, chosen))
            node_id = child_id
            depth += 1

        leaf_state = copy.deepcopy(get_node_state_fn(node_id))
        return leaf_state, edge_path, node_id

    if workers <= 1:
        for _ in range(iters):
            leaf_state, edge_path, leaf_id = select_and_expand_one()
            v = _rollout_value(leaf_state, config)
            backprop(v, edge_path, leaf_id)
            completed += 1
            maybe_print()
        maybe_print()
        return recommend_root_maximin(stats, root_id, root_state)

    pool = executor
    owns_pool = False
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=workers)
        owns_pool = True

    try:
        while len(inflight) < min(inflight_limit, iters):
            leaf_state, edge_path, leaf_id = select_and_expand_one()
            fut = pool.submit(_rollout_worker, (leaf_state, config))
            inflight.append((fut, edge_path, leaf_id))

        while inflight:
            still: List[Tuple[Any, List[Tuple[str, JointActionId]], str]] = []
            for fut, edge_path, leaf_id in inflight:
                if fut.done():
                    v = float(fut.result())
                    backprop(v, edge_path, leaf_id)
                    completed += 1
                    maybe_print()
                else:
                    still.append((fut, edge_path, leaf_id))
            inflight = still

            while completed + len(inflight) < iters and len(inflight) < inflight_limit:
                leaf_state, edge_path, leaf_id = select_and_expand_one()
                fut = pool.submit(_rollout_worker, (leaf_state, config))
                inflight.append((fut, edge_path, leaf_id))

            if inflight and all(not fut.done() for fut, _, _ in inflight):
                _ = next(as_completed([x[0] for x in inflight], timeout=None))
                time.sleep(0.001)

        maybe_print(force=True)
    finally:
        if owns_pool:
            pool.shutdown(wait=True)

    return recommend_root_maximin(stats, root_id, root_state)
