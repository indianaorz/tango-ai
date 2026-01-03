#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import copy
import threading

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------
# Grid constants (BN 6x3 = 18)
# ---------------------------------------------------------------------
ROWS = 3
COLS = 6
N = ROWS * COLS

OWNER_PLAYER = 0  # allowed
OWNER_ENEMY = 1   # blocked for player movement (phase 0)

# Tile types (match your CSS tile-* classes)
TILE_BASIC = 2

DIR_DELTAS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def opposite_dir(d: str) -> str:
    return {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(d, d)


# ---------------------------------------------------------------------
# Action system (fundamental game-logic object)
# ---------------------------------------------------------------------
ActionId = str


@dataclass(frozen=True)
class EventSpec:
    type: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ActionSpec:
    lock_cust: int
    timeline: List[Tuple[int, EventSpec]]


# Damage constants (phase 0)
DMG_BUSTER = 5
DMG_CHARGE = 50

ACTIONS: Dict[ActionId, ActionSpec] = {
    # Movement
    "MOVE_UP": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"actor": "P", "dir": "UP"})),
            (1, EventSpec("MOVE_APPLY", {"actor": "P", "dir": "UP"})),
        ],
    ),
    "MOVE_DOWN": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"actor": "P", "dir": "DOWN"})),
            (1, EventSpec("MOVE_APPLY", {"actor": "P", "dir": "DOWN"})),
        ],
    ),
    "MOVE_LEFT": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"actor": "P", "dir": "LEFT"})),
            (1, EventSpec("MOVE_APPLY", {"actor": "P", "dir": "LEFT"})),
        ],
    ),
    "MOVE_RIGHT": ActionSpec(
        lock_cust=2,
        timeline=[
            (0, EventSpec("LEAN", {"actor": "P", "dir": "RIGHT"})),
            (1, EventSpec("MOVE_APPLY", {"actor": "P", "dir": "RIGHT"})),
        ],
    ),

    # Player buster / charge (raycast right)
    "P_SHOOT": ActionSpec(
        lock_cust=2,
        timeline=[(0, EventSpec("RAY_SHOT", {"actor": "P", "dmg": DMG_BUSTER, "reset_charge": True, "shot_kind": "buster"}))],
    ),
    "P_RELEASE_CHARGE": ActionSpec(
        lock_cust=5,
        timeline=[(0, EventSpec("RAY_SHOT", {"actor": "P", "dmg": DMG_CHARGE, "reset_charge": True, "shot_kind": "charge"}))],
    ),

    # Enemy (raycast left) — for testing
    "E_SHOOT": ActionSpec(
        lock_cust=2,
        timeline=[(0, EventSpec("RAY_SHOT", {"actor": "E", "dmg": DMG_BUSTER, "reset_charge": True, "shot_kind": "buster"}))],
    ),
    "E_RELEASE_CHARGE": ActionSpec(
        lock_cust=5,
        timeline=[(0, EventSpec("RAY_SHOT", {"actor": "E", "dmg": DMG_CHARGE, "reset_charge": True, "shot_kind": "charge"}))],
    ),
}

# Special MCTS/control actions (not in ACTIONS dict)
ACT_NOOP: ActionId = "NOOP"
ACT_P_HOLD_ON: ActionId = "P_HOLD_ON"
ACT_P_HOLD_OFF: ActionId = "P_HOLD_OFF"

# (Optional) If you later want enemy controls in tree:
ACT_E_HOLD_ON: ActionId = "E_HOLD_ON"
ACT_E_HOLD_OFF: ActionId = "E_HOLD_OFF"


def rc_to_idx(r: int, c: int) -> int:
    return r * COLS + c


def idx_to_rc(idx: int) -> Tuple[int, int]:
    return (idx // COLS, idx % COLS)


def default_owners() -> list[int]:
    # Left 3 cols = player-owned, right 3 cols = enemy-owned
    owners: list[int] = []
    for r in range(ROWS):
        for c in range(COLS):
            owners.append(OWNER_PLAYER if c < 3 else OWNER_ENEMY)
    return owners


def default_tiles() -> list[int]:
    return [TILE_BASIC] * N


@dataclass
class ScheduledEvent:
    due_cust: int
    spec: EventSpec
    source_action: ActionId

    def to_json(self) -> Dict[str, Any]:
        return {
            "due_cust": self.due_cust,
            "type": self.spec.type,
            "payload": self.spec.payload,
            "source_action": self.source_action,
        }


@dataclass
class ChargeState:
    """
    Toggle-based approximation of 'holding B':
      - hold=True increments progress each cust (unless locked)
      - progress==0 => level 0
      - progress>=1 => level 1 (charging)
      - progress>=5 => level 2 (full)
    """
    hold: bool = False
    progress: int = 0  # consecutive cust ticks held (when unlocked)

    FULL_AT: int = 5

    @property
    def level(self) -> int:
        if self.progress >= self.FULL_AT:
            return 2
        if self.progress >= 1:
            return 1
        return 0

    def reset(self) -> None:
        self.progress = 0

    def tick_unlocked(self) -> None:
        if not self.hold:
            self.progress = 0
            return
        self.progress += 1


@dataclass
class ShotLineVisual:
    actor: str              # "P" or "E"
    kind: str               # "buster" or "charge"
    from_idx: int
    to_idx: int
    expires_cust: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "kind": self.kind,
            "from_idx": self.from_idx,
            "to_idx": self.to_idx,
            "expires_cust": self.expires_cust,
        }


@dataclass
class GameState:
    cust_gauge: int = 0

    # HP
    p_hp: int = 1000
    e_hp: int = 1000

    # Positions
    p_idx: int = rc_to_idx(1, 1)
    e_idx: int = rc_to_idx(1, 4)

    # Board
    grid_owner_state: list[int] = None  # type: ignore[assignment]
    grid_state: list[int] = None        # type: ignore[assignment]

    # Input intent (chosen; committed on SPACE)
    pending_action: Optional[ActionId] = None

    # Single lock channel for now
    locked_until: int = 0

    # Scheduled events
    scheduled: list[ScheduledEvent] = None  # type: ignore[assignment]

    # Charge
    p_charge: ChargeState = None  # type: ignore[assignment]
    e_charge: ChargeState = None  # type: ignore[assignment]

    # Visual state (server-authoritative)
    p_lean_dir: Optional[str] = None
    e_lean_dir: Optional[str] = None

    p_entry_dir: Optional[str] = None
    p_entry_until: int = 0
    e_entry_dir: Optional[str] = None
    e_entry_until: int = 0

    shot_line: Optional[ShotLineVisual] = None

    # Debug/transient
    last_action_started: Optional[ActionId] = None
    last_events: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.grid_owner_state is None:
            self.grid_owner_state = default_owners()
        if self.grid_state is None:
            self.grid_state = default_tiles()
        if self.scheduled is None:
            self.scheduled = []
        if self.last_events is None:
            self.last_events = []
        if self.p_charge is None:
            self.p_charge = ChargeState()
        if self.e_charge is None:
            self.e_charge = ChargeState()
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        if len(self.grid_owner_state) != N:
            raise ValueError(f"grid_owner_state must be len {N}")
        if len(self.grid_state) != N:
            raise ValueError(f"grid_state must be len {N}")
        if not (0 <= self.p_idx < N) or not (0 <= self.e_idx < N):
            raise ValueError("p_idx/e_idx out of bounds")
        if self.pending_action is not None and not isinstance(self.pending_action, str):
            raise ValueError("pending_action invalid")
        if self.locked_until < 0:
            raise ValueError("locked_until invalid")
        if self.p_hp < 0 or self.e_hp < 0:
            raise ValueError("hp < 0 invalid")
        if self.p_entry_until < 0 or self.e_entry_until < 0:
            raise ValueError("entry_until invalid")

    # Minimal summary for tree nodes
    def summary(self) -> Dict[str, Any]:
        pr, pc = idx_to_rc(self.p_idx)
        er, ec = idx_to_rc(self.e_idx)
        return {
            "cust": self.cust_gauge,
            "p_hp": self.p_hp,
            "e_hp": self.e_hp,
            "p_rc": [pr, pc],
            "e_rc": [er, ec],
            "locked": self.cust_gauge < self.locked_until,
            "lock_rem": max(0, self.locked_until - self.cust_gauge),
            "p_chg": self.p_charge.level,
            "e_chg": self.e_charge.level,
            "p_hold": self.p_charge.hold,
        }

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)

        # Scheduled JSON
        d["scheduled"] = [ev.to_json() for ev in self.scheduled]

        # Add rc convenience
        pr, pc = idx_to_rc(self.p_idx)
        er, ec = idx_to_rc(self.e_idx)
        d["p_rc"] = [pr, pc]
        d["e_rc"] = [er, ec]

        # Derived lock
        d["is_locked"] = self.cust_gauge < self.locked_until
        d["lock_remaining"] = max(0, self.locked_until - self.cust_gauge)

        # Charge derived
        d["p_charge_level"] = self.p_charge.level
        d["e_charge_level"] = self.e_charge.level
        d["p_charge_hold"] = self.p_charge.hold
        d["e_charge_hold"] = self.e_charge.hold
        d["p_charge_progress"] = self.p_charge.progress
        d["e_charge_progress"] = self.e_charge.progress
        d["charge_full_at"] = self.p_charge.FULL_AT

        # Entry derived flags
        d["p_is_entering"] = (self.p_entry_dir is not None) and (self.cust_gauge < self.p_entry_until)
        d["e_is_entering"] = (self.e_entry_dir is not None) and (self.cust_gauge < self.e_entry_until)

        # Shot visual
        if self.shot_line is not None and self.cust_gauge < self.shot_line.expires_cust:
            d["shot_line"] = self.shot_line.to_json()
        else:
            d["shot_line"] = None

        return d

    # -----------------------------------------------------------------
    # Cursor controls (do NOT mutate node snapshots)
    # -----------------------------------------------------------------
    def cursor_set_pending_action(self, action_id: Optional[ActionId]) -> None:
        self.pending_action = action_id
        self._validate_invariants()

    def cursor_apply_control_action(self, action_id: ActionId) -> None:
        """
        Applies immediate cursor edits that affect next step (e.g., hold on/off).
        This does NOT advance time.
        """
        if action_id == ACT_P_HOLD_ON:
            self.p_charge.hold = True
            self._log("cursor: P_HOLD_ON")
            return
        if action_id == ACT_P_HOLD_OFF:
            self.p_charge.hold = False
            self._log("cursor: P_HOLD_OFF")
            return
        if action_id == ACT_E_HOLD_ON:
            self.e_charge.hold = True
            self._log("cursor: E_HOLD_ON")
            return
        if action_id == ACT_E_HOLD_OFF:
            self.e_charge.hold = False
            self._log("cursor: E_HOLD_OFF")
            return
        if action_id == ACT_NOOP:
            self._log("cursor: NOOP")
            return

        # Default: set as pending action
        if action_id in ACTIONS:
            self.pending_action = action_id
            return

        raise ValueError(f"unknown control action: {action_id}")

    # -----------------------------------------------------------------
    # Time step (SPACE) — deterministic
    # -----------------------------------------------------------------
    def advance_cust(self) -> None:
        """
        Step semantics:
          1) If pending_action not None, try start it at t
          2) time t->t+1
          3) apply events due at t+1
          4) if UNLOCKED at t+1, process charge ticks (hold); if LOCKED, no charge processing
        """
        t = self.cust_gauge

        # 1) start pending at t
        if self.pending_action is not None:
            self._try_start_action(self.pending_action, t)

        self.pending_action = None

        # 2) advance time
        self.cust_gauge = t + 1

        # 3) apply due events
        self._apply_due_events(self.cust_gauge)

        # 4) charge only processes while unlocked
        if self.cust_gauge < self.locked_until:
            pass
        else:
            self.p_charge.tick_unlocked()
            self.e_charge.tick_unlocked()

        # clear leans unless re-applied this cust
        self.p_lean_dir = None
        self.e_lean_dir = None

        self._validate_invariants()

    # -----------------------------------------------------------------
    # Action execution
    # -----------------------------------------------------------------
    def _try_start_action(self, action_id: ActionId, t: int) -> None:
        if action_id not in ACTIONS:
            return

        if t < self.locked_until:
            self._log(f"blocked: locked ({self.locked_until - t} left)")
            return

        # Preconditions
        if action_id.startswith("MOVE_"):
            dir_name = action_id.split("_", 1)[1]
            if not self._can_move("P", dir_name):
                self._log(f"blocked: cannot move {dir_name}")
                return

        if action_id == "P_RELEASE_CHARGE" and self.p_charge.level != 2:
            self._log("blocked: P not full charge")
            return
        if action_id == "E_RELEASE_CHARGE" and self.e_charge.level != 2:
            self._log("blocked: E not full charge")
            return

        spec = ACTIONS[action_id]

        # Commit lock
        self.locked_until = t + spec.lock_cust
        self.last_action_started = action_id

        # Schedule events
        for offset, ev in spec.timeline:
            self.scheduled.append(ScheduledEvent(due_cust=t + offset, spec=ev, source_action=action_id))

        # Apply offset=0 immediately (still at t)
        self._apply_due_events(t)

        self._log(f"start: {action_id} (lock {spec.lock_cust})")

    def _apply_due_events(self, now_cust: int) -> None:
        if not self.scheduled:
            return

        due: list[ScheduledEvent] = []
        future: list[ScheduledEvent] = []
        for ev in self.scheduled:
            (due if ev.due_cust == now_cust else future).append(ev)
        self.scheduled = future

        for ev in due:
            self._apply_event(ev.spec, now_cust)

        # expire shot_line if needed
        if self.shot_line is not None and now_cust >= self.shot_line.expires_cust:
            self.shot_line = None

        # expire entry if needed
        if self.p_entry_dir is not None and now_cust >= self.p_entry_until:
            self.p_entry_dir = None
        if self.e_entry_dir is not None and now_cust >= self.e_entry_until:
            self.e_entry_dir = None

    # -----------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------
    def _apply_event(self, ev: EventSpec, now_cust: int) -> None:
        et = ev.type

        if et == "LEAN":
            actor = str(ev.payload.get("actor", "P"))
            dir_name = str(ev.payload.get("dir", ""))
            if dir_name in DIR_DELTAS:
                if actor == "P":
                    self.p_lean_dir = dir_name
                else:
                    self.e_lean_dir = dir_name
                self._log(f"event: {actor} LEAN {dir_name}")
            return

        if et == "MOVE_APPLY":
            actor = str(ev.payload.get("actor", "P"))
            dir_name = str(ev.payload.get("dir", ""))
            if dir_name in DIR_DELTAS and actor in ("P", "E"):
                self._try_apply_move(actor, dir_name)

                # Entry visual: lean back toward origin for 1 cust after arrival
                entry_dir = opposite_dir(dir_name)
                if actor == "P":
                    self.p_entry_dir = entry_dir
                    self.p_entry_until = now_cust + 1
                else:
                    self.e_entry_dir = entry_dir
                    self.e_entry_until = now_cust + 1

                self._log(f"event: {actor} MOVE_APPLY {dir_name}")
            return

        if et == "RAY_SHOT":
            actor = str(ev.payload.get("actor", "P"))
            dmg = int(ev.payload.get("dmg", 0))
            reset_charge = bool(ev.payload.get("reset_charge", False))
            shot_kind = str(ev.payload.get("shot_kind", "buster"))

            # Visual line
            from_idx, to_idx = self._compute_raycast_line(actor)
            self.shot_line = ShotLineVisual(
                actor=actor,
                kind=shot_kind,
                from_idx=from_idx,
                to_idx=to_idx,
                expires_cust=now_cust + 1,
            )

            # Damage if target in LOS
            self._raycast_and_damage(actor, dmg)

            if reset_charge:
                self._reset_charge(actor)

            self._log(f"event: {actor} RAY_SHOT kind={shot_kind} dmg={dmg}")
            return

        self._log(f"event: unknown {et}")

    # -----------------------------------------------------------------
    # Charge helpers
    # -----------------------------------------------------------------
    def _reset_charge(self, actor: str) -> None:
        if actor == "P":
            self.p_charge.reset()
        else:
            self.e_charge.reset()

    # -----------------------------------------------------------------
    # Movement rules (v0)
    # -----------------------------------------------------------------
    def _can_move(self, actor: str, dir_name: str) -> bool:
        if dir_name not in DIR_DELTAS:
            return False

        idx = self.p_idx if actor == "P" else self.e_idx
        dr, dc = DIR_DELTAS[dir_name]
        r, c = idx_to_rc(idx)
        nr, nc = r + dr, c + dc

        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            return False

        nidx = rc_to_idx(nr, nc)

        if actor == "P" and nidx == self.e_idx:
            return False
        if actor == "E" and nidx == self.p_idx:
            return False

        if actor == "P" and self.grid_owner_state[nidx] != OWNER_PLAYER:
            return False
        if actor == "E" and self.grid_owner_state[nidx] != OWNER_ENEMY:
            return False

        return True

    def _try_apply_move(self, actor: str, dir_name: str) -> None:
        if not self._can_move(actor, dir_name):
            return
        idx = self.p_idx if actor == "P" else self.e_idx
        dr, dc = DIR_DELTAS[dir_name]
        r, c = idx_to_rc(idx)
        nidx = rc_to_idx(r + dr, c + dc)
        if actor == "P":
            self.p_idx = nidx
        else:
            self.e_idx = nidx

    # -----------------------------------------------------------------
    # Raycast (row line-of-sight)
    # -----------------------------------------------------------------
    def _compute_raycast_line(self, actor: str) -> Tuple[int, int]:
        if actor == "P":
            a_idx = self.p_idx
            direction = +1
        else:
            a_idx = self.e_idx
            direction = -1

        ar, ac = idx_to_rc(a_idx)
        end_col = (COLS - 1) if direction > 0 else 0
        to_idx = rc_to_idx(ar, end_col)
        return a_idx, to_idx

    def _raycast_and_damage(self, actor: str, dmg: int) -> None:
        if dmg <= 0:
            return

        if actor == "P":
            a_idx, t_idx = self.p_idx, self.e_idx
            direction = +1
        else:
            a_idx, t_idx = self.e_idx, self.p_idx
            direction = -1

        ar, ac = idx_to_rc(a_idx)
        tr, tc = idx_to_rc(t_idx)

        if ar != tr:
            return

        c = ac + direction
        while 0 <= c < COLS:
            idx = rc_to_idx(ar, c)
            if idx == t_idx:
                if actor == "P":
                    self.e_hp = max(0, self.e_hp - dmg)
                else:
                    self.p_hp = max(0, self.p_hp - dmg)
                return
            c += direction

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.last_events.append(f"[cust {self.cust_gauge}] {msg}")
        if len(self.last_events) > 12:
            self.last_events = self.last_events[-12:]


def new_game() -> GameState:
    return GameState(
        cust_gauge=0,
        p_hp=1000,
        e_hp=1000,
        p_idx=rc_to_idx(1, 1),
        e_idx=rc_to_idx(1, 4),
        grid_owner_state=default_owners(),
        grid_state=default_tiles(),
        pending_action=None,
        locked_until=0,
        scheduled=[],
        p_charge=ChargeState(False, 0),
        e_charge=ChargeState(False, 0),
        p_lean_dir=None,
        e_lean_dir=None,
        p_entry_dir=None,
        p_entry_until=0,
        e_entry_dir=None,
        e_entry_until=0,
        shot_line=None,
        last_action_started=None,
        last_events=[],
    )


# ---------------------------------------------------------------------
# Tree store (immutable nodes + mutable cursor)
# ---------------------------------------------------------------------
@dataclass
class TreeNode:
    node_id: str
    parent_id: Optional[str]
    action_from_parent: Optional[ActionId]
    depth: int
    state: GameState  # immutable snapshot
    children: Dict[ActionId, str]

    def summary(self) -> Dict[str, Any]:
        s = self.state.summary()
        return {
            "id": self.node_id,
            "parent": self.parent_id,
            "action": self.action_from_parent,
            "depth": self.depth,
            "children_count": len(self.children),
            "s": s,
        }


class TreeStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: str = ""
        self.current_id: str = ""
        self._next_id: int = 1

        # cursor is editable working state derived from current node snapshot
        self.cursor_base_id: str = ""
        self.cursor: GameState = new_game()

        self.reset()

    def reset(self) -> None:
        self.nodes.clear()
        self._next_id = 1
        root_state = new_game()
        root = TreeNode(
            node_id=self._alloc_id(),
            parent_id=None,
            action_from_parent=None,
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

    def cursor_state_json(self) -> Dict[str, Any]:
        d = self.cursor.to_json()
        d["tree"] = {
            "root_id": self.root_id,
            "current_id": self.current_id,
            "cursor_base_id": self.cursor_base_id,
        }
        return d

    def available_actions(self, st: GameState) -> List[ActionId]:
        # “Legal-ish” options. MCTS can still try “blocked” actions if you want, but
        # keeping this pruned makes the tree readable.
        acts: List[ActionId] = [ACT_NOOP, ACT_P_HOLD_ON, ACT_P_HOLD_OFF]

        # Moves (only if can move)
        for a in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            dir_name = a.split("_", 1)[1]
            if st._can_move("P", dir_name):
                acts.append(a)

        # Shoot always allowed (but will be blocked by lock inside step)
        acts.append("P_SHOOT")

        # Release charge only if full
        if st.p_charge.level == 2:
            acts.append("P_RELEASE_CHARGE")

        return acts

    def expand_node(self, node_id: str) -> None:
        # IMPORTANT: For now, expanding must NOT generate children.
        # Nodes are created only via commit_cursor_step().
        if node_id not in self.nodes:
            raise ValueError("unknown node_id")
        return


    def _ensure_child(self, node: TreeNode, act: ActionId) -> str:
        if act in node.children:
            return node.children[act]

        # simulate one step from node.state (immutable)
        st = copy.deepcopy(node.state)
        self._apply_control_to_cursor_like(st, act)
        st.advance_cust()

        child = TreeNode(
            node_id=self._alloc_id(),
            parent_id=node.node_id,
            action_from_parent=act,
            depth=node.depth + 1,
            state=st,
            children={},
        )
        self.nodes[child.node_id] = child
        node.children[act] = child.node_id
        return child.node_id

    def _apply_control_to_cursor_like(self, st: GameState, act: ActionId) -> None:
        # Apply immediate control edits (hold) or pending action selection.
        if act == ACT_NOOP:
            st.pending_action = None
            return
        if act == ACT_P_HOLD_ON:
            st.p_charge.hold = True
            st.pending_action = None
            return
        if act == ACT_P_HOLD_OFF:
            st.p_charge.hold = False
            st.pending_action = None
            return

        # default: pending action
        if act in ACTIONS:
            st.pending_action = act
            return

        raise ValueError(f"unknown action: {act}")

    # --- Cursor interactions (what the UI uses) ---
    def cursor_key_event(self, key: str, is_down: bool) -> None:
        if not is_down:
            return

        # arrows select pending action
        if key == "ArrowUp":
            self.cursor.cursor_set_pending_action("MOVE_UP")
            return
        if key == "ArrowDown":
            self.cursor.cursor_set_pending_action("MOVE_DOWN")
            return
        if key == "ArrowLeft":
            self.cursor.cursor_set_pending_action("MOVE_LEFT")
            return
        if key == "ArrowRight":
            self.cursor.cursor_set_pending_action("MOVE_RIGHT")
            return

        # SPACE commits: cursor -> child node
        if key == " " or key == "Space":
            self.commit_cursor_step()
            return

        # shortcuts (same behavior as before)
        if key in ("z", "Z"):
            self.cursor.cursor_set_pending_action("P_SHOOT")
            return
        if key in ("c", "C"):
            self.cursor.cursor_set_pending_action("P_RELEASE_CHARGE")
            return
        if key in ("x", "X"):
            # Toggle is cursor-only; deterministic at commit time
            self.cursor.p_charge.hold = not self.cursor.p_charge.hold
            self.cursor._log(f"cursor: toggle hold -> {self.cursor.p_charge.hold}")
            return

    def cursor_ui_action(self, action: str) -> None:
        a = action.strip()

        # keep your existing buttons working
        if a == "P_TOGGLE_CHARGE":
            self.cursor.p_charge.hold = not self.cursor.p_charge.hold
            self.cursor._log(f"cursor: toggle hold -> {self.cursor.p_charge.hold}")
            return

        if a in ("P_SHOOT", "P_RELEASE_CHARGE", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            self.cursor.cursor_set_pending_action(a)
            return

        # expose explicit hold-on/off as well
        if a in (ACT_P_HOLD_ON, ACT_P_HOLD_OFF, ACT_NOOP):
            self.cursor.cursor_apply_control_action(a)
            return

        raise ValueError(f"unknown ui action: {action}")

    def commit_cursor_step(self) -> None:
        """
        Uses cursor_base_id as the branching source.
        Creates/reuses child node keyed by the *committed action* (control + pending),
        then advances current to that child and resets cursor.
        """
        if self.cursor_base_id != self.current_id:
            # safety: cursor should always match current; if not, re-sync
            self._reset_cursor_from_current()

        base_node = self.nodes[self.current_id]

        act = self._derive_committed_action(self.cursor)
        child_id = self._ensure_child(base_node, act)

        # move current pointer
        self.current_id = child_id
        self._reset_cursor_from_current()

    def _derive_committed_action(self, cur: GameState) -> ActionId:
        # priority: if hold differs from base snapshot, commit a hold action
        base = self.nodes[self.current_id].state
        if cur.p_charge.hold != base.p_charge.hold:
            return ACT_P_HOLD_ON if cur.p_charge.hold else ACT_P_HOLD_OFF

        # pending action or NOOP
        if cur.pending_action is None:
            return ACT_NOOP
        return cur.pending_action

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


# ---------------------------------------------------------------------
# Flask wiring
# ---------------------------------------------------------------------
app = Flask(__name__)

_LOCK = threading.Lock()
_TREE = TreeStore()


@app.get("/")
def index():
    return render_template("sim.html")


@app.get("/api/state")
def api_state():
    with _LOCK:
        return jsonify(_TREE.cursor_state_json())


@app.post("/api/reset")
def api_reset():
    with _LOCK:
        _TREE.reset()
        return jsonify(_TREE.cursor_state_json())


@app.post("/api/key_event")
def api_key_event():
    data = request.get_json(silent=True) or {}
    key = data.get("key", None)
    is_down = data.get("is_down", None)

    if not isinstance(key, str):
        return jsonify({"error": "key must be a string"}), 400
    if not isinstance(is_down, bool):
        return jsonify({"error": "is_down must be a boolean"}), 400

    with _LOCK:
        try:
            _TREE.cursor_key_event(key, is_down)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(_TREE.cursor_state_json())


@app.post("/api/ui_action")
def api_ui_action():
    data = request.get_json(silent=True) or {}
    action = data.get("action", None)
    if not isinstance(action, str):
        return jsonify({"error": "action must be a string"}), 400

    with _LOCK:
        try:
            _TREE.cursor_ui_action(action)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(_TREE.cursor_state_json())


# ---------------- Tree API ----------------

@app.get("/api/tree/subtree")
def api_tree_subtree():
    node_id = request.args.get("node_id", "").strip()
    depth = request.args.get("depth", "4").strip()
    if not node_id:
        node_id = "ROOT"

    with _LOCK:
        try:
            if node_id == "ROOT":
                node_id = _TREE.root_id
            payload = _TREE.subtree(node_id=node_id, depth=int(depth))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(payload)


@app.post("/api/tree/expand")
def api_tree_expand():
    # Expand is UI-only for now; server does not generate children.
    data = request.get_json(silent=True) or {}
    node_id = data.get("node_id", None)
    if not isinstance(node_id, str) or not node_id.strip():
        return jsonify({"error": "node_id must be a non-empty string"}), 400

    with _LOCK:
        try:
            # no-op expand
            payload = _TREE.subtree(node_id=node_id.strip(), depth=3)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(payload)



@app.post("/api/tree/set_current")
def api_tree_set_current():
    data = request.get_json(silent=True) or {}
    node_id = data.get("node_id", None)
    if not isinstance(node_id, str) or not node_id.strip():
        return jsonify({"error": "node_id must be a non-empty string"}), 400

    with _LOCK:
        try:
            _TREE.set_current(node_id.strip())
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(_TREE.cursor_state_json())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
