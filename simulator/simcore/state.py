# mmbn_sim/simcore/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Set, Iterable

from .actions import (
    ACTIONS,
    ACT_HOLD_OFF,
    ACT_HOLD_ON,
    ACT_NOOP,
    ACT_USE_CHIP,
    ActionId,
    EventSpec,
)
from .board import Board
from .chips import chip_name, chip_spec, chip_timeline, safe_chip_summary
from .constants import (
    COLS,
    DIR_DELTAS,
    N,
    OWNER_P1,
    OWNER_P2,
    ROWS,
    TILE_HOLE_PERM,
    opposite_dir,
)
from .coords import idx_to_rc, mirror_dir, rc_to_idx
from .forms import form_charge_spec, form_name, pattern_target_indices


ActorId = Literal["P1", "P2"]


def other(actor: ActorId) -> ActorId:
    return "P2" if actor == "P1" else "P1"


# -----------------------------------------------------------------------------
# AreaGrab tuning (cust-time)
# -----------------------------------------------------------------------------
# 64 cust = 8s => 1 cust = 0.125s; 30s => 240 cust
AREAGRAB_EXPIRE_CUST = 240
AREAGRAB_BLOCK_DMG = 10
AREAGRAB_BLOCK_HITSTUN = 1


def _base_owner_for_col(col: int) -> int:
    return OWNER_P1 if int(col) < 3 else OWNER_P2


def _other_owner(owner: int) -> int:
    return OWNER_P2 if int(owner) == OWNER_P1 else OWNER_P1


def _col_indices(col: int) -> List[int]:
    c = int(col)
    return [rc_to_idx(r, c) for r in range(ROWS)]


def _col_is_fully_owner(board_owners: List[int], board_tiles: List[int], col: int, owner: int) -> bool:
    """Treat permanent holes as 'neutral' for full-owner checks (they can't be stolen)."""
    for idx in _col_indices(col):
        if int(board_tiles[idx]) == TILE_HOLE_PERM:
            continue
        if int(board_owners[idx]) != int(owner):
            return False
    return True


@dataclass
class ChargeState:
    """
    Controller-faithful approximation of the single B button.

    - hold=True increments progress each cust (unless locked)
    - progress>=FULL_AT => "full charge" (level 2)

    queued_release=True means:
      - Player released B while at full charge. A charge shot MUST fire as soon
        as the actor is able to start actions (i.e., when unlocked).
      - While queued_release is true, we preserve progress (stay "full")
        until the shot actually fires.
    """
    hold: bool = False
    progress: int = 0
    queued_release: bool = False

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
        self.queued_release = False

    def set_hold(self, new_hold: bool) -> None:
        """
        Apply a hold toggle with controller semantics.

        Releasing B (hold: True -> False) at full charge queues a charge shot.
        """
        new_hold = bool(new_hold)

        if self.queued_release:
            self.hold = new_hold
            return

        was_hold = self.hold
        was_level = self.level

        self.hold = new_hold

        if was_hold and (not new_hold) and was_level == 2:
            self.queued_release = True

    def tick_unlocked(self) -> None:
        if self.queued_release:
            return

        if not self.hold:
            self.progress = 0
            return

        self.progress += 1


@dataclass
class ScheduledEvent:
    due_cust: int
    actor: ActorId
    spec: EventSpec
    source_action: ActionId

    def to_json(self) -> Dict[str, Any]:
        return {
            "due_cust": self.due_cust,
            "actor": self.actor,
            "type": self.spec.type,
            "payload": self.spec.payload,
            "source_action": self.source_action,
        }


@dataclass
class ShotLineVisual:
    actor: ActorId
    kind: str
    from_idx: int
    to_idx: int
    expires_cust: int

    def alive(self, now: int) -> bool:
        return now < self.expires_cust


@dataclass
class HotPanelVisual:
    actor: ActorId
    kind: str
    idx: int
    expires_cust: int

    def alive(self, now: int) -> bool:
        return now < self.expires_cust


@dataclass(frozen=True)
class AreaColTimer:
    col: int
    stolen_owner: int         # OWNER_P1 or OWNER_P2 (non-base owner holding this col)
    started_cust: int
    expires_cust: int


@dataclass
class ActorState:
    hp: int
    idx: int  # canonical
    form: int = 0

    charge: ChargeState = field(default_factory=ChargeState)

    chip_hand: List[int] = field(default_factory=list)

    pending_action: Optional[ActionId] = None
    locked_until: int = 0

    # Visual state stored in *local dir terms* so each actor UI is consistent
    lean_dir_local: Optional[str] = None
    entry_dir_local: Optional[str] = None
    entry_until: int = 0

    # Movement-buffer tracking (for AreaGrab blocking)
    move_dir_local: Optional[str] = None
    move_due_cust: int = 0  # when MOVE_APPLY is expected to resolve (t+1)

    def is_locked(self, cust: int) -> bool:
        return cust < self.locked_until


@dataclass
class GameState:
    cust: int = 0
    board: Board = field(default_factory=Board.new)

    actors: Dict[ActorId, ActorState] = field(default_factory=dict)

    scheduled: List[ScheduledEvent] = field(default_factory=list)
    shot_lines: List[ShotLineVisual] = field(default_factory=list)
    hot_panels: List[HotPanelVisual] = field(default_factory=list)

    # AreaGrab column timers (canonical col index -> timer)
    area_cols: Dict[int, AreaColTimer] = field(default_factory=dict)

    last_action_started: Optional[str] = None
    last_events: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.actors:
            # Default match-up: Fire vs Slash
            # P1: 1,4,1
            # P2: 163,1,163,4,4  (per your request)
            self.actors = {
                "P1": ActorState(hp=1000, idx=rc_to_idx(1, 1), form=1, chip_hand=[1, 4, 163, 163]),
                "P2": ActorState(hp=1000, idx=rc_to_idx(1, 4), form=3, chip_hand=[163, 1, 163, 4, 4]),
            }
        self._validate()

    def _validate(self) -> None:
        self.board.validate()
        for a, st in self.actors.items():
            if not (0 <= st.idx < N):
                raise ValueError(f"{a}.idx out of bounds")
            if st.hp < 0:
                raise ValueError(f"{a}.hp < 0")
            if st.locked_until < 0:
                raise ValueError(f"{a}.locked_until invalid")
            if st.entry_until < 0:
                raise ValueError(f"{a}.entry_until invalid")
            if st.move_due_cust < 0:
                raise ValueError(f"{a}.move_due_cust invalid")
            if not isinstance(st.form, int):
                raise ValueError(f"{a}.form must be int")
            if not isinstance(st.chip_hand, list):
                raise ValueError(f"{a}.chip_hand must be list")

        # sanity-check timers
        for col, rec in self.area_cols.items():
            if int(col) < 0 or int(col) >= COLS:
                raise ValueError("area_cols contains out-of-range col")
            if int(rec.col) != int(col):
                raise ValueError("area_cols key/record mismatch")
            if int(rec.stolen_owner) not in (OWNER_P1, OWNER_P2):
                raise ValueError("area_cols stolen_owner invalid")
            if int(rec.expires_cust) < int(rec.started_cust):
                raise ValueError("area_cols expires_cust < started_cust")

    # ----------------------------
    # Rules: transforms
    # ----------------------------
    def _canon_dir_for_actor(self, actor: ActorId, dir_local: str) -> str:
        return dir_local if actor == "P1" else mirror_dir(dir_local)

    def _local_dir_for_actor_from_canon(self, actor: ActorId, canon_dir: str) -> str:
        return canon_dir if actor == "P1" else mirror_dir(canon_dir)

    def _forward_col_step(self, actor: ActorId) -> int:
        return +1 if actor == "P1" else -1

    def _owner_required(self, actor: ActorId) -> int:
        return OWNER_P1 if actor == "P1" else OWNER_P2

    # ----------------------------
    # Action legality (single-actor, for UI + MCTS)
    # ----------------------------
    def legal_action_ids(self, actor: ActorId) -> List[ActionId]:
        st = self.actors[actor]
        out: Set[ActionId] = {ACT_NOOP}

        if not st.charge.hold:
            out.add(ACT_HOLD_ON)
        else:
            out.add(ACT_HOLD_OFF)

        if st.is_locked(self.cust):
            return sorted(out)

        if st.charge.queued_release:
            out.add("RELEASE_CHARGE")
            return sorted(out)

        for a in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            dir_local = a.split("_", 1)[1]
            if self.can_move(actor, dir_local):
                out.add(a)

        if st.charge.level != 2:
            out.add("SHOOT")
        else:
            out.add("RELEASE_CHARGE")

        if len(st.chip_hand) > 0:
            out.add(ACT_USE_CHIP)

        return sorted(out)

    # ----------------------------
    # Movement legality: dir is LOCAL to actor
    # ----------------------------
    def can_move(self, actor: ActorId, dir_local: str) -> bool:
        if dir_local not in DIR_DELTAS:
            return False

        canon_dir = self._canon_dir_for_actor(actor, dir_local)
        dr, dc = DIR_DELTAS[canon_dir]

        st = self.actors[actor]
        r, c = idx_to_rc(st.idx)
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            return False

        nidx = rc_to_idx(nr, nc)
        if nidx == self.actors[other(actor)].idx:
            return False

        if self.board.owners[nidx] != self._owner_required(actor):
            return False

        return True

    def _apply_move(self, actor: ActorId, dir_local: str, now: int) -> None:
        if not self.can_move(actor, dir_local):
            return
        canon_dir = self._canon_dir_for_actor(actor, dir_local)
        dr, dc = DIR_DELTAS[canon_dir]
        st = self.actors[actor]
        r, c = idx_to_rc(st.idx)
        st.idx = rc_to_idx(r + dr, c + dc)

        # Clear move-buffer state once movement resolves
        st.move_dir_local = None
        st.move_due_cust = 0

        st.entry_dir_local = opposite_dir(dir_local)
        st.entry_until = now + 1

    def _apply_forced_push(self, target: ActorId, canon_dir: str, steps: int, now: int) -> bool:
        steps = int(steps)
        if steps == 0:
            return False
        if canon_dir not in DIR_DELTAS:
            return False

        dr, dc = DIR_DELTAS[canon_dir]
        st = self.actors[target]
        r, c = idx_to_rc(st.idx)

        nr = r + dr * steps
        nc = c + dc * steps
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            return False

        nidx = rc_to_idx(nr, nc)
        if nidx == self.actors[other(target)].idx:
            return False

        st.idx = nidx

        local_dir = self._local_dir_for_actor_from_canon(target, canon_dir)
        st.entry_dir_local = opposite_dir(local_dir)
        st.entry_until = now + 1
        return True

    # ----------------------------
    # Ray shot (row LOS), forward in LOCAL
    # ----------------------------
    def _compute_raycast_line(self, actor: ActorId) -> Tuple[int, int]:
        st = self.actors[actor]
        r, c = idx_to_rc(st.idx)
        step = self._forward_col_step(actor)
        end_col = (COLS - 1) if step > 0 else 0
        return st.idx, rc_to_idx(r, end_col)

    def _ray_forward_indices(self, actor: ActorId) -> List[int]:
        st = self.actors[actor]
        r, c = idx_to_rc(st.idx)
        step = self._forward_col_step(actor)
        out: List[int] = []
        cc = c + step
        while 0 <= cc < COLS:
            out.append(rc_to_idx(r, cc))
            cc += step
        return out

    def _raycast_and_damage(self, actor: ActorId, dmg: int) -> bool:
        if dmg <= 0:
            return False

        a_st = self.actors[actor]
        t_st = self.actors[other(actor)]

        ar, ac = idx_to_rc(a_st.idx)
        tr, tc = idx_to_rc(t_st.idx)
        if ar != tr:
            return False

        step = self._forward_col_step(actor)
        c = ac + step
        while 0 <= c < COLS:
            idx = rc_to_idx(ar, c)
            if idx == t_st.idx:
                t_st.hp = max(0, t_st.hp - dmg)
                return True
            c += step
        return False

    # ----------------------------
    # Charge shot: form-dependent
    # ----------------------------
    def _charge_shot_targets(self, actor: ActorId) -> Tuple[str, List[int]]:
        a = self.actors[actor]
        spec = form_charge_spec(a.form)

        if "Pattern" in spec.ranges and spec.pattern is not None:
            targets = pattern_target_indices(
                actor_idx_canon=a.idx,
                actor_is_p1=(actor == "P1"),
                pattern=spec.pattern,
            )
            return ("Pattern", targets)

        _from, to = self._compute_raycast_line(actor)
        return ("MatchY", [to])

    def _apply_charge_shot(self, actor: ActorId, dmg: int, now: int) -> None:
        if dmg <= 0:
            return

        a = self.actors[actor]
        t = self.actors[other(actor)]
        spec = form_charge_spec(a.form)

        mode, targets = self._charge_shot_targets(actor)

        if mode == "Pattern" and spec.pattern is not None:
            for idx in targets:
                self.shot_lines.append(
                    ShotLineVisual(
                        actor=actor,
                        kind="charge",
                        from_idx=a.idx,
                        to_idx=idx,
                        expires_cust=now + 2,
                    )
                )
                self.hot_panels.append(
                    HotPanelVisual(
                        actor=actor,
                        kind="charge",
                        idx=idx,
                        expires_cust=now + 2,
                    )
                )

            if t.idx in set(targets):
                t.hp = max(0, t.hp - dmg)
            return

        from_idx, to_idx = self._compute_raycast_line(actor)
        self.shot_lines.append(
            ShotLineVisual(
                actor=actor,
                kind="charge",
                from_idx=from_idx,
                to_idx=to_idx,
                expires_cust=now + 2,
            )
        )

        for idx in self._ray_forward_indices(actor):
            self.hot_panels.append(
                HotPanelVisual(
                    actor=actor,
                    kind="charge",
                    idx=idx,
                    expires_cust=now + 2,
                )
            )

        self._raycast_and_damage(actor, dmg)

    # ----------------------------
    # Chips
    # ----------------------------
    def _start_use_chip(self, actor: ActorId, t: int) -> None:
        st = self.actors[actor]
        if st.is_locked(t):
            self._log(f"{actor}: blocked USE_CHIP (locked {st.locked_until - t} left)")
            return
        if st.charge.queued_release:
            self._log(f"{actor}: blocked USE_CHIP (queued charge release)")
            return
        if not st.chip_hand:
            self._log(f"{actor}: blocked USE_CHIP (empty hand)")
            return

        cid = int(st.chip_hand[0])
        spec = chip_spec(cid)

        st.chip_hand.pop(0)

        st.locked_until = t + int(spec.lock_cust)
        self.last_action_started = f"{actor}:{ACT_USE_CHIP}:{spec.name}[{cid}]"

        for offset, ev in chip_timeline(cid):
            self.scheduled.append(
                ScheduledEvent(
                    due_cust=t + int(offset),
                    actor=actor,
                    spec=ev,
                    source_action=ACT_USE_CHIP,
                )
            )

        self._apply_due_events(t)

        self._log(
            f"{actor}: start USE_CHIP {spec.name}[{cid}] (lock {spec.lock_cust}) "
            f"hand_now={safe_chip_summary(st.chip_hand)}"
        )

    def _apply_chip_raycast(
        self,
        actor: ActorId,
        *,
        dmg: int,
        kind: str,
        push_step: int,
        hitstun_cust: int,
        now: int,
    ) -> None:
        dmg = int(dmg)
        push_step = int(push_step)
        hitstun_cust = int(hitstun_cust)

        from_idx, to_idx = self._compute_raycast_line(actor)
        self.shot_lines.append(
            ShotLineVisual(
                actor=actor,
                kind=str(kind),
                from_idx=from_idx,
                to_idx=to_idx,
                expires_cust=now + 2,
            )
        )

        for idx in self._ray_forward_indices(actor):
            self.hot_panels.append(
                HotPanelVisual(
                    actor=actor,
                    kind="buster",
                    idx=idx,
                    expires_cust=now + 2,
                )
            )

        hit = self._raycast_and_damage(actor, dmg)
        if not hit:
            return

        tgt = other(actor)
        tgt_st = self.actors[tgt]

        if hitstun_cust > 0:
            tgt_st.locked_until = max(int(tgt_st.locked_until), int(now) + hitstun_cust)

        if push_step != 0:
            canon_dir = "RIGHT" if actor == "P1" else "LEFT"
            moved = self._apply_forced_push(tgt, canon_dir, push_step, now)
            if moved:
                self._log(f"{actor}: {kind} push -> {tgt} moved {canon_dir} by {push_step}")

    # ----------------------------
    # AreaGrab mechanics
    # ----------------------------
    def _movement_buffer_block_indices(self, actor: ActorId, now: int) -> Set[int]:
        """
        Indices considered "occupied" for purposes like AreaGrab blocking.

        If actor is buffering a vertical move resolving at `now`, they effectively occupy
        both source and destination panels (=> blocks 2 panels).
        """
        st = self.actors[actor]
        out: Set[int] = {int(st.idx)}

        if st.move_dir_local in ("UP", "DOWN") and int(st.move_due_cust) == int(now):
            canon_dir = self._canon_dir_for_actor(actor, st.move_dir_local)
            dr, dc = DIR_DELTAS[canon_dir]
            r, c = idx_to_rc(st.idx)
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                out.add(int(rc_to_idx(nr, nc)))

        return out

    def _is_column_fully_owned(self, actor: ActorId, col: int) -> bool:
        """
        A column is 'fully owned' if every stealable panel in that column is owned by actor.
        Permanent holes are not stealable and do not count against fullness.
        """
        my_owner = self._owner_required(actor)

        for r in range(ROWS):
            idx = rc_to_idx(r, col)

            # Permanent holes are never stealable/ownable; ignore them for fullness.
            if int(self.board.tiles[idx]) == TILE_HOLE_PERM:
                continue

            if int(self.board.owners[idx]) != int(my_owner):
                return False

        return True

    def _next_areagrab_target_col(self, actor: ActorId) -> Optional[int]:
        """
        AreaGrab/PanelGrab targets the first column (from your side toward the opponent)
        that is NOT fully owned by you (ignoring permanent holes).
        """
        if actor == "P1":
            scan = range(0, COLS)           # left -> right
        else:
            scan = range(COLS - 1, -1, -1)  # right -> left

        for c in scan:
            if not self._is_column_fully_owned(actor, c):
                return c

        return None

    def _col_has_any_stolen(self, col: int) -> bool:
        base = _base_owner_for_col(col)
        for idx in _col_indices(col):
            if int(self.board.tiles[idx]) == TILE_HOLE_PERM:
                continue
            if int(self.board.owners[idx]) != int(base):
                return True
        return False

    def _reconcile_area_cols(self, cols: Iterable[int], now: int, *, allow_start: bool = True) -> bool:
        """
        Keep `area_cols` consistent with current board ownership.

        - If a column is fully base-owned => remove timer.
        - If a column has any stolen panel (non-base) =>
            ensure timer exists with stolen_owner = other(base_owner),
            but DO NOT reset started/expires unless this is a brand new stolen column.
        - Timer starts only when column transitions from fully base-owned -> stolen.
          (We approximate transition by: 'no timer existed' AND 'col is stolen now'.)
        """
        changed = False
        for col in cols:
            col = int(col)
            base = _base_owner_for_col(col)

            fully_base = _col_is_fully_owner(self.board.owners, self.board.tiles, col, base)
            if fully_base:
                if col in self.area_cols:
                    self.area_cols.pop(col, None)
                    changed = True
                continue

            # Not fully base-owned. If any stolen panel exists, this column is "stolen".
            if not self._col_has_any_stolen(col):
                # Defensive: should be unreachable if not fully base-owned, but keep safe.
                if col in self.area_cols:
                    self.area_cols.pop(col, None)
                    changed = True
                continue

            stolen_owner = _other_owner(base)

            rec = self.area_cols.get(col)
            if rec is None:
                if allow_start:
                    self.area_cols[col] = AreaColTimer(
                        col=col,
                        stolen_owner=int(stolen_owner),
                        started_cust=int(now),
                        expires_cust=int(now) + AREAGRAB_EXPIRE_CUST,
                    )
                    changed = True
            else:
                if int(rec.stolen_owner) != int(stolen_owner):
                    # This shouldn't happen in current sim, but keep invariant sane
                    # without resetting the timer window.
                    self.area_cols[col] = AreaColTimer(
                        col=col,
                        stolen_owner=int(stolen_owner),
                        started_cust=int(rec.started_cust),
                        expires_cust=int(rec.expires_cust),
                    )
                    changed = True

        return changed

    def _area_outermost_for_owner(self, owner: int) -> Optional[int]:
        cols = [int(c) for c, rec in self.area_cols.items() if int(rec.stolen_owner) == int(owner)]
        if not cols:
            return None
        return max(cols) if int(owner) == OWNER_P1 else min(cols)

    def _process_areagrab_expiry(self, now: int) -> None:
        """
        Revert expired columns, but only when they are the outermost stolen column for that owner.

        Also supports same-cust cascade:
          - If +2 is removed (expired or stolen back), and +1 already expired,
            +1 reverts immediately in the same cust.
        """
        # First, ensure timers match the board (important after steal-backs).
        self._reconcile_area_cols(range(COLS), now, allow_start=True)

        blockers: Set[int] = set()
        blockers |= self._movement_buffer_block_indices("P1", now)
        blockers |= self._movement_buffer_block_indices("P2", now)

        # Loop until no further changes can occur this cust.
        while True:
            any_change = False

            for owner in (OWNER_P1, OWNER_P2):
                while True:
                    outer = self._area_outermost_for_owner(owner)
                    if outer is None:
                        break

                    rec = self.area_cols.get(int(outer))
                    if rec is None:
                        break

                    if int(now) < int(rec.expires_cust):
                        break

                    base_owner = _base_owner_for_col(outer)
                    changed_this_col = False

                    for idx in _col_indices(outer):
                        if int(self.board.tiles[idx]) == TILE_HOLE_PERM:
                            continue
                        # Only revert panels that are currently held by the stolen_owner.
                        if int(self.board.owners[idx]) != int(owner):
                            continue
                        if int(idx) in blockers:
                            continue
                        self.board.owners[idx] = int(base_owner)
                        changed_this_col = True

                    # Reconcile timer removal if column fully returned.
                    rec_change = self._reconcile_area_cols([outer], now, allow_start=False)
                    if changed_this_col or rec_change:
                        any_change = True

                    # If we could not change anything (fully blocked), stop trying this owner this cust.
                    # Future custs may unblock it.
                    if not (changed_this_col or rec_change):
                        break

                    # Otherwise, keep looping for this owner because:
                    # - same col might still have more to revert next iteration (rare)
                    # - or outermost might have changed (col removed), revealing an already-expired inner col

            if not any_change:
                break

    def _apply_areagrab(self, actor: ActorId, now: int) -> None:
        my_owner = self._owner_required(actor)
        enemy = other(actor)
        e = self.actors[enemy]

        tgt_col = self._next_areagrab_target_col(actor)
        if tgt_col is None:
            self._log(f"{actor}: AREA_GRAB -> no target col available")
            return

        base_owner = _base_owner_for_col(tgt_col)

        # IMPORTANT:
        # Base ownership does NOT block stealing.
        # If this column is base-owned by you but currently held by the opponent,
        # AreaGrab should steal it back (restoring toward default).
        was_fully_base = _col_is_fully_owner(self.board.owners, self.board.tiles, tgt_col, base_owner)

        blocked = self._movement_buffer_block_indices(enemy, now)

        changed = False
        blocked_panels = 0

        for idx in _col_indices(tgt_col):
            if int(self.board.tiles[idx]) == TILE_HOLE_PERM:
                continue

            if int(idx) in blocked:
                # Cannot steal; deal 10 dmg + flinch 1 cust
                e.hp = max(0, int(e.hp) - AREAGRAB_BLOCK_DMG)
                e.locked_until = max(int(e.locked_until), int(now) + AREAGRAB_BLOCK_HITSTUN)
                blocked_panels += 1

                self.hot_panels.append(
                    HotPanelVisual(
                        actor=actor,
                        kind="areagrab_block",
                        idx=int(idx),
                        expires_cust=int(now) + 2,
                    )
                )
                continue

            if int(self.board.owners[idx]) != int(my_owner):
                self.board.owners[idx] = int(my_owner)
                changed = True

        if blocked_panels > 0:
            self._log(
                f"{actor}: AREA_GRAB col={tgt_col} blocked_panels={blocked_panels} "
                f"-> {enemy} took {blocked_panels * AREAGRAB_BLOCK_DMG}"
            )

        # Reconcile timers based on actual board state.
        # Start timer only if this column was fully base-owned and is now stolen.
        # (If it was already stolen/partial, do NOT reset.)
        self._reconcile_area_cols(range(COLS), now, allow_start=True)

        # If we just created a brand new stolen column, enforce the transition rule explicitly.
        # (This keeps behavior stable even if some future effect mutated owners without a timer.)
        if changed and was_fully_base and self._col_has_any_stolen(tgt_col):
            # It might already exist due to reconcile; just ensure it exists.
            self._reconcile_area_cols([tgt_col], now, allow_start=True)

        if changed:
            self._log(f"{actor}: AREA_GRAB applied to col={tgt_col} (base_owner={base_owner})")
        else:
            self._log(f"{actor}: AREA_GRAB col={tgt_col} stole 0 panels")

        # Note: expiry processing happens after events in advance_cust(),
        # and it will cascade inner expiries immediately if outer was removed.

    # ----------------------------
    # Cursor edits (server-authoritative)
    # ----------------------------
    def cursor_set_pending(self, actor: ActorId, action: Optional[ActionId]) -> None:
        self.actors[actor].pending_action = action
        self._validate()

    def cursor_set_hold(self, actor: ActorId, hold: bool) -> None:
        st = self.actors[actor]
        st.charge.set_hold(bool(hold))

    def cursor_toggle_hold(self, actor: ActorId) -> None:
        st = self.actors[actor]
        st.charge.set_hold(not st.charge.hold)

    # ----------------------------
    # Deterministic step
    # ----------------------------
    def advance_cust(self) -> None:
        t = self.cust

        # 0) queued release fires ASAP when actor is able to act
        for actor in ("P1", "P2"):
            st = self.actors[actor]
            if st.charge.queued_release and (not st.is_locked(t)):
                self._try_start_action(actor, "RELEASE_CHARGE", t)
                st.pending_action = None

        # 1) start pending for both actors at t
        for actor in ("P1", "P2"):
            st = self.actors[actor]
            if st.pending_action is not None:
                self._try_start_action(actor, st.pending_action, t)
            st.pending_action = None

        # 2) advance time
        self.cust = t + 1

        # 3) apply due events
        self._apply_due_events(self.cust)

        # 3.5) area expiry processing (after events)
        self._process_areagrab_expiry(self.cust)

        # 4) charge tick only while unlocked
        for actor in ("P1", "P2"):
            st = self.actors[actor]
            if st.is_locked(self.cust):
                continue
            st.charge.tick_unlocked()

        # 5) clear leans
        for actor in ("P1", "P2"):
            self.actors[actor].lean_dir_local = None

        self.shot_lines = [sl for sl in self.shot_lines if sl.alive(self.cust)]
        self.hot_panels = [hp for hp in self.hot_panels if hp.alive(self.cust)]

        self._validate()

    def _try_start_action(self, actor: ActorId, action_id: ActionId, t: int) -> None:
        st = self.actors[actor]

        if st.is_locked(t):
            self._log(f"{actor}: blocked (locked {st.locked_until - t} left)")
            return

        if action_id == ACT_USE_CHIP:
            self._start_use_chip(actor, t)
            return

        if action_id not in ACTIONS:
            self._log(f"{actor}: unknown action {action_id}")
            return

        if action_id == "SHOOT" and st.charge.level == 2:
            self._log(f"{actor}: blocked SHOOT at full charge (would be charge release)")
            return

        if action_id == "RELEASE_CHARGE" and st.charge.level != 2 and (not st.charge.queued_release):
            self._log(f"{actor}: blocked not full charge")
            return

        if action_id.startswith("MOVE_"):
            dir_local = action_id.split("_", 1)[1]
            if not self.can_move(actor, dir_local):
                self._log(f"{actor}: blocked cannot move {dir_local}")
                return

            # Track movement buffering for AreaGrab blocking (resolve at t+1)
            st.move_dir_local = dir_local
            st.move_due_cust = int(t) + 1

        spec = ACTIONS[action_id]

        st.locked_until = t + spec.lock_cust
        self.last_action_started = f"{actor}:{action_id}"

        for offset, ev in spec.timeline:
            self.scheduled.append(
                ScheduledEvent(
                    due_cust=t + offset,
                    actor=actor,
                    spec=ev,
                    source_action=action_id,
                )
            )

        self._apply_due_events(t)

        self._log(f"{actor}: start {action_id} (lock {spec.lock_cust}) form={form_name(st.form)}[{st.form}]")

    def _apply_due_events(self, now: int) -> None:
        if not self.scheduled:
            self._expire_entry(now)
            return

        due: List[ScheduledEvent] = []
        future: List[ScheduledEvent] = []
        for ev in self.scheduled:
            (due if ev.due_cust == now else future).append(ev)
        self.scheduled = future

        # Deterministic, rule-friendly ordering:
        # movement resolves before AreaGrab this cust
        def _prio(e: ScheduledEvent) -> int:
            if e.spec.type == "MOVE_APPLY":
                return 10
            if e.spec.type == "AREA_GRAB":
                return 20
            return 50

        due.sort(key=_prio)

        for ev in due:
            self._apply_event(ev.actor, ev.spec, now)

        self._expire_entry(now)

    def _expire_entry(self, now: int) -> None:
        for actor in ("P1", "P2"):
            st = self.actors[actor]
            if st.entry_dir_local is not None and now >= st.entry_until:
                st.entry_dir_local = None

    def _apply_event(self, actor: ActorId, ev: EventSpec, now: int) -> None:
        et = ev.type
        st = self.actors[actor]

        if et == "LEAN":
            dir_local = str(ev.payload.get("dir", ""))
            if dir_local in DIR_DELTAS:
                st.lean_dir_local = dir_local
                self._log(f"{actor}: event LEAN {dir_local}")
            return

        if et == "MOVE_APPLY":
            dir_local = str(ev.payload.get("dir", ""))
            if dir_local in DIR_DELTAS:
                self._apply_move(actor, dir_local, now)
                self._log(f"{actor}: event MOVE_APPLY {dir_local}")
            return

        if et == "RAY_SHOT":
            dmg = int(ev.payload.get("dmg", 0))
            reset_charge = bool(ev.payload.get("reset_charge", False))
            kind = str(ev.payload.get("shot_kind", "buster"))

            if kind == "charge":
                self._apply_charge_shot(actor, dmg, now)
            else:
                from_idx, to_idx = self._compute_raycast_line(actor)
                self.shot_lines.append(
                    ShotLineVisual(
                        actor=actor,
                        kind=kind,
                        from_idx=from_idx,
                        to_idx=to_idx,
                        expires_cust=now + 2,
                    )
                )
                self._raycast_and_damage(actor, dmg)

            if reset_charge:
                st.charge.reset()

            self._log(f"{actor}: event RAY_SHOT {kind} dmg={dmg} form={form_name(st.form)}[{st.form}]")
            return

        if et == "CHIP_RAY":
            dmg = int(ev.payload.get("dmg", 0))
            kind = str(ev.payload.get("kind", "chip"))
            push_step = int(ev.payload.get("push_step", 0))
            hitstun_cust = int(ev.payload.get("hitstun_cust", 0))
            chip_id = int(ev.payload.get("chip_id", -1))

            self._apply_chip_raycast(
                actor,
                dmg=dmg,
                kind=kind,
                push_step=push_step,
                hitstun_cust=hitstun_cust,
                now=now,
            )
            self._log(
                f"{actor}: event CHIP_RAY {chip_name(chip_id)}[{chip_id}] kind={kind} "
                f"dmg={dmg} push={push_step} hitstun={hitstun_cust}"
            )
            return

        if et == "AREA_GRAB":
            self._apply_areagrab(actor, now)
            return

        self._log(f"{actor}: event unknown {et}")

    def _log(self, msg: str) -> None:
        self.last_events.append(f"[cust {self.cust}] {msg}")
        if len(self.last_events) > 16:
            self.last_events = self.last_events[-16:]
