# viewer/derived_stream.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# Keep these aligned with viewer/derived_state.py
INVALID_CHIP_IDS = {255, 65535}

# -----------------------------------------------------------------------------
# Cross / Form mapping (idx 0..10). idx 0 is Normal (not consumable)
# -----------------------------------------------------------------------------
_CROSS_ROWS = [
    {"idx": 0, "name": "Normal", "normal": 0, "beast": 11},
    {"idx": 1, "name": "Fire",   "normal": 1, "beast": 13},
    {"idx": 2, "name": "Elec",   "normal": 2, "beast": 14},
    {"idx": 3, "name": "Slash",  "normal": 3, "beast": 15},
    {"idx": 4, "name": "Erase",  "normal": 4, "beast": 16},
    {"idx": 5, "name": "Charge", "normal": 5, "beast": 17},
    {"idx": 6, "name": "Aqua",   "normal": 6, "beast": 18},
    {"idx": 7, "name": "Thawk",  "normal": 7, "beast": 19},
    {"idx": 8, "name": "Tengu",  "normal": 8, "beast": 20},
    {"idx": 9, "name": "Grnd",   "normal": 9, "beast": 21},
    {"idx": 10, "name": "Dust",  "normal": 10, "beast": 22},
]

# -----------------------------------------------------------------------------
# Version-limited crosses (optional filter for planner)
# -----------------------------------------------------------------------------
_VERSION_CROSS_IDXS = {
    "Gregar": [1, 2, 3, 4, 5],      # Fire, Elec, Slash, Erase, Charge
    "Falzar": [6, 7, 8, 9, 10],     # Aqua, Thawk, Tengu, Grnd, Dust
}

def parse_bn6_version_from_rom_path(rom_path: Any) -> Optional[str]:
    """
    Your convention:
      "bn6,0" -> Gregar
      "bn6,1" -> Falzar
    Returns "Gregar"/"Falzar" or None if unknown.
    """
    if not rom_path:
        return None
    s = str(rom_path).strip().lower()
    # Expected "bn6,0" or "bn6,1" (allow extra commas after)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    if parts[0] != "bn6":
        return None
    if len(parts) < 2:
        return None
    try:
        ver = int(parts[1])
    except Exception:
        return None

    if ver == 0:
        return "Gregar"
    if ver == 1:
        return "Falzar"
    return None


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def _chip_valid(chip_id: Any) -> bool:
    try:
        cid = int(chip_id)
    except Exception:
        return False
    return cid not in INVALID_CHIP_IDS


def _cross_state_from_game_emotion(v: Any) -> Optional[Dict[str, Any]]:
    raw = _as_int(v, 0)
    for row in _CROSS_ROWS:
        if raw == row["normal"]:
            return {"idx": row["idx"], "name": row["name"], "beast": False, "raw": raw}
        if raw == row["beast"]:
            return {"idx": row["idx"], "name": row["name"], "beast": True, "raw": raw}
    return None


def _is_beast(cur: Optional[Dict[str, Any]]) -> bool:
    return bool(cur and cur.get("beast") is True)


def _init_cross_used_mask() -> List[bool]:
    return [False] * (max(r["idx"] for r in _CROSS_ROWS) + 1)


def _update_cross_tracking(
    *,
    cur_state: Optional[Dict[str, Any]],
    prev_state: Optional[Dict[str, Any]],
    used_mask: List[bool],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    event = {"entered": None, "exited": None}

    if cur_state is not None:
        idx = int(cur_state["idx"])
        if idx >= 1 and 0 <= idx < len(used_mask):
            used_mask[idx] = True

    def key(s: Optional[Dict[str, Any]]) -> Tuple[int, bool]:
        if not s:
            return (-1, False)
        return (int(s["idx"]), bool(s["beast"]))

    if key(prev_state) != key(cur_state):
        if prev_state is not None:
            event["exited"] = dict(prev_state)
        if cur_state is not None:
            event["entered"] = dict(cur_state)

    return cur_state, event


def _extract_selection(frame: Dict[str, Any]) -> Tuple[bool, bool, List[int]]:
    idxs = [_as_int(x, 255) for x in _as_list(frame.get("selected_chip_indices"))]
    count = _as_int(frame.get("chip_select_count"), 0)

    # Authoritative: 0 means "no chips selected" (carry held hand).
    if count == 0:
        considered: List[int] = []
    else:
        considered = idxs[:count] if 0 < count <= len(idxs) else idxs

    beast_selected = 11 in considered

    slot_indices: List[int] = []
    for x in considered:
        if 0 <= x < 10:
            slot_indices.append(int(x))

    selected_any = beast_selected or (len(slot_indices) > 0)
    return selected_any, beast_selected, slot_indices


def _selected_chips_from_window(frame: Dict[str, Any], slot_indices: List[int]) -> List[Dict[str, int]]:
    chip_slots = _as_list(frame.get("chip_slots"))
    chip_codes = _as_list(frame.get("chip_codes"))

    out: List[Dict[str, int]] = []
    for si in slot_indices:
        if si < 0 or si >= len(chip_slots):
            continue
        raw_id = _as_int(chip_slots[si], 255)
        if raw_id in INVALID_CHIP_IDS:
            continue
        raw_code = _as_int(chip_codes[si], 0) if si < len(chip_codes) else 0
        out.append({"id": int(raw_id), "code": int(raw_code)})
    return out


def _has_any_selection(frame: Dict[str, Any]) -> bool:
    idxs = [_as_int(x, 255) for x in _as_list(frame.get("selected_chip_indices"))]
    if not idxs:
        return False

    count = _as_int(frame.get("chip_select_count"), 0)
    if count == 0:
        return False

    considered = idxs[:count] if 0 < count <= len(idxs) else idxs
    if 11 in considered:
        return True
    return any(0 <= x < 10 for x in considered)


def _ok_pred(f: Dict[str, Any]) -> bool:
    return (
        _as_bool(f.get("inside_window"), False)
        and _as_int(f.get("cust_gauge"), -1) == 0
        and _as_int(f.get("selected_menu_index"), -1) == 10
    )


def _find_window_commit_snapshot_from_buffer(
    window_frames: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, Dict[str, Any], int, Dict[str, Any], int]:
    """
    Same intent as viewer/derived_state._find_window_commit_snapshot(), but operates on the
    buffered in-window frames only (index space is 0..len(buffer)-1).
    """
    if not window_frames:
        return {}, -1, {}, -1, {}, -1

    end_idx = len(window_frames) - 1

    ok_end = -1
    i = end_idx
    while i >= 0:
        f = window_frames[i]
        if not _as_bool(f.get("inside_window"), False):
            break
        if _ok_pred(f):
            ok_end = i
            break
        i -= 1

    if ok_end < 0:
        ok_start = end_idx
        ok_start_frame = window_frames[ok_start]
        return ok_start_frame, ok_start, ok_start_frame, ok_start, ok_start_frame, ok_start

    ok_start = ok_end
    j = ok_end - 1
    while j >= 0:
        f = window_frames[j]
        if not _as_bool(f.get("inside_window"), False):
            break
        if not _ok_pred(f):
            break
        ok_start = j
        j -= 1

    ok_start_frame = window_frames[ok_start]
    ok_end_frame = window_frames[ok_end]

    if _has_any_selection(ok_start_frame):
        return ok_start_frame, ok_start, ok_end_frame, ok_end, ok_start_frame, ok_start

    for k in range(ok_start, ok_end + 1):
        f = window_frames[k]
        if not _ok_pred(f):
            continue
        if _has_any_selection(f):
            return ok_start_frame, ok_start, ok_end_frame, ok_end, f, k

    k = ok_start - 1
    while k >= 0:
        f = window_frames[k]
        if not _as_bool(f.get("inside_window"), False):
            break
        if _as_int(f.get("cust_gauge"), -1) != 0:
            k -= 1
            continue
        if _has_any_selection(f):
            return ok_start_frame, ok_start, ok_end_frame, ok_end, f, k
        k -= 1

    return ok_start_frame, ok_start, ok_end_frame, ok_end, ok_start_frame, ok_start


def _mark_used_in_folder(folder_ids: List[int], used_mask: List[bool], used_chip_id: int) -> None:
    for i, cid in enumerate(folder_ids):
        if not used_mask[i] and cid == used_chip_id:
            used_mask[i] = True
            return


def _remove_from_held(held: List[Dict[str, int]], used_chip_id: int) -> None:
    for i, ch in enumerate(held):
        if _as_int(ch.get("id"), -1) == used_chip_id:
            held.pop(i)
            return


def _drop_held_prefix_until_current(held: List[Dict[str, int]], current_chip_id: int) -> List[int]:
    if not held:
        return []
    target = int(current_chip_id)
    idx = -1
    for i, ch in enumerate(held):
        if _as_int(ch.get("id"), -1) == target:
            idx = i
            break
    if idx <= 0:
        return []

    dropped_ids: List[int] = []
    for _ in range(idx):
        dropped_ids.append(_as_int(held.pop(0).get("id"), -1))
    return [x for x in dropped_ids if x >= 0]


@dataclass
class _FolderTracker:
    ids: List[int]
    used_mask: List[bool]

    @classmethod
    def empty(cls) -> "_FolderTracker":
        return cls(ids=[], used_mask=[])

    def ensure_ids(self, ids: List[int]) -> None:
        if self.ids:
            return
        self.ids = [int(x) for x in ids]
        self.used_mask = [False] * len(self.ids)


class DerivedStateTracker:
    """
    Streaming equivalent of compute_derived().

    Call update() once per incoming frame; it returns the derived dict for *that* frame.
    """

    def __init__(self, *, game_version: Optional[str] = None, rom_path: Any = None):
        # Prefer explicit if given, else parse from rom_path.
        gv = (game_version or "").strip() or None
        if gv is None:
            gv = parse_bn6_version_from_rom_path(rom_path)
        self.game_version = gv
        self.reset()

    def reset(self) -> None:
        self.prev_inside: bool = False
        self.turn_index: int = 0

        self.player_folder = _FolderTracker.empty()
        self.enemy_folder = _FolderTracker.empty()

        self.prev_player_chip: Optional[int] = None
        self.prev_enemy_chip: Optional[int] = None

        self.player_used_cross_mask: List[bool] = _init_cross_used_mask()
        self.enemy_used_cross_mask: List[bool] = _init_cross_used_mask()
        self.prev_player_cross: Optional[Dict[str, Any]] = None
        self.prev_enemy_cross: Optional[Dict[str, Any]] = None

        self.player_beast_ever: bool = False
        self.enemy_beast_ever: bool = False
        self.player_last_beast_turn: Optional[int] = None
        self.enemy_last_beast_turn: Optional[int] = None
        self.player_last_beast_frame: Optional[int] = None
        self.enemy_last_beast_frame: Optional[int] = None

        self.held: List[Dict[str, int]] = []

        # Buffer only frames while inside_window=True (index local to buffer)
        self._window_buf: Deque[Dict[str, Any]] = deque()

        # Streaming frame index (monotonic, for “last_frame” fields)
        self._frame_i: int = -1

    def _available_cross_idxs(self) -> Optional[List[int]]:
        if not self.game_version:
            return None
        return list(_VERSION_CROSS_IDXS.get(self.game_version, [])) or None

    def update(self, frame: Dict[str, Any], *, static: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._frame_i += 1
        i = self._frame_i

        static = static or {}

        # Learn folders once (from live cache or server)
        pf = _as_list(static.get("player_folder_ids")) or _as_list(static.get("cached_player_folder"))
        ef = _as_list(static.get("enemy_folder_ids")) or _as_list(static.get("cached_enemy_folder"))
        if pf:
            self.player_folder.ensure_ids([_as_int(x, 255) for x in pf])
        if ef:
            self.enemy_folder.ensure_ids([_as_int(x, 255) for x in ef])

        inside = _as_bool(frame.get("inside_window"), False)

        # Turn boundary: window closed AND *current* frame has cust_gauge==0
        if self.prev_inside and (not inside):
            if _as_int(frame.get("cust_gauge"), -1) == 0:
                self.turn_index += 1

        # Cross tracking
        cur_player_cross = _cross_state_from_game_emotion(frame.get("player_game_emotion"))
        cur_enemy_cross = _cross_state_from_game_emotion(frame.get("enemy_game_emotion"))

        prev_p_beast = _is_beast(self.prev_player_cross)
        prev_e_beast = _is_beast(self.prev_enemy_cross)
        cur_p_beast = _is_beast(cur_player_cross)
        cur_e_beast = _is_beast(cur_enemy_cross)

        if (not prev_p_beast) and cur_p_beast:
            self.player_beast_ever = True
            self.player_last_beast_turn = int(self.turn_index)
            self.player_last_beast_frame = int(i)

        if (not prev_e_beast) and cur_e_beast:
            self.enemy_beast_ever = True
            self.enemy_last_beast_turn = int(self.turn_index)
            self.enemy_last_beast_frame = int(i)

        self.prev_player_cross, player_cross_event = _update_cross_tracking(
            cur_state=cur_player_cross,
            prev_state=self.prev_player_cross,
            used_mask=self.player_used_cross_mask,
        )
        self.prev_enemy_cross, enemy_cross_event = _update_cross_tracking(
            cur_state=cur_enemy_cross,
            prev_state=self.prev_enemy_cross,
            used_mask=self.enemy_used_cross_mask,
        )

        # Chip usage (battle only)
        used_player_chip_id: Optional[int] = None
        used_enemy_chip_id: Optional[int] = None

        if not inside:
            # Player
            curp_valid: Optional[int] = _as_int(frame.get("player_chip")) if _chip_valid(frame.get("player_chip")) else None
            if self.prev_player_chip is None:
                if curp_valid is not None:
                    self.prev_player_chip = curp_valid
            else:
                if curp_valid != self.prev_player_chip:
                    used_player_chip_id = self.prev_player_chip
                    if self.player_folder.ids and used_player_chip_id is not None:
                        _mark_used_in_folder(self.player_folder.ids, self.player_folder.used_mask, used_player_chip_id)
                    if used_player_chip_id is not None:
                        _remove_from_held(self.held, used_player_chip_id)
                self.prev_player_chip = curp_valid

            # Modifier-chain rule
            if curp_valid is not None and self.held:
                dropped_ids = _drop_held_prefix_until_current(self.held, curp_valid)
                for did in dropped_ids:
                    if self.player_folder.ids and did is not None:
                        _mark_used_in_folder(self.player_folder.ids, self.player_folder.used_mask, did)

            # Empty-hand rule
            cust = _as_int(frame.get("cust_gauge"), 0)
            if cust > 0 and _as_int(frame.get("player_chip"), 65535) == 65535:
                self.held.clear()

            # Enemy
            cure_valid: Optional[int] = _as_int(frame.get("enemy_chip")) if _chip_valid(frame.get("enemy_chip")) else None
            if self.prev_enemy_chip is None:
                if cure_valid is not None:
                    self.prev_enemy_chip = cure_valid
            else:
                if cure_valid != self.prev_enemy_chip:
                    used_enemy_chip_id = self.prev_enemy_chip
                    if self.enemy_folder.ids and used_enemy_chip_id is not None:
                        _mark_used_in_folder(self.enemy_folder.ids, self.enemy_folder.used_mask, used_enemy_chip_id)
                self.prev_enemy_chip = cure_valid

        # Chip-window buffering + commit on close
        window_commit: Dict[str, Any] = {
            "happened": False,
            "selected_any": False,
            "beast_selected": False,
            "selected_chips": [],
            "source": "",
            "ok_start_idx": -1,
            "ok_end_idx": -1,
            "sel_idx": -1,
            "close_idx": -1,
            "close_chip_select_count": 0,
        }

        if inside:
            # Buffer only relevant fields to avoid accidental bloat
            self._window_buf.append(
                {
                    "inside_window": True,
                    "cust_gauge": frame.get("cust_gauge"),
                    "selected_menu_index": frame.get("selected_menu_index"),
                    "chip_select_count": frame.get("chip_select_count"),
                    "selected_chip_indices": frame.get("selected_chip_indices"),
                    "chip_slots": frame.get("chip_slots"),
                    "chip_codes": frame.get("chip_codes"),
                }
            )

        if self.prev_inside and (not inside):
            buf = list(self._window_buf)
            self._window_buf.clear()

            close_idx = len(buf) - 1
            close_frame = buf[close_idx] if close_idx >= 0 else {}
            close_count = _as_int(close_frame.get("chip_select_count"), 0)

            ok_start_frame, ok_start_idx, ok_end_frame, ok_end_idx, sel_frame, sel_idx = _find_window_commit_snapshot_from_buffer(buf)

            if close_count == 0:
                window_commit = {
                    "happened": True,
                    "selected_any": False,
                    "beast_selected": False,
                    "selected_chips": [],
                    "source": f"closeCount0@{close_idx}|okStart@{ok_start_idx}|okEnd@{ok_end_idx}|sel@{sel_idx}",
                    "ok_start_idx": int(ok_start_idx),
                    "ok_end_idx": int(ok_end_idx),
                    "sel_idx": int(sel_idx),
                    "close_idx": int(close_idx),
                    "close_chip_select_count": int(close_count),
                }
            else:
                selected_any, beast_selected, slot_indices = _extract_selection(sel_frame)
                selected_chips = _selected_chips_from_window(sel_frame, slot_indices)

                window_commit = {
                    "happened": True,
                    "selected_any": bool(selected_any),
                    "beast_selected": bool(beast_selected),
                    "selected_chips": list(selected_chips),
                    "source": f"closeCount@{close_idx}={close_count}|okStart@{ok_start_idx}|okEnd@{ok_end_idx}|sel@{sel_idx}",
                    "ok_start_idx": int(ok_start_idx),
                    "ok_end_idx": int(ok_end_idx),
                    "sel_idx": int(sel_idx),
                    "close_idx": int(close_idx),
                    "close_chip_select_count": int(close_count),
                }

                if selected_any:
                    self.held = list(selected_chips)

        def _turns_since(last_turn: Optional[int]) -> Optional[int]:
            if last_turn is None:
                return None
            return int(self.turn_index) - int(last_turn)

        derived = {
            "turn_index": int(self.turn_index),
            "player": {
                "folder_used_mask": list(self.player_folder.used_mask),
                "held_chips": list(self.held),
                "window_commit": window_commit,
                "used_chip_id": used_player_chip_id,
                "used_cross_mask": list(self.player_used_cross_mask),
                "active_cross": dict(cur_player_cross) if cur_player_cross else None,
                "cross_event": dict(player_cross_event),
                "beast": {
                    "active": bool(cur_p_beast),
                    "ever": bool(self.player_beast_ever),
                    "last_turn": int(self.player_last_beast_turn) if self.player_last_beast_turn is not None else None,
                    "turns_since": _turns_since(self.player_last_beast_turn),
                    "last_frame": int(self.player_last_beast_frame) if self.player_last_beast_frame is not None else None,
                },
            },
            "enemy": {
                "folder_used_mask": list(self.enemy_folder.used_mask),
                "used_chip_id": used_enemy_chip_id,
                "used_cross_mask": list(self.enemy_used_cross_mask),
                "active_cross": dict(cur_enemy_cross) if cur_enemy_cross else None,
                "cross_event": dict(enemy_cross_event),
                "beast": {
                    "active": bool(cur_e_beast),
                    "ever": bool(self.enemy_beast_ever),
                    "last_turn": int(self.enemy_last_beast_turn) if self.enemy_last_beast_turn is not None else None,
                    "turns_since": _turns_since(self.enemy_last_beast_turn),
                    "last_frame": int(self.enemy_last_beast_frame) if self.enemy_last_beast_frame is not None else None,
                },
            },
            # Planner-facing constraint
            "available_cross_idxs": self._available_cross_idxs(),
            "game_version": self.game_version,
        }

        self.prev_inside = inside
        return derived
