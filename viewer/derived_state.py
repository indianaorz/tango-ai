# viewer/derived_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# These appear in your captures for "no chip" / invalid slots.
INVALID_CHIP_IDS = {255, 65535}

# These appear in your captures for "no chip" / invalid slots.
INVALID_CHIP_IDS = {255, 65535}

# -----------------------------------------------------------------------------
# Cross / Form mapping from game_emotion ids (mirrors your JS mapping)
#
# We treat "cross used" as: observed actor in a cross form at least once.
# Cross index 0 is "Normal" (not a consumable cross); we track it but do NOT
# mark it as "used".
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
    {"idx": 10,"name": "Dust",   "normal": 10,"beast": 22},
]
def _is_beast(cur: Optional[Dict[str, Any]]) -> bool:
    return bool(cur and cur.get("beast") is True)

def _cross_state_from_game_emotion(v: Any) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {
        "idx": int,          # 0..10
        "name": str,         # e.g. "Fire"
        "beast": bool,       # True if beast variant
        "raw": int,          # raw game_emotion id
      }
    Or None if unknown/unmapped.
    """
    raw = _as_int(v, 0)
    for row in _CROSS_ROWS:
        if raw == row["normal"]:
            return {"idx": row["idx"], "name": row["name"], "beast": False, "raw": raw}
        if raw == row["beast"]:
            return {"idx": row["idx"], "name": row["name"], "beast": True, "raw": raw}
    return None

def _init_cross_used_mask() -> List[bool]:
    # index == cross idx (0..10). idx 0 ("Normal") is not consumable, but kept for alignment.
    return [False] * (max(r["idx"] for r in _CROSS_ROWS) + 1)

def _update_cross_tracking(
    *,
    cur_state: Optional[Dict[str, Any]],
    prev_state: Optional[Dict[str, Any]],
    used_mask: List[bool],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Updates used_mask in-place and returns:
      new_prev_state,
      event = {"entered": <state or None>, "exited": <state or None>}
    """
    event = {"entered": None, "exited": None}

    # If we can identify the current cross and it's a real cross (idx>=1), mark used.
    if cur_state is not None:
        idx = int(cur_state["idx"])
        if idx >= 1 and 0 <= idx < len(used_mask):
            used_mask[idx] = True

    # Transition detection (entered/exited) based on idx/beast changes.
    def key(s: Optional[Dict[str, Any]]) -> Tuple[int, bool]:
        if not s:
            return (-1, False)
        return (int(s["idx"]), bool(s["beast"]))

    pk = key(prev_state)
    ck = key(cur_state)
    if pk != ck:
        if prev_state is not None:
            event["exited"] = dict(prev_state)
        if cur_state is not None:
            event["entered"] = dict(cur_state)

    return cur_state, event


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


def _extract_selection(frame: Dict[str, Any]) -> Tuple[bool, bool, List[int]]:
    """
    Returns:
      selected_any: bool  (chips selected OR beast out)
      beast_selected: bool
      selected_chip_slot_indices: List[int]  (0..9 indices into chip_slots/chip_codes)

    NOTE:
      In your pipeline, "beast out" may be represented as index 11 in selected_chip_indices.
    """
    idxs_raw = frame.get("selected_chip_indices")
    idxs = [_as_int(x, 255) for x in _as_list(idxs_raw)]

    count = _as_int(frame.get("chip_select_count"), 0)

    # chip_select_count == 0 is the "no chips selected" case (carry held hand).
    # Even if selected_chip_indices still has stale values, we must NOT treat them as a selection.
    if count == 0:
        considered: List[int] = []
    else:
        # If count is clearly bogus (negative or > len), fall back to scanning the whole list.
        considered = idxs[:count] if 0 < count <= len(idxs) else idxs

    beast_selected = 11 in considered


    slot_indices: List[int] = []
    for x in considered:
        if 0 <= x < 10:
            slot_indices.append(int(x))

    selected_any = beast_selected or (len(slot_indices) > 0)
    return selected_any, beast_selected, slot_indices


def _selected_chips_from_window(frame: Dict[str, Any], slot_indices: List[int]) -> List[Dict[str, int]]:
    """
    Convert selected slot indices into a stable list of chips for the held-hand:
      [{"id": <chip_id>, "code": <raw_code>}, ...]
    """
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


def _mark_used_in_folder(folder_ids: List[int], used_mask: List[bool], used_chip_id: int) -> None:
    """
    Mark the first not-yet-used occurrence of chip_id in the static folder.
    """
    for i, cid in enumerate(folder_ids):
        if not used_mask[i] and cid == used_chip_id:
            used_mask[i] = True
            return


def _remove_from_held(held: List[Dict[str, int]], used_chip_id: int) -> None:
    """
    Remove one occurrence of used_chip_id from held (duplicates allowed).
    Match by chip id only (we usually don't know code at usage time).
    """
    for i, ch in enumerate(held):
        if _as_int(ch.get("id"), -1) == used_chip_id:
            held.pop(i)
            return


def _has_any_selection(frame: Dict[str, Any]) -> bool:
    """
    True if this frame still contains a non-empty selected queue OR beast selection.

    Robustness:
      - chip_select_count can go stale/0 while selected_chip_indices still contains meaningful values.
      - We treat any 0..9 as a chip selection, and 11 as beast-out.
    """
    idxs_raw = frame.get("selected_chip_indices")
    idxs = [_as_int(x, 255) for x in _as_list(idxs_raw)]
    if not idxs:
        return False

    count = _as_int(frame.get("chip_select_count"), 0)

    # Authoritative: 0 means no selection (carry)
    if count == 0:
        return False

    considered = idxs[:count] if 0 < count <= len(idxs) else idxs

    if 11 in considered:
        return True
    for x in considered:
        if 0 <= x < 10:
            return True
    return False



def _ok_pred(f: Dict[str, Any]) -> bool:
    """
    "OK button" frame inside the chip window:
      - inside_window True
      - cust_gauge == 0
      - selected_menu_index == 10  (OK)
    """
    return (
        _as_bool(f.get("inside_window"), False)
        and _as_int(f.get("cust_gauge"), -1) == 0
        and _as_int(f.get("selected_menu_index"), -1) == 10
    )


def _find_window_commit_snapshot(
    frames: List[Dict[str, Any]],
    end_idx: int,
) -> Tuple[Dict[str, Any], int, Dict[str, Any], int, Dict[str, Any], int]:
    """
    We need the snapshot frame to map selected_chip_indices -> chip_slots/chip_codes.
    But two capture quirks exist:

      1) selected queue can clear BEFORE inside_window flips false.
      2) selected_menu_index can stay at 10 even after "close", so "last OK frame"
         is NOT a good anchor. Instead we want:
           - the FIRST frame of the FINAL contiguous run where selected_menu_index == 10
             (while inside_window & cust_gauge==0). This is when the OK state is first entered,
             and (in your data) still has the selected_chip_indices intact.

    Returns:
      ok_start_frame, ok_start_idx,
      ok_end_frame,   ok_end_idx,
      sel_frame,      sel_idx

    sel_frame is the frame we will use to read selection (usually ok_start).
    """
    if not frames:
        return {}, -1, {}, -1, {}, -1

    if end_idx < 0:
        return {}, -1, {}, -1, {}, -1
    if end_idx >= len(frames):
        end_idx = len(frames) - 1

    # Ensure end_idx is inside-window; if not, walk backward to last inside-window frame.
    i = end_idx
    while i >= 0 and not _as_bool(frames[i].get("inside_window"), False):
        i -= 1
    if i < 0:
        return {}, -1, {}, -1, {}, -1
    end_idx = i

    # 1) Find ok_end: last frame in-window satisfying OK predicate.
    ok_end = -1
    i = end_idx
    while i >= 0:
        f = frames[i]
        if not _as_bool(f.get("inside_window"), False):
            break
        if _ok_pred(f):
            ok_end = i
            break
        i -= 1

    # If we never find OK predicate, fall back to end_idx (still in window).
    if ok_end < 0:
        ok_start = end_idx
        ok_start_frame = frames[ok_start]
        return ok_start_frame, ok_start, ok_start_frame, ok_start, ok_start_frame, ok_start

    # 2) Walk backward to find ok_start: first frame of the final contiguous OK run.
    ok_start = ok_end
    j = ok_end - 1
    while j >= 0:
        f = frames[j]
        if not _as_bool(f.get("inside_window"), False):
            break
        if not _ok_pred(f):
            break
        ok_start = j
        j -= 1

    ok_start_frame = frames[ok_start]
    ok_end_frame = frames[ok_end]

    # 3) Choose selection snapshot:
    #    Prefer ok_start (user-requested). If it somehow has no selection, try forward within the OK run
    #    for the first frame that has selection. If still none, try backward (still in-window, cust==0)
    #    for a frame with selection.
    if _has_any_selection(ok_start_frame):
        return ok_start_frame, ok_start, ok_end_frame, ok_end, ok_start_frame, ok_start

    # Search forward within [ok_start..ok_end]
    for k in range(ok_start, ok_end + 1):
        f = frames[k]
        if not _ok_pred(f):
            continue
        if _has_any_selection(f):
            return ok_start_frame, ok_start, ok_end_frame, ok_end, f, k

    # Search backward before ok_start (still inside window, cust==0)
    k = ok_start - 1
    while k >= 0:
        f = frames[k]
        if not _as_bool(f.get("inside_window"), False):
            break
        if _as_int(f.get("cust_gauge"), -1) != 0:
            k -= 1
            continue
        if _has_any_selection(f):
            return ok_start_frame, ok_start, ok_end_frame, ok_end, f, k
        k -= 1

    # Last resort: ok_start
    return ok_start_frame, ok_start, ok_end_frame, ok_end, ok_start_frame, ok_start


@dataclass
class _FolderTracker:
    ids: List[int]
    used_mask: List[bool]

    @classmethod
    def from_static(cls, static: Dict[str, Any], prefix: str) -> "_FolderTracker":
        key = f"{prefix}_folder_ids"
        raw = _as_list(static.get(key))
        ids = [_as_int(x, 255) for x in raw]
        used_mask = [False] * len(ids)
        return cls(ids=ids, used_mask=used_mask)

def _drop_held_prefix_until_current(held: List[Dict[str, int]], current_chip_id: int) -> List[int]:
    """
    If current_chip_id appears later in held[], drop everything before the first
    occurrence and return the dropped chip ids.

    This models "modifier chips" that get consumed alongside an earlier chip:
      held = [Cannon, Atk+10, ZapRing]
      current becomes ZapRing -> drop [Cannon, Atk+10]
    """
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
    # Filter any weird -1s just in case
    return [x for x in dropped_ids if x >= 0]


def compute_derived(frames: List[Dict[str, Any]], static: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Derived per-frame state.

    Player:
      - folder_used_mask (based on player_chip changes during battle)
      - held_chips (persist across turns if no selection on window close)
      - window_commit (metadata at close frame, plus debug source indices)
    Enemy:
      - folder_used_mask (based on enemy_chip changes during battle)

    Semantics:
      - A "commit" happens when the chip window closes AND the player had
        (selected chips OR beast out).
      - If there is a commit, previous held_chips are discarded and replaced by the committed selection.
        If it's beast-only commit (no chips chosen), held becomes empty (intentional discard).
      - If there is NO commit (no selection and no beast), held_chips carry across to the next battle segment.
      - We remove chips from held_chips when we observe them used during battle (via player_chip transitions).
      - Enemy chip-window isn't visible; we only mark used_mask via enemy_chip transitions.

    IMPORTANT:
      Your capture can clear selected queue BEFORE inside_window flips false.
      Also, selected_menu_index can stick at 10, so we snapshot from:
        - the FIRST frame of the FINAL contiguous OK-run (menu==10, cust==0, inside_window True),
          which (in your data) still has selected_chip_indices.
    """
    static = static or {}
    player_folder = _FolderTracker.from_static(static, "player")
    enemy_folder = _FolderTracker.from_static(static, "enemy")

    derived: List[Dict[str, Any]] = []

    prev_inside: bool = False
    prev_player_chip: Optional[int] = None
    prev_enemy_chip: Optional[int] = None

    # Turn index: increments at each window close (inside_window True -> False).
    # This gives a stable "how many turns ago" timeline for strategy.
    turn_index: int = 0

    # Persistent cross usage tracking
    player_used_cross_mask: List[bool] = _init_cross_used_mask()
    enemy_used_cross_mask: List[bool] = _init_cross_used_mask()
    prev_player_cross: Optional[Dict[str, Any]] = None
    prev_enemy_cross: Optional[Dict[str, Any]] = None

    # Beast tracking (derived from *_game_emotion entering a beast variant)
    player_beast_ever: bool = False
    enemy_beast_ever: bool = False
    player_last_beast_turn: Optional[int] = None
    enemy_last_beast_turn: Optional[int] = None
    player_last_beast_frame: Optional[int] = None
    enemy_last_beast_frame: Optional[int] = None

    # The persistent "cards held" during battle turns
    held: List[Dict[str, int]] = []



    for i, f in enumerate(frames):
        inside = _as_bool(f.get("inside_window"), False)

        # Turn boundary only when window closes AND cust_gauge == 0.
        # This avoids counting spurious inside_window transitions.
        if prev_inside and not inside:
            if _as_int(f.get("cust_gauge"), -1) == 0:
                turn_index += 1



        # --- Cross tracking (both sides) ---
        cur_player_cross = _cross_state_from_game_emotion(f.get("player_game_emotion"))
        cur_enemy_cross = _cross_state_from_game_emotion(f.get("enemy_game_emotion"))

        # Beast trigger = entering a beast variant from a non-beast variant.
        prev_p_beast = _is_beast(prev_player_cross)
        prev_e_beast = _is_beast(prev_enemy_cross)
        cur_p_beast = _is_beast(cur_player_cross)
        cur_e_beast = _is_beast(cur_enemy_cross)

        if (not prev_p_beast) and cur_p_beast:
            player_beast_ever = True
            player_last_beast_turn = int(turn_index)
            player_last_beast_frame = int(i)

        if (not prev_e_beast) and cur_e_beast:
            enemy_beast_ever = True
            enemy_last_beast_turn = int(turn_index)
            enemy_last_beast_frame = int(i)

        prev_player_cross, player_cross_event = _update_cross_tracking(
            cur_state=cur_player_cross,
            prev_state=prev_player_cross,
            used_mask=player_used_cross_mask,
        )
        prev_enemy_cross, enemy_cross_event = _update_cross_tracking(
            cur_state=cur_enemy_cross,
            prev_state=prev_enemy_cross,
            used_mask=enemy_used_cross_mask,
        )


        # --- Detect chip usage (battle only) ---
        used_player_chip_id: Optional[int] = None
        used_enemy_chip_id: Optional[int] = None


        cur_player_chip = f.get("player_chip")
        cur_enemy_chip = f.get("enemy_chip")

        if not inside:
            # Normalize current chip: None means "no chip active" (65535 / 255 / invalid)
            curp_valid: Optional[int] = _as_int(cur_player_chip) if _chip_valid(cur_player_chip) else None

            if prev_player_chip is None:
                # Initialize once we see the first valid chip in battle
                if curp_valid is not None:
                    prev_player_chip = curp_valid
            else:
                # Any change away from the previous valid chip counts as "used previous chip",
                # INCLUDING transitioning to invalid/65535 (meaning you ran out / no hand left).
                if curp_valid != prev_player_chip:
                    used_player_chip_id = prev_player_chip
                    _mark_used_in_folder(player_folder.ids, player_folder.used_mask, used_player_chip_id)
                    _remove_from_held(held, used_player_chip_id)

                # Update prev to the new state (None when 65535)
                prev_player_chip = curp_valid

            # -----------------------------------------------------------------
            # Modifier-chain rule:
            # If the currently-active chip is not the first held chip, but appears
            # later in held[], drop the prefix as "consumed" (chip + modifiers).
            # -----------------------------------------------------------------
            if curp_valid is not None and held:
                dropped_ids = _drop_held_prefix_until_current(held, curp_valid)
                for did in dropped_ids:
                    _mark_used_in_folder(player_folder.ids, player_folder.used_mask, did)

            # -----------------------------------------------------------------
            # Empty-hand rule (battle-only):
            # If cust_gauge > 0 and player_chip is the "empty hand" sentinel (65535),
            # we force held to empty (you have nothing left to carry).
            # -----------------------------------------------------------------
            cust = _as_int(f.get("cust_gauge"), 0)
            if cust > 0 and _as_int(cur_player_chip, 65535) == 65535:
                held.clear()




        # --- Detect chip-window close: inside_window True -> False ---
        window_commit: Dict[str, Any] = {
            "happened": False,
            "selected_any": False,
            "beast_selected": False,
            "selected_chips": [],
            "source": "",
            "ok_start_idx": -1,
            "ok_end_idx": -1,
            "sel_idx": -1,
        }

        if prev_inside and not inside:
            ok_start_frame, ok_start_idx, ok_end_frame, ok_end_idx, sel_frame, sel_idx = _find_window_commit_snapshot(
                frames, i - 1
            )

            selected_any, beast_selected, slot_indices = _extract_selection(sel_frame)
            selected_chips = _selected_chips_from_window(sel_frame, slot_indices)

            window_commit = {
                "happened": True,
                "selected_any": bool(selected_any),
                "beast_selected": bool(beast_selected),
                "selected_chips": list(selected_chips),
                "source": f"okStart@{ok_start_idx}|okEnd@{ok_end_idx}|sel@{sel_idx}",
                "ok_start_idx": int(ok_start_idx),
                "ok_end_idx": int(ok_end_idx),
                "sel_idx": int(sel_idx),
            }

            if selected_any:
                held = list(selected_chips)

        def _turns_since(last_turn: Optional[int]) -> Optional[int]:
            if last_turn is None:
                return None
            return int(turn_index) - int(last_turn)

        derived.append(
            {
                "turn_index": int(turn_index),

                "player": {
                    "folder_used_mask": list(player_folder.used_mask),
                    "held_chips": list(held),
                    "window_commit": window_commit,
                    "used_chip_id": used_player_chip_id,

                    # Cross usage tracking
                    "used_cross_mask": list(player_used_cross_mask),          # idx 0..10
                    "active_cross": dict(cur_player_cross) if cur_player_cross else None,
                    "cross_event": dict(player_cross_event),

                    # Beast tracking
                    "beast": {
                        "active": bool(cur_p_beast),
                        "ever": bool(player_beast_ever),
                        "last_turn": int(player_last_beast_turn) if player_last_beast_turn is not None else None,
                        "turns_since": _turns_since(player_last_beast_turn),
                        "last_frame": int(player_last_beast_frame) if player_last_beast_frame is not None else None,
                    },
                },
                "enemy": {
                    "folder_used_mask": list(enemy_folder.used_mask),
                    "used_chip_id": used_enemy_chip_id,

                    # Cross usage tracking
                    "used_cross_mask": list(enemy_used_cross_mask),           # idx 0..10
                    "active_cross": dict(cur_enemy_cross) if cur_enemy_cross else None,
                    "cross_event": dict(enemy_cross_event),

                    # Beast tracking
                    "beast": {
                        "active": bool(cur_e_beast),
                        "ever": bool(enemy_beast_ever),
                        "last_turn": int(enemy_last_beast_turn) if enemy_last_beast_turn is not None else None,
                        "turns_since": _turns_since(enemy_last_beast_turn),
                        "last_frame": int(enemy_last_beast_frame) if enemy_last_beast_frame is not None else None,
                    },
                },
            }
        )



        prev_inside = inside

    return derived


def compute_derived_states(frames: List[Dict[str, Any]], static: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Public entrypoint expected by viewer/app.py.
    """
    return compute_derived(frames, static)


__all__ = [
    "compute_derived_states",
    "compute_derived",
]
