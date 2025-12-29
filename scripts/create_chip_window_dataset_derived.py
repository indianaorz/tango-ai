#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Import derived-state (single source of truth for chip-window commits)
# ---------------------------------------------------------------------
try:
    from derived_state import compute_derived
except Exception as e:
    raise SystemExit(
        "Failed to import derived_state.compute_derived. "
        "Run this script from repo root or ensure PYTHONPATH includes the project.\n"
        f"Import error: {e}"
    )

INVALID_CHIP_IDS = {255, 65535}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            x = json.load(f)
            return x if isinstance(x, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        out.append(row)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _find_replay_dirs(root: Path) -> List[Path]:
    if root.is_file() and root.name == "actions.jsonl":
        return [root.parent]
    if root.is_dir() and (root / "actions.jsonl").exists():
        return [root]

    dirs: List[Path] = []
    if not root.exists():
        return dirs

    for p in sorted(root.rglob("actions.jsonl")):
        if p.is_file():
            dirs.append(p.parent)
    return sorted(set(dirs))


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
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


def _chip_id_norm(v: Any) -> int:
    x = _as_int(v, 0)
    if x in INVALID_CHIP_IDS or x < 0:
        return 0
    return x


def _code_norm(v: Any, max_code: int = 63) -> int:
    x = _as_int(v, 0)
    if x < 0:
        return 0
    if x > max_code:
        return max_code
    return x


def _mask_hand_from_visible(
    chip_slots: List[Any],
    chip_codes: List[Any],
    visible_count: int,
    *,
    n_slots: int = 10,
) -> Tuple[List[int], List[int], List[float]]:
    hand_id: List[int] = [0] * n_slots
    hand_code: List[int] = [0] * n_slots
    hand_vis: List[float] = [0.0] * n_slots

    vis = max(0, min(n_slots, _as_int(visible_count, 0)))
    for i in range(n_slots):
        if i < len(chip_slots):
            hand_id[i] = _chip_id_norm(chip_slots[i])
            hand_code[i] = _code_norm(chip_codes[i] if i < len(chip_codes) else 0)
        else:
            hand_id[i] = 0
            hand_code[i] = 0
        hand_vis[i] = 1.0 if i < vis else 0.0

    # If a slot is not visible, hard-mask it to 0 for clean learning signals.
    for i in range(n_slots):
        if hand_vis[i] <= 0.0:
            hand_id[i] = 0
            hand_code[i] = 0

    return hand_id, hand_code, hand_vis


def _extract_grid(frame: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))

    gs = [max(0, _as_int(grid_state[i], 0)) for i in range(min(18, len(grid_state)))]
    go_raw = [_as_int(grid_owner[i], 2) for i in range(min(18, len(grid_owner)))]
    go = []
    for x in go_raw:
        if x == 0:
            go.append(0)
        elif x == 1:
            go.append(1)
        else:
            go.append(2)

    if len(gs) < 18:
        gs.extend([0] * (18 - len(gs)))
    if len(go) < 18:
        go.extend([2] * (18 - len(go)))
    return gs, go


def _extract_pos(frame: Dict[str, Any], key: str) -> Tuple[int, int]:
    v = frame.get(key)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _as_int(v[0], 0), _as_int(v[1], 0)
    if isinstance(v, str) and "," in v:
        a, b = v.split(",", 1)
        return _as_int(a, 0), _as_int(b, 0)
    return 0, 0


def _get_player_game_emotion(frame: Dict[str, Any]) -> int:
    """
    Prefer 'player_game_emotion' (explicit), fall back to existing captures 'player_emotion'.
    """
    if "player_game_emotion" in frame:
        return _as_int(frame.get("player_game_emotion"), 0)
    return _as_int(frame.get("player_emotion"), 0)


def _get_enemy_game_emotion(frame: Dict[str, Any]) -> int:
    if "enemy_game_emotion" in frame:
        return _as_int(frame.get("enemy_game_emotion"), 0)
    return _as_int(frame.get("enemy_emotion"), 0)


def _is_chip_window_frame(frame: Dict[str, Any]) -> bool:
    """
    Chip window (your definition):
      inside_window == True AND cust_gauge == 0
    """
    inside = _as_bool(frame.get("inside_window"), False)
    cust = _as_int(frame.get("cust_gauge"), 0)
    return inside and (cust == 0)


def _is_battle_start_frame(frame: Dict[str, Any]) -> bool:
    """
    Battle start (your definition):
      inside_window == False AND cust_gauge > 0
    """
    inside = _as_bool(frame.get("inside_window"), False)
    cust = _as_int(frame.get("cust_gauge"), 0)
    return (not inside) and (cust > 0)


def _find_pre_emotion_idx(frames: List[Dict[str, Any]], open_idx: int) -> int:
    """
    Find the last frame BEFORE entering the chip window.

    We anchor on the chip-window condition (inside_window==True, cust==0) and then
    scan backward to find the closest frame that represents "battle outside window":
      inside_window==False AND cust_gauge>0

    If none exists (edge cases), fall back to the closest frame with inside_window==False,
    else open_idx.
    """
    n = len(frames)
    if n == 0:
        return 0
    open_idx = max(0, min(open_idx, n - 1))

    i = open_idx - 1
    # closest "outside window + in battle" frame
    while i >= 0:
        f = frames[i]
        if _is_battle_start_frame(f):
            return i
        # If we somehow see "inside window" while scanning backward, still keep going.
        i -= 1

    # fallback: closest outside-window frame at all
    i = open_idx - 1
    while i >= 0:
        f = frames[i]
        if not _as_bool(f.get("inside_window"), False):
            return i
        i -= 1

    return open_idx


def _find_post_emotion_idx(
    frames: List[Dict[str, Any]],
    *,
    close_idx: int,
    search_end: int,
) -> int:
    """
    Find the first frame AFTER leaving the chip window that corresponds to battle start:
      inside_window==False AND cust_gauge>0

    We start at close_idx+1 (first frame after inside_window segment ends) and scan
    forward until search_end (exclusive).
    If not found, returns close_idx+1 clamped (best-effort).
    """
    n = len(frames)
    if n == 0:
        return 0
    close_idx = max(0, min(close_idx, n - 1))
    a = min(n - 1, close_idx + 1)
    b = max(a + 1, min(search_end, n))  # ensure b > a when possible

    for i in range(a, b):
        if _is_battle_start_frame(frames[i]):
            return i

    return a


def _derive_selected_cross_emotion_from_boundaries(
    frames: List[Dict[str, Any]],
    *,
    open_idx: int,
    close_idx: int,
    next_open_idx: int,
) -> Tuple[int, int, int, int, int, str]:
    """
    Returns:
      (selected_cross_emotion, pre_emotion, post_emotion, pre_idx, post_idx, reason)

    We store selected_cross as the raw player_game_emotion id (includes beast ids).
    Per your spec:
      - pre_emotion: from last frame BEFORE we enter chip window (battle outside window)
      - post_emotion: from first frame AFTER we leave chip window (battle start)
        i.e. inside_window==False AND cust_gauge>0

    selected_cross:
      - if post_emotion != pre_emotion => selected_cross = post_emotion
      - else 0

    reason helps debugging aggregate stats.
    """
    if not frames:
        return 0, 0, 0, 0, 0, "no_frames"

    n = len(frames)
    open_idx = max(0, min(open_idx, n - 1))
    close_idx = max(0, min(close_idx, n - 1))
    next_open_idx = max(0, min(next_open_idx, n))

    # If open_idx isn't actually a chip window frame (data oddities), still proceed.
    pre_idx = _find_pre_emotion_idx(frames, open_idx)

    # Search post until next_open_idx (or end) so we don't get stuck on cust==0 transition frames
    post_idx = _find_post_emotion_idx(frames, close_idx=close_idx, search_end=next_open_idx)

    pre_f = frames[pre_idx]
    post_f = frames[post_idx]

    pre_em = _get_player_game_emotion(pre_f)
    post_em = _get_player_game_emotion(post_f)

    if not _is_battle_start_frame(post_f):
        # post frame didn't satisfy the condition; caller will see very low "battle-start found"
        return 0, pre_em, post_em, pre_idx, post_idx, "post_not_battle_start"

    if pre_em == post_em:
        return 0, pre_em, post_em, pre_idx, post_idx, "no_change"

    return int(post_em), pre_em, post_em, pre_idx, post_idx, "changed"


# ---------------------------------------------------------------------
# Turn extraction logic
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WindowEvent:
    open_idx: int       # first frame inside_window True (regardless of cust; we treat as window segment start)
    close_idx: int      # last frame inside_window True
    commit_idx: int     # first frame after close (inside_window False) where derived.window_commit is recorded
    next_open_idx: int  # next open, or len(frames) if none


def _find_window_events(frames: List[Dict[str, Any]]) -> List[WindowEvent]:
    n = len(frames)
    if n == 0:
        return []

    inside = [_as_bool(f.get("inside_window"), False) for f in frames]

    opens: List[int] = []
    closes: List[int] = []  # close_idx = last inside frame
    prev = False
    for i in range(n):
        cur = inside[i]
        if cur and not prev:
            opens.append(i)
        if prev and not cur:
            closes.append(i - 1)
        prev = cur

    # Pair each open with the next close after it.
    events: List[WindowEvent] = []
    ci = 0
    for oi in opens:
        while ci < len(closes) and closes[ci] < oi:
            ci += 1
        if ci >= len(closes):
            break
        close_idx = closes[ci]
        if close_idx < oi:
            continue
        commit_idx = min(n - 1, close_idx + 1)  # first outside frame (may still have cust==0!)
        # Find next open after commit
        next_open = n
        for j in opens:
            if j > commit_idx:
                next_open = j
                break
        events.append(WindowEvent(open_idx=oi, close_idx=close_idx, commit_idx=commit_idx, next_open_idx=next_open))
        ci += 1

    return events


def _compute_outcome(frames: List[Dict[str, Any]], start: int, end: int) -> Tuple[int, int, int]:
    """
    Outcome over [start, end):
      - damage_dealt: enemy_hp decreases
      - damage_taken: player_hp decreases
      - net_yield: dealt - taken
    """
    if start < 0:
        start = 0
    if end > len(frames):
        end = len(frames)
    if end <= start:
        return 0, 0, 0

    p_prev = _as_int(frames[start].get("player_health"), 0)
    e_prev = _as_int(frames[start].get("enemy_health"), 0)

    dealt = 0
    taken = 0

    for i in range(start + 1, end):
        f = frames[i]
        p = _as_int(f.get("player_health"), p_prev)
        e = _as_int(f.get("enemy_health"), e_prev)

        dp = p_prev - p
        de = e_prev - e

        if 0 < dp < 2000:
            taken += dp
        if 0 < de < 2000:
            dealt += de

        p_prev = p
        e_prev = e

    return dealt, taken, dealt - taken


def _held_to_fixed(held: Any, *, n: int = 5) -> Tuple[List[int], List[int], List[bool]]:
    """
    held is list[{"id":int,"code":int}, ...]
    Returns fixed-length arrays.
    """
    items = held if isinstance(held, list) else []
    out_id = [0] * n
    out_code = [0] * n
    out_mask = [False] * n
    for i in range(min(n, len(items))):
        it = items[i] if isinstance(items[i], dict) else {}
        out_id[i] = _chip_id_norm(it.get("id"))
        out_code[i] = _code_norm(it.get("code"))
        out_mask[i] = True
    return out_id, out_code, out_mask


def _held_list_clamped(held: Any, *, n: int = 5) -> List[Dict[str, int]]:
    """
    Return a clean list-of-dicts form for UI/debugging.
    """
    items = held if isinstance(held, list) else []
    out: List[Dict[str, int]] = []
    for i in range(min(n, len(items))):
        it = items[i] if isinstance(items[i], dict) else {}
        out.append(
            {
                "id": _chip_id_norm(it.get("id")),
                "code": _code_norm(it.get("code")),
            }
        )
    return out


# ---------------------------------------------------------------------
# Debug summary structures
# ---------------------------------------------------------------------
@dataclass
class ReplayDebugSummary:
    replay: str
    total_events: int = 0
    had_window_commit: int = 0
    battle_start_found: int = 0
    changed_count: int = 0
    post_not_battle_start: int = 0
    examples: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []


def _fmt_frame_flags(frames: List[Dict[str, Any]], idx: int) -> str:
    if idx < 0 or idx >= len(frames):
        return "idx=OOB"
    f = frames[idx]
    inside = _as_bool(f.get("inside_window"), False)
    cust = _as_int(f.get("cust_gauge"), 0)
    em = _get_player_game_emotion(f)
    return f"idx={idx} inside={int(inside)} cust={cust} em={em}"


def build_samples_for_replay(replay_dir: Path, *, debug: bool = False, debug_examples_limit: int = 8) -> Tuple[List[Dict[str, Any]], ReplayDebugSummary]:
    actions_path = replay_dir / "actions.jsonl"
    static_path = replay_dir / "static_data.json"
    replay_name = replay_dir.name

    frames = _read_jsonl(actions_path)
    dbg = ReplayDebugSummary(replay=replay_name)

    if not frames:
        return [], dbg

    static = _read_json(static_path)
    derived = compute_derived(frames, static)

    events = _find_window_events(frames)
    if not events:
        return [], dbg

    samples: List[Dict[str, Any]] = []

    for ev in events:
        dbg.total_events += 1

        if ev.commit_idx < 0 or ev.commit_idx >= len(derived):
            continue

        # --- derived snapshots ---
        d_open = derived[ev.open_idx] if 0 <= ev.open_idx < len(derived) else {}
        d_commit = derived[ev.commit_idx] if isinstance(derived[ev.commit_idx], dict) else {}

        p_open = (d_open.get("player") or {}) if isinstance(d_open, dict) else {}
        p_commit = (d_commit.get("player") or {}) if isinstance(d_commit, dict) else {}
        e_open = (d_open.get("enemy") or {}) if isinstance(d_open, dict) else {}

        window_commit = p_commit.get("window_commit") or {}
        happened = bool(window_commit.get("happened", False))
        if not happened:
            continue
        dbg.had_window_commit += 1

        open_f = frames[ev.open_idx]
        close_f = frames[ev.close_idx]
        commit_f = frames[ev.commit_idx]

        # -----------------------------------------------------------------
        # Selected cross detection (FIXED):
        # - pre = last battle frame before window entry
        # - post = first battle-start frame after leaving window (cust>0)
        # -----------------------------------------------------------------
        selected_cross_emotion, pre_emotion, post_emotion, pre_idx, post_idx, reason = _derive_selected_cross_emotion_from_boundaries(
            frames,
            open_idx=ev.open_idx,
            close_idx=ev.close_idx,
            next_open_idx=ev.next_open_idx,
        )

        if reason != "post_not_battle_start":
            dbg.battle_start_found += 1
        else:
            dbg.post_not_battle_start += 1

        if reason == "changed":
            dbg.changed_count += 1

        if debug and len(dbg.examples) < debug_examples_limit:
            dbg.examples.append(
                " | ".join(
                    [
                        f"open={_fmt_frame_flags(frames, ev.open_idx)}",
                        f"close={_fmt_frame_flags(frames, ev.close_idx)}",
                        f"commitIdx={_fmt_frame_flags(frames, ev.commit_idx)}",
                        f"pre={_fmt_frame_flags(frames, pre_idx)}",
                        f"post={_fmt_frame_flags(frames, post_idx)}",
                        f"selected={selected_cross_emotion} reason={reason}",
                    ]
                )
            )

        # Input: visible chip window at open frame (masked by visible_count)
        chip_slots = _as_list(open_f.get("chip_slots"))
        chip_codes = _as_list(open_f.get("chip_codes"))
        chip_visible_count = _as_int(open_f.get("chip_visible_count"), 5)
        hand_id, hand_code, hand_vis = _mask_hand_from_visible(chip_slots, chip_codes, chip_visible_count)

        # Authoritative held after commit (enter battle) and context held (at open)
        held_after = p_commit.get("held_chips", [])
        ha_id, ha_code, ha_mask = _held_to_fixed(held_after, n=5)

        # "Held before window": use derived held from open snapshot (the decision context)
        # This is more stable than trying to guess it from raw frame boundaries.
        held_before = p_open.get("held_chips", [])
        hb_id, hb_code, hb_mask = _held_to_fixed(held_before, n=5)

        p_folder_used = _as_list(p_open.get("folder_used_mask"))
        e_folder_used = _as_list(e_open.get("folder_used_mask"))
        p_used_cross = _as_list(p_open.get("used_cross_mask"))
        e_used_cross = _as_list(e_open.get("used_cross_mask"))

        # Board snapshot at open
        grid_tile, grid_owner = _extract_grid(open_f)
        px, py = _extract_pos(open_f, "player_pos")
        ex, ey = _extract_pos(open_f, "enemy_pos")

        # Outcome (commit -> next open)
        dealt, taken, net = _compute_outcome(frames, ev.commit_idx, ev.next_open_idx)

        # Labels from derived commit
        selected_any = bool(window_commit.get("selected_any", False))
        beast_selected = bool(window_commit.get("beast_selected", False))
        selected_chips = window_commit.get("selected_chips", [])
        sc_id, sc_code, sc_mask = _held_to_fixed(selected_chips, n=5)

        sample = {
            "format": "chip_window_strategy_v2",
            "replay": replay_name,

            # --- indices ---
            "open_idx": int(ev.open_idx),
            "close_idx": int(ev.close_idx),
            "commit_idx": int(ev.commit_idx),
            "next_open_idx": int(ev.next_open_idx),

            # --- selected-cross boundary indices (debug + UI support) ---
            "selected_cross_pre_idx": int(pre_idx),
            "selected_cross_post_idx": int(post_idx),
            "selected_cross_reason": str(reason),

            # --- input snapshot (OPEN) ---
            "p_hp_open": _as_int(open_f.get("player_health"), 0),
            "e_hp_open": _as_int(open_f.get("enemy_health"), 0),
            "cust_open": _as_int(open_f.get("cust_gauge"), 0),
            "player_charge_open": _as_int(open_f.get("player_charge"), 0),
            "enemy_charge_open": _as_int(open_f.get("enemy_charge"), 0),

            "player_pos_open": [int(px), int(py)],
            "enemy_pos_open": [int(ex), int(ey)],

            # Keep these names for backward compatibility with existing consumers.
            # Values are sourced from *_game_emotion when present.
            "player_emotion_open": _get_player_game_emotion(open_f),
            "enemy_emotion_open": _get_enemy_game_emotion(open_f),

            "beast_mode_open": _as_int(open_f.get("beast_mode"), 0),

            "grid_tile_open": grid_tile,
            "grid_owner_open": grid_owner,

            # Chip window contents (masked by visibility)
            "window_hand_id": hand_id,
            "window_hand_code": hand_code,
            "window_hand_vis": hand_vis,
            "chip_visible_count": int(chip_visible_count),

            # Derived context at open
            "turn_index_open": _as_int(d_open.get("turn_index"), 0),
            "folder_used_mask_p_open": [bool(x) for x in p_folder_used],
            "folder_used_mask_e_open": [bool(x) for x in e_folder_used],
            "used_cross_mask_p_open": [bool(x) for x in p_used_cross],
            "used_cross_mask_e_open": [bool(x) for x in e_used_cross],
            "active_cross_p_open": p_open.get("active_cross", None),
            "active_cross_e_open": e_open.get("active_cross", None),
            "beast_p_open": p_open.get("beast", None),
            "beast_e_open": e_open.get("beast", None),

            # HELD BEFORE entering THIS chip window (decision context)
            "held_before_id": hb_id,
            "held_before_code": hb_code,
            "held_before_mask": hb_mask,
            "held_before": _held_list_clamped(held_before, n=5),

            # --- label (COMMIT) ---
            "selected_any": selected_any,
            "beast_selected": beast_selected,
            "selected_chips_id": sc_id,
            "selected_chips_code": sc_code,
            "selected_chips_mask": sc_mask,
            "selected_chips": _held_list_clamped(selected_chips, n=5),

            # Selected cross label (RAW player_game_emotion id; includes beast ids)
            # 0 means "no cross selection detected".
            "selected_cross": int(selected_cross_emotion),

            # Debug fields (what we compared)
            "selected_cross_pre_emotion": int(pre_emotion),
            "selected_cross_post_emotion": int(post_emotion),

            # Also record the raw boundary flags we used (helps spot bad matching)
            "selected_cross_pre_inside": int(_as_bool(frames[pre_idx].get("inside_window"), False)) if 0 <= pre_idx < len(frames) else 0,
            "selected_cross_pre_cust": int(_as_int(frames[pre_idx].get("cust_gauge"), 0)) if 0 <= pre_idx < len(frames) else 0,
            "selected_cross_post_inside": int(_as_bool(frames[post_idx].get("inside_window"), False)) if 0 <= post_idx < len(frames) else 0,
            "selected_cross_post_cust": int(_as_int(frames[post_idx].get("cust_gauge"), 0)) if 0 <= post_idx < len(frames) else 0,

            # HELD AFTER commit (authoritative enter-battle)
            "held_after_id": ha_id,
            "held_after_code": ha_code,
            "held_after_mask": ha_mask,
            "held_after": _held_list_clamped(held_after, n=5),

            # Debug / provenance
            "commit_source": window_commit.get("source", ""),
            "close_chip_select_count": _as_int(window_commit.get("close_chip_select_count"), 0),

            # --- outcome (commit -> next open) ---
            "damage_dealt": int(dealt),
            "damage_taken": int(taken),
            "net_yield": int(net),

            # Optional: close snapshot telemetry
            "p_hp_close": _as_int(close_f.get("player_health"), 0),
            "e_hp_close": _as_int(close_f.get("enemy_health"), 0),
            "p_hp_commit": _as_int(commit_f.get("player_health"), 0),
            "e_hp_commit": _as_int(commit_f.get("enemy_health"), 0),
        }

        samples.append(sample)

    return samples, dbg


def _worker_build_samples(args: Tuple[str, bool, int]) -> Tuple[str, List[Dict[str, Any]], ReplayDebugSummary, Optional[str]]:
    """
    ProcessPool worker.
    Returns (replay_name, samples, debug_summary, error_str)
    """
    replay_dir_str, debug, debug_examples_limit = args
    rd = Path(replay_dir_str)
    try:
        samples, dbg = build_samples_for_replay(rd, debug=debug, debug_examples_limit=debug_examples_limit)
        return rd.name, samples, dbg, None
    except Exception as e:
        return rd.name, [], ReplayDebugSummary(replay=rd.name), str(e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build chip window strategy dataset (derived-state v2, parallel)")
    ap.add_argument("--input", default="data/dataset", help="Dataset root containing replay dirs")
    ap.add_argument("--output_dir", default="data/chipwindows_v2", help="Output directory")
    ap.add_argument("--output_name", default="strategy_v2.jsonl", help="Output jsonl filename")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    ap.add_argument("--workers", type=int, default=0, help="Num worker processes (0 = os.cpu_count())")

    # Debug controls
    ap.add_argument("--debug", action="store_true", help="Print per-replay selected-cross detection stats + a few examples")
    ap.add_argument("--debug_examples", type=int, default=6, help="How many event examples to print per replay when --debug")
    ap.add_argument("--debug_top", type=int, default=12, help="Print detailed examples for the top-N most suspicious replays (lowest change rate)")

    args = ap.parse_args()

    in_root = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    replay_dirs = _find_replay_dirs(in_root)
    replay_dirs = sorted(replay_dirs, key=lambda p: p.name)
    print(f"[scan] input={in_root} replays={len(replay_dirs)}")

    workers = int(args.workers) if int(args.workers) > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)
    print(f"[pool] workers={workers}")

    # Deterministic output order: we gather results in replay order.
    results_by_name: Dict[str, List[Dict[str, Any]]] = {}
    debug_by_name: Dict[str, ReplayDebugSummary] = {}
    errors_by_name: Dict[str, str] = {}

    job_args = [(str(p), bool(args.debug), int(args.debug_examples)) for p in replay_dirs]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, samples, dbg, err in ex.map(_worker_build_samples, job_args):
            if err:
                errors_by_name[name] = err
                continue
            if samples:
                results_by_name[name] = samples
            debug_by_name[name] = dbg

    # Write output in replay order
    total = 0
    kept_replays = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for rd in replay_dirs:
            samples = results_by_name.get(rd.name)
            if not samples:
                continue
            kept_replays += 1
            for s in samples:
                out_f.write(json.dumps(s) + "\n")
            total += len(samples)
            print(f"  + {rd.name}: {len(samples)} samples")

    # Aggregate debug stats
    agg_total_events = 0
    agg_had_commit = 0
    agg_battle_found = 0
    agg_changed = 0
    agg_post_not_battle = 0

    for rd in replay_dirs:
        dbg = debug_by_name.get(rd.name)
        if not dbg:
            continue
        agg_total_events += dbg.total_events
        agg_had_commit += dbg.had_window_commit
        agg_battle_found += dbg.battle_start_found
        agg_changed += dbg.changed_count
        agg_post_not_battle += dbg.post_not_battle_start

    print(
        "[cross] "
        f"events={agg_total_events} with_commit={agg_had_commit} "
        f"battle_start_found={agg_battle_found} changed={agg_changed} "
        f"post_not_battle_start={agg_post_not_battle}"
    )
    if agg_battle_found > 0:
        rate = agg_changed / max(1, agg_battle_found)
        print(f"[cross] change_rate={rate:.4f} (changed/battle_start_found)")

    # Print suspicious replays (lowest change rate), with examples
    if args.debug:
        scored: List[Tuple[float, str]] = []
        for name, dbg in debug_by_name.items():
            denom = max(1, dbg.battle_start_found)
            rate = dbg.changed_count / denom
            # Prefer ones where we *did* find battle start but almost never changed
            if dbg.battle_start_found >= 3:
                scored.append((rate, name))
        scored.sort(key=lambda x: (x[0], x[1]))

        top_n = max(0, int(args.debug_top))
        if top_n > 0 and scored:
            print(f"[debug] lowest change-rate replays (top {min(top_n, len(scored))}):")
            for rate, name in scored[:top_n]:
                dbg = debug_by_name[name]
                print(
                    f"  - {name}: events={dbg.total_events} commit={dbg.had_window_commit} "
                    f"battle_found={dbg.battle_start_found} changed={dbg.changed_count} "
                    f"rate={rate:.4f} post_not_battle={dbg.post_not_battle_start}"
                )
                for exline in dbg.examples:
                    print(f"      {exline}")

    if errors_by_name:
        print(f"[warn] failures={len(errors_by_name)}")
        for i, (k, v) in enumerate(sorted(errors_by_name.items())[:10]):
            print(f"  - {k}: {v}")

    print(f"[done] samples={total} replays_with_samples={kept_replays} out={out_path}")


if __name__ == "__main__":
    main()
