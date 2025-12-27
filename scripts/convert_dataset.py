# scripts/convert_dataset.py
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from typing import Dict

from action_schema import (
    GBA_BITS,
    GBA_TO_NITROGEN,
    BUTTON_TOKENS,
    build_nitrogen_template,
)

# Fields that represent the "Player" and "Enemy" relative to the capture
SWAP_PAIRS = [
    ("player_health", "enemy_health"),
    ("player_pos", "enemy_pos"),
    ("player_charge", "enemy_charge"),
    ("player_chip", "enemy_chip"),
]


def parse_input_bitmask(bitmask_int: int) -> OrderedDict:
    """
    Produces a Nitrogen-shaped action row:
      - axes are present (list-wrapped), left at 0.0
      - only GBA buttons are ever set to 1.0
      - all other Nitrogen buttons remain 0.0
    """
    action = build_nitrogen_template()

    if not isinstance(bitmask_int, int):
        bitmask_int = 0

    for bit, gba_btn in GBA_BITS.items():
        if (bitmask_int >> bit) & 1:
            nitro_key = GBA_TO_NITROGEN.get(gba_btn)
            if nitro_key is not None:
                action[nitro_key] = 1.0

    return action


def extract_state_v1(obj: dict, swap: bool) -> dict:
    state_wrapper = obj.get("state", {})
    st = state_wrapper.get("V1", state_wrapper) if isinstance(state_wrapper, dict) else {}
    if not st:
        return {}

    if swap:
        st = st.copy()
        for k1, k2 in SWAP_PAIRS:
            st[k1], st[k2] = st.get(k2), st.get(k1)

    return st


def _update_press_stats(stats: Dict[str, int], row: dict) -> None:
    """
    Count presses ONLY for canonical buttons.
    (Avoid counting telemetry fields like player_health.)
    """
    for k in BUTTON_TOKENS:
        v = row.get(k, 0.0)
        if isinstance(v, (int, float)) and float(v) > 0.5:
            stats[k] = stats.get(k, 0) + 1

from typing import Any, List, Optional, Tuple

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _stable_mode(values: List[int]) -> Optional[int]:
    """
    Returns the mode of non-negative ints in `values`.
    If no usable values, returns None.
    """
    counts: Dict[int, int] = {}
    for v in values:
        if v is None:
            continue
        iv = _safe_int(v, -1)
        if iv < 0:
            continue
        counts[iv] = counts.get(iv, 0) + 1
    if not counts:
        return None
    # mode by count, tie-break by smaller value for determinism
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _range_mode(frames: List[dict], i0: int, i1: int, key: str) -> Optional[int]:
    n = len(frames)
    a = max(0, min(i0, n))
    b = max(0, min(i1, n))
    if b <= a:
        return None
    vals = [frames[i].get(key) for i in range(a, b)]
    return _stable_mode(vals)


def _find_inside_window_segments(frames: List[dict]) -> List[Tuple[int, int]]:
    """
    Returns list of (start, end_exclusive) segments where inside_window==True.
    """
    segs: List[Tuple[int, int]] = []
    n = len(frames)
    i = 0
    while i < n:
        if bool(frames[i].get("inside_window", False)):
            j = i + 1
            while j < n and bool(frames[j].get("inside_window", False)):
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def _find_battle_start_by_cust_gauge(frames: List[dict], start_idx: int) -> Optional[int]:
    """
    First index >= start_idx where cust_gauge > 0.

    We intentionally do NOT require inside_window == False here because:
      - if cust_gauge > 0, gameplay is starting/started
      - and inside_window may be noisy depending on capture timing.
    """
    n = len(frames)
    for i in range(max(0, start_idx), n):
        if _safe_int(frames[i].get("cust_gauge", 0), 0) > 0:
            return i
    return None

def _first_value_change(frames: List[dict], i0: int, i1: int, key: str, base: int) -> Optional[int]:
    """
    Return first index in [i0, i1) where key's int value != base.
    Ignores missing/None by skipping them.
    """
    n = len(frames)
    a = max(0, min(i0, n))
    b = max(0, min(i1, n))
    for i in range(a, b):
        v = frames[i].get(key)
        if v is None:
            continue
        iv = _safe_int(v, base)
        if iv != base:
            return i
    return None

def _find_window_intent(frames: List[dict], s0: int, s1: int) -> bool:
    """
    Local player's 'cross intent' proxy:
      - if player_emotion changes during the cust window, assume we intended a cross change.
    Uses modes over small windows to avoid 1-frame glitches.
    """
    # early window mode
    a0 = s0
    a1 = min(s0 + 12, s1)
    # late window mode
    b0 = max(s0, s1 - 12)
    b1 = s1

    pre = _range_mode(frames, a0, a1, "player_emotion")
    post = _range_mode(frames, b0, b1, "player_emotion")
    if pre is None or post is None:
        return False
    return int(pre) != int(post)


def detect_and_fix_game_emotion_swap(frames: List[dict]) -> bool:
    """
    Detect whether player_game_emotion and enemy_game_emotion should be swapped.

    Primary evidence (intent windows):
      - If local intent happened (player_emotion changes during window),
        then near round-start (scan 180 frames before first cust_gauge>0),
        the *local* game_emotion should be the one that changes first.

    Strong tell-tale (NO-intent windows):
      - If local intent did NOT happen, but player_game_emotion changes (and enemy doesn't)
        near round-start, that's a strong indicator player_game_emotion is actually enemy's.
        -> Immediately decide SWAP and stop processing.
    """
    if not frames:
        return False

    segs = _find_inside_window_segments(frames)
    if not segs:
        return False

    # Tunables
    PRE_MODE_FRAMES = 12        # stable estimate at end of window
    PRE_BATTLE_SCAN = 180       # 3s @ 60fps (scan BEFORE battle_start)
    MIN_VOTES = 2
    LEAD_FRAMES = 2             # require this much lead to count "first"
    EARLY_EXIT_MARGIN = 2

    votes_keep = 0
    votes_swap = 0

    for (s0, s1) in segs:
        intent = _find_window_intent(frames, s0, s1)

        # Anchor: first cust_gauge > 0 after window closes
        battle_start = _find_battle_start_by_cust_gauge(frames, s1)
        if battle_start is None:
            continue

        # Stable "before" values (end-of-window)
        p_base = _range_mode(frames, s1 - PRE_MODE_FRAMES, s1, "player_game_emotion")
        e_base = _range_mode(frames, s1 - PRE_MODE_FRAMES, s1, "enemy_game_emotion")
        if p_base is None or e_base is None:
            continue

        # Scan region: 180 frames BEFORE battle_start (but never before window end)
        scan_start = max(s1, battle_start - PRE_BATTLE_SCAN)
        scan_end = battle_start  # exclusive
        if scan_end <= scan_start:
            continue

        p_chg_i = _first_value_change(frames, scan_start, scan_end, "player_game_emotion", base=int(p_base))
        e_chg_i = _first_value_change(frames, scan_start, scan_end, "enemy_game_emotion", base=int(e_base))

        dp = (p_chg_i is not None)
        de = (e_chg_i is not None)

        # ------------------------------------------------------------
        # STRONG TELL-TALE RULE (your request):
        # If we did NOT show local intent in the window, but "player_game_emotion"
        # is the one that changes near round-start (while enemy doesn't),
        # that strongly suggests the ownership is flipped.
        # -> swap and stop processing immediately.
        # ------------------------------------------------------------
        if not intent:
            if dp and not de:
                return True  # SWAP immediately
            if de and not dp:
                # This is strong evidence for KEEP (opponent changed cross; local didn't)
                votes_keep += 2
                if votes_keep >= votes_swap + EARLY_EXIT_MARGIN and (votes_keep + votes_swap) >= MIN_VOTES:
                    return False
            # both/no change -> ignore
            continue

        # ------------------------------------------------------------
        # Normal voting for INTENT windows (local intended to change cross)
        # ------------------------------------------------------------
        if not dp and not de:
            continue

        if dp and not de:
            votes_keep += 1
        elif de and not dp:
            votes_swap += 1
        else:
            # both changed: decide who changed first (with lead margin)
            assert p_chg_i is not None and e_chg_i is not None
            if p_chg_i + LEAD_FRAMES < e_chg_i:
                votes_keep += 1
            elif e_chg_i + LEAD_FRAMES < p_chg_i:
                votes_swap += 1
            else:
                pass  # too close/ambiguous

        if (votes_keep + votes_swap) >= MIN_VOTES:
            if votes_swap >= votes_keep + EARLY_EXIT_MARGIN:
                return True
            if votes_keep >= votes_swap + EARLY_EXIT_MARGIN:
                return False

    if (votes_keep + votes_swap) < MIN_VOTES:
        return False

    return votes_swap > votes_keep

def apply_game_emotion_swap_inplace(st: dict, do_swap: bool) -> None:
    if not do_swap:
        return
    if "player_game_emotion" in st or "enemy_game_emotion" in st:
        st["player_game_emotion"], st["enemy_game_emotion"] = st.get("enemy_game_emotion"), st.get("player_game_emotion")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--swap-players", action="store_true", help="Swap P1/P2 telemetry data")
    parser.add_argument("--print-stats", action="store_true", help="Print button press-rate sanity stats")
    args = parser.parse_args()

    count = 0
    bad = 0
    prev_hp = {"p": None, "e": None}
    press_stats: Dict[str, int] = {}

    # Pass 1: read & extract state (respecting --swap-players) so we can infer game_emotion ownership
    raw_rows: List[dict] = []
    st_frames_for_detect: List[dict] = []

    with open(args.input, "r") as fin:
        tmp_count = 0
        tmp_bad = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                joyflags = data.get("input", 0)
                if isinstance(joyflags, dict):
                    joyflags = joyflags.get("local", 0)
                joyflags_i = int(joyflags or 0)

                frame_idx = data.get("frame", tmp_count)
                tick = data.get("tick", tmp_count)

                st = extract_state_v1(data, args.swap_players)

                raw_rows.append({
                    "joyflags": joyflags_i,
                    "frame_idx": frame_idx,
                    "tick": tick,
                    "state": st,
                })

                # Only the state dict matters for detection
                st_frames_for_detect.append(st)
                tmp_count += 1
            except Exception:
                tmp_bad += 1
                continue

    # Infer whether player_game_emotion/enemy_game_emotion are swapped (even after --swap-players)
    swap_game_emotions = detect_and_fix_game_emotion_swap(st_frames_for_detect)

    if swap_game_emotions:
        print("🧩 Detected swapped game_emotions -> correcting player_game_emotion/enemy_game_emotion for all frames.")
    else:
        print("🧩 game_emotions look consistent (no correction applied).")

    # Pass 2: write final rows
    with open(args.output, "w") as fout:
        for row in raw_rows:
            try:
                final_row = parse_input_bitmask(int(row["joyflags"]))

                # Metadata
                final_row["frame_idx"] = row["frame_idx"]
                final_row["tick"] = row["tick"]

                # State (apply game_emotion correction here)
                st = row["state"]
                apply_game_emotion_swap_inplace(st, swap_game_emotions)
                for k, v in st.items():
                    final_row[k] = v

                # Metrics
                p_hp = final_row.get("player_health")
                e_hp = final_row.get("enemy_health")

                curr_p = p_hp if isinstance(p_hp, (int, float)) else prev_hp["p"]
                curr_e = e_hp if isinstance(e_hp, (int, float)) else prev_hp["e"]

                if curr_p is not None and prev_hp["p"] is not None:
                    diff = prev_hp["p"] - curr_p
                    final_row["player_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["player_damage_taken"] = 0

                if curr_e is not None and prev_hp["e"] is not None:
                    diff = prev_hp["e"] - curr_e
                    final_row["enemy_damage_taken"] = diff if 0 < diff < 1000 else 0
                else:
                    final_row["enemy_damage_taken"] = 0

                prev_hp["p"] = curr_p
                prev_hp["e"] = curr_e

                if args.print_stats:
                    _update_press_stats(press_stats, final_row)

                fout.write(json.dumps(final_row) + "\n")
                count += 1

            except Exception:
                bad += 1
                continue

    print(f"Converted {count} frames. Skipped {bad} bad lines.")

    if args.print_stats and count > 0:
        pairs = [(k, press_stats.get(k, 0) / float(count)) for k in BUTTON_TOKENS]
        pairs.sort(key=lambda kv: kv[1], reverse=True)

        print("\n=== Button press-rate (top 12) ===")
        for k, r in pairs[:12]:
            print(f"{k:16s} {r*100:6.2f}%")

        # Heuristic warnings for obvious bitmask mismatch / stuck input
        for k, r in pairs:
            if r > 0.95:
                print(
                    f"⚠️ WARNING: '{k}' is pressed {r*100:.1f}% of frames. "
                    "Bitmask mapping may be wrong or stuck input."
                )


if __name__ == "__main__":
    main()
