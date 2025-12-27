#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


FPS = 60


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _mode_int(vals: List[int]) -> Optional[int]:
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]


def _stable_value(series: List[int], start: int, end: int) -> Optional[int]:
    """Mode over [start, end) clamped."""
    n = len(series)
    if n == 0:
        return None
    a = max(0, min(start, n))
    b = max(0, min(end, n))
    if b <= a:
        return None
    return _mode_int(series[a:b])


@dataclass
class WindowSegment:
    start: int  # inclusive
    end: int    # exclusive


def _find_window_segments(inside: List[bool]) -> List[WindowSegment]:
    segs: List[WindowSegment] = []
    n = len(inside)
    i = 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i + 1
        while j < n and inside[j]:
            j += 1
        segs.append(WindowSegment(start=i, end=j))
        i = j
    return segs


def analyze_actions_jsonl(path: str, *, pre_s: float = 0.5, post_s: float = 1.0) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except Exception:
                continue

    if not frames:
        return {"events": 0, "player_changed": 0, "enemy_changed": 0, "both_changed": 0, "neither_changed": 0}

    inside = [bool(fr.get("inside_window", False)) for fr in frames]
    p_ge = [_safe_int(fr.get("player_game_emotion", 0), 0) for fr in frames]
    e_ge = [_safe_int(fr.get("enemy_game_emotion", 0), 0) for fr in frames]
    sel  = [_safe_int(fr.get("selected_cross_index", 0), 0) for fr in frames]

    segs = _find_window_segments(inside)

    pre_k = max(1, int(pre_s * FPS))
    post_k = max(1, int(post_s * FPS))

    events = 0
    player_changed = 0
    enemy_changed = 0
    both_changed = 0
    neither_changed = 0

    last_sel: Optional[int] = None

    for seg_idx, seg in enumerate(segs):
        last_in_window = seg.end - 1
        if last_in_window < 0:
            continue

        sel_now = sel[last_in_window]

        # Need a prior sel to compare; if none, just set and continue
        if last_sel is None:
            last_sel = sel_now
            continue

        # Only count when the selection actually changed
        if sel_now == last_sel:
            continue

        # Determine stable before/after around the window close
        # "before" = just before window opens
        pre_end = seg.start
        pre_start = pre_end - pre_k

        # "after" = first battle-valid frame after window closes (cust_gauge != 0)
        battle_start = _find_first_battle_frame_after(frames, seg.end)
        if battle_start is None:
            last_sel = sel_now
            continue

        post_start = battle_start
        post_end = battle_start + post_k

        p_before = _stable_value(p_ge, pre_start, pre_end)
        e_before = _stable_value(e_ge, pre_start, pre_end)
        p_after  = _stable_value(p_ge, post_start, post_end)
        e_after  = _stable_value(e_ge, post_start, post_end)


        # If any of these are None (near ends), skip (don’t poison stats)
        if p_before is None or e_before is None or p_after is None or e_after is None:
            last_sel = sel_now
            continue

        p_chg = (p_after != p_before)
        e_chg = (e_after != e_before)

        events += 1
        if p_chg and e_chg:
            both_changed += 1
        elif p_chg:
            player_changed += 1
        elif e_chg:
            enemy_changed += 1
        else:
            neither_changed += 1

        last_sel = sel_now

    return {
        "events": events,
        "player_changed": player_changed,
        "enemy_changed": enemy_changed,
        "both_changed": both_changed,
        "neither_changed": neither_changed,
    }

def _find_first_battle_frame_after(frames: List[Dict[str, Any]], start_idx: int) -> Optional[int]:
    """
    First index >= start_idx where we're in battle flow:
      - inside_window must be False
      - cust_gauge != 0
    """
    n = len(frames)
    for i in range(max(0, start_idx), n):
        fr = frames[i]
        if bool(fr.get("inside_window", False)):
            continue
        if _safe_int(fr.get("cust_gauge", 0), 0) != 0:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/dataset")
    ap.add_argument("--pre-s", type=float, default=0.5, help="seconds to look back for stable 'before'")
    ap.add_argument("--post-s", type=float, default=1.0, help="seconds to look forward for stable 'after'")
    ap.add_argument("--limit", type=int, default=0, help="limit number of replays (0 = no limit)")
    args = ap.parse_args()

    totals = Counter()

    paths: List[Tuple[str, str]] = []
    for root, _, files in os.walk(args.dataset_dir):
        if "actions.jsonl" in files:
            replay = os.path.basename(root)
            paths.append((replay, os.path.join(root, "actions.jsonl")))
    paths.sort()

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    for replay, p in paths:
        r = analyze_actions_jsonl(p, pre_s=args.pre_s, post_s=args.post_s)
        for k, v in r.items():
            totals[k] += int(v)

    ev = totals["events"]
    pc = totals["player_changed"]
    ec = totals["enemy_changed"]
    bc = totals["both_changed"]
    nc = totals["neither_changed"]

    print("=== Cross ownership inference via selection→emotion-change correlation ===")
    print(f"Replays scanned: {len(paths)}")
    print(f"Selection-changed events: {ev}")
    print(f"player_game_emotion changed: {pc}")
    print(f"enemy_game_emotion changed: {ec}")
    print(f"both changed: {bc}")
    print(f"neither changed: {nc}")

    if ev > 0:
        print("")
        print(f"P(change is player): {pc/ev:.3f}")
        print(f"P(change is enemy) : {ec/ev:.3f}")
        print(f"P(change is both)  : {bc/ev:.3f}")
        print(f"P(change is neither): {nc/ev:.3f}")

        if pc > ec * 1.25:
            print("\nLikely: selected_cross_index corresponds to PLAYER; player_game_emotion is the one that should track it.")
        elif ec > pc * 1.25:
            print("\nLikely: selected_cross_index corresponds to PLAYER but the *enemy_game_emotion* field is actually the player (swap bug for game_emotion).")
        else:
            print("\nInconclusive: could be delayed application, missing telemetry, or cross switching not reflected in game_emotion reliably.")


if __name__ == "__main__":
    main()
