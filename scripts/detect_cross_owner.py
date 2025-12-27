#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# JSONL helpers (tolerant)
# -----------------------------

def read_jsonl(
    path: str,
    *,
    allow_bad_lines: bool = True,
    max_bad_lines: int = 50,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                if not allow_bad_lines:
                    raise
                bad += 1
                if bad <= 5:
                    print(f"[WARN] {path} bad json at line {line_no} (skipping)")
                if bad >= max_bad_lines:
                    raise RuntimeError(f"Too many bad json lines in {path} (>= {max_bad_lines}). Aborting.")
                continue

            if isinstance(obj, dict):
                frames.append(obj)
            else:
                if not allow_bad_lines:
                    raise TypeError(f"{path} line {line_no} parsed to non-dict JSON: {type(obj)}")
                bad += 1
                if bad <= 5:
                    print(f"[WARN] {path} non-dict json at line {line_no} (skipping)")
                if bad >= max_bad_lines:
                    raise RuntimeError(f"Too many bad json lines in {path} (>= {max_bad_lines}). Aborting.")
    return frames


# -----------------------------
# Core helpers (convert_dataset parity)
# -----------------------------

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _stable_mode(values: List[Any]) -> Optional[int]:
    """
    Mode of non-negative ints; ignores missing/None/negative.
    Deterministic tie-break by smaller value.
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
    Intentionally does NOT require inside_window == False (parity with convert_dataset).
    """
    n = len(frames)
    for i in range(max(0, start_idx), n):
        if _safe_int(frames[i].get("cust_gauge", 0), 0) > 0:
            return i
    return None


def _first_value_change(frames: List[dict], i0: int, i1: int, key: str, base: int) -> Optional[int]:
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


def _mode_diff(frames: List[dict], s0: int, s1: int, key: str, *, edge: int = 12) -> Optional[bool]:
    """
    Returns True if mode(early edge) != mode(late edge), False if equal.
    Returns None if insufficient data.
    """
    a0 = s0
    a1 = min(s0 + edge, s1)
    b0 = max(s0, s1 - edge)
    b1 = s1
    pre = _range_mode(frames, a0, a1, key)
    post = _range_mode(frames, b0, b1, key)
    if pre is None or post is None:
        return None
    return int(pre) != int(post)


def _find_window_intent(frames: List[dict], s0: int, s1: int) -> bool:
    """
    Local 'cross intent' proxy.

    IMPORTANT FIX:
      player_emotion alone is not reliable. Use:
        - player_emotion mode change OR
        - selected_cross_index mode change

    (selected_cross_index is ALWAYS local per your invariant.)
    """
    pe = _mode_diff(frames, s0, s1, "player_emotion", edge=12)
    sc = _mode_diff(frames, s0, s1, "selected_cross_index", edge=12)

    # If both are unavailable, treat as no intent (but it will reduce confidence later)
    if pe is None and sc is None:
        return False

    return bool(pe) or bool(sc)


# -----------------------------
# Evidence + decision
# -----------------------------

@dataclass(frozen=True)
class CrossOwnerEvidence:
    windows: int
    windows_used: int
    intent_windows: int
    no_intent_windows: int
    votes_keep: int
    votes_swap: int
    early_swap_triggered: int
    early_keep_bonus: int
    skipped_no_battle_start: int
    skipped_missing_bases: int
    skipped_bad_scan: int
    decision: str   # "swap" | "keep" | "unclear"
    reason: str


def analyze_actions_frames(
    frames: List[dict],
    *,
    pre_mode_frames: int = 12,
    pre_battle_scan: int = 180,
    lead_frames: int = 2,
    early_exit_margin: int = 2,
    min_votes: int = 2,
) -> CrossOwnerEvidence:
    if not frames:
        return CrossOwnerEvidence(
            windows=0, windows_used=0, intent_windows=0, no_intent_windows=0,
            votes_keep=0, votes_swap=0, early_swap_triggered=0, early_keep_bonus=0,
            skipped_no_battle_start=0, skipped_missing_bases=0, skipped_bad_scan=0,
            decision="unclear", reason="no_frames",
        )

    segs = _find_inside_window_segments(frames)

    windows = len(segs)
    windows_used = 0
    intent_windows = 0
    no_intent_windows = 0

    votes_keep = 0
    votes_swap = 0

    early_swap_triggered = 0
    early_keep_bonus = 0

    skipped_no_battle_start = 0
    skipped_missing_bases = 0
    skipped_bad_scan = 0

    for (s0, s1) in segs:
        intent = _find_window_intent(frames, s0, s1)
        if intent:
            intent_windows += 1
        else:
            no_intent_windows += 1

        battle_start = _find_battle_start_by_cust_gauge(frames, s1)
        if battle_start is None:
            skipped_no_battle_start += 1
            continue

        p_base = _range_mode(frames, s1 - pre_mode_frames, s1, "player_game_emotion")
        e_base = _range_mode(frames, s1 - pre_mode_frames, s1, "enemy_game_emotion")
        if p_base is None or e_base is None:
            skipped_missing_bases += 1
            continue

        scan_start = max(s1, battle_start - pre_battle_scan)
        scan_end = battle_start
        if scan_end <= scan_start:
            skipped_bad_scan += 1
            continue

        p_chg_i = _first_value_change(frames, scan_start, scan_end, "player_game_emotion", base=int(p_base))
        e_chg_i = _first_value_change(frames, scan_start, scan_end, "enemy_game_emotion", base=int(e_base))

        dp = (p_chg_i is not None)
        de = (e_chg_i is not None)

        windows_used += 1

        # Strong no-intent tell (unchanged), but now "no-intent" is much harder to hit incorrectly.
        if not intent:
            if dp and not de:
                early_swap_triggered += 1
                return CrossOwnerEvidence(
                    windows=windows, windows_used=windows_used,
                    intent_windows=intent_windows, no_intent_windows=no_intent_windows,
                    votes_keep=votes_keep, votes_swap=votes_swap,
                    early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
                    skipped_no_battle_start=skipped_no_battle_start,
                    skipped_missing_bases=skipped_missing_bases,
                    skipped_bad_scan=skipped_bad_scan,
                    decision="swap",
                    reason="no_intent_player_game_emotion_changed_first",
                )

            if de and not dp:
                votes_keep += 2
                early_keep_bonus += 1
                if (votes_keep + votes_swap) >= min_votes and votes_keep >= votes_swap + early_exit_margin:
                    return CrossOwnerEvidence(
                        windows=windows, windows_used=windows_used,
                        intent_windows=intent_windows, no_intent_windows=no_intent_windows,
                        votes_keep=votes_keep, votes_swap=votes_swap,
                        early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
                        skipped_no_battle_start=skipped_no_battle_start,
                        skipped_missing_bases=skipped_missing_bases,
                        skipped_bad_scan=skipped_bad_scan,
                        decision="keep",
                        reason="no_intent_enemy_changed_first_keep_bonus",
                    )
            continue

        # Intent windows: vote based on who changes first pre-battle
        if not dp and not de:
            continue

        if dp and not de:
            votes_keep += 1
        elif de and not dp:
            votes_swap += 1
        else:
            assert p_chg_i is not None and e_chg_i is not None
            if p_chg_i + lead_frames < e_chg_i:
                votes_keep += 1
            elif e_chg_i + lead_frames < p_chg_i:
                votes_swap += 1

        if (votes_keep + votes_swap) >= min_votes:
            if votes_swap >= votes_keep + early_exit_margin:
                return CrossOwnerEvidence(
                    windows=windows, windows_used=windows_used,
                    intent_windows=intent_windows, no_intent_windows=no_intent_windows,
                    votes_keep=votes_keep, votes_swap=votes_swap,
                    early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
                    skipped_no_battle_start=skipped_no_battle_start,
                    skipped_missing_bases=skipped_missing_bases,
                    skipped_bad_scan=skipped_bad_scan,
                    decision="swap",
                    reason="intent_votes_decisive_swap",
                )
            if votes_keep >= votes_swap + early_exit_margin:
                return CrossOwnerEvidence(
                    windows=windows, windows_used=windows_used,
                    intent_windows=intent_windows, no_intent_windows=no_intent_windows,
                    votes_keep=votes_keep, votes_swap=votes_swap,
                    early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
                    skipped_no_battle_start=skipped_no_battle_start,
                    skipped_missing_bases=skipped_missing_bases,
                    skipped_bad_scan=skipped_bad_scan,
                    decision="keep",
                    reason="intent_votes_decisive_keep",
                )

    # Final decision
    if (votes_keep + votes_swap) < min_votes:
        decision = "unclear"
        reason = f"insufficient_votes<{min_votes}"
    else:
        decision = "swap" if votes_swap > votes_keep else "keep"
        reason = "final_vote_compare"

    return CrossOwnerEvidence(
        windows=windows, windows_used=windows_used,
        intent_windows=intent_windows, no_intent_windows=no_intent_windows,
        votes_keep=votes_keep, votes_swap=votes_swap,
        early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
        skipped_no_battle_start=skipped_no_battle_start,
        skipped_missing_bases=skipped_missing_bases,
        skipped_bad_scan=skipped_bad_scan,
        decision=decision,
        reason=reason,
    )


def analyze_actions_jsonl(
    path: str,
    *,
    allow_bad_lines: bool = True,
    max_bad_lines: int = 50,
    pre_mode_frames: int = 12,
    pre_battle_scan: int = 180,
    lead_frames: int = 2,
    early_exit_margin: int = 2,
    min_votes: int = 2,
) -> Dict[str, Any]:
    frames = read_jsonl(path, allow_bad_lines=allow_bad_lines, max_bad_lines=max_bad_lines)
    ev = analyze_actions_frames(
        frames,
        pre_mode_frames=pre_mode_frames,
        pre_battle_scan=pre_battle_scan,
        lead_frames=lead_frames,
        early_exit_margin=early_exit_margin,
        min_votes=min_votes,
    )
    return {
        "windows": ev.windows,
        "windows_used": ev.windows_used,
        "intent_windows": ev.intent_windows,
        "no_intent_windows": ev.no_intent_windows,
        "votes_keep": ev.votes_keep,
        "votes_swap": ev.votes_swap,
        "early_swap_triggered": ev.early_swap_triggered,
        "early_keep_bonus": ev.early_keep_bonus,
        "skipped_no_battle_start": ev.skipped_no_battle_start,
        "skipped_missing_bases": ev.skipped_missing_bases,
        "skipped_bad_scan": ev.skipped_bad_scan,
        "decision": ev.decision,
        "reason": ev.reason,
    }


def decide_cross_owner_swap(evidence: Dict[str, Any]) -> Tuple[str, str]:
    verdict = str(evidence.get("decision", "unclear"))
    reason = str(evidence.get("reason", ""))
    if verdict not in ("swap", "keep", "unclear"):
        return ("unclear", "invalid_decision_field")
    return (verdict, reason)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/dataset")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--pre-mode-frames", type=int, default=12)
    ap.add_argument("--pre-battle-scan", type=int, default=180)
    ap.add_argument("--lead-frames", type=int, default=2)
    ap.add_argument("--early-exit-margin", type=int, default=2)
    ap.add_argument("--min-votes", type=int, default=2)

    ap.add_argument("--max-bad-lines", type=int, default=50)
    args = ap.parse_args()

    paths: List[Tuple[str, str]] = []
    for root, _, files in os.walk(args.dataset_dir):
        if "actions.jsonl" in files:
            replay = os.path.basename(root)
            paths.append((replay, os.path.join(root, "actions.jsonl")))
    paths.sort()

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    counts: Dict[str, int] = {"swap": 0, "keep": 0, "unclear": 0}

    for replay, p in paths:
        evidence = analyze_actions_jsonl(
            p,
            allow_bad_lines=True,
            max_bad_lines=args.max_bad_lines,
            pre_mode_frames=args.pre_mode_frames,
            pre_battle_scan=args.pre_battle_scan,
            lead_frames=args.lead_frames,
            early_exit_margin=args.early_exit_margin,
            min_votes=args.min_votes,
        )
        verdict, reason = decide_cross_owner_swap(evidence)
        counts[verdict] = counts.get(verdict, 0) + 1

        print(
            f"[{verdict.upper()}] {replay} "
            f"used={evidence.get('windows_used',0)} "
            f"intent={evidence.get('intent_windows',0)} "
            f"no_intent={evidence.get('no_intent_windows',0)} "
            f"votes(K={evidence.get('votes_keep',0)},S={evidence.get('votes_swap',0)}) "
            f"early_swap={evidence.get('early_swap_triggered',0)} "
            f"reason={reason}"
        )

    print("")
    print(f"Replays scanned: {len(paths)}")
    print(f"swap   : {counts.get('swap',0)}")
    print(f"keep   : {counts.get('keep',0)}")
    print(f"unclear: {counts.get('unclear',0)}")


if __name__ == "__main__":
    main()
