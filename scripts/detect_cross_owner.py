#!/usr/bin/env python3
# scripts/detect_cross_owner.py
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


def _enemy_emotion_present_impossible(frames: List[dict]) -> bool:
    """
    Your invariant: game never populates enemy_emotion.
    If it exists with a non-null/non-zero value anywhere, labels are swapped/mislabeled.
    """
    for fr in frames:
        if "enemy_emotion" in fr and fr.get("enemy_emotion") is not None and fr.get("enemy_emotion", 0) != 0:
            return True
    return False


def _cust_is_zero(fr: Dict[str, Any]) -> bool:
    return _safe_int(fr.get("cust_gauge", 0), 0) == 0


def _cust_is_pos(fr: Dict[str, Any]) -> bool:
    return _safe_int(fr.get("cust_gauge", 0), 0) > 0


def _first_index_where(frames: List[dict], i0: int, i1: int, pred) -> Optional[int]:
    n = len(frames)
    a = max(0, min(i0, n))
    b = max(0, min(i1, n))
    for i in range(a, b):
        if pred(frames[i]):
            return i
    return None


def _last_run_first_index(
    frames: List[dict],
    i0: int,
    i1: int,
    pred,
) -> Optional[int]:
    """
    Find the last contiguous run where pred(frame) is True in [i0,i1),
    and return the first index of that last run.
    """
    n = len(frames)
    a = max(0, min(i0, n))
    b = max(0, min(i1, n))
    if b <= a:
        return None

    last_run_start: Optional[int] = None
    in_run = False
    run_start = -1

    for i in range(a, b):
        if pred(frames[i]):
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                last_run_start = run_start
                in_run = False

    if in_run:
        last_run_start = run_start

    return last_run_start


def _menu_ok_player_emotion_swap_tell(
    frames: List[dict],
    s0: int,
    s1: int,
    *,
    ok_menu_index: int = 10,
    menu_index_key: str = "selected_menu_index",
    battle_emotion_lookahead: int = 8,
) -> Tuple[bool, str, Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Corrected menu-confirm tell:

    - In the inside_window segment [s0,s1), restrict to cust_gauge==0.
    - baseline = player_emotion at the first frame in the segment where inside_window==True and cust_gauge==0.
    - Find the LAST contiguous run where selected_menu_index == ok_menu_index within that segment (cust==0),
      and take the FIRST frame of that LAST run (this corresponds to the actual OK click segment).
    - If player_emotion at that "OK segment start" equals baseline (unchanged),
      then at the next battle start (first cust_gauge>0 after s1),
      if player_emotion changes vs baseline => SWAP.

    Returns:
      (triggered_swap, reason, base_idx, ok_idx, battle_idx, base_emotion, battle_emotion_mode)
    """
    base_idx = _first_index_where(
        frames, s0, s1,
        lambda fr: bool(fr.get("inside_window", False)) and _cust_is_zero(fr)
    )
    if base_idx is None:
        return (False, "menu_ok:no_base", None, None, None, None, None)

    base_em = frames[base_idx].get("player_emotion")
    if base_em is None:
        return (False, "menu_ok:no_base_emotion", base_idx, None, None, None, None)
    base_em_i = _safe_int(base_em, -1)
    if base_em_i < 0:
        return (False, "menu_ok:bad_base_emotion", base_idx, None, None, None, None)

    ok_idx = _last_run_first_index(
        frames,
        base_idx,
        s1,
        lambda fr: (
            bool(fr.get("inside_window", False))
            and _cust_is_zero(fr)
            and _safe_int(fr.get(menu_index_key, -9999), -9999) == ok_menu_index
        ),
    )
    if ok_idx is None:
        return (False, "menu_ok:no_ok_run", base_idx, None, None, base_em_i, None)

    ok_em = frames[ok_idx].get("player_emotion")
    if ok_em is None:
        return (False, "menu_ok:no_ok_emotion", base_idx, ok_idx, None, base_em_i, None)
    ok_em_i = _safe_int(ok_em, -1)
    if ok_em_i < 0:
        return (False, "menu_ok:bad_ok_emotion", base_idx, ok_idx, None, base_em_i, None)

    if ok_em_i != base_em_i:
        return (False, "menu_ok:emotion_changed_in_menu", base_idx, ok_idx, None, base_em_i, None)

    battle_idx = _find_battle_start_by_cust_gauge(frames, s1)
    if battle_idx is None:
        return (False, "menu_ok:no_battle_start", base_idx, ok_idx, None, base_em_i, None)

    b0 = battle_idx
    b1 = min(len(frames), battle_idx + max(1, int(battle_emotion_lookahead)))
    battle_mode = _range_mode(frames, b0, b1, "player_emotion")
    if battle_mode is None:
        return (False, "menu_ok:no_battle_emotion_mode", base_idx, ok_idx, battle_idx, base_em_i, None)

    battle_mode_i = int(battle_mode)

    if battle_mode_i != base_em_i:
        return (
            True,
            "menu_ok_player_emotion_changes_at_battle_start",
            base_idx,
            ok_idx,
            battle_idx,
            base_em_i,
            battle_mode_i,
        )

    return (False, "menu_ok:no_change_at_battle", base_idx, ok_idx, battle_idx, base_em_i, battle_mode_i)


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


# -----------------------------
# Debug instrumentation
# -----------------------------

@dataclass(frozen=True)
class WindowDebug:
    window_idx: int
    s0: int
    s1: int

    intent: bool
    intent_pe_changed: Optional[bool]
    intent_sc_changed: Optional[bool]

    menu_ok_swap: bool
    menu_ok_reason: str
    menu_base_idx: Optional[int]
    menu_ok_idx: Optional[int]
    menu_battle_idx: Optional[int]
    menu_base_em: Optional[int]
    menu_battle_em: Optional[int]

    battle_start: Optional[int]
    p_base: Optional[int]
    e_base: Optional[int]

    scan_start: Optional[int]
    scan_end: Optional[int]

    p_chg_i: Optional[int]
    e_chg_i: Optional[int]

    voted: str  # "keep"|"swap"|"none"|"early_swap"|"keep_bonus"
    note: str


def _debug_intent_components(frames: List[dict], s0: int, s1: int) -> Tuple[bool, Optional[bool], Optional[bool]]:
    pe = _mode_diff(frames, s0, s1, "player_emotion", edge=12)
    sc = _mode_diff(frames, s0, s1, "selected_cross_index", edge=12)
    if pe is None and sc is None:
        return (False, pe, sc)
    return (bool(pe) or bool(sc), pe, sc)


def _print_debug(dbg: List[WindowDebug], debug_limit: int) -> None:
    lim = debug_limit if debug_limit and debug_limit > 0 else len(dbg)
    for d in dbg[:lim]:
        print(
            f"[WIN {d.window_idx}] seg=({d.s0},{d.s1}) "
            f"intent={d.intent} pe={d.intent_pe_changed} sc={d.intent_sc_changed} "
            f"menu_ok_swap={d.menu_ok_swap} menu_ok_reason={d.menu_ok_reason} "
            f"menu(base={d.menu_base_idx},ok={d.menu_ok_idx},battle={d.menu_battle_idx},"
            f"base_em={d.menu_base_em},battle_em={d.menu_battle_em}) "
            f"battle_start={d.battle_start} "
            f"base(P={d.p_base},E={d.e_base}) "
            f"scan=({d.scan_start},{d.scan_end}) "
            f"chg(P={d.p_chg_i},E={d.e_chg_i}) "
            f"voted={d.voted} note={d.note}"
        )


# -----------------------------
# Analysis
# -----------------------------

def analyze_actions_frames(
    frames: List[dict],
    *,
    pre_mode_frames: int = 12,
    pre_battle_scan: int = 180,
    lead_frames: int = 2,
    early_exit_margin: int = 2,
    min_votes: int = 2,
    # Menu-ok tell tunables
    ok_menu_index: int = 10,
    menu_index_key: str = "selected_menu_index",
    battle_emotion_lookahead: int = 8,
    # Debug
    debug: bool = False,
    debug_limit: int = 0,
) -> CrossOwnerEvidence:
    if not frames:
        return CrossOwnerEvidence(
            windows=0, windows_used=0, intent_windows=0, no_intent_windows=0,
            votes_keep=0, votes_swap=0, early_swap_triggered=0, early_keep_bonus=0,
            skipped_no_battle_start=0, skipped_missing_bases=0, skipped_bad_scan=0,
            decision="unclear", reason="no_frames",
        )

    # Hard tell: enemy_emotion should never exist with a non-zero value.
    if _enemy_emotion_present_impossible(frames):
        return CrossOwnerEvidence(
            windows=0, windows_used=0, intent_windows=0, no_intent_windows=0,
            votes_keep=0, votes_swap=0, early_swap_triggered=1, early_keep_bonus=0,
            skipped_no_battle_start=0, skipped_missing_bases=0, skipped_bad_scan=0,
            decision="swap", reason="enemy_emotion_present_impossible",
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

    dbg: List[WindowDebug] = []
    win_idx = 0

    for (s0, s1) in segs:
        intent, pe_changed, sc_changed = _debug_intent_components(frames, s0, s1)
        if intent:
            intent_windows += 1
        else:
            no_intent_windows += 1

        # NEW EARLY SWAP TELL (your menu-confirm rule)
        menu_ok_swap, menu_ok_reason, base_idx, ok_idx, menu_battle_idx, base_em, battle_em = _menu_ok_player_emotion_swap_tell(
            frames,
            s0,
            s1,
            ok_menu_index=ok_menu_index,
            menu_index_key=menu_index_key,
            battle_emotion_lookahead=battle_emotion_lookahead,
        )
        if menu_ok_swap:
            early_swap_triggered += 1
            if debug:
                dbg.append(
                    WindowDebug(
                        window_idx=win_idx,
                        s0=s0, s1=s1,
                        intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                        menu_ok_swap=True, menu_ok_reason=menu_ok_reason,
                        menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                        menu_base_em=base_em, menu_battle_em=battle_em,
                        battle_start=menu_battle_idx,
                        p_base=None, e_base=None,
                        scan_start=None, scan_end=None,
                        p_chg_i=None, e_chg_i=None,
                        voted="early_swap",
                        note="menu_ok:player_emotion_changed_at_battle_start => SWAP",
                    )
                )
                _print_debug(dbg, debug_limit)
            return CrossOwnerEvidence(
                windows=windows, windows_used=windows_used,
                intent_windows=intent_windows, no_intent_windows=no_intent_windows,
                votes_keep=votes_keep, votes_swap=votes_swap,
                early_swap_triggered=early_swap_triggered, early_keep_bonus=early_keep_bonus,
                skipped_no_battle_start=skipped_no_battle_start,
                skipped_missing_bases=skipped_missing_bases,
                skipped_bad_scan=skipped_bad_scan,
                decision="swap",
                reason=menu_ok_reason,
            )

        battle_start = _find_battle_start_by_cust_gauge(frames, s1)
        if battle_start is None:
            skipped_no_battle_start += 1
            if debug:
                dbg.append(
                    WindowDebug(
                        window_idx=win_idx,
                        s0=s0, s1=s1,
                        intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                        menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                        menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                        menu_base_em=base_em, menu_battle_em=battle_em,
                        battle_start=None,
                        p_base=None, e_base=None,
                        scan_start=None, scan_end=None,
                        p_chg_i=None, e_chg_i=None,
                        voted="none",
                        note="skip:no_battle_start",
                    )
                )
            win_idx += 1
            continue

        p_base = _range_mode(frames, s1 - pre_mode_frames, s1, "player_game_emotion")
        e_base = _range_mode(frames, s1 - pre_mode_frames, s1, "enemy_game_emotion")
        if p_base is None or e_base is None:
            skipped_missing_bases += 1
            if debug:
                dbg.append(
                    WindowDebug(
                        window_idx=win_idx,
                        s0=s0, s1=s1,
                        intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                        menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                        menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                        menu_base_em=base_em, menu_battle_em=battle_em,
                        battle_start=battle_start,
                        p_base=p_base, e_base=e_base,
                        scan_start=None, scan_end=None,
                        p_chg_i=None, e_chg_i=None,
                        voted="none",
                        note="skip:missing_bases",
                    )
                )
            win_idx += 1
            continue

        scan_start = max(s1, battle_start - pre_battle_scan)
        scan_end = battle_start
        if scan_end <= scan_start:
            skipped_bad_scan += 1
            if debug:
                dbg.append(
                    WindowDebug(
                        window_idx=win_idx,
                        s0=s0, s1=s1,
                        intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                        menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                        menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                        menu_base_em=base_em, menu_battle_em=battle_em,
                        battle_start=battle_start,
                        p_base=int(p_base), e_base=int(e_base),
                        scan_start=scan_start, scan_end=scan_end,
                        p_chg_i=None, e_chg_i=None,
                        voted="none",
                        note="skip:bad_scan",
                    )
                )
            win_idx += 1
            continue

        p_chg_i = _first_value_change(frames, scan_start, scan_end, "player_game_emotion", base=int(p_base))
        e_chg_i = _first_value_change(frames, scan_start, scan_end, "enemy_game_emotion", base=int(e_base))

        dp = (p_chg_i is not None)
        de = (e_chg_i is not None)

        windows_used += 1

        voted = "none"
        note = ""

        # No-intent windows: strong tells
        if not intent:
            if dp and not de:
                early_swap_triggered += 1
                voted = "early_swap"
                note = "no_intent:player_game_emotion_changed_first => SWAP"
                if debug:
                    dbg.append(
                        WindowDebug(
                            window_idx=win_idx,
                            s0=s0, s1=s1,
                            intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                            menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                            menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                            menu_base_em=base_em, menu_battle_em=battle_em,
                            battle_start=battle_start,
                            p_base=int(p_base), e_base=int(e_base),
                            scan_start=scan_start, scan_end=scan_end,
                            p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                            voted=voted,
                            note=note,
                        )
                    )
                    _print_debug(dbg, debug_limit)
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
                voted = "keep_bonus"
                note = "no_intent:enemy_changed_player_not (+2 keep)"
                if (votes_keep + votes_swap) >= min_votes and votes_keep >= votes_swap + early_exit_margin:
                    if debug:
                        dbg.append(
                            WindowDebug(
                                window_idx=win_idx,
                                s0=s0, s1=s1,
                                intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                                menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                                menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                                menu_base_em=base_em, menu_battle_em=battle_em,
                                battle_start=battle_start,
                                p_base=int(p_base), e_base=int(e_base),
                                scan_start=scan_start, scan_end=scan_end,
                                p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                                voted=voted,
                                note=note + " => KEEP(decisive)",
                            )
                        )
                        _print_debug(dbg, debug_limit)
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

            if debug:
                dbg.append(
                    WindowDebug(
                        window_idx=win_idx,
                        s0=s0, s1=s1,
                        intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                        menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                        menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                        menu_base_em=base_em, menu_battle_em=battle_em,
                        battle_start=battle_start,
                        p_base=int(p_base), e_base=int(e_base),
                        scan_start=scan_start, scan_end=scan_end,
                        p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                        voted=voted,
                        note=note or "no_intent:no_vote",
                    )
                )
            win_idx += 1
            continue

        # Intent windows: vote based on who changes first pre-battle
        if not dp and not de:
            voted = "none"
            note = "intent:no_changes"
        elif dp and not de:
            votes_keep += 1
            voted = "keep"
            note = "intent:player_changed_only => keep"
        elif de and not dp:
            votes_swap += 1
            voted = "swap"
            note = "intent:enemy_changed_only => swap"
        else:
            assert p_chg_i is not None and e_chg_i is not None
            if p_chg_i + lead_frames < e_chg_i:
                votes_keep += 1
                voted = "keep"
                note = "intent:player_changed_first => keep"
            elif e_chg_i + lead_frames < p_chg_i:
                votes_swap += 1
                voted = "swap"
                note = "intent:enemy_changed_first => swap"
            else:
                voted = "none"
                note = "intent:too_close_no_vote"

        if (votes_keep + votes_swap) >= min_votes:
            if votes_swap >= votes_keep + early_exit_margin:
                if debug:
                    dbg.append(
                        WindowDebug(
                            window_idx=win_idx,
                            s0=s0, s1=s1,
                            intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                            menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                            menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                            menu_base_em=base_em, menu_battle_em=battle_em,
                            battle_start=battle_start,
                            p_base=int(p_base), e_base=int(e_base),
                            scan_start=scan_start, scan_end=scan_end,
                            p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                            voted=voted,
                            note=note + " => SWAP(decisive)",
                        )
                    )
                    _print_debug(dbg, debug_limit)
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
                if debug:
                    dbg.append(
                        WindowDebug(
                            window_idx=win_idx,
                            s0=s0, s1=s1,
                            intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                            menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                            menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                            menu_base_em=base_em, menu_battle_em=battle_em,
                            battle_start=battle_start,
                            p_base=int(p_base), e_base=int(e_base),
                            scan_start=scan_start, scan_end=scan_end,
                            p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                            voted=voted,
                            note=note + " => KEEP(decisive)",
                        )
                    )
                    _print_debug(dbg, debug_limit)
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

        if debug:
            dbg.append(
                WindowDebug(
                    window_idx=win_idx,
                    s0=s0, s1=s1,
                    intent=intent, intent_pe_changed=pe_changed, intent_sc_changed=sc_changed,
                    menu_ok_swap=False, menu_ok_reason=menu_ok_reason,
                    menu_base_idx=base_idx, menu_ok_idx=ok_idx, menu_battle_idx=menu_battle_idx,
                    menu_base_em=base_em, menu_battle_em=battle_em,
                    battle_start=battle_start,
                    p_base=int(p_base), e_base=int(e_base),
                    scan_start=scan_start, scan_end=scan_end,
                    p_chg_i=p_chg_i, e_chg_i=e_chg_i,
                    voted=voted,
                    note=note,
                )
            )

        win_idx += 1

    # Final decision
    if debug and dbg:
        _print_debug(dbg, debug_limit)

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
    # Menu-ok tell tunables
    ok_menu_index: int = 10,
    menu_index_key: str = "selected_menu_index",
    battle_emotion_lookahead: int = 8,
    # Debug
    debug: bool = False,
    debug_limit: int = 0,
) -> Dict[str, Any]:
    frames = read_jsonl(path, allow_bad_lines=allow_bad_lines, max_bad_lines=max_bad_lines)
    ev = analyze_actions_frames(
        frames,
        pre_mode_frames=pre_mode_frames,
        pre_battle_scan=pre_battle_scan,
        lead_frames=lead_frames,
        early_exit_margin=early_exit_margin,
        min_votes=min_votes,
        ok_menu_index=ok_menu_index,
        menu_index_key=menu_index_key,
        battle_emotion_lookahead=battle_emotion_lookahead,
        debug=debug,
        debug_limit=debug_limit,
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
    ap.add_argument("--early-exit-margin", type=int, default=3)
    ap.add_argument("--min-votes", type=int, default=2)

    ap.add_argument("--ok-menu-index", type=int, default=10)
    ap.add_argument("--menu-index-key", type=str, default="selected_menu_index")
    ap.add_argument("--battle-emotion-lookahead", type=int, default=8)

    ap.add_argument("--max-bad-lines", type=int, default=50)

    # Debugging a specific replay folder name
    ap.add_argument("--debug-replay", type=str, default="")
    ap.add_argument("--debug-limit", type=int, default=50)

    args = ap.parse_args()

    paths: List[Tuple[str, str]] = []
    for root, _, files in os.walk(args.dataset_dir):
        if "actions.jsonl" in files:
            replay = os.path.basename(root)
            paths.append((replay, os.path.join(root, "actions.jsonl")))
    paths.sort()

    if args.debug_replay:
        paths = [(r, p) for (r, p) in paths if r == args.debug_replay]
        if not paths:
            raise SystemExit(f"Replay not found under {args.dataset_dir}: {args.debug_replay}")

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
            ok_menu_index=args.ok_menu_index,
            menu_index_key=args.menu_index_key,
            battle_emotion_lookahead=args.battle_emotion_lookahead,
            debug=bool(args.debug_replay),
            debug_limit=args.debug_limit,
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

    if not args.debug_replay:
        print("")
        print(f"Replays scanned: {len(paths)}")
        print(f"swap   : {counts.get('swap',0)}")
        print(f"keep   : {counts.get('keep',0)}")
        print(f"unclear: {counts.get('unclear',0)}")


if __name__ == "__main__":
    main()
