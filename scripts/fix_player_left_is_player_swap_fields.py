#!/usr/bin/env python3
# scripts/fix_player_left_is_player_swap_fields.py
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional


# -----------------------------------------------------------------------------
# What we swap (ONLY these fields)
# -----------------------------------------------------------------------------

SWAP_PAIRS: List[Tuple[str, str]] = [
    ("player_health", "enemy_health"),
    ("player_pos", "enemy_pos"),
    ("player_charge", "enemy_charge"),
    ("player_chip", "enemy_chip"),
    ("player_game_emotion", "enemy_game_emotion"),
]


# -----------------------------------------------------------------------------
# JSONL helpers (tolerant + atomic)
# -----------------------------------------------------------------------------

def iter_jsonl_tolerant(path: Path, *, max_bad_lines: int = 50) -> Iterator[Dict[str, Any]]:
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] {path} bad json at line {line_no} (skipping)")
                if bad >= max_bad_lines:
                    raise RuntimeError(f"Too many bad json lines in {path} (>= {max_bad_lines}). Aborting.")
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl_atomic(path: Path, rows: Iterator[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for obj in rows:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Detection: "who is left more often?"
# -----------------------------------------------------------------------------

def _get_pos_xy(fr: Dict[str, Any], key: str) -> Optional[Tuple[float, float]]:
    v = fr.get(key)
    if not isinstance(v, list) or len(v) < 2:
        return None
    try:
        x = float(v[0])
        y = float(v[1])
    except Exception:
        return None
    if not (x == x and y == y):  # NaN check
        return None
    return (x, y)


def _is_battle_frame(fr: Dict[str, Any]) -> bool:
    # Same conservative heuristic used elsewhere: cust_gauge>0 => in battle flow.
    v = fr.get("cust_gauge", 0)
    try:
        return int(v) > 0
    except Exception:
        return False


@dataclass(frozen=True)
class LeftDecision:
    verdict: str  # "swap" | "keep" | "unclear"
    reason: str
    battle_frames_scanned: int
    comparable_frames: int
    player_left_votes: int
    enemy_left_votes: int
    ties: int


def decide_left_is_player_streaming(
    actions_path: Path,
    *,
    max_bad_lines: int,
    max_scan_frames: int,
    min_votes: int,
    ratio: float,
    dx_deadzone: float,
    battle_only: bool,
) -> LeftDecision:
    """
    Vote per comparable frame:
      - if player_pos.x < enemy_pos.x - dx_deadzone => player_left_votes++
      - if enemy_pos.x < player_pos.x - dx_deadzone => enemy_left_votes++
      - else tie

    Decision:
      - if enemy_left_votes >= player_left_votes*ratio => SWAP (because "enemy" is left more, so enemy is actually player)
      - if player_left_votes >= enemy_left_votes*ratio => KEEP
      - else UNCLEAR

    Streaming + early exit when decisive.
    """
    scanned_battle = 0
    comparable = 0
    p_left = 0
    e_left = 0
    ties = 0

    def dominance() -> Optional[str]:
        total_votes = p_left + e_left
        if total_votes < min_votes:
            return None
        # if "enemy" is left much more => the labels are flipped => swap
        if e_left >= int(p_left * ratio + 0.5):
            return "swap"
        if p_left >= int(e_left * ratio + 0.5):
            return "keep"
        return None

    for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=max_bad_lines):
        if battle_only and not _is_battle_frame(fr):
            continue

        if max_scan_frames > 0 and scanned_battle >= max_scan_frames:
            break

        ppos = _get_pos_xy(fr, "player_pos")
        epos = _get_pos_xy(fr, "enemy_pos")
        if ppos is None or epos is None:
            # still counts as a scanned battle frame if we’re in battle stream,
            # but not a comparable frame
            scanned_battle += 1
            continue

        comparable += 1
        dx = ppos[0] - epos[0]
        if dx < -dx_deadzone:
            p_left += 1
        elif dx > dx_deadzone:
            e_left += 1
        else:
            ties += 1

        scanned_battle += 1

        dv = dominance()
        if dv is not None:
            if dv == "swap":
                return LeftDecision(
                    verdict="swap",
                    reason=f"early:enemy_left>=player_left*{ratio}",
                    battle_frames_scanned=scanned_battle,
                    comparable_frames=comparable,
                    player_left_votes=p_left,
                    enemy_left_votes=e_left,
                    ties=ties,
                )
            return LeftDecision(
                verdict="keep",
                reason=f"early:player_left>=enemy_left*{ratio}",
                battle_frames_scanned=scanned_battle,
                comparable_frames=comparable,
                player_left_votes=p_left,
                enemy_left_votes=e_left,
                ties=ties,
            )

    total_votes = p_left + e_left
    if total_votes < min_votes:
        return LeftDecision(
            verdict="unclear",
            reason=f"votes<{min_votes}",
            battle_frames_scanned=scanned_battle,
            comparable_frames=comparable,
            player_left_votes=p_left,
            enemy_left_votes=e_left,
            ties=ties,
        )

    if e_left >= int(p_left * ratio + 0.5):
        return LeftDecision(
            verdict="swap",
            reason=f"enemy_left>=player_left*{ratio}",
            battle_frames_scanned=scanned_battle,
            comparable_frames=comparable,
            player_left_votes=p_left,
            enemy_left_votes=e_left,
            ties=ties,
        )

    if p_left >= int(e_left * ratio + 0.5):
        return LeftDecision(
            verdict="keep",
            reason=f"player_left>=enemy_left*{ratio}",
            battle_frames_scanned=scanned_battle,
            comparable_frames=comparable,
            player_left_votes=p_left,
            enemy_left_votes=e_left,
            ties=ties,
        )

    return LeftDecision(
        verdict="unclear",
        reason="no_clear_dominance",
        battle_frames_scanned=scanned_battle,
        comparable_frames=comparable,
        player_left_votes=p_left,
        enemy_left_votes=e_left,
        ties=ties,
    )


# -----------------------------------------------------------------------------
# Apply: swap ONLY the requested fields
# -----------------------------------------------------------------------------

def _swap_pair_inplace(fr: Dict[str, Any], a: str, b: str) -> bool:
    """
    Swap values for keys a<->b if at least one exists.
    Returns True if any change occurred.
    """
    if a not in fr and b not in fr:
        return False
    fr[a], fr[b] = fr.get(b), fr.get(a)
    return True


def apply_swap_fields_atomic(actions_path: Path, *, max_bad_lines: int) -> None:
    def transformed() -> Iterator[Dict[str, Any]]:
        for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=max_bad_lines):
            for a, b in SWAP_PAIRS:
                _swap_pair_inplace(fr, a, b)
            yield fr

    write_jsonl_atomic(actions_path, transformed())


def find_actions_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/actions.jsonl"))


# -----------------------------------------------------------------------------
# Parallel worker
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkerResult:
    folder: str
    path: str
    verdict: str
    reason: str
    scanned: int
    comparable: int
    p_left: int
    e_left: int
    ties: int
    applied: bool
    error: str


def _process_one(args: Tuple[str, bool, int, int, int, float, float, bool]) -> WorkerResult:
    path_str, do_apply, max_bad_lines, max_scan_frames, min_votes, ratio, dx_deadzone, battle_only = args
    actions_path = Path(path_str)
    folder = actions_path.parent.name
    path_out = str(actions_path).replace("\\", "/")

    try:
        dec = decide_left_is_player_streaming(
            actions_path,
            max_bad_lines=max_bad_lines,
            max_scan_frames=max_scan_frames,
            min_votes=min_votes,
            ratio=ratio,
            dx_deadzone=dx_deadzone,
            battle_only=battle_only,
        )

        applied = False
        if do_apply and dec.verdict == "swap":
            apply_swap_fields_atomic(actions_path, max_bad_lines=max_bad_lines)
            applied = True

        return WorkerResult(
            folder=folder,
            path=path_out,
            verdict=dec.verdict,
            reason=dec.reason,
            scanned=dec.battle_frames_scanned,
            comparable=dec.comparable_frames,
            p_left=dec.player_left_votes,
            e_left=dec.enemy_left_votes,
            ties=dec.ties,
            applied=applied,
            error="",
        )
    except Exception as e:
        return WorkerResult(
            folder=folder,
            path=path_out,
            verdict="error",
            reason="",
            scanned=0,
            comparable=0,
            p_left=0,
            e_left=0,
            ties=0,
            applied=False,
            error=str(e),
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect whether player/enemy labels are flipped using left-right position heuristic, then swap ONLY selected fields."
    )
    ap.add_argument("--root", type=str, default="data/dataset")
    ap.add_argument("--apply", action="store_true", help="rewrite actions.jsonl for verdict=swap")
    ap.add_argument("--report", type=str, default="player_left_is_player_swap_fields_report.csv")

    ap.add_argument("--max-bad-lines", type=int, default=50)
    ap.add_argument("--max-scan-frames", type=int, default=50_000, help="battle frames to scan per replay (0 = no limit)")
    ap.add_argument("--min-votes", type=int, default=500, help="min decisive votes needed to decide swap/keep")
    ap.add_argument("--ratio", type=float, default=1.15, help="dominance ratio needed to decide (>=1.0)")
    ap.add_argument("--dx-deadzone", type=float, default=2.0, help="ignore frames where |dx| <= deadzone (reduces jitter)")
    ap.add_argument("--battle-only", action="store_true", help="only consider frames where cust_gauge>0")

    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel workers")

    args = ap.parse_args()

    root = Path(args.root)
    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    if not args.apply:
        print("\n[DRY RUN MODE] No files will be modified. Use --apply to execute swaps.\n")
    else:
        print("\n[APPLY MODE] Files WILL be modified in place (atomic rewrite).\n")

    tasks: List[Tuple[str, bool, int, int, int, float, float, bool]] = [
        (
            str(p),
            bool(args.apply),
            int(args.max_bad_lines),
            int(args.max_scan_frames),
            int(args.min_votes),
            float(args.ratio),
            float(args.dx_deadzone),
            bool(args.battle_only),
        )
        for p in files
    ]

    results: List[WorkerResult] = []
    use_parallel = (args.workers or 1) > 1 and len(tasks) > 1

    def log_line(res: WorkerResult) -> None:
        prefix = "[APPLY]" if args.apply else "[DRY]"
        if res.verdict == "error":
            msg = f"{prefix} {res.folder}: ERROR {res.error}"
        else:
            msg = (
                f"{prefix} {res.folder}: {res.verdict}  "
                f"L(p={res.p_left},e={res.e_left},ties={res.ties})  "
                f"scanned={res.scanned} comparable={res.comparable}  reason={res.reason}"
            )
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)

    if use_parallel:
        with multiprocessing.Pool(processes=args.workers) as pool:
            it = pool.imap_unordered(_process_one, tasks, chunksize=1)
            if tqdm is not None:
                it = tqdm(it, total=len(tasks), desc="LeftSwap", unit="file")
            for res in it:
                results.append(res)
                log_line(res)
    else:
        for t in tasks:
            res = _process_one(t)
            results.append(res)
            log_line(res)

    # Stats
    n_swap = sum(1 for r in results if r.verdict == "swap")
    n_keep = sum(1 for r in results if r.verdict == "keep")
    n_unclear = sum(1 for r in results if r.verdict == "unclear")
    n_error = sum(1 for r in results if r.verdict == "error")
    n_applied = sum(1 for r in results if r.applied)

    # Report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "folder",
        "path",
        "verdict",
        "reason",
        "player_left_votes",
        "enemy_left_votes",
        "ties",
        "battle_frames_scanned",
        "comparable_frames",
        "applied",
        "error",
    ]

    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "folder": r.folder,
                    "path": r.path,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "player_left_votes": r.p_left,
                    "enemy_left_votes": r.e_left,
                    "ties": r.ties,
                    "battle_frames_scanned": r.scanned,
                    "comparable_frames": r.comparable,
                    "applied": str(r.applied),
                    "error": r.error,
                }
            )

    print("")
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Done ({mode}). files={len(files)}  swap={n_swap}  keep={n_keep}  unclear={n_unclear}  error={n_error}")
    if args.apply:
        print(f"Applied rewrites: {n_applied}")
    print(f"Workers: {args.workers}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
