#!/usr/bin/env python3
# scripts/batch_fix_cross_owner.py
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from tqdm import tqdm


# --------------------------------------------------------------------
# Import scripts/detect_cross_owner.py by file path (no package needed)
# --------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DETECT_PATH = _SCRIPTS_DIR / "detect_cross_owner.py"

_spec = importlib.util.spec_from_file_location("detect_cross_owner", str(_DETECT_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load detect_cross_owner from {_DETECT_PATH}")

detect = importlib.util.module_from_spec(_spec)
sys.modules["detect_cross_owner"] = detect
_spec.loader.exec_module(detect)  # type: ignore

analyze_actions_jsonl = detect.analyze_actions_jsonl
decide_cross_owner_swap = detect.decide_cross_owner_swap


# --------------------------------------------------------------------
# JSONL rewrite (tolerant)
# --------------------------------------------------------------------

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


def apply_swap_game_emotion(fr: Dict[str, Any]) -> None:
    # match convert_dataset swap style: tolerate missing keys
    fr["player_game_emotion"], fr["enemy_game_emotion"] = fr.get("enemy_game_emotion"), fr.get("player_game_emotion")


# --------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------

def find_actions_files(root: Path) -> List[Path]:
    # Match your other batch scripts: direct children run folders
    out: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            p = child / "actions.jsonl"
            if p.is_file():
                out.append(p)
    return out


def _parse_name_set(csv_or_empty: str) -> Set[str]:
    s = (csv_or_empty or "").strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


# --------------------------------------------------------------------
# Parallel worker
# --------------------------------------------------------------------

@dataclass(frozen=True)
class WorkerArgs:
    actions_path: str
    max_bad_lines: int

    pre_mode_frames: int
    pre_battle_scan: int
    lead_frames: int
    early_exit_margin: int
    min_votes: int

    apply: bool

    force_swap: Set[str]
    force_keep: Set[str]

    debug_replay: str
    debug_limit: int


@dataclass(frozen=True)
class WorkerResult:
    folder: str
    path: str
    verdict: str
    reason: str
    forced: str  # "", "swap", "keep"
    modified: bool

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


def _process_one(arg: WorkerArgs) -> WorkerResult:
    actions_path = Path(arg.actions_path)
    replay = actions_path.parent.name

    # Enable detector debug only for one replay (avoid interleaved output)
    debug = bool(arg.debug_replay) and (arg.debug_replay == replay)

    evidence = analyze_actions_jsonl(
        str(actions_path),
        allow_bad_lines=True,
        max_bad_lines=arg.max_bad_lines,
        pre_mode_frames=arg.pre_mode_frames,
        pre_battle_scan=arg.pre_battle_scan,
        lead_frames=arg.lead_frames,
        early_exit_margin=arg.early_exit_margin,
        min_votes=arg.min_votes,
        debug=debug,
        debug_limit=arg.debug_limit,
    )

    verdict, reason = decide_cross_owner_swap(evidence)

    forced = ""
    if replay in arg.force_swap:
        verdict, reason = ("swap", "forced_swap")
        forced = "swap"
    elif replay in arg.force_keep:
        verdict, reason = ("keep", "forced_keep")
        forced = "keep"

    modified = False
    if arg.apply and verdict == "swap":
        def transformed() -> Iterator[Dict[str, Any]]:
            for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=arg.max_bad_lines):
                apply_swap_game_emotion(fr)
                yield fr

        write_jsonl_atomic(actions_path, transformed())
        modified = True

    def _i(k: str) -> int:
        try:
            return int(evidence.get(k, 0) or 0)
        except Exception:
            return 0

    return WorkerResult(
        folder=replay,
        path=str(actions_path).replace("\\", "/"),
        verdict=verdict,
        reason=str(reason),
        forced=forced,
        modified=modified,

        windows=_i("windows"),
        windows_used=_i("windows_used"),
        intent_windows=_i("intent_windows"),
        no_intent_windows=_i("no_intent_windows"),
        votes_keep=_i("votes_keep"),
        votes_swap=_i("votes_swap"),
        early_swap_triggered=_i("early_swap_triggered"),
        early_keep_bonus=_i("early_keep_bonus"),
        skipped_no_battle_start=_i("skipped_no_battle_start"),
        skipped_missing_bases=_i("skipped_missing_bases"),
        skipped_bad_scan=_i("skipped_bad_scan"),
    )


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Accept both flags (alias). --root is preferred, --dataset-dir kept for back-compat.
    ap.add_argument("--root", type=str, default="", help="dataset root containing run folders (preferred)")
    ap.add_argument("--dataset-dir", type=str, default="", help="alias of --root (back-compat)")

    # Detector tunables (same as detect_cross_owner.py)
    ap.add_argument("--pre-mode-frames", type=int, default=12)
    ap.add_argument("--pre-battle-scan", type=int, default=180)
    ap.add_argument("--lead-frames", type=int, default=2)
    ap.add_argument("--early-exit-margin", type=int, default=2)
    ap.add_argument("--min-votes", type=int, default=2)

    ap.add_argument("--apply", action="store_true", help="rewrite actions.jsonl for verdict=swap")
    ap.add_argument("--report", type=str, default="cross_owner_report.csv")
    ap.add_argument("--max-bad-lines", type=int, default=50)

    ap.add_argument("--force-swap", type=str, default="", help="comma-separated replay folder names to force verdict=swap")
    ap.add_argument("--force-keep", type=str, default="", help="comma-separated replay folder names to force verdict=keep")

    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel worker processes")

    ap.add_argument("--debug-replay", type=str, default="", help="exact replay folder name to print window debug for")
    ap.add_argument("--debug-limit", type=int, default=0, help="limit number of debug windows printed (0 = all)")

    args = ap.parse_args()

    root_str = (args.root or "").strip() or (args.dataset_dir or "").strip() or "data/dataset"
    root = Path(root_str)

    force_swap = _parse_name_set(args.force_swap)
    force_keep = _parse_name_set(args.force_keep)
    overlap = force_swap & force_keep
    if overlap:
        raise SystemExit(f"Same replay listed in both --force-swap and --force-keep: {sorted(overlap)}")

    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    # If debug is requested, force workers=1 so debug output is readable and deterministic.
    workers = int(args.workers)
    if args.debug_replay:
        workers = 1

    task_args: List[WorkerArgs] = [
        WorkerArgs(
            actions_path=str(p),
            max_bad_lines=int(args.max_bad_lines),
            pre_mode_frames=int(args.pre_mode_frames),
            pre_battle_scan=int(args.pre_battle_scan),
            lead_frames=int(args.lead_frames),
            early_exit_margin=int(args.early_exit_margin),
            min_votes=int(args.min_votes),
            apply=bool(args.apply),
            force_swap=force_swap,
            force_keep=force_keep,
            debug_replay=str(args.debug_replay),
            debug_limit=int(args.debug_limit),
        )
        for p in files
    ]

    t0 = time.time()

    # Run
    results: List[WorkerResult] = []
    if workers <= 1:
        for a in tqdm(task_args, total=len(task_args), desc="CrossOwner", unit="file"):
            results.append(_process_one(a))
    else:
        with mp.Pool(processes=workers) as pool:
            it = pool.imap_unordered(_process_one, task_args, chunksize=1)
            for r in tqdm(it, total=len(task_args), desc="CrossOwner", unit="file"):
                results.append(r)

    # Deterministic report order by folder
    results.sort(key=lambda r: r.folder)

    # Summary counts
    n_swap = sum(1 for r in results if r.verdict == "swap")
    n_keep = sum(1 for r in results if r.verdict == "keep")
    n_unclear = sum(1 for r in results if r.verdict == "unclear")
    n_forced = sum(1 for r in results if r.forced)

    # Console output (compact)
    for r in results:
        prefix = "[APPLY]" if args.apply else "[DRY]"
        forced = f"  [FORCED:{r.forced.upper()}]" if r.forced else ""
        print(
            f"{prefix} [{r.verdict.upper()}] {r.folder}  "
            f"used={r.windows_used}  intent={r.intent_windows}  no_intent={r.no_intent_windows}  "
            f"votes(K={r.votes_keep},S={r.votes_swap})  early_swap={r.early_swap_triggered}"
            f"{forced}"
        )

    # Write report
    report_rows: List[Dict[str, Any]] = []
    for r in results:
        report_rows.append(
            {
                "folder": r.folder,
                "path": r.path,
                "verdict": r.verdict,
                "reason": r.reason,
                "forced": r.forced,

                "windows": r.windows,
                "windows_used": r.windows_used,
                "intent_windows": r.intent_windows,
                "no_intent_windows": r.no_intent_windows,
                "votes_keep": r.votes_keep,
                "votes_swap": r.votes_swap,
                "early_swap_triggered": r.early_swap_triggered,
                "early_keep_bonus": r.early_keep_bonus,
                "skipped_no_battle_start": r.skipped_no_battle_start,
                "skipped_missing_bases": r.skipped_missing_bases,
                "skipped_bad_scan": r.skipped_bad_scan,
            }
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    dt = time.time() - t0
    print("")
    print(f"Done. files={len(results)}  swap={n_swap}  keep={n_keep}  unclear={n_unclear}  forced={n_forced}  workers={workers}  dt={dt:.2f}s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
