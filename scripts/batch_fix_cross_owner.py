#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set


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


def find_actions_files(root: Path) -> List[Path]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/dataset")

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
    args = ap.parse_args()

    force_swap = _parse_name_set(args.force_swap)
    force_keep = _parse_name_set(args.force_keep)
    overlap = force_swap & force_keep
    if overlap:
        raise SystemExit(f"Same replay listed in both --force-swap and --force-keep: {sorted(overlap)}")

    root = Path(args.root)
    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    report_rows: List[Dict[str, Any]] = []
    n_swap = 0
    n_keep = 0
    n_unclear = 0
    n_forced = 0

    for actions_path in files:
        replay = actions_path.parent.name

        evidence = analyze_actions_jsonl(
            str(actions_path),
            allow_bad_lines=True,
            max_bad_lines=args.max_bad_lines,
            pre_mode_frames=args.pre_mode_frames,
            pre_battle_scan=args.pre_battle_scan,
            lead_frames=args.lead_frames,
            early_exit_margin=args.early_exit_margin,
            min_votes=args.min_votes,
        )

        verdict, reason = decide_cross_owner_swap(evidence)

        forced = ""
        if replay in force_swap:
            verdict, reason = ("swap", "forced_swap")
            forced = "swap"
            n_forced += 1
        elif replay in force_keep:
            verdict, reason = ("keep", "forced_keep")
            forced = "keep"
            n_forced += 1

        if verdict == "swap":
            n_swap += 1
        elif verdict == "keep":
            n_keep += 1
        else:
            n_unclear += 1

        report_rows.append(
            {
                "folder": replay,
                "path": str(actions_path).replace("\\", "/"),
                "verdict": verdict,
                "reason": reason,
                "forced": forced,

                "windows": int(evidence.get("windows", 0) or 0),
                "windows_used": int(evidence.get("windows_used", 0) or 0),
                "intent_windows": int(evidence.get("intent_windows", 0) or 0),
                "no_intent_windows": int(evidence.get("no_intent_windows", 0) or 0),
                "votes_keep": int(evidence.get("votes_keep", 0) or 0),
                "votes_swap": int(evidence.get("votes_swap", 0) or 0),
                "early_swap_triggered": int(evidence.get("early_swap_triggered", 0) or 0),
                "early_keep_bonus": int(evidence.get("early_keep_bonus", 0) or 0),
                "skipped_no_battle_start": int(evidence.get("skipped_no_battle_start", 0) or 0),
                "skipped_missing_bases": int(evidence.get("skipped_missing_bases", 0) or 0),
                "skipped_bad_scan": int(evidence.get("skipped_bad_scan", 0) or 0),
            }
        )

        print(
            f"[{verdict.upper()}] {replay}  "
            f"used={evidence.get('windows_used',0)}  "
            f"intent={evidence.get('intent_windows',0)}  "
            f"no_intent={evidence.get('no_intent_windows',0)}  "
            f"votes(K={evidence.get('votes_keep',0)},S={evidence.get('votes_swap',0)})  "
            f"early_swap={evidence.get('early_swap_triggered',0)}"
            + (f"  [FORCED:{forced.upper()}]" if forced else "")
        )

        if args.apply and verdict == "swap":
            def transformed() -> Iterator[Dict[str, Any]]:
                for fr in iter_jsonl_tolerant(actions_path, max_bad_lines=args.max_bad_lines):
                    apply_swap_game_emotion(fr)
                    yield fr

            write_jsonl_atomic(actions_path, transformed())

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    print("")
    print(f"Done. files={len(files)}  swap={n_swap}  keep={n_keep}  unclear={n_unclear}  forced={n_forced}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
