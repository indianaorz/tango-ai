from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import importlib.util
import sys

# Robustly import scripts/detect_grid_flip.py even when scripts/ is not a package.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_DETECT_PATH = _SCRIPTS_DIR / "detect_grid_flip.py"

_spec = importlib.util.spec_from_file_location("detect_grid_flip", str(_DETECT_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load detect_grid_flip from {_DETECT_PATH}")

_detect = importlib.util.module_from_spec(_spec)
sys.modules["detect_grid_flip"] = _detect
_spec.loader.exec_module(_detect)

GRID_W = int(_detect.GRID_W)
GRID_H = int(_detect.GRID_H)
analyze_flip_owner_based = _detect.analyze_flip_owner_based



# -----------------------------
# JSONL streaming helpers
# -----------------------------

def iter_jsonl(path: Path, *, allow_bad_lines: bool = True, max_bad_lines: int = 50) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL frames. If allow_bad_lines is True, malformed lines are skipped.
    Stops if too many bad lines are encountered (to avoid silently processing garbage files).
    """
    bad = 0
    with path.open("r", encoding="utf-8") as f:
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
                yield obj
            else:
                # Non-dict JSON lines are unexpected; treat as bad.
                bad += 1
                if bad <= 5:
                    print(f"[WARN] {path} non-dict json at line {line_no} (skipping)")
                if bad >= max_bad_lines:
                    raise RuntimeError(f"Too many bad json lines in {path} (>= {max_bad_lines}). Aborting.")


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for obj in rows:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


# -----------------------------
# Grid transform primitives
# -----------------------------

def _is_grid_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) == GRID_W * GRID_H


def _mirror_idx(idx: int) -> int:
    r = idx // GRID_W
    c = idx % GRID_W
    mc = (GRID_W - 1) - c
    return r * GRID_W + mc


def mirror_grid_list(arr: List[Any]) -> List[Any]:
    # Mirror within each row (column flip)
    out = [None] * len(arr)
    for i in range(len(arr)):
        out[_mirror_idx(i)] = arr[i]
    return out


def swap_owner_bits_in_grid_owner_state(arr: List[Any]) -> List[Any]:
    # Only for grid_owner_state: swap 0<->1, leave others untouched
    out: List[Any] = []
    for x in arr:
        if x == 0:
            out.append(1)
        elif x == 1:
            out.append(0)
        else:
            out.append(x)
    return out


# -----------------------------
# Entity swap helpers
# -----------------------------

def swap_player_enemy_keys(frame: Dict[str, Any]) -> None:
    """
    Swap any 'player_*' <-> 'enemy_*' pairs if both exist.
    Also swap 'player' <-> 'enemy' if both exist.
    """
    # Swap dict-level "player"/"enemy" if present
    if "player" in frame and "enemy" in frame:
        frame["player"], frame["enemy"] = frame["enemy"], frame["player"]

    # Collect suffixes for paired keys
    to_swap: List[Tuple[str, str]] = []
    for k in list(frame.keys()):
        if k.startswith("player_"):
            suffix = k[len("player_") :]
            ek = "enemy_" + suffix
            if ek in frame:
                to_swap.append((k, ek))

    # Perform swaps (avoid double-swapping)
    swapped = set()
    for pk, ek in to_swap:
        if pk in swapped or ek in swapped:
            continue
        frame[pk], frame[ek] = frame[ek], frame[pk]
        swapped.add(pk)
        swapped.add(ek)


# -----------------------------
# Apply fix according to best hypothesis
# -----------------------------

@dataclass(frozen=True)
class FixPlan:
    mirror: bool
    swap_owner: bool
    swap_entities: bool


def apply_fix_in_place(frame: Dict[str, Any], plan: FixPlan) -> None:
    # 1) Swap entities first (so positions/fields align to expected roles)
    if plan.swap_entities:
        swap_player_enemy_keys(frame)

    # 2) Mirror grid arrays (any grid_* list of correct length)
    if plan.mirror:
        for k, v in list(frame.items()):
            if k.startswith("grid_") and _is_grid_list(v):
                frame[k] = mirror_grid_list(v)

    # 3) Swap owner labels in owner-state grid ONLY (canonicalize player=0, enemy=1)
    if plan.swap_owner:
        v = frame.get("grid_owner_state")
        if _is_grid_list(v):
            frame["grid_owner_state"] = swap_owner_bits_in_grid_owner_state(v)


# -----------------------------
# Batch driver
# -----------------------------

def find_actions_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        p = child / "actions.jsonl"
        if p.is_file():
            out.append(p)
    return out


def load_for_detection(path: Path, *, max_frames: int) -> List[Dict[str, Any]]:
    """
    Detection needs a list of frames. Keep it bounded for huge files.
    """
    frames: List[Dict[str, Any]] = []
    for i, fr in enumerate(iter_jsonl(path, allow_bad_lines=True)):
        frames.append(fr)
        if len(frames) >= max_frames:
            break
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/dataset", help="dataset root containing run folders")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--min-obs", type=int, default=1000)
    ap.add_argument("--center-method", type=str, default="auto", choices=["auto", "quantile", "kmeans"])
    ap.add_argument("--max-frames", type=int, default=250_000, help="cap frames loaded for detection per file")
    ap.add_argument("--apply", action="store_true", help="actually rewrite actions.jsonl (otherwise dry-run)")
    ap.add_argument("--report", type=str, default="grid_flip_report.csv")
    args = ap.parse_args()

    root = Path(args.root)
    files = find_actions_files(root)
    if not files:
        raise SystemExit(f"No actions.jsonl found under {root}")

    report_path = Path(args.report)

    rows: List[Dict[str, Any]] = []
    n_flip = 0
    n_unclear = 0
    n_normal = 0

    for actions_path in files:
        frames = load_for_detection(actions_path, max_frames=args.max_frames)

        x_centers, y_centers, best_normal, best_flipped, all_scores, decision, chosen_method = analyze_flip_owner_based(
            frames,
            margin=args.margin,
            min_obs=args.min_obs,
            center_method=args.center_method,
        )

        # Decide plan:
        # - If decision == "flip": apply *best_flipped* transforms to canonicalize to normal.
        # - Otherwise: no rewrite.
        plan = FixPlan(
            mirror=best_flipped.mirror,
            swap_owner=best_flipped.swap_owner,
            swap_entities=best_flipped.swap_entities,
        )

        if decision == "flip":
            n_flip += 1
        elif decision == "normal":
            n_normal += 1
        else:
            n_unclear += 1

        rows.append(
            {
                "folder": actions_path.parent.name,
                "path": str(actions_path).replace("\\", "/"),
                "decision": decision,
                "center_method": chosen_method,
                "best_normal_rate": f"{best_normal.match_rate:.6f}",
                "best_flipped_rate": f"{best_flipped.match_rate:.6f}",
                "best_normal_label": best_normal.label(),
                "best_flipped_label": best_flipped.label(),
                "apply_mirror": str(plan.mirror),
                "apply_swap_owner": str(plan.swap_owner),
                "apply_swap_entities": str(plan.swap_entities),
            }
        )

        if args.apply and decision == "flip":
            # Stream read + transform + atomic replace
            def transformed() -> Iterator[Dict[str, Any]]:
                for fr in iter_jsonl(actions_path, allow_bad_lines=True):
                    apply_fix_in_place(fr, plan)
                    yield fr

            write_jsonl_atomic(actions_path, transformed())

            print(f"[FLIP] {actions_path.parent.name}  ({best_flipped.match_rate:.3f} vs {best_normal.match_rate:.3f})")
        else:
            print(f"[SKIP:{decision.upper()}] {actions_path.parent.name}  ({best_flipped.match_rate:.3f} vs {best_normal.match_rate:.3f})")

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("")
    print(f"Done. files={len(files)}  flip={n_flip}  normal={n_normal}  unclear={n_unclear}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
