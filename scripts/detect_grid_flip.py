from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


GRID_W = 6
GRID_H = 3


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# -----------------------------------------------------------------------------
# Safe conversions
# -----------------------------------------------------------------------------

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _safe_get_pos(fr: Dict[str, Any], key: str) -> Optional[Tuple[float, float]]:
    v = fr.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    x = _as_float(v[0])
    y = _as_float(v[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _is_grid_ok(arr: Any) -> bool:
    return isinstance(arr, (list, tuple)) and len(arr) == GRID_W * GRID_H


# -----------------------------------------------------------------------------
# Grid indexing helpers
# -----------------------------------------------------------------------------

def _nearest_center_index(centers: Sequence[float], x: float) -> int:
    best_i = 0
    best_d = abs(x - centers[0])
    for i in range(1, len(centers)):
        d = abs(x - centers[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _mirror_col(c: int) -> int:
    return (GRID_W - 1) - c


def _mirror_idx(idx: int) -> int:
    r = idx // GRID_W
    c = idx % GRID_W
    return r * GRID_W + _mirror_col(c)


def _cell_index_from_pos(
    x_centers: Sequence[float],
    y_centers: Sequence[float],
    pos: Tuple[float, float],
) -> int:
    col = _nearest_center_index(x_centers, pos[0])
    row = _nearest_center_index(y_centers, pos[1])
    return row * GRID_W + col


# -----------------------------------------------------------------------------
# Deterministic 1D k-means (kept as fallback)
# -----------------------------------------------------------------------------

def _kmeans_1d(samples: Sequence[float], k: int, iters: int = 30) -> Optional[List[float]]:
    xs = [float(x) for x in samples if math.isfinite(float(x))]
    if len(xs) < k:
        return None

    xs.sort()

    centers = []
    for i in range(k):
        q = (i + 0.5) / k
        idx = int(q * (len(xs) - 1))
        centers.append(xs[idx])
    centers.sort()

    for _ in range(iters):
        buckets: List[List[float]] = [[] for _ in range(k)]
        for x in xs:
            ci = _nearest_center_index(centers, x)
            buckets[ci].append(x)

        new_centers = []
        for i in range(k):
            if buckets[i]:
                new_centers.append(sum(buckets[i]) / len(buckets[i]))
            else:
                new_centers.append(centers[i])

        new_centers.sort()
        if all(abs(new_centers[i] - centers[i]) < 1e-6 for i in range(k)):
            centers = new_centers
            break
        centers = new_centers

    return centers


# -----------------------------------------------------------------------------
# More robust deterministic centers: trimmed quantiles + even spacing + refine
# -----------------------------------------------------------------------------

def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    # q in [0,1]
    if not sorted_xs:
        return float("nan")
    if q <= 0.0:
        return float(sorted_xs[0])
    if q >= 1.0:
        return float(sorted_xs[-1])
    idx = q * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_xs[lo])
    t = idx - lo
    return float((1.0 - t) * sorted_xs[lo] + t * sorted_xs[hi])


def _infer_centers_quantile(
    samples: Sequence[float],
    k: int,
    *,
    trim_q: float = 0.02,
    refine_iters: int = 2,
) -> Optional[List[float]]:
    xs = [float(x) for x in samples if math.isfinite(float(x))]
    if len(xs) < k:
        return None
    xs.sort()

    lo = _quantile(xs, trim_q)
    hi = _quantile(xs, 1.0 - trim_q)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None

    step = (hi - lo) / (k - 1)
    centers = [lo + i * step for i in range(k)]

    # Refinement: reassign to nearest and recompute mean; keep deterministic
    for _ in range(refine_iters):
        buckets: List[List[float]] = [[] for _ in range(k)]
        for x in xs:
            ci = _nearest_center_index(centers, x)
            buckets[ci].append(x)
        new_centers: List[float] = []
        for i in range(k):
            if buckets[i]:
                new_centers.append(sum(buckets[i]) / len(buckets[i]))
            else:
                new_centers.append(centers[i])
        new_centers.sort()
        centers = new_centers

    # Sanity: strictly increasing-ish (avoid collapsed centers)
    for i in range(1, k):
        if not (centers[i] > centers[i - 1]):
            # If collapse happened, bail out to allow other method
            return None

    return centers


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Score:
    match_rate: float          # combined (player + enemy) match rate
    used_obs: int              # number of entity-observations used (player+enemy)
    mirror: bool
    swap_owner: bool
    swap_entities: bool

    player_match_rate: float
    enemy_match_rate: float
    used_player: int
    used_enemy: int

    def label(self) -> str:
        m = "flipped" if self.mirror else "normal"
        o = "owner=swap" if self.swap_owner else "owner=0->P,1->E"
        e = "entities=swap" if self.swap_entities else "entities=as-is"
        return f"{m} | {o} | {e}"


def _score_owner_consistency(
    frames: Sequence[Dict[str, Any]],
    x_centers: Sequence[float],
    y_centers: Sequence[float],
    *,
    mirror: bool,
    swap_owner: bool,
    swap_entities: bool,
) -> Score:
    """
    Uses BOTH player_pos and enemy_pos.

    IMPORTANT CHANGE vs old logic:
      - We score player and enemy independently (entity-observations).
      - We do NOT require both to match in the same frame.

    Hypotheses:
      swap_owner=False => player=0, enemy=1
      swap_owner=True  => player=1, enemy=0

      swap_entities=False => player_pos uses player_owner, enemy_pos uses enemy_owner
      swap_entities=True  => player_pos uses enemy_owner, enemy_pos uses player_owner

    mirror=True mirrors the grid lookup indices (column flip).
    """
    player_owner = 1 if swap_owner else 0
    enemy_owner = 0 if swap_owner else 1

    used_p = 0
    used_e = 0
    matched_p = 0
    matched_e = 0

    for fr in frames:
        if _as_bool(fr.get("inside_window"), False):
            continue

        owner = fr.get("grid_owner_state")
        if not _is_grid_ok(owner):
            continue

        ppos_raw = _safe_get_pos(fr, "player_pos")
        epos_raw = _safe_get_pos(fr, "enemy_pos")
        if ppos_raw is None or epos_raw is None:
            continue

        if not swap_entities:
            ppos = ppos_raw
            epos = epos_raw
            expected_p = player_owner
            expected_e = enemy_owner
        else:
            ppos = ppos_raw
            epos = epos_raw
            expected_p = enemy_owner
            expected_e = player_owner

        # Player observation
        pidx = _cell_index_from_pos(x_centers, y_centers, ppos)
        if mirror:
            pidx = _mirror_idx(pidx)
        op = _as_int(owner[pidx], -1)
        if op in (0, 1):  # only score binary owner states; ignore unknown encodings
            used_p += 1
            if op == expected_p:
                matched_p += 1

        # Enemy observation
        eidx = _cell_index_from_pos(x_centers, y_centers, epos)
        if mirror:
            eidx = _mirror_idx(eidx)
        oe = _as_int(owner[eidx], -1)
        if oe in (0, 1):  # only score binary owner states; ignore unknown encodings
            used_e += 1
            if oe == expected_e:
                matched_e += 1

    used_obs = used_p + used_e
    matched_obs = matched_p + matched_e
    combined = (matched_obs / used_obs) if used_obs > 0 else 0.0

    p_rate = (matched_p / used_p) if used_p > 0 else 0.0
    e_rate = (matched_e / used_e) if used_e > 0 else 0.0

    return Score(
        match_rate=combined,
        used_obs=used_obs,
        mirror=mirror,
        swap_owner=swap_owner,
        swap_entities=swap_entities,
        player_match_rate=p_rate,
        enemy_match_rate=e_rate,
        used_player=used_p,
        used_enemy=used_e,
    )


def _evaluate_all_hypotheses(
    frames: Sequence[Dict[str, Any]],
    x_centers: Sequence[float],
    y_centers: Sequence[float],
) -> List[Score]:
    scores: List[Score] = []
    for mirror in (False, True):
        for swap_owner in (False, True):
            for swap_entities in (False, True):
                scores.append(
                    _score_owner_consistency(
                        frames,
                        x_centers,
                        y_centers,
                        mirror=mirror,
                        swap_owner=swap_owner,
                        swap_entities=swap_entities,
                    )
                )
    return scores


def analyze_flip_owner_based(
    frames: Sequence[Dict[str, Any]],
    *,
    margin: float = 0.03,
    min_obs: int = 1000,
    center_method: str = "auto",  # auto|quantile|kmeans
) -> Tuple[Optional[List[float]], Optional[List[float]], Score, Score, List[Score], str, str]:
    """
    Evaluate all combinations of:
      - mirror (False/True)
      - swap_owner (False/True)
      - swap_entities (False/True)

    IMPORTANT CHANGE vs old version:
      - Use entity-observations (player+enemy) instead of requiring both match in same frame.
      - Infer centers using quantile method (more stable) and/or kmeans; auto picks best.

    Returns:
      x_centers, y_centers,
      best_normal, best_flipped,
      all_scores,
      decision,
      chosen_center_method
    """
    xs: List[float] = []
    ys: List[float] = []

    for fr in frames:
        if _as_bool(fr.get("inside_window"), False):
            continue
        ppos = _safe_get_pos(fr, "player_pos")
        epos = _safe_get_pos(fr, "enemy_pos")
        if ppos is None or epos is None:
            continue
        xs.extend([ppos[0], epos[0]])
        ys.extend([ppos[1], epos[1]])

    def infer(method: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if method == "quantile":
            return (
                _infer_centers_quantile(xs, GRID_W),
                _infer_centers_quantile(ys, GRID_H),
            )
        if method == "kmeans":
            return (
                _kmeans_1d(xs, GRID_W),
                _kmeans_1d(ys, GRID_H),
            )
        raise ValueError(f"unknown center_method: {method}")

    # Choose centers
    candidates: List[Tuple[str, Optional[List[float]], Optional[List[float]], List[Score]]] = []

    if center_method in ("auto", "quantile"):
        xc, yc = infer("quantile")
        if xc is not None and yc is not None:
            candidates.append(("quantile", xc, yc, _evaluate_all_hypotheses(frames, xc, yc)))

    if center_method in ("auto", "kmeans"):
        xc, yc = infer("kmeans")
        if xc is not None and yc is not None:
            candidates.append(("kmeans", xc, yc, _evaluate_all_hypotheses(frames, xc, yc)))

    if not candidates:
        dummy = Score(0.0, 0, False, False, False, 0.0, 0.0, 0, 0)
        return None, None, dummy, dummy, [], "unknown (could not infer centers)", "none"

    # Pick the center method that yields the best overall hypothesis score
    def best_overall(scores: List[Score]) -> float:
        return max(scores, key=lambda s: s.match_rate).match_rate if scores else 0.0

    chosen_method, x_centers, y_centers, all_scores = max(
        candidates, key=lambda t: best_overall(t[3])
    )

    best_normal = max((s for s in all_scores if not s.mirror), key=lambda s: s.match_rate)
    best_flipped = max((s for s in all_scores if s.mirror), key=lambda s: s.match_rate)

    usable = max(best_normal.used_obs, best_flipped.used_obs)
    if usable < min_obs:
        decision = f"unknown (only {usable} entity-observations; need >= {min_obs})"
        return x_centers, y_centers, best_normal, best_flipped, all_scores, decision, chosen_method

    if best_flipped.match_rate - best_normal.match_rate >= margin:
        decision = "flip"
    elif best_normal.match_rate - best_flipped.match_rate >= margin:
        decision = "normal"
    else:
        decision = "unclear"

    return x_centers, y_centers, best_normal, best_flipped, all_scores, decision, chosen_method


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("actions_jsonl", type=str)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--min-obs", type=int, default=1000, help="minimum player+enemy observations (not frames)")
    ap.add_argument("--center-method", type=str, default="auto", choices=["auto", "quantile", "kmeans"])
    ap.add_argument("--show-all", action="store_true", help="print all hypothesis scores")
    args = ap.parse_args()

    path = Path(args.actions_jsonl)
    frames = read_jsonl(path)

    x_centers, y_centers, best_normal, best_flipped, all_scores, decision, chosen_method = analyze_flip_owner_based(
        frames,
        margin=args.margin,
        min_obs=args.min_obs,
        center_method=args.center_method,
    )

    print(f"File: {path}")
    print(f"Total frames: {len(frames)}")
    print(f"Center method: {chosen_method}")
    print(f"Usable entity-observations (best): {max(best_normal.used_obs, best_flipped.used_obs)}")
    print("")

    print(
        f"BEST normal : {best_normal.match_rate:.3f}   ({best_normal.label()})"
        f"   [P {best_normal.player_match_rate:.3f} ({best_normal.used_player}),"
        f" E {best_normal.enemy_match_rate:.3f} ({best_normal.used_enemy})]"
    )
    print(
        f"BEST flipped: {best_flipped.match_rate:.3f}   ({best_flipped.label()})"
        f"   [P {best_flipped.player_match_rate:.3f} ({best_flipped.used_player}),"
        f" E {best_flipped.enemy_match_rate:.3f} ({best_flipped.used_enemy})]"
    )
    print(f"Decision: {decision}")

    if x_centers:
        print("x centers (cols): " + ", ".join(f"{x:.2f}" for x in x_centers))
    else:
        print("x centers (cols): <none>")
    if y_centers:
        print("y centers (rows): " + ", ".join(f"{y:.2f}" for y in y_centers))
    else:
        print("y centers (rows): <none>")

    if args.show_all:
        print("\nAll hypothesis scores (sorted):")
        for s in sorted(all_scores, key=lambda t: t.match_rate, reverse=True):
            print(
                f"  {s.match_rate:.3f}  used_obs={s.used_obs:6d}"
                f"  P={s.player_match_rate:.3f}({s.used_player:5d})"
                f"  E={s.enemy_match_rate:.3f}({s.used_enemy:5d})"
                f"  {s.label()}"
            )


if __name__ == "__main__":
    main()
