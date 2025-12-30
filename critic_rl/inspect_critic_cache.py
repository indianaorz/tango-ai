#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# -----------------------------
# Robust parsing (match dataset.py spirit)
# -----------------------------
def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return frames


def _hp_delta_series(frames: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    p = 0
    e = 0
    for fr in frames:
        if "player_health" in fr:
            p = _as_int(fr.get("player_health"), p)
        if "enemy_health" in fr:
            e = _as_int(fr.get("enemy_health"), e)
        out.append(int(p - e))
    return out


def _sampled_transition_reward(hp_delta: List[int], t: int, t_next: int) -> float:
    """
    Stride-aware reward between sampled frames:
      r = (p_hp - e_hp)[t_next] - (p_hp - e_hp)[t]
    """
    if not hp_delta:
        return 0.0
    n = len(hp_delta)
    t = max(0, min(int(t), n - 1))
    t_next = max(0, min(int(t_next), n - 1))
    return float(hp_delta[t_next] - hp_delta[t])


def _torch_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _quantiles(x: torch.Tensor, qs: List[float]) -> Dict[float, float]:
    x = x.detach().flatten()
    if x.numel() == 0:
        return {q: float("nan") for q in qs}
    x = x.float()
    # torch.quantile exists in modern torch; fallback if needed
    try:
        out = torch.quantile(x, torch.tensor(qs, device=x.device))
        return {qs[i]: float(out[i].item()) for i in range(len(qs))}
    except Exception:
        # fallback: sort on CPU (slower but fine for inspector)
        xs = x.cpu().numpy()
        xs.sort()
        n = xs.shape[0]
        res: Dict[float, float] = {}
        for q in qs:
            idx = int(round(q * (n - 1)))
            res[q] = float(xs[idx])
        return res


# -----------------------------
# Reporting
# -----------------------------
@dataclass
class Issue:
    kind: str
    file: str
    detail: str


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _short(p: Path) -> str:
    return p.name


# -----------------------------
# Cache invariant checks
# -----------------------------
def _check_cache_file_struct(
    ck: Dict[str, Any],
    path: Path,
    issues: List[Issue],
    *,
    expected_params: Optional[Dict[str, Any]],
    expected_seq_len: Optional[int],
) -> None:
    name = _short(path)

    # required keys
    for k in ["replay", "params", "start_frame", "r", "done", "valid", "x"]:
        if k not in ck:
            issues.append(Issue("missing_key", name, f"Missing key: {k}"))
            return

    if expected_params is not None and ck.get("params") != expected_params:
        issues.append(Issue("params_mismatch", name, "ck['params'] != manifest/expected params"))

    r = ck["r"]
    done = ck["done"]
    valid = ck["valid"]
    start_frame = ck["start_frame"]
    x = ck["x"]

    if not isinstance(x, dict):
        issues.append(Issue("bad_type", name, f"x is {type(x)} not dict"))
        return

    # basic tensor checks
    for tname, t in [("r", r), ("done", done), ("valid", valid), ("start_frame", start_frame)]:
        if not isinstance(t, torch.Tensor):
            issues.append(Issue("bad_type", name, f"{tname} is {type(t)} not torch.Tensor"))
            return

    if r.ndim != 2:
        issues.append(Issue("bad_shape", name, f"r.ndim={r.ndim}, expected 2"))
        return
    N, T = int(r.shape[0]), int(r.shape[1])

    if expected_seq_len is not None and T != expected_seq_len:
        issues.append(Issue("seq_len_mismatch", name, f"r.shape[1]={T} != expected_seq_len={expected_seq_len}"))

    for tname, t in [("done", done), ("valid", valid)]:
        if t.shape != r.shape:
            issues.append(Issue("bad_shape", name, f"{tname}.shape={tuple(t.shape)} != r.shape={tuple(r.shape)}"))

    if start_frame.ndim != 1:
        issues.append(Issue("bad_shape", name, f"start_frame.ndim={start_frame.ndim}, expected 1"))
    else:
        if int(start_frame.shape[0]) != N:
            issues.append(Issue("bad_shape", name, f"start_frame.shape[0]={int(start_frame.shape[0])} != N={N}"))

    # dtypes
    if r.dtype not in (torch.float16, torch.float32, torch.float64):
        issues.append(Issue("bad_dtype", name, f"r.dtype={r.dtype} expected float"))
    if done.dtype != torch.bool:
        issues.append(Issue("bad_dtype", name, f"done.dtype={done.dtype} expected bool"))
    if valid.dtype != torch.bool:
        issues.append(Issue("bad_dtype", name, f"valid.dtype={valid.dtype} expected bool"))

    # finite
    if not _torch_finite(r):
        issues.append(Issue("non_finite", name, "r contains NaN/Inf"))

    # valid mask sanity: must be prefix-true then false (no gaps)
    # and last valid timestep should typically be true (unless N=0)
    if N > 0:
        v = valid
        # detect gaps: once false, should never become true later
        # Compute for each row: find first false; ensure no true after it.
        vf = v.float()
        # cumulative min along time: once 0 appears, cummin stays 0
        try:
            cummin = torch.cummin(vf, dim=1).values
            # positions where cummin==0 but v==1 means a gap
            gaps = ((cummin == 0.0) & (vf == 1.0)).any(dim=1)
            if bool(gaps.any().item()):
                issues.append(Issue("valid_gaps", name, f"{int(gaps.sum().item())} rows have non-prefix valid mask"))
        except Exception:
            # fallback slower
            gaps_count = 0
            for i in range(N):
                row = v[i].tolist()
                seen_false = False
                bad = False
                for b in row:
                    if not b:
                        seen_false = True
                    elif seen_false and b:
                        bad = True
                        break
                if bad:
                    gaps_count += 1
            if gaps_count:
                issues.append(Issue("valid_gaps", name, f"{gaps_count} rows have non-prefix valid mask"))

    # x towers shape agreement
    for k, v in x.items():
        if not isinstance(v, torch.Tensor):
            issues.append(Issue("bad_type", name, f"x['{k}'] is {type(v)} not torch.Tensor"))
            continue
        if v.ndim < 2:
            issues.append(Issue("bad_shape", name, f"x['{k}'].ndim={v.ndim} expected >=2 (N,T,...)"))
            continue
        if int(v.shape[0]) != N or int(v.shape[1]) != T:
            issues.append(
                Issue(
                    "bad_shape",
                    name,
                    f"x['{k}'].shape[:2]={tuple(v.shape[:2])} != (N,T)=({N},{T})",
                )
            )


def _reward_stats(r: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    if r.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "abs_mean": float("nan"),
            "sparsity": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    mask = valid
    if mask.dtype != torch.bool:
        mask = mask.bool()

    rv = r[mask]
    if rv.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "sparsity": 1.0,
            "min": 0.0,
            "max": 0.0,
        }

    rv = rv.float()
    mean = float(rv.mean().item())
    std = float(rv.std(unbiased=False).item())
    abs_mean = float(rv.abs().mean().item())
    # define “zero” as exactly 0.0 in cache; you can widen this if you want eps
    sparsity = float((rv == 0.0).float().mean().item())
    mn = float(rv.min().item())
    mx = float(rv.max().item())
    return {
        "mean": mean,
        "std": std,
        "abs_mean": abs_mean,
        "sparsity": sparsity,
        "min": mn,
        "max": mx,
    }


def _check_reward_and_done_against_raw(
    ck: Dict[str, Any],
    cache_path: Path,
    dataset_dir: Path,
    issues: List[Issue],
    *,
    stride: int,
    seq_len: int,
    sample_rows_limit: int,
    rng: random.Random,
) -> None:
    """
    Recompute stride-aware rewards and done flags from raw replay to detect cache bugs.
    """
    name = _short(cache_path)
    replay = str(ck.get("replay") or cache_path.stem)

    replay_dir = dataset_dir / replay
    actions_path = replay_dir / "actions.jsonl"
    if not actions_path.exists():
        issues.append(Issue("missing_raw", name, f"Raw actions.jsonl not found at {actions_path}"))
        return

    frames = _read_jsonl(actions_path)
    if not frames:
        issues.append(Issue("empty_raw", name, "Raw actions.jsonl empty/unreadable"))
        return

    hp_delta = _hp_delta_series(frames)
    n_frames = len(frames)

    start_frame: torch.Tensor = ck["start_frame"].to(torch.int64).cpu()
    r: torch.Tensor = ck["r"].cpu()
    done: torch.Tensor = ck["done"].cpu()
    valid: torch.Tensor = ck["valid"].cpu()

    N = int(r.shape[0])
    if N == 0:
        return

    # choose rows to check
    rows = list(range(N))
    rng.shuffle(rows)
    rows = rows[: max(1, min(sample_rows_limit, N))]

    for row_i in rows:
        sf = int(start_frame[row_i].item())
        # Validate start_frame in bounds
        if sf < 0 or sf >= n_frames:
            issues.append(Issue("start_oob", name, f"row={row_i} start_frame={sf} out of bounds (n_frames={n_frames})"))
            continue

        # Determine how many valid steps
        vrow = valid[row_i].tolist()
        # valid is expected prefix-true then false; take prefix length
        vlen = 0
        for b in vrow:
            if b:
                vlen += 1
            else:
                break

        if vlen == 0:
            continue

        # Recompute expected rewards/done for each valid timestep
        # timestep k samples raw_i = sf + k*stride
        for k in range(vlen):
            raw_i = sf + k * stride
            if raw_i >= n_frames:
                # cache valid says it exists but raw_i doesn't -> inconsistent
                issues.append(Issue("valid_inconsistent", name, f"row={row_i} k={k} raw_i={raw_i} >= n_frames"))
                break

            raw_next = raw_i + stride
            if raw_next >= n_frames:
                raw_next = n_frames - 1

            exp_r = _sampled_transition_reward(hp_delta, raw_i, raw_next)
            got_r = float(r[row_i, k].item())

            # rewards should match exactly if cache used same definition; allow tiny float noise
            if not math.isfinite(got_r):
                issues.append(Issue("non_finite", name, f"row={row_i} k={k} cached reward non-finite"))
            else:
                if abs(got_r - exp_r) > 1e-4:
                    issues.append(
                        Issue(
                            "reward_mismatch",
                            name,
                            f"row={row_i} k={k} start={sf} raw_i={raw_i} exp_r={exp_r:.3f} got_r={got_r:.3f}",
                        )
                    )
                    # if we see mismatches, no need to spam every k
                    break

            # done should be True only if raw_i is last frame
            exp_done = bool(raw_i >= n_frames - 1)
            got_done = bool(done[row_i, k].item())
            if got_done != exp_done:
                issues.append(
                    Issue(
                        "done_mismatch",
                        name,
                        f"row={row_i} k={k} raw_i={raw_i} exp_done={exp_done} got_done={got_done}",
                    )
                )
                break

        # Also: after vlen, cached reward should be 0 and done False (padding region)
        for k in range(vlen, seq_len):
            if bool(valid[row_i, k].item()):
                issues.append(Issue("valid_gaps", name, f"row={row_i} valid True after prefix at k={k}"))
                break
            if abs(float(r[row_i, k].item())) > 1e-6:
                issues.append(Issue("pad_reward_nonzero", name, f"row={row_i} k={k} padded reward is nonzero"))
                break
            if bool(done[row_i, k].item()):
                issues.append(Issue("pad_done_true", name, f"row={row_i} k={k} padded done is True"))
                break


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect critic RL cache tensors for consistency and sanity.")
    ap.add_argument("--dataset_dir", type=str, default="data/dataset", help="Root dir containing replay folders.")
    ap.add_argument("--cache_dir", type=str, required=True, help="Cache dir containing *.pt and _manifest.pt")
    ap.add_argument("--max_files", type=int, default=50, help="Max cache files to inspect (random sample).")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for file/row sampling.")
    ap.add_argument("--check_raw", action="store_true", help="Verify cached rewards/done against raw actions.jsonl.")
    ap.add_argument("--rows_per_file", type=int, default=16, help="Rows to verify per file when --check_raw.")
    ap.add_argument("--expect_stride", type=int, default=None, help="Assert stride equals this value.")
    ap.add_argument("--expect_seq_len", type=int, default=None, help="Assert seq_len equals this value.")
    ap.add_argument("--fail_on_issue", action="store_true", help="Exit non-zero if any issue is found.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"ERROR: cache_dir not found: {cache_dir}", file=sys.stderr)
        return 2

    manifest_path = cache_dir / "_manifest.pt"
    expected_params: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        try:
            mf = torch.load(manifest_path, map_location="cpu")
            if isinstance(mf, dict) and isinstance(mf.get("params"), dict):
                expected_params = mf["params"]
        except Exception:
            pass

    # optional param assertions
    if expected_params is not None:
        if args.expect_stride is not None:
            if int(expected_params.get("stride", -999)) != int(args.expect_stride):
                print(f"WARNING: manifest stride={expected_params.get('stride')} != --expect_stride={args.expect_stride}")
        if args.expect_seq_len is not None:
            if int(expected_params.get("seq_len", -999)) != int(args.expect_seq_len):
                print(f"WARNING: manifest seq_len={expected_params.get('seq_len')} != --expect_seq_len={args.expect_seq_len}")

    files = sorted([p for p in cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
    if not files:
        print(f"ERROR: no cache files found in {cache_dir}", file=sys.stderr)
        return 2

    rng = random.Random(int(args.seed))
    rng.shuffle(files)
    files = files[: max(1, min(int(args.max_files), len(files)))]

    issues: List[Issue] = []
    total_rows = 0
    total_valid_steps = 0

    # aggregate reward stats
    agg_r_all: List[torch.Tensor] = []
    agg_v_all: List[torch.Tensor] = []

    print(f"Inspecting {len(files)} cache files from {cache_dir} ...")
    if expected_params is not None:
        print(f"Manifest format: {expected_params.get('format')}  stride={expected_params.get('stride')}  seq_len={expected_params.get('seq_len')}")

    for p in files:
        try:
            ck = torch.load(p, map_location="cpu")
        except Exception as e:
            issues.append(Issue("load_fail", _short(p), f"torch.load failed: {e}"))
            continue

        if not isinstance(ck, dict):
            issues.append(Issue("bad_type", _short(p), f"top-level is {type(ck)} not dict"))
            continue

        exp_seq_len = int(args.expect_seq_len) if args.expect_seq_len is not None else (int(expected_params["seq_len"]) if expected_params and "seq_len" in expected_params else None)
        _check_cache_file_struct(ck, p, issues, expected_params=expected_params, expected_seq_len=exp_seq_len)

        # reward stats aggregation
        if isinstance(ck.get("r"), torch.Tensor) and isinstance(ck.get("valid"), torch.Tensor):
            r = ck["r"]
            v = ck["valid"]
            if r.ndim == 2 and v.shape == r.shape:
                total_rows += int(r.shape[0])
                total_valid_steps += int(v.sum().item())
                agg_r_all.append(r.float().flatten())
                agg_v_all.append(v.flatten())

        # raw verification (optional)
        if args.check_raw:
            # determine stride/seq_len to use for expected recompute
            if args.expect_stride is not None:
                stride = int(args.expect_stride)
            elif expected_params and "stride" in expected_params:
                stride = int(expected_params["stride"])
            else:
                # fall back: guess from ck params
                stride = int(ck.get("params", {}).get("stride", 1)) if isinstance(ck.get("params"), dict) else 1

            if exp_seq_len is not None:
                seq_len = int(exp_seq_len)
            else:
                seq_len = int(ck.get("params", {}).get("seq_len", ck["r"].shape[1])) if isinstance(ck.get("params"), dict) else int(ck["r"].shape[1])

            _check_reward_and_done_against_raw(
                ck,
                p,
                dataset_dir,
                issues,
                stride=stride,
                seq_len=seq_len,
                sample_rows_limit=int(args.rows_per_file),
                rng=rng,
            )

    print()
    print("=== Summary ===")
    print(f"Files checked:       {len(files)}")
    print(f"Total rows (N):      {total_rows}")
    print(f"Total valid steps:   {total_valid_steps}")

    if agg_r_all and agg_v_all:
        r_all = torch.cat(agg_r_all, dim=0)
        v_all = torch.cat(agg_v_all, dim=0).bool()
        rv = r_all[v_all]
        stats = _reward_stats(rv.unsqueeze(0), torch.ones_like(rv.unsqueeze(0), dtype=torch.bool))  # reuse helper
        qs = _quantiles(rv, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
        print("Reward stats over VALID steps:")
        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  abs_mean={stats['abs_mean']:.4f}")
        print(f"  sparsity(==0)={_fmt_pct(stats['sparsity'])}  min={stats['min']:.3f}  max={stats['max']:.3f}")
        print("  quantiles:", "  ".join([f"q{int(k*100):02d}={v:.3f}" for k, v in qs.items()]))

    print()
    if issues:
        print(f"=== Issues ({len(issues)}) ===")
        # group by kind
        by_kind: Dict[str, List[Issue]] = {}
        for it in issues:
            by_kind.setdefault(it.kind, []).append(it)
        for kind in sorted(by_kind.keys()):
            group = by_kind[kind]
            print(f"- {kind}: {len(group)}")
            # print a few examples
            for ex in group[:10]:
                print(f"    {ex.file}: {ex.detail}")
            if len(group) > 10:
                print(f"    ... {len(group) - 10} more")
    else:
        print("No issues found ✅")

    if issues and args.fail_on_issue:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
