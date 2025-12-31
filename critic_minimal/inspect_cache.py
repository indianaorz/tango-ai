# critic_minimal/inspect_cache.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from critic_minimal.features import BUTTON_KEYS

# ----------------------------
# Small utilities
# ----------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _quantiles(arr: np.ndarray, qs: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {}
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(round(q * 100)):02d}"] = float(np.quantile(arr, q))
    return out


def _normalize_target_tanh(y: torch.Tensor, *, norm_factor: float) -> torch.Tensor:
    nf = float(norm_factor)
    if not math.isfinite(nf) or nf <= 1e-6:
        nf = 1.0
    return torch.tanh(y.to(torch.float32) / nf)


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x
        delta = x - self.mean
        self.mean += delta / float(self.n)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def extend(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.add(float(x))

    def to_dict(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {"n": 0}
        var = self.m2 / float(self.n)
        std = math.sqrt(max(0.0, var))
        return {
            "n": int(self.n),
            "min": float(self.min_v),
            "max": float(self.max_v),
            "mean": float(self.mean),
            "std": float(std),
        }


class Reservoir:
    """
    Reservoir sampling for approximate quantiles without loading everything.
    Deterministic given seed + consistent call order.
    """
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = random.Random(int(seed))
        self.n_seen = 0
        self.buf: List[float] = []

    def add_many(self, xs: np.ndarray) -> None:
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        for x in xs:
            self.n_seen += 1
            if len(self.buf) < self.capacity:
                self.buf.append(float(x))
            else:
                j = self.rng.randrange(0, self.n_seen)
                if j < self.capacity:
                    self.buf[j] = float(x)

    def values(self) -> np.ndarray:
        return np.asarray(self.buf, dtype=np.float64)


def _bool_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return x.to(torch.bool)


def _load_pt(path: Path) -> Optional[Dict[str, Any]]:
    try:
        ck = torch.load(path, map_location="cpu")
        return ck if isinstance(ck, dict) else None
    except Exception:
        return None


def _infer_schema(ck: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Returns (seq_len, action_dim, scalar_dim). Raises if missing.
    """
    x = ck.get("x", None)
    y = ck.get("y", None)
    if not isinstance(x, dict) or not torch.is_tensor(y) or y.ndim != 2:
        raise ValueError("missing/invalid x or y")
    T = int(y.shape[1])

    action = x.get("action", None)
    scalars = x.get("scalars", None)
    if not torch.is_tensor(action) or action.ndim != 3:
        raise ValueError("missing/invalid x[action]")
    if not torch.is_tensor(scalars) or scalars.ndim != 3:
        raise ValueError("missing/invalid x[scalars]")

    A = int(action.shape[2])
    S = int(scalars.shape[2])
    return T, A, S


def _print_human_summary(r: Dict[str, Any]) -> None:
    schema = r["schema"]
    q = r["quality"]
    t = r["targets"]
    feats = r["features"]

    print("\n==============================", flush=True)
    print(" critic_minimal cache inspect ", flush=True)
    print("==============================", flush=True)
    print(f"cache_dir: {r['cache_dir']}", flush=True)
    print(f"files_scanned: {r['files_scanned']} | sequences_seen: {q['sequences_seen']}", flush=True)
    print(f"schema: T={schema['seq_len']} action_dim={schema['action_dim']} scalar_dim={schema['scalar_dim']}", flush=True)
    print(f"y_norm_factor: {r['y_norm']['norm_factor']:.3f}", flush=True)

    vr = q["valid_ratio"]
    oz = q["outside_valid_frac_nonzero"]
    print("", flush=True)
    print(
        f"valid_ratio (fraction of T): mean={vr.get('mean', 0):.3f} min={vr.get('min', 0):.3f} max={vr.get('max', 0):.3f}",
        flush=True,
    )
    print(f"outside_valid_frac_nonzero(y): mean={oz.get('mean', 0):.6f} (should be ~0)", flush=True)
    print(f"params_mismatch_files: {q['params_mismatch_count']} | file_errors: {q['file_error_count']}", flush=True)

    yr = t["y_raw"]
    yn = t["y_norm"]
    print("", flush=True)
    print(
        f"y_raw:  mean={yr['stats'].get('mean', 0):.3f} std={yr['stats'].get('std', 0):.3f} "
        f"abs_p90={yr['abs_quantiles'].get('p90', 0):.3f} abs_p95={yr['abs_quantiles'].get('p95', 0):.3f}",
        flush=True,
    )
    print(
        f"y_norm: mean={yn['stats'].get('mean', 0):.3f} std={yn['stats'].get('std', 0):.3f} "
        f"abs_p90={yn['abs_quantiles'].get('p90', 0):.3f} abs_p95={yn['abs_quantiles'].get('p95', 0):.3f}",
        flush=True,
    )

    print("", flush=True)
    print("action press rates (valid frames):", flush=True)
    pr = feats["actions"]["press_rate_over_valid_frames"]
    for k in BUTTON_KEYS:
        print(f"  {k:>14s}: {pr.get(k, 0.0)*100.0:6.2f}%", flush=True)


def inspect_cache(
    cache_dir: Path,
    *,
    max_files: Optional[int],
    max_rows_per_file: int,
    max_valid_values: int,
    seed: int,
    y_norm_factor: float,
    action_press_threshold: float,
    show_progress: bool,
) -> Dict[str, Any]:
    pt_files = sorted([p for p in cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
    if not pt_files:
        raise SystemExit(f"No .pt cache files found in {cache_dir}")

    manifest = _read_json(cache_dir / "manifest.json")

    # shuffle file order deterministically for sampling passes
    rng = random.Random(int(seed))
    rng.shuffle(pt_files)

    if max_files is not None:
        pt_files = pt_files[: int(max_files)]

    # Global trackers
    file_errors: List[Dict[str, Any]] = []
    params_mismatch: List[str] = []
    schema_ref: Optional[Tuple[int, int, int]] = None  # (T,A,S)
    params_ref: Optional[Dict[str, Any]] = None

    # Targets
    y_raw_stats = RunningStats()
    y_norm_stats = RunningStats()
    y_raw_res = Reservoir(capacity=max_valid_values, seed=seed + 1)
    y_norm_res = Reservoir(capacity=max_valid_values, seed=seed + 2)

    # Valid mask / zeros outside valid
    valid_ratio_stats = RunningStats()
    outside_valid_nonzero_stats = RunningStats()  # fraction of |y|>0 outside valid per seq

    done_count = 0
    seq_count = 0

    # Scalars per-dim (we expect 5: p_hp,e_hp,p_charge,e_charge,cust)
    scalar_stats = [RunningStats() for _ in range(5)]
    scalar_res = [Reservoir(capacity=max(10_000, max_valid_values // 8), seed=seed + 10 + i) for i in range(5)]

    # Actions press rates (over valid frames)
    action_press_counts = np.zeros((len(BUTTON_KEYS),), dtype=np.float64)
    action_valid_frames = 0.0

    it = pt_files
    if show_progress:
        it = tqdm(pt_files, desc="Inspecting cache files", dynamic_ncols=True)

    for p in it:
        ck = _load_pt(p)
        if ck is None:
            file_errors.append({"file": p.name, "error": "failed_to_load"})
            continue

        try:
            T, A, S = _infer_schema(ck)
            if schema_ref is None:
                schema_ref = (T, A, S)
            elif (T, A, S) != schema_ref:
                file_errors.append(
                    {
                        "file": p.name,
                        "error": "schema_mismatch",
                        "got": {"seq_len": T, "action_dim": A, "scalar_dim": S},
                        "expected": {"seq_len": schema_ref[0], "action_dim": schema_ref[1], "scalar_dim": schema_ref[2]},
                    }
                )
                continue

            # params compare (best effort)
            params = ck.get("params", None)
            if isinstance(params, dict):
                if params_ref is None:
                    params_ref = params
                elif params != params_ref:
                    params_mismatch.append(p.name)

            x = ck["x"]
            y_raw = ck["y"]  # [N,T]
            valid = _bool_tensor(ck["valid"])  # [N,T]
            done = ck.get("done", None)
            if torch.is_tensor(done) and done.shape == valid.shape:
                done_count += int(done.sum().item())

            N = int(y_raw.shape[0])
            if N <= 0:
                continue

            # pick rows to inspect
            idxs = list(range(N))
            if N > int(max_rows_per_file):
                rng.shuffle(idxs)
                idxs = idxs[: int(max_rows_per_file)]

            scalars = x["scalars"]  # [N,T,5]
            action = x["action"]    # [N,T,A]

            # Discrete tensors (optional but expected in your cache)
            pemo = x.get("p_emotion_id", None)
            eemo = x.get("e_emotion_id", None)
            chip = x.get("player_chip", None)

            for i in idxs:
                seq_count += 1

                v = valid[i]  # [T]
                v_sum = float(v.to(torch.float32).sum().item())
                valid_ratio_stats.add(v_sum / float(T))

                yr = y_raw[i].to(torch.float32)  # [T]

                outside = (~v)
                if int(outside.sum().item()) > 0:
                    frac_nonzero = float((yr[outside].abs() > 1e-6).to(torch.float32).mean().item())
                    outside_valid_nonzero_stats.add(frac_nonzero)

                if v_sum <= 0:
                    continue

                yr_valid = yr[v].detach().cpu().numpy()
                y_raw_stats.extend(yr_valid.tolist())
                y_raw_res.add_many(yr_valid)

                yn_valid = _normalize_target_tanh(torch.from_numpy(yr_valid), norm_factor=y_norm_factor).numpy()
                y_norm_stats.extend(yn_valid.tolist())
                y_norm_res.add_many(yn_valid)

                sc = scalars[i].to(torch.float32)  # [T,5]
                sc_valid = sc[v].detach().cpu().numpy()
                if sc_valid.size:
                    for d in range(min(5, sc_valid.shape[1])):
                        col = sc_valid[:, d]
                        scalar_stats[d].extend(col.tolist())
                        scalar_res[d].add_many(col)

                act = action[i].to(torch.float32)  # [T,A]
                act_valid = act[v].detach().cpu().numpy()
                if act_valid.size:
                    action_valid_frames += float(act_valid.shape[0])
                    pressed = (act_valid > float(action_press_threshold)).astype(np.float64)
                    action_press_counts[: pressed.shape[1]] += pressed.sum(axis=0)

        except Exception as e:
            file_errors.append({"file": p.name, "error": str(e)})
            continue

    if schema_ref is None:
        raise SystemExit("No valid cache files could be parsed.")

    y_raw_vals = y_raw_res.values()
    y_norm_vals = y_norm_res.values()

    scalar_q = []
    scalar_names = ["p_hp/2500", "e_hp/2500", "p_charge/2", "e_charge/2", "cust/100"]
    for d in range(5):
        vals = scalar_res[d].values()
        scalar_q.append(
            {
                "dim": d,
                "name": scalar_names[d],
                "stats": scalar_stats[d].to_dict(),
                "quantiles": _quantiles(vals, [0.01, 0.10, 0.50, 0.90, 0.95, 0.99]),
                "sample_n": int(vals.size),
            }
        )

    action_press_rate: Dict[str, float] = {}
    if action_valid_frames > 0:
        for i, k in enumerate(BUTTON_KEYS):
            action_press_rate[k] = float(action_press_counts[i] / action_valid_frames)

    report: Dict[str, Any] = {
        "version": 1,
        "created_at": _now_iso(),
        "cache_dir": str(cache_dir),
        "files_scanned": int(len(pt_files)),
        "manifest_present": bool(manifest is not None),
        "schema": {"seq_len": int(schema_ref[0]), "action_dim": int(schema_ref[1]), "scalar_dim": int(schema_ref[2])},
        "y_norm": {"norm_factor": float(y_norm_factor), "def": "y_norm = tanh(y_raw / norm_factor)"},
        "quality": {
            "sequences_seen": int(seq_count),
            "done_true_count": int(done_count),
            "valid_ratio": valid_ratio_stats.to_dict(),
            "outside_valid_frac_nonzero": outside_valid_nonzero_stats.to_dict(),
            "params_mismatch_count": int(len(params_mismatch)),
            "params_mismatch_files_sample": params_mismatch[:50],
            "file_error_count": int(len(file_errors)),
            "file_errors_sample": file_errors[:50],
        },
        "targets": {
            "y_raw": {
                "stats": y_raw_stats.to_dict(),
                "quantiles": _quantiles(y_raw_vals, [0.01, 0.10, 0.50, 0.90, 0.95, 0.99]),
                "abs_quantiles": _quantiles(np.abs(y_raw_vals), [0.50, 0.90, 0.95, 0.99]),
                "sample_n": int(y_raw_vals.size),
            },
            "y_norm": {
                "stats": y_norm_stats.to_dict(),
                "quantiles": _quantiles(y_norm_vals, [0.01, 0.10, 0.50, 0.90, 0.95, 0.99]),
                "abs_quantiles": _quantiles(np.abs(y_norm_vals), [0.50, 0.90, 0.95, 0.99]),
                "sample_n": int(y_norm_vals.size),
            },
        },
        "features": {
            "scalars": {
                "names": scalar_names,
                "per_dim": scalar_q,
            },
            "actions": {
                "keys": list(BUTTON_KEYS),
                "press_threshold": float(action_press_threshold),
                "press_rate_over_valid_frames": action_press_rate,
                "valid_frames_counted": float(action_valid_frames),
            },
        },
    }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect critic_minimal cache for schema/target/feature sanity.")
    ap.add_argument("--cache_dir", type=str, default="data/cache_minimal_q/h4_t192")
    ap.add_argument("--out_json", type=str, default="", help="If set, write full report JSON to this path.")

    # sampling / speed
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of .pt files scanned (None = all).")
    ap.add_argument("--max_rows_per_file", type=int, default=256, help="Limit sequences inspected per file.")
    ap.add_argument("--max_valid_values", type=int, default=200_000, help="Reservoir capacity for target/scalar quantiles.")
    ap.add_argument("--seed", type=int, default=1337)

    # normalization check
    ap.add_argument("--y_norm_factor", type=float, default=325.0, help="Used only for y_norm=tanh(y_raw/y_norm_factor) stats.")
    ap.add_argument("--action_press_threshold", type=float, default=0.5, help=">threshold counts as pressed.")

    # UX
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar.")

    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"cache_dir not found: {cache_dir}")

    # PRINT IMMEDIATELY so it never looks “hung”
    pt_files_total = len([p for p in cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
    mf = "ALL" if args.max_files is None else str(args.max_files)
    print(
        f"[inspect_cache] cache_dir={cache_dir} | pt_files={pt_files_total} | scanning={mf} | rows/file<={int(args.max_rows_per_file)}",
        flush=True,
    )

    report = inspect_cache(
        cache_dir,
        max_files=args.max_files,
        max_rows_per_file=int(args.max_rows_per_file),
        max_valid_values=int(args.max_valid_values),
        seed=int(args.seed),
        y_norm_factor=float(args.y_norm_factor),
        action_press_threshold=float(args.action_press_threshold),
        show_progress=(not bool(args.no_progress)),
    )

    _print_human_summary(report)

    out_json = (args.out_json or "").strip()
    if out_json:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote: {outp}", flush=True)

    q = report["quality"]
    if int(q["file_error_count"]) > 0 or int(q["params_mismatch_count"]) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
