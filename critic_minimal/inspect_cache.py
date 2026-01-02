# critic_minimal/inspect_cache.py
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


INT_KEYS = [
    "p_emotion_id",
    "e_emotion_id",
    "grid_tile",
    "grid_owner",
    "p_grid_idx",
    "e_grid_idx",
    "rel_pe_idx",
    "player_chip",
]

FLOAT_KEYS = [
    "scalars",
    "action",
]


def _load_manifest(cache_dir: Path) -> Dict[str, Any] | None:
    mp = cache_dir / "manifest.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_pt_files(cache_dir: Path, max_files: int) -> List[Path]:
    files = sorted(cache_dir.glob("*.pt"))
    return files[:max_files]


def _tensor_stats_float(x: torch.Tensor) -> Dict[str, float]:
    t = x.detach().to(torch.float32).flatten().cpu()
    if t.numel() == 0:
        return {"n": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "n": float(t.numel()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def _tensor_int_summary(x: torch.Tensor, topk: int = 12) -> Dict[str, Any]:
    t = x.detach().to(torch.int64).flatten().cpu()
    if t.numel() == 0:
        return {"n": 0, "unique": 0, "top": []}
    vals = t.tolist()
    c = Counter(vals)
    top = c.most_common(topk)
    return {
        "n": int(t.numel()),
        "unique": int(len(c)),
        "min": int(min(vals)),
        "max": int(max(vals)),
        "top": top,
    }


def _action_sparsity(action_last: torch.Tensor) -> Dict[str, Any]:
    a = action_last.detach().to(torch.float32).flatten().cpu()
    if a.numel() == 0:
        return {"n": 0, "zero_frac": 1.0}
    zero = float((a == 0.0).sum().item())
    return {"n": int(a.numel()), "zero_frac": zero / float(a.numel())}


def _extract_last_tokens(payload: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_last: dict of last-token features, flattened per-sample into [N, ...] (still tensors)
      valid:  [N,T] bool
      y:      [N,T] float32
    """
    x = payload["x"]
    valid = payload["valid"]
    y = payload["y"]

    # last token across time dimension
    x_last: Dict[str, torch.Tensor] = {}
    for k, v in x.items():
        # v: [N,T,...] -> [N,...] taking last time step
        x_last[k] = v[:, -1].contiguous()
    return x_last, valid, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect cached critic_minimal_q tensors and last-token distributions.")
    ap.add_argument("--cache_dir", type=str, default="data/cache_minimal_q/h4_t192")
    ap.add_argument("--max_files", type=int, default=12, help="How many .pt files to scan.")
    ap.add_argument("--samples_per_file", type=int, default=128, help="How many sequences per file to sample.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--show_examples", type=int, default=2, help="Print a couple of raw last-token examples.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"cache_dir does not exist: {cache_dir}")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    manifest = _load_manifest(cache_dir)
    if manifest is not None:
        print(f"[manifest] files={manifest.get('num_files')} total_sequences={manifest.get('total_sequences')}")
        ts = (manifest.get("targets_summary") or {})
        if isinstance(ts, dict):
            print(f"[manifest] abs_p90_median={ts.get('abs_p90_median_across_files')} abs_p95_median={ts.get('abs_p95_median_across_files')}")

    files = _pick_pt_files(cache_dir, int(args.max_files))
    if not files:
        raise SystemExit(f"No .pt files found in {cache_dir}")

    # aggregate collectors
    agg_float: Dict[str, List[torch.Tensor]] = {k: [] for k in FLOAT_KEYS}
    agg_int: Dict[str, List[torch.Tensor]] = {k: [] for k in INT_KEYS}

    examples_printed = 0

    for fp in files:
        payload = torch.load(fp, map_location="cpu")
        if not isinstance(payload, dict) or "x" not in payload:
            print(f"[skip] {fp.name}: not a minimal_q payload dict")
            continue

        x_last, valid, y = _extract_last_tokens(payload)
        n = int(next(iter(x_last.values())).shape[0])

        # sample indices
        take = min(int(args.samples_per_file), n)
        idxs = list(range(n))
        random.shuffle(idxs)
        idxs = idxs[:take]

        for k in FLOAT_KEYS:
            if k in x_last:
                agg_float[k].append(x_last[k][idxs].detach().cpu())

        for k in INT_KEYS:
            if k in x_last:
                agg_int[k].append(x_last[k][idxs].detach().cpu())

        # show a few examples
        if examples_printed < int(args.show_examples):
            i = idxs[0]
            print(f"\n[example] file={fp.name} seq_i={i} replay={payload.get('replay')}")
            # scalars + ids + action prefix
            if "scalars" in x_last:
                print(f"  scalars_last={x_last['scalars'][i].tolist()}")
            if "p_emotion_id" in x_last and "e_emotion_id" in x_last:
                print(f"  p_emo={int(x_last['p_emotion_id'][i].item())} e_emo={int(x_last['e_emotion_id'][i].item())}")
            if "p_grid_idx" in x_last and "e_grid_idx" in x_last and "rel_pe_idx" in x_last:
                print(
                    f"  p_grid={int(x_last['p_grid_idx'][i].item())} "
                    f"e_grid={int(x_last['e_grid_idx'][i].item())} "
                    f"rel={int(x_last['rel_pe_idx'][i].item())}"
                )
            if "player_chip" in x_last:
                print(f"  player_chip={int(x_last['player_chip'][i].item())}")
            if "action" in x_last:
                a = x_last["action"][i].to(torch.float32)
                pref = a[: min(24, a.numel())].tolist()
                print(f"  action_last_prefix({len(pref)})={pref} sparsity={_action_sparsity(a)}")
            examples_printed += 1

    def _cat(xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=0) if xs else torch.empty((0,))

    print("\n=== LAST-TOKEN FLOAT FEATURES ===")
    for k in FLOAT_KEYS:
        t = _cat(agg_float[k])
        if t.numel() == 0:
            print(f"{k}: (missing)")
            continue
        stats = _tensor_stats_float(t)
        print(f"{k}: shape={tuple(t.shape)} stats={stats}")
        if k == "action":
            # action sparsity summary over all last tokens
            a = t.to(torch.float32)
            zero = float((a == 0.0).sum().item())
            total = float(a.numel())
            print(f"  action_zero_frac_overall={zero / max(1.0, total):.6f}")

    print("\n=== LAST-TOKEN INT FEATURES ===")
    for k in INT_KEYS:
        t = _cat(agg_int[k])
        if t.numel() == 0:
            print(f"{k}: (missing)")
            continue
        # flatten across batch and feature dims
        summ = _tensor_int_summary(t, topk=12)
        print(f"{k}: shape={tuple(t.shape)} summary={summ}")


if __name__ == "__main__":
    main()
