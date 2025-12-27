# critic/precache.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from critic.dataset import (
    _read_actions_jsonl,
    _read_json,
    _final_hp_delta,
    tensorize_sample,
    _as_int,
)
from viewer.derived_state import compute_derived  # same as dataset path


def _version_tag() -> str:
    """
    Bump this if you change tensorize_sample semantics or derived_state logic.
    (You can also compute a file hash; this simple tag is reliable if you remember to bump.)
    """
    return "hpdelta_v1"


def _hash_cfg(*, stride: int, require_cust_gt0: bool, folder_len: int) -> str:
    s = json.dumps(
        {
            "version": _version_tag(),
            "stride": int(stride),
            "require_cust_gt0": bool(require_cust_gt0),
            "folder_len": int(folder_len),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]


def _scan_replays(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "actions.jsonl").exists():
            out.append(child)
    if not out:
        raise RuntimeError(f"No replays found under {root}")
    return out


def _build_cache_for_replay(
    replay_dir: Path,
    out_dir: Path,
    *,
    stride: int,
    require_cust_gt0: bool,
    folder_len: int,
) -> Optional[Path]:
    actions_path = replay_dir / "actions.jsonl"
    static = _read_json(replay_dir / "static_data.json") or {}
    frames = _read_actions_jsonl(actions_path)
    if not frames:
        return None

    derived = compute_derived(frames, static)
    y = float(_final_hp_delta(frames))

    eligible: List[int] = []
    for fi in range(0, len(frames), max(1, int(stride))):
        if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
            continue
        eligible.append(fi)

    if not eligible:
        return None

    # Build stacked tensors per key
    first = tensorize_sample(frames[eligible[0]], derived[min(eligible[0], len(derived) - 1)], static, folder_len=folder_len)
    keys = list(first.keys())
    stacks: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

    for fi in eligible:
        di = min(max(0, fi), len(derived) - 1)
        x = tensorize_sample(frames[fi], derived[di], static, folder_len=folder_len)
        for k in keys:
            stacks[k].append(x[k])

    packed: Dict[str, torch.Tensor] = {}
    for k in keys:
        packed[k] = torch.stack(stacks[k], dim=0)

    payload = {
        "version": _version_tag(),
        "replay": replay_dir.name,
        "stride": int(stride),
        "require_cust_gt0": bool(require_cust_gt0),
        "folder_len": int(folder_len),
        "eligible_frame_idxs": torch.tensor(eligible, dtype=torch.int32),
        "y_final_hp_delta": torch.tensor([y], dtype=torch.float32),
        "x": packed,
    }

    out_path = out_dir / f"{replay_dir.name}.pt"
    tmp_path = out_dir / f"{replay_dir.name}.pt.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--out_dir", type=str, default="data/cache_hpdelta")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--require_cust_gt0", action="store_true")
    ap.add_argument("--folder_len", type=int, default=30)
    ap.add_argument("--max_replays", type=int, default=0, help="0 = no limit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    base_out = Path(args.out_dir)
    cfg_hash = _hash_cfg(stride=args.stride, require_cust_gt0=args.require_cust_gt0, folder_len=args.folder_len)
    out_dir = base_out / f"{_version_tag()}_{cfg_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)

    replays = _scan_replays(root)
    if args.max_replays and args.max_replays > 0:
        replays = replays[: int(args.max_replays)]

    print(f"precache out_dir={out_dir} replays={len(replays)}", flush=True)

    n_ok = 0
    for r in tqdm(replays, desc="precaching", dynamic_ncols=True):
        out_path = out_dir / f"{r.name}.pt"
        if out_path.exists() and not args.overwrite:
            n_ok += 1
            continue
        built = _build_cache_for_replay(
            r,
            out_dir,
            stride=args.stride,
            require_cust_gt0=bool(args.require_cust_gt0),
            folder_len=args.folder_len,
        )
        if built is not None:
            n_ok += 1

    print(f"done. cached_replays={n_ok}/{len(replays)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
