# critic_minimal/precache.py
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from critic_minimal.features import (
    ACTION_DIM,
    BUTTON_KEYS,
    _as_int,
    _as_list,
    _chip_id_norm,
    _owner_norm,
    _pos_norm,
    _tile_norm,
    aggregate_action,
    bellman_hp_return_sampled,
    compute_done_from_valid,
    pos_to_grid_idx,
    rel_pe_index,
)

# Optional fast JSON parsing
try:
    import orjson  # type: ignore

    _json_loads = orjson.loads
except Exception:
    _json_loads = None


MANIFEST_NAME = "manifest.json"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(str(tmp_path), str(path))
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


def _meta_path_for(pt_path: Path) -> Path:
    return pt_path.with_suffix(".meta.json")


def _write_meta(pt_path: Path, meta: Dict[str, Any]) -> None:
    _atomic_write_text(_meta_path_for(pt_path), json.dumps(meta, indent=2, sort_keys=True))


def _load_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tensor_stats_1d(x: torch.Tensor) -> Dict[str, Any]:
    t = x.detach().to(torch.float32).flatten().cpu()
    n = int(t.numel())
    if n == 0:
        return {"n": 0}
    abs_t = t.abs()
    q = torch.tensor([0.01, 0.50, 0.90, 0.95, 0.99], dtype=torch.float32)
    try:
        p = torch.quantile(t, q).tolist()
        ap = torch.quantile(abs_t, q).tolist()
    except Exception:
        tn = t.numpy()
        an = abs_t.numpy()
        p = np.quantile(tn, q.numpy()).tolist()
        ap = np.quantile(an, q.numpy()).tolist()
    return {
        "n": n,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "p01": float(p[0]),
        "p50": float(p[1]),
        "p90": float(p[2]),
        "p95": float(p[3]),
        "p99": float(p[4]),
        "abs_p01": float(ap[0]),
        "abs_p50": float(ap[1]),
        "abs_p90": float(ap[2]),
        "abs_p95": float(ap[3]),
        "abs_p99": float(ap[4]),
    }


def _read_actions_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = _json_loads(line) if _json_loads is not None else json.loads(line.decode("utf-8"))
                    if isinstance(row, dict):
                        out.append(row)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _cache_params_dict(
    *,
    hold: int,
    seq_len: int,
    start_stride: int,
    gamma: float,
    reward_scale: float,
    positive_only: bool,
    ema_alpha: float,
) -> Dict[str, Any]:
    return {
        "format": "critic_minimal_q_v1",
        "hold": int(hold),
        "seq_len": int(seq_len),
        "start_stride": int(start_stride),
        "action_dim": int(ACTION_DIM),
        "action_keys": list(BUTTON_KEYS),
        "reward": "bellman_hp_return_sampled",
        "bellman": {
            "gamma": float(gamma),
            "reward_scale": float(reward_scale),
            "positive_only": bool(positive_only),
            "ema_alpha": float(ema_alpha),
            "reward_def": "max(0,dEHP)-max(0,dPHP) on sampled points",
        },
        # Scalars are normalized in-cache (dataset does NOT re-normalize).
        "scalars": {
            "names": ["p_hp", "e_hp", "p_charge", "e_charge", "cust_gauge"],
            "norm": {
                "p_hp_div": 2500.0,
                "e_hp_div": 2500.0,
                "p_charge_div": 2.0,
                "e_charge_div": 2.0,
                "cust_div": 100.0,
            },
        },
    }


def _extract_step_features(frames: List[Dict[str, Any]], raw_i: int) -> Dict[str, Any]:
    f = frames[raw_i]

    # Scalars (raw)
    p_hp = float(_as_int(f.get("player_health"), 0))
    e_hp = float(_as_int(f.get("enemy_health"), 0))
    p_chg = float(_as_int(f.get("player_charge"), 0))
    e_chg = float(_as_int(f.get("enemy_charge"), 0))
    cust = float(_as_int(f.get("cust_gauge"), 0))

    # Emotion ids (raw ids, keep as ints)
    p_emo = _as_int(f.get("player_game_emotion"), 0)
    e_emo = _as_int(f.get("enemy_game_emotion"), 0)

    # Grid (tile + owner)
    grid_state = _as_list(f.get("grid_state"))
    grid_owner = _as_list(f.get("grid_owner_state"))
    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0] * (18 - min(18, len(grid_state)))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2] * (18 - min(18, len(grid_owner)))

    # Positions -> indices
    px_i, py_i = _pos_norm(f.get("player_pos"))
    ex_i, ey_i = _pos_norm(f.get("enemy_pos"))
    p_grid_idx = pos_to_grid_idx(float(px_i), float(py_i))
    e_grid_idx = pos_to_grid_idx(float(ex_i), float(ey_i))
    rel_idx = rel_pe_index(p_grid_idx, e_grid_idx)

    # Chip id
    player_chip = _chip_id_norm(f.get("player_chip"))

    return {
        "p_hp": p_hp,
        "e_hp": e_hp,
        "p_chg": p_chg,
        "e_chg": e_chg,
        "cust": cust,
        "p_emo": p_emo,
        "e_emo": e_emo,
        "grid_tile": gs,
        "grid_owner": go,
        "p_grid_idx": p_grid_idx,
        "e_grid_idx": e_grid_idx,
        "rel_pe_idx": rel_idx,
        "player_chip": player_chip,
    }


def process_replay(
    folder: Path,
    *,
    out_dir: Path,
    overwrite: bool,
    hold: int,
    seq_len: int,
    start_stride: int,
    gamma: float,
    reward_scale: float,
    positive_only: bool,
    ema_alpha: float,
) -> Optional[Dict[str, int]]:
    name = folder.name
    save_file = out_dir / f"{name}.pt"

    params = _cache_params_dict(
        hold=hold,
        seq_len=seq_len,
        start_stride=start_stride,
        gamma=gamma,
        reward_scale=reward_scale,
        positive_only=positive_only,
        ema_alpha=ema_alpha,
    )

    if save_file.exists() and not overwrite:
        # Fast skip if params match
        try:
            ck = torch.load(save_file, map_location="cpu")
            if isinstance(ck, dict) and ck.get("params") == params:
                return None
        except Exception:
            pass

    act_path = folder / "actions.jsonl"
    if not act_path.exists():
        return None

    frames = _read_actions_jsonl(act_path)
    if not frames:
        return {"kept": 0, "total": 0, "skipped": 1}

    n_frames = len(frames)
    hold = max(1, int(hold))
    seq_len = max(1, int(seq_len))
    start_stride = max(1, int(start_stride))

    # sampled indices (raw frame indices at hold-stride)
    sample_raw: List[int] = list(range(0, n_frames, hold))
    S = len(sample_raw)
    if S == 0:
        return {"kept": 0, "total": n_frames, "skipped": 1}

    # battle mask at sampled points: cust>0 at that raw index
    battle_mask_sample: List[bool] = [(_as_int(frames[ri].get("cust_gauge"), 0) > 0) for ri in sample_raw]
    if not any(battle_mask_sample):
        return {"kept": 0, "total": n_frames, "skipped": 1}

    # Precompute per-step features on sampled timeline
    scalars = torch.zeros((S, 5), dtype=torch.float32)  # p/e hp, p/e charge, cust
    p_emo = torch.zeros((S,), dtype=torch.int64)
    e_emo = torch.zeros((S,), dtype=torch.int64)
    grid_tile = torch.zeros((S, 18), dtype=torch.int64)
    grid_owner = torch.zeros((S, 18), dtype=torch.int64)
    p_grid_idx = torch.zeros((S,), dtype=torch.int64)
    e_grid_idx = torch.zeros((S,), dtype=torch.int64)
    rel_pe_idx = torch.zeros((S,), dtype=torch.int64)
    player_chip = torch.zeros((S,), dtype=torch.int64)
    actions = torch.zeros((S, ACTION_DIM), dtype=torch.float32)

    for t, ri in enumerate(sample_raw):
        feats = _extract_step_features(frames, ri)

        # Normalize scalars into-cache (dataset will not redo this).
        scalars[t, 0] = float(feats["p_hp"]) / 2500.0
        scalars[t, 1] = float(feats["e_hp"]) / 2500.0
        scalars[t, 2] = float(feats["p_chg"]) / 2.0
        scalars[t, 3] = float(feats["e_chg"]) / 2.0
        scalars[t, 4] = float(feats["cust"]) / 100.0

        p_emo[t] = int(feats["p_emo"])
        e_emo[t] = int(feats["e_emo"])

        grid_tile[t] = torch.tensor(feats["grid_tile"], dtype=torch.int64)
        grid_owner[t] = torch.tensor(feats["grid_owner"], dtype=torch.int64)

        p_grid_idx[t] = int(feats["p_grid_idx"])
        e_grid_idx[t] = int(feats["e_grid_idx"])
        rel_pe_idx[t] = int(feats["rel_pe_idx"])

        player_chip[t] = int(feats["player_chip"])

        actions[t] = aggregate_action(frames, ri, hold)

    # Bellman targets on sampled timeline (length S)
    values_list = bellman_hp_return_sampled(
        frames,
        sample_raw,
        battle_mask_sample,
        gamma=float(gamma),
        reward_scale=float(reward_scale),
        positive_only=bool(positive_only),
        ema_alpha=float(ema_alpha),
    )
    values = torch.tensor(values_list, dtype=torch.float32)

    # Build sequences starting at each sampled index where battle is on.
    starts: List[int] = [i for i, ok in enumerate(battle_mask_sample) if ok]
    starts = starts[::start_stride]
    if not starts:
        return {"kept": 0, "total": n_frames, "skipped": 1}

    xs: Dict[str, List[torch.Tensor]] = {
        "scalars": [],
        "p_emotion_id": [],
        "e_emotion_id": [],
        "grid_tile": [],
        "grid_owner": [],
        "p_grid_idx": [],
        "e_grid_idx": [],
        "rel_pe_idx": [],
        "player_chip": [],
        "action": [],
    }
    ys: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    dones: List[torch.Tensor] = []
    start_raw_frames: List[int] = []

    for s0 in starts:
        v = torch.zeros((seq_len,), dtype=torch.bool)

        max_t = min(seq_len, S - s0)
        if max_t <= 0:
            continue

        valid_len = 0
        for t in range(max_t):
            if not bool(battle_mask_sample[s0 + t]):
                break
            valid_len += 1
        if valid_len <= 0:
            continue

        v[:valid_len] = True

        def _slice_pad_1d(src: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((seq_len,), dtype=src.dtype)
            n = min(seq_len, src.shape[0] - s0)
            if n > 0:
                out[:n] = src[s0 : s0 + n]
            return out

        def _slice_pad_2d(src: torch.Tensor, d1: int) -> torch.Tensor:
            out = torch.zeros((seq_len, d1), dtype=src.dtype)
            n = min(seq_len, src.shape[0] - s0)
            if n > 0:
                out[:n] = src[s0 : s0 + n]
            return out

        xs["scalars"].append(_slice_pad_2d(scalars, 5))
        xs["p_emotion_id"].append(_slice_pad_1d(p_emo))
        xs["e_emotion_id"].append(_slice_pad_1d(e_emo))
        xs["grid_tile"].append(_slice_pad_2d(grid_tile, 18))
        xs["grid_owner"].append(_slice_pad_2d(grid_owner, 18))
        xs["p_grid_idx"].append(_slice_pad_1d(p_grid_idx))
        xs["e_grid_idx"].append(_slice_pad_1d(e_grid_idx))
        xs["rel_pe_idx"].append(_slice_pad_1d(rel_pe_idx))
        xs["player_chip"].append(_slice_pad_1d(player_chip))
        xs["action"].append(_slice_pad_2d(actions, ACTION_DIM))

        # Targets: keep RAW y in cache; zero it outside valid for safety.
        y = _slice_pad_1d(values)
        y = torch.where(v, y, torch.zeros_like(y))
        ys.append(y)

        valids.append(v)
        dones.append(compute_done_from_valid(v))

        start_raw_frames.append(int(sample_raw[s0]))

    if not ys:
        return {"kept": 0, "total": n_frames, "skipped": 1}

    x_stacked = {k: torch.stack(vs, dim=0) for k, vs in xs.items()}  # [N, T, ...]
    y_stacked = torch.stack(ys, dim=0)  # [N, T] RAW bellman return
    valid_stacked = torch.stack(valids, dim=0)  # [N, T]
    done_stacked = torch.stack(dones, dim=0)  # [N, T]
    start_raw_t = torch.tensor(start_raw_frames, dtype=torch.int64)

    # Stats for meta (only on valid points)
    y_valid = y_stacked[valid_stacked]
    y_stats = _tensor_stats_1d(y_valid) if y_valid.numel() > 0 else {"n": 0}

    payload = {
        "replay": name,
        "params": params,
        "hold": int(hold),
        "seq_len": int(seq_len),
        "start_stride": int(start_stride),
        "start_raw_frame": start_raw_t,  # [N]
        "y": y_stacked,  # [N,T] RAW bellman return
        "valid": valid_stacked,  # [N,T]
        "done": done_stacked,  # [N,T]
        "x": x_stacked,  # dict of [N,T,...]
    }

    _atomic_torch_save(payload, save_file)

    meta = {
        "version": 1,
        "replay": name,
        "pt_file": save_file.name,
        "created_at": _now_iso(),
        "num_sequences": int(y_stacked.shape[0]),
        "seq_len": int(seq_len),
        "hold": int(hold),
        "start_stride": int(start_stride),
        "action_dim": int(ACTION_DIM),
        "action_keys": list(BUTTON_KEYS),
        "target_stats": y_stats,
        "target_recommendation": {
            "norm_factor_suggested_abs_p90": float(y_stats.get("abs_p90", 0.0) or 0.0),
            "norm_factor_suggested_abs_p95": float(y_stats.get("abs_p95", 0.0) or 0.0),
            "note": "Training uses y_norm=tanh(y_raw/norm_factor). Start with abs_p95-median across files.",
        },
        "bellman": params["bellman"],
        "scalars": params["scalars"],
    }
    _write_meta(save_file, meta)

    return {"kept": int(y_stacked.shape[0]), "total": int(n_frames), "skipped": 0}


def _write_manifest(out_dir: Path, *, overwrite: bool = True) -> Dict[str, Any]:
    metas: List[Dict[str, Any]] = []
    for mp in sorted(out_dir.glob("*.meta.json")):
        m = _load_meta(mp)
        if m is not None:
            metas.append(m)

    by_pt: Dict[str, Dict[str, Any]] = {}
    for m in metas:
        pf = str(m.get("pt_file", "") or "")
        if pf:
            by_pt[pf] = m
    files = [by_pt[k] for k in sorted(by_pt.keys())]

    total_seq = int(sum(int(m.get("num_sequences", 0) or 0) for m in files))
    abs_p90s = [
        m["target_stats"]["abs_p90"]
        for m in files
        if "target_stats" in m and isinstance(m["target_stats"], dict) and "abs_p90" in m["target_stats"]
    ]
    abs_p95s = [
        m["target_stats"]["abs_p95"]
        for m in files
        if "target_stats" in m and isinstance(m["target_stats"], dict) and "abs_p95" in m["target_stats"]
    ]

    manifest = {
        "version": 1,
        "created_at": _now_iso(),
        "out_dir": str(out_dir),
        "num_files": len(files),
        "total_sequences": total_seq,
        "files": files,
        "targets_summary": {
            "files_with_stats": int(len(abs_p90s)),
            "abs_p90_median_across_files": float(np.median(abs_p90s)) if abs_p90s else None,
            "abs_p95_median_across_files": float(np.median(abs_p95s)) if abs_p95s else None,
        },
    }

    if overwrite:
        _atomic_write_text(out_dir / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_job(args: Tuple[str, str, bool, int, int, int, float, float, bool, float]) -> Optional[Dict[str, int]]:
    (
        folder_str,
        out_dir_str,
        overwrite,
        hold,
        seq_len,
        start_stride,
        gamma,
        reward_scale,
        positive_only,
        ema_alpha,
    ) = args
    folder = Path(folder_str)
    out_dir = Path(out_dir_str)
    try:
        return process_replay(
            folder,
            out_dir=out_dir,
            overwrite=overwrite,
            hold=hold,
            seq_len=seq_len,
            start_stride=start_stride,
            gamma=gamma,
            reward_scale=reward_scale,
            positive_only=positive_only,
            ema_alpha=ema_alpha,
        )
    except Exception as e:
        print(f"❌ Error {folder.name}: {e}")
        return {"kept": 0, "total": 0, "skipped": 1}


def main() -> None:
    ap = argparse.ArgumentParser(description="Precache minimal Q(s,a) sequences with Bellman HP-return targets.")
    ap.add_argument("--dataset_dir", type=str, default="data/dataset")
    ap.add_argument("--out_dir", type=str, default="data/cache_minimal_q/h4_t192")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--hold", type=int, default=4, help="Hold/stride in raw frames (60fps). Default 4.")
    ap.add_argument("--seq_len", type=int, default=192)
    ap.add_argument("--start_stride", type=int, default=1, help="Keep every Nth possible start to control cache size.")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--positive_only", action="store_true", default=False)
    ap.add_argument("--ema_alpha", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=min(48, (os.cpu_count() or 1)))
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # best-effort cleanup of stray tmp
    for tmp in list(out_dir.glob("*.tmp")) + list(out_dir.glob("*.pt.tmp")) + list(out_dir.glob("*.meta.json.tmp")):
        with contextlib.suppress(Exception):
            tmp.unlink()

    replays = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and (d / "actions.jsonl").exists()])
    print(f"📦 minimal-q precache | replays={len(replays)}")
    print(f"   out_dir={out_dir}")
    print(f"   hold={args.hold} seq_len={args.seq_len} start_stride={args.start_stride}")
    print(
        f"   bellman: gamma={args.gamma} reward_scale={args.reward_scale} "
        f"positive_only={bool(args.positive_only)} ema_alpha={args.ema_alpha}"
    )
    print(f"   workers={args.num_workers}")
    print("   json: orjson=on" if _json_loads is not None else "   json: stdlib=json")

    jobs: List[Tuple[str, str, bool, int, int, int, float, float, bool, float]] = []
    for folder in replays:
        jobs.append(
            (
                str(folder),
                str(out_dir),
                bool(args.overwrite),
                int(args.hold),
                int(args.seq_len),
                int(args.start_stride),
                float(args.gamma),
                float(args.reward_scale),
                bool(args.positive_only),
                float(args.ema_alpha),
            )
        )

    stats = {"kept": 0, "total": 0, "skipped": 0}
    interrupted = False

    pbar = tqdm(total=len(jobs), desc="Precaching", dynamic_ncols=True)
    try:
        nw = max(1, int(args.num_workers))
        if nw == 1:
            for job in jobs:
                res = _worker_job(job)
                if res:
                    stats["kept"] += int(res["kept"])
                    stats["total"] += int(res["total"])
                    stats["skipped"] += int(res["skipped"])
                pbar.update(1)
                pbar.set_postfix(
                    kept=f"{stats['kept']:,}",
                    scanned=f"{stats['total']:,}",
                    skipped=f"{stats['skipped']:,}",
                )
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
            with ctx.Pool(processes=nw, initializer=_init_worker) as pool:
                for res in pool.imap_unordered(_worker_job, jobs, chunksize=1):
                    if res:
                        stats["kept"] += int(res["kept"])
                        stats["total"] += int(res["total"])
                        stats["skipped"] += int(res["skipped"])
                    pbar.update(1)
                    pbar.set_postfix(
                        kept=f"{stats['kept']:,}",
                        scanned=f"{stats['total']:,}",
                        skipped=f"{stats['skipped']:,}",
                    )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pbar.close()

        manifest = _write_manifest(out_dir, overwrite=True)
        print(
            f"🧾 Wrote manifest: {out_dir / MANIFEST_NAME} | "
            f"files={manifest['num_files']} sequences={manifest['total_sequences']:,}"
        )

        if interrupted:
            print("\n🛑 Interrupted (Ctrl+C). Partial cache is valid; manifest reflects completed replays.")
            raise SystemExit(1)

        print("\n✅ Precache Complete.")
        print(f"  Frames Scanned:   {stats['total']:,}")
        print(f"  Sequences Cached: {stats['kept']:,}")
        print(f"  Replays Skipped:  {stats['skipped']:,}")


if __name__ == "__main__":
    main()
