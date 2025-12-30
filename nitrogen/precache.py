# nitrogen/precache.py
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

from action_schema import BUTTON_TOKENS, ACTION_DIM

# Optional fast JSON parsing (huge speedup on actions.jsonl)
try:
    import orjson  # type: ignore

    _json_loads = orjson.loads
except Exception:
    _json_loads = None

# -----------------------------------------------------------------------------
# Defaults (override via CLI)
# -----------------------------------------------------------------------------
DEFAULT_SOURCE_DIR = "data/dataset"
DEFAULT_OUT_DIR = "data/nitrogen_battle_cache_bellman"

RESOLUTION_HW = (256, 256)  # (H, W)
NATIVE_RES_HW = (160, 240)  # (H, W)

DEFAULT_ACTION_HORIZON = 18

# Video decode batching
DEFAULT_FRAME_BATCH = 256  # was 64

# Bellman HP-return defaults (no critic)
# Reward per-frame: (enemy_damage - player_damage) using raw HP deltas (not normalized)
DEFAULT_GAMMA = 0.997
DEFAULT_REWARD_SCALE = 1.0
DEFAULT_USE_POSITIVE_ONLY = False  # if True: clamp reward to >=0 (damage dealt only)

# Optional smoothing (applied only inside battle segments)
DEFAULT_EMA_ALPHA = 0.0  # 0.0 = off

# Workers: default to 48 if available, else cpu_count
_DEFAULT_CPU = os.cpu_count() or 1
DEFAULT_NUM_WORKERS = 48 if _DEFAULT_CPU >= 48 else _DEFAULT_CPU

# Manifest/meta
MANIFEST_NAME = "manifest.json"


# -----------------------------------------------------------------------------
# Atomic IO helpers (interrupt-safe)
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def _tensor_stats_1d(x: torch.Tensor) -> Dict[str, Any]:
    """
    Compute robust stats for a 1D float tensor on CPU.
    """
    t = x.detach().to(torch.float32).flatten().cpu()
    n = int(t.numel())
    if n == 0:
        return {"n": 0}

    abs_t = t.abs()
    q = torch.tensor([0.01, 0.50, 0.90, 0.95, 0.99], dtype=torch.float32)

    # torch.quantile exists on recent torch; if not, fallback to numpy.
    try:
        p = torch.quantile(t, q).tolist()
        ap = torch.quantile(abs_t, q).tolist()
    except Exception:
        import numpy as _np
        tn = t.numpy()
        an = abs_t.numpy()
        p = _np.quantile(tn, q.numpy()).tolist()
        ap = _np.quantile(an, q.numpy()).tolist()

    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    return {
        "n": n,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": mean,
        "std": std,
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


def _meta_path_for(save_file: Path) -> Path:
    return save_file.with_suffix(".meta.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_torch_save(payload: Dict[str, Any], save_path: Path) -> None:
    """
    Save a .pt atomically: write to .tmp then os.replace().
    Prevents partial/corrupt .pt when interrupted.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    torch.save(payload, tmp)
    tmp.replace(save_path)


def _write_meta(save_file: Path, meta: Dict[str, Any]) -> None:
    mp = _meta_path_for(save_file)
    _atomic_write_text(mp, json.dumps(meta, indent=2, sort_keys=True))


def _load_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_meta_for_existing_pt(pt_path: Path) -> Optional[Dict[str, Any]]:
    """
    If <replay>.meta.json is missing but <replay>.pt exists, load the pt once
    to derive N and write meta. Used only when --rebuild_manifest is set.
    """
    meta_path = _meta_path_for(pt_path)
    if meta_path.exists():
        return _load_meta(meta_path)

    try:
        payload = torch.load(pt_path, map_location="cpu")
        n = int(payload["frames"].shape[0])
        ah = int(payload["actions"].shape[1])
        ad = int(payload["actions"].shape[2])
        rh = int(payload["frames"].shape[2])
        rw = int(payload["frames"].shape[3])

        meta = {
            "version": 1,
            "replay": pt_path.stem,
            "pt_file": pt_path.name,
            "num_samples": n,
            "action_horizon": ah,
            "action_dim": ad,
            "resolution_hw": [rh, rw],
            "created_at": _now_iso(),
            "source": "rebuild_manifest",
        }
        _write_meta(pt_path, meta)
        return meta
    except Exception:
        return None


def _write_manifest(out_dir: Path, *, overwrite: bool = True, rebuild_missing_meta: bool = False) -> Dict[str, Any]:
    """
    Build manifest from *.meta.json (fast). Optionally rebuild missing meta by loading pt once.
    """
    metas: List[Dict[str, Any]] = []

    if rebuild_missing_meta:
        for pt in sorted(out_dir.glob("*.pt")):
            m = _ensure_meta_for_existing_pt(pt)
            if m is not None:
                metas.append(m)

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



    total_samples = int(sum(int(m.get("num_samples", 0) or 0) for m in files))
    manifest = {
        "version": 1,
        "created_at": _now_iso(),
        "out_dir": str(out_dir),
        "num_files": len(files),
        "total_samples": total_samples,
        "files": files,
    }

    abs_p90s = [m["values_stats"]["abs_p90"] for m in files if "values_stats" in m]
    manifest["values_summary"] = {
        "files_with_stats": len(abs_p90s),
        "abs_p90_median_across_files": float(np.median(abs_p90s)) if abs_p90s else None,
        "abs_p95_median_across_files": float(np.median([m["values_stats"]["abs_p95"] for m in files if "values_stats" in m])) if abs_p90s else None,
    }


    if overwrite:
        _atomic_write_text(out_dir / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))

    return manifest


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _scalar(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        return float(v[0]) if v else float(default)
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size > 0 else float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _norm_axis(v: float) -> float:
    fv = float(v)
    if abs(fv) > 1.5:
        fv = fv / 32767.0
    return max(-1.0, min(1.0, fv))


def _btn01(v: float) -> float:
    return 1.0 if float(v) > 0.5 else 0.0


def process_batch_gba(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    frames: [B, 3, H, W] uint8/float
    Produces centered padded [B, 3, target_h, target_w]
    """
    B, C, H, W = frames.shape
    native_h, native_w = NATIVE_RES_HW

    if (H, W) != (native_h, native_w):
        frames_native = F.interpolate(frames.float(), size=(native_h, native_w), mode="nearest")
    else:
        frames_native = frames.float()

    canvas = torch.zeros((B, C, target_h, target_w), dtype=frames_native.dtype)
    y_off = (target_h - native_h) // 2
    x_off = (target_w - native_w) // 2

    paste_h, paste_w = min(native_h, target_h), min(native_w, target_w)
    canvas[:, :, y_off : y_off + paste_h, x_off : x_off + paste_w] = frames_native[:, :, :paste_h, :paste_w]
    return canvas


def _ema_smooth_masked(values: List[float], mask: List[bool], alpha: float) -> List[float]:
    """
    EMA is reset whenever mask is False.
    """
    if alpha <= 0.0:
        return values

    out = [0.0] * len(values)
    ema: Optional[float] = None
    for i, v in enumerate(values):
        if not mask[i]:
            ema = None
            out[i] = 0.0
            continue
        ema = float(v) if ema is None else (alpha * float(v) + (1.0 - alpha) * ema)
        out[i] = float(ema)
    return out


def _bellman_hp_return(
    frames: List[Dict[str, Any]],
    battle_mask: List[bool],
    *,
    gamma: float,
    reward_scale: float,
    positive_only: bool,
) -> List[float]:
    """
    Computes a dense per-frame discounted return using raw HP deltas.

    reward_t = (max(0, enemy_hp[t-1]-enemy_hp[t]) - max(0, player_hp[t-1]-player_hp[t])) * reward_scale
    return_t = reward_t + gamma * return_{t+1} within contiguous battle segments
    Outside battle_mask => 0.0, and the return chain resets at segment boundaries.
    """
    n = len(frames)
    if n == 0:
        return []

    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    # Rewards (use prev->curr delta); reward[0] = 0
    r = [0.0] * n
    prev_p = _scalar(frames[0].get("player_health"), 0.0)
    prev_e = _scalar(frames[0].get("enemy_health"), 0.0)

    for i in range(1, n):
        p = _scalar(frames[i].get("player_health"), prev_p)
        e = _scalar(frames[i].get("enemy_health"), prev_e)

        dmg_taken = max(0.0, prev_p - p)
        dmg_dealt = max(0.0, prev_e - e)

        rt = (dmg_dealt - dmg_taken) * float(reward_scale)
        if positive_only:
            rt = max(0.0, rt)
        r[i] = rt

        prev_p, prev_e = p, e

    # Discounted return (backward), reset outside mask
    out = [0.0] * n
    nxt = 0.0
    for i in range(n - 1, -1, -1):
        if not battle_mask[i]:
            nxt = 0.0
            out[i] = 0.0
            continue
        nxt = r[i] + g * nxt
        out[i] = nxt

    return out


# -----------------------------------------------------------------------------
# Core replay processing
# -----------------------------------------------------------------------------
def _load_static(static_path: Path) -> Dict[str, Any]:
    try:
        with open(static_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def process_replay(
    folder: Path,
    *,
    out_dir: Path,
    overwrite: bool,
    action_horizon: int,
    frame_batch: int,
    gamma: float,
    reward_scale: float,
    positive_only: bool,
    ema_alpha: float,
) -> Optional[Dict[str, int]]:
    name = folder.name
    save_file = out_dir / f"{name}.pt"
    if save_file.exists() and not overwrite:
        return None

    vid_path = folder / "video.mp4"
    act_path = folder / "actions.jsonl"
    static_path = folder / "static_data.json"

    if not (vid_path.exists() and act_path.exists()):
        return None

    # Loaded for parity / future extensions; not required for Bellman return
    _ = _load_static(static_path)

    frames_data: List[Dict[str, Any]] = []
    raw_actions: List[List[float]] = []
    raw_states: List[List[float]] = []  # [p_hp_norm, e_hp_norm, charge_norm]
    raw_chips: List[int] = []
    battle_map: List[Tuple[int, int]] = []  # (action_idx, video_idx)

    # --- Parse actions.jsonl with strict sync ---
    action_idx = 0

    with open(act_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue

            if _json_loads is not None:
                row = _json_loads(line)
            else:
                row = json.loads(line.decode("utf-8"))

            frames_data.append(row)

            vec = [
                _norm_axis(_scalar(row.get("AXIS_LEFTX"))),
                _norm_axis(_scalar(row.get("AXIS_LEFTY"))),
                _norm_axis(_scalar(row.get("AXIS_RIGHTX"))),
                _norm_axis(_scalar(row.get("AXIS_RIGHTY"))),
            ]
            for btn in BUTTON_TOKENS:
                vec.append(_btn01(_scalar(row.get(btn))))
            raw_actions.append(vec)

            # keep your normalized state tensor (still useful for training)
            p_hp = _scalar(row.get("player_health"), 0.0) / 2500.0
            e_hp = _scalar(row.get("enemy_health"), 0.0) / 2500.0
            p_chg = _scalar(row.get("player_charge"), 0.0) / 2.0
            p_chip = int(_scalar(row.get("player_chip"), 0))
            raw_states.append([p_hp, e_hp, p_chg])
            raw_chips.append(p_chip)

            vid_idx = int(row.get("frame_idx", action_idx))

            cust_gauge = int(row.get("cust_gauge", 0))
            if cust_gauge > 0:
                battle_map.append((action_idx, vid_idx))

            action_idx += 1

    if not frames_data or not battle_map:
        return {"kept": 0, "total": len(frames_data), "skipped": 1}

    n_frames = len(frames_data)

    # --- battle mask ---
    battle_mask = [int(frames_data[i].get("cust_gauge", 0)) > 0 for i in range(n_frames)]

    # --- Bellman HP-return (dense, then optional EMA inside battle) ---
    bellman_values = _bellman_hp_return(
        frames_data,
        battle_mask,
        gamma=gamma,
        reward_scale=reward_scale,
        positive_only=positive_only,
    )
    if ema_alpha > 0.0:
        bellman_values = _ema_smooth_masked(bellman_values, battle_mask, alpha=ema_alpha)

    # --- Bake base tensors ---
    all_actions_tensor = torch.tensor(raw_actions, dtype=torch.float32)  # [T, ACTION_DIM]
    all_values_tensor = torch.tensor(bellman_values, dtype=torch.float32)  # [T]
    all_states_tensor = torch.tensor(raw_states, dtype=torch.float32)  # [T, 3]
    all_chips_tensor = torch.tensor(raw_chips, dtype=torch.long)  # [T]

    # --- Vectorized action windows for battle frames only ---
    battle_action_indices = torch.tensor([a for a, _ in battle_map], dtype=torch.long)
    kept_video_indices = torch.tensor([v for _, v in battle_map], dtype=torch.long)

    T = int(all_actions_tensor.shape[0])
    H = int(action_horizon)
    D = int(all_actions_tensor.shape[1])
    if D != ACTION_DIM:
        raise RuntimeError(f"ACTION_DIM mismatch: got {D}, expected {ACTION_DIM}")

    pad = torch.zeros((H - 1, D), dtype=all_actions_tensor.dtype)
    actions_padded = torch.cat([all_actions_tensor, pad], dim=0)  # [T+H-1, D]

    # unfold(0, H, 1) returns [T, D, H] for a 2D tensor; permute to [T, H, D]
    all_windows = actions_padded.unfold(0, H, 1).permute(0, 2, 1).contiguous()  # [T, H, D]

    final_actions = all_windows.index_select(0, battle_action_indices).contiguous()

    # fail fast if anything is off
    if final_actions.shape[1] != H or final_actions.shape[2] != D:
        raise RuntimeError(
            f"Bad window shape for {name}: got {tuple(final_actions.shape)}, expected [N,{H},{D}]"
        )


    final_values = all_values_tensor.index_select(0, battle_action_indices).contiguous()
    value_stats = _tensor_stats_1d(final_values)
    final_states = all_states_tensor.index_select(0, battle_action_indices).contiguous()
    final_chips = all_chips_tensor.index_select(0, battle_action_indices).contiguous()

    kept_action_indices = battle_action_indices

    # --- Video read ---
    vr = VideoReader(str(vid_path), ctx=cpu(0))
    real_vid_len = int(len(vr))

    # Filter any out-of-range video indices once (vectorized)
    valid = kept_video_indices < real_vid_len
    if int(valid.sum().item()) == 0:
        return {"kept": 0, "total": n_frames, "skipped": 1}

    kept_video_indices = kept_video_indices[valid]
    kept_action_indices = kept_action_indices[valid]
    final_actions = final_actions[valid]
    final_values = final_values[valid]
    final_states = final_states[valid]
    final_chips = final_chips[valid]

    # --- Extract frames in chunks ---
    target_h, target_w = RESOLUTION_HW
    N = int(kept_video_indices.shape[0])
    final_frames = torch.empty((N, 3, target_h, target_w), dtype=torch.uint8)

    write_ptr = 0
    batch_vid: List[int] = []
    frame_batch = max(1, int(frame_batch))

    for i in range(N):
        vid_idx = int(kept_video_indices[i].item())
        batch_vid.append(vid_idx)

        if len(batch_vid) == frame_batch or i == N - 1:
            try:
                v_batch = vr.get_batch(batch_vid).asnumpy()  # [B, H, W, C]
                t_batch = torch.from_numpy(v_batch).permute(0, 3, 1, 2)  # [B, C, H, W]
                t_proc = process_batch_gba(t_batch, target_h, target_w)  # [B, C, target_h, target_w]
                n_in = int(t_proc.shape[0])
                final_frames[write_ptr : write_ptr + n_in] = t_proc.to(torch.uint8)
                write_ptr += n_in
            except Exception as e:
                print(f"⚠️ Read error {name} @write_ptr={write_ptr}: {e}")
                write_ptr += len(batch_vid)
            batch_vid = []

    payload = {
        "frames": final_frames,  # [N, 3, H, W] uint8
        "actions": final_actions,  # [N, action_horizon, ACTION_DIM] float32
        "values": final_values,  # [N] float32 (Bellman HP-return)
        "states": final_states,  # [N, 3] float32
        "chips": final_chips,  # [N] int64
        "action_indices": kept_action_indices.to(torch.long),
        "video_indices": kept_video_indices.to(torch.long),
    }

    # --- Atomic write .pt then write meta ---
    _atomic_torch_save(payload, save_file)

    meta = {
        "version": 1,
        "replay": name,
        "pt_file": save_file.name,
        "num_samples": int(final_frames.shape[0]),
        "total_frames_in_actions": int(n_frames),
        "action_horizon": int(action_horizon),
        "action_dim": int(final_actions.shape[2]),
        "resolution_hw": [int(target_h), int(target_w)],
        "bellman": {
            "gamma": float(gamma),
            "reward_scale": float(reward_scale),
            "positive_only": bool(positive_only),
            "ema_alpha": float(ema_alpha),
            "reward_def": "max(0, dEHP) - max(0, dPHP) using prev->curr HP deltas",
        },
        "video": {"frame_batch": int(frame_batch)},
        "created_at": _now_iso(),
        "source": "precache_bellman",
        "values_stats": value_stats,
        "values_recommendation": {
            "norm_factor_suggested": float(value_stats.get("abs_p90", 0.0) or 0.0),
            "note": "Use abs_p90 (or abs_p95) as a starting norm_factor for tanh-based weighting.",
        },

    }
    _write_meta(save_file, meta)

    return {"kept": int(final_frames.shape[0]), "total": int(n_frames), "skipped": 0}


# -----------------------------------------------------------------------------
# Multiprocessing wiring
# -----------------------------------------------------------------------------
def _init_worker() -> None:
    # Let the parent handle Ctrl+C cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_job(
    args: Tuple[str, str, bool, int, int, float, float, bool, float]
) -> Optional[Dict[str, int]]:
    (
        folder_str,
        out_dir_str,
        overwrite,
        action_horizon,
        frame_batch,
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
            action_horizon=action_horizon,
            frame_batch=frame_batch,
            gamma=gamma,
            reward_scale=reward_scale,
            positive_only=positive_only,
            ema_alpha=ema_alpha,
        )
    except Exception as e:
        print(f"❌ Error {folder.name}: {e}")
        return {"kept": 0, "total": 0, "skipped": 1}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache battle-only frames with Bellman HP-return values for Nitrogen.")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--action_horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--frame_batch", type=int, default=DEFAULT_FRAME_BATCH)

    # Bellman knobs
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--reward_scale", type=float, default=DEFAULT_REWARD_SCALE)
    parser.add_argument("--positive_only", action="store_true", default=DEFAULT_USE_POSITIVE_ONLY)
    parser.add_argument("--ema_alpha", type=float, default=DEFAULT_EMA_ALPHA)

    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

    parser.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="At end, ensure every .pt has a .meta.json (loading only those missing), then write manifest.json.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort cleanup of stray tmp files from prior interrupts
    for tmp in (
        list(out_dir.glob("*.pt.tmp"))
        + list(out_dir.glob("*.meta.json.tmp"))
        + list(out_dir.glob("manifest.json.tmp"))
    ):
        with contextlib.suppress(Exception):
            tmp.unlink()

    replays = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"📦 Battle-only precache | Bellman HP-return (no critic) | replays={len(replays)}")
    print(f"   out_dir={out_dir}")
    print(f"   horizon={args.action_horizon}")
    print(
        "   bellman:"
        f" gamma={args.gamma}"
        f" reward_scale={args.reward_scale}"
        f" positive_only={bool(args.positive_only)}"
        f" ema_alpha={args.ema_alpha}"
    )
    print(f"   video: frame_batch={args.frame_batch}")
    print(f"   workers={args.num_workers}")
    print("   json: orjson=on" if _json_loads is not None else "   json: stdlib=json")

    stats = {"kept": 0, "total": 0, "skipped": 0}
    interrupted = False

    jobs: List[Tuple[str, str, bool, int, int, float, float, bool, float]] = []
    for folder in replays:
        jobs.append(
            (
                str(folder),
                str(out_dir),
                bool(args.overwrite),
                int(args.action_horizon),
                int(args.frame_batch),
                float(args.gamma),
                float(args.reward_scale),
                bool(args.positive_only),
                float(args.ema_alpha),
            )
        )

    nw = max(1, int(args.num_workers))
    pbar = tqdm(total=len(jobs), desc="Precaching", dynamic_ncols=True)

    try:
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

        manifest = _write_manifest(
            out_dir,
            overwrite=True,
            rebuild_missing_meta=bool(args.rebuild_manifest),
        )
        print(
            f"🧾 Wrote manifest: {out_dir / MANIFEST_NAME} | "
            f"files={manifest['num_files']} samples={manifest['total_samples']:,}"
        )

        if interrupted:
            print("\n🛑 Interrupted (Ctrl+C). Partial cache is valid; manifest reflects completed replays.")
            raise SystemExit(1)

        print("\n✅ Precache Complete (battle-only + bellman).")
        print(f"  Frames Scanned:   {stats['total']:,}")
        print(f"  Frames Cached:    {stats['kept']:,}")
        print(f"  Replays Skipped:  {stats['skipped']:,}")


if __name__ == "__main__":
    main()
