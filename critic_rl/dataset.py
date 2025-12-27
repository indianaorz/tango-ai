from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# IMPORTANT: import your derived_state compute_derived
from viewer.derived_state import compute_derived  # type: ignore


INVALID_CHIP_IDS = {255, 65535}


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else []


def _chip_id_norm(v: Any) -> int:
    """
    Normalize chip ids to a non-negative int for hashing.
    0 is reserved for "none/invalid".
    """
    try:
        x = int(v)
    except Exception:
        return 0
    if x in INVALID_CHIP_IDS or x < 0:
        return 0
    return x


def _code_norm(v: Any, max_code: int = 63) -> int:
    try:
        x = int(v)
    except Exception:
        return 0
    if x < 0:
        return 0
    if x > max_code:
        return max_code
    return x


def _pos_norm(v: Any) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _as_int(v[0], 0), _as_int(v[1], 0)
    if isinstance(v, str) and "," in v:
        parts = v.split(",")
        if len(parts) >= 2:
            return _as_int(parts[0], 0), _as_int(parts[1], 0)
    return 0, 0


def _tile_norm(v: Any) -> int:
    if v is None:
        return 0
    try:
        x = int(v)
    except Exception:
        return 0
    if x < 0:
        return 0
    return x


def _owner_norm(v: Any) -> int:
    # 0 or 1 known, else 2
    x = _as_int(v, 2)
    if x == 0:
        return 0
    if x == 1:
        return 1
    return 2


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_actions_jsonl(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except Exception:
                continue
    return frames


# ---------------------------------------------------------------------------
# RL signals: hp_delta rewards + terminals
# ---------------------------------------------------------------------------

def _hp_delta_series(frames: List[Dict[str, Any]]) -> List[int]:
    """
    hp_delta[t] = player_hp[t] - enemy_hp[t]
    Uses carry-forward for missing hp fields.
    """
    out: List[int] = []
    p = 0
    e = 0
    for f in frames:
        if "player_health" in f:
            p = _as_int(f.get("player_health"), p)
        if "enemy_health" in f:
            e = _as_int(f.get("enemy_health"), e)
        out.append(int(p - e))
    return out


def _hp_delta_rewards(hp_delta: List[int]) -> List[float]:
    """
    reward r_t = hp_delta[t+1] - hp_delta[t]
    last reward = 0.
    """
    if not hp_delta:
        return []
    r = [0.0] * len(hp_delta)
    for t in range(len(hp_delta) - 1):
        r[t] = float(hp_delta[t + 1] - hp_delta[t])
    r[-1] = 0.0
    return r


# ---------------------------------------------------------------------------
# Replay metadata and small RAM cache (JSONL-mode)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayMeta:
    name: str
    path: Path
    n_frames: int
    static: Dict[str, Any]


class _ReplayCache:
    """
    LRU-ish cache of parsed replay data and derived state.
    Keeps memory bounded while allowing random access.
    """
    def __init__(self, *, max_replays: int = 2):
        self.max_replays = int(max_replays)
        self._order: List[str] = []
        self._frames: Dict[str, List[Dict[str, Any]]] = {}
        self._derived: Dict[str, List[Dict[str, Any]]] = {}
        self._rewards: Dict[str, List[float]] = {}

    def get(self, meta: ReplayMeta) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
        key = meta.name
        if key in self._frames and key in self._derived and key in self._rewards:
            self._touch(key)
            return self._frames[key], self._derived[key], self._rewards[key]

        frames = _read_actions_jsonl(meta.path / "actions.jsonl")
        derived = compute_derived(frames, meta.static)

        hp_delta = _hp_delta_series(frames)
        rewards = _hp_delta_rewards(hp_delta)

        self._frames[key] = frames
        self._derived[key] = derived
        self._rewards[key] = rewards
        self._touch(key)
        self._evict_if_needed()
        return frames, derived, rewards

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.max_replays:
            victim = self._order.pop(0)
            self._frames.pop(victim, None)
            self._derived.pop(victim, None)
            self._rewards.pop(victim, None)


# ---------------------------------------------------------------------------
# Tensorization (per-frame)
# ---------------------------------------------------------------------------

def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask


def tensorize_frame(
    frame: Dict[str, Any],
    d: Dict[str, Any],
    static: Dict[str, Any],
    *,
    folder_len: int = 30,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build model input tensors from:
      - raw frame dict
      - derived dict at same idx
      - static dict for replay

    Returns tensors on CPU unless device provided.
    """
    p_hp = _as_int(frame.get("player_health"), 0)
    e_hp = _as_int(frame.get("enemy_health"), 0)
    p_chg = _as_int(frame.get("player_charge"), 0)
    e_chg = _as_int(frame.get("enemy_charge"), 0)
    cust = _as_int(frame.get("cust_gauge"), 0)
    inside = 1.0 if _as_bool(frame.get("inside_window"), False) else 0.0
    turn_idx = _as_int(d.get("turn_index"), 0)

    px, py = _pos_norm(frame.get("player_pos"))
    ex, ey = _pos_norm(frame.get("enemy_pos"))

    scalars = torch.zeros(32, dtype=torch.float32)
    scalars[0] = float(p_hp) / 1000.0
    scalars[1] = float(e_hp) / 1000.0
    scalars[2] = float(p_chg) / 100.0
    scalars[3] = float(e_chg) / 100.0
    scalars[4] = float(cust) / 100.0
    scalars[5] = float(inside)
    scalars[6] = float(turn_idx) / 50.0
    scalars[7] = float(px) / 5.0
    scalars[8] = float(py) / 5.0
    scalars[9] = float(ex) / 5.0
    scalars[10] = float(ey) / 5.0

    grid_state = _as_list(frame.get("grid_state"))
    grid_owner = _as_list(frame.get("grid_owner_state"))
    gs = [0] * 18
    go = [2] * 18
    for i in range(min(18, len(grid_state))):
        gs[i] = _tile_norm(grid_state[i])
    for i in range(min(18, len(grid_owner))):
        go[i] = _owner_norm(grid_owner[i])

    chip_slots = _as_list(frame.get("chip_slots"))
    chip_codes = _as_list(frame.get("chip_codes"))
    vis = _as_int(frame.get("chip_visible_count"), 5)

    hand_id = [0] * 10
    hand_code = [0] * 10
    hand_vis = [0.0] * 10
    for i in range(10):
        if i < len(chip_slots):
            hand_id[i] = _chip_id_norm(chip_slots[i])
            hand_code[i] = _code_norm(chip_codes[i] if i < len(chip_codes) else 0)
            hand_vis[i] = 1.0 if i < vis else 0.0

    p_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("player_folder_ids"))]
    e_folder_ids = [_chip_id_norm(x) for x in _as_list(static.get("enemy_folder_ids"))]
    p_ids, p_mask = _pad_ids(p_folder_ids, folder_len)
    e_ids, e_mask = _pad_ids(e_folder_ids, folder_len)

    p_used_mask_raw = _as_list(d.get("player", {}).get("folder_used_mask"))
    e_used_mask_raw = _as_list(d.get("enemy", {}).get("folder_used_mask"))

    p_used = [0.0] * folder_len
    e_used = [0.0] * folder_len
    for i in range(min(folder_len, len(p_used_mask_raw))):
        p_used[i] = 1.0 if bool(p_used_mask_raw[i]) else 0.0
    for i in range(min(folder_len, len(e_used_mask_raw))):
        e_used[i] = 1.0 if bool(e_used_mask_raw[i]) else 0.0

    held = _as_list(d.get("player", {}).get("held_chips"))
    held_id = [0] * 5
    held_code = [0] * 5
    held_mask = [False] * 5
    for i in range(min(5, len(held))):
        h = held[i] if isinstance(held[i], dict) else {}
        held_id[i] = _chip_id_norm(h.get("id"))
        held_code[i] = _code_norm(h.get("code"))
        held_mask[i] = True

    p_used_cross = _as_list(d.get("player", {}).get("used_cross_mask"))
    e_used_cross = _as_list(d.get("enemy", {}).get("used_cross_mask"))
    used_cross = [0.0] * 22
    for i in range(min(11, len(p_used_cross))):
        used_cross[i] = 1.0 if bool(p_used_cross[i]) else 0.0
    for i in range(min(11, len(e_used_cross))):
        used_cross[11 + i] = 1.0 if bool(e_used_cross[i]) else 0.0

    p_active = d.get("player", {}).get("active_cross") or None
    e_active = d.get("enemy", {}).get("active_cross") or None
    p_idx = _as_int(p_active.get("idx"), 11) if isinstance(p_active, dict) else 11
    e_idx = _as_int(e_active.get("idx"), 11) if isinstance(e_active, dict) else 11

    p_beast = d.get("player", {}).get("beast") or {}
    e_beast = d.get("enemy", {}).get("beast") or {}
    p_b_active = 1.0 if bool(p_beast.get("active")) else 0.0
    e_b_active = 1.0 if bool(e_beast.get("active")) else 0.0
    p_b_ever = 1.0 if bool(p_beast.get("ever")) else 0.0
    e_b_ever = 1.0 if bool(e_beast.get("ever")) else 0.0
    p_ts = p_beast.get("turns_since")
    e_ts = e_beast.get("turns_since")
    p_ts_f = float(_as_int(p_ts, 999)) / 50.0 if p_ts is not None else 9.99
    e_ts_f = float(_as_int(e_ts, 999)) / 50.0 if e_ts is not None else 9.99

    beast_feats = torch.tensor([p_b_active, p_ts_f, p_b_ever, e_b_active, e_ts_f, e_b_ever], dtype=torch.float32)

    out: Dict[str, torch.Tensor] = {
        "scalars": scalars,
        "grid_tile": torch.tensor(gs, dtype=torch.int64),
        "grid_owner": torch.tensor(go, dtype=torch.int64),

        "hand_id": torch.tensor(hand_id, dtype=torch.int64),
        "hand_code": torch.tensor(hand_code, dtype=torch.int64),
        "hand_vis": torch.tensor(hand_vis, dtype=torch.float32),

        "folder_id_p": torch.tensor(p_ids, dtype=torch.int64),
        "folder_used_p": torch.tensor(p_used, dtype=torch.float32),
        "folder_mask_p": torch.tensor(p_mask, dtype=torch.bool),

        "folder_id_e": torch.tensor(e_ids, dtype=torch.int64),
        "folder_used_e": torch.tensor(e_used, dtype=torch.float32),
        "folder_mask_e": torch.tensor(e_mask, dtype=torch.bool),

        "held_id": torch.tensor(held_id, dtype=torch.int64),
        "held_code": torch.tensor(held_code, dtype=torch.int64),
        "held_mask": torch.tensor(held_mask, dtype=torch.bool),

        "used_cross": torch.tensor(used_cross, dtype=torch.float32),
        "active_cross_idx_p": torch.tensor(p_idx, dtype=torch.int64),
        "active_cross_idx_e": torch.tensor(e_idx, dtype=torch.int64),
        "beast_feats": beast_feats,
    }

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}

    return out


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def _stack_time(xs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # xs: list of per-frame dicts (len=T), each tensor is (...)
    out: Dict[str, List[torch.Tensor]] = {}
    for x in xs:
        for k, v in x.items():
            out.setdefault(k, []).append(v)
    # stack -> (T, ...)
    return {k: torch.stack(vs, dim=0) for k, vs in out.items()}


def _pad_sequence_time(
    seq: Dict[str, torch.Tensor],
    *,
    target_len: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Pad a (T, ...) sequence dict up to target_len on time dimension with zeros.
    Returns (padded_seq, valid_mask[T]).
    """
    if not seq:
        raise ValueError("empty seq")
    # infer current T from any key
    any_t = next(iter(seq.values()))
    if any_t.ndim < 1:
        raise ValueError("sequence tensors must have time dimension")
    T = int(any_t.shape[0])

    valid = torch.zeros((target_len,), dtype=torch.bool)
    n = min(T, target_len)
    valid[:n] = True

    if T >= target_len:
        return {k: v[:target_len] for k, v in seq.items()}, valid

    padded: Dict[str, torch.Tensor] = {}
    pad_T = target_len - T
    for k, v in seq.items():
        pad_shape = (pad_T,) + tuple(v.shape[1:])
        padded[k] = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)], dim=0)
    return padded, valid


# ---------------------------------------------------------------------------
# Cache format + builder (sequence cache)
# ---------------------------------------------------------------------------

def _cache_params_dict(
    *,
    stride: int,
    folder_len: int,
    require_cust_gt0: bool,
    seq_len: int,
) -> Dict[str, Any]:
    return {
        "format": "critic_rl_td_lambda_v1",
        "stride": int(stride),
        "folder_len": int(folder_len),
        "require_cust_gt0": bool(require_cust_gt0),
        "seq_len": int(seq_len),
        "reward": "hp_delta_diff",
        "target": "td_lambda_bootstrap_in_train",
    }


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def build_rl_sequence_cache(
    *,
    dataset_dir: str,
    cache_dir: str,
    stride: int,
    folder_len: int,
    seq_len: int,
    require_cust_gt0: bool,
    overwrite: bool = False,
) -> None:
    """
    Precompute tensorized *sequences* + rewards into cache_dir.

    Produces one .pt per replay:
      {
        "replay": str,
        "params": {...},
        "start_frame": int64 [N],
        "r": float32 [N, T],          # per-step rewards aligned to states in sequence (last typically 0)
        "done": bool   [N, T],        # done True at terminal state positions (end of replay)
        "valid": bool  [N, T],        # valid True for real steps, False for padding
        "x": {k: tensor [N, T, ...]}
      }

    Targets are computed during training via TD(λ) using bootstrap values.
    """
    root = Path(dataset_dir)
    out_root = Path(cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stride = int(max(1, stride))
    folder_len = int(folder_len)
    seq_len = int(max(2, seq_len))  # at least 2 steps to have transitions

    params = _cache_params_dict(
        stride=stride,
        folder_len=folder_len,
        require_cust_gt0=require_cust_gt0,
        seq_len=seq_len,
    )

    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "actions.jsonl").exists()]
    if not dirs:
        raise RuntimeError(f"No valid replay folders found under {root}")

    print(f"[cache] dir={out_root}", flush=True)
    print(f"[cache] params={params}", flush=True)
    print(f"[cache] found replay dirs={len(dirs)}", flush=True)

    t0_all = time.time()

    replay_bar = tqdm(
        dirs,
        desc="cache: replays",
        unit="replay",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.2,
    )

    for child in replay_bar:
        name = child.name
        actions_path = child / "actions.jsonl"
        static = _read_json(child / "static_data.json") or {}

        cache_path = out_root / f"{name}.pt"

        if cache_path.exists() and not overwrite:
            try:
                ck = torch.load(cache_path, map_location="cpu")
                if isinstance(ck, dict) and ck.get("params") == params:
                    replay_bar.set_postfix(skip="hit")
                    continue
            except Exception:
                pass

        tqdm.write(f"[cache] {name}: reading actions.jsonl ...", file=sys.stdout)
        frames = _read_actions_jsonl(actions_path)
        if not frames:
            tqdm.write(f"[cache] {name}: no frames; skipping", file=sys.stdout)
            continue

        tqdm.write(f"[cache] {name}: compute_derived (frames={len(frames)}) ...", file=sys.stdout)
        derived = compute_derived(frames, static)

        hp_delta = _hp_delta_series(frames)
        rewards = _hp_delta_rewards(hp_delta)  # len == n_frames

        # start indices on raw frame timeline
        starts: List[int] = []
        for fi in range(0, len(frames), stride):
            if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                continue
            starts.append(fi)

        if not starts:
            payload = {
                "replay": name,
                "params": params,
                "start_frame": torch.empty((0,), dtype=torch.int64),
                "r": torch.empty((0, seq_len), dtype=torch.float32),
                "done": torch.empty((0, seq_len), dtype=torch.bool),
                "valid": torch.empty((0, seq_len), dtype=torch.bool),
                "x": {},
            }
            _atomic_torch_save(payload, cache_path)
            tqdm.write(f"[cache] {name}: 0 samples after filter; wrote empty cache", file=sys.stdout)
            continue

        xs: Dict[str, List[torch.Tensor]] = {}
        rs: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        valids: List[torch.Tensor] = []

        sample_bar = tqdm(
            starts,
            desc=f"  {name}: seqs",
            unit="seq",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.2,
        )

        n_frames = len(frames)

        for start in sample_bar:
            # build a sequence of raw indices: start + k*stride
            idxs: List[int] = []
            for k in range(seq_len):
                idxs.append(start + k * stride)

            per_x: List[Dict[str, torch.Tensor]] = []
            per_r: List[float] = []
            per_done: List[bool] = []

            for raw_i in idxs:
                if raw_i >= n_frames:
                    break
                di = min(raw_i, len(derived) - 1) if derived else 0
                per_x.append(
                    tensorize_frame(frames[raw_i], derived[di] if derived else {}, static, folder_len=folder_len, device=None)
                )
                per_r.append(float(rewards[raw_i]))
                # terminal: if this raw_i is last frame in replay
                per_done.append(bool(raw_i >= (n_frames - 1)))

            if not per_x:
                continue

            seq = _stack_time(per_x)  # (T, ...)
            seq, valid = _pad_sequence_time(seq, target_len=seq_len)

            # r/done also padded to seq_len
            r_t = torch.zeros((seq_len,), dtype=torch.float32)
            d_t = torch.zeros((seq_len,), dtype=torch.bool)

            for i in range(min(len(per_r), seq_len)):
                r_t[i] = float(per_r[i])
            for i in range(min(len(per_done), seq_len)):
                d_t[i] = bool(per_done[i])

            # if we padded due to exceeding replay end, mark done at last valid index
            if valid.any():
                last_valid = int(valid.nonzero(as_tuple=False)[-1].item())
                if last_valid >= 0:
                    d_t[last_valid] = True

            for k, v in seq.items():
                xs.setdefault(k, []).append(v)
            rs.append(r_t)
            dones.append(d_t)
            valids.append(valid)

        if not rs:
            payload = {
                "replay": name,
                "params": params,
                "start_frame": torch.empty((0,), dtype=torch.int64),
                "r": torch.empty((0, seq_len), dtype=torch.float32),
                "done": torch.empty((0, seq_len), dtype=torch.bool),
                "valid": torch.empty((0, seq_len), dtype=torch.bool),
                "x": {},
            }
            _atomic_torch_save(payload, cache_path)
            tqdm.write(f"[cache] {name}: 0 sequences built; wrote empty cache", file=sys.stdout)
            continue

        x_stacked: Dict[str, torch.Tensor] = {k: torch.stack(vs, dim=0) for k, vs in xs.items()}  # [N,T,...]
        payload = {
            "replay": name,
            "params": params,
            "start_frame": torch.tensor(starts[: x_stacked[next(iter(x_stacked))].shape[0]], dtype=torch.int64),
            "r": torch.stack(rs, dim=0),       # [N,T]
            "done": torch.stack(dones, dim=0), # [N,T]
            "valid": torch.stack(valids, dim=0),  # [N,T]
            "x": x_stacked,
        }

        _atomic_torch_save(payload, cache_path)
        replay_bar.set_postfix(seqs=int(payload["r"].shape[0]))

    replay_bar.close()

    manifest = {
        "params": params,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "num_replay_dirs": len(dirs),
        "built_at_unix": time.time(),
        "build_seconds": time.time() - t0_all,
    }
    _atomic_torch_save(manifest, out_root / "_manifest.pt")

    print(f"[cache] done in {time.time()-t0_all:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CriticRLTDDataset(Dataset):
    """
    JSONL-backed dataset yielding short sequences for TD(λ) training.

    Returns:
      x_seq: Dict[str, Tensor] where each is (T, ...)
      r:     Tensor (T,) float32
      done:  Tensor (T,) bool
      valid: Tensor (T,) bool
    """
    def __init__(
        self,
        root_dir: str,
        *,
        stride: int = 6,
        seq_len: int = 16,
        max_samples: Optional[int] = None,
        cache_replays: int = 2,
        folder_len: int = 30,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        require_cust_gt0: bool = True,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.stride = int(max(1, stride))
        self.seq_len = int(max(2, seq_len))
        self.max_samples = max_samples
        self.folder_len = int(folder_len)
        self.cache = _ReplayCache(max_replays=cache_replays)
        self.require_cust_gt0 = bool(require_cust_gt0)

        metas = self._scan_replays()
        metas = self._split(metas, split=split, val_ratio=val_ratio, seed=seed)
        self.metas = metas

        # index is (replay_idx, start_frame)
        self.index: List[Tuple[int, int]] = []
        for r_i, m in enumerate(self.metas):
            frames = _read_actions_jsonl(m.path / "actions.jsonl")
            if not frames:
                continue
            for fi in range(0, len(frames), self.stride):
                if self.require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                    continue
                self.index.append((r_i, fi))
                if self.max_samples is not None and len(self.index) >= self.max_samples:
                    break
            if self.max_samples is not None and len(self.index) >= self.max_samples:
                break

        if not self.index:
            raise RuntimeError(
                "No sequences found after filtering. "
                "If this is unexpected, set require_cust_gt0=False or check cust_gauge values."
            )

    def _scan_replays(self) -> List[ReplayMeta]:
        metas: List[ReplayMeta] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            actions_path = child / "actions.jsonl"
            if not actions_path.exists():
                continue

            static = _read_json(child / "static_data.json") or {}
            frames = _read_actions_jsonl(actions_path)
            if not frames:
                continue

            metas.append(
                ReplayMeta(
                    name=child.name,
                    path=child,
                    n_frames=len(frames),
                    static=static,
                )
            )
        if not metas:
            raise RuntimeError(f"No valid replays found under {self.root}")
        return metas

    @staticmethod
    def _split(metas: List[ReplayMeta], *, split: str, val_ratio: float, seed: int) -> List[ReplayMeta]:
        rng = __import__("random").Random(seed)
        names = [m.name for m in metas]
        rng.shuffle(names)
        n_val = max(1, int(len(names) * float(val_ratio)))
        val_set = set(names[:n_val])
        if split == "val":
            return [m for m in metas if m.name in val_set]
        return [m for m in metas if m.name not in val_set]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        r_i, start = self.index[idx]
        meta = self.metas[r_i]
        frames, derived, rewards = self.cache.get(meta)

        if not frames or not rewards:
            raise RuntimeError(f"Replay {meta.name} unexpectedly empty after load")

        n_frames = len(frames)

        per_x: List[Dict[str, torch.Tensor]] = []
        per_r: List[float] = []
        per_done: List[bool] = []

        for k in range(self.seq_len):
            raw_i = start + k * self.stride
            if raw_i >= n_frames:
                break
            di = min(raw_i, len(derived) - 1) if derived else 0
            per_x.append(
                tensorize_frame(frames[raw_i], derived[di] if derived else {}, meta.static, folder_len=self.folder_len, device=None)
            )
            per_r.append(float(rewards[raw_i]))
            per_done.append(bool(raw_i >= (n_frames - 1)))

        if not per_x:
            # Should not happen due to index build, but be safe.
            empty_x = tensorize_frame(frames[min(start, n_frames - 1)], derived[min(start, len(derived) - 1)] if derived else {}, meta.static, folder_len=self.folder_len)
            seq = _stack_time([empty_x])
            seq, valid = _pad_sequence_time(seq, target_len=self.seq_len)
            r_t = torch.zeros((self.seq_len,), dtype=torch.float32)
            d_t = torch.zeros((self.seq_len,), dtype=torch.bool)
            d_t[0] = True
            return seq, r_t, d_t, valid

        seq = _stack_time(per_x)
        seq, valid = _pad_sequence_time(seq, target_len=self.seq_len)

        r_t = torch.zeros((self.seq_len,), dtype=torch.float32)
        d_t = torch.zeros((self.seq_len,), dtype=torch.bool)

        for i in range(min(len(per_r), self.seq_len)):
            r_t[i] = float(per_r[i])
        for i in range(min(len(per_done), self.seq_len)):
            d_t[i] = bool(per_done[i])

        if valid.any():
            last_valid = int(valid.nonzero(as_tuple=False)[-1].item())
            d_t[last_valid] = True

        return seq, r_t, d_t, valid


class CachedCriticRLTDDataset(Dataset):
    """
    Cache-backed dataset for TD(λ) training.

    Each replay file contains:
      {
        "replay": str,
        "params": {...},
        "start_frame": int64 [N],
        "r": float32 [N, T],
        "done": bool   [N, T],
        "valid": bool  [N, T],
        "x": {k: tensor [N, T, ...]}
      }
    """
    def __init__(
        self,
        dataset_dir: str,
        *,
        cache_dir: str,
        stride: int,
        folder_len: int,
        seq_len: int,
        require_cust_gt0: bool,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.params = _cache_params_dict(
            stride=int(stride),
            folder_len=int(folder_len),
            require_cust_gt0=bool(require_cust_gt0),
            seq_len=int(seq_len),
        )
        self.max_samples = max_samples

        all_files = sorted([p for p in self.cache_dir.glob("*.pt") if p.name != "_manifest.pt"])
        if not all_files:
            raise RuntimeError(f"No cache files found under {self.cache_dir}")

        names = [p.stem for p in all_files]
        rng = __import__("random").Random(seed)
        rng.shuffle(names)
        n_val = max(1, int(len(names) * float(val_ratio)))
        val_set = set(names[:n_val])

        keep = val_set if split == "val" else set([n for n in names if n not in val_set])
        self.files: List[Path] = [p for p in all_files if p.stem in keep]
        if not self.files:
            raise RuntimeError(f"Cache split '{split}' is empty under {self.cache_dir}")

        self._index: List[Tuple[int, int]] = []

        for file_idx, p in enumerate(self.files):
            ck = torch.load(p, map_location="cpu")
            if not isinstance(ck, dict):
                raise RuntimeError(f"Bad cache file (not dict): {p}")

            found = ck.get("params", None)
            if found != self.params:
                raise RuntimeError(
                    f"Cache params mismatch for {p.name}.\n"
                    f"Expected: {self.params}\n"
                    f"Found:    {found}\n"
                    f"Rebuild cache with --build_cache --rebuild_cache."
                )

            x = ck.get("x") or {}
            r = ck.get("r")
            done = ck.get("done")
            valid = ck.get("valid")

            if not isinstance(x, dict) or not x:
                continue  # empty replay after filtering
            if not isinstance(r, torch.Tensor) or r.ndim != 2:
                raise RuntimeError(f"Bad cache r in {p.name}: expected 2D tensor [N,T]")
            if not isinstance(done, torch.Tensor) or done.ndim != 2:
                raise RuntimeError(f"Bad cache done in {p.name}: expected 2D tensor [N,T]")
            if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
                raise RuntimeError(f"Bad cache valid in {p.name}: expected 2D tensor [N,T]")

            first = next(iter(x.values()))
            if not isinstance(first, torch.Tensor) or first.ndim < 2:
                raise RuntimeError(f"Bad cache tensors in {p.name}: expected [N,T,...]")
            n_rows = int(first.shape[0])
            if int(r.shape[0]) != n_rows or int(done.shape[0]) != n_rows or int(valid.shape[0]) != n_rows:
                raise RuntimeError(f"Cache row mismatch in {p.name}: x has {n_rows}, r/done/valid mismatch")

            for row in range(n_rows):
                self._index.append((file_idx, row))
                if self.max_samples is not None and len(self._index) >= self.max_samples:
                    break
            if self.max_samples is not None and len(self._index) >= self.max_samples:
                break

        if not self._index:
            raise RuntimeError(
                "Cached dataset has 0 samples. Likely everything got filtered out "
                "(require_cust_gt0/stride), or cache build produced empty tensors."
            )

        self._lru_key: Optional[int] = None
        self._lru_obj: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self._index)

    def _load_file(self, file_idx: int) -> Dict[str, Any]:
        if self._lru_key == file_idx and self._lru_obj is not None:
            return self._lru_obj
        p = self.files[file_idx]
        ck = torch.load(p, map_location="cpu")
        if not isinstance(ck, dict):
            raise RuntimeError(f"Bad cache file (not dict): {p}")
        self._lru_key = file_idx
        self._lru_obj = ck
        return ck

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, row_idx = self._index[idx]
        ck = self._load_file(file_idx)

        x_all: Dict[str, torch.Tensor] = ck["x"]
        r_all: torch.Tensor = ck["r"]
        d_all: torch.Tensor = ck["done"]
        v_all: torch.Tensor = ck["valid"]

        x: Dict[str, torch.Tensor] = {k: v[row_idx] for k, v in x_all.items()}  # [T,...]
        r = r_all[row_idx].clone()
        done = d_all[row_idx].clone()
        valid = v_all[row_idx].clone()
        return x, r, done, valid


def collate_batch(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, rs, ds, vs = zip(*batch)
    out: Dict[str, torch.Tensor] = {}
    for k in xs[0].keys():
        out[k] = torch.stack([x[k] for x in xs], dim=0)  # [B,T,...]
    r = torch.stack(list(rs), dim=0)     # [B,T]
    done = torch.stack(list(ds), dim=0)  # [B,T]
    valid = torch.stack(list(vs), dim=0) # [B,T]
    return out, r, done, valid


__all__ = [
    "CriticRLTDDataset",
    "CachedCriticRLTDDataset",
    "build_rl_sequence_cache",
    "collate_batch",
]
