# critic/dataset.py
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
# RL target: hp_delta rewards and discounted returns
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


def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    G_t = r_t + gamma * G_{t+1}
    """
    if not rewards:
        return []
    gg = float(gamma)
    if not (0.0 < gg <= 1.0):
        raise ValueError(f"gamma must be in (0,1], got {gamma}")
    g = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gg * running
        g[t] = running
    return g


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
    def __init__(self, *, max_replays: int = 3, gamma: float = 0.999):
        self.max_replays = int(max_replays)
        self.gamma = float(gamma)
        self._order: List[str] = []
        self._frames: Dict[str, List[Dict[str, Any]]] = {}
        self._derived: Dict[str, List[Dict[str, Any]]] = {}
        self._returns: Dict[str, List[float]] = {}

    def get(self, meta: ReplayMeta) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
        key = meta.name
        if key in self._frames and key in self._derived and key in self._returns:
            self._touch(key)
            return self._frames[key], self._derived[key], self._returns[key]

        frames = _read_actions_jsonl(meta.path / "actions.jsonl")
        derived = compute_derived(frames, meta.static)

        hp_delta = _hp_delta_series(frames)
        rewards = _hp_delta_rewards(hp_delta)
        returns = _discounted_returns(rewards, gamma=self.gamma)

        self._frames[key] = frames
        self._derived[key] = derived
        self._returns[key] = returns
        self._touch(key)
        self._evict_if_needed()
        return frames, derived, returns

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.max_replays:
            victim = self._order.pop(0)
            self._frames.pop(victim, None)
            self._derived.pop(victim, None)
            self._returns.pop(victim, None)


# ---------------------------------------------------------------------------
# Tensorization
# ---------------------------------------------------------------------------

def _pad_ids(ids: List[int], n: int) -> Tuple[List[int], List[bool]]:
    out = ids[:n]
    mask = [True] * len(out)
    if len(out) < n:
        pad = n - len(out)
        out.extend([0] * pad)
        mask.extend([False] * pad)
    return out, mask


def tensorize_sample(
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
# Cache format + builder
# ---------------------------------------------------------------------------

def _cache_params_dict(*, stride: int, folder_len: int, require_cust_gt0: bool, gamma: float) -> Dict[str, Any]:
    return {
        "format": "critic_value_v2",
        "stride": int(stride),
        "folder_len": int(folder_len),
        "require_cust_gt0": bool(require_cust_gt0),
        "gamma": float(gamma),
        "reward": "hp_delta_diff",
        "target": "discounted_return",
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


def build_hpdelta_cache(
    *,
    dataset_dir: str,
    cache_dir: str,
    stride: int,
    folder_len: int,
    require_cust_gt0: bool,
    gamma: float,
    overwrite: bool = False,
) -> None:
    """
    Precompute tensorized inputs + PER-SAMPLE return targets per replay into cache_dir.

    Produces one .pt per replay:
      {
        "replay": str,
        "params": {...},
        "frame_idx": int64 [N],
        "y": float32 [N],             # discounted return at each selected frame
        "x": {k: tensor [N, ...]}
      }
    """
    root = Path(dataset_dir)
    out_root = Path(cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stride = int(max(1, stride))
    folder_len = int(folder_len)
    gamma = float(gamma)
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0,1], got {gamma}")

    params = _cache_params_dict(stride=stride, folder_len=folder_len, require_cust_gt0=require_cust_gt0, gamma=gamma)

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

        # If not overwriting and looks compatible, skip
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

        # compute returns for ALL frames
        hp_delta = _hp_delta_series(frames)
        rewards = _hp_delta_rewards(hp_delta)
        returns = _discounted_returns(rewards, gamma=gamma)

        tqdm.write(f"[cache] {name}: compute_derived (frames={len(frames)}) ...", file=sys.stdout)
        derived = compute_derived(frames, static)

        sel: List[int] = []
        for fi in range(0, len(frames), stride):
            if require_cust_gt0 and _as_int(frames[fi].get("cust_gauge"), 0) <= 0:
                continue
            sel.append(fi)

        if not sel:
            payload = {
                "replay": name,
                "params": params,
                "frame_idx": torch.empty((0,), dtype=torch.int64),
                "y": torch.empty((0,), dtype=torch.float32),
                "x": {},
            }
            _atomic_torch_save(payload, cache_path)
            tqdm.write(f"[cache] {name}: 0 samples after filter; wrote empty cache", file=sys.stdout)
            continue

        # tensorize and stack
        sample_bar = tqdm(
            sel,
            desc=f"  {name}: samples",
            unit="samp",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.2,
        )

        xs: Dict[str, List[torch.Tensor]] = {}
        ys: List[float] = []

        for fi in sample_bar:
            di = min(fi, len(derived) - 1) if derived else 0
            x = tensorize_sample(frames[fi], derived[di] if derived else {}, static, folder_len=folder_len, device=None)
            for k, v in x.items():
                xs.setdefault(k, []).append(v)
            ys.append(float(returns[fi]))

        x_stacked: Dict[str, torch.Tensor] = {k: torch.stack(vs, dim=0) for k, vs in xs.items()}

        payload = {
            "replay": name,
            "params": params,
            "frame_idx": torch.tensor(sel, dtype=torch.int64),
            "y": torch.tensor(ys, dtype=torch.float32),   # [N]
            "x": x_stacked,
        }

        _atomic_torch_save(payload, cache_path)
        replay_bar.set_postfix(samples=len(sel))

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

class CriticHPDeltaDataset(Dataset):
    """
    JSONL-backed dataset (on-the-fly parsing + derived + returns).
    Indexes (replay, frame_idx). Target is discounted return G_t of hp_delta advantage.

    Filtering:
      - If require_cust_gt0 True, only include frames where cust_gauge > 0.
    """
    def __init__(
        self,
        root_dir: str,
        *,
        stride: int = 6,
        gamma: float = 0.999,
        max_samples: Optional[int] = None,
        cache_replays: int = 3,
        folder_len: int = 30,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        require_cust_gt0: bool = True,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.stride = int(max(1, stride))
        self.gamma = float(gamma)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0,1], got {gamma}")
        self.max_samples = max_samples
        self.folder_len = int(folder_len)
        self.cache = _ReplayCache(max_replays=cache_replays, gamma=self.gamma)
        self.require_cust_gt0 = bool(require_cust_gt0)

        metas = self._scan_replays()
        metas = self._split(metas, split=split, val_ratio=val_ratio, seed=seed)
        self.metas = metas

        # Build index with filtering at source
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
                "No samples found after filtering. "
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

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        r_i, f_i = self.index[idx]
        meta = self.metas[r_i]
        frames, derived, returns = self.cache.get(meta)

        if not frames or not returns:
            raise RuntimeError(f"Replay {meta.name} unexpectedly empty after load")

        f_i = min(max(0, f_i), len(frames) - 1)
        d_i = min(max(0, f_i), len(derived) - 1) if derived else 0

        # Safety: if require_cust_gt0, nudge to nearest >0 frame
        if self.require_cust_gt0 and _as_int(frames[f_i].get("cust_gauge"), 0) <= 0:
            j = f_i
            while j < len(frames) and _as_int(frames[j].get("cust_gauge"), 0) <= 0:
                j += 1
            if j >= len(frames):
                j = f_i
                while j >= 0 and _as_int(frames[j].get("cust_gauge"), 0) <= 0:
                    j -= 1
            if 0 <= j < len(frames):
                f_i = j
                d_i = min(f_i, len(derived) - 1) if derived else 0

        x = tensorize_sample(frames[f_i], derived[d_i] if derived else {}, meta.static, folder_len=self.folder_len, device=None)
        y = torch.tensor(float(returns[f_i]), dtype=torch.float32)
        return x, y


class CachedCriticHPDeltaDataset(Dataset):
    """
    Cache-backed dataset. Loads pre-tensorized samples from cache_dir/*.pt.

    Each replay file contains:
      {
        "replay": str,
        "params": {...},
        "frame_idx": int64 [N],
        "y": float32 [N],             # discounted returns at each selected frame
        "x": {k: tensor [N, ...]}
      }
    """
    def __init__(
        self,
        dataset_dir: str,
        *,
        cache_dir: str,
        stride: int,
        folder_len: int,
        require_cust_gt0: bool,
        gamma: float,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1337,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.params = _cache_params_dict(stride=stride, folder_len=folder_len, require_cust_gt0=require_cust_gt0, gamma=float(gamma))
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
            y = ck.get("y")
            if not isinstance(x, dict) or not x:
                continue  # empty replay after filtering
            if not isinstance(y, torch.Tensor) or y.ndim != 1:
                raise RuntimeError(f"Bad cache y in {p.name}: expected 1D tensor")

            first = next(iter(x.values()))
            if not isinstance(first, torch.Tensor) or first.ndim < 1:
                raise RuntimeError(f"Bad cache tensors in {p.name}")
            n_rows = int(first.shape[0])
            if int(y.shape[0]) != n_rows:
                raise RuntimeError(f"Cache row mismatch in {p.name}: x has {n_rows} rows but y has {int(y.shape[0])}")

            for r in range(n_rows):
                self._index.append((file_idx, r))
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

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        file_idx, row_idx = self._index[idx]
        ck = self._load_file(file_idx)

        x_all: Dict[str, torch.Tensor] = ck["x"]
        y_all: torch.Tensor = ck["y"]

        x: Dict[str, torch.Tensor] = {k: v[row_idx] for k, v in x_all.items()}
        y_out = y_all[row_idx].clone()
        return x, y_out


def collate_batch(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    xs, ys = zip(*batch)
    out: Dict[str, torch.Tensor] = {}
    for k in xs[0].keys():
        out[k] = torch.stack([x[k] for x in xs], dim=0)
    y = torch.stack(list(ys), dim=0)
    return out, y


__all__ = [
    "CriticHPDeltaDataset",
    "CachedCriticHPDeltaDataset",
    "build_hpdelta_cache",
    "collate_batch",
]
