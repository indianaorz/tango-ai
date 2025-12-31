# critic_minimal/feature_audit.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import torch

from critic_minimal.dataset import StreamingMinimalQDataset
from critic_minimal.features import BUTTON_KEYS, ACTION_DIM


# -----------------------------
# Running stats (Welford)
# -----------------------------
@dataclass
class RunningMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    mn: float = float("inf")
    mx: float = float("-inf")

    def add(self, x: float) -> None:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return
        self.n += 1
        dx = x - self.mean
        self.mean += dx / float(self.n)
        dx2 = x - self.mean
        self.m2 += dx * dx2
        self.mn = min(self.mn, x)
        self.mx = max(self.mx, x)

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / float(self.n))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": int(self.n),
            "min": float(self.mn if self.n else 0.0),
            "max": float(self.mx if self.n else 0.0),
            "mean": float(self.mean if self.n else 0.0),
            "std": float(self.std()),
        }


@dataclass
class ReferenceStats:
    # scalars: [5]
    scalars: List[RunningMoments]
    # discrete sets/counters
    p_emo_seen: Counter
    e_emo_seen: Counter
    chip_seen: Counter
    p_idx_seen: Counter
    e_idx_seen: Counter
    rel_seen: Counter
    # grid: per-cell tile/owner ranges
    grid_tile_min: int
    grid_tile_max: int
    grid_owner_min: int
    grid_owner_max: int
    # action: press rates per key at last token
    action_press_counts: List[int]
    action_samples: int

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "scalars": [m.as_dict() for m in self.scalars],
            "p_emo_top": self.p_emo_seen.most_common(20),
            "e_emo_top": self.e_emo_seen.most_common(20),
            "chip_top": self.chip_seen.most_common(20),
            "p_idx_top": self.p_idx_seen.most_common(18),
            "e_idx_top": self.e_idx_seen.most_common(18),
            "rel_top": self.rel_seen.most_common(20),
            "grid_tile_range": [int(self.grid_tile_min), int(self.grid_tile_max)],
            "grid_owner_range": [int(self.grid_owner_min), int(self.grid_owner_max)],
            "action_press_rate_last": {
                BUTTON_KEYS[i]: (float(self.action_press_counts[i]) / float(self.action_samples or 1))
                for i in range(min(ACTION_DIM, len(self.action_press_counts)))
            },
            "action_samples": int(self.action_samples),
        }


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _pick_last_valid_t(valid_1d: torch.Tensor) -> Optional[int]:
    # valid_1d: [T] bool
    if not torch.is_tensor(valid_1d) or valid_1d.ndim != 1:
        return None
    if valid_1d.numel() == 0:
        return None
    v = valid_1d.to(torch.bool)
    nz = torch.nonzero(v, as_tuple=False)
    if nz.numel() == 0:
        return None
    return int(nz[-1].item())


def build_reference_from_cache(
    cache_dir: str,
    *,
    num_sequences: int = 2048,
    seed: int = 1337,
) -> ReferenceStats:
    """
    Builds a lightweight reference summary from your cached TRAINING tensors.
    Compares *inference tensors* against what training actually contained.

    Note: uses split="train" with val_ratio=0 to avoid accidental leakage.
    """
    ds = StreamingMinimalQDataset(
        cache_dir=cache_dir,
        split="train",
        val_ratio=0.0,
        seed=int(seed),
        max_sequences=int(num_sequences),
    )

    scal = [RunningMoments() for _ in range(5)]
    p_emo = Counter()
    e_emo = Counter()
    chip = Counter()
    pidx = Counter()
    eidx = Counter()
    rel = Counter()

    tile_min = 1 << 30
    tile_max = -(1 << 30)
    own_min = 1 << 30
    own_max = -(1 << 30)

    action_press = [0 for _ in range(ACTION_DIM)]
    action_samples = 0

    for x, _y, valid in ds:
        # x: dict [T,...] (not batched)
        # valid: [T]
        t_last = _pick_last_valid_t(valid)
        if t_last is None:
            continue

        # scalars [T,5]
        sc = x.get("scalars", None)
        if torch.is_tensor(sc) and sc.ndim == 2 and sc.shape[1] == 5:
            for i in range(5):
                scal[i].add(float(sc[t_last, i].item()))

        def _count_1d(name: str, ctr: Counter) -> None:
            v = x.get(name, None)
            if torch.is_tensor(v) and v.ndim == 1:
                ctr[_safe_int(v[t_last].item(), 0)] += 1

        _count_1d("p_emotion_id", p_emo)
        _count_1d("e_emotion_id", e_emo)
        _count_1d("player_chip", chip)
        _count_1d("p_grid_idx", pidx)
        _count_1d("e_grid_idx", eidx)
        _count_1d("rel_pe_idx", rel)

        gt = x.get("grid_tile", None)
        go = x.get("grid_owner", None)
        if torch.is_tensor(gt) and gt.ndim == 2 and gt.shape[1] == 18:
            t = gt[t_last].to(torch.int64)
            tile_min = min(tile_min, int(t.min().item()))
            tile_max = max(tile_max, int(t.max().item()))
        if torch.is_tensor(go) and go.ndim == 2 and go.shape[1] == 18:
            o = go[t_last].to(torch.int64)
            own_min = min(own_min, int(o.min().item()))
            own_max = max(own_max, int(o.max().item()))

        act = x.get("action", None)
        if torch.is_tensor(act) and act.ndim == 2 and act.shape[1] == ACTION_DIM:
            a = act[t_last].to(torch.float32)
            for i in range(ACTION_DIM):
                if float(a[i].item()) > 0.5:
                    action_press[i] += 1
            action_samples += 1

    if tile_min == (1 << 30):
        tile_min, tile_max = 0, 0
    if own_min == (1 << 30):
        own_min, own_max = 0, 0

    return ReferenceStats(
        scalars=scal,
        p_emo_seen=p_emo,
        e_emo_seen=e_emo,
        chip_seen=chip,
        p_idx_seen=pidx,
        e_idx_seen=eidx,
        rel_seen=rel,
        grid_tile_min=int(tile_min),
        grid_tile_max=int(tile_max),
        grid_owner_min=int(own_min),
        grid_owner_max=int(own_max),
        action_press_counts=action_press,
        action_samples=int(action_samples),
    )


def snapshot_last_token_from_xb(xb_base: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    xb_base is your inference "base" dict: [1,T,...] on CPU.
    Returns a compact JSON-able snapshot for T-1.
    """
    T = int(xb_base["scalars"].shape[1])
    t = T - 1

    def _t1(name: str) -> int:
        v = xb_base.get(name)
        if not torch.is_tensor(v):
            return 0
        return int(v[0, t].item())

    scal = xb_base["scalars"][0, t].to(torch.float32).tolist()
    act = xb_base["action"][0, t].to(torch.float32).tolist()

    # grid summaries (not full 18 by default; enough to detect mapping issues)
    gt = xb_base["grid_tile"][0, t].to(torch.int64)
    go = xb_base["grid_owner"][0, t].to(torch.int64)

    return {
        "t": int(t),
        "scalars_norm": [float(round(x, 6)) for x in scal],  # [p_hp,e_hp,p_chg,e_chg,cust] already normalized
        "p_emotion_id": _t1("p_emotion_id"),
        "e_emotion_id": _t1("e_emotion_id"),
        "player_chip": _t1("player_chip"),
        "p_grid_idx": _t1("p_grid_idx"),
        "e_grid_idx": _t1("e_grid_idx"),
        "rel_pe_idx": _t1("rel_pe_idx"),
        "grid_tile_minmax": [int(gt.min().item()), int(gt.max().item())],
        "grid_owner_counts": {
            "0": int((go == 0).sum().item()),
            "1": int((go == 1).sum().item()),
            "2": int((go == 2).sum().item()),
        },
        "action": {BUTTON_KEYS[i]: float(round(act[i], 3)) for i in range(min(ACTION_DIM, len(act)))},
    }


def compare_snapshot_to_reference(
    snap: Dict[str, Any],
    ref: ReferenceStats,
    *,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Returns structured warnings + quick diffs.
    This is what you want in logs to catch mismapped fields.
    """
    warnings: List[str] = []

    scal = snap.get("scalars_norm", [])
    if isinstance(scal, list) and len(scal) == 5:
        for i, name in enumerate(["p_hp", "e_hp", "p_charge", "e_charge", "cust_gauge"]):
            v = float(scal[i])
            m = ref.scalars[i]
            if m.n > 0:
                if v < (m.mn - eps) or v > (m.mx + eps):
                    warnings.append(f"scalar[{name}]={v:.6f} outside train_range=[{m.mn:.6f},{m.mx:.6f}]")
            # extra “smells wrong” checks that commonly happen with BN6:
            if name in ("p_hp", "e_hp") and (v < -0.01 or v > 1.5):
                warnings.append(f"scalar[{name}]={v:.6f} looks unnormalized (expected ~0..1)")
            if name == "cust_gauge" and (v > 1.25):  # common if raw is 0..255 but you divide by 100
                warnings.append("cust_gauge_norm > 1.25 (possible raw 0..255 mapped as /100)")

    def _seen_check(field: str, ctr: Counter) -> None:
        v = int(snap.get(field, 0))
        if sum(ctr.values()) > 0 and ctr.get(v, 0) == 0:
            warnings.append(f"{field}={v} never seen in sampled training cache")

    _seen_check("p_emotion_id", ref.p_emo_seen)
    _seen_check("e_emotion_id", ref.e_emo_seen)
    _seen_check("player_chip", ref.chip_seen)
    _seen_check("p_grid_idx", ref.p_idx_seen)
    _seen_check("e_grid_idx", ref.e_idx_seen)
    _seen_check("rel_pe_idx", ref.rel_seen)

    # hard validity bounds
    pg = int(snap.get("p_grid_idx", 0))
    eg = int(snap.get("e_grid_idx", 0))
    rr = int(snap.get("rel_pe_idx", 0))
    if not (0 <= pg <= 17):
        warnings.append(f"p_grid_idx={pg} outside [0..17]")
    if not (0 <= eg <= 17):
        warnings.append(f"e_grid_idx={eg} outside [0..17]")
    if not (0 <= rr <= 54):
        warnings.append(f"rel_pe_idx={rr} outside [0..54]")

    gt_minmax = snap.get("grid_tile_minmax", [0, 0])
    if isinstance(gt_minmax, list) and len(gt_minmax) == 2:
        mn, mx = int(gt_minmax[0]), int(gt_minmax[1])
        if mx > ref.grid_tile_max or mn < ref.grid_tile_min:
            warnings.append(
                f"grid_tile_minmax=[{mn},{mx}] outside train_range=[{ref.grid_tile_min},{ref.grid_tile_max}]"
            )

    goc = snap.get("grid_owner_counts", {})
    if isinstance(goc, dict):
        # if owner is all "2" constantly, you likely failed to map ownership
        if int(goc.get("0", 0)) == 0 and int(goc.get("1", 0)) == 0 and int(goc.get("2", 0)) == 18:
            warnings.append("grid_owner is all 2 (unknown) — likely owner mapping missing/broken")

    return {
        "warnings": warnings,
        "train_scalar_stats": [m.as_dict() for m in ref.scalars],
        "train_grid_tile_range": [ref.grid_tile_min, ref.grid_tile_max],
        "train_grid_owner_range": [ref.grid_owner_min, ref.grid_owner_max],
    }


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
