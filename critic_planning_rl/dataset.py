# planning_critic/dataset.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _pad_bool_mask(vals: Any, n: int) -> torch.Tensor:
    lst = vals if isinstance(vals, list) else []
    out = [1.0 if bool(x) else 0.0 for x in lst]
    if len(out) < n:
        out.extend([0.0] * (n - len(out)))
    return torch.tensor(out[:n], dtype=torch.float32)


def _extract_visible_hand(
    draw_id_raw: Any,
    draw_code_raw: Any,
    vis_raw: Any,
    *,
    max_slots: int = 10,
) -> List[Tuple[int, int]]:
    ids = draw_id_raw if isinstance(draw_id_raw, list) else []
    codes = draw_code_raw if isinstance(draw_code_raw, list) else []
    vis = vis_raw if isinstance(vis_raw, list) else []

    out: List[Tuple[int, int]] = []
    for i in range(max_slots):
        v = _as_float(vis[i], 0.0) if i < len(vis) else 0.0
        if v > 0.5:
            cid = _as_int(ids[i], 0) if i < len(ids) else 0
            ccode = _as_int(codes[i], 0) if i < len(codes) else 0
            out.append((max(0, cid), _clamp_int(max(0, ccode), 0, 127)))
    return out


def _rebuild_padded_hand(
    visible: List[Tuple[int, int]],
    *,
    max_slots: int = 10,
) -> Tuple[List[int], List[int]]:
    draw_id: List[int] = []
    draw_code: List[int] = []
    for i in range(max_slots):
        if i < len(visible):
            draw_id.append(max(0, _as_int(visible[i][0], 0)))
            draw_code.append(_clamp_int(_as_int(visible[i][1], 0), 0, 127))
        else:
            draw_id.append(0)
            draw_code.append(0)
    return draw_id, draw_code


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    # scalars normalization
    hp_scale: float = 1000.0
    turn_scale: float = 50.0

    # fixed sizes
    n_grid: int = 18
    n_folder: int = 30
    n_cross: int = 11
    n_held: int = 5
    n_draw: int = 10
    n_selected: int = 5

    # clamping (defensive)
    tile_max: int = 31
    owner_max: int = 2
    code_max: int = 127
    cross_max: int = 63


class PlanningDataset(Dataset):
    """
    Produces one row for PlanningCritic.

    IMPORTANT: trains on WEIGHTED target, not raw.
      - target_key defaults to 'net_yield_weighted'
      - if missing, falls back to 'net_yield' (but you should fix generation to include weighted)
    """

    def __init__(
        self,
        jsonl_path: str,
        *,
        max_samples: Optional[int] = None,
        shuffle_hand: bool = False,
        target_key: str = "net_yield_weighted",
        target_transform: str = "signed_log1p",
        cfg: DatasetConfig = DatasetConfig(),
    ):
        self.path = Path(jsonl_path)
        self.shuffle_hand = bool(shuffle_hand)
        self.target_key = str(target_key)
        self.target_transform = str(target_transform)
        self.cfg = cfg

        self.data: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    self.data.append(row)
                if max_samples is not None and len(self.data) >= int(max_samples):
                    break

    def __len__(self) -> int:
        return len(self.data)

    def _get_target_raw(self, row: Dict[str, Any]) -> float:
        # Prefer weighted, fall back (but this should be rare once generator is updated)
        if self.target_key in row:
            return _as_float(row.get(self.target_key), 0.0)
        if self.target_key != "net_yield" and "net_yield" in row:
            return _as_float(row.get("net_yield"), 0.0)
        return 0.0

    def _transform_target(self, y: float) -> float:
        mode = self.target_transform.strip().lower()
        if mode == "none":
            return float(y)
        if mode == "signed_log1p":
            return _signed_log1p(float(y))
        raise ValueError(f"Unknown target_transform: {self.target_transform}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        c = self.cfg

        # --- SCALARS ---
        p_hp = _as_float(row.get("p_hp_open", 0.0), 0.0) / c.hp_scale
        e_hp = _as_float(row.get("e_hp_open", 0.0), 0.0) / c.hp_scale
        turn = _as_float(row.get("turn_index_open", 0.0), 0.0) / c.turn_scale

        # --- GRID ---
        grid_tile = torch.tensor(row.get("grid_tile_open", [0] * c.n_grid), dtype=torch.long)[: c.n_grid]
        grid_owner = torch.tensor(row.get("grid_owner_open", [2] * c.n_grid), dtype=torch.long)[: c.n_grid]
        grid_tile = grid_tile.clamp(0, c.tile_max)
        grid_owner = grid_owner.clamp(0, c.owner_max)

        # --- MASKS ---
        folder_p = _pad_bool_mask(row.get("folder_used_mask_p_open", []), c.n_folder)
        folder_e = _pad_bool_mask(row.get("folder_used_mask_e_open", []), c.n_folder)
        cross_p = _pad_bool_mask(row.get("used_cross_mask_p_open", []), c.n_cross)
        cross_e = _pad_bool_mask(row.get("used_cross_mask_e_open", []), c.n_cross)

        # --- HELD (context) ---
        held_id = torch.tensor(row.get("held_before_id", [0] * c.n_held), dtype=torch.long)[: c.n_held].clamp(min=0)
        held_code = (
            torch.tensor(row.get("held_before_code", [0] * c.n_held), dtype=torch.long)[: c.n_held].clamp(0, c.code_max)
        )

        # --- DRAW HAND (visible only, optionally shuffled) ---
        visible = _extract_visible_hand(
            row.get("window_hand_id", [0] * c.n_draw),
            row.get("window_hand_code", [0] * c.n_draw),
            row.get("window_hand_vis", [0.0] * c.n_draw),
            max_slots=c.n_draw,
        )
        if self.shuffle_hand:
            random.shuffle(visible)
        draw_id_list, draw_code_list = _rebuild_padded_hand(visible, max_slots=c.n_draw)

        draw_id = torch.tensor(draw_id_list, dtype=torch.long).clamp(min=0)
        draw_code = torch.tensor(draw_code_list, dtype=torch.long).clamp(0, c.code_max)

        # --- ACTION (the decision) ---
        sel_id = torch.tensor(row.get("selected_chips_id", [0] * c.n_selected), dtype=torch.long)[: c.n_selected].clamp(min=0)
        sel_code = (
            torch.tensor(row.get("selected_chips_code", [0] * c.n_selected), dtype=torch.long)[: c.n_selected].clamp(0, c.code_max)
        )

        sel_cross_val = _as_int(row.get("selected_cross", 0), 0)
        sel_cross = torch.tensor(_clamp_int(sel_cross_val, 0, c.cross_max), dtype=torch.long)

        sel_beast = torch.tensor(1 if bool(row.get("beast_selected", False)) else 0, dtype=torch.long)

        # --- TARGET (WEIGHTED) ---
        target_raw = self._get_target_raw(row)
        target = self._transform_target(target_raw)

        out: Dict[str, torch.Tensor] = {
            "p_hp": torch.tensor(p_hp, dtype=torch.float32),
            "e_hp": torch.tensor(e_hp, dtype=torch.float32),
            "turn_idx": torch.tensor(turn, dtype=torch.float32),
            "grid_tile": grid_tile,
            "grid_owner": grid_owner,
            "folder_p": folder_p,
            "folder_e": folder_e,
            "cross_hist_p": cross_p,
            "cross_hist_e": cross_e,
            "held_id": held_id,
            "held_code": held_code,
            "draw_id": draw_id,
            "draw_code": draw_code,
            "sel_id": sel_id,
            "sel_code": sel_code,
            "sel_cross": sel_cross,
            "sel_beast": sel_beast,
            "target": torch.tensor(float(target), dtype=torch.float32),
            # Keep raw around for debugging/metrics if you want it in train.py
            "target_raw": torch.tensor(float(target_raw), dtype=torch.float32),
        }
        return out


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # stacks everything; assumes all tensors are same shape per key
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
