# viewer/critic_minimal_infer.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from critic_minimal.model import MinimalQConfig, MinimalQCritic
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
    pos_to_grid_idx,
    rel_pe_index,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _clamp_for_atanh(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # atanh blows up at +/-1; clamp tightly.
    lo = -1.0 + float(eps)
    hi = 1.0 - float(eps)
    return torch.clamp(x, lo, hi)

def _atanh(x: torch.Tensor) -> torch.Tensor:
    # torch.atanh exists, but keep compatibility safe.
    if hasattr(torch, "atanh"):
        return torch.atanh(x)
    # atanh(x) = 0.5 * ln((1+x)/(1-x))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass(frozen=True)
class MinimalCriticInferResult:
    values_by_frame: Dict[int, float]         # raw_frame_idx -> y_raw_hat
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Feature extraction (mirrors critic_minimal.precache exactly)
# ---------------------------------------------------------------------

def _extract_step_features(frames: List[Dict[str, Any]], raw_i: int) -> Dict[str, Any]:
    f = frames[raw_i]

    # Scalars
    p_hp = float(_as_int(f.get("player_health"), 0))
    e_hp = float(_as_int(f.get("enemy_health"), 0))
    p_chg = float(_as_int(f.get("player_charge"), 0))
    e_chg = float(_as_int(f.get("enemy_charge"), 0))
    cust = float(_as_int(f.get("cust_gauge"), 0))

    # Emotion ids
    p_emo = _as_int(f.get("player_game_emotion"), 0)
    e_emo = _as_int(f.get("enemy_game_emotion"), 0)

    # Grid
    grid_state = _as_list(f.get("grid_state"))
    grid_owner = _as_list(f.get("grid_owner_state"))
    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0] * (18 - min(18, len(grid_state)))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2] * (18 - min(18, len(grid_owner)))

    # Positions
    px_i, py_i = _pos_norm(f.get("player_pos"))
    ex_i, ey_i = _pos_norm(f.get("enemy_pos"))
    p_grid_idx = pos_to_grid_idx(float(px_i), float(py_i))
    e_grid_idx = pos_to_grid_idx(float(ex_i), float(ey_i))
    rel_idx = rel_pe_index(p_grid_idx, e_grid_idx)

    # Chip on deck (int bucket)
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


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

class MinimalCriticRunner:
    """
    Loads a critic_minimal checkpoint (best.pt / last.pt from critic_minimal.train)
    and runs inference on a replay's raw actions frames.

    Output is a sparse mapping raw_frame_idx -> y_raw_hat.
    You can densify + smooth in the viewer using your existing masked helpers.
    """

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: str = "cuda",
        use_amp: bool = True,
        batch_seqs: int = 256,
    ) -> None:
        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.batch_seqs = int(batch_seqs)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Invalid checkpoint (not a dict): {self.ckpt_path}")

        cfg_d = ckpt.get("cfg", None)
        if not isinstance(cfg_d, dict):
            raise RuntimeError("Checkpoint missing cfg dict")

        # Schema is persisted by train.py; we validate lightly.
        schema_d = ckpt.get("schema", None)
        if not isinstance(schema_d, dict):
            raise RuntimeError("Checkpoint missing schema dict")

        self.seq_len = int(schema_d.get("seq_len", cfg_d.get("max_seq_len", 192)) or 192)
        self.action_dim = int(schema_d.get("action_dim", cfg_d.get("action_dim", ACTION_DIM)) or ACTION_DIM)
        self.scalar_dim = int(schema_d.get("scalar_dim", cfg_d.get("scalar_dim", 5)) or 5)

        if self.action_dim != ACTION_DIM:
            raise RuntimeError(f"Checkpoint action_dim={self.action_dim} but features ACTION_DIM={ACTION_DIM}")

        # y_norm_factor saved by train.py
        self.y_norm_factor = float(ckpt.get("y_norm_factor", 0.0) or 0.0)
        if self.y_norm_factor <= 0.0:
            # safe default, but you really want it in ckpt
            self.y_norm_factor = 325.0

        cfg = MinimalQConfig(
            max_seq_len=int(cfg_d.get("max_seq_len", self.seq_len)),
            d_model=int(cfg_d.get("d_model", 256)),
            time_layers=int(cfg_d.get("time_layers", 4)),
            n_heads=int(cfg_d.get("n_heads", 4)),
            dropout=float(cfg_d.get("dropout", 0.10)),
            action_dim=int(cfg_d.get("action_dim", self.action_dim)),
            scalar_dim=int(cfg_d.get("scalar_dim", self.scalar_dim)),
            emotion_vocab=int(cfg_d.get("emotion_vocab", 256)),
            chip_vocab=int(cfg_d.get("chip_vocab", 512)),
            tile_vocab=int(cfg_d.get("tile_vocab", 128)),
        )

        model = MinimalQCritic(cfg)
        sd = ckpt.get("model", None)
        if not isinstance(sd, dict):
            raise RuntimeError("Checkpoint missing model state_dict")
        model.load_state_dict(sd, strict=True)

        model.to(self.device)
        model.eval()

        self.cfg = cfg
        self.model = model

    def _maybe_enable_fast_cuda(self) -> None:
        """Safe inference speed knobs (no accuracy guarantees needed for a critic UI probe)."""
        if self.device.type != "cuda":
            return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.no_grad()
    def infer_last_raw_batch(
        self,
        *,
        frames_window: List[Dict[str, Any]],
        overrides_list: List[Dict[str, int]],
        hold: int = 4,
        seq_len: Optional[int] = None,
        start_stride: int = 1,              # kept for signature parity; not used here
        require_cust_gt0: bool = True,
        batch_seqs: Optional[int] = None,   # max candidates per GPU batch
    ) -> List[Optional[float]]:
        """
        Extremely fast batched evaluator for /api/critic_find_best.

        Computes y_raw_hat at the *endpoint* of frames_window for many candidate button overrides,
        by constructing the exact same "sequence start at the endpoint" input that your
        current slow path effectively uses when it reads raw_dense[-1].

        Behavior:
          - Only the endpoint frame's ACTION vector is overridden (button keys only).
          - Model input sequence is length T with timestep 0 = endpoint features, timestep 1..T-1 = padding zeros,
            matching slice_pad behavior when s0 is the last sampled point.
          - Returns list aligned with overrides_list, each element is raw y_hat (float) or None on failure.

        This is designed so you can push batch sizes into the thousands to saturate VRAM.
        """
        if not frames_window:
            raise ValueError("frames_window is empty")
        if not isinstance(overrides_list, list) or not overrides_list:
            raise ValueError("overrides_list must be a non-empty list")

        hold = max(1, int(hold))
        T = int(seq_len or self.seq_len)
        T = max(1, min(T, int(self.cfg.max_seq_len)))

        # Optional: you can require battle at endpoint (matches your API behavior)
        end_frame = frames_window[-1]
        if require_cust_gt0:
            try:
                cust = int(end_frame.get("cust_gauge") or 0)
            except Exception:
                cust = 0
            if cust <= 0:
                return [None for _ in overrides_list]

        # Ensure speed knobs are on (idempotent)
        self._maybe_enable_fast_cuda()

        # Build endpoint features
        raw_i = len(frames_window) - 1
        feats = _extract_step_features(frames_window, raw_i)

        # Base action at endpoint (aggregate over hold frames starting at endpoint; typically just the endpoint)
        base_action = aggregate_action(frames_window, raw_i, hold)
        if isinstance(base_action, list):
            base_action = torch.tensor(base_action, dtype=torch.float32)
        elif not isinstance(base_action, torch.Tensor):
            base_action = torch.tensor(base_action, dtype=torch.float32)

        # Mapping for button overrides (only touches known BUTTON_KEYS)
        key_to_idx = {k: i for i, k in enumerate(BUTTON_KEYS)}

        # Decide chunk size for candidates
        bs = int(batch_seqs or self.batch_seqs)
        bs = max(1, bs)

        results: List[Optional[float]] = [None] * len(overrides_list)

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        # Pre-make the constant per-candidate timestep-0 tensors (CPU) then copy once per batch
        # Scalars normalized exactly like infer_from_frames
        scalars0 = torch.tensor(
            [
                float(feats["p_hp"]) / 2500.0,
                float(feats["e_hp"]) / 2500.0,
                float(feats["p_chg"]) / 2.0,
                float(feats["e_chg"]) / 2.0,
                float(feats["cust"]) / 100.0,
            ],
            dtype=torch.float32,
        )
        p_emo0 = torch.tensor(int(feats["p_emo"]), dtype=torch.int64)
        e_emo0 = torch.tensor(int(feats["e_emo"]), dtype=torch.int64)
        grid_tile0 = torch.tensor(feats["grid_tile"], dtype=torch.int64)    # [18]
        grid_owner0 = torch.tensor(feats["grid_owner"], dtype=torch.int64)  # [18]
        p_grid0 = torch.tensor(int(feats["p_grid_idx"]), dtype=torch.int64)
        e_grid0 = torch.tensor(int(feats["e_grid_idx"]), dtype=torch.int64)
        rel0 = torch.tensor(int(feats["rel_pe_idx"]), dtype=torch.int64)
        chip0 = torch.tensor(int(feats["player_chip"]), dtype=torch.int64)

        base_action0 = base_action.to(torch.float32)

        for off in range(0, len(overrides_list), bs):
            chunk = overrides_list[off : off + bs]
            B = len(chunk)

            # Allocate batch xb on device (zeros padded after t=0)
            xb: Dict[str, torch.Tensor] = {
                "scalars": torch.zeros((B, T, 5), dtype=torch.float32, device=self.device),
                "p_emotion_id": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "e_emotion_id": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "grid_tile": torch.zeros((B, T, 18), dtype=torch.int64, device=self.device),
                "grid_owner": torch.zeros((B, T, 18), dtype=torch.int64, device=self.device),
                "p_grid_idx": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "e_grid_idx": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "rel_pe_idx": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "player_chip": torch.zeros((B, T), dtype=torch.int64, device=self.device),
                "action": torch.zeros((B, T, ACTION_DIM), dtype=torch.float32, device=self.device),
            }

            # Fill timestep 0 with endpoint features
            xb["scalars"][:, 0, :] = scalars0.to(self.device)
            xb["p_emotion_id"][:, 0] = p_emo0.to(self.device)
            xb["e_emotion_id"][:, 0] = e_emo0.to(self.device)
            xb["grid_tile"][:, 0, :] = grid_tile0.to(self.device)
            xb["grid_owner"][:, 0, :] = grid_owner0.to(self.device)
            xb["p_grid_idx"][:, 0] = p_grid0.to(self.device)
            xb["e_grid_idx"][:, 0] = e_grid0.to(self.device)
            xb["rel_pe_idx"][:, 0] = rel0.to(self.device)
            xb["player_chip"][:, 0] = chip0.to(self.device)

            # Build per-candidate overridden action for timestep 0
            a0 = base_action0.unsqueeze(0).repeat(B, 1).to(self.device)  # [B, ACTION_DIM]
            # Apply overrides
            for i, ov in enumerate(chunk):
                if not isinstance(ov, dict):
                    continue
                for k, v in ov.items():
                    idx = key_to_idx.get(str(k))
                    if idx is None:
                        continue
                    a0[i, idx] = 1.0 if int(v) != 0 else 0.0

            xb["action"][:, 0, :] = a0

            with amp_ctx:
                q = self.model(xb)  # [B, T] in tanh-space

            q0 = q[:, 0].to(torch.float32)
            q0 = _clamp_for_atanh(q0, eps=1e-3)
            y_raw_hat = float(self.y_norm_factor) * _atanh(q0)  # [B] tensor

            y = y_raw_hat.detach().cpu().tolist()
            for i, val in enumerate(y):
                results[off + i] = float(val) if (val is not None and math.isfinite(float(val))) else None

        return results
        
    @torch.no_grad()
    def infer_from_frames(
        self,
        *,
        frames: List[Dict[str, Any]],
        hold: int = 4,
        seq_len: Optional[int] = None,
        start_stride: int = 1,
        require_cust_gt0: bool = True,
        batch_seqs: Optional[int] = None,
    ) -> MinimalCriticInferResult:
        n_frames = int(len(frames))
        if n_frames <= 0:
            return MinimalCriticInferResult(values_by_frame={}, meta={"n_frames": 0, "reason": "empty"})

        hold = max(1, int(hold))
        T = int(seq_len or self.seq_len)
        T = max(1, min(T, int(self.cfg.max_seq_len)))
        start_stride = max(1, int(start_stride))
        bs = int(batch_seqs or self.batch_seqs)
        bs = max(1, bs)

        # sampled indices along raw timeline
        sample_raw: List[int] = list(range(0, n_frames, hold))
        S = int(len(sample_raw))
        if S <= 0:
            return MinimalCriticInferResult(values_by_frame={}, meta={"n_frames": n_frames, "reason": "no_samples"})

        # battle mask at sampled points
        battle_mask_sample: List[bool] = []
        for ri in sample_raw:
            cust = _as_int(frames[ri].get("cust_gauge"), 0) if 0 <= ri < n_frames else 0
            battle_mask_sample.append(bool(cust > 0))

        if require_cust_gt0 and (not any(battle_mask_sample)):
            return MinimalCriticInferResult(
                values_by_frame={},
                meta={"n_frames": n_frames, "hold": hold, "seq_len": T, "reason": "no_battle_frames"},
            )

        # build sampled features tensors [S,...] (mirrors precache)
        scalars = torch.zeros((S, 5), dtype=torch.float32)
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

        # starts on sampled timeline
        if require_cust_gt0:
            starts = [i for i, ok in enumerate(battle_mask_sample) if ok]
        else:
            starts = list(range(S))

        starts = starts[::start_stride]
        if not starts:
            return MinimalCriticInferResult(values_by_frame={}, meta={"n_frames": n_frames, "reason": "no_starts"})

        # helper slicing
        def _slice_pad_1d(src: torch.Tensor, s0: int) -> torch.Tensor:
            out = torch.zeros((T,), dtype=src.dtype)
            n = min(T, int(src.shape[0]) - int(s0))
            if n > 0:
                out[:n] = src[s0 : s0 + n]
            return out

        def _slice_pad_2d(src: torch.Tensor, s0: int, d1: int) -> torch.Tensor:
            out = torch.zeros((T, d1), dtype=src.dtype)
            n = min(T, int(src.shape[0]) - int(s0))
            if n > 0:
                out[:n] = src[s0 : s0 + n]
            return out

        values_by_frame: Dict[int, float] = {}

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        for b0 in range(0, len(starts), bs):
            batch_starts = starts[b0 : b0 + bs]
            B = len(batch_starts)

            xb: Dict[str, torch.Tensor] = {
                "scalars": torch.zeros((B, T, 5), dtype=torch.float32),
                "p_emotion_id": torch.zeros((B, T), dtype=torch.int64),
                "e_emotion_id": torch.zeros((B, T), dtype=torch.int64),
                "grid_tile": torch.zeros((B, T, 18), dtype=torch.int64),
                "grid_owner": torch.zeros((B, T, 18), dtype=torch.int64),
                "p_grid_idx": torch.zeros((B, T), dtype=torch.int64),
                "e_grid_idx": torch.zeros((B, T), dtype=torch.int64),
                "rel_pe_idx": torch.zeros((B, T), dtype=torch.int64),
                "player_chip": torch.zeros((B, T), dtype=torch.int64),
                "action": torch.zeros((B, T, ACTION_DIM), dtype=torch.float32),
            }

            for i, s0 in enumerate(batch_starts):
                xb["scalars"][i] = _slice_pad_2d(scalars, s0, 5)
                xb["p_emotion_id"][i] = _slice_pad_1d(p_emo, s0)
                xb["e_emotion_id"][i] = _slice_pad_1d(e_emo, s0)
                xb["grid_tile"][i] = _slice_pad_2d(grid_tile, s0, 18)
                xb["grid_owner"][i] = _slice_pad_2d(grid_owner, s0, 18)
                xb["p_grid_idx"][i] = _slice_pad_1d(p_grid_idx, s0)
                xb["e_grid_idx"][i] = _slice_pad_1d(e_grid_idx, s0)
                xb["rel_pe_idx"][i] = _slice_pad_1d(rel_pe_idx, s0)
                xb["player_chip"][i] = _slice_pad_1d(player_chip, s0)
                xb["action"][i] = _slice_pad_2d(actions, s0, ACTION_DIM)

            xb = {k: v.to(self.device, non_blocking=True) for k, v in xb.items()}

            with amp_ctx:
                q = self.model(xb)  # [B,T] in tanh-space

            q0 = q[:, 0].detach().to(torch.float32)  # [B]
            q0 = _clamp_for_atanh(q0, eps=1e-3)
            y_raw_hat = float(self.y_norm_factor) * _atanh(q0)  # [B]

            y_raw_hat_cpu = y_raw_hat.detach().cpu().tolist()
            for s0, v in zip(batch_starts, y_raw_hat_cpu):
                raw_idx = int(sample_raw[int(s0)])
                # deterministic, "start-value" assignment
                values_by_frame[raw_idx] = float(v)

        meta = {
            "source": "critic_minimal",
            "ckpt": self.ckpt_path,
            "device": str(self.device),
            "use_amp": bool(self.use_amp),
            "hold": int(hold),
            "seq_len": int(T),
            "start_stride": int(start_stride),
            "batch_seqs": int(bs),
            "n_frames": int(n_frames),
            "sampled_points": int(S),
            "starts": int(len(starts)),
            "action_dim": int(ACTION_DIM),
            "action_keys": list(BUTTON_KEYS),
            "y_norm_factor": float(self.y_norm_factor),
            "note": "values_by_frame is sparse at raw indices (0,hold,2hold,...). Viewer should densify/smooth within cust_mask.",
        }

        return MinimalCriticInferResult(values_by_frame=values_by_frame, meta=meta)
