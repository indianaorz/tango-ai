# viewer/critic_frame_infer.py
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from critic_minimal.model import StackedConfig, StackedQCritic
from critic_minimal.features import (
    ACTION_DIM, BUTTON_KEYS, _as_int, _as_list, _btn01, 
    _chip_id_norm, _owner_norm, _pos_norm, _tile_norm, 
    pos_to_grid_idx, rel_pe_index
)

@dataclass(frozen=True)
class FrameInferResult:
    values_by_frame: List[Optional[float]]
    meta: Dict[str, Any]

def _extract_flat_features(f: Dict[str, Any]) -> List[float]:
    """
    Must match precache.py _extract_frame_features exactly.
    """
    # 1. Scalars
    s = [
        float(_as_int(f.get("player_health"), 0)) / 2500.0,
        float(_as_int(f.get("enemy_health"), 0)) / 2500.0,
        float(_as_int(f.get("player_charge"), 0)) / 2.0,
        float(_as_int(f.get("enemy_charge"), 0)) / 2.0,
        float(_as_int(f.get("cust_gauge"), 0)) / 100.0,
    ]
    # 2. Categoricals
    cats = [
        float(_as_int(f.get("player_game_emotion"), 0)),
        float(_as_int(f.get("enemy_game_emotion"), 0)),
        float(_chip_id_norm(f.get("player_chip"))),
    ]
    # 3. Grid
    gs = _as_list(f.get("grid_state"))
    go = _as_list(f.get("grid_owner_state"))
    grid_feats = []
    for i in range(18):
        t_val = _tile_norm(gs[i]) if i < len(gs) else 0
        o_val = _owner_norm(go[i]) if i < len(go) else 2
        grid_feats.append(float(t_val))
        grid_feats.append(float(o_val))
        
    # 4. Positions
    px, py = _pos_norm(f.get("player_pos"))
    ex, ey = _pos_norm(f.get("enemy_pos"))
    pidx = pos_to_grid_idx(float(px), float(py))
    eidx = pos_to_grid_idx(float(ex), float(ey))
    rel = rel_pe_index(pidx, eidx)
    pos_feats = [float(pidx), float(eidx), float(rel)]
    
    return s + cats + grid_feats + pos_feats

def _buttons_from_frame(frame: Dict[str, Any]) -> List[float]:
    out = [0.0] * ACTION_DIM
    for i, k in enumerate(BUTTON_KEYS):
        # Handle manual override keys from Lab (floats) or raw captures
        val = frame.get(k, 0.0)
        out[i] = 1.0 if (float(val) > 0.5) else 0.0
    return out

class FrameCriticRunner:
    def __init__(self, *, ckpt_path: str, device: str = "cuda", use_amp: bool = True, batch_frames: int = 256) -> None:
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.batch_frames = batch_frames
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_d = ckpt.get("cfg", {})
        
        # Load Stacked Config
        self.cfg = StackedConfig(**cfg_d)
        self.model = StackedQCritic(self.cfg)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval().to(self.device)
        self.ckpt_path = ckpt_path
        
        # Stride used in training (Current + T-Stride)
        self.stack_stride = 4 

    @torch.no_grad()
    def infer_values_dense(self, *, frames: Sequence[Dict[str, Any]], require_cust_gt0: bool = True) -> FrameInferResult:
        """
        Infers Q-values for a sequence of frames.
        Automatically handles stacking (pairing Frame i with Frame i-4).
        """
        n = len(frames)
        out: List[Optional[float]] = [None] * n
        
        # 1. Pre-calculate features for all frames
        # We need this because frame i might rely on frame i-4
        all_feats = []
        for f in frames:
            all_feats.append(_extract_flat_features(f if f else {}))
            
        # 2. Identify frames to score
        cust_mask = [(int((f or {}).get("cust_gauge") or 0) > 0) for f in frames] if require_cust_gt0 else [True] * n
        idxs_to_score = [i for i in range(n) if cust_mask[i]]
        
        if not idxs_to_score:
            return FrameInferResult(out, {})

        # 3. Batch Processing
        for off in range(0, len(idxs_to_score), self.batch_frames):
            batch_idxs = idxs_to_score[off : off + self.batch_frames]
            
            # Prepare Tensors
            x_stacked_list = []
            action_list = []
            prev_action_list = []
            
            for i in batch_idxs:
                # -- State Stack --
                # Curr: i
                # Prev: i - 4 (if i < 4, clamp to 0)
                prev_i = max(0, i - self.stack_stride)
                
                feat_curr = all_feats[i]
                feat_prev = all_feats[prev_i]
                
                # Combine [Dim] + [Dim] -> [Dim*2]
                x_stacked_list.append(feat_curr + feat_prev)
                
                # -- Actions --
                # Current frame action (what we are evaluating)
                action_list.append(_buttons_from_frame(frames[i]))
                
                # Prev frame action (context from T-1)
                if i > 0:
                    prev_action_list.append(_buttons_from_frame(frames[i-1]))
                else:
                    prev_action_list.append([0.0]*ACTION_DIM)

            # To Device
            xb = {
                "x_stacked": torch.tensor(x_stacked_list, dtype=torch.float32).to(self.device),
                "action": torch.tensor(action_list, dtype=torch.float32).to(self.device),
                "prev_action": torch.tensor(prev_action_list, dtype=torch.float32).to(self.device)
            }
            
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                q = self.model(xb).detach().float().cpu().view(-1)
                
            for k, original_idx in enumerate(batch_idxs):
                out[original_idx] = float(q[k].item())

        return FrameInferResult(out, {"ckpt": self.ckpt_path})