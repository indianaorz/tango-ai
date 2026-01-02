from __future__ import annotations

import time
import math
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Ensure these imports exist in your project structure
from critic_minimal.features import (
    ACTION_DIM,
    BUTTON_KEYS,
    extract_flow_features, # Ensure this is exposed in features.py
    _as_int,
    _pos_norm,
    pos_to_grid_idx,
)
# We import the model class you provided. 
# Assuming you saved the model code in critic_minimal/model_flow.py
from critic_minimal.model_flow import ActionFlowDiT

# =============================================================================
# Small helpers
# =============================================================================

def _int_to_bin16(mask: int) -> str:
    return format(int(mask) & 0xFFFF, "016b")

def _now_ms() -> float:
    return time.time() * 1000.0

def _grid_idx_to_pos(idx):
    # Maps grid index back to approximate pixel coordinates if needed
    # (reusing logic from your app.py if required, otherwise using raw pos)
    row, col = idx // 6, idx % 6
    x = 20 + (col * 40)
    y = [260, 515, 770][row] if row < 3 else 260
    return [x, y]

def _normalize_pos_for_model(pos_any: Any) -> List[float]:
    # Ensure [x, y] list for the feature extractor
    xi, yi = _pos_norm(pos_any)
    return [float(xi), float(yi)]

# =============================================================================
# Key Mapping (Vector Index -> Game Bitmask)
# =============================================================================

@dataclass(frozen=True)
class _KeyMap:
    # Mapping from BUTTON_KEYS index names to Emulator Bit Indices
    mapping: Dict[str, int]

def _build_keymap(key_bit_positions: Dict[str, int]) -> _KeyMap:
    # Map the canonical BUTTON_KEYS names to your specific emulator bits
    # BUTTON_KEYS = ['DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'EAST', 'SOUTH', ...]
    
    # Standardize config keys to match BUTTON_KEYS where possible
    lookup = {
        "DPAD_UP": ["UP", "DPAD_UP", "DpadUp"],
        "DPAD_DOWN": ["DOWN", "DPAD_DOWN", "DpadDown"],
        "DPAD_LEFT": ["LEFT", "DPAD_LEFT", "DpadLeft"],
        "DPAD_RIGHT": ["RIGHT", "DPAD_RIGHT", "DpadRight"],
        "START": ["RETURN", "START"],
        "BACK": ["ESCAPE", "BACK"],
        "LEFT_SHOULDER": ["L", "LB", "LEFT_SHOULDER"],
        "RIGHT_SHOULDER": ["R", "RB", "RIGHT_SHOULDER"],
        "EAST": ["Z", "A", "EAST"],   # A button usually mapped to Z key
        "SOUTH": ["X", "B", "SOUTH"], # B button usually mapped to X key
    }
    
    final_map = {}
    for btn_name in BUTTON_KEYS:
        candidates = lookup.get(btn_name, [btn_name])
        for c in candidates:
            if c in key_bit_positions:
                final_map[btn_name] = int(key_bit_positions[c])
                break
    
    return _KeyMap(mapping=final_map)

def _vec_to_bitmask(action_vec: np.ndarray, km: _KeyMap, threshold: float = 0.5) -> int:
    """
    Converts 10-dim float vector -> 16-bit integer mask
    """
    mask = 0
    for i, btn_name in enumerate(BUTTON_KEYS):
        val = action_vec[i]
        if val > threshold:
            bit = km.mapping.get(btn_name)
            if bit is not None:
                mask |= (1 << bit)
    return int(mask) & 0xFFFF

# =============================================================================
# History Buffer (Stores 47-dim Feature Vectors)
# =============================================================================

class _History:
    def __init__(self, context_len: int = 256, feat_dim: int = 47):
        self.context_len = context_len
        self.feat_dim = feat_dim
        # Initialize with zeros (padding)
        self.buffer = deque(maxlen=context_len)
        for _ in range(context_len):
            self.buffer.append(np.zeros(feat_dim, dtype=np.float32))
            
    def push(self, feat_vec: np.ndarray):
        # feat_vec expected shape (47,)
        self.buffer.append(feat_vec)
        
    def get_tensor(self, device: torch.device) -> torch.Tensor:
        # Returns [1, 256, 47] tensor
        arr = np.array(self.buffer, dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        return t
        
    def reset(self):
        for _ in range(self.context_len):
            self.buffer.append(np.zeros(self.feat_dim, dtype=np.float32))

# =============================================================================
# Model Runner (Inference Loop)
# =============================================================================

class _Runner:
    def __init__(self, ckpt_path: str, device: torch.device, use_amp: bool):
        self.device = device
        self.use_amp = use_amp
        
        print(f"[Battle] Loading V8 AdaLN Flow Model from {ckpt_path}...")
        
        # === V8 / V7 Architecture Config ===
        # Must match training: embed_dim=384, heads=6
        self.model = ActionFlowDiT(
            feat_dim=47,
            act_dim=10,
            hist_len=256,
            seq_len=18,
            embed_dim=384, 
            depth=6,
            heads=6 
        ).to(device)
        
        try:
            state = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(state)
            self.model.eval()
            print("[Battle] Model loaded successfully.")
        except Exception as e:
            print(f"[Battle] CRITICAL ERROR loading model: {e}")
            raise e

    @torch.no_grad()
    def generate_plan(self, history: torch.Tensor, prompt_reward: float = 1.0, cfg_scale: float = 2.5) -> np.ndarray:
        """
        Runs the 10-step Euler Diffusion loop.
        Returns: [18, 10] numpy array (Time, Buttons)
        """
        # 1. Setup Conditions
        # History: [1, 256, 47] -> [2, 256, 47] for CFG
        hist_batch = history.repeat(2, 1, 1)
        
        # Rewards: [Target, Null] -> [1.0, 0.0]
        ret_batch = torch.tensor([[prompt_reward], [0.0]], device=self.device, dtype=torch.float32)
        
        # 2. Initialize Noise (t=0)
        # Shape: [1, 18, 10]
        x = torch.randn(1, 18, 10, device=self.device)
        
        # 3. Euler Loop (NitroGen Style)
        num_steps = 10
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i / num_steps
            
            # Prepare inputs for CFG batch [Cond, Uncond]
            x_in = x.repeat(2, 1, 1) # [2, 18, 10]
            t_in = torch.full((2,), t_val, device=self.device, dtype=torch.float32)
            
            # Predict Velocity field
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                v_out = self.model(x_in, t_in, hist_batch, ret_batch)
            
            v_cond, v_uncond = v_out[0], v_out[1]
            
            # Classifier-Free Guidance extrapolation
            v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Euler Step
            x = x + v_final * dt
            
        # 4. Return Plan
        return x[0].cpu().numpy()

# =============================================================================
# Strategy Class
# =============================================================================

@dataclass
class _PortState:
    hist: _History
    
class MinimalCriticBattleStrategy:
    """
    V8/V7 AdaLN Diffusion Strategy.
    Generates 18-frame plans based on 256-frame feature history.
    Uses Receding Horizon Control (generates plan every frame, executes step 0).
    """
    def __init__(
        self,
        *,
        ckpt_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
        # Legacy args maintained for compatibility with loader
        hold: int = 4, 
        seq_len: int = 192,
        start_stride: int = 1,
        require_cust_gt0: bool = True,
        use_amp: bool = True,
        **kwargs
    ):
        self.requires_image = False # Feature-based
        self._km = _build_keymap(key_bit_positions)
        self._runner = _Runner(ckpt_path, device, use_amp)
        self._by_port: Dict[int, _PortState] = {}
        
        # Config
        self.target_reward = 1.0 # "Win"
        self.cfg_scale = 2.5     # Strong guidance
        self.require_cust_gt0 = require_cust_gt0

    def reset_state(self, port: int):
        self._by_port.pop(int(port), None)

    def _get_state(self, port: int) -> _PortState:
        p = int(port)
        if p not in self._by_port:
            self._by_port[p] = _PortState(
                hist=_History(context_len=256, feat_dim=47)
            )
        return self._by_port[p]

    def decide_action(self, port: int, game_state: dict) -> dict:
        st = self._get_state(port)
        
        # 1. Check Battle State
        cust = float(game_state.get("cust_gauge", 0))
        if self.require_cust_gt0 and cust <= 0:
            # Not in battle (Menuing/Cutscene) - Reset history and idle
            st.hist.reset()
            return {
                "button_command": {"type": "key_press", "key": _int_to_bin16(0)},
                "debug": {"active_model": "battle", "note": "not_in_battle"}
            }

        # 2. Extract Features (Same logic as app.py / extract_flow_features)
        # We assume game_state has keys like 'player_health', 'grid_tiles', etc.
        # normalized or raw. extract_flow_features handles dict -> vector.
        
        # Ensure we pass expected keys to the extractor
        p_pos = _normalize_pos_for_model(game_state.get("player_pos"))
        e_pos = _normalize_pos_for_model(game_state.get("enemy_pos"))
        
        # Construct the state dict expected by extract_flow_features
        # This mirrors viewer/app.py logic
        sim_state = {
            "player_health": _as_int(game_state.get("player_health")),
            "enemy_health": _as_int(game_state.get("enemy_health")),
            "player_charge": float(game_state.get("player_charge", 0)),
            "enemy_charge": float(game_state.get("enemy_charge", 0)),
            "cust_gauge": int(cust),
            "player_game_emotion": _as_int(game_state.get("player_game_emotion")),
            "enemy_game_emotion": _as_int(game_state.get("enemy_game_emotion")),
            "player_pos": p_pos,
            "enemy_pos": e_pos,
            # Handle potential key variations from different hook versions
            "grid_state": game_state.get("grid_state") or game_state.get("grid_tile") or [2]*18,
            "grid_owner_state": game_state.get("grid_owner_state") or game_state.get("grid_owner") or ([0]*9 + [1]*9),
            "player_chip": _as_int(game_state.get("player_chip"), 65535)
        }
        
        # [47] float32 vector
        feat_vec = extract_flow_features(sim_state)
        
        # 3. Update History
        st.hist.push(feat_vec)
        
        # 4. Generate Plan (Diffusion)
        t0 = _now_ms()
        hist_tensor = st.hist.get_tensor(self._runner.device)
        
        # Get 18-step plan [18, 10]
        plan = self._runner.generate_plan(
            hist_tensor, 
            prompt_reward=self.target_reward, 
            cfg_scale=self.cfg_scale
        )
        infer_ms = _now_ms() - t0
        
        # 5. Extract Immediate Action (Step 0)
        # We take the first step of the plan for execution (Receding Horizon)
        action_vec_0 = plan[0] 
        
        # Convert to Bitmask
        mask = _vec_to_bitmask(action_vec_0, self._km)
        key_bin = _int_to_bin16(mask)
        
        return {
            "button_command": {"type": "key_press", "key": key_bin},
            "ng_key_bin": "", # Used for Nitrogen visualization if needed
            "debug": {
                "active_model": "battle_flow_v8",
                "infer_ms": infer_ms,
                "action_raw": action_vec_0.tolist(), # For logging/plotting
                "plan_preview": plan[:3].tolist(),   # Log first 3 steps
                "cfg_scale": self.cfg_scale
            }
        }