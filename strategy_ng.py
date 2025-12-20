# ── Begin: strategy_ng.py ──
# strategy_ng.py
from __future__ import annotations

import base64
import os
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from PIL import Image

from ng_policy import NgNitroGenPolicy, load_ng_checkpoint, NgPolicyError

# -----------------------------------------------------------------------------
# Small, explicit model->intent mapping (configurable)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NgActionSchema:
    move_x_idx: int = 0
    move_y_idx: int = 1
    a_idx: int = 2
    b_idx: int = 3
    start_idx: int = 4

    deadzone: float = 0.25
    button_threshold: float = 0.25

    @staticmethod
    def from_env() -> "NgActionSchema":
        dz = float(os.getenv("NG_MOVE_DEADZONE", "0.25"))
        bt = float(os.getenv("NG_BTN_THRESH", "0.25"))
        return NgActionSchema(deadzone=dz, button_threshold=bt)

def _tanh_scalar(x: float) -> float:
    return float(torch.tanh(torch.tensor(x)).item())

def _safe_get(action_vec_1d: torch.Tensor, idx: int) -> float:
    if idx < 0: return 0.0
    if action_vec_1d.numel() <= idx: return 0.0
    return float(action_vec_1d[idx].item())

# -----------------------------------------------------------------------------
# Image decode + preprocessing
# -----------------------------------------------------------------------------

def _decode_pil_from_b64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64: return None
    try:
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None

def _pil_to_chw_float01(pil_img: Image.Image, *, out_h: int, out_w: int) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), resample=Image.BICUBIC)
    px = torch.tensor(list(pil_img.getdata()), dtype=torch.float32)
    px = px.view(out_h, out_w, 3).permute(2, 0, 1).contiguous()
    return px / 255.0

# -----------------------------------------------------------------------------
# Mapping
# -----------------------------------------------------------------------------

_DEFAULT_BUTTON_ALIAS = { "A": "Z", "B": "X", "START": "RETURN" }

def _mask_to_bin16(mask: int) -> str:
    s = format(int(mask) & 0xFFFF, "016b")
    return s

def _logical_buttons_to_mask(key_bit_positions: Dict[str, int], logical_buttons: List[str], *, button_alias: Dict[str, str]) -> int:
    mask = 0
    for b in logical_buttons:
        b = str(b).upper()
        actual = button_alias.get(b, b)
        bit = key_bit_positions.get(actual)
        if bit is None: continue
        mask |= (1 << int(bit))
    return int(mask)

def _intent_from_action_vec(action_vec_1d: torch.Tensor, schema: NgActionSchema) -> List[str]:
    mx = _tanh_scalar(_safe_get(action_vec_1d, schema.move_x_idx))
    my = _tanh_scalar(_safe_get(action_vec_1d, schema.move_y_idx))
    btns: List[str] = []

    if abs(mx) >= schema.deadzone: btns.append("RIGHT" if mx > 0 else "LEFT")
    if abs(my) >= schema.deadzone: btns.append("DOWN" if my > 0 else "UP")

    if _tanh_scalar(_safe_get(action_vec_1d, schema.a_idx)) >= schema.button_threshold: btns.append("A")
    if _tanh_scalar(_safe_get(action_vec_1d, schema.b_idx)) >= schema.button_threshold: btns.append("B")
    if _tanh_scalar(_safe_get(action_vec_1d, schema.start_idx)) >= schema.button_threshold: btns.append("START")

    return btns

def _apply_gating(logical_buttons: List[str], *, inside_window: bool, allow_actions_in_window: bool, in_battle: bool, forbid_actions_in_battle: List[str]) -> List[str]:
    btns = list(logical_buttons)
    if inside_window and not allow_actions_in_window: return []
    if in_battle and forbid_actions_in_battle:
        forb = {s.upper() for s in forbid_actions_in_battle}
        btns = [b for b in btns if b.upper() not in forb]
    return btns

# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------

class NGAgentStrategy:
    def __init__(self, *, ckpt_path: str, device: torch.device, key_bit_positions: Dict[str, int], discrete_actions: List[str], util_fns: Dict[str, Any], frame_h: int, frame_w: int, seq_len_frames: int, use_images: bool = True, allow_actions_in_window: bool = False, forbid_actions_in_battle: Optional[List[str]] = None, button_alias: Optional[Dict[str, str]] = None):
        self.dev = torch.device(device)
        self.key_bits = dict(key_bit_positions or {})
        self.discrete_actions = list(discrete_actions or [])
        self.util = dict(util_fns or {})
        self.frame_h = int(frame_h)
        self.frame_w = int(frame_w)
        self.T = int(seq_len_frames)
        self.use_images = bool(use_images)
        self.allow_actions_in_window = bool(allow_actions_in_window)
        self.forbid_actions_in_battle = list(forbid_actions_in_battle or [])
        self.button_alias = dict(button_alias or _DEFAULT_BUTTON_ALIAS)
        self.schema = NgActionSchema.from_env()

        # Debug & Determinism
        self._emit_ng_raw = os.getenv("NG_DEBUG_RAW", "0").strip() in ("1", "true", "True")
        self._raw_max_items = int(os.getenv("NG_DEBUG_RAW_MAX_ITEMS", "16"))
        self._seed_base = os.getenv("NG_SEED_BASE", "").strip()
        self._use_seed = bool(self._seed_base)

        # Performance flags
        self.use_fp16 = bool(int(os.getenv("NG_USE_FP16", "1")))
        
        # Buffer
        self._frames: Dict[int, Deque[torch.Tensor]] = {}
        self._last_ts: Dict[int, float] = {}

        loaded = load_ng_checkpoint(ckpt_path, device=self.dev)
        self.policy = NgNitroGenPolicy(loaded, default_game_id=None).to(self.dev).eval()
        
        # Optimize model precision immediately
        if self.use_fp16:
            self.policy.half()

    def reset_state(self, port: int):
        p = int(port)
        self._frames.pop(p, None)
        self._last_ts.pop(p, None)

    def _get_in_battle(self, gs: Dict[str, Any]) -> bool:
        for k in ("in_battle", "battle_active", "is_in_battle"):
            v = gs.get(k, None)
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)) and v in (0, 1): return bool(v)
        bs = gs.get("battle_state", None)
        if isinstance(bs, (int, float)): return bool(bs > 0)
        return True

    def _push_frame(self, port: int, frame_chw: torch.Tensor) -> torch.Tensor:
        """
        Prepare sequence tensor.
        If T=1, simply unsqueeze the current frame (stateless optimization).
        Otherwise maintain history deque.
        """
        # OPTIMIZATION: If config says 1 frame, skip the deque overhead completely.
        if self.T == 1:
            return frame_chw.unsqueeze(0).unsqueeze(0) # [1, 1, 3, H, W]

        p = int(port)
        dq = self._frames.get(p)
        if dq is None:
            dq = deque()
            self._frames[p] = dq

        dq.append(frame_chw)
        while len(dq) > self.T:
            dq.popleft()

        # Pad if history not full
        xs = list(dq)
        if len(xs) < self.T:
            pad = [xs[0]] * (self.T - len(xs))
            xs = pad + xs

        seq = torch.stack(xs, dim=0).unsqueeze(0).contiguous()
        return seq

    def _seed_for(self, port: int) -> Optional[int]:
        if not self._use_seed: return None
        try:
            base = int(self._seed_base)
        except Exception:
            base = 0
        step = int(time.time() * 20.0)
        return int(base + step + int(port) * 1000003)

    def decide_action(self, port: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        p = int(port)
        now = time.time()
        self._last_ts[p] = now

        def _no_op(debug: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "button_command": {"type": "key_press", "key": _mask_to_bin16(0)},
                "ng_key_bin": "",
                "debug": debug,
            }

        if not self.use_images:
            return _no_op({"reason": "NG requires images"})

        pil = _decode_pil_from_b64(game_state.get("image"))
        if pil is None:
            return _no_op({"reason": "no image in game_state"})

        # Preprocess frame -> Tensor
        frame_chw: torch.Tensor
        pp = self.util.get("preprocess_frame", None)
        if callable(pp):
            try:
                try: frame_chw = pp(pil, self.frame_h, self.frame_w)
                except TypeError: frame_chw = pp(pil)
                if not isinstance(frame_chw, torch.Tensor): raise TypeError("not tensor")
                if frame_chw.dim() == 4: frame_chw = frame_chw.squeeze(0)
                frame_chw = frame_chw.float().contiguous()
            except Exception:
                frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)
        else:
            frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)

        # Get Sequence [1, T, 3, H, W]
        seq = self._push_frame(p, frame_chw)

        try:
            seq_dev = seq.to(self.dev, non_blocking=True)
            
            # OPTIMIZATION: Cast to FP16 if enabled
            if self.use_fp16:
                seq_dev = seq_dev.half()

            seed = self._seed_for(p)
            
            # Inference
            if self._emit_ng_raw:
                primary, raw = self.policy(
                    seq_dev, seed=seed, take_step=0,
                    return_continuous=True, return_raw=True, raw_max_items=self._raw_max_items,
                )
                action_vec = primary
            else:
                action_vec = self.policy(
                    seq_dev, seed=seed, take_step=0,
                    return_continuous=True, return_raw=False,
                )
                raw = None

            # [1,D] -> [D]
            action_1d = action_vec.squeeze(0).detach().float().cpu()

        except (NgPolicyError, Exception) as e:
            # Fallback so we don't crash the connection loop
            print(f"NG Inference Error on port {port}: {e}")
            return _no_op({"reason": "ng_inference_failed", "error": repr(e)[:400]})

        # Interpret & Map
        intended_logical = _intent_from_action_vec(action_1d, self.schema)
        
        ng_intended_mask = _logical_buttons_to_mask(self.key_bits, intended_logical, button_alias=self.button_alias)
        ng_key_bin = _mask_to_bin16(ng_intended_mask)

        mapped_logical = _apply_gating(
            intended_logical,
            inside_window=bool(game_state.get("inside_window", False)),
            allow_actions_in_window=self.allow_actions_in_window,
            in_battle=self._get_in_battle(game_state),
            forbid_actions_in_battle=self.forbid_actions_in_battle,
        )
        mapped_mask = _logical_buttons_to_mask(self.key_bits, mapped_logical, button_alias=self.button_alias)
        mapped_key_bin = _mask_to_bin16(mapped_mask)

        debug = {
            "intent": { "intended_logical": intended_logical, "mapped_logical": mapped_logical },
        }
        if raw is not None: debug["ng_raw"] = raw

        return {
            "button_command": {"type": "key_press", "key": mapped_key_bin},
            "ng_key_bin": ng_key_bin,
            "debug": debug,
        }
# ── End: strategy_ng.py ──