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
    """
    Interprets NitroGen continuous action_vec [D] into a "logical intent".

    Default heuristic:
      action[0] = move_x  (- = left, + = right)
      action[1] = move_y  (- = up,   + = down)
      action[2] = A (chip)  (press if > threshold)
      action[3] = B (buster)(press if > threshold)
      action[4] = START     (press if > threshold)

    If D < needed indices, missing channels are treated as 0.

    This is meant for *debug + bring-up*; once you know the true schema,
    lock it in (or load prototypes / a learned head).
    """
    move_x_idx: int = 0
    move_y_idx: int = 1
    a_idx: int = 2
    b_idx: int = 3
    start_idx: int = 4

    deadzone: float = 0.25
    button_threshold: float = 0.25  # after tanh

    @staticmethod
    def from_env() -> "NgActionSchema":
        # Keep it simple: allow tuning without code changes.
        # Example:
        #   NG_MOVE_DEADZONE=0.35 NG_BTN_THRESH=0.5
        dz = float(os.getenv("NG_MOVE_DEADZONE", "0.25"))
        bt = float(os.getenv("NG_BTN_THRESH", "0.25"))
        return NgActionSchema(deadzone=dz, button_threshold=bt)


def _tanh_scalar(x: float) -> float:
    # Stable squashing for unknown scale.
    return float(torch.tanh(torch.tensor(x)).item())


def _safe_get(action_vec_1d: torch.Tensor, idx: int) -> float:
    if idx < 0:
        return 0.0
    if action_vec_1d.numel() <= idx:
        return 0.0
    return float(action_vec_1d[idx].item())


# -----------------------------------------------------------------------------
# Image decode + preprocessing
# -----------------------------------------------------------------------------

def _decode_pil_from_b64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def _pil_to_chw_float01(pil_img: Image.Image, *, out_h: int, out_w: int) -> torch.Tensor:
    """
    Convert PIL RGB -> float tensor [3,H,W] in [0,1].
    Uses bicubic resize (good for vision encoders).
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), resample=Image.BICUBIC)

    # PIL -> bytes -> tensor (avoid numpy dependency here)
    # torch doesn't natively read PIL; simplest is getdata then reshape.
    px = torch.tensor(list(pil_img.getdata()), dtype=torch.float32)  # [H*W, 3]
    px = px.view(out_h, out_w, 3).permute(2, 0, 1).contiguous()      # [3,H,W]
    return px / 255.0


# -----------------------------------------------------------------------------
# Mapping to Tango key bitmask
# -----------------------------------------------------------------------------

_DEFAULT_BUTTON_ALIAS = {
    "A": "Z",
    "B": "X",
    "START": "RETURN",
}


def _mask_to_bin16(mask: int) -> str:
    # Tango expects a 16-bit string; keep MSB..LSB ordering consistent with int_to_binary_string usage.
    # Your utils.int_to_binary_string likely does this, but we keep it local and deterministic.
    s = format(int(mask) & 0xFFFF, "016b")
    return s


def _logical_buttons_to_mask(
    key_bit_positions: Dict[str, int],
    logical_buttons: List[str],
    *,
    button_alias: Dict[str, str],
) -> int:
    """
    Convert logical button names into the *actual* in-game bits (Z/X/RETURN + directions).
    """
    mask = 0
    for b in logical_buttons:
        b = str(b).upper()
        actual = button_alias.get(b, b)
        bit = key_bit_positions.get(actual)
        if bit is None:
            continue
        mask |= (1 << int(bit))
    return int(mask)


def _intent_from_action_vec(action_vec_1d: torch.Tensor, schema: NgActionSchema) -> List[str]:
    """
    Returns a list of *logical* buttons: UP/DOWN/LEFT/RIGHT/A/B/START
    derived from the continuous action vector.
    """
    # movement
    mx = _tanh_scalar(_safe_get(action_vec_1d, schema.move_x_idx))
    my = _tanh_scalar(_safe_get(action_vec_1d, schema.move_y_idx))

    btns: List[str] = []

    if abs(mx) >= schema.deadzone:
        btns.append("RIGHT" if mx > 0 else "LEFT")
    if abs(my) >= schema.deadzone:
        # NOTE: typical screen coords: -y is up, +y is down (we assume that here)
        btns.append("DOWN" if my > 0 else "UP")

    # buttons
    a = _tanh_scalar(_safe_get(action_vec_1d, schema.a_idx))
    b = _tanh_scalar(_safe_get(action_vec_1d, schema.b_idx))
    st = _tanh_scalar(_safe_get(action_vec_1d, schema.start_idx))

    if a >= schema.button_threshold:
        btns.append("A")
    if b >= schema.button_threshold:
        btns.append("B")
    if st >= schema.button_threshold:
        btns.append("START")

    # stable ordering for easier diffing in logs
    order = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "A": 4, "B": 5, "START": 6}
    btns.sort(key=lambda x: order.get(x, 99))
    return btns


def _apply_gating(
    logical_buttons: List[str],
    *,
    inside_window: bool,
    allow_actions_in_window: bool,
    in_battle: bool,
    forbid_actions_in_battle: List[str],
) -> List[str]:
    btns = list(logical_buttons)

    if inside_window and not allow_actions_in_window:
        # "do nothing" in window
        return []

    if in_battle and forbid_actions_in_battle:
        forb = {s.upper() for s in forbid_actions_in_battle}
        btns = [b for b in btns if b.upper() not in forb]

    return btns


# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------

class NGAgentStrategy:
    """
    NitroGen-powered inference strategy, instrumented for mapping validation.

    Contract:
      decide_action(port, game_state) -> dict with:
        - button_command: {"type":"key_press","key":"0101..."}
        - ng_key_bin: raw NG "intent" key bits (pre-gating)
        - debug: mapping + (optional) ng raw summary
    """

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
        discrete_actions: List[str],  # kept for compatibility; not used directly by NG mapping yet
        util_fns: Dict[str, Any],
        frame_h: int,
        frame_w: int,
        seq_len_frames: int,
        use_images: bool = True,
        allow_actions_in_window: bool = False,
        forbid_actions_in_battle: Optional[List[str]] = None,
        button_alias: Optional[Dict[str, str]] = None,
    ):
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

        # Optional heavy debug
        self._emit_ng_raw = os.getenv("NG_DEBUG_RAW", "0").strip() in ("1", "true", "True")
        self._raw_max_items = int(os.getenv("NG_DEBUG_RAW_MAX_ITEMS", "16"))

        # Determinism (optional)
        self._seed_base = os.getenv("NG_SEED_BASE", "").strip()
        self._use_seed = bool(self._seed_base)

        # Per-port rolling frames
        self._frames: Dict[int, Deque[torch.Tensor]] = {}
        self._last_ts: Dict[int, float] = {}

        # Load model once
        loaded = load_ng_checkpoint(ckpt_path, device=self.dev)
        self.policy = NgNitroGenPolicy(loaded, default_game_id=None).to(self.dev).eval()

    def reset_state(self, port: int):
        p = int(port)
        self._frames.pop(p, None)
        self._last_ts.pop(p, None)

    def _get_in_battle(self, gs: Dict[str, Any]) -> bool:
        # Be tolerant: different servers encode this differently.
        # If nothing is present, assume in battle (safer for START gating if you use it).
        for k in ("in_battle", "battle_active", "is_in_battle"):
            v = gs.get(k, None)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and v in (0, 1):
                return bool(v)
        # Some payloads use battle_state > 0
        bs = gs.get("battle_state", None)
        if isinstance(bs, (int, float)):
            return bool(bs > 0)
        return True

    def _push_frame(self, port: int, frame_chw: torch.Tensor) -> torch.Tensor:
        """
        Store and return a [1,T,3,H,W] tensor (padded if needed).
        """
        p = int(port)
        dq = self._frames.get(p)
        if dq is None:
            dq = deque()
            self._frames[p] = dq

        dq.append(frame_chw)
        while len(dq) > self.T:
            dq.popleft()

        if len(dq) == 0:
            # Should never happen because caller pushes.
            x = torch.zeros((1, self.T, 3, self.frame_h, self.frame_w), dtype=torch.float32)
            return x

        # Pad on the left with the oldest frame to reach T
        xs = list(dq)
        if len(xs) < self.T:
            pad = [xs[0]] * (self.T - len(xs))
            xs = pad + xs

        # [T,3,H,W] -> [1,T,3,H,W]
        seq = torch.stack(xs, dim=0).unsqueeze(0).contiguous()
        return seq

    def _seed_for(self, port: int) -> Optional[int]:
        if not self._use_seed:
            return None
        # Simple deterministic evolution: base + floor(time*20) + port
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

        # Default NO_OP command
        def _no_op(debug: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "button_command": {"type": "key_press", "key": _mask_to_bin16(0)},
                "ng_key_bin": "",
                "debug": debug,
            }

        if not self.use_images:
            return _no_op({"reason": "NG requires images; use_images=False"})

        pil = _decode_pil_from_b64(game_state.get("image"))
        if pil is None:
            return _no_op({"reason": "no image in game_state"})

        # Prefer your project preprocessing if available; otherwise do a safe local conversion.
        frame_chw: torch.Tensor
        pp = self.util.get("preprocess_frame", None)
        if callable(pp):
            try:
                # Expecting pp(pil, H, W) or pp(pil) depending on your utils.
                try:
                    frame_chw = pp(pil, self.frame_h, self.frame_w)
                except TypeError:
                    frame_chw = pp(pil)
                if not isinstance(frame_chw, torch.Tensor):
                    raise TypeError("utils.preprocess_frame did not return torch.Tensor")
                if frame_chw.dim() == 4:
                    # sometimes returns [1,C,H,W]
                    frame_chw = frame_chw.squeeze(0)
                if frame_chw.dim() != 3:
                    raise ValueError(f"preprocess_frame returned shape {tuple(frame_chw.shape)}")
                frame_chw = frame_chw.float().contiguous()
            except Exception:
                frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)
        else:
            frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)

        seq = self._push_frame(p, frame_chw)  # [1,T,3,H,W]

        # Run NitroGen (correct API path is inside NgNitroGenPolicy.forward())
        try:
            seq_dev = seq.to(self.dev, non_blocking=True)

            seed = self._seed_for(p)
            if self._emit_ng_raw:
                primary, raw = self.policy(
                    seq_dev,
                    seed=seed,
                    take_step=0,
                    return_continuous=True,
                    return_raw=True,
                    raw_max_items=self._raw_max_items,
                )
                action_vec = primary  # [B,D]
            else:
                action_vec = self.policy(
                    seq_dev,
                    seed=seed,
                    take_step=0,
                    return_continuous=True,
                    return_raw=False,
                )
                raw = None

            # [1,D] -> [D]
            action_1d = action_vec.squeeze(0).detach().to("cpu", non_blocking=True).float().contiguous()

        except (NgPolicyError, Exception) as e:
            return _no_op({"reason": "ng_inference_failed", "error": repr(e)[:400]})

        # Derive NG "intent" as logical buttons (this is what we will log as raw intent)
        intended_logical = _intent_from_action_vec(action_1d, self.schema)

        inside_window = bool(game_state.get("inside_window", False))
        in_battle = self._get_in_battle(game_state)

        # Convert intended -> raw bitmask (pre-gating)
        ng_intended_mask = _logical_buttons_to_mask(
            self.key_bits, intended_logical, button_alias=self.button_alias
        )
        ng_key_bin = _mask_to_bin16(ng_intended_mask)

        # Apply gating/rules for what we actually send
        mapped_logical = _apply_gating(
            intended_logical,
            inside_window=inside_window,
            allow_actions_in_window=self.allow_actions_in_window,
            in_battle=in_battle,
            forbid_actions_in_battle=self.forbid_actions_in_battle,
        )
        mapped_mask = _logical_buttons_to_mask(
            self.key_bits, mapped_logical, button_alias=self.button_alias
        )
        mapped_key_bin = _mask_to_bin16(mapped_mask)

        debug: Dict[str, Any] = {
            "schema": {
                "deadzone": self.schema.deadzone,
                "button_threshold": self.schema.button_threshold,
                "move_x_idx": self.schema.move_x_idx,
                "move_y_idx": self.schema.move_y_idx,
                "a_idx": self.schema.a_idx,
                "b_idx": self.schema.b_idx,
                "start_idx": self.schema.start_idx,
            },
            "context": {
                "inside_window": inside_window,
                "in_battle": in_battle,
                "allow_actions_in_window": self.allow_actions_in_window,
                "forbid_actions_in_battle": list(self.forbid_actions_in_battle),
            },
            "intent": {
                "action_vec_sample": action_1d[:16].tolist(),
                "action_dim": int(action_1d.numel()),
                "intended_logical": intended_logical,
                "mapped_logical": mapped_logical,
                "ng_intended_mask_int": int(ng_intended_mask),
                "mapped_mask_int": int(mapped_mask),
            },
        }
        if raw is not None:
            debug["ng_raw"] = raw  # already JSON-safe summary

        return {
            "button_command": {"type": "key_press", "key": mapped_key_bin},
            # ✅ This is what DebugState will store as "NG wants to press (raw)"
            "ng_key_bin": ng_key_bin,
            # keep everything else in debug so you can inspect mapping/inference health
            "debug": debug,
        }
