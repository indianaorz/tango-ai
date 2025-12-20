# ── Begin: strategy_ng.py ──
# strategy_ng.py
from __future__ import annotations

import base64
import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torchvision.io
import torch.nn.functional as F
from PIL import Image

from ng_policy import NgNitroGenPolicy, load_ng_checkpoint, NgPolicyError

# -----------------------------------------------------------------------------
# Global Batch Manager
# -----------------------------------------------------------------------------
class GlobalBatchManager:
    """
    Collects frames from multiple threads/agents, runs ONE model inference,
    and distributes results back.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, policy, device, use_fp16=False):
        self.policy = policy
        self.device = device
        self.use_fp16 = use_fp16
        
        self.batch_timeout = 0.005  # Wait up to 5ms for other agents
        self.pending_inputs = []    # [(event, result_container, tensor_frame), ...]
        self.batch_lock = threading.Lock()

    @classmethod
    def get(cls, policy=None, device=None, use_fp16=False):
        with cls._lock:
            if cls._instance is None:
                if policy is None: raise ValueError("BatchManager not init")
                cls._instance = cls(policy, device, use_fp16)
            return cls._instance

    def infer(self, frame_tensor, emit_raw=False, seed=None):
        my_event = threading.Event()
        my_result = {}
        
        with self.batch_lock:
            self.pending_inputs.append({
                "event": my_event,
                "result": my_result,
                "frame": frame_tensor,
                "seed": seed,
                "emit_raw": emit_raw
            })
            should_trigger = (len(self.pending_inputs) == 1)

        if should_trigger:
            time.sleep(self.batch_timeout) 
            self._execute_batch()
        else:
            my_event.wait()

        if "error" in my_result:
            raise RuntimeError(my_result["error"])
        return my_result["action_vec"], my_result["raw"]

    def _execute_batch(self):
        with self.batch_lock:
            batch_data = self.pending_inputs[:]
            self.pending_inputs.clear()

        if not batch_data: return

        # Stack: List[1, T, C, H, W] -> [B, T, C, H, W]
        frames = [item["frame"] for item in batch_data]
        batch_tensor = torch.cat(frames, dim=0)

        try:
            batch_dev = batch_tensor.to(self.device, non_blocking=True)
            if self.use_fp16:
                batch_dev = batch_dev.half()

            seed = batch_data[0]["seed"]
            emit_raw = batch_data[0]["emit_raw"]

            with torch.no_grad():
                if emit_raw:
                    primary, raw_out = self.policy(
                        batch_dev, seed=seed, take_step=0,
                        return_continuous=True, return_raw=True, raw_max_items=8
                    )
                    raw_list = [raw_out] * len(batch_data)
                else:
                    primary = self.policy(
                        batch_dev, seed=seed, take_step=0,
                        return_continuous=True, return_raw=False
                    )
                    raw_list = [None] * len(batch_data)

            primary_cpu = primary.detach().float().cpu()
            
            for i, item in enumerate(batch_data):
                item["result"]["action_vec"] = primary_cpu[i].unsqueeze(0)
                item["result"]["raw"] = raw_list[i]

        except Exception as e:
            print(f"Batch Inference Failed: {e}")
            for item in batch_data:
                item["result"]["error"] = str(e)
        finally:
            for item in batch_data:
                item["event"].set()


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
# Helpers
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

        # Frame Skipping Config
        # 4 = 1 action every 4 frames (effectively 15Hz decisions on 60Hz game)
        self.frame_skip = int(os.getenv("NG_FRAME_SKIP", "4"))
        self.frame_counter = 0
        self.last_decision = None

        # [NEW] Force allow actions in window via ENV
        if os.getenv("NG_ALLOW_IN_WINDOW", "1") == "1":
            self.allow_actions_in_window = True
            print("NG: Actions inside window ENABLED via env var.")

        self._emit_ng_raw = os.getenv("NG_DEBUG_RAW", "0").strip() in ("1", "true", "True")
        self._raw_max_items = int(os.getenv("NG_DEBUG_RAW_MAX_ITEMS", "16"))
        self._seed_base = os.getenv("NG_SEED_BASE", "").strip()
        self._use_seed = bool(self._seed_base)
        self.use_fp16 = bool(int(os.getenv("NG_USE_FP16", "1")))
        
        self._frames: Dict[int, Deque[torch.Tensor]] = {}
        self._last_ts: Dict[int, float] = {}

        if GlobalBatchManager._instance is None:
            print(f"Loading NitroGen policy from {ckpt_path}...")
            loaded = load_ng_checkpoint(ckpt_path, device=self.dev)
            policy = NgNitroGenPolicy(loaded, default_game_id=None).to(self.dev).eval()
            
            if self.use_fp16:
                print("Enabling FP16...")
                policy.half()

            if hasattr(torch, "compile"):
                print("Compiling model (default mode)...")
                try:
                    policy = torch.compile(policy)
                except Exception as e:
                    print(f"Warning: torch.compile failed ({e})")
            
            print("⚡ Warming up model... (Wait ~30s)")
            try:
                with torch.no_grad():
                    dummy = torch.zeros((1, self.T, 3, self.frame_h, self.frame_w), dtype=torch.float32, device=self.dev)
                    if self.use_fp16: dummy = dummy.half()
                    policy(dummy, seed=None, take_step=0, return_continuous=True)
                print("✅ Warmup complete.")
            except Exception as e:
                print(f"⚠️ Warmup failed: {e}")

            self.batch_mgr = GlobalBatchManager.get(policy, self.dev, self.use_fp16)
        else:
            print("Reusing existing BatchManager.")
            self.batch_mgr = GlobalBatchManager.get()

    def reset_state(self, port: int):
        p = int(port)
        self._frames.pop(p, None)
        self._last_ts.pop(p, None)
        self.frame_counter = 0
        self.last_decision = None

    def _get_in_battle(self, gs: Dict[str, Any]) -> bool:
        for k in ("in_battle", "battle_active", "is_in_battle"):
            v = gs.get(k, None)
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)) and v in (0, 1): return bool(v)
        bs = gs.get("battle_state", None)
        if isinstance(bs, (int, float)): return bool(bs > 0)
        return True

    def _push_frame(self, port: int, frame_chw: torch.Tensor) -> torch.Tensor:
        if self.T == 1:
            return frame_chw.unsqueeze(0).unsqueeze(0) 

        p = int(port)
        dq = self._frames.get(p)
        if dq is None:
            dq = deque()
            self._frames[p] = dq

        dq.append(frame_chw)
        while len(dq) > self.T:
            dq.popleft()

        xs = list(dq)
        if len(xs) < self.T:
            pad = [xs[0]] * (self.T - len(xs))
            xs = pad + xs

        seq = torch.stack(xs, dim=0).unsqueeze(0).contiguous()
        return seq

    def _seed_for(self, port: int) -> Optional[int]:
        if not self._use_seed: return None
        try: base = int(self._seed_base)
        except: base = 0
        step = int(time.time() * 20.0)
        return int(base + step + int(port) * 1000003)

    def decide_action(self, port: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        p = int(port)
        now = time.time()
        self._last_ts[p] = now
        self.frame_counter += 1

        # [FRAME SKIPPING]
        # Return last decision immediately if skipping. 
        # This saves image decoding + inference time.
        if self.last_decision and (self.frame_counter % self.frame_skip != 0):
            # Important: if T > 1, we technically 'miss' a frame in history here.
            # Since T=1 (stateless), this is perfectly fine.
            return self.last_decision

        def _no_op(debug: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "button_command": {"type": "key_press", "key": _mask_to_bin16(0)},
                "ng_key_bin": "",
                "debug": debug,
            }

        if not self.use_images:
            return _no_op({"reason": "NG requires images"})

        # [OPTIMIZATION] Direct GPU Decode
        b64_str = game_state.get("image")
        if not b64_str: return _no_op({"reason": "no image"})

        try:
            # 1. Decode base64 (CPU)
            raw_bytes = base64.b64decode(b64_str)
            # 2. Create byte tensor (CPU -> Pinned Mem would be better, but simple tensor is OK)
            byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            
            # 3. Decode JPEG directly to GPU
            # This requires torchvision 0.13+ and a compatible backend (nvjpeg)
            # If not available, it might fallback to cpu, which is still okay.
            frame_gpu = torchvision.io.decode_jpeg(byte_tensor, device=self.dev)

            # 4. Resize & Normalize on GPU
            # [C, H, W] -> [1, C, H, W] for interpolate
            frame_gpu = frame_gpu.unsqueeze(0).float()
            frame_chw = F.interpolate(
                frame_gpu, 
                size=(self.frame_h, self.frame_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) / 255.0

        except Exception as e:
            # Fallback to PIL (CPU)
            # print(f"GPU Decode Warning: {e}") # Uncomment to debug why GPU decode fails
            pil = _decode_pil_from_b64(b64_str)
            if pil is None: return _no_op({"reason": "bad image"})
            frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)

        seq = self._push_frame(p, frame_chw)

        # [BATCH INFERENCE]
        try:
            seed = self._seed_for(p)
            action_vec, raw = self.batch_mgr.infer(seq, emit_raw=self._emit_ng_raw, seed=seed)
            action_1d = action_vec.squeeze(0)

        except Exception as e:
            return _no_op({"reason": "ng_batch_failed", "error": str(e)})

        # Interpret
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

        result = {
            "button_command": {"type": "key_press", "key": mapped_key_bin},
            "ng_key_bin": ng_key_bin,
            "debug": debug,
        }
        
        self.last_decision = result
        return result
# ── End: strategy_ng.py ──