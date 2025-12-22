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
import torchvision.transforms.functional as TF
from PIL import Image
from ng_policy import NgNitroGenPolicy, load_ng_checkpoint, NgPolicyError
import shutil
import math

# -----------------------------------------------------------------------------
# Dashboard & Logging Utilities
# -----------------------------------------------------------------------------

# Make sure this matches your model's output order exactly
BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER',
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST',
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP'
]

def print_action_dashboard(probs, chosen_idx, fps=0, threshold=0.05):
    """
    probs: List of probabilities for all buttons
    chosen_idx: The index of the highest probability
    fps: Current inference FPS
    threshold: The activation threshold (usually 0.05)
    """
    # ANSI Colors
    GREEN = '\033[92m'    # Active
    YELLOW = '\033[93m'   # Contender
    GRAY = '\033[90m'     # Inactive
    RED = '\033[91m'      # Selected but weak
    RESET = '\033[0m'
    
    # 1. Determine if the "Selected" action is actually strong enough to press
    chosen_prob = float(probs[chosen_idx])
    
    if chosen_prob > threshold:
        # Valid press
        selected_text = f"{GREEN}{BUTTON_TOKENS[chosen_idx]} ({chosen_prob:.2f}){RESET}"
    else:
        # Highest value, but below threshold (Neutral state)
        selected_text = f"{GRAY}NEUTRAL (Max: {BUTTON_TOKENS[chosen_idx]} {chosen_prob:.2f}){RESET}"

    output = [f"⚡ {fps:.1f} FPS | Action: {selected_text}"]
    
    # 2. Format grid
    row = ""
    for i, token in enumerate(BUTTON_TOKENS):
        if i >= len(probs): break
        
        p = float(probs[i])
        
        # VISUAL LOGIC
        if i == chosen_idx and p > threshold:
            # This is the winner AND it's pressed
            color = GREEN
            prefix = ">>"
        elif p > threshold:
            # Pressed, but not the max (e.g. running + shooting)
            color = YELLOW
            prefix = " +"
        elif i == chosen_idx:
            # Max value, but not pressed (Ghost/Noise)
            color = GRAY
            prefix = " ~"
        else:
            # Low value
            color = GRAY
            prefix = "  "
            
        entry = f"{color}{prefix} {token[:10]:<10} {p:.2f}{RESET}"
        row += entry + "  "
        
        if (i + 1) % 4 == 0:
            output.append(row)
            row = ""
            
    if row: output.append(row)
    
    print("\n".join(output))
    print("-" * 50)

def _resolve_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    dt = os.getenv("NG_DTYPE", "").strip().lower()
    if device.type != "cuda": return None
    if dt == "fp16": return torch.float16
    if dt == "fp32": return None
    return torch.bfloat16

# -----------------------------------------------------------------------------
# Global Batch Manager
# -----------------------------------------------------------------------------
class GlobalBatchManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, policy, device, use_fp16=False):
        self.policy = policy
        self.device = device
        self.use_fp16 = use_fp16 
        self.autocast_dtype = _resolve_autocast_dtype(torch.device(device))
        self.batch_lock = threading.Lock()
        self.pending_inputs = []
        self.batch_timeout = 0.002 

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

        frames = [item["frame"] for item in batch_data]
        batch_tensor = torch.cat(frames, dim=0)

        try:
            batch_dev = batch_tensor.to(self.device, non_blocking=True).to(dtype=torch.float32)
            seed = batch_data[0]["seed"]
            emit_raw = batch_data[0]["emit_raw"]
            
            autocast_enabled = (self.autocast_dtype is not None)
            
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
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
# Action Schema & Helpers
# -----------------------------------------------------------------------------
_TRAIN_BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER',
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST',
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP'
]

def _btn_index(name: str) -> int:
    name = name.strip().upper()
    try:
        return 4 + _TRAIN_BUTTON_TOKENS.index(name)
    except ValueError:
        return -1

THRESHOLD = 0.1

@dataclass(frozen=True)
class NgActionSchema:
    axis_leftx: int = 0
    axis_lefty: int = 1
    dpad_up: int = _btn_index("DPAD_UP")
    dpad_down: int = _btn_index("DPAD_DOWN")
    dpad_left: int = _btn_index("DPAD_LEFT")
    dpad_right: int = _btn_index("DPAD_RIGHT")
    
    a_btn: str = os.getenv("NG_BTN_A", "SOUTH").strip().upper()
    b_btn: str = os.getenv("NG_BTN_B", "EAST").strip().upper()
    start_btn: str = os.getenv("NG_BTN_START", "START").strip().upper()
    
    # NEW: Full GBA Support
    l_btn: int = _btn_index("LEFT_SHOULDER")
    r_btn: int = _btn_index("RIGHT_SHOULDER")
    select_btn: int = _btn_index("BACK") # 'Back' is usually Select
    
    btn_activation: str = os.getenv("NG_BTN_ACTIVATION", "raw01").strip().lower()
    deadzone: float = 0.10
    # Lower threshold even more to capture flicker
    button_threshold: float = float(os.getenv("NG_BTN_THRESH", "0.4")) 

    @staticmethod
    def from_env() -> "NgActionSchema":
        dz = float(os.getenv("NG_MOVE_DEADZONE", "0.10"))
        bt = float(os.getenv("NG_BTN_THRESH", THRESHOLD))
        act = os.getenv("NG_BTN_ACTIVATION", "raw01").strip().lower()
        return NgActionSchema(deadzone=dz, button_threshold=bt, btn_activation=act)

def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _axis_act(x: float) -> float:
    return float(math.tanh(float(x)))

def _btn_act_raw01(x: float) -> float:
    x = float(x)
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _btn_act_logits(x: float) -> float:
    return _sigmoid(x)

def _safe_get(action_vec_1d: torch.Tensor, idx: int) -> float:
    if idx < 0: return 0.0
    if action_vec_1d.numel() <= idx: return 0.0
    return float(action_vec_1d[idx].item())

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
    
    # 1. Force Native Resolution (240x160)
    NATIVE_W, NATIVE_H = 240, 160
    img = pil_img.resize((NATIVE_W, NATIVE_H), resample=Image.NEAREST)
    
    # 2. Center Pad
    new_img = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    left = (out_w - NATIVE_W) // 2
    top = (out_h - NATIVE_H) // 2
    new_img.paste(img, (left, top))
    
    # 3. To Tensor [0.0, 1.0]
    tensor = TF.to_tensor(new_img)
    
    # 4. --- NEW: SigLIP Normalization ---
    # This is critical for matching the pre-trained vision encoder
    tensor = (tensor - 0.5) / 0.5
    
    return tensor

_DEFAULT_BUTTON_ALIAS = { "A": "Z", "B": "X", "START": "RETURN", "SELECT": "BACKSPACE", "L": "A", "R": "S" }

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
    btns: List[str] = []
    
    if schema.btn_activation == "logits":
        btn_decode = _btn_act_logits
    else:
        btn_decode = _btn_act_raw01

    up_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_up))
    dn_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_down))
    lf_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_left))
    rt_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_right))
    
    used_dpad = False
    if up_v >= schema.button_threshold: btns.append("UP"); used_dpad = True
    if dn_v >= schema.button_threshold: btns.append("DOWN"); used_dpad = True
    if lf_v >= schema.button_threshold: btns.append("LEFT"); used_dpad = True
    if rt_v >= schema.button_threshold: btns.append("RIGHT"); used_dpad = True

    # --- FIX: DISABLE ANALOG STICK READING FOR GBA ---
    # Foundation models often output noise (e.g. 0.12) on sticks. 
    # Since GBA is digital, we only want to listen to the D-Pad.
    # We comment this out so the "Stick Drift" never triggers a button press.
    
    # if not used_dpad:
    #     mx = _axis_act(_safe_get(action_vec_1d, schema.axis_leftx))
    #     my = _axis_act(_safe_get(action_vec_1d, schema.axis_lefty))
    #     if abs(mx) >= schema.deadzone: btns.append("RIGHT" if mx > 0 else "LEFT")
    #     if abs(my) >= schema.deadzone: btns.append("DOWN" if my > 0 else "UP")
    # -------------------------------------------------

    a_idx = _btn_index(schema.a_btn)
    b_idx = _btn_index(schema.b_btn)
    s_idx = _btn_index(schema.start_btn)
    
    l_idx = schema.l_btn
    r_idx = schema.r_btn
    sel_idx = schema.select_btn

    if btn_decode(_safe_get(action_vec_1d, a_idx)) >= schema.button_threshold: btns.append("A")
    if btn_decode(_safe_get(action_vec_1d, b_idx)) >= schema.button_threshold: btns.append("B")
    if btn_decode(_safe_get(action_vec_1d, s_idx)) >= schema.button_threshold: btns.append("START")
    if btn_decode(_safe_get(action_vec_1d, sel_idx)) >= schema.button_threshold: btns.append("SELECT")
    if btn_decode(_safe_get(action_vec_1d, l_idx)) >= schema.button_threshold: btns.append("L")
    if btn_decode(_safe_get(action_vec_1d, r_idx)) >= schema.button_threshold: btns.append("R")
    
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
        self.forbid_actions_in_battle = [] 
        
        # --- INPUT SMOOTHING STATE ---
        self.held_buttons: Dict[str, int] = {} # {button_name: frames_remaining}
        
        # FIX FOR 27 FPS vs 60 FPS: 
        # Reduced from 5 to 1. Holding for 5 frames at 27fps = ~185ms stuck button.
        self.sticky_frames = 1  

        if os.getenv("NG_ALLOW_IN_WINDOW", "1") == "1":
            self.allow_actions_in_window = True
        
        self.button_alias = dict(button_alias or _DEFAULT_BUTTON_ALIAS)
        self.schema = NgActionSchema.from_env()
        
        self.frame_skip = int(os.getenv("NG_FRAME_SKIP", "4"))
        self.frame_counter = 0
        self.last_decision = None
        self.last_inference_time = time.time() # For FPS calc
        
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
            
            if os.getenv("NG_CAST_WEIGHTS", "0").strip() in ("1", "true", "True"):
                if os.getenv("NG_DTYPE", "").strip().lower() == "fp16":
                    print("Casting policy weights to FP16...")
                    policy.half()
            
            # --- WARMUP ---
            try:
                print("⚡ Warming up model...")
                with torch.no_grad():
                    dummy = torch.zeros((1, self.T, 3, self.frame_h, self.frame_w), dtype=torch.float32, device=self.dev)
                    if self.use_fp16: dummy = dummy.half()
                    policy(dummy, seed=None, take_step=0, return_continuous=True)
                print("✅ Warmup complete.")
            except Exception as e:
                print(f"⚠️ Warmup failed: {e}")
                
            self.batch_mgr = GlobalBatchManager.get(policy, self.dev, self.use_fp16)
        else:
            self.batch_mgr = GlobalBatchManager.get()

    def reset_state(self, port: int):
        p = int(port)
        self._frames.pop(p, None)
        self._last_ts.pop(p, None)
        self.frame_counter = 0
        self.last_decision = None
        self.held_buttons.clear() # Reset sticky buttons

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
        self._last_ts[p] = time.time()
        self.frame_counter += 1

        if self.last_decision and (self.frame_counter % self.frame_skip != 0):
            return self.last_decision

        def _no_op(debug: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "button_command": {"type": "key_press", "key": _mask_to_bin16(0)},
                "ng_key_bin": "",
                "debug": debug,
            }

        if not self.use_images:
            return _no_op({"reason": "NG requires images"})

        b64_str = game_state.get("image")
        if not b64_str: return _no_op({"reason": "no image"})

        try:
            pil = _decode_pil_from_b64(b64_str)
            if pil is None: return _no_op({"reason": "bad image"})
            frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)
        except Exception as e:
            return _no_op({"reason": f"decode_err: {e}"})

        if frame_chw.is_cuda: frame_chw = frame_chw.cpu()
        frame_chw = frame_chw.to(dtype=torch.float32, copy=False)

        # === DEBUG: SAVE WHAT THE MODEL SEES ===
        # Save one frame every 10 seconds (approx 300 frames at 30fps) to check quality
        if self.frame_counter % 300 == 0:
            try:
                from torchvision.utils import save_image
                # Save to the current directory
                save_image(frame_chw, f"debug_inference_{port}.png")
                print(f"📸 Saved debug image: debug_inference_{port}.png")
            except ImportError:
                print("⚠️ Could not save debug image (torchvision not found?)")
        # =======================================

        seq = self._push_frame(p, frame_chw)

        try:
            seed = self._seed_for(p)
            action_vec, raw = self.batch_mgr.infer(seq, emit_raw=self._emit_ng_raw, seed=seed)
            action_1d = action_vec.squeeze(0)
            
            # --- START DASHBOARD LOGGING ---
            if port == 12350:
                now = time.time()
                fps = 1.0 / (now - self.last_inference_time) if hasattr(self, 'last_inference_time') else 0.0
                self.last_inference_time = now

                if self.schema.btn_activation == "logits":
                    raw_probs = torch.sigmoid(action_1d).tolist()
                else:
                    raw_probs = action_1d.tolist()

                # --- NEW: DEBUG FOUNDATION MODEL NOISE ---
                sticks = raw_probs[:4]
                print(f"\n🎮 STICKS: LX={sticks[0]:.3f} LY={sticks[1]:.3f} RX={sticks[2]:.3f} RY={sticks[3]:.3f}")
                # ----------------------------------------
                
                # Slice: skip first 4 (sticks)
                btn_data = raw_probs[4:] 
                
                # --- FIX: GHOST MASKING ---
                # These tokens are unused in GBA but have high "bias" (approx 0.5).
                # We force them to -1.0 so they never win the selection.
                IGNORED_TOKENS = {
                    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP', 
                    'LEFT_THUMB', 'RIGHT_THUMB', 'GUIDE', 'LEFT_TRIGGER', 'RIGHT_TRIGGER',
                    'WEST', 'NORTH' # Assuming BN6 uses A(South)/B(East)
                }

                masked_data = list(btn_data) # Copy to avoid altering original tensor data if needed elsewhere
                for i, token in enumerate(BUTTON_TOKENS):
                    if i < len(masked_data) and token in IGNORED_TOKENS:
                        masked_data[i] = -1.0
                
                # Find Chosen Index for display
                if masked_data:
                    chosen_val = max(masked_data)
                    chosen_idx = masked_data.index(chosen_val)
                    # Pass the MASKED data to the print function so the green text is correct
                    print_action_dashboard(masked_data, chosen_idx, fps, threshold=THRESHOLD)
            # --- END DASHBOARD LOGGING ---

        except Exception as e:
            return _no_op({"reason": "ng_batch_failed", "error": str(e)})

        # 1. Get raw intent from model
        # NOTE: _intent_from_action_vec ALREADY ignores the Right Stick tokens 
        # because they aren't in your NgActionSchema. The issue was just visual in the log.
        raw_intent = _intent_from_action_vec(action_1d, self.schema)
        
        # 2. Update Sticky State
        expired = []
        for btn in self.held_buttons:
            self.held_buttons[btn] -= 1
            if self.held_buttons[btn] <= 0:
                expired.append(btn)
        for btn in expired:
            del self.held_buttons[btn]
            
        for btn in raw_intent:
            self.held_buttons[btn] = self.sticky_frames
            
        sticky_intent = list(self.held_buttons.keys())

        # 3. Calculate masks
        ng_intended_mask = _logical_buttons_to_mask(self.key_bits, sticky_intent, button_alias=self.button_alias)
        ng_key_bin = _mask_to_bin16(ng_intended_mask)

        mapped_logical = _apply_gating(
            sticky_intent,
            inside_window=bool(game_state.get("inside_window", False)),
            allow_actions_in_window=self.allow_actions_in_window,
            in_battle=self._get_in_battle(game_state),
            forbid_actions_in_battle=self.forbid_actions_in_battle,
        )
        mapped_mask = _logical_buttons_to_mask(self.key_bits, mapped_logical, button_alias=self.button_alias)
        mapped_key_bin = _mask_to_bin16(mapped_mask)

        debug = {
            "intent": { "raw": raw_intent, "sticky": sticky_intent, "mapped": mapped_logical },
        }
        if raw is not None: debug["ng_raw"] = raw

        result = {
            "button_command": {"type": "key_press", "key": mapped_key_bin},
            "ng_key_bin": ng_key_bin,
            "debug": debug,
        }
        
        self.last_decision = result
        return result