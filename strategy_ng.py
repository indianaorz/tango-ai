#!/usr/bin/env python3
from __future__ import annotations

import base64
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Import policy loader
from ng_policy import NgNitroGenPolicy, load_ng_checkpoint

# -----------------------------------------------------------------------------
# Token layout (Dynamic - Matches Training)
# -----------------------------------------------------------------------------
try:
    from nitrogen.shared import BUTTON_ACTION_TOKENS as _LIB_TOKENS
    BUTTON_TOKENS = list(_LIB_TOKENS)
    # print(f"[NG] Loaded {len(BUTTON_TOKENS)} canonical button tokens from nitrogen.shared")
except ImportError:
    print("[NG] WARNING: nitrogen.shared not found. Using hardcoded alphabetical fallback (LIKELY INCORRECT).")
    BUTTON_TOKENS = [
        "BACK",
        "DPAD_DOWN",
        "DPAD_LEFT",
        "DPAD_RIGHT",
        "DPAD_UP",
        "EAST",
        "GUIDE",
        "LEFT_SHOULDER",
        "LEFT_THUMB",
        "LEFT_TRIGGER",
        "NORTH",
        "RIGHT_SHOULDER",
        "RIGHT_THUMB",
        "RIGHT_TRIGGER",
        "SOUTH",
        "START",
        "WEST",
        "RIGHT_BOTTOM",
        "RIGHT_LEFT",
        "RIGHT_RIGHT",
        "RIGHT_UP",
    ]

# NitroGen action head output:
#   [buttons(N), j_left(x,y), j_right(x,y)]
N_BUTTONS = len(BUTTON_TOKENS)
IDX_JLEFT = N_BUTTONS
IDX_JRIGHT = N_BUTTONS + 2
ACTION_VEC_DIM = N_BUTTONS + 4


# -----------------------------------------------------------------------------
# Autocast / dtype
# -----------------------------------------------------------------------------
def _resolve_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    dt = os.getenv("NG_DTYPE", "").strip().lower()
    if device.type != "cuda":
        return None
    if dt == "fp16":
        return torch.float16
    if dt == "fp32":
        return None
    return torch.bfloat16


_autocast_enabled = os.getenv("NG_AUTOMATIC_MIXED_PRECISION", "1").strip() == "1"


# -----------------------------------------------------------------------------
# Global Batch Manager (shared across ports/handlers)
# -----------------------------------------------------------------------------
class GlobalBatchManager:
    """
    Batches concurrent inference calls across instances to amortize overhead.

    Thread safety:
      - infer() is thread-safe.
      - one of the callers executes the batch after a short timeout.
    """
    _instance: Optional["GlobalBatchManager"] = None
    _lock = threading.Lock()

    def __init__(self, policy: torch.nn.Module, device: torch.device, use_fp16: bool = False):
        self.policy = policy
        self.device = device
        self.use_fp16 = bool(use_fp16)

        self.autocast_dtype = _resolve_autocast_dtype(torch.device(device))

        self.batch_lock = threading.Lock()
        self.pending_inputs: List[Dict[str, Any]] = []
        self.batch_timeout = float(os.getenv("NG_BATCH_TIMEOUT_S", "0.002").strip() or "0.002")

    @classmethod
    def get(cls, policy: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None, use_fp16: bool = False) -> "GlobalBatchManager":
        with cls._lock:
            if cls._instance is None:
                if policy is None or device is None:
                    raise ValueError("GlobalBatchManager not initialized (policy/device missing).")
                cls._instance = cls(policy, device, use_fp16)
            return cls._instance

    def infer(
        self,
        frame_tensor: torch.Tensor,
        *,
        emit_raw: bool = False,
        seed: Optional[int] = None,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        """
        frame_tensor: [B, V, 3, H, W] float32 on CPU or CUDA
        returns: (primary, raw)
          primary: CPU float32
            - [B, 25] if return_sequence=False and model returns a single step
            - [B, T, 25] if return_sequence=True
        """
        my_event = threading.Event()
        my_result: Dict[str, Any] = {}

        with self.batch_lock:
            self.pending_inputs.append(
                {
                    "event": my_event,
                    "result": my_result,
                    "frame": frame_tensor,
                    "seed": seed,
                    "emit_raw": bool(emit_raw),
                    "return_sequence": bool(return_sequence),
                }
            )
            should_trigger = (len(self.pending_inputs) == 1)

        if should_trigger:
            time.sleep(self.batch_timeout)
            self._execute_batch()
        else:
            my_event.wait()

        if "error" in my_result:
            raise RuntimeError(my_result["error"])

        return my_result["action_vec"], my_result["raw"]

    def _execute_batch(self) -> None:
        with self.batch_lock:
            batch_data = self.pending_inputs[:]
            self.pending_inputs.clear()

        if not batch_data:
            return

        frames = [item["frame"] for item in batch_data]
        try:
            batch_tensor = torch.cat(frames, dim=0)
        except Exception as e:
            for item in batch_data:
                item["result"]["error"] = f"batch_cat_failed: {e!r}"
                item["event"].set()
            return

        try:
            batch_dev = batch_tensor.to(self.device, non_blocking=True).to(dtype=torch.float32)

            seed = batch_data[0]["seed"]
            emit_raw = bool(batch_data[0]["emit_raw"])
            return_sequence = bool(batch_data[0]["return_sequence"])

            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype if _autocast_enabled else torch.float32,
                enabled=bool(_autocast_enabled),
            ):
                if emit_raw:
                    primary, raw_out = self.policy(
                        batch_dev,
                        seed=seed,
                        take_step=0,
                        return_continuous=True,
                        return_raw=True,
                        raw_max_items=int(os.getenv("NG_DEBUG_RAW_MAX_ITEMS", "8")),
                        return_sequence=return_sequence,
                    )
                    raw_list = [raw_out] * len(batch_data)
                else:
                    primary = self.policy(
                        batch_dev,
                        seed=seed,
                        take_step=0,
                        return_continuous=True,
                        return_raw=False,
                        return_sequence=return_sequence,
                    )
                    raw_list = [None] * len(batch_data)

            primary_cpu = primary.detach().float().cpu()

            # primary_cpu is either:
            #   [B, 25] OR [B, T, 25]
            for i, item in enumerate(batch_data):
                item["result"]["action_vec"] = primary_cpu[i].unsqueeze(0)  # [1, ...]
                item["result"]["raw"] = raw_list[i]

        except Exception as e:
            for item in batch_data:
                item["result"]["error"] = str(e)
        finally:
            for item in batch_data:
                item["event"].set()


# -----------------------------------------------------------------------------
# Action Schema & helpers
# -----------------------------------------------------------------------------
_TRAIN_BUTTON_TOKENS = list(BUTTON_TOKENS)


def _btn_index(name: str) -> int:
    name = name.strip().upper()
    try:
        return _TRAIN_BUTTON_TOKENS.index(name)
    except ValueError:
        return -1


@dataclass(frozen=True)
class NgActionSchema:
    axis_leftx: int = IDX_JLEFT + 0
    axis_lefty: int = IDX_JLEFT + 1
    axis_rightx: int = IDX_JRIGHT + 0
    axis_righty: int = IDX_JRIGHT + 1

    dpad_up: int = _btn_index("DPAD_UP")
    dpad_down: int = _btn_index("DPAD_DOWN")
    dpad_left: int = _btn_index("DPAD_LEFT")
    dpad_right: int = _btn_index("DPAD_RIGHT")

    a_btn: str = os.getenv("NG_BTN_A", "EAST").strip().upper()
    b_btn: str = os.getenv("NG_BTN_B", "SOUTH").strip().upper()
    start_btn: str = os.getenv("NG_BTN_START", "START").strip().upper()

    l_btn: int = _btn_index("LEFT_SHOULDER")
    r_btn: int = _btn_index("RIGHT_SHOULDER")
    select_btn: int = _btn_index("BACK")

    btn_activation: str = os.getenv("NG_BTN_ACTIVATION", "raw01").strip().lower()
    deadzone: float = float(os.getenv("NG_MOVE_DEADZONE", "0.10").strip() or "0.10")
    button_threshold: float = float(os.getenv("NG_BTN_THRESH", "0.4").strip() or "0.4")

    @staticmethod
    def from_env() -> "NgActionSchema":
        return NgActionSchema(
            deadzone=float(os.getenv("NG_MOVE_DEADZONE", "0.10").strip() or "0.10"),
            button_threshold=float(os.getenv("NG_BTN_THRESH", "0.4").strip() or "0.4"),
            btn_activation=os.getenv("NG_BTN_ACTIVATION", "raw01").strip().lower(),
        )


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _btn_act_raw01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _btn_act_logits(x: float) -> float:
    return _sigmoid(x)


def _safe_get(action_vec_1d: torch.Tensor, idx: int) -> float:
    if idx < 0:
        return 0.0
    if action_vec_1d.numel() <= idx:
        return 0.0
    return float(action_vec_1d[idx].item())


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
    IMPORTANT:
      - force native GBA res (240x160) then center-pad to model input (out_w,out_h)
      - SigLIP-style normalize to [-1,1] via (x-0.5)/0.5
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    NATIVE_W, NATIVE_H = 240, 160
    img = pil_img.resize((NATIVE_W, NATIVE_H), resample=Image.NEAREST)

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    left = (out_w - NATIVE_W) // 2
    top = (out_h - NATIVE_H) // 2
    canvas.paste(img, (left, top))

    t = TF.to_tensor(canvas)  # [3,H,W] float in [0,1]
    t = (t - 0.5) / 0.5       # [-1,1]
    return t


_DEFAULT_BUTTON_ALIAS = {
    "A": "Z",
    "B": "X",
    "START": "RETURN",
    "SELECT": "BACKSPACE",
    "L": "A",
    "R": "S",
}


def _mask_to_bin16(mask: int) -> str:
    return format(int(mask) & 0xFFFF, "016b")


def _logical_buttons_to_mask(
    key_bit_positions: Dict[str, int],
    logical_buttons: List[str],
    *,
    button_alias: Dict[str, str],
) -> int:
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
    Map the model output (tokenizer space) -> logical buttons list
    using only D-Pad (GBA is digital; ignore analog drift).
    """
    btns: List[str] = []

    btn_decode = _btn_act_logits if schema.btn_activation == "logits" else _btn_act_raw01

    up_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_up))
    dn_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_down))
    lf_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_left))
    rt_v = btn_decode(_safe_get(action_vec_1d, schema.dpad_right))

    if up_v >= schema.button_threshold:
        btns.append("UP")
    if dn_v >= schema.button_threshold:
        btns.append("DOWN")
    if lf_v >= schema.button_threshold:
        btns.append("LEFT")
    if rt_v >= schema.button_threshold:
        btns.append("RIGHT")

    a_idx = _btn_index(schema.a_btn)
    b_idx = _btn_index(schema.b_btn)
    s_idx = _btn_index(schema.start_btn)

    if btn_decode(_safe_get(action_vec_1d, a_idx)) >= schema.button_threshold:
        btns.append("A")
    if btn_decode(_safe_get(action_vec_1d, b_idx)) >= schema.button_threshold:
        btns.append("B")
    if btn_decode(_safe_get(action_vec_1d, s_idx)) >= schema.button_threshold:
        btns.append("START")

    if btn_decode(_safe_get(action_vec_1d, schema.select_btn)) >= schema.button_threshold:
        btns.append("SELECT")
    if btn_decode(_safe_get(action_vec_1d, schema.l_btn)) >= schema.button_threshold:
        btns.append("L")
    if btn_decode(_safe_get(action_vec_1d, schema.r_btn)) >= schema.button_threshold:
        btns.append("R")

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
        return []
    if in_battle and forbid_actions_in_battle:
        forb = {s.upper() for s in forbid_actions_in_battle}
        btns = [b for b in btns if b.upper() not in forb]
    return btns


# -----------------------------------------------------------------------------
# Strategy: NG Agent with prefetching plan buffer (default overwrite mode)
# -----------------------------------------------------------------------------
class NGAgentStrategy:
    """
    Produces a *continuous* key stream by playing back a model-generated plan.

    Key idea:
      - Always keep an active plan buffer.
      - Prefetch the next plan in a background thread before the active plan runs out.
      - Default behavior: overwrite (swap in new plan immediately when ready),
        controlled by NG_PLAN_REFRESH_MODE.

    Env:
      NG_PLAN_REFRESH_MODE:
        - "overwrite" (DEFAULT): swap to newly prefetched plan as soon as it's ready
        - "prefetch": swap only when current plan ends (still no stall if prefetch keeps up)
      NG_PLAN_PREFETCH_WATERMARK:
        - <=1.0: fraction remaining at which to start prefetch (default 0.5)
        - > 1.0: ticks remaining at which to start prefetch (e.g. 16)
      NG_FRAME_SKIP:
        - kept for compatibility, but in plan mode we always emit every tick
      NG_STICKY_FRAMES:
        - how many ticks to hold newly-intended buttons (default 1)
    """

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
        discrete_actions: List[str],
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
        if os.getenv("NG_ALLOW_IN_WINDOW", "1").strip() == "1":
            self.allow_actions_in_window = True

        self.forbid_actions_in_battle = list(forbid_actions_in_battle or [])
        self.button_alias = dict(button_alias or _DEFAULT_BUTTON_ALIAS)
        self.schema = NgActionSchema.from_env()

        # Sticky buttons
        self.held_buttons: Dict[str, int] = {}
        self.sticky_frames = int(os.getenv("NG_STICKY_FRAMES", "1").strip() or "1")

        # Debug / seeding
        self._emit_ng_raw = os.getenv("NG_DEBUG_RAW", "0").strip().lower() in ("1", "true", "yes")
        self._seed_base = os.getenv("NG_SEED_BASE", "").strip()
        self._use_seed = bool(self._seed_base)
        self.use_fp16 = bool(int(os.getenv("NG_USE_FP16", "1").strip() or "1"))

        # Vision horizon V (from policy/tokenizer when available)
        self.V = 1

        # Per-port frame history for V>1
        self._frames: Dict[int, Deque[torch.Tensor]] = {}
        self._last_ts: Dict[int, float] = {}

        # Plan buffers per port
        self._plan_keys: Dict[int, List[str]] = {}
        self._plan_ng_keys: Dict[int, List[str]] = {}
        self._plan_idx: Dict[int, int] = {}
        self._plan_id: Dict[int, int] = {}

        # Prefetch controls (DEFAULT overwrite as requested)
        self.plan_refresh_mode = os.getenv("NG_PLAN_REFRESH_MODE", "overwrite").strip().lower()
        self.plan_prefetch_watermark = float(os.getenv("NG_PLAN_PREFETCH_WATERMARK", "0.5").strip() or "0.5")

        # Pending (prefetched) plan per port
        self._plan_lock = threading.Lock()
        self._prefetch_inflight: Dict[int, bool] = {}
        self._pending_ready: Dict[int, bool] = {}
        self._pending_keys: Dict[int, List[str]] = {}
        self._pending_ng_keys: Dict[int, List[str]] = {}
        self._pending_plan_id: Dict[int, int] = {}

        # Build / load policy (singleton global)
        if GlobalBatchManager._instance is None:
            print(f"Loading NitroGen policy from {ckpt_path}...")
            loaded = load_ng_checkpoint(ckpt_path, device=self.dev)
            policy = NgNitroGenPolicy(loaded, default_game_id=None).to(self.dev).eval()

            # Prefer trained horizon if present
            vh = None
            for obj in (policy, getattr(policy, "tokenizer", None), getattr(loaded, "tokenizer", None)):
                if obj is None:
                    continue
                vh = getattr(obj, "vision_horizon", None)
                if isinstance(vh, int) and vh > 0:
                    break
            self.V = int(vh) if isinstance(vh, int) and vh > 0 else 1

            # Optional weight casting
            if os.getenv("NG_CAST_WEIGHTS", "0").strip().lower() in ("1", "true", "yes"):
                if os.getenv("NG_DTYPE", "").strip().lower() == "fp16":
                    print("Casting policy weights to FP16...")
                    policy.half()

            # Warmup
            try:
                print("⚡ Warming up model...")
                with torch.no_grad():
                    dummy = torch.zeros((1, self.V, 3, self.frame_h, self.frame_w), dtype=torch.float32, device=self.dev)
                    policy(dummy, seed=None, take_step=0, return_continuous=True, return_sequence=True)
                print("✅ Warmup complete.")
            except Exception as e:
                print(f"⚠️ Warmup failed: {e!r}")

            self.batch_mgr = GlobalBatchManager.get(policy, self.dev, self.use_fp16)
        else:
            self.batch_mgr = GlobalBatchManager.get()

        self.last_decision: Optional[Dict[str, Any]] = None
        self.frame_skip = int(os.getenv("NG_FRAME_SKIP", "1").strip() or "1")
        self.frame_counter = 0

    # -------------------------
    # Per-port lifecycle
    # -------------------------
    def reset_state(self, port: int) -> None:
        p = int(port)
        self._frames.pop(p, None)
        self._last_ts.pop(p, None)

        self._plan_keys.pop(p, None)
        self._plan_ng_keys.pop(p, None)
        self._plan_idx.pop(p, None)
        self._plan_id.pop(p, None)

        with self._plan_lock:
            self._prefetch_inflight.pop(p, None)
            self._pending_ready.pop(p, None)
            self._pending_keys.pop(p, None)
            self._pending_ng_keys.pop(p, None)
            self._pending_plan_id.pop(p, None)

        self.held_buttons.clear()
        self.last_decision = None

    def _get_in_battle(self, gs: Dict[str, Any]) -> bool:
        for k in ("in_battle", "battle_active", "is_in_battle"):
            v = gs.get(k, None)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and v in (0, 1):
                return bool(v)
        bs = gs.get("battle_state", None)
        if isinstance(bs, (int, float)):
            return bool(bs > 0)
        # Conservative default: treat as in-battle so we don't menu-spam unless allowed
        return True

    # -------------------------
    # Frames
    # -------------------------
    def _push_frame(self, port: int, frame_chw: torch.Tensor) -> torch.Tensor:
        V = int(getattr(self, "V", 1))
        if V <= 1:
            return frame_chw.unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]

        p = int(port)
        dq = self._frames.get(p)
        if dq is None:
            dq = deque()
            self._frames[p] = dq

        dq.append(frame_chw)
        while len(dq) > V:
            dq.popleft()

        xs = list(dq)
        if len(xs) < V:
            xs = [xs[0]] * (V - len(xs)) + xs

        return torch.stack(xs, dim=0).unsqueeze(0).contiguous()  # [1,V,3,H,W]

    def _seed_for(self, port: int) -> Optional[int]:
        if not self._use_seed:
            return None
        try:
            base = int(self._seed_base)
        except Exception:
            base = 0
        step = int(time.time() * 20.0)
        return int(base + step + int(port) * 1_000_003)

    # -------------------------
    # Prefetch controls
    # -------------------------
    def _should_prefetch(self, *, remaining: int, total: int) -> bool:
        wm = float(self.plan_prefetch_watermark)
        if total <= 0:
            return True
        if wm <= 1.0:
            frac = float(remaining) / float(total)
            return frac <= wm
        return remaining <= int(wm)

    def _build_plan_from_model(
        self,
        *,
        port: int,
        seq: torch.Tensor,
        seed: Optional[int],
        game_state: Dict[str, Any],
        plan_id: int,
    ) -> Tuple[List[str], List[str], int, Any]:
        """
        Blocking: runs model once, returns (keys, ng_keys, new_plan_id, raw_debug).
        keys/ng_keys are 16-bit binary strings (mapped + raw intent).
        """
        action_seq, raw = self.batch_mgr.infer(
            seq,
            emit_raw=self._emit_ng_raw,
            seed=seed,
            return_sequence=True,
        )

        # action_seq comes back as [1, T, 25] OR [1, 25]
        x = action_seq.squeeze(0)
        if x.ndim == 1:
            # [25] -> treat as T=1
            if x.numel() < N_BUTTONS:
                raise RuntimeError(f"bad_action_vec: {tuple(action_seq.shape)}")
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] < N_BUTTONS:
            raise RuntimeError(f"bad_action_seq_shape: {tuple(action_seq.shape)}")

        T_plan = int(x.shape[0])

        base_plan_keys: List[str] = []
        base_plan_ng_keys: List[str] = []

        # reset sticky per fresh plan
        self.held_buttons.clear()

        for t in range(T_plan):
            action_1d = x[t]
            raw_intent = _intent_from_action_vec(action_1d, self.schema)

            # sticky update
            expired = []
            for btn in list(self.held_buttons.keys()):
                self.held_buttons[btn] -= 1
                if self.held_buttons[btn] <= 0:
                    expired.append(btn)
            for btn in expired:
                del self.held_buttons[btn]
            for btn in raw_intent:
                self.held_buttons[btn] = self.sticky_frames

            sticky_intent = list(self.held_buttons.keys())

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

            base_plan_keys.append(mapped_key_bin)
            base_plan_ng_keys.append(ng_key_bin)

        return base_plan_keys, base_plan_ng_keys, int(plan_id + 1), raw

    def _start_prefetch_if_needed(self, *, port: int, seq: torch.Tensor, seed: Optional[int], game_state: Dict[str, Any]) -> None:
        p = int(port)
        with self._plan_lock:
            if self._prefetch_inflight.get(p, False):
                return
            self._prefetch_inflight[p] = True
            plan_id = int(self._plan_id.get(p, 0))

        def _worker():
            try:
                keys, ng_keys, new_id, _raw = self._build_plan_from_model(
                    port=p, seq=seq, seed=seed, game_state=game_state, plan_id=plan_id
                )
                with self._plan_lock:
                    self._pending_keys[p] = keys
                    self._pending_ng_keys[p] = ng_keys
                    self._pending_plan_id[p] = new_id
                    self._pending_ready[p] = True
            except Exception as e:
                print(f"[NG] Prefetch failed on port {p}: {e!r}")
                with self._plan_lock:
                    self._pending_ready[p] = False
            finally:
                with self._plan_lock:
                    self._prefetch_inflight[p] = False

        threading.Thread(target=_worker, daemon=True).start()

    def _try_swap_in_pending(self, *, port: int) -> bool:
        p = int(port)
        with self._plan_lock:
            if not self._pending_ready.get(p, False):
                return False
            keys = self._pending_keys.get(p) or []
            ng_keys = self._pending_ng_keys.get(p) or []
            new_id = int(self._pending_plan_id.get(p, int(self._plan_id.get(p, 0)) + 1))
            if not keys:
                self._pending_ready[p] = False
                return False

            self._plan_keys[p] = list(keys)
            self._plan_ng_keys[p] = list(ng_keys)
            self._plan_idx[p] = 0
            self._plan_id[p] = new_id

            # clear pending
            self._pending_ready[p] = False
            self._pending_keys.pop(p, None)
            self._pending_ng_keys.pop(p, None)
            self._pending_plan_id.pop(p, None)

            return True

    # -------------------------
    # Main decision API
    # -------------------------
    def decide_action(self, port: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        p = int(port)
        self._last_ts[p] = time.time()
        self.frame_counter += 1

        def _no_op(reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            dbg = {"reason": reason}
            if extra:
                dbg.update(extra)
            return {
                "button_command": {"type": "key_press", "key": _mask_to_bin16(0)},
                "ng_key_bin": "",
                "debug": dbg,
                "action_plan_keys": [],
                "action_plan_ng_keys": [],
            }

        if not self.use_images:
            return _no_op("NG requires images")

        b64_str = game_state.get("image")
        if not b64_str:
            return _no_op("no image")

        pil = _decode_pil_from_b64(b64_str)
        if pil is None:
            return _no_op("bad image")

        try:
            frame_chw = _pil_to_chw_float01(pil, out_h=self.frame_h, out_w=self.frame_w)
        except Exception as e:
            return _no_op("preprocess failed", {"err": repr(e)})

        if frame_chw.is_cuda:
            frame_chw = frame_chw.cpu()
        frame_chw = frame_chw.to(dtype=torch.float32, copy=False)

        seq = self._push_frame(p, frame_chw)
        seed = self._seed_for(p)

        # Active plan state
        plan_keys = self._plan_keys.get(p) or []
        plan_ng_keys = self._plan_ng_keys.get(p) or []
        plan_idx = int(self._plan_idx.get(p, 0))
        plan_id = int(self._plan_id.get(p, 0))

        total_ticks = len(plan_keys)
        remaining = max(0, total_ticks - plan_idx)

        # Default overwrite mode: swap immediately if pending is ready
        if self.plan_refresh_mode == "overwrite":
            self._try_swap_in_pending(port=p)
            plan_keys = self._plan_keys.get(p) or []
            plan_ng_keys = self._plan_ng_keys.get(p) or []
            plan_idx = int(self._plan_idx.get(p, 0))
            total_ticks = len(plan_keys)
            remaining = max(0, total_ticks - plan_idx)
        else:
            # prefetch mode: swap only when current plan ends
            if total_ticks > 0 and plan_idx >= total_ticks:
                self._try_swap_in_pending(port=p)
                plan_keys = self._plan_keys.get(p) or []
                plan_ng_keys = self._plan_ng_keys.get(p) or []
                plan_idx = int(self._plan_idx.get(p, 0))
                total_ticks = len(plan_keys)
                remaining = max(0, total_ticks - plan_idx)

        # If no plan yet, must build one synchronously (first tick only)
        raw = None
        if not plan_keys:
            try:
                keys, ng_keys, new_id, raw = self._build_plan_from_model(
                    port=p, seq=seq, seed=seed, game_state=game_state, plan_id=plan_id
                )
                self._plan_keys[p] = keys
                self._plan_ng_keys[p] = ng_keys
                self._plan_idx[p] = 0
                self._plan_id[p] = new_id

                plan_keys = keys
                plan_ng_keys = ng_keys
                plan_idx = 0
                total_ticks = len(plan_keys)
                remaining = total_ticks
            except Exception as e:
                return _no_op("initial_plan_infer_failed", {"err": repr(e)})

        # Prefetch when under watermark
        if self._should_prefetch(remaining=remaining, total=total_ticks):
            self._start_prefetch_if_needed(port=p, seq=seq, seed=seed, game_state=game_state)

        # Playback one tick
        if plan_idx >= len(plan_keys):
            # best-effort swap now (prefetch miss)
            swapped = self._try_swap_in_pending(port=p)
            if swapped:
                plan_keys = self._plan_keys.get(p) or []
                plan_ng_keys = self._plan_ng_keys.get(p) or []
                plan_idx = int(self._plan_idx.get(p, 0))
            else:
                # hold last if possible
                if plan_keys:
                    plan_idx = len(plan_keys) - 1
                else:
                    return _no_op("plan_empty_after_prefetch_miss")

        step_key = plan_keys[plan_idx]
        step_ng_key = plan_ng_keys[plan_idx] if plan_idx < len(plan_ng_keys) else ""

        # Advance
        self._plan_idx[p] = plan_idx + 1

        remaining_keys = plan_keys[plan_idx:]
        remaining_ng_keys = plan_ng_keys[plan_idx:] if plan_ng_keys else []

        debug_extra = {
            "plan_id": int(self._plan_id.get(p, 0)),
            "plan_step_idx": int(plan_idx),
            "plan_total_ticks": int(len(plan_keys)),
            "plan_remaining_ticks": int(max(0, len(plan_keys) - (plan_idx + 1))),
            "prefetch_mode": self.plan_refresh_mode,
            "prefetch_watermark": float(self.plan_prefetch_watermark),
        }

        with self._plan_lock:
            debug_extra["prefetch_inflight"] = bool(self._prefetch_inflight.get(p, False))
            debug_extra["prefetch_pending_ready"] = bool(self._pending_ready.get(p, False))

        if raw is not None:
            debug_extra["ng_raw"] = raw

        result = {
            "button_command": {"type": "key_press", "key": step_key},
            "ng_key_bin": step_ng_key,
            "action_plan_keys": remaining_keys,
            "action_plan_ng_keys": remaining_ng_keys,
            "debug": debug_extra,
        }
        self.last_decision = result
        return result