from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class HistoryEntry:
    ts: float
    inside_window: bool
    action_type: str

    ng_key_bin: str
    ng_key_int: int
    ng_pressed_buttons: List[str]

    mapped_key_bin: str
    mapped_key_int: int
    mapped_pressed_buttons: List[str]


@dataclass
class PortDebugSnapshot:
    ts: float = 0.0
    inside_window: bool = False
    action_type: str = ""

    # "mapped" = what we actually send to the game (current tick)
    mapped_key_bin: str = ""
    mapped_key_int: int = 0
    mapped_pressed_buttons: List[str] = None

    # "ng" = raw model intent (if provided) (current tick)
    ng_key_bin: str = ""
    ng_key_int: int = 0
    ng_pressed_buttons: List[str] = None

    # FUTURE BUFFER (predicted/queued next actions)
    # Each step is { "ng_pressed_buttons": [...], "mapped_pressed_buttons": [...] } (either side may be missing)
    next_actions: Optional[List[Dict[str, Any]]] = None

    jpg_bytes: Optional[bytes] = None
    img_w: int = 0
    img_h: int = 0

    # Inference metrics
    infer_count_total: int = 0
    infer_hz: float = 0.0
    infer_window_s: float = 0.0
    last_update_age_s: float = 0.0



class DebugState:
    """
    Thread-safe store for per-port debug data and history.
    Uses Condition variables to allow the server to stream updates
    immediately (push) instead of polling.
    """

    def __init__(
        self,
        key_bit_positions: Dict[str, int],
        *,
        rate_window_s: float = 5.0,
        max_events_per_port: int = 4096,
        max_history_per_port: int = 200,
    ):
        if rate_window_s <= 0:
            raise ValueError("rate_window_s must be > 0")
        if max_history_per_port <= 0:
            raise ValueError("max_history_per_port must be > 0")

        self._lock = threading.Lock()
        self._by_port: Dict[int, PortDebugSnapshot] = {}

        # Original mapping (as provided)
        self._key_bits = dict(key_bit_positions or {})

        # Normalized lookup so we can accept "up", "UP", "DPAD_UP", "z", etc.
        # We normalize keys to: UPPERCASE with '_' separators.
        self._key_bits_norm: Dict[str, int] = {}
        for btn, bit in self._key_bits.items():
            try:
                b = int(bit)
            except Exception:
                continue
            self._key_bits_norm[self._norm_btn(btn)] = b

        # Event notification for streaming (Push vs Poll)
        self._port_conditions: Dict[int, threading.Condition] = {}

        # Rolling inference tracking
        self._rate_window_s = float(rate_window_s)
        self._max_events_per_port = int(max_events_per_port)

        # Per-port deque of inference timestamps
        self._infer_ts: Dict[int, Deque[float]] = {}
        self._infer_total: Dict[int, int] = {}

        # Per-port history
        self._max_history_per_port = int(max_history_per_port)
        self._hist: Dict[int, Deque[HistoryEntry]] = {}

    from collections import deque
    from typing import Sequence

    def _extract_future_buffer(self, decision: Dict[str, Any]) -> Optional[Any]:
        if not decision:
            return None

        def _as_steps(x: Any) -> Optional[List[Any]]:
            # Accept list/tuple/deque; reject str/bytes
            if isinstance(x, (list, tuple, deque)):
                return list(x)
            # Some code wraps it: {"actions":[...]} or {"steps":[...]}
            if isinstance(x, dict):
                for k in ("next_actions", "future_actions", "actions", "steps", "items"):
                    v = x.get(k)
                    if isinstance(v, (list, tuple, deque)):
                        return list(v)
            return None

        dbg = decision.get("debug") or {}
        buf = decision.get("buffer") or {}
        plan = decision.get("plan") or {}
        model = decision.get("model") or {}

        candidates = [
            decision.get("next_actions"),
            decision.get("future_actions"),
            decision.get("future_buffer"),
            decision.get("next_actions_buffer"),
            decision.get("action_buffer"),
            decision.get("future"),
            dbg.get("next_actions"),
            dbg.get("future_actions"),
            dbg.get("future_buffer"),
            dbg.get("next_actions_buffer"),
            buf.get("next_actions"),
            buf.get("future_actions"),
            buf.get("future_buffer"),
            plan.get("actions"),
            plan.get("next_actions"),
            plan.get("future_actions"),
            model.get("next_actions"),
            model.get("future_actions"),
            model.get("future_buffer"),
        ]

        # Prefer non-empty
        for c in candidates:
            steps = _as_steps(c)
            if steps is not None and len(steps) > 0:
                return steps

        # Allow empty-but-present
        for c in candidates:
            steps = _as_steps(c)
            if steps is not None:
                return steps

        return None


    def _future_step_to_buttons(self, step: Any) -> Dict[str, Any]:
        """
        Normalize one future step into:
          {"ng_pressed_buttons":[...], "mapped_pressed_buttons":[...]}
        """
        out: Dict[str, Any] = {"ng_pressed_buttons": [], "mapped_pressed_buttons": []}

        if step is None:
            return out

        # Case: already dict with pressed buttons lists
        if isinstance(step, dict):
            if isinstance(step.get("ng_pressed_buttons"), list):
                out["ng_pressed_buttons"] = [str(x) for x in step.get("ng_pressed_buttons") if x is not None]
            if isinstance(step.get("mapped_pressed_buttons"), list):
                out["mapped_pressed_buttons"] = [str(x) for x in step.get("mapped_pressed_buttons") if x is not None]

            # Case: keys provided
            if not out["ng_pressed_buttons"] and ("ng_key" in step or "ng_key_bin" in step):
                raw = step.get("ng_key", step.get("ng_key_bin"))
                _, _, pressed = self._coerce_key_to_mask_and_buttons(raw)
                out["ng_pressed_buttons"] = pressed

            if not out["mapped_pressed_buttons"] and ("mapped_key" in step or "key" in step or "mapped_key_bin" in step):
                raw = step.get("mapped_key", step.get("mapped_key_bin", step.get("key")))
                _, _, pressed = self._coerce_key_to_mask_and_buttons(raw)
                out["mapped_pressed_buttons"] = pressed

            return out

        # Case: scalar (int / bitstring / "up+z") -> treat as mapped by default
        _, _, pressed = self._coerce_key_to_mask_and_buttons(step)
        out["mapped_pressed_buttons"] = pressed
        return out


    # ----------------------------
    # Normalization / parsing
    # ----------------------------

    @staticmethod
    def _norm_btn(x: Any) -> str:
        return (
            str(x or "")
            .strip()
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )

    _SPLIT_RE = re.compile(r"[,\+\|/]+|\s+")

    def _button_list_from_any(self, raw: Any) -> List[str]:
        """
        Accept:
          - "up" / "up+z" / "up,z" / "up z"
          - ["up","z"]
          - ("up","z")
        Returns normalized tokens (original casing not preserved).
        """
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            out: List[str] = []
            for v in raw:
                s = self._norm_btn(v)
                if s:
                    out.append(s)
            return out

        s = self._norm_btn(raw)
        if not s:
            return []
        parts = [p for p in self._SPLIT_RE.split(s) if p]
        return parts

    def _mask_from_button_tokens(self, tokens_norm: List[str]) -> int:
        """
        Convert normalized button tokens to a bitmask using key_bit_positions.

        We support some common aliases so "UP" can match "DPAD_UP" keys, etc.
        """
        def expand_aliases(k: str) -> List[str]:
            # Canonical -> possible names present in key_bit_positions
            if k == "UP":
                return ["UP", "DPAD_UP"]
            if k == "DOWN":
                return ["DOWN", "DPAD_DOWN"]
            if k == "LEFT":
                return ["LEFT", "DPAD_LEFT"]
            if k == "RIGHT":
                return ["RIGHT", "DPAD_RIGHT"]

            # If your KEY_BIT_POSITIONS uses keyboard names z/x directly, keep them.
            # If it uses EAST/SOUTH, allow mapping:
            if k == "Z":
                return ["Z", "EAST", "A"]
            if k == "X":
                return ["X", "SOUTH", "B"]

            if k == "A":
                return ["A", "EAST", "Z"]
            if k == "B":
                return ["B", "SOUTH", "X"]

            if k == "SELECT":
                return ["SELECT", "BACK"]
            if k == "BACK":
                return ["BACK", "SELECT"]

            if k == "L":
                return ["L", "LEFT_SHOULDER", "LB", "L1"]
            if k == "R":
                return ["R", "RIGHT_SHOULDER", "RB", "R1"]

            return [k]

        mask = 0
        for tok in tokens_norm:
            for cand in expand_aliases(tok):
                bit = self._key_bits_norm.get(cand)
                if bit is None:
                    continue
                mask |= (1 << int(bit))
                break
        return int(mask)

    def _pressed_buttons_from_mask(self, mask_int: int) -> List[str]:
        pressed: List[str] = []
        for btn, bit in self._key_bits.items():
            try:
                b = int(bit)
            except Exception:
                continue
            if (int(mask_int) >> b) & 1:
                pressed.append(str(btn))
        pressed.sort()
        return pressed

    def _parse_key_bin(self, key_bin: str) -> Tuple[str, int]:
        """
        If key_bin is a valid binary string, return (s, int(s,2)).
        Otherwise return (original string, 0).
        """
        if not key_bin:
            return "", 0
        s = str(key_bin)
        if not all(c in "01" for c in s):
            return s, 0
        try:
            return s, int(s, 2)
        except Exception:
            return s, 0

    def _coerce_key_to_mask_and_buttons(self, key_raw: Any) -> Tuple[str, int, List[str]]:
        """
        Accepts key_raw in multiple formats:

          1) int mask:  123
          2) bitstring: "010101"
          3) button name(s): "up", "up+z", "up z", ["up","z"], etc.

        Returns:
          (key_bin_display, key_int_mask, pressed_buttons_list)
        """
        if key_raw is None:
            return "", 0, []

        # (1) int mask
        if isinstance(key_raw, int):
            key_int = int(key_raw)
            key_bin = format(key_int, "b") if key_int != 0 else "0"
            pressed = self._pressed_buttons_from_mask(key_int) if key_int != 0 else []
            return key_bin, key_int, pressed

        # (2) bitstring
        if isinstance(key_raw, str):
            key_bin, key_int = self._parse_key_bin(key_raw)
            if key_int != 0 or (key_bin and all(c in "01" for c in key_bin)):
                pressed = self._pressed_buttons_from_mask(key_int) if key_int != 0 else []
                return key_bin, key_int, pressed

        # (3) button names
        tokens = self._button_list_from_any(key_raw)
        if not tokens:
            # Keep whatever string form for display, but no mask/buttons.
            return str(key_raw), 0, []

        key_int = self._mask_from_button_tokens(tokens)
        if key_int != 0:
            pressed = self._pressed_buttons_from_mask(key_int)
            key_bin = format(key_int, "b")
            return key_bin, key_int, pressed

        # Fallback: can't map tokens to bits, but still surface tokens to UI
        # (This keeps debugging useful even if key_bit_positions doesn't include them.)
        return "+".join(tokens), 0, [t.lower() for t in tokens]

    # ----------------------------
    # Rate tracking
    # ----------------------------

    def get_render_condition(self, port: int) -> threading.Condition:
        with self._lock:
            if port not in self._port_conditions:
                self._port_conditions[port] = threading.Condition()
            return self._port_conditions[port]

    def _record_infer(self, port: int, ts: float) -> Tuple[int, float, float]:
        dq = self._infer_ts.get(port)
        if dq is None:
            dq = deque()
            self._infer_ts[port] = dq

        dq.append(ts)
        if len(dq) > self._max_events_per_port:
            while len(dq) > self._max_events_per_port:
                dq.popleft()

        cutoff = ts - self._rate_window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

        total = self._infer_total.get(port, 0) + 1
        self._infer_total[port] = total

        if len(dq) <= 1:
            hz = 0.0
        else:
            span = max(1e-6, dq[-1] - dq[0])
            hz = float((len(dq) - 1) / span)

        return total, hz, self._rate_window_s

    # ----------------------------
    # Decision extraction
    # ----------------------------

    def _extract_action_type_and_mapped_key(self, decision: Dict[str, Any]) -> Tuple[str, Any]:
        action_type = ""
        mapped_key = None
        try:
            cmd = (decision or {}).get("button_command") or {}
            action_type = str(cmd.get("type") or "")
            if action_type == "key_press":
                mapped_key = cmd.get("key")
            else:
                action_type = action_type or "unknown"
        except Exception:
            return "error_parsing_decision", None
        return action_type, mapped_key

    def _extract_ng_key(self, decision: Dict[str, Any]) -> Any:
        if not decision:
            return None
        v = decision.get("ng_key_bin")
        if v is not None and v != "":
            return v
        dbg = decision.get("debug") or {}
        v = dbg.get("ng_key_bin")
        if v is not None and v != "":
            return v
        mdl = decision.get("model") or {}
        v = mdl.get("ng_key_bin")
        if v is not None and v != "":
            return v
        return None

    # ----------------------------
    # Public update / export
    # ----------------------------

    def update(
        self,
        port: int,
        game_state: Dict[str, Any],
        decision: Dict[str, Any],
        pil_img: Optional[Image.Image],
    ) -> None:

        ts = time.time()
        p = int(port)
        inside_window = bool(game_state.get("inside_window", False))

        action_type, mapped_key_raw = self._extract_action_type_and_mapped_key(decision or {})
        ng_key_raw = self._extract_ng_key(decision or {})

        # Compute mapped key mask/buttons (supports int, bitstring, or "up+z" etc.)
        if action_type == "key_press":
            mapped_key_bin, mapped_key_int, mapped_pressed = self._coerce_key_to_mask_and_buttons(mapped_key_raw)
        else:
            mapped_key_bin, mapped_key_int, mapped_pressed = "", 0, []

        # Compute NG key mask/buttons (same flexible decoding)
        if ng_key_raw is not None and ng_key_raw != "":
            ng_key_bin, ng_key_int, ng_pressed = self._coerce_key_to_mask_and_buttons(ng_key_raw)
        else:
            ng_key_bin, ng_key_int, ng_pressed = "", 0, []

        # --- FUTURE BUFFER (predicted next actions) ---
        # Your strategy returns the horizon as:
        #   decision["action_plan_keys"]      -> mapped 16-bit bin strings (full horizon)
        #   decision["action_plan_ng_keys"]   -> ng 16-bit bin strings (full horizon)  (optional, add it)
        #
        # The UI wants NEXT actions, so we drop index 0 (current tick).
        next_actions: Optional[List[Dict[str, Any]]] = None

        plan_keys = (decision or {}).get("action_plan_keys")
        plan_ng_keys = (decision or {}).get("action_plan_ng_keys")

        if isinstance(plan_keys, list) and plan_keys:
            # bounded + skip current
            MAX_T = 64
            ks = plan_keys[1:1 + MAX_T]

            if isinstance(plan_ng_keys, list) and len(plan_ng_keys) == len(plan_keys):
                ngs = plan_ng_keys[1:1 + MAX_T]
                next_actions = []
                for mk, nk in zip(ks, ngs):
                    # normalize using your existing coercion
                    _, _, mapped_pressed = self._coerce_key_to_mask_and_buttons(mk)
                    _, _, ng_pressed = self._coerce_key_to_mask_and_buttons(nk)
                    next_actions.append({
                        "mapped_pressed_buttons": mapped_pressed,
                        "ng_pressed_buttons": ng_pressed,
                    })
            else:
                # mapped-only plan
                next_actions = []
                for mk in ks:
                    _, _, mapped_pressed = self._coerce_key_to_mask_and_buttons(mk)
                    next_actions.append({
                        "mapped_pressed_buttons": mapped_pressed,
                        "ng_pressed_buttons": [],
                    })
        else:
            # Fallback to your generic extractor if plan_keys isn't present
            future_raw = self._extract_future_buffer(decision or {})
            if isinstance(future_raw, list):
                MAX_T = 64
                next_actions = [self._future_step_to_buttons(s) for s in future_raw[:MAX_T]]



        jpg_bytes: Optional[bytes] = None
        img_w = 0
        img_h = 0
        if pil_img is not None:
            try:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img_w, img_h = pil_img.size
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=92, optimize=True)
                jpg_bytes = buf.getvalue()
            except Exception:
                jpg_bytes = None
                img_w = 0
                img_h = 0

        with self._lock:
            total, hz, win = self._record_infer(p, ts)

            snap = PortDebugSnapshot(
                ts=ts,
                inside_window=inside_window,
                action_type=action_type,
                mapped_key_bin=mapped_key_bin,
                mapped_key_int=mapped_key_int,
                mapped_pressed_buttons=mapped_pressed,
                ng_key_bin=ng_key_bin,
                ng_key_int=ng_key_int,
                ng_pressed_buttons=ng_pressed,
                next_actions=next_actions,  # <-- add this
                jpg_bytes=jpg_bytes,
                img_w=img_w,
                img_h=img_h,
                infer_count_total=total,
                infer_hz=hz,
                infer_window_s=win,
                last_update_age_s=0.0,
            )

            self._by_port[p] = snap

            dq = self._hist.get(p)
            if dq is None:
                dq = deque()
                self._hist[p] = dq

            dq.append(
                HistoryEntry(
                    ts=ts,
                    inside_window=inside_window,
                    action_type=action_type,
                    ng_key_bin=ng_key_bin,
                    ng_key_int=ng_key_int,
                    ng_pressed_buttons=ng_pressed,
                    mapped_key_bin=mapped_key_bin,
                    mapped_key_int=mapped_key_int,
                    mapped_pressed_buttons=mapped_pressed,
                )
            )
            while len(dq) > self._max_history_per_port:
                dq.popleft()

        cond = self.get_render_condition(p)
        with cond:
            cond.notify_all()

    def to_json(self) -> Dict[str, Any]:
        now = time.time()

        with self._lock:
            per_port: Dict[str, Any] = {}

            overall_events = 0
            overall_min_ts: Optional[float] = None
            overall_max_ts: Optional[float] = None

            for port, s in self._by_port.items():
                infer_dq = self._infer_ts.get(port) or deque()
                if infer_dq:
                    overall_events += len(infer_dq)
                    overall_min_ts = infer_dq[0] if overall_min_ts is None else min(overall_min_ts, infer_dq[0])
                    overall_max_ts = infer_dq[-1] if overall_max_ts is None else max(overall_max_ts, infer_dq[-1])

                last_age = float(now - s.ts) if s.ts else 0.0

                hist_dq = self._hist.get(port) or deque()
                hist = list(reversed(hist_dq))

                per_port[str(port)] = {
                    "ts": s.ts,
                    "inside_window": s.inside_window,
                    "action_type": s.action_type,
                    "ng_key_bin": s.ng_key_bin,
                    "ng_key_int": s.ng_key_int,
                    "ng_pressed_buttons": s.ng_pressed_buttons or [],
                    "mapped_key_bin": s.mapped_key_bin,
                    "mapped_key_int": s.mapped_key_int,
                    "mapped_pressed_buttons": s.mapped_pressed_buttons or [],
                    "img_w": s.img_w,
                    "img_h": s.img_h,
                    "next_actions": s.next_actions or [],
                    "has_image": bool(s.jpg_bytes),
                    "infer_count_total": s.infer_count_total,
                    "infer_hz": s.infer_hz,
                    "infer_window_s": self._rate_window_s,
                    "last_update_age_s": last_age,
                    "history": [
                        {
                            "ts": h.ts,
                            "inside_window": h.inside_window,
                            "action_type": h.action_type,
                            "ng_key_bin": h.ng_key_bin,
                            "ng_key_int": h.ng_key_int,
                            "ng_pressed_buttons": h.ng_pressed_buttons or [],
                            "mapped_key_bin": h.mapped_key_bin,
                            "mapped_key_int": h.mapped_key_int,
                            "mapped_pressed_buttons": h.mapped_pressed_buttons or [],
                        }
                        for h in hist
                    ],
                }

            if overall_min_ts is not None and overall_max_ts is not None and overall_events > 1:
                span = max(1e-6, overall_max_ts - overall_min_ts)
                overall_hz = float((overall_events - 1) / span)
            else:
                overall_hz = 0.0

            return {
                "meta": {
                    "rate_window_s": self._rate_window_s,
                    "overall_infer_hz": overall_hz,
                    "ports_reporting": len(per_port),
                    "ts": now,
                },
                "ports": per_port,
            }

    def get_image_jpg(self, port: int) -> Optional[bytes]:
        with self._lock:
            s = self._by_port.get(int(port))
            return None if s is None else s.jpg_bytes
