# ── Begin: selfplay_debug_ui/state.py ──
from __future__ import annotations

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

    # "mapped" = what we actually send to the game
    mapped_key_bin: str = ""
    mapped_key_int: int = 0
    mapped_pressed_buttons: List[str] = None

    # "ng" = raw model intent (if provided)
    ng_key_bin: str = ""
    ng_key_int: int = 0
    ng_pressed_buttons: List[str] = None

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
        self._key_bits = dict(key_bit_positions or {})

        # Event notification for streaming (Push vs Poll)
        # Each port gets a Condition variable that the web server waits on.
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

    def get_render_condition(self, port: int) -> threading.Condition:
        """
        Returns a threading.Condition specific to this port. 
        Waiters can wait() on this to be notified of new frames.
        """
        with self._lock:
            if port not in self._port_conditions:
                self._port_conditions[port] = threading.Condition()
            return self._port_conditions[port]

    def _pressed_buttons_from_mask(self, mask_int: int) -> List[str]:
        pressed: List[str] = []
        for btn, bit in self._key_bits.items():
            try:
                b = int(bit)
            except Exception:
                continue
            if (mask_int >> b) & 1:
                pressed.append(str(btn))
        pressed.sort()
        return pressed

    def _parse_key_bin(self, key_bin: str) -> Tuple[str, int]:
        if not key_bin:
            return "", 0
        s = str(key_bin)
        if not all(c in "01" for c in s):
            return s, 0
        try:
            return s, int(s, 2)
        except Exception:
            return s, 0

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

    def _extract_action_type_and_mapped_key(self, decision: Dict[str, Any]) -> Tuple[str, str]:
        action_type = ""
        mapped_key_bin = ""
        try:
            cmd = (decision or {}).get("button_command") or {}
            action_type = str(cmd.get("type") or "")
            if action_type == "key_press":
                mapped_key_bin = str(cmd.get("key") or "")
            else:
                action_type = action_type or "unknown"
        except Exception:
            return "error_parsing_decision", ""
        return action_type, mapped_key_bin

    def _extract_ng_key(self, decision: Dict[str, Any]) -> str:
        if not decision:
            return ""
        v = decision.get("ng_key_bin")
        if v:
            return str(v)
        dbg = decision.get("debug") or {}
        v = dbg.get("ng_key_bin")
        if v:
            return str(v)
        mdl = decision.get("model") or {}
        v = mdl.get("ng_key_bin")
        if v:
            return str(v)
        return ""

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

        action_type, mapped_key_bin_raw = self._extract_action_type_and_mapped_key(decision or {})
        ng_key_bin_raw = self._extract_ng_key(decision or {})

        mapped_key_bin, mapped_key_int = self._parse_key_bin(mapped_key_bin_raw)
        ng_key_bin, ng_key_int = self._parse_key_bin(ng_key_bin_raw)

        try:
            mapped_pressed = self._pressed_buttons_from_mask(mapped_key_int)
        except Exception:
            mapped_pressed = []

        if ng_key_bin:
            try:
                ng_pressed = self._pressed_buttons_from_mask(ng_key_int)
            except Exception:
                ng_pressed = []
        else:
            ng_pressed = []

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

        # Update data store
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

        # Notification phase (Outside of data lock, but inside Condition lock)
        # This wakes up the streaming server immediately
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
# ── End: selfplay_debug_ui/state.py ──