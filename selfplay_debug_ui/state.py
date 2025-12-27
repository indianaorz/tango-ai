# ── Begin: selfplay_debug_ui/state.py ──
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
    next_actions: Optional[List[Dict[str, Any]]] = None

    jpg_bytes: Optional[bytes] = None
    img_w: int = 0
    img_h: int = 0

    # Inference metrics
    infer_count_total: int = 0
    infer_hz: float = 0.0
    infer_window_s: float = 0.0
    last_update_age_s: float = 0.0

    # 🚀 NEW: Game State Features
    player_hp: int = 0
    enemy_hp: int = 0
    player_cross_id: int = 0
    player_used_crosses: List[int] = field(default_factory=list)
    enemy_used_crosses: List[int] = field(default_factory=list)
    beast_mode: bool = False
    full_synchro: bool = False
    chip_window: List[Dict[str, Any]] = field(default_factory=list) # List of {id, code}


class DebugState:
    """
    Thread-safe store for per-port debug data and history.
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

        # Original mapping
        self._key_bits = dict(key_bit_positions or {})

        # Normalized lookup
        self._key_bits_norm: Dict[str, int] = {}
        for btn, bit in self._key_bits.items():
            try:
                b = int(bit)
            except Exception:
                continue
            self._key_bits_norm[self._norm_btn(btn)] = b

        # Event notification
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

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _norm_btn(x: Any) -> str:
        return str(x or "").strip().upper().replace("-", "_").replace(" ", "_")

    _SPLIT_RE = re.compile(r"[,\+\|/]+|\s+")

    def _button_list_from_any(self, raw: Any) -> List[str]:
        if raw is None: return []
        if isinstance(raw, (list, tuple)):
            out = []
            for v in raw:
                s = self._norm_btn(v)
                if s: out.append(s)
            return out
        s = self._norm_btn(raw)
        if not s: return []
        return [p for p in self._SPLIT_RE.split(s) if p]

    def _mask_from_button_tokens(self, tokens_norm: List[str]) -> int:
        def expand_aliases(k: str) -> List[str]:
            if k == "UP": return ["UP", "DPAD_UP"]
            if k == "DOWN": return ["DOWN", "DPAD_DOWN"]
            if k == "LEFT": return ["LEFT", "DPAD_LEFT"]
            if k == "RIGHT": return ["RIGHT", "DPAD_RIGHT"]
            if k in ("Z", "A", "EAST"): return ["A", "EAST", "Z"]
            if k in ("X", "B", "SOUTH"): return ["B", "SOUTH", "X"]
            if k in ("SELECT", "BACK"): return ["SELECT", "BACK"]
            if k in ("L", "LEFT_SHOULDER", "LB"): return ["L", "LEFT_SHOULDER"]
            if k in ("R", "RIGHT_SHOULDER", "RB"): return ["R", "RIGHT_SHOULDER"]
            return [k]

        mask = 0
        for tok in tokens_norm:
            for cand in expand_aliases(tok):
                bit = self._key_bits_norm.get(cand)
                if bit is not None:
                    mask |= (1 << int(bit))
                    break
        return int(mask)

    def _pressed_buttons_from_mask(self, mask_int: int) -> List[str]:
        pressed = []
        for btn, bit in self._key_bits.items():
            if (int(mask_int) >> int(bit)) & 1:
                pressed.append(str(btn))
        pressed.sort()
        return pressed

    def _parse_key_bin(self, key_bin: str) -> Tuple[str, int]:
        if not key_bin: return "", 0
        s = str(key_bin)
        if not all(c in "01" for c in s): return s, 0
        try: return s, int(s, 2)
        except: return s, 0

    def _coerce_key_to_mask_and_buttons(self, key_raw: Any) -> Tuple[str, int, List[str]]:
        if key_raw is None: return "", 0, []
        if isinstance(key_raw, int):
            ki = int(key_raw)
            return format(ki, "b"), ki, self._pressed_buttons_from_mask(ki)
        if isinstance(key_raw, str):
            kb, ki = self._parse_key_bin(key_raw)
            if ki != 0 or (kb and all(c in "01" for c in kb)):
                return kb, ki, self._pressed_buttons_from_mask(ki)
        tokens = self._button_list_from_any(key_raw)
        if not tokens: return str(key_raw), 0, []
        ki = self._mask_from_button_tokens(tokens)
        return format(ki, "b"), ki, self._pressed_buttons_from_mask(ki)

    def _extract_future_buffer(self, decision: Dict[str, Any]) -> Optional[Any]:
        return decision.get("next_actions")

    def _future_step_to_buttons(self, step: Any) -> Dict[str, Any]:
        out = {"ng_pressed_buttons": [], "mapped_pressed_buttons": []}
        if not step: return out
        if isinstance(step, dict):
            if "mapped_pressed_buttons" in step: out["mapped_pressed_buttons"] = step["mapped_pressed_buttons"]
            if "ng_pressed_buttons" in step: out["ng_pressed_buttons"] = step["ng_pressed_buttons"]
            if not out["mapped_pressed_buttons"] and "key" in step:
                _, _, out["mapped_pressed_buttons"] = self._coerce_key_to_mask_and_buttons(step["key"])
            return out
        _, _, out["mapped_pressed_buttons"] = self._coerce_key_to_mask_and_buttons(step)
        return out

    def get_render_condition(self, port: int) -> threading.Condition:
        with self._lock:
            if port not in self._port_conditions:
                self._port_conditions[port] = threading.Condition()
            return self._port_conditions[port]

    def _record_infer(self, port: int, ts: float) -> Tuple[int, float, float]:
        dq = self._infer_ts.setdefault(port, deque())
        dq.append(ts)
        while dq and dq[0] < ts - self._rate_window_s: dq.popleft()
        total = self._infer_total.get(port, 0) + 1
        self._infer_total[port] = total
        hz = (len(dq) - 1) / max(1e-6, dq[-1] - dq[0]) if len(dq) > 1 else 0.0
        return total, hz, self._rate_window_s

    # ----------------------------
    # Decision extraction
    # ----------------------------
    def _extract_action_type_and_mapped_key(self, decision: Dict[str, Any]) -> Tuple[str, Any]:
        try:
            cmd = (decision or {}).get("button_command") or {}
            return str(cmd.get("type") or ""), cmd.get("key")
        except: return "error", None

    def _extract_ng_key(self, decision: Dict[str, Any]) -> Any:
        return (decision or {}).get("ng_key_bin") or ((decision or {}).get("debug") or {}).get("ng_key_bin")

    # ----------------------------
    # Public Update
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
        inside_window = bool(float(game_state.get("inside_window", 0)))

        at, mk_raw = self._extract_action_type_and_mapped_key(decision)
        nk_raw = self._extract_ng_key(decision)
        mk_bin, mk_int, mk_btns = self._coerce_key_to_mask_and_buttons(mk_raw)
        nk_bin, nk_int, nk_btns = self._coerce_key_to_mask_and_buttons(nk_raw)

        # Future Buffer Logic
        next_actions = []
        plan_keys = (decision or {}).get("action_plan_keys")
        if isinstance(plan_keys, list) and plan_keys:
            # Skip T=0
            for k in plan_keys[1:65]:
                _, _, btns = self._coerce_key_to_mask_and_buttons(k)
                next_actions.append({"mapped_pressed_buttons": btns, "ng_pressed_buttons": []})
        
        # Image
        jpg_bytes, w, h = None, 0, 0
        if pil_img:
            try:
                if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
                w, h = pil_img.size
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                jpg_bytes = buf.getvalue()
            except: pass

        # 🚀 EXTRACT GAME STATE
        p_hp = int(float(game_state.get("player_health", 0)))
        e_hp = int(float(game_state.get("enemy_health", 0)))
        p_cross = int(float(game_state.get("player_cross_id", 0)))
        p_used = game_state.get("player_used_crosses_list", [])
        e_used = game_state.get("enemy_used_crosses_list", [])
        beast = int(float(game_state.get("beast_mode", 0))) > 0
        emo = int(float(game_state.get("player_emotion", 0)))
        sync = (emo == 1)

        chip_window = []
        if inside_window:
            raw_slots = game_state.get("chip_slots", [])
            raw_codes = game_state.get("chip_codes", [])
            count = int(float(game_state.get("chip_visible_count", 5)))
            for i in range(min(len(raw_slots), count)):
                chip_window.append({"id": int(raw_slots[i]), "code": int(raw_codes[i])})

        with self._lock:
            total, hz, win = self._record_infer(p, ts)

            snap = PortDebugSnapshot(
                ts=ts, inside_window=inside_window, action_type=at,
                mapped_key_bin=mk_bin, mapped_key_int=mk_int, mapped_pressed_buttons=mk_btns,
                ng_key_bin=nk_bin, ng_key_int=nk_int, ng_pressed_buttons=nk_btns,
                next_actions=next_actions,
                jpg_bytes=jpg_bytes, img_w=w, img_h=h,
                infer_count_total=total, infer_hz=hz, infer_window_s=win,
                
                # Game State
                player_hp=p_hp, enemy_hp=e_hp,
                player_cross_id=p_cross,
                player_used_crosses=p_used, enemy_used_crosses=e_used,
                beast_mode=beast, full_synchro=sync,
                chip_window=chip_window
            )
            self._by_port[p] = snap

            dq = self._hist.setdefault(p, deque())
            dq.append(HistoryEntry(ts, inside_window, at, nk_bin, nk_int, nk_btns, mk_bin, mk_int, mk_btns))
            while len(dq) > self._max_history_per_port: dq.popleft()

        cond = self.get_render_condition(p)
        with cond: cond.notify_all()

    def to_json(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            ports = {}
            for p, s in self._by_port.items():
                ports[str(p)] = {
                    "ts": s.ts, "inside_window": s.inside_window, "infer_hz": s.infer_hz,
                    "mapped_pressed_buttons": s.mapped_pressed_buttons,
                    "ng_pressed_buttons": s.ng_pressed_buttons,
                    "next_actions": s.next_actions,
                    "has_image": bool(s.jpg_bytes), "img_w": s.img_w, "img_h": s.img_h,
                    
                    # 🚀 SERIALIZE GAME STATE
                    "player_hp": s.player_hp, "enemy_hp": s.enemy_hp,
                    "player_cross_id": s.player_cross_id,
                    "player_used_crosses": s.player_used_crosses,
                    "enemy_used_crosses": s.enemy_used_crosses,
                    "beast_mode": s.beast_mode, "full_synchro": s.full_synchro,
                    "chip_window": s.chip_window
                }
            return {"meta": {"ts": now}, "ports": ports}

    def get_image_jpg(self, port: int) -> Optional[bytes]:
        with self._lock:
            s = self._by_port.get(int(port))
            return s.jpg_bytes if s else None
# ── End: selfplay_debug_ui/state.py ──