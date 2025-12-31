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

# Optional import to compute the same grid indices your critic_minimal uses.
# If this import fails (e.g. module layout changes), the UI still works; it just
# won’t show p/e grid idx.
try:
    from critic_minimal.features import pos_to_grid_idx  # type: ignore
except Exception:  # pragma: no cover
    pos_to_grid_idx = None  # type: ignore[assignment]


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

    # Strategy / model mode (e.g. "battle"/"plan"/"bootstrap")
    active_model: str = ""
    note: str = ""
    infer_ms: float = 0.0

    # "mapped" = what we actually send to the game (current tick)
    mapped_key_bin: str = ""
    mapped_key_int: int = 0
    mapped_pressed_buttons: List[str] = field(default_factory=list)

    # "ng" = raw model intent (if provided) (current tick)
    ng_key_bin: str = ""
    ng_key_int: int = 0
    ng_pressed_buttons: List[str] = field(default_factory=list)

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

    # ── Game State Features (existing) ────────────────────────────────────────
    player_hp: int = 0
    enemy_hp: int = 0
    player_cross_id: int = 0
    player_used_crosses: List[int] = field(default_factory=list)
    enemy_used_crosses: List[int] = field(default_factory=list)
    beast_mode: bool = False
    full_synchro: bool = False
    chip_window: List[Dict[str, Any]] = field(default_factory=list)  # {id, code}

    # ── NEW: Raw inputs that the battle policy actually consumes ─────────────
    player_charge: int = 0
    enemy_charge: int = 0
    cust_gauge: int = 0
    player_emotion_id: int = 0
    enemy_emotion_id: int = 0
    player_chip_id: int = 0

    # Grid (6x3 flattened => 18)
    grid_tile: List[int] = field(default_factory=list)
    grid_owner: List[int] = field(default_factory=list)

    # Positions
    player_pos_raw: Any = None
    enemy_pos_raw: Any = None
    player_grid_idx: int = -1
    enemy_grid_idx: int = -1


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

    @staticmethod
    def _safe_int(v: Any, default: int = 0) -> int:
        try:
            if v is None:
                return default
            if isinstance(v, bool):
                return int(v)
            return int(float(v))
        except Exception:
            return default

    @staticmethod
    def _as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        return []

    def _button_list_from_any(self, raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            out = []
            for v in raw:
                s = self._norm_btn(v)
                if s:
                    out.append(s)
            return out
        s = self._norm_btn(raw)
        if not s:
            return []
        return [p for p in self._SPLIT_RE.split(s) if p]

    def _mask_from_button_tokens(self, tokens_norm: List[str]) -> int:
        def expand_aliases(k: str) -> List[str]:
            if k == "UP":
                return ["UP", "DPAD_UP"]
            if k == "DOWN":
                return ["DOWN", "DPAD_DOWN"]
            if k == "LEFT":
                return ["LEFT", "DPAD_LEFT"]
            if k == "RIGHT":
                return ["RIGHT", "DPAD_RIGHT"]
            if k in ("Z", "A", "EAST"):
                return ["A", "EAST", "Z"]
            if k in ("X", "B", "SOUTH"):
                return ["B", "SOUTH", "X"]
            if k in ("SELECT", "BACK"):
                return ["SELECT", "BACK"]
            if k in ("L", "LEFT_SHOULDER", "LB"):
                return ["L", "LEFT_SHOULDER"]
            if k in ("R", "RIGHT_SHOULDER", "RB"):
                return ["R", "RIGHT_SHOULDER"]
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
        if key_raw is None:
            return "", 0, []
        if isinstance(key_raw, int):
            ki = int(key_raw)
            return format(ki, "b"), ki, self._pressed_buttons_from_mask(ki)
        if isinstance(key_raw, str):
            kb, ki = self._parse_key_bin(key_raw)
            if ki != 0 or (kb and all(c in "01" for c in kb)):
                return kb, ki, self._pressed_buttons_from_mask(ki)
        tokens = self._button_list_from_any(key_raw)
        if not tokens:
            return str(key_raw), 0, []
        ki = self._mask_from_button_tokens(tokens)
        return format(ki, "b"), ki, self._pressed_buttons_from_mask(ki)

    # ----------------------------
    # Grid/pos normalization (match battle policy expectations)
    # ----------------------------
    _X_CENTERS = [20.0, 60.0, 100.0, 140.0, 180.0, 220.0]
    _Y_CENTERS = [260.0, 515.0, 770.0]

    @classmethod
    def _looks_like_panel_coords(cls, x: int, y: int) -> bool:
        if x < 0 or y < 0:
            return False
        return (x <= 6 and y <= 3)

    @classmethod
    def _panel_to_pixel_cluster(cls, x: int, y: int) -> Tuple[float, float]:
        col = (x - 1) if (1 <= x <= 6) else x
        row = (y - 1) if (1 <= y <= 3) else y
        col = max(0, min(5, int(col)))
        row = max(0, min(2, int(row)))
        return float(cls._X_CENTERS[col]), float(cls._Y_CENTERS[row])

    @classmethod
    def _pos_xy(cls, pos_any: Any) -> Tuple[float, float]:
        """
        Robust parse:
          - [x,y], (x,y)
          - {"x":..., "y":...}
          - {"X":..., "Y":...}
        """
        if pos_any is None:
            return 0.0, 0.0
        if isinstance(pos_any, (list, tuple)) and len(pos_any) >= 2:
            return float(cls._safe_int(pos_any[0], 0)), float(cls._safe_int(pos_any[1], 0))
        if isinstance(pos_any, dict):
            x = pos_any.get("x", pos_any.get("X", 0))
            y = pos_any.get("y", pos_any.get("Y", 0))
            return float(cls._safe_int(x, 0)), float(cls._safe_int(y, 0))
        # last resort: try to parse "x,y"
        try:
            s = str(pos_any)
            parts = re.split(r"[,\s]+", s.strip())
            if len(parts) >= 2:
                return float(cls._safe_int(parts[0], 0)), float(cls._safe_int(parts[1], 0))
        except Exception:
            pass
        return 0.0, 0.0

    @classmethod
    def _normalize_pos_for_model(cls, pos_any: Any) -> Tuple[float, float]:
        """
        If it looks like panel coords, map to the same pixel cluster centers the
        critic_minimal cache expected.
        """
        xi, yi = cls._pos_xy(pos_any)
        x = int(xi)
        y = int(yi)
        if cls._looks_like_panel_coords(x, y):
            return cls._panel_to_pixel_cluster(x, y)
        return float(xi), float(yi)

    @staticmethod
    def _tile_class_norm(v: Any) -> int:
        # UI-side: keep ints, default 0 (unknown/none)
        try:
            if v is None:
                return 0
            return int(float(v))
        except Exception:
            return 0

    @staticmethod
    def _owner_class_norm(v: Any) -> int:
        # Expect 0=P side, 1=E side, else neutral
        try:
            if v is None:
                return 2
            return int(float(v))
        except Exception:
            return 2

    # ----------------------------
    # Public
    # ----------------------------
    def get_render_condition(self, port: int) -> threading.Condition:
        with self._lock:
            if port not in self._port_conditions:
                self._port_conditions[port] = threading.Condition()
            return self._port_conditions[port]

    def _record_infer(self, port: int, ts: float) -> Tuple[int, float, float]:
        dq = self._infer_ts.setdefault(port, deque())
        dq.append(ts)
        while dq and dq[0] < ts - self._rate_window_s:
            dq.popleft()
        total = self._infer_total.get(port, 0) + 1
        self._infer_total[port] = total
        hz = (len(dq) - 1) / max(1e-6, dq[-1] - dq[0]) if len(dq) > 1 else 0.0
        return total, hz, self._rate_window_s

    def _extract_action_type_and_mapped_key(self, decision: Dict[str, Any]) -> Tuple[str, Any]:
        try:
            cmd = (decision or {}).get("button_command") or {}
            return str(cmd.get("type") or ""), cmd.get("key")
        except Exception:
            return "error", None

    def _extract_ng_key(self, decision: Dict[str, Any]) -> Any:
        return (decision or {}).get("ng_key_bin") or ((decision or {}).get("debug") or {}).get("ng_key_bin")

    @staticmethod
    def _dbg_str(decision: Dict[str, Any], key: str, default: str = "") -> str:
        try:
            d = (decision or {}).get("debug") or {}
            v = d.get(key, default)
            return str(v) if v is not None else default
        except Exception:
            return default

    @staticmethod
    def _dbg_float(decision: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            d = (decision or {}).get("debug") or {}
            v = d.get(key, default)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

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

        # Future Buffer Logic (kept)
        next_actions: List[Dict[str, Any]] = []
        plan_keys = (decision or {}).get("action_plan_keys")
        if isinstance(plan_keys, list) and plan_keys:
            for k in plan_keys[1:65]:
                _, _, btns = self._coerce_key_to_mask_and_buttons(k)
                next_actions.append({"mapped_pressed_buttons": btns, "ng_pressed_buttons": []})

        # Image
        jpg_bytes, w, h = None, 0, 0
        if pil_img:
            try:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                w, h = pil_img.size
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                jpg_bytes = buf.getvalue()
            except Exception:
                pass

        # ── Existing extracted state ───────────────────────────────────────────
        p_hp = self._safe_int(game_state.get("player_health", 0))
        e_hp = self._safe_int(game_state.get("enemy_health", 0))
        p_cross = self._safe_int(game_state.get("player_cross_id", 0))

        p_used = game_state.get("player_used_crosses_list", [])
        e_used = game_state.get("enemy_used_crosses_list", [])
        if not isinstance(p_used, list):
            p_used = []
        if not isinstance(e_used, list):
            e_used = []

        beast = self._safe_int(game_state.get("beast_mode", 0)) > 0
        emo_for_sync = self._safe_int(game_state.get("player_emotion", 0))
        sync = (emo_for_sync == 1)

        # Chip Window (kept)
        chip_window: List[Dict[str, Any]] = []
        if inside_window:
            raw_slots = self._as_list(game_state.get("chip_slots", []))
            raw_codes = self._as_list(game_state.get("chip_codes", []))
            count = self._safe_int(game_state.get("chip_visible_count", 5), 5)
            for i in range(min(len(raw_slots), len(raw_codes), count)):
                chip_window.append({"id": self._safe_int(raw_slots[i], 0), "code": self._safe_int(raw_codes[i], 0)})

        # ── NEW: raw features used by battle policy ────────────────────────────
        p_chg = self._safe_int(game_state.get("player_charge", 0))
        e_chg = self._safe_int(game_state.get("enemy_charge", 0))
        cust = self._safe_int(game_state.get("cust_gauge", 0))

        p_emo_id = self._safe_int(game_state.get("player_game_emotion", game_state.get("player_emotion", 0)))
        e_emo_id = self._safe_int(game_state.get("enemy_game_emotion", 0))

        player_chip_id = self._safe_int(game_state.get("player_chip", 0))

        # Grid arrays
        gs_raw = self._as_list(game_state.get("grid_state", game_state.get("grid_tile", [])))
        go_raw = self._as_list(game_state.get("grid_owner_state", game_state.get("grid_owner", [])))
        grid_tile = [self._tile_class_norm(gs_raw[i]) for i in range(min(18, len(gs_raw)))]
        grid_owner = [self._owner_class_norm(go_raw[i]) for i in range(min(18, len(go_raw)))]

        # Right-pad to 18 for stable UI rendering
        if len(grid_tile) < 18:
            grid_tile.extend([0] * (18 - len(grid_tile)))
        if len(grid_owner) < 18:
            grid_owner.extend([2] * (18 - len(grid_owner)))

        # Positions → grid idx (same intent as battle policy)
        p_pos_raw = game_state.get("player_pos")
        e_pos_raw = game_state.get("enemy_pos")
        p_grid_idx = -1
        e_grid_idx = -1
        if pos_to_grid_idx is not None:
            try:
                px, py = self._normalize_pos_for_model(p_pos_raw)
                ex, ey = self._normalize_pos_for_model(e_pos_raw)
                p_grid_idx = int(pos_to_grid_idx(px, py))
                e_grid_idx = int(pos_to_grid_idx(ex, ey))
            except Exception:
                p_grid_idx = -1
                e_grid_idx = -1

        # Decision debug tags
        active_model = self._dbg_str(decision, "active_model", "")
        note = self._dbg_str(decision, "note", "")
        infer_ms = self._dbg_float(decision, "infer_ms", 0.0)

        with self._lock:
            total, hz, win = self._record_infer(p, ts)

            snap = PortDebugSnapshot(
                ts=ts,
                inside_window=inside_window,
                action_type=at,
                active_model=active_model,
                note=note,
                infer_ms=infer_ms,
                mapped_key_bin=mk_bin,
                mapped_key_int=mk_int,
                mapped_pressed_buttons=mk_btns,
                ng_key_bin=nk_bin,
                ng_key_int=nk_int,
                ng_pressed_buttons=nk_btns,
                next_actions=next_actions,
                jpg_bytes=jpg_bytes,
                img_w=w,
                img_h=h,
                infer_count_total=total,
                infer_hz=hz,
                infer_window_s=win,
                player_hp=p_hp,
                enemy_hp=e_hp,
                player_cross_id=p_cross,
                player_used_crosses=[self._safe_int(x, 0) for x in p_used],
                enemy_used_crosses=[self._safe_int(x, 0) for x in e_used],
                beast_mode=bool(beast),
                full_synchro=bool(sync),
                chip_window=chip_window,
                player_charge=p_chg,
                enemy_charge=e_chg,
                cust_gauge=cust,
                player_emotion_id=p_emo_id,
                enemy_emotion_id=e_emo_id,
                player_chip_id=player_chip_id,
                grid_tile=grid_tile,
                grid_owner=grid_owner,
                player_pos_raw=p_pos_raw,
                enemy_pos_raw=e_pos_raw,
                player_grid_idx=int(p_grid_idx),
                enemy_grid_idx=int(e_grid_idx),
            )
            self._by_port[p] = snap

            dq = self._hist.setdefault(p, deque())
            dq.append(
                HistoryEntry(
                    ts,
                    inside_window,
                    at,
                    nk_bin,
                    nk_int,
                    nk_btns,
                    mk_bin,
                    mk_int,
                    mk_btns,
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
            ports: Dict[str, Any] = {}
            for p, s in self._by_port.items():
                ports[str(p)] = {
                    "ts": s.ts,
                    "inside_window": s.inside_window,
                    "infer_hz": s.infer_hz,
                    "active_model": s.active_model,
                    "note": s.note,
                    "infer_ms": s.infer_ms,
                    "mapped_pressed_buttons": s.mapped_pressed_buttons,
                    "ng_pressed_buttons": s.ng_pressed_buttons,
                    "next_actions": s.next_actions,
                    "has_image": bool(s.jpg_bytes),
                    "img_w": s.img_w,
                    "img_h": s.img_h,
                    "player_hp": s.player_hp,
                    "enemy_hp": s.enemy_hp,
                    "player_cross_id": s.player_cross_id,
                    "player_used_crosses": s.player_used_crosses,
                    "enemy_used_crosses": s.enemy_used_crosses,
                    "beast_mode": s.beast_mode,
                    "full_synchro": s.full_synchro,
                    "chip_window": s.chip_window,
                    # NEW
                    "player_charge": s.player_charge,
                    "enemy_charge": s.enemy_charge,
                    "cust_gauge": s.cust_gauge,
                    "player_emotion_id": s.player_emotion_id,
                    "enemy_emotion_id": s.enemy_emotion_id,
                    "player_chip_id": s.player_chip_id,
                    "grid_tile": s.grid_tile,
                    "grid_owner": s.grid_owner,
                    "player_pos_raw": s.player_pos_raw,
                    "enemy_pos_raw": s.enemy_pos_raw,
                    "player_grid_idx": s.player_grid_idx,
                    "enemy_grid_idx": s.enemy_grid_idx,
                }
            return {"meta": {"ts": now}, "ports": ports}

    def get_image_jpg(self, port: int) -> Optional[bytes]:
        with self._lock:
            s = self._by_port.get(int(port))
            return s.jpg_bytes if s else None


__all__ = ["DebugState", "PortDebugSnapshot", "HistoryEntry"]
# ── End: selfplay_debug_ui/state.py ──
