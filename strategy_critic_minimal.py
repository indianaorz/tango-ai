# strategy_critic_minimal.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from critic_minimal.features import (
    ACTION_DIM,
    BUTTON_KEYS,
    _as_int,
    _as_list,
    _chip_id_norm,
    _owner_norm,
    _pos_norm,
    _tile_norm,
    pos_to_grid_idx,
    rel_pe_index,
)
from critic_minimal.model import MinimalQCritic

# =============================================================================
# Small helpers (your runner expects 16-bit key strings)
# =============================================================================


def _int_to_bin16(mask: int) -> str:
    return format(int(mask) & 0xFFFF, "016b")


def _now_ms() -> float:
    return time.time() * 1000.0


# =============================================================================
# Position normalization (match your training/caching mapping expectations)
# =============================================================================

_X_CENTERS = [20.0, 60.0, 100.0, 140.0, 180.0, 220.0]
_Y_CENTERS = [260.0, 515.0, 770.0]


def _looks_like_panel_coords(x: int, y: int) -> bool:
    # Typical panel coords are small (0..5,0..2) or (1..6,1..3).
    if x < 0 or y < 0:
        return False
    return (x <= 6 and y <= 3)


def _panel_to_pixel_cluster(x: int, y: int) -> Tuple[float, float]:
    # Accept 0-indexed or 1-indexed (common in some hooks)
    col = (x - 1) if (1 <= x <= 6) else x
    row = (y - 1) if (1 <= y <= 3) else y
    col = max(0, min(5, int(col)))
    row = max(0, min(2, int(row)))
    return float(_X_CENTERS[col]), float(_Y_CENTERS[row])


def _normalize_pos_for_model(pos_any: Any) -> Tuple[float, float]:
    """
    Your cache/training used pos_to_grid_idx() on pixel-like coords.
    Live state sometimes may be (col,row) panels; convert those to the
    same pixel cluster centers the cache mapping expects.
    """
    xi, yi = _pos_norm(pos_any)
    x = int(xi)
    y = int(yi)
    if _looks_like_panel_coords(x, y):
        return _panel_to_pixel_cluster(x, y)
    return float(x), float(y)


# =============================================================================
# Feature extraction (mirror critic_minimal/precache.py semantics)
# =============================================================================


def _extract_step_features_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    # Scalars (raw)
    p_hp = float(_as_int(state.get("player_health"), 0))
    e_hp = float(_as_int(state.get("enemy_health"), 0))
    p_chg = float(_as_int(state.get("player_charge"), 0))
    e_chg = float(_as_int(state.get("enemy_charge"), 0))
    cust = float(_as_int(state.get("cust_gauge"), 0))

    # Emotion ids (raw ints)
    p_emo = _as_int(state.get("player_game_emotion"), 0)
    e_emo = _as_int(state.get("enemy_game_emotion"), 0)

    # Grid state (support alt keys, but normalize exactly like precache)
    grid_state = _as_list(state.get("grid_state", state.get("grid_tile", [])))
    grid_owner = _as_list(state.get("grid_owner_state", state.get("grid_owner", [])))

    gs = [_tile_norm(grid_state[i]) for i in range(min(18, len(grid_state)))] + [0] * (18 - min(18, len(grid_state)))
    go = [_owner_norm(grid_owner[i]) for i in range(min(18, len(grid_owner)))] + [2] * (18 - min(18, len(grid_owner)))

    # Positions -> indices (ensure same coord space as training)
    px, py = _normalize_pos_for_model(state.get("player_pos"))
    ex, ey = _normalize_pos_for_model(state.get("enemy_pos"))
    p_grid_idx = int(pos_to_grid_idx(px, py))
    e_grid_idx = int(pos_to_grid_idx(ex, ey))
    rel_idx = int(rel_pe_index(p_grid_idx, e_grid_idx))

    # Chip id
    player_chip = int(_chip_id_norm(state.get("player_chip")))

    return {
        "p_hp": p_hp,
        "e_hp": e_hp,
        "p_chg": p_chg,
        "e_chg": e_chg,
        "cust": cust,
        "p_emo": int(p_emo),
        "e_emo": int(e_emo),
        "grid_tile": gs,
        "grid_owner": go,
        "p_grid_idx": p_grid_idx,
        "e_grid_idx": e_grid_idx,
        "rel_pe_idx": rel_idx,
        "player_chip": player_chip,
    }


# =============================================================================
# Action bitmask <-> ACTION_DIM vector (CRITICAL: use BUTTON_KEYS, not ACTION_DIM index)
# =============================================================================


def _action_mask_to_action_vec(action_mask: int) -> torch.Tensor:
    """
    action_mask is in critic_minimal BUTTON_KEYS bitspace (len(BUTTON_KEYS) == ACTION_DIM).
    Vector output matches training: [ACTION_DIM] float32 aligned to BUTTON_KEYS order.
    """
    m = int(action_mask)
    a = torch.zeros((ACTION_DIM,), dtype=torch.float32)
    for i in range(ACTION_DIM):
        if (m >> i) & 1:
            a[i] = 1.0
    return a


# =============================================================================
# Candidate generation (safe default: no START/BACK)
# =============================================================================


def _default_candidate_masks_no_menu() -> List[int]:
    """
    Generates a compact but useful set in critic_minimal BUTTON_KEYS bitspace:
      - D-pad: none or one direction
      - Face: none/A/B/A+B
      - Shoulders: none/L/R/L+R
    Excludes START/BACK entirely (we never set those bits).
    """
    idx = {k: i for i, k in enumerate(BUTTON_KEYS)}

    dpad = [
        0,
        1 << idx["DPAD_UP"],
        1 << idx["DPAD_DOWN"],
        1 << idx["DPAD_LEFT"],
        1 << idx["DPAD_RIGHT"],
    ]
    face = [
        0,
        1 << idx["EAST"],
        1 << idx["SOUTH"],
        (1 << idx["EAST"]) | (1 << idx["SOUTH"]),
    ]
    shoulder = [
        0,
        1 << idx["LEFT_SHOULDER"],
        1 << idx["RIGHT_SHOULDER"],
        (1 << idx["LEFT_SHOULDER"]) | (1 << idx["RIGHT_SHOULDER"]),
    ]

    out: List[int] = []
    for d in dpad:
        for f in face:
            for s in shoulder:
                out.append(int(d | f | s))

    # stable uniq
    seen = set()
    uniq: List[int] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


# =============================================================================
# History buffer (we write our own actions; game won’t)
# =============================================================================


@dataclass(frozen=True)
class _HistCfg:
    hold: int
    seq_len: int
    require_cust_gt0: bool


class _History:
    """
    Live analogue of critic_minimal/precache.py:
      - Raw ticks arrive every inference step.
      - We "sample" every hold ticks (t % hold == 0) and append one token.
      - The token action is OR-aggregate over last `hold` raw actions, matching aggregate_action(... hold).
    IMPORTANT: Since Tango does not echo controller state, we must record what we sent.
    This module expects record_raw_action() to be called exactly once per tick, before update_from_state().
    """

    def __init__(self, cfg: _HistCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._raw_t = 0
        self._in_battle = False

        # raw actions we "sent" each tick
        self._raw_actions: List[torch.Tensor] = []

        # sampled streams (each entry corresponds to one sampled token)
        self.scalars: List[torch.Tensor] = []     # [5] float32 normalized
        self.p_emo: List[int] = []
        self.e_emo: List[int] = []
        self.grid_tile: List[torch.Tensor] = []   # [18] int64
        self.grid_owner: List[torch.Tensor] = []  # [18] int64
        self.p_grid_idx: List[int] = []
        self.e_grid_idx: List[int] = []
        self.rel_pe_idx: List[int] = []
        self.player_chip: List[int] = []
        self.action: List[torch.Tensor] = []      # [ACTION_DIM] float32 OR across hold raw ticks

    def in_battle(self) -> bool:
        return bool(self._in_battle)

    def record_raw_action(self, a_raw: torch.Tensor) -> None:
        if a_raw.ndim != 1 or int(a_raw.shape[0]) != ACTION_DIM:
            raise ValueError(f"record_raw_action expects [ACTION_DIM], got {tuple(a_raw.shape)}")
        self._raw_actions.append(a_raw.detach().to(torch.float32).cpu())

    def update_from_state(self, state: Dict[str, Any]) -> None:
        feats = _extract_step_features_from_state(state)

        in_battle_now = True
        if self.cfg.require_cust_gt0:
            in_battle_now = float(feats.get("cust", 0.0)) > 0.0

        # Segment boundary: if we were in battle and now we're not, wipe history.
        if self._in_battle and not in_battle_now:
            self.reset()
            return

        self._in_battle = in_battle_now
        if not self._in_battle:
            self._raw_t += 1
            return

        hold = max(1, int(self.cfg.hold))

        # Sample exactly when raw_t % hold == 0, matching precache sample_raw = range(0, n, hold)
        if (self._raw_t % hold) == 0:
            sc = torch.tensor(
                [
                    float(feats["p_hp"]) / 2500.0,
                    float(feats["e_hp"]) / 2500.0,
                    float(feats["p_chg"]) / 2.0,
                    float(feats["e_chg"]) / 2.0,
                    float(feats["cust"]) / 100.0,
                ],
                dtype=torch.float32,
            )

            self.scalars.append(sc)
            self.p_emo.append(int(feats["p_emo"]))
            self.e_emo.append(int(feats["e_emo"]))
            self.grid_tile.append(torch.tensor(feats["grid_tile"], dtype=torch.int64))
            self.grid_owner.append(torch.tensor(feats["grid_owner"], dtype=torch.int64))
            self.p_grid_idx.append(int(feats["p_grid_idx"]))
            self.e_grid_idx.append(int(feats["e_grid_idx"]))
            self.rel_pe_idx.append(int(feats["rel_pe_idx"]))
            self.player_chip.append(int(feats["player_chip"]))

            # OR aggregate of last `hold` raw actions (live analogue of aggregate_action over hold frames)
            lo = max(0, len(self._raw_actions) - hold)
            a = torch.zeros((ACTION_DIM,), dtype=torch.float32)
            for j in range(lo, len(self._raw_actions)):
                a = torch.maximum(a, self._raw_actions[j].to(torch.float32))
            self.action.append(a)

            # rolling window to seq_len
            T = int(self.cfg.seq_len)
            if len(self.scalars) > T:
                self.scalars.pop(0)
                self.p_emo.pop(0)
                self.e_emo.pop(0)
                self.grid_tile.pop(0)
                self.grid_owner.pop(0)
                self.p_grid_idx.pop(0)
                self.e_grid_idx.pop(0)
                self.rel_pe_idx.pop(0)
                self.player_chip.pop(0)
                self.action.pop(0)

        self._raw_t += 1

    def build_x(self) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        Returns:
          xb: dict with shapes [1,T,...]
          cur_t: last valid token index (0-based)
          hist_n: number of valid tokens
        Note: We RIGHT-pad with zeros (like cache did by slicing then padding).
        """
        T = int(self.cfg.seq_len)
        n_total = int(len(self.scalars))
        if n_total <= 0:
            xb = {
                "scalars": torch.zeros((1, T, 5), dtype=torch.float32),
                "p_emotion_id": torch.zeros((1, T), dtype=torch.int64),
                "e_emotion_id": torch.zeros((1, T), dtype=torch.int64),
                "grid_tile": torch.zeros((1, T, 18), dtype=torch.int64),
                "grid_owner": torch.zeros((1, T, 18), dtype=torch.int64),
                "p_grid_idx": torch.zeros((1, T), dtype=torch.int64),
                "e_grid_idx": torch.zeros((1, T), dtype=torch.int64),
                "rel_pe_idx": torch.zeros((1, T), dtype=torch.int64),
                "player_chip": torch.zeros((1, T), dtype=torch.int64),
                "action": torch.zeros((1, T, ACTION_DIM), dtype=torch.float32),
            }
            return xb, 0, 0

        n = min(n_total, T)
        cur_t = n - 1

        def _stack_pad_2d(src: List[torch.Tensor], d1: int, dtype: torch.dtype) -> torch.Tensor:
            out = torch.zeros((T, d1), dtype=dtype)
            tail = src[-n:]
            for i in range(n):
                out[i] = tail[i].to(dtype=dtype)
            return out

        def _stack_pad_1d_int(src: List[int]) -> torch.Tensor:
            out = torch.zeros((T,), dtype=torch.int64)
            tail = src[-n:]
            for i in range(n):
                out[i] = int(tail[i])
            return out

        xb = {
            "scalars": _stack_pad_2d(self.scalars, 5, torch.float32).unsqueeze(0),
            "p_emotion_id": _stack_pad_1d_int(self.p_emo).unsqueeze(0),
            "e_emotion_id": _stack_pad_1d_int(self.e_emo).unsqueeze(0),
            "grid_tile": _stack_pad_2d(self.grid_tile, 18, torch.int64).unsqueeze(0),
            "grid_owner": _stack_pad_2d(self.grid_owner, 18, torch.int64).unsqueeze(0),
            "p_grid_idx": _stack_pad_1d_int(self.p_grid_idx).unsqueeze(0),
            "e_grid_idx": _stack_pad_1d_int(self.e_grid_idx).unsqueeze(0),
            "rel_pe_idx": _stack_pad_1d_int(self.rel_pe_idx).unsqueeze(0),
            "player_chip": _stack_pad_1d_int(self.player_chip).unsqueeze(0),
            "action": _stack_pad_2d(self.action, ACTION_DIM, torch.float32).unsqueeze(0),
        }
        return xb, int(cur_t), int(n)


# =============================================================================
# Model runner (loads ckpt produced by critic_minimal/train.py)
# =============================================================================


class _Runner:
    def __init__(self, ckpt_path: str, device: torch.device, *, use_amp: bool):
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        ck = torch.load(str(ckpt_path), map_location="cpu")
        if not isinstance(ck, dict) or "model" not in ck or "cfg" not in ck:
            raise RuntimeError(f"Bad checkpoint: {ckpt_path}")

        cfg_dict = ck.get("cfg")
        if not isinstance(cfg_dict, dict):
            raise RuntimeError(f"Bad ckpt cfg: {ckpt_path}")

        # Prefer attribute-style construction (matches your existing MinimalQCritic(cfg) pattern)
        try:
            cfg_obj = type("MinimalQCfg", (), cfg_dict)()
            model = MinimalQCritic(cfg_obj)
        except Exception:
            model = MinimalQCritic(**cfg_dict)  # type: ignore[arg-type]

        model.load_state_dict(ck["model"], strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

        ynf = ck.get("y_norm_factor", None)
        self.y_norm_factor = float(ynf) if isinstance(ynf, (int, float)) and float(ynf) > 1e-6 else 350.0

    @torch.no_grad()
    def score_batch_last_token_raw(self, xb: Dict[str, torch.Tensor], cur_t: int) -> torch.Tensor:
        """
        xb: dict with leading batch dim [B,T,...]
        returns: raw-estimated score [B] (atanh(tanh_q)*y_norm_factor)
        """
        xb_d = {k: v.to(self.device, non_blocking=True) for k, v in xb.items()}

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        with amp_ctx:
            q = self.model(xb_d)  # [B,T] tanh-space

        qt = q[:, int(cur_t)]
        qt = torch.clamp(qt, -0.999, 0.999)

        # atanh(x) = 0.5*ln((1+x)/(1-x))
        raw = 0.5 * torch.log((1.0 + qt) / (1.0 - qt)) * float(self.y_norm_factor)
        return raw


# =============================================================================
# Key mapping (internal BUTTON_KEYS -> your Tango KEY_BIT_POSITIONS)
# =============================================================================


def _first_bit(key_bits: Dict[str, int], names: Sequence[str]) -> Optional[int]:
    for n in names:
        if n in key_bits and key_bits[n] is not None:
            return int(key_bits[n])
    return None


@dataclass(frozen=True)
class _KeyMap:
    up: Optional[int]
    down: Optional[int]
    left: Optional[int]
    right: Optional[int]
    start: Optional[int]
    back: Optional[int]
    l: Optional[int]
    r: Optional[int]
    a: Optional[int]
    b: Optional[int]


def _build_keymap(key_bit_positions: Dict[str, int]) -> _KeyMap:
    """
    CRITICAL: preserve your repo's historical mapping preference:
      - A / EAST  => Z (preferred) then A then EAST
      - B / SOUTH => X (preferred) then B then SOUTH
    This is what made the old version press SOUTH correctly.
    """
    kb = dict(key_bit_positions or {})
    return _KeyMap(
        up=_first_bit(kb, ("UP", "DPAD_UP", "DpadUp")),
        down=_first_bit(kb, ("DOWN", "DPAD_DOWN", "DpadDown")),
        left=_first_bit(kb, ("LEFT", "DPAD_LEFT", "DpadLeft")),
        right=_first_bit(kb, ("RIGHT", "DPAD_RIGHT", "DpadRight")),
        start=_first_bit(kb, ("RETURN", "START")),
        back=_first_bit(kb, ("ESCAPE", "BACK")),
        l=_first_bit(kb, ("L", "LB", "LEFT_SHOULDER")),
        r=_first_bit(kb, ("R", "RB", "RIGHT_SHOULDER")),
        a=_first_bit(kb, ("Z", "A", "EAST")),      # <-- key fix
        b=_first_bit(kb, ("X", "B", "SOUTH")),     # <-- key fix
    )


def _action_mask_to_key_mask(action_mask: int, km: _KeyMap) -> int:
    """
    action_mask is in critic_minimal BUTTON_KEYS bitspace.
    key_mask is in emulator KEY_BIT_POSITIONS bitspace (16-bit).
    """
    idx = {k: i for i, k in enumerate(BUTTON_KEYS)}

    def bit_on(name: str) -> bool:
        return bool((int(action_mask) >> int(idx[name])) & 1)

    out = 0
    if bit_on("DPAD_UP") and km.up is not None:
        out |= (1 << km.up)
    if bit_on("DPAD_DOWN") and km.down is not None:
        out |= (1 << km.down)
    if bit_on("DPAD_LEFT") and km.left is not None:
        out |= (1 << km.left)
    if bit_on("DPAD_RIGHT") and km.right is not None:
        out |= (1 << km.right)

    # We do not intentionally emit START/BACK for this policy, but keep mapping for completeness.
    if bit_on("START") and km.start is not None:
        out |= (1 << km.start)
    if bit_on("BACK") and km.back is not None:
        out |= (1 << km.back)

    if bit_on("LEFT_SHOULDER") and km.l is not None:
        out |= (1 << km.l)
    if bit_on("RIGHT_SHOULDER") and km.r is not None:
        out |= (1 << km.r)

    if bit_on("EAST") and km.a is not None:
        out |= (1 << km.a)
    if bit_on("SOUTH") and km.b is not None:
        out |= (1 << km.b)

    return int(out) & 0xFFFF


# =============================================================================
# Per-port state
# =============================================================================


@dataclass
class _PortState:
    hist: _History
    last_sent_action_mask: int  # in critic_minimal action bitspace


# =============================================================================
# Public Strategy (THIS is what your run_selfplay_inference imports)
# =============================================================================


class MinimalCriticBattleStrategy:
    """
    Drop-in for your selfplay code.

    Expected methods:
      - reset_state(port)
      - decide_action(port, game_state) -> dict with:
          button_command: {"type":"key_press","key":<bin16>}
          ng_key_bin: ""
          debug: {...}

    Notes:
      - requires_image=False so SafeHybridStrategy won't block it on missing image
      - we do NOT rely on game_state containing controller keys; we write our own history
    """

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: torch.device,
        key_bit_positions: Dict[str, int],
        hold: int = 4,
        seq_len: int = 192,
        start_stride: int = 1,  # kept for API compatibility (not used live)
        require_cust_gt0: bool = True,
        use_amp: bool = True,
        warmup_tokens: int = 4,
        candidates: Optional[List[int]] = None,
        top_k_debug: int = 5,
    ):
        self.requires_image = False  # important for SafeHybridStrategy gating

        self._runner = _Runner(str(ckpt_path), device=torch.device(device), use_amp=bool(use_amp))
        self._km = _build_keymap(key_bit_positions)

        self._hold = max(1, int(hold))
        self._seq_len = max(8, int(seq_len))
        self._require_cust_gt0 = bool(require_cust_gt0)

        self._warmup_tokens = max(0, int(warmup_tokens))
        self._top_k_debug = max(0, int(top_k_debug))

        # candidates in critic_minimal action-bitspace
        self._candidates = list(candidates) if candidates is not None else _default_candidate_masks_no_menu()
        if not self._candidates:
            self._candidates = [0]

        # per-port state
        self._by_port: Dict[int, _PortState] = {}

        # kept for API compatibility with your constructor calls
        self._start_stride = max(1, int(start_stride))

    def reset_state(self, port: int):
        self._by_port.pop(int(port), None)

    def _get_state(self, port: int) -> _PortState:
        p = int(port)
        st = self._by_port.get(p)
        if st is None:
            hist = _History(_HistCfg(hold=self._hold, seq_len=self._seq_len, require_cust_gt0=self._require_cust_gt0))
            st = _PortState(hist=hist, last_sent_action_mask=0)
            self._by_port[p] = st
        return st

    def decide_action(self, port: int, game_state: dict) -> dict:
        p = int(port)
        st = self._get_state(p)

        # Always write what we *sent last call* as the raw action for THIS tick.
        # Since Tango doesn't echo the controller, this is the only consistent history.
        st.hist.record_raw_action(_action_mask_to_action_vec(st.last_sent_action_mask))

        # Update from latest state (may segment-reset if battle ends)
        st.hist.update_from_state(game_state)

        inside_window = bool(float(game_state.get("inside_window", 0) or 0))
        cust = int(float(game_state.get("cust_gauge", 0) or 0))

        # If we are not in battle, do nothing (also keeps history stable).
        if not st.hist.in_battle():
            st.last_sent_action_mask = 0
            key_mask = 0
            return {
                "button_command": {"type": "key_press", "key": _int_to_bin16(key_mask)},
                "ng_key_bin": "",
                "debug": {
                    "active_model": "battle",
                    "note": "not_in_battle",
                    "inside_window": bool(inside_window),
                    "cust": int(cust),
                },
            }

        xb1, cur_t, hist_n = st.hist.build_x()

        # Warmup: until we have a few tokens, emit no-op.
        if hist_n < self._warmup_tokens:
            st.last_sent_action_mask = 0
            key_mask = 0
            return {
                "button_command": {"type": "key_press", "key": _int_to_bin16(key_mask)},
                "ng_key_bin": "",
                "debug": {
                    "active_model": "battle",
                    "note": "warmup",
                    "hist_n": int(hist_n),
                    "cur_t": int(cur_t),
                    "inside_window": bool(inside_window),
                    "cust": int(cust),
                },
            }

        # Evaluate candidates by batching: tile xb and override action at cur_t.
        C = int(len(self._candidates))
        if C <= 0:
            st.last_sent_action_mask = 0
            return {
                "button_command": {"type": "key_press", "key": _int_to_bin16(0)},
                "ng_key_bin": "",
                "debug": {"active_model": "battle", "note": "no_candidates"},
            }

        xb: Dict[str, torch.Tensor] = {}
        for k, v in xb1.items():
            if k == "action":
                continue
            # v shape: [1,T,...] -> [C,T,...]
            reps = [C] + [1] * (v.ndim - 1)
            xb[k] = v.repeat(*reps)

        action = xb1["action"].repeat(C, 1, 1)  # [C,T,ACTION_DIM]
        for i, m in enumerate(self._candidates):
            action[i, int(cur_t)] = _action_mask_to_action_vec(int(m))
        xb["action"] = action

        t0 = _now_ms()
        scores = self._runner.score_batch_last_token_raw(xb, cur_t=int(cur_t))  # [C]
        infer_ms = _now_ms() - t0

        best_i = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
        best_action_mask = int(self._candidates[best_i])

        # store as "sent" for next tick history
        st.last_sent_action_mask = best_action_mask

        # Map to emulator key bits
        key_mask = _action_mask_to_key_mask(best_action_mask, self._km)
        key_bin16 = _int_to_bin16(key_mask)

        # Debug top-K
        top_dbg: List[Dict[str, Any]] = []
        if self._top_k_debug > 0 and scores.numel() > 0:
            k = min(self._top_k_debug, int(scores.numel()))
            vals, idxs = torch.topk(scores, k=k, largest=True)
            for vv, ii in zip(vals.tolist(), idxs.tolist()):
                am = int(self._candidates[int(ii)])
                top_dbg.append({"action_mask": int(am), "score_raw": float(vv)})

        return {
            "button_command": {"type": "key_press", "key": key_bin16},
            "ng_key_bin": "",
            "debug": {
                "active_model": "battle",
                "note": "critic_minimal",
                "infer_ms": float(infer_ms),
                "inside_window": bool(inside_window),
                "cust": int(cust),
                "hist_n": int(hist_n),
                "cur_t": int(cur_t),
                "cands": int(C),
                "best_action_mask": int(best_action_mask),
                "best_key_mask": int(key_mask),
                "top": top_dbg,
            },
        }


__all__ = ["MinimalCriticBattleStrategy"]
