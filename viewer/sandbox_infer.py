# viewer/sandbox_infer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request


# ----------------------------
# Small helpers (pure)
# ----------------------------
def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple)) and x:
            return int(x[0])
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple)) and x:
            return float(x[0])
        return float(x)
    except Exception:
        return default


def _is_finite(x: Any) -> bool:
    try:
        xf = float(x)
        return xf == xf and xf not in (float("inf"), float("-inf"))
    except Exception:
        return False


def _pad_list(x: Any, n: int, fill: Any) -> List[Any]:
    xs = list(x) if isinstance(x, (list, tuple)) else []
    xs = xs[:n]
    if len(xs) < n:
        xs = xs + [fill] * (n - len(xs))
    return xs


def _coerce_values_by_frame(values_by_frame: Any, n_frames: int) -> List[Optional[float]]:
    """
    Normalize CriticRunner output into a dense list of length n_frames containing
    floats or None (no NaN/inf).
    Supports:
      - list/tuple of per-frame values (may include None/NaN)
      - dict mapping frame_idx -> value
    """
    out: List[Optional[float]] = [None] * n_frames
    if n_frames <= 0:
        return out

    if isinstance(values_by_frame, dict):
        for k, v in values_by_frame.items():
            try:
                i = int(k)
            except Exception:
                continue
            if 0 <= i < n_frames and _is_finite(v):
                out[i] = float(v)
        return out

    if isinstance(values_by_frame, (list, tuple)):
        m = min(len(values_by_frame), n_frames)
        for i in range(m):
            v = values_by_frame[i]
            if _is_finite(v):
                out[i] = float(v)
        return out

    return out


# ----------------------------
# Config / Contracts
# ----------------------------
@dataclass(frozen=True)
class SandboxCfg:
    # Must match critic_minimal runtime config
    hold: int
    seq_len: int
    start_stride: int
    require_cust_gt0: bool


@dataclass(frozen=True)
class SandboxDeps:
    # UI
    ui_buttons: List[str]
    # chips
    chip_db: Dict[str, Dict[str, Any]]
    # critic access (avoid circular import)
    get_critic: Callable[[], Any]


def _read_cfg_from_env() -> SandboxCfg:
    def _env_int(k: str, d: int) -> int:
        try:
            return int(os.environ.get(k, str(d)).strip() or str(d))
        except Exception:
            return d

    def _env_bool(k: str, d: bool) -> bool:
        v = (os.environ.get(k, "1" if d else "0") or "").strip()
        return v != "0"

    hold = max(1, _env_int("CRITIC_MINIMAL_HOLD", 4))
    seq_len = max(2, _env_int("CRITIC_MINIMAL_SEQ_LEN", 192))
    start_stride = max(1, _env_int("CRITIC_MINIMAL_START_STRIDE", 1))
    require_cust_gt0 = _env_bool("CRITIC_MINIMAL_REQUIRE_CUST_GT0", True)

    return SandboxCfg(
        hold=int(hold),
        seq_len=int(seq_len),
        start_stride=int(start_stride),
        require_cust_gt0=bool(require_cust_gt0),
    )


def _default_grid_owner() -> List[int]:
    # 18 tiles: left half (cols 0-2) owner=0, right half (cols 3-5) owner=1
    owners: List[int] = []
    for r in range(3):
        for c in range(6):
            owners.append(0 if c <= 2 else 1)
    return owners


def _build_synthetic_frame_from_state(
    state: Dict[str, Any],
    *,
    buttons: Dict[str, float],
) -> Dict[str, Any]:
    """
    Create a frame dict in the same *shape* as actions.jsonl frames.
    Keep types conservative: use scalars (0/1 floats) for buttons.
    """
    grid_owner = [int(x) for x in _pad_list(state.get("grid_owner", _default_grid_owner()), 18, 2)]
    grid_tile = [int(x) for x in _pad_list(state.get("grid_tile", [2] * 18), 18, 2)]

    # Positions are grid indices 0..17 in the UI; store both idx + derived-ish x/y placeholders.
    # Your real pipeline uses x/y, but many feature builders also use idx directly.
    p_idx = max(0, min(17, _as_int(state.get("player_grid_idx", 6), 6)))
    e_idx = max(0, min(17, _as_int(state.get("enemy_grid_idx", 11), 11)))

    # Minimal-but-safe “battle on” fields
    cust_gauge = max(1, min(100, _as_int(state.get("cust_gauge", 100), 100)))

    # Chip value: your frames typically use 65535 for NONE
    player_chip = _as_int(state.get("player_chip", 65535), 65535)
    enemy_chip = _as_int(state.get("enemy_chip", 65535), 65535)

    # Charges
    p_charge = max(0, _as_int(state.get("player_charge", 0), 0))
    e_charge = max(0, _as_int(state.get("enemy_charge", 0), 0))

    # Emotions/forms (keep as ints, even if model ignores)
    p_emotion = _as_int(state.get("player_emotion", 0), 0)
    e_emotion = _as_int(state.get("enemy_emotion", 0), 0)

    # HP (optional but common in reward/value shaping)
    p_hp = max(1, _as_int(state.get("player_health", 2000), 2000))
    e_hp = max(1, _as_int(state.get("enemy_health", 2000), 2000))

    frame: Dict[str, Any] = {
        # Battle mask driver
        "cust_gauge": cust_gauge,
        # Common fields used in your pipeline
        "player_health": p_hp,
        "enemy_health": e_hp,
        "player_charge": p_charge,
        "enemy_charge": e_charge,
        "player_chip": player_chip,
        "enemy_chip": enemy_chip,
        "player_emotion": p_emotion,
        "enemy_emotion": e_emotion,
        # Grid
        "grid_owner": grid_owner,
        "grid_tile": grid_tile,
        # Store idx explicitly (helpful even if x/y mapping differs)
        "player_grid_idx": p_idx,
        "enemy_grid_idx": e_idx,
    }

    # Buttons (action features)
    for k, v in (buttons or {}).items():
        # normalize to 0.0/1.0
        vv = 1.0 if float(v) > 0.5 else 0.0
        frame[k] = vv

    return frame


def _build_window_frames(
    state: Dict[str, Any],
    deps: SandboxDeps,
    cfg: SandboxCfg,
) -> List[Dict[str, Any]]:
    """
    Build a deterministic “history” window of length:
      win_len = (seq_len - 1) * hold + 1

    Semantics:
      - most frames use prev_buttons (or 0)
      - the endpoint uses next_buttons
    This lets you test “given this state/history, how good is the next controller choice?”
    """
    hold = int(cfg.hold)
    seq_len = int(cfg.seq_len)
    win_len = (seq_len - 1) * hold + 1

    prev_buttons = state.get("prev_buttons", {}) or {}
    next_buttons = state.get("next_buttons", {}) or {}

    # sanitize button dicts to only known keys
    def _filter(btns: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in deps.ui_buttons:
            if k in btns:
                out[k] = _as_float(btns.get(k), 0.0)
        return out

    prev_b = _filter(prev_buttons)
    next_b = _filter(next_buttons)

    # base “history” uses prev buttons
    window = [_build_synthetic_frame_from_state(state, buttons=prev_b) for _ in range(win_len)]

    # endpoint uses next buttons
    window[-1] = _build_synthetic_frame_from_state(state, buttons=next_b)
    return window


def create_sandbox_blueprint(deps: SandboxDeps) -> Blueprint:
    bp = Blueprint("sandbox_infer", __name__)

    @bp.get("/sandbox")
    def sandbox_page():
        # Expose buttons list to JS
        return render_template(
            "sandbox_infer.html",
            ui_buttons=deps.ui_buttons,
        )

    @bp.post("/api/sandbox_score")
    def api_sandbox_score():
        critic = deps.get_critic()
        if critic is None:
            return jsonify({"ok": False, "error": "Critic not loaded"}), 500

        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400

        cfg = _read_cfg_from_env()

        try:
            frames_window = _build_window_frames(payload, deps, cfg)
        except Exception as e:
            return jsonify({"ok": False, "error": f"window build failed: {e}"}), 400

        try:
            res = critic.infer_from_frames(
                frames=frames_window,
                hold=int(cfg.hold),
                seq_len=int(cfg.seq_len),
                start_stride=int(cfg.start_stride),
                require_cust_gt0=bool(cfg.require_cust_gt0),
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"infer failed: {e}"}), 500

        dense = _coerce_values_by_frame(getattr(res, "values_by_frame", None), len(frames_window))
        last = dense[-1] if dense else None

        return jsonify(
            {
                "ok": True,
                "critic_v_raw_last": float(last) if (last is not None and _is_finite(last)) else None,
                "meta": {
                    "hold": int(cfg.hold),
                    "seq_len": int(cfg.seq_len),
                    "start_stride": int(cfg.start_stride),
                    "require_cust_gt0": bool(cfg.require_cust_gt0),
                    "window_len": int(len(frames_window)),
                },
            }
        )

    return bp
