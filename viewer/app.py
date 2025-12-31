# viewer/app.py
from __future__ import annotations

import os
import sys
import json
import math
import io
import base64
import subprocess
import copy
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import torch
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify, request
import torchvision.transforms.functional as TF

from action_schema import get_button_tokens

UI_BUTTONS = get_button_tokens()

# -----------------------------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # ../
sys.path.append(parent_dir)

# Add 'strategy' folder to path (your repo uses this for some modules)
sys.path.append(os.path.join(parent_dir, "strategy"))

# -----------------------------------------------------------------------------
# IMPORT ACTION SCHEMA
# -----------------------------------------------------------------------------
try:
    from action_schema import BUTTON_TOKENS, GBA_UI_BUTTONS
except ImportError:
    print("⚠️ action_schema.py not found. Using defaults.")
    BUTTON_TOKENS = []
    GBA_UI_BUTTONS = []

# -----------------------------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# CONFIG & DIRECTORIES
# -----------------------------------------------------------------------------
DATASET_DIR = os.path.join(parent_dir, "data/dataset")
ASSETS_DIR = os.path.join(parent_dir, "data/assets")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
CHIPS_JSON_PATHS = [os.path.join(ASSETS_DIR, "chips.json")]
MASK_PATH = os.path.join(parent_dir, "chip_window_mask.png")

# Cache Locations
CACHE_DIRS: Dict[str, str] = {
    "Legacy": os.path.join(parent_dir, "data/dataset_cached"),
    "Plan": os.path.join(parent_dir, "data/planning_cache"),
    "Battle": os.path.join(parent_dir, "data/battle_cache"),
}

# Checkpoint Locations
CKPT_ROOT = os.path.join(parent_dir, "checkpoints")
#pring to validate
# print(f"Checkpoint root: {CKPT_ROOT}")
PLANNING_CKPT_DIR = ""#os.path.join(CKPT_ROOT, "planning")
BATTLE_CKPT_DIR = os.path.join(CKPT_ROOT, "nitrogen_battle_critic")
#validate battle
# print(f"Battle Checkpoint dir: {BATTLE_CKPT_DIR}")
# Strategy & RL Paths
STRATEGY_DB_PATH = os.path.join(parent_dir, "data/chipwindows_v2/strategy_v2.jsonl")
STRATEGY_MODEL_PATH = os.path.join(parent_dir, "checkpoints_strategy/strategy_model.pt")
RL_WEIGHTS_PATH = os.path.join(parent_dir, "data/nitrogen_rl/frame_weights.jsonl")
RL_EVENTS_PATH = os.path.join(parent_dir, "data/nitrogen_rl/rl_events.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-memory caches
RL_EVENTS_CACHE: List[Dict[str, Any]] = []
RL_DATA_CACHE: List[Dict[str, Any]] = []  # legacy weights cache (kept; not used by RL inspector now)

# -----------------------------------------------------------------------------
# STRATEGY MODEL CONFIG (kept consistent with your latest notes)
# -----------------------------------------------------------------------------
STRATEGY_CONFIG = {
    "num_chip_ids": 512,
    "num_codes": 32,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "dropout": 0.0,
}

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
# --- COORDINATE MATH ---
def pos_to_grid_idx(x: float, y: float) -> int:
    """
    Data-driven mapping of (x,y) to 0-17 grid index.
    Based on dataset analysis:
      X clusters: 20, 60, 100, 140, 180, 220 (Standard 40px spacing)
      Y clusters: 260, 515, 770 (Internal fixed-point coords?)
    """
    # X: 40px spacing is confirmed by data.
    # Simple int division snaps 20->0, 60->1, 100->2 safely.
    col = int(x // 40)
    col = max(0, min(5, col))

    # Y: Use midpoints between the clusters seen in data.
    # Row 0 (260)  <-- boundary 387 --> Row 1 (515)
    # Row 1 (515)  <-- boundary 642 --> Row 2 (770)
    row = 1
    if y < 387:
        row = 0
    elif y > 642:
        row = 2
    
    return (row * 6) + col
def resolve_chip_info_from_id_code(raw_id: int, raw_code: int) -> Optional[Dict[str, Any]]:
    """
    Accepts numeric chip id + raw code parity format (same as resolve_chip_info),
    but works even when the source is derived-state (id/code dicts).
    """
    try:
        rid = int(raw_id)
    except Exception:
        return None
    if rid in (255, 65535) or rid <= 0:
        return None
    try:
        rc = int(raw_code)
    except Exception:
        rc = 0
    return resolve_chip_info(rid, rc)


def _rich_from_chip_dict_list(chips: Any) -> List[Dict[str, Any]]:
    """
    chips: [{"id": int, "code": int}, ...] (derived_state format)
    Returns a list of resolved chip info objects (no Nones).
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(chips, list):
        return out
    for ch in chips:
        if not isinstance(ch, dict):
            continue
        info = resolve_chip_info_from_id_code(ch.get("id", 0), ch.get("code", 0))
        if info:
            out.append(info)
    return out


def find_cache_info(filename: str, hint_label: str | None = None) -> Tuple[Optional[str], str]:
    """
    Returns (full_path, cache_type_label).
    If hint_label is provided, checks that specific folder first.
    """
    if hint_label and hint_label in CACHE_DIRS:
        candidate = os.path.join(CACHE_DIRS[hint_label], filename)
        if os.path.exists(candidate):
            return candidate, hint_label

    for label, path in CACHE_DIRS.items():
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate, label

    return None, "Unknown"


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _detect_logits_like(values: List[float]) -> bool:
    if not values:
        return False
    mn, mx = min(values), max(values)
    return (mn < -0.05) or (mx > 1.05)


def _detect_sticks_01_like(stick2: List[float]) -> bool:
    mn, mx = min(stick2), max(stick2)
    return (mn >= -0.05) and (mx <= 1.05)


def _clamp_index(i: int, n: int) -> int:
    return max(0, min(int(i), int(n) - 1))


# -----------------------------------------------------------------------------
# VIDEO METADATA (FPS) – authoritative for UI time<->frame mapping
# -----------------------------------------------------------------------------
def _find_video_path_for_replay(replay_name: str) -> Optional[str]:
    folder = os.path.join(DATASET_DIR, replay_name)
    candidates = ["video.mp4", "video.webm", "video.mkv", "video.mov"]
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p

    if os.path.isdir(folder):
        for name in os.listdir(folder):
            lower = name.lower()
            if lower.endswith((".mp4", ".webm", ".mkv", ".mov")):
                return os.path.join(folder, name)

    return None


@lru_cache(maxsize=4096)
def _ffprobe_video_info(video_path: str) -> Dict[str, Any]:
    """
    Returns:
      - fps: float
      - duration: float|None
      - nb_frames: int|None
      - r_frame_rate: str|None
      - avg_frame_rate: str|None

    Never raises (falls back to fps=60.0).
    """
    s0: Dict[str, Any] = {}
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            video_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
        j = json.loads(out)
        streams = j.get("streams") or []
        s0 = streams[0] if streams else {}
    except Exception:
        s0 = {}

    def _parse_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _parse_int(x: Any) -> Optional[int]:
        try:
            if x is None:
                return None
            return int(x)
        except Exception:
            return None

    def _parse_rate(r: Any) -> Optional[float]:
        if not r:
            return None
        try:
            return float(Fraction(str(r)))
        except Exception:
            return None

    duration = _parse_float(s0.get("duration"))
    nb_frames = _parse_int(s0.get("nb_frames"))
    r_rate = s0.get("r_frame_rate")
    avg_rate = s0.get("avg_frame_rate")

    fps: Optional[float] = None

    # Prefer nb_frames/duration when present (often best for "almost VFR" encodes)
    if duration and duration > 0 and nb_frames and nb_frames > 0:
        fps = float(nb_frames) / float(duration)

    # Fall back to explicit rate strings
    if not fps or fps <= 0:
        fps = _parse_rate(r_rate) or _parse_rate(avg_rate)

    # Hard fallback
    if not fps or fps <= 0:
        fps = 60.0

    return {
        "fps": float(fps),
        "duration": duration,
        "nb_frames": nb_frames,
        "r_frame_rate": str(r_rate) if r_rate is not None else None,
        "avg_frame_rate": str(avg_rate) if avg_rate is not None else None,
    }


# -----------------------------------------------------------------------------
# MASK LOADING (for cache inspector masking)
# -----------------------------------------------------------------------------
_CACHED_MASK_TENSOR: Optional[torch.Tensor] = None


def get_mask_tensor() -> Optional[torch.Tensor]:
    global _CACHED_MASK_TENSOR
    if _CACHED_MASK_TENSOR is None:
        if os.path.exists(MASK_PATH):
            try:
                img = Image.open(MASK_PATH).convert("RGBA").resize((256, 256), Image.NEAREST)
                alpha = TF.to_tensor(img)[3, :, :]
                _CACHED_MASK_TENSOR = (alpha == 0.0).bool()  # True = keep
            except Exception as e:
                print(f"Failed to load mask: {e}")
                _CACHED_MASK_TENSOR = None
    return _CACHED_MASK_TENSOR


# -----------------------------------------------------------------------------
# NITROGEN ENGINE (DUAL MODEL SUPPORT)
# -----------------------------------------------------------------------------
class DualModelEngine:
    def __init__(self):
        self.models: Dict[str, Any] = {"Battle": None, "Plan": None}
        self.configs: Dict[str, Dict[str, Any]] = {
            "Battle": {"action_horizon": 18, "vision_horizon": 1, "old_layout": False},
            "Plan": {"action_horizon": 18, "vision_horizon": 1, "old_layout": False},
        }

        try:
            from ng_policy import NgNitroGenPolicy, load_ng_checkpoint  # type: ignore

            self.PolicyClass = NgNitroGenPolicy
            self.loader_func = load_ng_checkpoint
        except ImportError:
            print("⚠️ Nitrogen policy code not found.")
            self.PolicyClass = None
            self.loader_func = None
            return

        self._load_best_ckpt(BATTLE_CKPT_DIR, "Battle")
        self._load_best_ckpt(PLANNING_CKPT_DIR, "Plan")

    def _load_best_ckpt(self, ckpt_dir: str, key: str) -> None:
        if not ckpt_dir or not os.path.exists(ckpt_dir):
            return

        max_step = -1
        best_file = None
        for f in os.listdir(ckpt_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                try:
                    step = int(f.split("_")[1].split(".")[0])
                    if step > max_step:
                        max_step = step
                        best_file = f
                except Exception:
                    continue

        if not best_file:
            return

        path = os.path.join(ckpt_dir, best_file)
        print(f"🧠 Loading {key} Model: {best_file}...")
        try:
            loaded = self.loader_func(path, device=torch.device(DEVICE))  # type: ignore[misc]
            policy = self.PolicyClass(loaded).to(DEVICE).eval()  # type: ignore[misc]
            self.models[key] = policy

            tok = getattr(policy, "tokenizer", None)
            if tok:
                self.configs[key]["action_horizon"] = getattr(tok, "action_horizon", 18)
                self.configs[key]["vision_horizon"] = getattr(tok, "vision_horizon", 1)
                self.configs[key]["old_layout"] = getattr(tok, "old_layout", False)

            print(f"✅ {key} Model Loaded.")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")

    def infer(self, frames_tensor: torch.Tensor, model_key: str = "Battle") -> Optional[List[List[float]]]:
        if model_key not in self.models:
            model_key = "Battle"
        policy = self.models.get(model_key)
        if policy is None:
            return None

        if frames_tensor.ndim == 3:
            frames_tensor = frames_tensor.unsqueeze(0)  # [V,C,H,W]
        frames_batch = frames_tensor.unsqueeze(0).to(DEVICE, non_blocking=True)  # [B,V,C,H,W]

        with torch.inference_mode():
            action_seq = policy(frames_batch, take_step=0, return_continuous=True, return_sequence=True)

        return action_seq.squeeze(0).detach().cpu().tolist()


engine = DualModelEngine()


def _coerce_scalar_float(v: object, default: float = 0.0) -> float:
    """
    Accept values that may be scalars or 1-element lists (common in telemetry dumps).
    Return a float, never raising.
    """
    try:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return float(default)
            v = v[0]
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _json_action_row_to_display(row: dict, ui_buttons: list[str]) -> dict:
    """
    Base dataset -> RL inspector row.

    Contract:
      - always include frame_idx (int, if present)
      - buttons_raw uses the SAME token list the UI uses (GBA_UI_BUTTONS)
    """
    buttons: Dict[str, float] = {}
    for k in ui_buttons:
        buttons[k] = _coerce_scalar_float(row.get(k, 0.0), 0.0)

    fi = row.get("frame_idx", None)
    try:
        fi_i = int(fi) if fi is not None else None
    except Exception:
        fi_i = None

    out: Dict[str, Any] = {"buttons_raw": buttons}
    if fi_i is not None:
        out["frame_idx"] = fi_i
    return out


# -----------------------------------------------------------------------------
# CHIP DATABASE
# -----------------------------------------------------------------------------
CHIP_DB: Dict[str, Dict[str, Any]] = {}


def load_chip_db() -> None:
    global CHIP_DB
    print(f"📂 Scanning chip data in: {ASSETS_DIR}")
    for path in CHIPS_JSON_PATHS:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for chip in data:
                        if chip.get("SId") is not None:
                            CHIP_DB[f"S{chip['SId']}"] = chip
                        if chip.get("MId") is not None:
                            CHIP_DB[f"M{chip['MId']}"] = chip
            except Exception as e:
                print(f"  ❌ Error loading {path}: {e}")


load_chip_db()

CODE_INDEXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"
def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x) != 0.0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    return False


def _to_list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, (list, tuple)) else []

def resolve_chip_info(raw_id: int, raw_code: int) -> Optional[Dict[str, Any]]:
    """
    Strategy page resolver: uses code parity to decide S vs M.
    raw_id 255 means empty.
    """
    if raw_id == 255:
        return None

    is_mid = (raw_code % 2 == 1)
    code_idx = raw_code // 2
    char_code = CODE_INDEXES[code_idx] if 0 <= code_idx < len(CODE_INDEXES) else "?"

    lookup_key = f"M{raw_id}" if is_mid else f"S{raw_id}"
    chip_data = CHIP_DB.get(lookup_key)

    image_file = chip_data.get("Image") if chip_data else None
    name = chip_data.get("Name", f"ID:{raw_id}") if chip_data else f"ID:{raw_id}"

    return {
        "id": raw_id,
        "name": name,
        "char_code": char_code,
        "image_url": f"/assets/images/{image_file}" if image_file else None,
    }


# -----------------------------------------------------------------------------
# ACTION DISPLAY HELPERS (policy vector + cache vectors + jsonl rows)
# -----------------------------------------------------------------------------
def _split_policy_action_layout(v25: List[float], *, old_layout: bool) -> Tuple[List[float], List[float], List[float]]:
    if len(v25) != 25:
        return [0.0] * 21, [0.0, 0.0], [0.0, 0.0]
    if old_layout:
        jl = [float(v25[0]), float(v25[1])]
        jr = [float(v25[2]), float(v25[3])]
        buttons = [float(x) for x in v25[4:]]
    else:
        buttons = [float(x) for x in v25[:-4]]
        jl = [float(v25[-4]), float(v25[-3])]
        jr = [float(v25[-2]), float(v25[-1])]
    return buttons, jl, jr


def _policy_vec_to_display(v25: List[float], *, old_layout: bool) -> Dict[str, Any]:
    buttons_raw, jl_raw, jr_raw = _split_policy_action_layout(v25, old_layout=old_layout)

    jl01 = _detect_sticks_01_like(jl_raw)
    jr01 = _detect_sticks_01_like(jr_raw)
    jl_axis = [x * 2.0 - 1.0 for x in jl_raw] if jl01 else jl_raw[:]
    jr_axis = [x * 2.0 - 1.0 for x in jr_raw] if jr01 else jr_raw[:]

    buttons_are_logits = _detect_logits_like(buttons_raw)
    buttons_prob = (
        [_sigmoid(x) for x in buttons_raw]
        if buttons_are_logits
        else [max(0.0, min(1.0, float(x))) for x in buttons_raw]
    )

    return {
        "axes": {
            "AXIS_LEFTX": float(jl_axis[0]),
            "AXIS_LEFTY": float(jl_axis[1]),
            "AXIS_RIGHTX": float(jr_axis[0]),
            "AXIS_RIGHTY": float(jr_axis[1]),
        },
        "sticks_raw": {"j_left": jl_raw, "j_right": jr_raw},
        "buttons_raw": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_raw)},
        "buttons_prob": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_prob)},
        "meta": {"buttons_are_logits": buttons_are_logits},
    }


def _cache_vec_to_gt(v: List[float]) -> Dict[str, Any]:
    # Cache format: [LX, LY, RX, RY, BTNS...]
    d: Dict[str, Any] = {
        "AXIS_LEFTX": float(v[0]),
        "AXIS_LEFTY": float(v[1]),
        "AXIS_RIGHTX": float(v[2]),
        "AXIS_RIGHTY": float(v[3]),
    }
    btn_vals = v[4:]
    for name, val in zip(BUTTON_TOKENS, btn_vals):
        d[name] = float(val)
    return d


def _json_action_to_display(json_row: dict) -> dict:
    axes = {
        "AXIS_LEFTX": float(json_row.get("AXIS_LEFTX", 0.0)),
        "AXIS_LEFTY": float(json_row.get("AXIS_LEFTY", 0.0)),
        "AXIS_RIGHTX": float(json_row.get("AXIS_RIGHTX", 0.0)),
        "AXIS_RIGHTY": float(json_row.get("AXIS_RIGHTY", 0.0)),
    }

    buttons_raw: Dict[str, float] = {}
    for btn in BUTTON_TOKENS:
        buttons_raw[btn] = float(json_row.get(btn, 0.0))

    return {
        "axes": axes,
        "buttons_raw": buttons_raw,
        "buttons_prob": buttons_raw,
        "meta": {"source": "jsonl"},
    }


# -----------------------------------------------------------------------------
# RL: LOADERS
# -----------------------------------------------------------------------------
def load_rl_events() -> None:
    global RL_EVENTS_CACHE
    RL_EVENTS_CACHE = []

    if not os.path.exists(RL_EVENTS_PATH):
        print(f"⚠️  RL events not found: {RL_EVENTS_PATH}")
        return

    print(f"📌 Loading RL Events from {RL_EVENTS_PATH}...")
    try:
        with open(RL_EVENTS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    RL_EVENTS_CACHE.append(json.loads(line))
                except Exception:
                    continue

        # Sort: more "interesting" first
        RL_EVENTS_CACHE.sort(key=lambda e: abs(float(e.get("weight", 1.0)) - 1.0), reverse=True)
        print(f"✅ Loaded {len(RL_EVENTS_CACHE)} RL events.")
    except Exception as e:
        print(f"❌ Error loading RL events: {e}")


def load_rl_data() -> None:
    """
    Loads frame_weights.jsonl into memory (legacy cache).
    The RL inspector page now uses RL_EVENTS_CACHE for list/detail.
    """
    global RL_DATA_CACHE
    RL_DATA_CACHE = []

    if not os.path.exists(RL_WEIGHTS_PATH):
        return

    print(f"⚖️  Loading RL Weights from {RL_WEIGHTS_PATH}...")
    try:
        temp_list = []
        with open(RL_WEIGHTS_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                parts = obj["key"].split("/")
                if len(parts) >= 2:
                    temp_list.append({"replay": parts[0], "frame": int(parts[1]), "weight": obj["val"]})
        RL_DATA_CACHE = sorted(temp_list, key=lambda x: x["weight"], reverse=True)
        print(f"✅ Loaded {len(RL_DATA_CACHE)} RL samples.")
    except Exception as e:
        print(f"❌ Error loading RL weights: {e}")


load_rl_events()
load_rl_data()

# -----------------------------------------------------------------------------
# STRATEGY MODEL LOADING (single unified load)
# -----------------------------------------------------------------------------
strategy_model = None
meta_proc = None

try:
    from planning.strategy_model import StrategyTransformer, META_DIM  # type: ignore
    from planning.train_strategy_bc import ChipMetaProcessor  # type: ignore

    if os.path.exists(STRATEGY_MODEL_PATH) and os.path.exists(CHIPS_JSON_PATHS[0]):
        print(f"🧠 Loading Strategy Model from {STRATEGY_MODEL_PATH}...")
        try:
            meta_proc = ChipMetaProcessor(CHIPS_JSON_PATHS[0])

            strategy_model = StrategyTransformer(
                num_chip_ids=STRATEGY_CONFIG["num_chip_ids"],
                num_codes=STRATEGY_CONFIG["num_codes"],
                d_model=STRATEGY_CONFIG["d_model"],
                nhead=STRATEGY_CONFIG["nhead"],
                num_layers=STRATEGY_CONFIG["num_layers"],
                meta_dim=META_DIM,
                dropout=STRATEGY_CONFIG["dropout"],
            )

            # In case the checkpoint expects a different context dim
            EXPECTED_CTX = 36
            if strategy_model.context_proj.in_features != EXPECTED_CTX:
                print(f"⚠️ Resizing context projection to {EXPECTED_CTX}")
                strategy_model.context_dim = EXPECTED_CTX
                strategy_model.context_proj = torch.nn.Linear(EXPECTED_CTX, STRATEGY_CONFIG["d_model"])

            state_dict = torch.load(STRATEGY_MODEL_PATH, map_location=DEVICE)
            strategy_model.load_state_dict(state_dict)
            strategy_model.to(DEVICE).eval()
            print("✅ Strategy Model Loaded!")
        except Exception as e:
            print(f"❌ Failed to load Strategy Model: {e}")
except ImportError as e:
    print(f"⚠️ Strategy modules not found: {e}")

# =============================================================================
# RL INSPECTOR CONSTANTS
# =============================================================================
RL_PRE_EVENT_FRAMES = 18  # horizon lead-in (for actions window sizing; event_frame is not used for video loop)


def _parse_chip_id_from_label(label: str) -> Optional[int]:
    # label examples: "CHIP:123" or "CROSS:Normal" etc.
    if not label:
        return None
    if not label.startswith("CHIP:"):
        return None
    try:
        return int(label.split(":", 1)[1])
    except Exception:
        return None


def _chip_id_to_name_image(chip_id: int) -> tuple[Optional[str], Optional[str]]:
    """
    RL labels store numeric chip id. Our CHIP_DB is keyed by SId/MId with a prefix.
    We don't know whether the RL id is SId or MId, so we attempt both.
    """
    if chip_id is None:
        return None, None

    chip = CHIP_DB.get(f"S{chip_id}") or CHIP_DB.get(f"M{chip_id}")
    if not chip:
        return None, None

    name = chip.get("Name")
    image_file = chip.get("Image")
    image_url = f"/assets/images/{image_file}" if image_file else None
    return name, image_url


def _display_label(label: str) -> str:
    # nicer label in UI
    if not label:
        return ""
    if label.startswith("CHIP:"):
        cid = _parse_chip_id_from_label(label)
        return f"CHIP:{cid}" if cid is not None else label
    if label.startswith("CROSS:"):
        return label.replace("CROSS:", "CROSS: ")
    return label


def _event_frame_window(evt: Dict[str, Any]) -> tuple[int, int, int]:
    """
    Returns:
      start_frame, event_frame, end_frame

    Semantics:
      - video loops from start_frame..end_frame (inclusive)
      - event_frame is metadata only (DO NOT drive the loop)
    """
    def _as_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    event_frame = _as_int(evt.get("event_frame", 0), 0)
    start_frame = _as_int(evt.get("start_frame", 0), 0)
    end_frame = _as_int(evt.get("end_frame", start_frame), start_frame)

    if start_frame < 0:
        start_frame = 0
    if end_frame < start_frame:
        end_frame = start_frame

    return start_frame, event_frame, end_frame


def _chip_info_for_numeric_id(chip_id: int) -> Optional[Dict[str, Any]]:
    """
    Best-effort resolve for CHIP:<id> when we don't have code/is_mid.
    Preference order:
      1) SId match
      2) MId match
    """
    try:
        cid = int(chip_id)
    except Exception:
        return None

    s = CHIP_DB.get(f"S{cid}")
    m = CHIP_DB.get(f"M{cid}")
    chip = s or m
    if not chip:
        return None

    image_file = chip.get("Image")
    return {
        "id": cid,
        "name": chip.get("Name", f"ID:{cid}"),
        "image_url": f"/assets/images/{image_file}" if image_file else None,
        "kind": ("S" if s else "M") if (s or m) else None,
    }


def _filter_events(actor: str, kind: str, outcome: str, label: str) -> list[Dict[str, Any]]:
    data = RL_EVENTS_CACHE

    if actor in ("player", "enemy"):
        data = [x for x in data if x.get("actor") == actor]
    if kind in ("chip", "charge"):
        data = [x for x in data if x.get("kind") == kind]
    if outcome in ("good", "bad"):
        data = [x for x in data if x.get("outcome") == outcome]
    if label:
        data = [x for x in data if x.get("label") == label]

    return data


# -----------------------------------------------------------------------------
# RL ACTIONS LOADING (actions.jsonl indexed by frame_idx)
# -----------------------------------------------------------------------------
def _btn_scalar(v: Any) -> float:
    """
    Base dataset values can be:
      - number (0/1/float)
      - list like [0.0]
      - missing / None
    Return a clean float.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list) and v:
        x = v[0]
        return float(x) if isinstance(x, (int, float)) else 0.0
    return 0.0


def _load_actions_by_frame_idx(actions_jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Read base dataset actions.jsonl and index by row["frame_idx"].
    This avoids any mismatch if lines are missing or duplicated.
    """
    by_frame: Dict[int, Dict[str, Any]] = {}
    if not actions_jsonl_path.exists():
        return by_frame

    with actions_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            fi = row.get("frame_idx")
            if fi is None:
                continue
            try:
                fi_i = int(fi)
            except Exception:
                continue
            by_frame[fi_i] = row

    return by_frame


def _window_actions(
    *,
    actions_by_frame: Dict[int, Dict[str, Any]],
    ui_buttons: List[str],
    start_frame: int,
    end_frame: int,
) -> List[Dict[str, Any]]:
    """
    Return a dense list for [start_frame..end_frame] inclusive.
    Each entry contains:
      - frame_idx
      - buttons_raw: {token: float}
    Missing frames become all-zeros.
    """
    out: List[Dict[str, Any]] = []
    s = int(start_frame)
    e = int(end_frame)
    if e < s:
        e = s

    for fi in range(s, e + 1):
        row = actions_by_frame.get(fi, {})
        buttons = {k: _btn_scalar(row.get(k)) for k in ui_buttons}
        out.append({"frame_idx": fi, "buttons_raw": buttons})

    return out


def _load_actions_sequence_for_event(
    replay_name: str,
    *,
    event_frame: int,
    horizon: int = 18,
) -> tuple[list, str]:
    """
    RL inspector ground-truth MUST come from base actions.jsonl, indexed by row["frame_idx"].
    """
    jsonl_path = Path(DATASET_DIR) / replay_name / "actions.jsonl"
    if not jsonl_path.exists():
        return [], "none"

    try:
        actions_by_frame = _load_actions_by_frame_idx(jsonl_path)
        seq: List[Dict[str, Any]] = []

        s = int(max(0, event_frame))
        for fi in range(s, s + int(max(0, horizon))):
            row = actions_by_frame.get(fi, {})
            seq.append(_json_action_row_to_display(row, list(GBA_UI_BUTTONS)))

        return seq, "jsonl"
    except Exception as e:
        print(f"⚠️ RL: JSONL actions seq load failed for {replay_name}: {e}")
        return [], "none"


def _load_actions_window_for_replay(
    replay_name: str,
    *,
    start_frame: int,
    end_frame: int,
    horizon: int = 18,
) -> tuple[list, str]:
    """
    RL inspector window actions MUST come from base actions.jsonl, using frame_idx indexing.
    Returns rows for [start_frame .. end_frame + horizon - 1] inclusive.
    """
    start_frame = int(max(0, start_frame))
    end_frame = int(max(start_frame, end_frame))
    tail_end = int(end_frame + max(0, int(horizon) - 1))

    jsonl_path = Path(DATASET_DIR) / replay_name / "actions.jsonl"
    if not jsonl_path.exists():
        return [], "none"

    try:
        actions_by_frame = _load_actions_by_frame_idx(jsonl_path)
        out = _window_actions(
            actions_by_frame=actions_by_frame,
            ui_buttons=list(GBA_UI_BUTTONS),
            start_frame=start_frame,
            end_frame=tail_end,
        )
        return out, "jsonl"
    except Exception as e:
        print(f"⚠️ RL: JSONL window actions load failed for {replay_name}: {e}")
        return [], "none"


def _load_image_for_event(replay_name: str, *, start_frame: int) -> tuple[Optional[str], str]:
    """
    The "Model Input (start)" image should come from start_frame.
    Prefer cached PT frames; no JSONL fallback image.
    """
    pt_path, _ = find_cache_info(f"{replay_name}.pt")
    if not pt_path:
        return None, "none"
    try:
        data = torch.load(pt_path, map_location="cpu")
        frames = data.get("frames")
        if frames is None:
            return None, "none"
        n = int(frames.shape[0])
        idx = max(0, min(int(start_frame), n - 1))

        frame_uint8 = frames[idx]
        img_np = frame_uint8.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        return "data:image/png;base64," + b64_img, "pt"
    except Exception as e:
        print(f"⚠️ RL: PT image load failed for {replay_name}: {e}")
        return None, "none"


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    replays.sort()

    cached_files: Dict[str, List[str]] = {}
    for label, path in CACHE_DIRS.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".pt")]
            files.sort()
            cached_files[label] = files

    has_strategy = os.path.exists(STRATEGY_DB_PATH)
    return render_template("index.html", replays=replays, cached_files=cached_files, has_strategy=has_strategy)


@app.route("/view/<path:replay_name>")
def view_replay(replay_name):
    # Get sorted list of all replays to find neighbors
    replays = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    try:
        idx = replays.index(replay_name)
        prev_replay = replays[idx - 1] if idx > 0 else None
        next_replay = replays[idx + 1] if idx < len(replays) - 1 else None
    except ValueError:
        prev_replay = None
        next_replay = None

    return render_template("view.html", 
                           replay_name=replay_name, 
                           prev_replay=prev_replay, 
                           next_replay=next_replay)


@app.route("/video/<path:replay_name>")
def serve_video(replay_name):
    folder = os.path.join(DATASET_DIR, replay_name)

    # Prefer the canonical name, but fall back to common alternates.
    candidates = ["video.mp4", "video.webm", "video.mkv", "video.mov"]
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return send_from_directory(folder, name)

    # Last resort: serve the first file that looks like a video
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            lower = name.lower()
            if lower.endswith((".mp4", ".webm", ".mkv", ".mov")):
                return send_from_directory(folder, name)

    return jsonify({"error": f"Video not found for replay '{replay_name}'"}), 404


@app.route("/assets/images/<path:filename>")
def serve_chip_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/api/chip_library")
def api_chip_library():
    return jsonify(CHIP_DB)





def _pad_list(x: Any, n: int, fill: Any) -> List[Any]:
    xs = list(x) if isinstance(x, (list, tuple)) else []
    xs = xs[:n]
    if len(xs) < n:
        xs = xs + [fill] * (n - len(xs))
    return xs


def _rich_from_id_code_arrays(ids: Any, codes: Any, mask: Any, *, n: int) -> List[Dict[str, Any]]:
    ids_l = _pad_list(ids, n, 0)
    codes_l = _pad_list(codes, n, 0)
    mask_l = _pad_list(mask, n, False)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        if not _as_bool(mask_l[i]):
            continue
        info = resolve_chip_info_from_id_code(ids_l[i], codes_l[i])
        if info:
            out.append(info)
    return out


def _compute_selected_indices_v2(
    *,
    hand_ids: List[int],
    hand_codes: List[int],
    hand_vis: List[float],
    sel_ids: List[int],
    sel_codes: List[int],
    sel_mask: List[bool],
) -> List[int]:
    """
    Map selected chips back onto hand indices for highlighting.
    Uses first unused matching visible slot per selected chip.
    """
    used = set()
    out: List[int] = []

    n_hand = min(10, len(hand_ids), len(hand_codes), len(hand_vis))
    for k in range(min(5, len(sel_ids), len(sel_codes), len(sel_mask))):
        if not bool(sel_mask[k]):
            continue
        sid = int(sel_ids[k] or 0)
        sc = int(sel_codes[k] or 0)
        if sid <= 0:
            continue

        found = None
        for i in range(n_hand):
            if i in used:
                continue
            if float(hand_vis[i] or 0.0) <= 0.0:
                continue
            if int(hand_ids[i] or 0) == sid and int(hand_codes[i] or 0) == sc:
                found = i
                break

        if found is None:
            # fallback: match by id only (codes can be noisy)
            for i in range(n_hand):
                if i in used:
                    continue
                if float(hand_vis[i] or 0.0) <= 0.0:
                    continue
                if int(hand_ids[i] or 0) == sid:
                    found = i
                    break

        if found is not None:
            used.add(found)
            out.append(found)

    return out
# viewer/app.py (Partial - replace the solver section)

# ... imports ... (ensure copy is imported)

# =============================================================================
# STRATEGY SOLVER LOGIC
# =============================================================================

# =============================================================================
# STRATEGY SOLVER LOGIC
# =============================================================================

WILDCARD_CODE = 26  # * Code
MAX_SLOTS = 5
MAX_CANDIDATES_LIMIT = 20000 

def _is_chain_valid(chain_indices: List[int], all_ids: List[int], all_codes: List[int]) -> bool:
    """
    Validates if a specific sequence of chips is legal.
    CRITICAL FIX: Normalizes codes (c // 2) so 8 and 9 are treated as the same letter.
    """
    if not chain_indices:
        return True
        
    c_ids = [all_ids[i] for i in chain_indices]
    
    # 1. Check "Same ID" Rule (Overrides Code rules)
    first_id = c_ids[0]
    if all(idn == first_id for idn in c_ids):
        return True
        
    # 2. Check "Same Code" Rule (with Wildcards)
    c_codes = [all_codes[i] for i in chain_indices]
    active_code_idx = None  # We track the NORMALIZED index (0-26)
    
    for raw_c in c_codes:
        c_idx = raw_c // 2  # Normalize: 8->4, 9->4
        
        if c_idx == WILDCARD_CODE:
            continue
            
        if active_code_idx is None:
            active_code_idx = c_idx
        elif c_idx != active_code_idx:
            # Real mismatch (e.g. 4 vs 5)
            return False
            
    return True

def _generate_chip_chains(ids: List[int], codes: List[int], valid_indices: List[int]) -> List[List[int]]:
    """
    Generates all valid permutations of chips up to MAX_SLOTS.
    """
    # We use a stack for DFS: (current_chain_list, used_indices_set)
    stack = []
    
    # Init with single chips
    for i in valid_indices:
        stack.append(([i], {i}))
        
    valid_chains = [[]] # Empty hand is always an option
    
    # Optimization: To prevent 20,000 permutations of the same 5 chips just reordered 
    # (if they are all compatible), we might hit the limit. 
    # But MMBN order matters, so A->B is distinct from B->A. We must keep them.
    
    while stack:
        if len(valid_chains) >= MAX_CANDIDATES_LIMIT:
            print(f"[Solver] Hit safety limit of {MAX_CANDIDATES_LIMIT} chains.")
            break
            
        curr_chain, used = stack.pop()
        
        # Add to results
        valid_chains.append(curr_chain)
        
        if len(curr_chain) >= MAX_SLOTS:
            continue
            
        # Try to extend
        for i in valid_indices:
            if i in used:
                continue
            
            # Optimization: Pre-check validity before adding to stack
            # This prunes the tree significantly compared to generating then checking.
            
            # Check if adding 'i' to 'curr_chain' maintains validity
            # We only need to check the LAST transition constraint relative to the whole group.
            # Actually, because "Same ID" allows ANY code, and "Same Code" allows DIFFERENT IDs,
            # the rule is global to the hand, not just the transition.
            # So we effectively check: Is (curr_chain + [i]) valid?
            
            test_chain = curr_chain + [i]
            
            if _is_chain_valid(test_chain, ids, codes):
                new_used = used.copy()
                new_used.add(i)
                stack.append((test_chain, new_used))
                
    return valid_chains

def _solve_combinations(game_state):
    """
    Generates all valid moves (Chips + Crosses + Beast) and prepares 
    the SIMULATED game state dictionaries for the Critic.
    """
    # --- 1. Parse Hand ---
    hand_slots = [int(x) for x in game_state.get("hand_slots", [])]
    hand_codes = [int(x) for x in game_state.get("hand_codes", [])]
    
    # --- 2. Generate Chip Chains ---
    valid_indices = [i for i, x in enumerate(hand_slots) if x > 0 and x != 255 and x != 65535]
    chains = _generate_chip_chains(hand_slots, hand_codes, valid_indices)
    
    # --- 3. Generate Valid Forms (Crosses) ---
    derived_p = game_state.get("derived", {}).get("player", {})
    used_mask = derived_p.get("used_cross_mask", [])
    current_cross = int(derived_p.get("active_cross", {}).get("idx", 0))
    
    if len(used_mask) < 11:
        used_mask = used_mask + [False] * (11 - len(used_mask))

    has_gregar_history = any(used_mask[1:6])
    has_falzar_history = any(used_mask[6:11])
    
    if 1 <= current_cross <= 5: has_gregar_history = True
    if 6 <= current_cross <= 10: has_falzar_history = True

    # 0 is always allowed. It means "No Selection" (Stay in current form).
    valid_cross_selections = [0] 

    # Pool of potential NEW crosses
    pool_indices = []
    if has_gregar_history and not has_falzar_history:
        pool_indices = [1, 2, 3, 4, 5]
    elif has_falzar_history and not has_gregar_history:
        pool_indices = [6, 7, 8, 9, 10]
    else:
        # Ambiguous/Start: Check all
        pool_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in pool_indices:
        # Validity Rule:
        # 1. New Cross must not be used yet.
        # 2. New Cross cannot be the current one (that's what 0 is for).
        if i != current_cross and not used_mask[i]:
            valid_cross_selections.append(i)
            
    # --- 4. Generate Valid Beast Options ---
    beast_state = derived_p.get("beast", {})
    is_beast_active = beast_state.get("active", False)
    has_beasted_ever = beast_state.get("ever", False)
    
    beast_opts = []
    
    if is_beast_active:
        # If ALREADY in Beast, we cannot press the button again.
        # The 'beast_selected' action must be False.
        # (The Critic will infer we are still in Beast via derived state).
        beast_opts = [False]
    elif has_beasted_ever:
        # If used previously and finished, we cannot use it again.
        beast_opts = [False]
    else:
        # Available to toggle
        beast_opts = [False, True]

    # --- 5. Cartesian Product ---
    candidates = []
    
    for ch in chains:
        for cr in valid_cross_selections:
            for b in beast_opts:
                
                # CONSTRAINT: Cannot Change Cross AND Beast Out in same turn
                # cr > 0 means we are actively selecting a new cross.
                # b=True means we are pressing the Beast Button.
                if cr > 0 and b:
                    continue
                
                # Rule: Beast Out consumes 1 chip slot (limits hand to 4)
                # Only applies if we are PERFORMING the Beast Out action now.
                # If we are already in beast (b=False, is_beast_active=True), standard limits apply.
                # Actually, standard limit is always 5 unless selecting beast reduces it.
                # Wait: Does being IN Beast reduce slots? No, usually just the turn you pick it.
                
                max_allowed = MAX_SLOTS
                if b is True: # Action: Beast Out
                    max_allowed -= 1
                
                if len(ch) <= max_allowed:
                    candidates.append({
                        "chain": ch,
                        "cross_idx": cr,
                        "do_beast": b
                    })
    
    # --- 6. Build Simulated Rows ---
    sim_rows = []
    
    for cand in candidates:
        chain_idxs = cand["chain"]
        
        sim_ids = [hand_slots[i] for i in chain_idxs]
        sim_codes = [hand_codes[i] for i in chain_idxs]
        
        sim_ids_padded = sim_ids + [0] * (5 - len(sim_ids))
        sim_codes_padded = sim_codes + [0] * (5 - len(sim_codes))
        
        sim_row = game_state.copy()
        
        sim_row['selected_chips_id'] = sim_ids_padded
        sim_row['selected_chips_code'] = sim_codes_padded
        sim_row['selected_cross'] = cand['cross_idx']
        sim_row['beast_selected'] = cand['do_beast']
        
        if 'window_hand_id' not in sim_row: sim_row['window_hand_id'] = sim_row.get('hand_slots', [])
        if 'window_hand_code' not in sim_row: sim_row['window_hand_code'] = sim_row.get('hand_codes', [])
        if 'window_hand_vis' not in sim_row:
             sim_row['window_hand_vis'] = [1.0 if x > 0 else 0.0 for x in hand_slots]

        sim_rows.append(sim_row)
        
    return candidates, sim_rows


@app.route("/api/solve_strategy", methods=["POST"])
def api_solve_strategy():
    if not _PLANNING_CRITIC:
        return jsonify({"error": "Critic not loaded"}), 500

    try:
        game_state = request.json
        
        # 1. Generate Candidates
        candidates, sim_rows = _solve_combinations(game_state)
        
        count = len(candidates)
        if count == 0:
            return jsonify({"results": [], "count": 0})
            
        print(f"[API] Evaluating {count} permutations...")

        # 2. Run Inference
        preds = _PLANNING_CRITIC.infer_rows(sim_rows)
            
        # 3. Combine
        results = []
        rich_hand = game_state.get("rich_hand", [])
        
        for i, cand in enumerate(candidates):
            chips_ui = []
            for idx in cand["chain"]:
                if idx < len(rich_hand) and rich_hand[idx]:
                    chips_ui.append(rich_hand[idx])
            
            results.append({
                "value": float(preds[i]),
                "chain_indices": cand["chain"],
                "chips": chips_ui,
                "cross": cand["cross_idx"],
                "beast": cand["do_beast"]
            })
            
        # 4. Sort Descending
        results.sort(key=lambda x: x["value"], reverse=True)
        
        return jsonify({
            "results": results[:2000],
            "total_evaluated": count
        }) 
        
    except Exception as e:
        print(f"[API] Solve failed: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _decorate_strategy_v2_row(raw_turn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert chip_window_strategy_v2 row into the legacy-ish shape the template expects.
    This lets view_strategy.html render without needing template edits.
    """
    # ---- enforce v2 ----
    if str(raw_turn.get("format", "")) != "chip_window_strategy_v2":
        raise ValueError("not v2")

    # -------------------------------------------------------------------------
    # DISPLAY-ONLY mapping: raw player_game_emotion -> {idx, beast, raw}
    #
    # IMPORTANT:
    # - Dataset semantics stay: selected_cross is the RAW emotion id.
    # - This mapping is ONLY for UI pretty labels.
    # - Built from the same table used in derived_state.py.
    # -------------------------------------------------------------------------
    _CROSS_ROWS = [
        {"idx": 0, "name": "Normal", "normal": 0,  "beast": 11},
        {"idx": 1, "name": "Fire",   "normal": 1,  "beast": 13},
        {"idx": 2, "name": "Elec",   "normal": 2,  "beast": 14},
        {"idx": 3, "name": "Slash",  "normal": 3,  "beast": 15},
        {"idx": 4, "name": "Erase",  "normal": 4,  "beast": 16},
        {"idx": 5, "name": "Charge", "normal": 5,  "beast": 17},
        {"idx": 6, "name": "Aqua",   "normal": 6,  "beast": 18},
        {"idx": 7, "name": "Thawk",  "normal": 7,  "beast": 19},
        {"idx": 8, "name": "Tengu",  "normal": 8,  "beast": 20},
        {"idx": 9, "name": "Grnd",   "normal": 9,  "beast": 21},
        {"idx": 10, "name": "Dust",  "normal": 10, "beast": 22},
    ]

    _EMOTION_TO_STATE: Dict[int, Tuple[int, bool]] = {}
    for row in _CROSS_ROWS:
        idx = int(row["idx"])
        n = int(row["normal"])
        b = int(row["beast"])
        _EMOTION_TO_STATE[n] = (idx, False)
        _EMOTION_TO_STATE[b] = (idx, True)

    def _emotion_state(raw_emotion: Any) -> Optional[Dict[str, Any]]:
        try:
            r = int(raw_emotion)
        except Exception:
            return None
        if r < 0:
            return None
        if r in _EMOTION_TO_STATE:
            idx, beast = _EMOTION_TO_STATE[r]
            return {"idx": int(idx), "beast": bool(beast), "raw": int(r)}
        # Unknown emotion id: keep stable
        return {"idx": int(r), "beast": False, "raw": int(r)}

    # ---- canonical v2 fields ----
    replay = str(raw_turn.get("replay", ""))
    open_idx = _as_int(raw_turn.get("open_idx", 0), 0)

    # Hand arrays (always 10)
    hand_ids = [int(x) for x in _pad_list(raw_turn.get("window_hand_id", []), 10, 0)]
    hand_codes = [int(x) for x in _pad_list(raw_turn.get("window_hand_code", []), 10, 0)]
    hand_vis = [float(x) for x in _pad_list(raw_turn.get("window_hand_vis", []), 10, 0.0)]
    visible_count = _as_int(raw_turn.get("chip_visible_count", 5), 5)

    # Selected arrays (5)
    sel_ids = [int(x) for x in _pad_list(raw_turn.get("selected_chips_id", []), 5, 0)]
    sel_codes = [int(x) for x in _pad_list(raw_turn.get("selected_chips_code", []), 5, 0)]
    sel_mask = [bool(x) for x in _pad_list(raw_turn.get("selected_chips_mask", []), 5, False)]

    # Held-before arrays (5)
    held_before_ids = [int(x) for x in _pad_list(raw_turn.get("held_before_id", []), 5, 0)]
    held_before_codes = [int(x) for x in _pad_list(raw_turn.get("held_before_code", []), 5, 0)]
    held_before_mask = [bool(x) for x in _pad_list(raw_turn.get("held_before_mask", []), 5, False)]

    # Held-after arrays (5) (enter battle)
    held_after_ids = [int(x) for x in _pad_list(raw_turn.get("held_after_id", []), 5, 0)]
    held_after_codes = [int(x) for x in _pad_list(raw_turn.get("held_after_code", []), 5, 0)]
    held_after_mask = [bool(x) for x in _pad_list(raw_turn.get("held_after_mask", []), 5, False)]

    # ---- build rich hand (always length 10) ----
    rich_hand: List[Optional[Dict[str, Any]]] = []
    for i in range(10):
        if float(hand_vis[i]) <= 0.0:
            rich_hand.append(None)
            continue
        cid = int(hand_ids[i] or 0)
        ccode = int(hand_codes[i] or 0)
        if cid <= 0 or cid in (255, 65535):
            rich_hand.append(None)
            continue
        rich_hand.append(resolve_chip_info(cid, ccode))

    # ---- selected_indices for highlighting ----
    selected_indices = _compute_selected_indices_v2(
        hand_ids=hand_ids,
        hand_codes=hand_codes,
        hand_vis=hand_vis,
        sel_ids=sel_ids,
        sel_codes=sel_codes,
        sel_mask=sel_mask,
    )

    # ---- rich selected / held ----
    rich_selected = _rich_from_id_code_arrays(sel_ids, sel_codes, sel_mask, n=5)

    hb_list = raw_turn.get("held_before", None)
    if isinstance(hb_list, list) and len(hb_list) > 0:
        rich_held_before = _rich_from_chip_dict_list(hb_list)
    else:
        rich_held_before = _rich_from_id_code_arrays(held_before_ids, held_before_codes, held_before_mask, n=5)

    ha_list = raw_turn.get("held_after", None)
    if isinstance(ha_list, list) and len(ha_list) > 0:
        rich_held_after = _rich_from_chip_dict_list(ha_list)
    else:
        rich_held_after = _rich_from_id_code_arrays(held_after_ids, held_after_codes, held_after_mask, n=5)

    # ---- normalize context keys the template uses ----
    raw_turn["replay_file"] = replay
    raw_turn["frame_open"] = open_idx
    raw_turn["frame_start"] = open_idx

    raw_turn["p_hp_start"] = _as_int(raw_turn.get("p_hp_open", 0), 0)
    raw_turn["e_hp_start"] = _as_int(raw_turn.get("e_hp_open", 0), 0)
    raw_turn["player_emotion"] = _as_int(raw_turn.get("player_emotion_open", 0), 0)
    raw_turn["enemy_emotion"] = _as_int(raw_turn.get("enemy_emotion_open", 0), 0)
    raw_turn["beast_mode"] = _as_int(raw_turn.get("beast_mode_open", 0), 0)
    raw_turn["cust_gauge"] = _as_int(raw_turn.get("cust_open", 0), 0)
    raw_turn["player_charge"] = _as_int(raw_turn.get("player_charge_open", 0), 0)
    raw_turn["enemy_charge"] = _as_int(raw_turn.get("enemy_charge_open", 0), 0)

    raw_turn["player_pos"] = raw_turn.get("player_pos_open", [0, 0])
    raw_turn["enemy_pos"] = raw_turn.get("enemy_pos_open", [0, 0])

    raw_turn["grid_state"] = raw_turn.get("grid_tile_open", [0] * 18)
    raw_turn["grid_owner_state"] = raw_turn.get("grid_owner_open", [2] * 18)
    if not isinstance(raw_turn["grid_state"], list):
        raw_turn["grid_state"] = [0] * 18
    if not isinstance(raw_turn["grid_owner_state"], list):
        raw_turn["grid_owner_state"] = [2] * 18
    raw_turn["grid_state"] = _pad_list(raw_turn["grid_state"], 18, 0)
    raw_turn["grid_owner_state"] = _pad_list(raw_turn["grid_owner_state"], 18, 2)

    raw_turn["hand_slots"] = hand_ids
    raw_turn["hand_codes"] = hand_codes
    raw_turn["chip_visible_count"] = visible_count

    raw_turn["rich_hand"] = rich_hand
    raw_turn["selected_indices"] = selected_indices
    raw_turn["rich_selected"] = rich_selected

    raw_turn["rich_held_before"] = rich_held_before
    raw_turn["rich_held_after"] = rich_held_after
    raw_turn["rich_held"] = rich_held_after  # legacy alias

    # ---- build a top-level window_commit (template convenience) ----
    raw_turn["window_commit"] = {
        "happened": True,
        "selected_any": bool(raw_turn.get("selected_any", False)),
        "beast_selected": bool(raw_turn.get("beast_selected", False)),
        "close_chip_select_count": _as_int(raw_turn.get("close_chip_select_count", 0), 0),
        "source": str(raw_turn.get("commit_source", "") or ""),
        "selected_chips": [
            {"id": int(sel_ids[i]), "code": int(sel_codes[i])}
            for i in range(5)
            if bool(sel_mask[i]) and int(sel_ids[i]) > 0
        ],
    }

    # -------------------------------------------------------------------------
    # Selected cross (RAW emotion id) -> pretty UI states
    # -------------------------------------------------------------------------
    sel_raw = raw_turn.get("selected_cross", None)
    if sel_raw is None:
        sel_raw = raw_turn.get("selected_cross_emotion", None)

    pre_raw = raw_turn.get("selected_cross_pre_emotion", None)
    post_raw = raw_turn.get("selected_cross_post_emotion", None)
    commit_raw = raw_turn.get("selected_cross_commit_emotion", None)  # legacy/debug

    # Preserve raw integer (including 0) when present
    try:
        if sel_raw is not None:
            raw_turn["selected_cross"] = int(sel_raw)
            raw_turn["window_commit"]["selected_cross"] = int(sel_raw)
    except Exception:
        pass

    raw_turn["selected_cross_state"] = _emotion_state(sel_raw) if sel_raw is not None else None
    raw_turn["selected_cross_pre_state"] = _emotion_state(pre_raw) if pre_raw is not None else None
    raw_turn["selected_cross_post_state"] = _emotion_state(post_raw) if post_raw is not None else None
    raw_turn["selected_cross_commit_state"] = _emotion_state(commit_raw) if commit_raw is not None else None

    # ---- build a minimal derived object so the template's dp/de/wc works ----
    raw_turn["derived"] = {
        "player": {
            "active_cross": raw_turn.get("active_cross_p_open", None),
            "used_cross_mask": raw_turn.get("used_cross_mask_p_open", []),
            "folder_used_mask": raw_turn.get("folder_used_mask_p_open", []),
            "beast": raw_turn.get("beast_p_open", None),
            "window_commit": dict(raw_turn["window_commit"]),
        },
        "enemy": {
            "active_cross": raw_turn.get("active_cross_e_open", None),
            "used_cross_mask": raw_turn.get("used_cross_mask_e_open", []),
            "folder_used_mask": raw_turn.get("folder_used_mask_e_open", []),
            "beast": raw_turn.get("beast_e_open", None),
        },
    }

    return raw_turn



_PLANNING_CRITIC = None

# Config
_DEFAULT_PLANNING_CKPT = "checkpoints/planning_critic.pt"

def _init_planning_critic():
    global _PLANNING_CRITIC
    try:
        from viewer.critic_planner_infer import PlanningCriticRunner
        
        ckpt = os.environ.get("PLANNING_CRITIC_CKPT", _DEFAULT_PLANNING_CKPT)
        if os.path.exists(ckpt):
            _PLANNING_CRITIC = PlanningCriticRunner(ckpt)
        else:
            print(f"[PlanningCritic] Checkpoint not found at {ckpt}")
    except Exception as e:
        print(f"[PlanningCritic] Failed to load: {e}")

# Call init at startup
_init_planning_critic()


@app.route("/api/predict_strategy", methods=["POST"])
def api_predict_strategy():
    """
    Run the Planning Critic on a hypothetical turn state provided by the UI.
    """
    if not _PLANNING_CRITIC:
        return jsonify({"error": "Critic not loaded", "value": 0.0})

    try:
        # The UI sends a single dictionary representing the modified turn
        # We wrap it in a list because infer_rows expects a batch
        modified_turn = request.json
        
        # The infer_rows method expects raw keys like 'p_hp_open', 'window_hand_id', etc.
        # The JS will ensure these are present and updated based on user selection.
        preds = _PLANNING_CRITIC.infer_rows([modified_turn])
        
        val = preds[0] if preds else 0.0
        return jsonify({"value": val})
        
    except Exception as e:
        print(f"[API] Prediction failed: {e}")
        return jsonify({"error": str(e), "value": 0.0}), 500



FPS_DEFAULT = 60.0
TURN_TAU_DEFAULT_S = 6.0  # must match your dataset builder default (or pass through)

def _turn_time_weight(duration_s: float, tau_s: float) -> float:
    d = max(0.0, float(duration_s))
    tau = max(1e-6, float(tau_s))
    return 1.0 / (1.0 + (d / tau))

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0

def _ensure_turn_time_metrics(row: Dict[str, Any], *, fps: float = FPS_DEFAULT, tau_s: float = TURN_TAU_DEFAULT_S) -> None:
    """
    Ensure duration + per-second + weighted metrics exist on a strategy row.
    This makes older rows display correctly and gives the viewer consistent keys.

    Writes:
      - turn_duration_frames, turn_duration_s
      - damage_*_per_s, net_yield_per_s
      - turn_time_weight, net_yield_weighted (and dealt/taken weighted)
    """
    # Derive duration from indices if missing
    if "turn_duration_frames" not in row or "turn_duration_s" not in row:
        commit_idx = int(row.get("commit_idx", 0) or 0)
        next_open_idx = int(row.get("next_open_idx", commit_idx) or commit_idx)

        dur_frames = max(0, next_open_idx - commit_idx)
        row["turn_duration_frames"] = int(dur_frames)

        fps_f = max(1e-6, float(fps))
        row["turn_duration_s"] = float(dur_frames) / fps_f

    dur_s = float(row.get("turn_duration_s", 0.0) or 0.0)

    dealt = float(row.get("damage_dealt", 0) or 0)
    taken = float(row.get("damage_taken", 0) or 0)

    # net_yield fallback (raw)
    if "net_yield" not in row:
        row["net_yield"] = int(dealt - taken)

    net = float(row.get("net_yield", 0) or 0)

    # per-second rates
    if "damage_dealt_per_s" not in row:
        row["damage_dealt_per_s"] = _safe_div(dealt, dur_s)
    if "damage_taken_per_s" not in row:
        row["damage_taken_per_s"] = _safe_div(taken, dur_s)
    if "net_yield_per_s" not in row:
        row["net_yield_per_s"] = _safe_div(net, dur_s)

    # time-discounted weighting
    if "turn_time_weight" not in row:
        row["turn_time_weight"] = float(_turn_time_weight(dur_s, tau_s=float(tau_s)))

    w = float(row.get("turn_time_weight", 0.0) or 0.0)
    if "damage_dealt_weighted" not in row:
        row["damage_dealt_weighted"] = dealt * w
    if "damage_taken_weighted" not in row:
        row["damage_taken_weighted"] = taken * w
    if "net_yield_weighted" not in row:
        row["net_yield_weighted"] = net * w



@app.route("/strategy")
def view_strategy():
    turns: List[Dict[str, Any]] = []

    if not os.path.exists(STRATEGY_DB_PATH):
        return render_template("view_strategy.html", turns=[])

    try:
        with open(STRATEGY_DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw_turn = json.loads(line)
                except Exception:
                    continue
                if not isinstance(raw_turn, dict):
                    continue

                # v2 only
                if str(raw_turn.get("format", "")) != "chip_window_strategy_v2":
                    continue

                # Decorate first (adds rich_hand, selected_indices, cross states, etc.)
                try:
                    raw_turn = _decorate_strategy_v2_row(raw_turn)
                except Exception:
                    continue

                # Ensure time metrics exist for display + consistency (old rows too)
                _ensure_turn_time_metrics(raw_turn, fps=FPS_DEFAULT, tau_s=TURN_TAU_DEFAULT_S)

                # Grid indices for visualization
                p_pos = raw_turn.get("player_pos") or raw_turn.get("player_pos_open") or [0, 0]
                e_pos = raw_turn.get("enemy_pos") or raw_turn.get("enemy_pos_open") or [0, 0]
                try:
                    raw_turn["p_grid_idx"] = pos_to_grid_idx(float(p_pos[0]), float(p_pos[1]))
                except Exception:
                    raw_turn["p_grid_idx"] = 0
                try:
                    raw_turn["e_grid_idx"] = pos_to_grid_idx(float(e_pos[0]), float(e_pos[1]))
                except Exception:
                    raw_turn["e_grid_idx"] = 0

                turns.append(raw_turn)

        # --- BATCH INFERENCE ---
        if _PLANNING_CRITIC and turns:
            try:
                preds = _PLANNING_CRITIC.infer_rows(turns)
                for i, pred in enumerate(preds):
                    turns[i]["critic_yield"] = pred
            except Exception as e:
                print(f"[PlanningCritic] Inference error: {e}")

    except Exception as e:
        print(f"Error reading strategy DB: {e}")

    return render_template("view_strategy.html", turns=turns[::-1])


import math
import os
import json
import time
import gc
import torch
import orjson
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union
from flask import Flask, jsonify

from derived_state import compute_derived

# -----------------------------------------------------------------------------
# Critic smoothing helpers (masked to match training: only when cust_gauge > 0)
# -----------------------------------------------------------------------------

def _is_finite(x: Any) -> bool:
    try:
        if x is None:
            return False
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False


def _coerce_values_by_frame(
    values_by_frame: Union[List[Any], Dict[Any, Any]],
    n_frames: int,
) -> List[Optional[float]]:
    """
    Normalize CriticRunner output into a dense list of length n_frames containing
    floats or None (no NaN/inf).
    Supports:
      - list/tuple of per-frame values (may include None/NaN)
      - dict mapping frame_idx -> value
    """
    out: List[Optional[float]] = [None] * n_frames

    if isinstance(values_by_frame, dict):
        for k, v in values_by_frame.items():
            try:
                i = int(k)
            except Exception:
                continue
            if 0 <= i < n_frames and _is_finite(v):
                out[i] = float(v)
        return out

    # list-like
    try:
        m = min(len(values_by_frame), n_frames)
        for i in range(m):
            v = values_by_frame[i]
            if _is_finite(v):
                out[i] = float(v)
        return out
    except Exception:
        return out


def _densify_linear_masked(values: List[Optional[float]], mask: List[bool]) -> List[Optional[float]]:
    """
    Linear densify, but ONLY within contiguous True segments of mask.
    Outside mask => forced None.
    Inside mask:
      - interpolate between known points
      - lead/trail filled with nearest known (within that segment)
      - if a segment has no known points, it stays all None
    """
    n = len(values)
    if n == 0:
        return values
    if len(mask) != n:
        raise ValueError(f"mask len {len(mask)} != values len {n}")

    out = list(values)

    # Collect known points only inside mask
    known = [i for i, v in enumerate(out) if mask[i] and v is not None]

    # Ensure masked-out frames are None
    for i in range(n):
        if not mask[i]:
            out[i] = None

    if not known:
        return out

    # Fill each contiguous True segment independently
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue

        j = i
        while j < n and mask[j]:
            j += 1

        seg_known = [k for k in known if i <= k < j]
        if seg_known:
            # leading
            first = seg_known[0]
            for t in range(i, first):
                out[t] = out[first]

            # middle gaps
            for a, b in zip(seg_known, seg_known[1:]):
                va, vb = out[a], out[b]
                if va is None or vb is None:
                    continue
                gap = b - a
                if gap > 1:
                    for t in range(a + 1, b):
                        u = (t - a) / gap
                        out[t] = (1.0 - u) * va + u * vb

            # trailing
            last = seg_known[-1]
            for t in range(last + 1, j):
                out[t] = out[last]
        else:
            # No known values in this segment
            for t in range(i, j):
                out[t] = None

        i = j

    return out


def _ema_smooth_masked(values: List[Optional[float]], mask: List[bool], alpha: float) -> List[Optional[float]]:
    """
    EMA smoothing but ONLY inside mask. Leaving mask hard-resets EMA so we don't
    "bleed" values into cust_gauge==0 regions.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    if len(mask) != len(values):
        raise ValueError("mask length mismatch")

    out: List[Optional[float]] = [None] * len(values)
    ema: Optional[float] = None

    for i, v in enumerate(values):
        if not mask[i]:
            ema = None
            out[i] = None
            continue

        if v is None:
            out[i] = ema if ema is not None else None
            continue

        ema = v if ema is None else (alpha * v + (1.0 - alpha) * ema)
        out[i] = ema

    return out


def _densify_and_smooth_masked(
    sparse: Union[List[Any], Dict[Any, Any]],
    n_frames: int,
    *,
    mask: List[bool],
    ema_alpha: float,
) -> tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Returns:
      (smoothed, densified_pre_ema)

    (1) coerce -> dense (None where missing / NaN)
    (2) linear densify within mask segments
    (3) EMA smooth within mask segments, resetting at mask boundaries
    """
    dense = _coerce_values_by_frame(sparse, n_frames)
    densified = _densify_linear_masked(dense, mask)
    smoothed = _ema_smooth_masked(densified, mask, alpha=ema_alpha)
    return smoothed, densified


# -----------------------------------------------------------------------------
# Critic init (critic_minimal)
# -----------------------------------------------------------------------------

_CRITIC = None
_CRITIC_ERR = None

_DEFAULT_CRITIC_MINIMAL_CKPT = r"C:\Users\leeor\FFCO\ai\tango-ai\checkpoints\critic_minimal\minq_v2_tanh\best.pt"
_DEFAULT_CRITIC_DEVICE = "cuda"
_DEFAULT_BATCH_SEQS = 256

def _init_critic() -> None:
    global _CRITIC, _CRITIC_ERR
    try:
        from viewer.critic_minimal_infer import MinimalCriticRunner

        ckpt = (os.environ.get("CRITIC_MINIMAL_CKPT", "").strip() or _DEFAULT_CRITIC_MINIMAL_CKPT).strip()
        device = (os.environ.get("CRITIC_MINIMAL_DEVICE", "").strip() or _DEFAULT_CRITIC_DEVICE).strip()
        batch_seqs = int(os.environ.get("CRITIC_MINIMAL_BATCH_SEQS", str(_DEFAULT_BATCH_SEQS)).strip() or str(_DEFAULT_BATCH_SEQS))
        use_amp = (os.environ.get("CRITIC_MINIMAL_AMP", "1").strip() != "0")

        if not ckpt:
            print("[CriticMinimal] No checkpoint path configured.")
            return

        print(f"[CriticMinimal] Loading model from: {ckpt} on {device}...")
        _CRITIC = MinimalCriticRunner(ckpt_path=ckpt, device=device, use_amp=use_amp, batch_seqs=batch_seqs)
        _CRITIC_ERR = None
        print("[CriticMinimal] Model loaded successfully.")
    except Exception as e:
        _CRITIC = None
        _CRITIC_ERR = str(e)
        print(f"[CriticMinimal] Failed to load model: {e}")

_init_critic()


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------

INPUT_CACHE = OrderedDict()
MAX_CACHE_SIZE = 5
# --- Critic aux cache for exact probe matching (NOT sent to UI) ---
CRITIC_AUX_CACHE = OrderedDict()  # replay -> {"mask": [...], "densified": [...], "ema_alpha": float, "stride": int, "seq_len": int}


# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (list, tuple)) and v:
            return int(v[0])
        return int(v)
    except Exception:
        return default


def _compute_hp_reward(frames: List[Dict[str, Any]]) -> List[float]:
    """
    Per-frame reward based purely on observed HP deltas:
      r_t = (enemy_hp[t-1] - enemy_hp[t]) - (player_hp[t-1] - player_hp[t])
    i.e. damage dealt minus damage taken.
    """
    n = len(frames)
    if n == 0:
        return []
    r = [0.0] * n
    prev_p = _safe_int(frames[0].get("player_health"), 0)
    prev_e = _safe_int(frames[0].get("enemy_health"), 0)
    for t in range(1, n):
        p = _safe_int(frames[t].get("player_health"), prev_p)
        e = _safe_int(frames[t].get("enemy_health"), prev_e)
        dp = max(0, prev_p - p)  # damage taken
        de = max(0, prev_e - e)  # damage dealt
        r[t] = float(de - dp)
        prev_p, prev_e = p, e
    return r


def _bellman_return(rewards: List[float], gamma: float) -> List[float]:
    """
    Backward discounted return:
      G_t = r_t + gamma * G_{t+1}
    """
    n = len(rewards)
    out = [0.0] * n
    g = 0.0
    for t in range(n - 1, -1, -1):
        g = float(rewards[t]) + float(gamma) * g
        out[t] = g
    return out
def _chip_use_events(chip: List[int]) -> tuple[List[bool], List[int]]:
    """
    Telemetry semantics (your script):
      - chip[t] is on-deck chip (fires on next A)
      - NO_CHIP means none
      - "use" when on-deck chip changes away from a previous non-NO_CHIP chip
    Returns:
      used_mask[t], used_chip_id[t] (prev chip id)
    """
    n = len(chip)
    used = [False] * n
    used_id = [NO_CHIP] * n
    if n <= 1:
        return used, used_id

    for t in range(1, n):
        prev = int(chip[t - 1])
        cur = int(chip[t])
        if prev != NO_CHIP and cur != prev:
            used[t] = True
            used_id[t] = prev
    return used, used_id


def _charge_release_events(charge: List[int], threshold: int) -> List[bool]:
    """
    True at t when charge[t-1] >= threshold and charge[t] == 0.
    """
    n = len(charge)
    out = [False] * n
    if n <= 1:
        return out
    for t in range(1, n):
        if int(charge[t - 1]) >= int(threshold) and int(charge[t]) == 0:
            out[t] = True
    return out
def _spread_add(buf: List[float], start: int, end_inclusive: int, amt: float, mode: str) -> None:
    """
    Add 'amt' across [start..end_inclusive].
    uniform: split evenly
    ramp: more weight near event frame
    """
    n = len(buf)
    a = max(0, min(int(start), n - 1))
    b = max(0, min(int(end_inclusive), n - 1))
    if b < a:
        return

    L = (b - a + 1)
    if L <= 0:
        return

    if mode == "ramp" and L > 1:
        # linear ramp 1..L
        denom = (L * (L + 1)) / 2.0
        for k in range(L):
            w = (k + 1) / denom
            buf[a + k] += amt * w
    else:
        each = amt / float(L)
        for i in range(a, b + 1):
            buf[i] += each
def _compute_commit_shaping(frames: List[Dict[str, Any]], cfg: ShapingCfg) -> List[float]:
    """
    Dense per-frame shaping reward built from commit events.
    We look forward LOOKAHEAD frames to measure:
      - player_damage_taken (from player HP drops)
      - enemy_damage_taken  (from enemy HP drops)
    Then assign shaped reward to the PRE_EVENT_FRAMES leading into the commit frame.
    """
    n = len(frames)
    if n == 0:
        return []

    # --- pull series ---
    p_hp = [_safe_int(f.get("player_health"), 0) for f in frames]
    e_hp = [_safe_int(f.get("enemy_health"), 0) for f in frames]

    p_charge = [_safe_int(f.get("player_charge"), 0) for f in frames]
    e_charge = [_safe_int(f.get("enemy_charge"), 0) for f in frames]

    p_chip = [_safe_int(f.get("player_chip"), NO_CHIP) for f in frames]
    e_chip = [_safe_int(f.get("enemy_chip"), NO_CHIP) for f in frames]

    # per-frame damage (same as your dataset script)
    p_dmg = [0] * n
    e_dmg = [0] * n
    for t in range(1, n):
        p_dmg[t] = max(0, p_hp[t - 1] - p_hp[t])  # damage taken by player
        e_dmg[t] = max(0, e_hp[t - 1] - e_hp[t])  # damage taken by enemy

    # event masks
    p_chip_use, _ = _chip_use_events(p_chip)
    e_chip_use, _ = _chip_use_events(e_chip)
    p_charge_rel = _charge_release_events(p_charge, cfg.charge_threshold)
    e_charge_rel = _charge_release_events(e_charge, cfg.charge_threshold)

    shaped = [0.0] * n

    def sum_window(arr: List[int], t0: int, t1_excl: int) -> int:
        a = max(0, min(int(t0), n))
        b = max(0, min(int(t1_excl), n))
        if b <= a:
            return 0
        return int(sum(arr[a:b]))

    def apply_actor_event(*, actor: str, t: int) -> None:
        # window
        start = max(0, t - cfg.pre_event_frames)
        end = min(n, t + cfg.lookahead_frames)  # exclusive

        dmg_taken = sum_window(p_dmg, t, end)   # player got hit?
        dmg_dealt = sum_window(e_dmg, t, end)   # player dealt dmg?

        if actor == "enemy":
            # enemy committed: reward if you dodged, punish if you got hit
            amt = cfg.enemy_dodge_bonus if dmg_taken == 0 else cfg.enemy_hit_penalty
            _spread_add(shaped, start, t, amt, cfg.spread_mode)

        elif actor == "player":
            # player committed: punish if no damage dealt (your request)
            if dmg_dealt <= 0:
                amt = cfg.player_miss_penalty
            else:
                # dealt damage: reward clean, penalize trades
                if dmg_taken == 0:
                    amt = cfg.player_clean_bonus
                else:
                    amt = cfg.player_trade_penalty
            _spread_add(shaped, start, t, amt, cfg.spread_mode)

    # apply for all commits (chip + charge)
    for t in range(1, n):
        if p_chip_use[t] or p_charge_rel[t]:
            apply_actor_event(actor="player", t=t)
        if e_chip_use[t] or e_charge_rel[t]:
            apply_actor_event(actor="enemy", t=t)

    return shaped

# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
from dataclasses import dataclass
NO_CHIP = 65535

@dataclass(frozen=True)
class ShapingCfg:
    fps: int = 60
    lookahead_frames: int = 4 * 60      # 4 seconds
    pre_event_frames: int = 18
    charge_threshold: int = 2

    # enemy commit shaping
    enemy_dodge_bonus: float = 40.0
    enemy_hit_penalty: float = -40.0

    # player commit shaping
    player_clean_bonus: float = 20.0
    player_trade_penalty: float = -15.0
    player_miss_penalty: float = -25.0

    # how to smear reward over [t-pre .. t]
    spread_mode: str = "uniform"  # "uniform" | "ramp"

@app.route("/inputs/<path:replay_name>")
def serve_inputs(replay_name: str):
    # 1) Memory Cache (Fastest)
    if replay_name in INPUT_CACHE:
        print(f"[Cache] Serving {replay_name} from memory...")
        INPUT_CACHE.move_to_end(replay_name)
        return jsonify(INPUT_CACHE[replay_name])

    print(f"\n[Serve] Loading replay: {replay_name}")
    replay_path = os.path.join(DATASET_DIR, replay_name)
    jsonl_path = os.path.join(replay_path, "actions.jsonl")
    static_path = os.path.join(replay_path, "static_data.json")

    response: Dict[str, Any] = {"frames": [], "static": None, "derived": []}

    # 2) Load frames
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "rb") as f:
                for line in f:
                    if line.strip():
                        try:
                            response["frames"].append(orjson.loads(line))
                        except Exception:
                            continue
        except Exception:
            pass

    print(f"[Serve] Loaded {len(response['frames'])} frames.")


    # --- Bellman HP-return (deterministic baseline) ---
    try:
        gamma = float(os.environ.get("BELL_GAMMA", "0.997").strip() or "0.997")  # ~1s half-life-ish at 60fps
    except Exception:
        gamma = 0.997

    hp_r = _compute_hp_reward(response["frames"])
    hp_G = _bellman_return(hp_r, gamma=gamma)

    # --- Shaping ---
    cfg = ShapingCfg(
        fps=60,
        lookahead_frames=int(os.environ.get("SHAPE_LOOKAHEAD", str(4 * 60))),
        pre_event_frames=int(os.environ.get("SHAPE_PRE", "18")),
        charge_threshold=int(os.environ.get("SHAPE_CHARGE_THR", "2")),
        enemy_dodge_bonus=float(os.environ.get("SHAPE_ENEMY_DODGE_BONUS", "40.0")),
        enemy_hit_penalty=float(os.environ.get("SHAPE_ENEMY_HIT_PENALTY", "-40.0")),
        player_clean_bonus=float(os.environ.get("SHAPE_PLAYER_CLEAN_BONUS", "20.0")),
        player_trade_penalty=float(os.environ.get("SHAPE_PLAYER_TRADE_PENALTY", "-15.0")),
        player_miss_penalty=float(os.environ.get("SHAPE_PLAYER_MISS_PENALTY", "-25.0")),
        spread_mode=str(os.environ.get("SHAPE_SPREAD", "uniform")),
    )

    shape_r = _compute_commit_shaping(response["frames"], cfg)

    # Combined reward (dense)
    combo_r = [float(hr) + float(sr) for hr, sr in zip(hp_r, shape_r)]

    # Bellman on combined reward
    combo_G = _bellman_return(combo_r, gamma=gamma)

    response["bellman"] = {
        "gamma": gamma,
        "values_by_frame": combo_G,
        "reward_by_frame": combo_r,
        "reward_hp_by_frame": hp_r,
        "reward_shape_by_frame": shape_r,
        "shape_cfg": {
            "lookahead_frames": cfg.lookahead_frames,
            "pre_event_frames": cfg.pre_event_frames,
            "charge_threshold": cfg.charge_threshold,
            "enemy_dodge_bonus": cfg.enemy_dodge_bonus,
            "enemy_hit_penalty": cfg.enemy_hit_penalty,
            "player_clean_bonus": cfg.player_clean_bonus,
            "player_trade_penalty": cfg.player_trade_penalty,
            "player_miss_penalty": cfg.player_miss_penalty,
            "spread_mode": cfg.spread_mode,
        },
    }



    # 3) Load static
    if os.path.exists(static_path):
        try:
            with open(static_path, "rb") as f:
                response["static"] = orjson.loads(f.read())
        except Exception:
            pass

    # 4) Derived
    try:
        t0 = time.time()
        derived_data = compute_derived(response["frames"], response["static"])
        print(f"[Serve] Derived state computed in {time.time() - t0:.3f}s")
    except Exception as e:
        derived_data = []
        print(f"⚠️ derived_state compute failed: {e}")

    critic_values: Optional[List[Optional[float]]] = None
    critic_meta: Optional[Dict[str, Any]] = None
    n_frames = len(derived_data)

    # -------------------------------------------------------------------------
    # NEW: Check Disk Cache (Nitrogen Pre-computed)
    # -------------------------------------------------------------------------
    # We look for the .pt file generated by nitrogen/precache.py
    # If found, we use those values instead of running slow inference.
    cache_dir = os.environ.get("NITROGEN_BATTLE_CACHE_DIR", os.path.join("data", "nitrogen_battle_cache_bellman"))
    cache_pt_path = os.path.join(cache_dir, f"{replay_name}.pt")
    cache_hit = False

    if os.path.exists(cache_pt_path):
        try:
            print(f"[Serve] Found disk cache: {cache_pt_path}")
            payload = torch.load(cache_pt_path, map_location="cpu")

            # values are per cached sample
            values_t = payload.get("values", None)
            if values_t is None:
                raise KeyError("cache missing key: values")

            # NEW format uses action_indices (preferred)
            # OLD format used original_indices
            idx_t = payload.get("action_indices", None)
            idx_key = "action_indices"
            if idx_t is None:
                idx_t = payload.get("original_indices", None)
                idx_key = "original_indices"

            if idx_t is None:
                raise KeyError("cache missing key: action_indices (or original_indices)")

            vals = values_t.detach().to(torch.float32).cpu().numpy()
            idxs = idx_t.detach().to(torch.long).cpu().numpy()

            if len(vals) != len(idxs):
                raise ValueError(f"cache mismatch: len(values)={len(vals)} != len({idx_key})={len(idxs)}")

            # Map cached (battle-only) values back onto full replay timeline (action/derived index space)
            critic_values = [None] * n_frames
            inserted = 0
            for idx, val in zip(idxs, vals):
                j = int(idx)
                if 0 <= j < n_frames:
                    critic_values[j] = float(val)
                    inserted += 1

            critic_meta = {
                "source": "nitrogen_cache",
                "path": cache_pt_path,
                "index_key": idx_key,
                "cached_samples": int(len(idxs)),
                "inserted": int(inserted),
                "n_frames": int(n_frames),
            }

            cache_hit = True
            print(f"[Serve] Loaded {len(idxs)} critic values from disk cache ({idx_key}); inserted={inserted}.")

        except Exception as e:
            print(f"⚠️ Failed to load disk cache: {e}")
            cache_hit = False


    # 5) Critic inference (critic_minimal)
    if _CRITIC is not None and response["frames"] and derived_data:
        try:
            # Canonical mask: match training notion of "battle" (cust_gauge > 0)
            cust_mask = [(int((f or {}).get("cust_gauge") or 0) > 0) for f in response["frames"]]
            if len(cust_mask) != n_frames:
                cust_mask = cust_mask[:n_frames] + [False] * max(0, n_frames - len(cust_mask))

            hold = int(os.environ.get("CRITIC_MINIMAL_HOLD", "4").strip() or "4")
            seq_len = int(os.environ.get("CRITIC_MINIMAL_SEQ_LEN", "192").strip() or "192")
            start_stride = int(os.environ.get("CRITIC_MINIMAL_START_STRIDE", "1").strip() or "1")
            ema_alpha = float(os.environ.get("CRITIC_MINIMAL_EMA_ALPHA", "0.25").strip() or "0.25")

            # If 1, require cust>0 starts (recommended for matching your cache/training intent)
            require_cust_gt0 = (os.environ.get("CRITIC_MINIMAL_REQUIRE_CUST_GT0", "1").strip() != "0")

            print(f"[Serve] Inferencing CriticMinimal (hold={hold}, seq_len={seq_len}, start_stride={start_stride}, ema_alpha={ema_alpha})...")

            t0 = time.time()
            res = _CRITIC.infer_from_frames(
                frames=response["frames"],
                hold=hold,
                seq_len=seq_len,
                start_stride=start_stride,
                require_cust_gt0=require_cust_gt0,
            )
            dt = time.time() - t0

            # res.values_by_frame is sparse at raw indices (0,hold,2hold,...)
            # Densify + smooth ONLY inside cust_mask segments (same pattern as before)
            critic_values, densified_pre_ema = _densify_and_smooth_masked(
                res.values_by_frame,
                n_frames,
                mask=cust_mask,
                ema_alpha=float(ema_alpha),
            )

            # IMPORTANT: store aux needed by probe (exact EMA-at-idx override)
            CRITIC_AUX_CACHE[replay_name] = {
                "mask": cust_mask,
                "densified": densified_pre_ema,   # pre-EMA so probe can override a raw point
                "ema_alpha": float(ema_alpha),
                "hold": int(hold),
                "seq_len": int(seq_len),
                "start_stride": int(start_stride),
                "require_cust_gt0": bool(require_cust_gt0),
            }

            critic_meta = dict(res.meta or {})
            critic_meta.update({
                "source": "critic_minimal",
                "took_s": float(dt),
                "ema_alpha": float(ema_alpha),
                "hold": int(hold),
                "seq_len": int(seq_len),
                "start_stride": int(start_stride),
                "require_cust_gt0": bool(require_cust_gt0),
            })


            print(f"[Serve] CriticMinimal inference complete in {dt:.3f}s.")

        except Exception as e:
            critic_values = None
            critic_meta = {"error": str(e)}
            print(f"[Serve] CriticMinimal inference failed: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # 6) Attach critic_v
    if critic_values is not None:
        m = min(len(critic_values), len(derived_data))
        for i in range(m):
            v = critic_values[i]
            # Only attach if finite and present
            if v is not None and _is_finite(v):
                derived_data[i]["critic_v"] = float(v)
            else:
                derived_data[i].pop("critic_v", None)

    response["derived"] = derived_data
    response["critic"] = {
        "enabled": _CRITIC is not None,
        "error": _CRITIC_ERR,
        "meta": critic_meta,
    }

    # 7) Cache store/evict
    INPUT_CACHE[replay_name] = response
    if len(INPUT_CACHE) > MAX_CACHE_SIZE:
        evicted = INPUT_CACHE.popitem(last=False)
        # Clean aux cache if it exists for this item
        CRITIC_AUX_CACHE.pop(evicted[0], None)
        print(f"[Cache] Evicted {evicted[0]} to free up memory.")

    return jsonify(response)

def _as_button_value(x) -> float:
    """Normalize incoming 0/1-ish values to 0.0/1.0 floats."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 1.0 if v > 0.5 else 0.0


def _apply_button_overrides(frame: dict, overrides: dict) -> dict:
    """
    Return a shallow-copied frame with controller button fields overridden.

    Preserves original shape: if the existing value is a list (e.g. [0.0]),
    we keep it as a single-element list. Otherwise we store a scalar.
    """
    out = dict(frame)
    for k, raw in (overrides or {}).items():
        v = _as_button_value(raw)
        if k in out and isinstance(out[k], list):
            out[k] = [v]
        else:
            out[k] = v
    return out


def _load_replay_frames_for_probe(replay_name: str) -> list:
    """
    Reuse the SAME source of truth as your /inputs/<replay> route.
    If you already have a function that loads frames, call it here.

    EXPECTATION: returns list[dict] where each dict is a frame record.
    """
    # ---- OPTION A (recommended): call your existing loader used by /inputs/<replay> ----
    # return load_inputs_for_replay(replay_name)["frames"]

    # ---- OPTION B: if /inputs/<replay> already reads from a JSON file in replay dir ----
    # Replace this with your actual path logic.
    import json
    from pathlib import Path

    replay_dir = Path(DATASET_DIR) / replay_name
    frames_path = replay_dir / "actions.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"actions file not found: {frames_path}")

    frames: list[dict] = []
    with frames_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ValueError(f"invalid JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object on line {line_no}, got {type(obj).__name__}")
            frames.append(obj)

    if not frames:
        raise ValueError(f"no frames parsed from: {frames_path}")

    return frames

def _ema_value_at_idx(
    *,
    densified: List[Optional[float]],
    mask: List[bool],
    alpha: float,
    idx: int,
    override_value: Optional[float] = None,
) -> Optional[float]:
    """
    Compute the exact EMA-smoothed value at `idx` using the SAME rules as /inputs:
      - EMA resets whenever mask is False
      - uses densified values inside mask segments
    If override_value is provided, it is used at idx instead of densified[idx].
    """
    n = len(densified)
    if n == 0 or idx < 0 or idx >= n:
        return None
    if len(mask) != n:
        return None
    if not mask[idx]:
        return None

    # Find segment start (mask transitions from False->True)
    s = idx
    while s > 0 and mask[s - 1]:
        s -= 1

    ema: Optional[float] = None
    out: Optional[float] = None

    for i in range(s, idx + 1):
        v = override_value if (override_value is not None and i == idx) else densified[i]

        if v is None:
            out = ema
            continue

        ema = v if ema is None else (alpha * v + (1.0 - alpha) * ema)
        out = ema

    return out


def _critic_probe_last_value(
    runner,
    *,
    frames_window: list[dict],
    static: dict | None,
    seq_len: int,
) -> float:
    """
    Probe path that matches /inputs/<replay>: infer_from_derived().
    We pass a *pre-sampled* window (length == seq_len), so we use stride=1.
    Returns the value for the last frame in frames_window.
    """
    if not hasattr(runner, "infer_from_derived"):
        raise RuntimeError("CriticRunner missing infer_from_derived(); probe cannot run.")

    # Derived must align 1:1 with frames_window
    derived_window = compute_derived(frames_window, static)

    res = runner.infer_from_derived(
        frames=frames_window,
        static=static,
        derived=derived_window,
        stride=1,                 # window already sampled
        seq_len=int(seq_len),
        require_cust_gt0=None,    # keep checkpoint-trained behavior
    )

    values = getattr(res, "values_by_frame", None)
    if values is None:
        raise RuntimeError("infer_from_derived returned no values_by_frame")

    last_i = len(frames_window) - 1

    v = None
    if isinstance(values, dict):
        v = values.get(last_i)
    elif isinstance(values, (list, tuple)) and 0 <= last_i < len(values):
        v = values[last_i]

    if v is None or not _is_finite(v):
        raise RuntimeError("probe produced no finite value for last frame")

    return float(v)


def _bitmask_to_buttons(mask: int, keys: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, k in enumerate(keys):
        out[k] = 1 if ((mask >> i) & 1) else 0
    return out


def _build_minimal_probe_window(
    *,
    frames: List[Dict[str, Any]],
    end_frame: int,
    hold: int,
    seq_len: int,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Builds the same kind of window as api_critic_probe, ending exactly at end_frame.
    Window length in frames = (seq_len - 1) * hold + 1

    Returns: (window_frames, window_abs_start, window_abs_end)
    """
    n = len(frames)
    end_frame = max(0, min(int(end_frame), n - 1))

    win_len = (int(seq_len) - 1) * int(hold) + 1
    start_abs = int(end_frame) - (win_len - 1)

    window: List[Dict[str, Any]] = []
    for t in range(win_len):
        src_i = start_abs + t
        if src_i < 0:
            src_i = 0
        if src_i >= n:
            src_i = n - 1
        row = frames[src_i]
        window.append(row if isinstance(row, dict) else {})

    return window, int(start_abs), int(end_frame)


def _minimal_raw_value_for_overrides(
    runner,
    *,
    base_window: List[Dict[str, Any]],
    overrides: Dict[str, int],
    hold: int,
    seq_len: int,
    start_stride: int,
    require_cust_gt0: bool,
) -> float:
    """
    Runs critic_minimal on base_window, overriding ONLY the endpoint buttons.
    Returns the raw model output at the endpoint (last element).
    """
    if not base_window:
        raise ValueError("empty window")

    # shallow copy window list; copy endpoint dict only (avoid mutating base)
    window = list(base_window)
    window[-1] = _apply_button_overrides(window[-1], overrides)

    res = runner.infer_from_frames(
        frames=window,
        hold=int(hold),
        seq_len=int(seq_len),
        start_stride=int(start_stride),
        require_cust_gt0=bool(require_cust_gt0),
    )

    raw_dense = _coerce_values_by_frame(getattr(res, "values_by_frame", []), len(window))
    raw_last = raw_dense[-1] if raw_dense else None
    if raw_last is None or not _is_finite(raw_last):
        raise RuntimeError("no finite raw value at endpoint")
    return float(raw_last)


def _ema_prev_at_idx(
    *,
    densified: List[Optional[float]],
    mask: List[bool],
    alpha: float,
    idx: int,
) -> Optional[float]:
    """
    EMA at idx-1 inside the current mask segment.
    Used to cheaply recompute EMA at idx for many candidate raw values:
      ema_idx = raw if ema_prev is None else alpha*raw + (1-alpha)*ema_prev
    """
    if idx <= 0:
        return None
    return _ema_value_at_idx(densified=densified, mask=mask, alpha=alpha, idx=idx - 1, override_value=None)

@app.route("/api/critic_find_best", methods=["POST"])
def api_critic_find_best():
    """
    Exhaustive search over all possible PROBE_KEYS button states (pruned).
    Returns the best controller mask for the probed frame (snapped to hold grid).

    Body:
      {
        replay: str,
        frame_idx: int,
        keys?: [str],          # optional; default is standard keys (no START/BACK)
        top_k?: int            # optional; default 5
      }
    """
    global _CRITIC, _CRITIC_ERR

    # ---------------------------------------------------------------------
    # Globals / consts (NO env vars)
    # ---------------------------------------------------------------------
    CRITIC_FIND_BEST_BATCH_SEQS = 16384  # push GPU; reduce if OOM
    CRITIC_FIND_BEST_MAX_STATES = None   # None => no cap

    # Keys we will never search over (ignored entirely)
    _IGNORED_KEYS = {"START", "BACK"}  # START + SELECT(BACK)

    # DPAD mutual exclusion constraints
    _DPAD_VERT = ("DPAD_UP", "DPAD_DOWN")
    _DPAD_HORZ = ("DPAD_LEFT", "DPAD_RIGHT")

    # Shoulder constraint: only allowed when cust_gauge==100
    _SHOULDERS = ("LEFT_SHOULDER", "RIGHT_SHOULDER")

    print("[API] /api/critic_find_best hit")

    payload = request.get_json(silent=True) or {}
    replay = payload.get("replay", "")
    frame_idx = payload.get("frame_idx", None)

    # Default probe set (no START/BACK)
    default_keys = [
        "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
        "EAST", "SOUTH",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
    ]

    keys_in = payload.get("keys", None)
    if not isinstance(keys_in, list) or not keys_in:
        keys_in = default_keys

    # normalize + filter ignored
    keys = []
    for k in keys_in:
        ks = str(k)
        if ks in _IGNORED_KEYS:
            continue
        keys.append(ks)

    # de-dupe preserving order
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    try:
        top_k = int(payload.get("top_k", 5))
    except Exception:
        top_k = 5
    top_k = max(1, min(top_k, 20))

    if not isinstance(replay, str) or not replay.strip():
        return jsonify({"ok": False, "note": "bad request: missing replay"}), 400
    try:
        frame_idx = int(frame_idx)
    except Exception:
        return jsonify({"ok": False, "note": "bad request: frame_idx must be int"}), 400

    # Ensure critic is initialized
    if _CRITIC is None and _CRITIC_ERR is None:
        _init_critic()
    if _CRITIC is None:
        err = _CRITIC_ERR or "critic not available"
        return jsonify({"ok": False, "note": f"critic init error: {err}"}), 500

    # Load frames
    try:
        frames = _load_replay_frames_for_probe(replay)
    except Exception as e:
        return jsonify({"ok": False, "note": f"load frames error: {e}"}), 500

    n = len(frames)
    if n <= 0:
        return jsonify({"ok": False, "note": "no frames"}), 400
    frame_idx = max(0, min(int(frame_idx), n - 1))

    # Match /inputs config (keep your current sources of truth here)
    # NOTE: You asked "no env var"; if you want these const too, say so and I’ll inline them.
    try:
        hold = int(os.environ.get("CRITIC_MINIMAL_HOLD", "4").strip() or "4")
    except Exception:
        hold = 4
    try:
        seq_len = int(os.environ.get("CRITIC_MINIMAL_SEQ_LEN", "192").strip() or "192")
    except Exception:
        seq_len = 192
    try:
        start_stride = int(os.environ.get("CRITIC_MINIMAL_START_STRIDE", "1").strip() or "1")
    except Exception:
        start_stride = 1
    try:
        ema_alpha = float(os.environ.get("CRITIC_MINIMAL_EMA_ALPHA", "0.25").strip() or "0.25")
    except Exception:
        ema_alpha = 0.25

    require_cust_gt0 = (os.environ.get("CRITIC_MINIMAL_REQUIRE_CUST_GT0", "1").strip() != "0")

    if hold <= 0:
        return jsonify({"ok": False, "note": f"bad config: hold<=0 ({hold})"}), 500
    if seq_len <= 1:
        return jsonify({"ok": False, "note": f"bad config: seq_len<=1 ({seq_len})"}), 500

    # Ensure aux exists (same as probe)
    aux = CRITIC_AUX_CACHE.get(replay)
    if aux is None:
        cust_mask = [(int((f or {}).get("cust_gauge") or 0) > 0) for f in frames]
        if len(cust_mask) != n:
            cust_mask = cust_mask[:n] + [False] * max(0, n - len(cust_mask))

        try:
            res_full = _CRITIC.infer_from_frames(
                frames=frames,
                hold=int(hold),
                seq_len=int(seq_len),
                start_stride=int(start_stride),
                require_cust_gt0=bool(require_cust_gt0),
            )
            dense = _coerce_values_by_frame(res_full.values_by_frame, n)
            densified = _densify_linear_masked(dense, cust_mask)
            aux = {
                "mask": cust_mask,
                "densified": densified,
                "ema_alpha": float(ema_alpha),
                "hold": int(hold),
                "seq_len": int(seq_len),
                "start_stride": int(start_stride),
                "require_cust_gt0": bool(require_cust_gt0),
            }
            CRITIC_AUX_CACHE[replay] = aux
        except Exception as e:
            return jsonify({"ok": False, "note": f"aux build failed: {e}"}), 500

    mask: List[bool] = aux["mask"]
    densified_full: List[Optional[float]] = aux["densified"]
    alpha_full: float = float(aux["ema_alpha"])

    # Snap to hold grid
    effective_frame = int(frame_idx) - (int(frame_idx) % int(hold))
    effective_frame = max(0, min(effective_frame, n - 1))

    # If we’re outside battle mask, searching is meaningless
    if not (0 <= effective_frame < len(mask)) or not mask[effective_frame]:
        return jsonify(
            {
                "ok": True,
                "best": None,
                "top": [],
                "meta": {
                    "frame_requested": int(frame_idx),
                    "frame_effective": int(effective_frame),
                    "hold": int(hold),
                    "seq_len": int(seq_len),
                    "ema_alpha": float(alpha_full),
                    "cust_mask_at_effective": False,
                    "keys": list(keys),
                },
                "note": "effective frame is outside battle mask (cust_gauge==0); no search performed",
            }
        )

    # Shoulder gating
    cust_here = int((frames[effective_frame] or {}).get("cust_gauge") or 0)
    allow_shoulders = (cust_here == 100)

    # Precompute EMA at idx-1 once
    ema_prev = _ema_prev_at_idx(
        densified=densified_full, mask=mask, alpha=alpha_full, idx=int(effective_frame)
    )

    # Build base window once
    base_window, win_start, win_end = _build_minimal_probe_window(
        frames=frames,
        end_frame=int(effective_frame),
        hold=int(hold),
        seq_len=int(seq_len),
    )

    # ---------------------------------------------------------------------
    # Pruned enumeration
    # ---------------------------------------------------------------------
    t0 = time.time()
    K = len(keys)
    if K <= 0 or K > 24:
        return jsonify({"ok": False, "note": f"refusing search: keys size {K} (expected 1..24)"}), 400

    bit = {k: i for i, k in enumerate(keys)}

    def _set_bit(m: int, k: str, v: int) -> int:
        i = bit.get(k, None)
        if i is None:
            return m
        if v:
            return m | (1 << i)
        return m & ~(1 << i)

    # Build DPAD combo masks (mutual exclusion)
    dpad_masks: List[int] = [0]

    has_up = _DPAD_VERT[0] in bit
    has_dn = _DPAD_VERT[1] in bit
    has_lt = _DPAD_HORZ[0] in bit
    has_rt = _DPAD_HORZ[1] in bit

    if has_up or has_dn or has_lt or has_rt:
        dpad_masks = []
        vert_states = [
            {},  # neither
            {_DPAD_VERT[0]: 1},  # up
            {_DPAD_VERT[1]: 1},  # down
        ]
        horz_states = [
            {},  # neither
            {_DPAD_HORZ[0]: 1},  # left
            {_DPAD_HORZ[1]: 1},  # right
        ]
        for vs in vert_states:
            for hs in horz_states:
                mm = 0
                for k, v in vs.items():
                    mm = _set_bit(mm, k, v)
                for k, v in hs.items():
                    mm = _set_bit(mm, k, v)
                dpad_masks.append(mm)

    # Remaining keys (independent toggles) except shoulders (gated)
    free_keys: List[str] = []
    for k in keys:
        if k in _IGNORED_KEYS:
            continue
        if k in _DPAD_VERT or k in _DPAD_HORZ:
            continue
        if k in _SHOULDERS:
            continue
        free_keys.append(k)

    # Shoulder keys included only if allowed; otherwise forced 0 (so excluded from enumeration)
    shoulder_keys: List[str] = []
    if allow_shoulders:
        for k in _SHOULDERS:
            if k in bit:
                shoulder_keys.append(k)

    # Build all valid bitmasks (usually tiny: 9 * 2^(free + shoulder_allowed))
    masks: List[int] = []
    base_list = dpad_masks

    # enumerate free keys
    for base in base_list:
        m0s = [base]
        for k in free_keys:
            i = bit[k]
            nexts = []
            for mm in m0s:
                nexts.append(mm)              # k=0
                nexts.append(mm | (1 << i))   # k=1
            m0s = nexts
        # enumerate shoulders if allowed
        if shoulder_keys:
            for k in shoulder_keys:
                i = bit[k]
                nexts = []
                for mm in m0s:
                    nexts.append(mm)
                    nexts.append(mm | (1 << i))
                m0s = nexts
        masks.extend(m0s)

    # Optional cap
    if CRITIC_FIND_BEST_MAX_STATES is not None:
        masks = masks[: int(CRITIC_FIND_BEST_MAX_STATES)]

    total = 1 << K
    best = {"mask": 0, "raw": -1e30, "v": -1e30}
    top: List[Dict[str, Any]] = []

    def push_top(item: Dict[str, Any]) -> None:
        top.append(item)
        top.sort(key=lambda x: float(x["v"]), reverse=True)
        del top[top_k:]

    def score_from_raw(raw_val: float) -> float:
        if ema_prev is None:
            return float(raw_val)
        return float(alpha_full) * float(raw_val) + (1.0 - float(alpha_full)) * float(ema_prev)

    # Score in big GPU batches if supported
    batch_seqs = int(CRITIC_FIND_BEST_BATCH_SEQS)
    batch_seqs = max(1, min(batch_seqs, 32768))

    if hasattr(_CRITIC, "infer_last_raw_batch"):
        for off in range(0, len(masks), batch_seqs):
            chunk_masks = masks[off : off + batch_seqs]
            overrides_list = [_bitmask_to_buttons(int(m), keys) for m in chunk_masks]

            # If shoulders are NOT allowed, forcibly clear them in overrides (belt+suspenders).
            if not allow_shoulders:
                for ov in overrides_list:
                    for sk in _SHOULDERS:
                        if sk in ov:
                            ov[sk] = 0

            try:
                try:
                    raws = _CRITIC.infer_last_raw_batch(
                        frames_window=base_window,
                        overrides_list=overrides_list,
                        hold=int(hold),
                        seq_len=int(seq_len),
                        start_stride=int(start_stride),
                        require_cust_gt0=bool(require_cust_gt0),
                        batch_seqs=int(batch_seqs),
                    )
                except TypeError:
                    # older signature
                    raws = _CRITIC.infer_last_raw_batch(
                        frames_window=base_window,
                        overrides_list=overrides_list,
                        hold=int(hold),
                        seq_len=int(seq_len),
                        start_stride=int(start_stride),
                        require_cust_gt0=bool(require_cust_gt0),
                    )
            except Exception as e:
                return jsonify({"ok": False, "note": f"batch infer failed: {e}"}), 500

            for m, raw in zip(chunk_masks, raws):
                if raw is None or not _is_finite(raw):
                    continue
                rawf = float(raw)
                vf = float(score_from_raw(rawf))
                if vf > float(best["v"]):
                    best = {"mask": int(m), "raw": rawf, "v": vf}
                push_top({"mask": int(m), "raw": rawf, "v": vf})
    else:
        # fallback (slow)
        for m in masks:
            overrides = _bitmask_to_buttons(int(m), keys)
            if not allow_shoulders:
                for sk in _SHOULDERS:
                    if sk in overrides:
                        overrides[sk] = 0

            try:
                raw = _minimal_raw_value_for_overrides(
                    _CRITIC,
                    base_window=base_window,
                    overrides=overrides,
                    hold=int(hold),
                    seq_len=int(seq_len),
                    start_stride=int(start_stride),
                    require_cust_gt0=bool(require_cust_gt0),
                )
            except Exception:
                continue

            vf = float(score_from_raw(float(raw)))
            if vf > float(best["v"]):
                best = {"mask": int(m), "raw": float(raw), "v": vf}
            push_top({"mask": int(m), "raw": float(raw), "v": vf})

    dt = time.time() - t0

    best_buttons = _bitmask_to_buttons(int(best["mask"]), keys) if top else None
    if best_buttons is not None and not allow_shoulders:
        for sk in _SHOULDERS:
            if sk in best_buttons:
                best_buttons[sk] = 0

    return jsonify(
        {
            "ok": True,
            "best": {
                "buttons": best_buttons,
                "mask": int(best["mask"]),
                "critic_v": float(best["v"]),
                "critic_v_raw": float(best["raw"]),
            } if best_buttons is not None else None,
            "top": [
                {
                    "mask": int(it["mask"]),
                    "buttons": _bitmask_to_buttons(int(it["mask"]), keys),
                    "critic_v": float(it["v"]),
                    "critic_v_raw": float(it["raw"]),
                }
                for it in top
            ],
            "meta": {
                "frame_requested": int(frame_idx),
                "frame_effective": int(effective_frame),
                "hold": int(hold),
                "seq_len": int(seq_len),
                "start_stride": int(start_stride),
                "ema_alpha": float(alpha_full),
                "keys": list(keys),
                "ignored_keys": sorted(list(_IGNORED_KEYS)),
                "cust_gauge_at_effective": int(cust_here),
                "allow_shoulders": bool(allow_shoulders),
                "states_evaluated": int(len(masks)),
                "states_total_naive": int(total),
                "window_abs_start": int(win_start),
                "window_abs_end": int(win_end),
                "batch_seqs": int(batch_seqs),
                "took_s": float(dt),
            },
            "note": "" if int(effective_frame) == int(frame_idx) else "snapped to hold grid",
        }
    )


@app.route("/api/critic_probe", methods=["POST"])
def api_critic_probe():
    """
    Probe CriticMinimal V(s) at a specific frame with overridden controller button states.

    IMPORTANT:
      - critic_minimal produces sparse raw values on a HOLD grid (0, hold, 2*hold, ...)
      - for exact, deterministic matching, we snap to the hold grid.

    Body: { replay: str, frame_idx: int, buttons: {KEY:0/1,...} }

    Response:
      {
        critic_v: float|None,            # EMA-smoothed value at effective_frame WITH override
        critic_v_cached: float|None,     # EMA-smoothed baseline at effective_frame (no override)
        critic_v_raw: float|None,        # raw model value at endpoint in the probe window
        meta: {...},
        note: str
      }
    """
    global _CRITIC, _CRITIC_ERR

    payload = request.get_json(silent=True) or {}
    replay = payload.get("replay", "")
    frame_idx = payload.get("frame_idx", None)
    buttons = payload.get("buttons", {}) or {}

    if not isinstance(replay, str) or not replay.strip():
        return jsonify({"critic_v": None, "note": "bad request: missing replay"}), 400
    if not isinstance(buttons, dict):
        return jsonify({"critic_v": None, "note": "bad request: buttons must be object/dict"}), 400
    try:
        frame_idx = int(frame_idx)
    except Exception:
        return jsonify({"critic_v": None, "note": "bad request: frame_idx must be int"}), 400

    # Ensure critic is initialized
    if _CRITIC is None and _CRITIC_ERR is None:
        _init_critic()
    if _CRITIC is None:
        err = _CRITIC_ERR or "critic not available"
        return jsonify({"critic_v": None, "note": f"critic init error: {err}"}), 500

    # Load frames
    try:
        frames = _load_replay_frames_for_probe(replay)
    except Exception as e:
        return jsonify({"critic_v": None, "note": f"load frames error: {e}"}), 500

    n = len(frames)
    if n <= 0:
        return jsonify({"critic_v": None, "note": "no frames"}), 400

    frame_idx = max(0, min(int(frame_idx), n - 1))

    # Match /inputs config (critic_minimal)
    try:
        hold = int(os.environ.get("CRITIC_MINIMAL_HOLD", "4").strip() or "4")
    except Exception:
        hold = 4
    try:
        seq_len = int(os.environ.get("CRITIC_MINIMAL_SEQ_LEN", "192").strip() or "192")
    except Exception:
        seq_len = 192
    try:
        start_stride = int(os.environ.get("CRITIC_MINIMAL_START_STRIDE", "1").strip() or "1")
    except Exception:
        start_stride = 1
    try:
        ema_alpha = float(os.environ.get("CRITIC_MINIMAL_EMA_ALPHA", "0.25").strip() or "0.25")
    except Exception:
        ema_alpha = 0.25

    require_cust_gt0 = (os.environ.get("CRITIC_MINIMAL_REQUIRE_CUST_GT0", "1").strip() != "0")

    if hold <= 0:
        return jsonify({"critic_v": None, "note": f"bad request: hold<=0 ({hold})"}), 400
    if seq_len <= 1:
        return jsonify({"critic_v": None, "note": f"bad request: seq_len<=1 ({seq_len})"}), 400

    # Ensure aux exists for this replay (build on-demand if user probed before /inputs loaded)
    aux = CRITIC_AUX_CACHE.get(replay)
    if aux is None:
        cust_mask = [(int((f or {}).get("cust_gauge") or 0) > 0) for f in frames]
        if len(cust_mask) != n:
            cust_mask = cust_mask[:n] + [False] * max(0, n - len(cust_mask))

        try:
            res_full = _CRITIC.infer_from_frames(
                frames=frames,
                hold=int(hold),
                seq_len=int(seq_len),
                start_stride=int(start_stride),
                require_cust_gt0=bool(require_cust_gt0),
            )
            dense = _coerce_values_by_frame(res_full.values_by_frame, n)
            densified = _densify_linear_masked(dense, cust_mask)

            aux = {
                "mask": cust_mask,
                "densified": densified,
                "ema_alpha": float(ema_alpha),
                "hold": int(hold),
                "seq_len": int(seq_len),
                "start_stride": int(start_stride),
                "require_cust_gt0": bool(require_cust_gt0),
            }
            CRITIC_AUX_CACHE[replay] = aux
        except Exception as e:
            return jsonify({"critic_v": None, "note": f"probe aux build failed: {e}"}), 500

    mask: List[bool] = aux["mask"]
    densified_full: List[Optional[float]] = aux["densified"]
    alpha_full: float = float(aux["ema_alpha"])

    # Snap to hold grid for deterministic "raw point" probing
    effective_frame = int(frame_idx) - (int(frame_idx) % int(hold))
    effective_frame = max(0, min(effective_frame, n - 1))

    # Build a window that ends exactly on a hold-aligned index so the last point has a raw value.
    # Window length in frames = (seq_len - 1) * hold + 1
    win_len = (int(seq_len) - 1) * int(hold) + 1
    start_abs = int(effective_frame) - (win_len - 1)

    window: list[dict] = []
    for t in range(win_len):
        src_i = start_abs + t
        if src_i < 0:
            src_i = 0
        if src_i >= n:
            src_i = n - 1
        row = frames[src_i]
        window.append(row if isinstance(row, dict) else {})

    # Apply button overrides ONLY to the endpoint frame
    window[-1] = _apply_button_overrides(window[-1], buttons)

    # Run critic_minimal on the window
    try:
        res = _CRITIC.infer_from_frames(
            frames=window,
            hold=int(hold),
            seq_len=int(seq_len),
            start_stride=int(start_stride),
            require_cust_gt0=bool(require_cust_gt0),
        )

        raw_dense = _coerce_values_by_frame(getattr(res, "values_by_frame", []), len(window))
        raw_last = raw_dense[-1] if raw_dense else None
        override_raw = float(raw_last) if (raw_last is not None and _is_finite(raw_last)) else None
        if override_raw is None:
            return jsonify({"critic_v": None, "note": "probe produced no finite raw value"}), 500

        # Baseline (no override) EMA at effective_frame
        baseline = _ema_value_at_idx(
            densified=densified_full,
            mask=mask,
            alpha=alpha_full,
            idx=int(effective_frame),
            override_value=None,
        )

        # Override EMA at effective_frame using the probed raw endpoint value
        probed = _ema_value_at_idx(
            densified=densified_full,
            mask=mask,
            alpha=alpha_full,
            idx=int(effective_frame),
            override_value=float(override_raw),
        )

        note = ""
        if int(effective_frame) != int(frame_idx):
            note = "probe snapped to hold grid for exact match"

        return jsonify(
            {
                "critic_v": float(probed) if (probed is not None and _is_finite(probed)) else None,
                "critic_v_cached": float(baseline) if (baseline is not None and _is_finite(baseline)) else None,
                "critic_v_raw": float(override_raw),
                "meta": {
                    "frame_requested": int(frame_idx),
                    "frame_effective": int(effective_frame),
                    "hold": int(hold),
                    "seq_len": int(seq_len),
                    "start_stride": int(start_stride),
                    "ema_alpha": float(alpha_full),
                    "cust_mask_at_effective": bool(mask[int(effective_frame)]) if 0 <= int(effective_frame) < len(mask) else False,
                    "window_abs_start": int(start_abs),
                    "window_abs_end": int(effective_frame),
                    "window_len": int(len(window)),
                },
                "note": note,
            }
        )

    except Exception as e:
        return jsonify({"critic_v": None, "note": f"probe error: {e}"}), 500




@app.route("/api/cache_meta/<path:filename>")
def cache_meta(filename):
    req_type = request.args.get("type")
    path, _ = find_cache_info(filename, hint_label=req_type)

    if not path:
        return jsonify({"error": "File not found"}), 404

    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        return jsonify({"count": int(data["frames"].shape[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache_frame/<path:filename>/<int:idx>")
def cache_frame(filename, idx):
    req_type = request.args.get("type")
    apply_mask = request.args.get("mask") == "1"

    path, cache_type = find_cache_info(filename, hint_label=req_type)
    if not path:
        return jsonify({"error": "File not found"}), 404

    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")

        actions_tensor = data["actions"]
        n_frames = int(data["frames"].shape[0])
        idx = _clamp_index(idx, n_frames)

        is_baked_3d = (actions_tensor.ndim == 3)

        # 1) Image
        frame_uint8 = data["frames"][idx]
        if apply_mask:
            mask = get_mask_tensor()
            if mask is not None:
                mask_3ch = mask.unsqueeze(0).expand_as(frame_uint8)
                frame_uint8 = frame_uint8.masked_fill(~mask_3ch, 0)

        img_np = frame_uint8.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 2) Input window (vision horizon)
        model_key = cache_type if cache_type in ["Plan", "Battle"] else "Battle"
        cfg = engine.configs.get(model_key, engine.configs["Battle"])
        V = int(cfg["vision_horizon"])

        indices = [max(0, idx - (V - 1) + i) for i in range(V)]
        seq_uint8 = data["frames"][indices]

        if apply_mask:
            mask = get_mask_tensor()
            if mask is not None:
                mask_seq = mask.unsqueeze(0).unsqueeze(0).expand_as(seq_uint8)
                seq_uint8 = seq_uint8.masked_fill(~mask_seq, 0)

        seq_float = seq_uint8.float().div(255.0).mul(2.0).sub(1.0)

        # 3) Ground truth sequence (next 18)
        gt_seq_display: List[Dict[str, Any]] = []
        T = 18
        if is_baked_3d:
            baked_seq = actions_tensor[idx]
            limit = min(T, baked_seq.shape[0])
            for t in range(limit):
                gt_seq_display.append(_cache_vec_to_gt(baked_seq[t].tolist()))
        else:
            for t in range(T):
                target_idx = min(idx + t, actions_tensor.shape[0] - 1)
                gt_seq_display.append(_cache_vec_to_gt(actions_tensor[target_idx].tolist()))

        # 4) Inference
        pred_seq_display = None
        pred_meta: Dict[str, Any] = {}
        pred_vecs = engine.infer(seq_float, model_key=model_key)

        if pred_vecs:
            pred_seq_display = [_policy_vec_to_display(v, old_layout=bool(cfg["old_layout"])) for v in pred_vecs]
            pred_meta = {"model_used": model_key, "horizon": len(pred_vecs)}
        else:
            pred_meta = {"error": f"Model {model_key} not loaded"}

        return jsonify(
            {
                "image": "data:image/png;base64," + b64_img,
                "ground_truth_seq": gt_seq_display,
                "ground_truth": gt_seq_display[0] if gt_seq_display else {},
                "prediction_seq": pred_seq_display,
                "prediction": pred_seq_display[0] if pred_seq_display else {},
                "pred_meta": pred_meta,
                "idx": idx,
                "is_baked": is_baked_3d,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/inspect/<path:filename>")
def inspect_cache(filename):
    return render_template("inspect_cache.html", filename=filename)


@app.route("/rl")
def view_rl_inspector():
    stats = {
        "total": len(RL_EVENTS_CACHE),
        "good": len([x for x in RL_EVENTS_CACHE if x.get("outcome") == "good"]),
        "bad": len([x for x in RL_EVENTS_CACHE if x.get("outcome") == "bad"]),
    }

    # RL inspector MUST use the same token list the base dataset uses.
    rl_ui_buttons = list(GBA_UI_BUTTONS)
    return render_template("rl_inspector.html", stats=stats, ui_buttons=rl_ui_buttons)


# -----------------------------------------------------------------------------
# RL API
# -----------------------------------------------------------------------------
@app.route("/api/rl_list")
def api_rl_list():
    """
    Returns a filtered list of RL events.
    IMPORTANT: idx is the index into RL_EVENTS_CACHE (stable for /api/rl_detail/<idx>).
    """
    actor = request.args.get("actor", "player")
    kind = request.args.get("kind", "chip")
    outcome = request.args.get("outcome", "all")
    label = request.args.get("label", "")

    out: List[Dict[str, Any]] = []
    for idx, e in enumerate(RL_EVENTS_CACHE):
        if actor in ("player", "enemy") and e.get("actor") != actor:
            continue
        if kind in ("chip", "charge") and e.get("kind") != kind:
            continue
        if outcome in ("good", "bad") and e.get("outcome") != outcome:
            continue
        if label and e.get("label") != label:
            continue

        lbl = str(e.get("label", ""))
        chip_id = _parse_chip_id_from_label(lbl)
        chip_name, chip_image_url = (None, None)
        if chip_id is not None:
            chip_name, chip_image_url = _chip_id_to_name_image(chip_id)

        out.append(
            {
                "idx": idx,
                "replay": e.get("replay", ""),
                "weight": float(e.get("weight", 1.0)),
                "start_frame": int(e.get("start_frame", 0)),
                "event_frame": int(e.get("event_frame", 0)),
                "end_frame": int(e.get("end_frame", 0)),
                "actor": e.get("actor", ""),
                "kind": e.get("kind", ""),
                "outcome": e.get("outcome", ""),
                "label": lbl,
                "display_label": _display_label(lbl),
                "chip_id": chip_id,
                "chip_name": chip_name,
                "chip_image_url": chip_image_url,
            }
        )

    # keep UI responsive
    return jsonify(out[:3000])


@app.route("/api/rl_groups")
def api_rl_groups():
    actor = request.args.get("actor", "player")
    kind = request.args.get("kind", "chip")

    data = _filter_events(actor=actor, kind=kind, outcome="all", label="")

    groups: Dict[str, Dict[str, Any]] = {}
    for e in data:
        lbl = str(e.get("label", ""))
        g = groups.get(lbl)
        if g is None:
            chip_id = _parse_chip_id_from_label(lbl)
            chip_name, chip_image_url = (None, None)
            if chip_id is not None:
                chip_name, chip_image_url = _chip_id_to_name_image(chip_id)

            g = {
                "label": lbl,
                "display_label": _display_label(lbl),
                "chip_id": chip_id,
                "chip_name": chip_name,
                "chip_image_url": chip_image_url,
                "good": 0,
                "bad": 0,
                "count": 0,
            }
            groups[lbl] = g

        g["count"] += 1
        if e.get("outcome") == "good":
            g["good"] += 1
        else:
            g["bad"] += 1

    out = list(groups.values())
    out.sort(key=lambda x: (x["count"], x["good"]), reverse=True)
    return jsonify(out[:500])


@app.route("/api/rl_detail/<int:idx>")
def api_rl_detail(idx: int):
    """
    Detail view for ONE RL event (by RL_EVENTS_CACHE index).

    Contract:
      - video loops from start_frame..end_frame (inclusive)
      - IGNORE event_frame for loop math (event_frame is metadata only)
      - UI time->frame mapping uses the REAL video fps (ffprobe)
      - window_actions covers [start_frame .. end_frame + 18 - 1]
    """
    if idx < 0 or idx >= len(RL_EVENTS_CACHE):
        return jsonify({"error": "Index out of bounds"}), 404

    evt = RL_EVENTS_CACHE[idx]
    replay_name = str(evt.get("replay", ""))

    start_frame, event_frame, end_frame = _event_frame_window(evt)

    # video info (authoritative fps for timeline mapping)
    video_path = _find_video_path_for_replay(replay_name)
    vinfo = _ffprobe_video_info(video_path) if video_path else {"fps": 60.0, "nb_frames": None, "duration": None}
    video_fps = float(vinfo.get("fps") or 60.0)

    # clamp frames against actual video length if known
    nb_frames = vinfo.get("nb_frames")
    if isinstance(nb_frames, int) and nb_frames > 0:
        start_frame = min(start_frame, nb_frames - 1)
        event_frame = min(event_frame, nb_frames - 1)
        end_frame = min(end_frame, nb_frames - 1)
        if end_frame < start_frame:
            end_frame = start_frame

    # loop timestamps (end is exclusive)
    start_ts = float(start_frame) / video_fps
    end_ts_excl = float(end_frame + 1) / video_fps

    # also clamp timestamps to duration if known
    dur = vinfo.get("duration")
    if isinstance(dur, (int, float)) and float(dur) > 0:
        d = float(dur)
        start_ts = max(0.0, min(start_ts, d - 0.001))
        end_ts_excl = max(start_ts + 0.001, min(end_ts_excl, d))

    # image at start_frame
    image_b64, img_src_type = _load_image_for_event(replay_name, start_frame=start_frame)

    # first 18 frames starting at start_frame
    actions, act_src_type = _load_actions_sequence_for_event(
        replay_name,
        event_frame=start_frame,
        horizon=18,
    )

    # window actions: [start .. end + 18 - 1]
    window_actions, win_src_type = _load_actions_window_for_replay(
        replay_name,
        start_frame=start_frame,
        end_frame=end_frame,
        horizon=18,
    )

    lbl = str(evt.get("label", ""))
    chip_id = _parse_chip_id_from_label(lbl)
    chip_name, chip_image_url = (None, None)
    if chip_id is not None:
        chip_name, chip_image_url = _chip_id_to_name_image(chip_id)

    return jsonify(
        {
            "replay": replay_name,
            "actor": evt.get("actor", ""),
            "kind": evt.get("kind", ""),
            "outcome": evt.get("outcome", ""),
            "reason": evt.get("reason", ""),
            "label": lbl,
            "display_label": _display_label(lbl),
            "weight": float(evt.get("weight", 1.0)),
            "start_frame": int(start_frame),
            "event_frame": int(event_frame),
            "end_frame": int(end_frame),
            # VIDEO LOOP CONTROL (authoritative: derived from real video fps)
            "start_timestamp": start_ts,
            "end_timestamp": end_ts_excl,
            "fps": video_fps,
            # Optional debug fields (handy if anything is still off)
            "video_nb_frames": nb_frames,
            "video_duration": dur,
            "video_r_frame_rate": vinfo.get("r_frame_rate"),
            "video_avg_frame_rate": vinfo.get("avg_frame_rate"),
            # chip decoration
            "chip_id": chip_id,
            "chip_name": chip_name,
            "chip_image_url": chip_image_url,
            # damage stats carried through from rl_events.jsonl
            "damage_dealt": int(evt.get("damage_dealt", 0)),
            "damage_taken": int(evt.get("damage_taken", 0)),
            # window actions stream
            "window_base_frame": int(start_frame),
            "window_actions": window_actions,
            # visuals
            "image": image_b64,
            # initial render compatibility
            "actions": actions,
            # preserve your source typing
            "source_type": (win_src_type if win_src_type != "none" else act_src_type)
            if act_src_type != "none"
            else img_src_type,
        }
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Viewer running at http://127.0.0.1:5011")
    app.run(debug=True, port=5011, host="0.0.0.0")
