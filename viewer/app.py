# viewer/app.py
from __future__ import annotations

import os
import sys
import json
import math
import io
import base64
import subprocess
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
CKPT_ROOT = ""  # e.g. os.path.join(parent_dir, "checkpoints")
PLANNING_CKPT_DIR = ""  # e.g. os.path.join(CKPT_ROOT, "planning")
BATTLE_CKPT_DIR = ""  # e.g. os.path.join(CKPT_ROOT, "battle")

# Strategy & RL Paths
STRATEGY_DB_PATH = os.path.join(parent_dir, "data/chipwindows/strategy.jsonl")
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
def pos_to_grid_idx(x: float, y: float) -> int:
    # Rough approximation of BN6 grid
    col = max(0, min(5, int((x - 40) / 40)))
    row = max(0, min(2, int((y - 75) / 30)))
    return row * 6 + col


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
    return render_template("view.html", replay_name=replay_name)


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


@app.route("/strategy")
def view_strategy():
    turns: List[Dict[str, Any]] = []

    if os.path.exists(STRATEGY_DB_PATH):
        try:
            with open(STRATEGY_DB_PATH, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    raw_turn = json.loads(line)

                    # Net yield fallback
                    if "net_yield" not in raw_turn:
                        d = raw_turn.get("damage_dealt", 0)
                        t = raw_turn.get("damage_taken", 0)
                        raw_turn["net_yield"] = d - t

                    # Hand processing
                    processed_hand: List[Optional[Dict[str, Any]]] = []
                    hand_slots = raw_turn.get("hand_slots", [])
                    hand_codes = raw_turn.get("hand_codes", [])
                    visible_count = raw_turn.get("chip_visible_count", 5)

                    for i, chip_id in enumerate(hand_slots):
                        if i >= visible_count:
                            processed_hand.append(None)
                        else:
                            chip_code = hand_codes[i]
                            info = resolve_chip_info(chip_id, chip_code)
                            processed_hand.append(info)
                    raw_turn["rich_hand"] = processed_hand

                    # Grid indices
                    p_pos = raw_turn.get("player_pos", [0, 0])
                    e_pos = raw_turn.get("enemy_pos", [0, 0])
                    raw_turn["p_grid_idx"] = pos_to_grid_idx(p_pos[0], p_pos[1])
                    raw_turn["e_grid_idx"] = pos_to_grid_idx(e_pos[0], e_pos[1])

                    # AI inference
                    if strategy_model and meta_proc:
                        try:
                            with torch.no_grad():
                                h_ids_list = list(hand_slots)
                                h_codes_list = list(hand_codes)
                                for i in range(visible_count, 10):
                                    if i < len(h_ids_list):
                                        h_ids_list[i] = 255
                                    if i < len(h_codes_list):
                                        h_codes_list[i] = 0

                                # If hand arrays are shorter than 10, pad
                                while len(h_ids_list) < 10:
                                    h_ids_list.append(255)
                                while len(h_codes_list) < 10:
                                    h_codes_list.append(0)

                                h_ids = torch.tensor([h_ids_list], dtype=torch.long).to(DEVICE)
                                san_codes = [min(int(c), 31) for c in h_codes_list]
                                h_codes = torch.tensor([san_codes], dtype=torch.long).to(DEVICE)

                                meta_list = [meta_proc.get_meta(cid) for cid in h_ids_list]
                                h_meta = torch.stack(meta_list).unsqueeze(0).to(DEVICE)

                                # Extended Context (36 dims)
                                p_hp = raw_turn.get("p_hp_start", 1000) / 1000.0
                                e_hp = raw_turn.get("e_hp_start", 2000) / 2000.0

                                used_vec = [0.0] * 6
                                c_list = raw_turn.get("player_used_crosses", raw_turn.get("used_crosses", []))
                                for c in c_list:
                                    if 1 <= c <= 6:
                                        used_vec[c - 1] = 1.0

                                cur_cross_id = raw_turn.get("current_cross", 0)
                                cur_cross_vec = [0.0] * 6
                                if 0 <= cur_cross_id <= 5:
                                    cur_cross_vec[cur_cross_id] = 1.0
                                else:
                                    cur_cross_vec[0] = 1.0

                                beast_val = 1.0 if raw_turn.get("beast_mode", 0) > 0 else 0.0

                                grid_raw = raw_turn.get("grid_state", [0] * 18)
                                if len(grid_raw) < 18:
                                    grid_raw = list(grid_raw) + [0] * (18 - len(grid_raw))
                                grid_vec = [float(x) / 10.0 for x in grid_raw[:18]]

                                is_full_sync = 1.0 if raw_turn.get("player_emotion", 0) == 1 else 0.0

                                owner_sum = sum(raw_turn.get("grid_owner_state", [0] * 18))
                                area_adv = ((18 - owner_sum) - 9.0) / 9.0

                                e_col = raw_turn["e_grid_idx"] % 6
                                e_col_norm = e_col / 5.0

                                ctx_list = (
                                    [p_hp, e_hp]
                                    + used_vec
                                    + cur_cross_vec
                                    + [beast_val]
                                    + grid_vec
                                    + [is_full_sync, area_adv, e_col_norm]
                                )
                                full_ctx = torch.tensor([ctx_list], dtype=torch.float32).to(DEVICE)

                                pos = torch.arange(10, device=DEVICE).unsqueeze(0)
                                src = (
                                    strategy_model.chip_embedding(h_ids)
                                    + strategy_model.code_embedding(h_codes)
                                    + strategy_model.meta_proj(h_meta)
                                    + strategy_model.pos_embedding(pos)
                                )
                                src = src + strategy_model.context_proj(full_ctx).unsqueeze(1)

                                memory = strategy_model.encoder(src)
                                cross_logits = strategy_model.cross_head(memory.mean(dim=1))
                                _, seq_tokens = strategy_model.inference(memory, cross_logits, max_chip_index=visible_count)

                                raw_turn["ai_indices"] = [t for t in seq_tokens[0].tolist() if t < 10 or t == 11]
                                raw_turn["ai_cross"] = torch.argmax(cross_logits, dim=1).item()
                        except Exception as e:
                            print(f"Inference error: {e}")

                    turns.append(raw_turn)
        except Exception as e:
            print(f"Error reading strategy DB: {e}")

    return render_template("view_strategy.html", turns=turns[::-1])

from derived_state import compute_derived
import time

_CRITIC = None
_CRITIC_ERR = None

# ---------------------------------------------------------------------------
# Critic config
# ---------------------------------------------------------------------------

_DEFAULT_CRITIC_CKPT = "C:\\Users\\leeor\\FFCO\\ai\\tango-ai\\checkpoints\\critic_rl\\tdlam_overfit_bs32_lr1e3_wd0\\last.pt"
# _DEFAULT_CRITIC_CKPT = "C:\\Users\\leeor\\FFCO\\ai\\tango-ai\\checkpoints\\critic_rl\\tdlam_v1\\last.pt"
_DEFAULT_CRITIC_DEVICE = "cuda"

# Increase batch size significantly to saturate GPU
_DEFAULT_BATCH_SEQS = 2048  


def _init_critic():
    global _CRITIC, _CRITIC_ERR
    try:
        from critic_infer import CriticRunner

        ckpt = (os.environ.get("CRITIC_RL_CKPT", "").strip() or _DEFAULT_CRITIC_CKPT).strip()
        device = (os.environ.get("CRITIC_RL_DEVICE", "").strip() or _DEFAULT_CRITIC_DEVICE).strip()

        if not ckpt:
            print("[Critic] No checkpoint path configured.")
            return

        print(f"[Critic] Loading model from: {ckpt} on {device}...")
        # Use a large batch size for inference speed
        _CRITIC = CriticRunner(ckpt_path=ckpt, device=device, use_amp=True, batch_seqs=_DEFAULT_BATCH_SEQS)
        _CRITIC_ERR = None
        print("[Critic] Model loaded successfully.")
    except Exception as e:
        _CRITIC = None
        _CRITIC_ERR = str(e)
        print(f"[Critic] Failed to load model: {e}")


_init_critic()

@app.route("/inputs/<path:replay_name>")
def serve_inputs(replay_name):
    print(f"\n[Serve] Loading replay: {replay_name}")
    replay_path = os.path.join(DATASET_DIR, replay_name)
    jsonl_path = os.path.join(replay_path, "actions.jsonl")
    static_path = os.path.join(replay_path, "static_data.json")
    response: Dict[str, Any] = {"frames": [], "static": None, "derived": []}

    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            response["frames"].append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            pass
    
    print(f"[Serve] Loaded {len(response['frames'])} frames.")

    if os.path.exists(static_path):
        try:
            with open(static_path, "r") as f:
                response["static"] = json.load(f)
        except Exception:
            pass

    # 1) Compute Derived State
    try:
        t0 = time.time()
        derived_data = compute_derived(response["frames"], response["static"])
        print(f"[Serve] Derived state computed in {time.time()-t0:.3f}s")
    except Exception as e:
        derived_data = []
        print(f"⚠️ derived_state compute failed: {e}")

    critic_values = None
    critic_meta = None

    # 2) Critic Inference
    if _CRITIC is not None and derived_data:
        try:
            trained_seq_len = getattr(_CRITIC, "trained_seq_len", 16)
            
            # FORCE STRIDE 1 to get a value for every single frame
            stride = 1 

            print(f"[Serve] Inferencing Critic (stride={stride}, seq_len={trained_seq_len})...")
            
            t0 = time.time()
            res = _CRITIC.infer_from_derived(
                frames=response["frames"],
                static=response["static"],
                derived=derived_data,
                stride=stride,
                seq_len=trained_seq_len,
                require_cust_gt0=False 
            )
            dt = time.time() - t0

            critic_values = res.values_by_frame
            critic_meta = res.meta
            
            print(f"[Serve] Inference complete in {dt:.3f}s. Coverage: {res.meta.get('coverage_frames')} frames.")

        except Exception as e:
            critic_values = None
            critic_meta = {"error": str(e)}
            print(f"[Serve] Critic inference failed: {e}")
            import traceback
            traceback.print_exc()

    # 3) Attach
    if critic_values:
        if len(critic_values) == len(derived_data):
            for i, val in enumerate(critic_values):
                derived_data[i]["critic_v"] = val
        else:
            print(f"⚠️ Mismatch: derived len {len(derived_data)} vs critic len {len(critic_values)}")

    response["derived"] = derived_data
    response["critic"] = {
        "enabled": _CRITIC is not None,
        "error": _CRITIC_ERR,
        "meta": critic_meta,
    }

    return jsonify(response)

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
