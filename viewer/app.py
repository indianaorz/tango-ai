# viewer/app.py
import os
import sys
import json
import math
import torch
import io
import base64
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify
from typing import Any, Optional, Dict, List, Tuple

# --- PATH HACK: Look 1 folder up for ng_policy + action_schema ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # ../
sys.path.append(parent_dir)
# -------------------------------------------------

# --- IMPORT POLICY ---
try:
    from ng_policy import NgNitroGenPolicy, load_ng_checkpoint
except ImportError as e:
    print(f"⚠️ Warning: ng_policy.py not found in {parent_dir}. Inference disabled.")
    print(f"Error details: {e}")
    NgNitroGenPolicy = None
    load_ng_checkpoint = None

# --- IMPORT ACTION SCHEMA (single source of truth) ---
from action_schema import BUTTON_TOKENS, GBA_UI_BUTTONS

app = Flask(__name__)

# --- CONFIG ---
DATASET_DIR = os.path.join(parent_dir, "data/dataset")
CACHE_DIR = os.path.join(parent_dir, "data/dataset_cached")
# CHECKPOINT_PATH = os.path.join(parent_dir, "weights/ng.pt")
CHECKPOINT_PATH = os.path.join(parent_dir, "checkpoints/step_3000.pt")
# CHECKPOINT_PATH = os.path.join(parent_dir, "checkpoints_overfit/overfit_20230929001213-ummm-bn6-vs-DthKrdMnSP-round1-p1_idx3206_step1000.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UI buttons: ONLY the GBA-relevant set we want to display
UI_BUTTONS = list(GBA_UI_BUTTONS)

ACTION_OFFSET = int(os.getenv("ACTION_OFFSET", "0"))  # keep in sync with training


def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _get_frame_window_uint8(frames_uint8: torch.Tensor, idx: int, V: int) -> torch.Tensor:
    n = int(frames_uint8.shape[0])
    if n <= 0:
        raise ValueError("Empty frames tensor")

    idx = max(0, min(idx, n - 1))
    start = idx - (V - 1)

    out = []
    for t in range(V):
        src_i = start + t
        if src_i < 0:
            src_i = 0
        elif src_i >= n:
            src_i = n - 1
        out.append(frames_uint8[src_i])
    return torch.stack(out, dim=0)  # [V,3,H,W]


def _clamp_index(i: int, n: int) -> int:
    return max(0, min(int(i), int(n) - 1))


def _vec_cache_to_gt_dict(v_cache: List[float]) -> Dict[str, float]:
    """
    Cache-space layout (your precache_dataset.py):
      [AXIS_LEFTX, AXIS_LEFTY, AXIS_RIGHTX, AXIS_RIGHTY, ...buttons in BUTTON_TOKENS...]
    """
    d: Dict[str, float] = {
        "AXIS_LEFTX": float(v_cache[0]),
        "AXIS_LEFTY": float(v_cache[1]),
        "AXIS_RIGHTX": float(v_cache[2]),
        "AXIS_RIGHTY": float(v_cache[3]),
    }
    btn = v_cache[4:4 + len(BUTTON_TOKENS)]
    for name, value in zip(BUTTON_TOKENS, btn):
        d[name] = float(value)
    return d


def _split_policy_action_layout(
    v25: List[float],
    *,
    old_layout: bool,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Policy/tokenizer-space layout (per NitrogenTokenizer.unpack_actions):

    If old_layout=False (your current run):
      buttons = actions[..., :-4]    # first 21
      j_left  = actions[..., -4:-2]  # next 2
      j_right = actions[..., -2:]    # last 2

    If old_layout=True:
      j_left  = actions[..., :2]
      j_right = actions[..., 2:4]
      buttons = actions[..., 4:]
    """
    if len(v25) != 25:
        raise ValueError(f"Expected action dim 25, got {len(v25)}")

    if old_layout:
        jl = [float(v25[0]), float(v25[1])]
        jr = [float(v25[2]), float(v25[3])]
        buttons = [float(x) for x in v25[4:]]
    else:
        buttons = [float(x) for x in v25[:-4]]
        jl = [float(v25[-4]), float(v25[-3])]
        jr = [float(v25[-2]), float(v25[-1])]

    if len(buttons) != len(BUTTON_TOKENS):
        raise ValueError(f"Expected {len(BUTTON_TOKENS)} buttons, got {len(buttons)}")

    return buttons, jl, jr


def _detect_logits_like(values: List[float]) -> bool:
    """
    Heuristic:
      - probabilities should mostly live in [0,1]
      - logits often spill outside that
    """
    if not values:
        return False
    mn = min(values)
    mx = max(values)
    return (mn < -0.05) or (mx > 1.05)


def _detect_sticks_01_like(stick2: List[float]) -> bool:
    """
    If sticks are in training-space, they are typically ~[0,1] centered at 0.5.
    If they are already [-1,1], center is 0.0.
    """
    mn = min(stick2)
    mx = max(stick2)
    return (mn >= -0.05) and (mx <= 1.05)


def _policy_vec_to_display(
    v25: List[float],
    *,
    old_layout: bool,
) -> Dict[str, Any]:
    """
    Returns a viewer-friendly dict:

    - axes are reported in [-1,1] as AXIS_* (derived from j_left/j_right)
    - buttons_raw: raw model outputs for buttons (either logits or probs)
    - buttons_prob: probabilities (always 0..1)
    - meta: whether buttons were treated as logits
    """
    buttons_raw, jl_raw, jr_raw = _split_policy_action_layout(v25, old_layout=old_layout)

    # sticks: convert to [-1,1] if they look like 0..1
    jl01 = _detect_sticks_01_like(jl_raw)
    jr01 = _detect_sticks_01_like(jr_raw)

    if jl01:
        jl_axis = [x * 2.0 - 1.0 for x in jl_raw]
    else:
        jl_axis = jl_raw[:]

    if jr01:
        jr_axis = [x * 2.0 - 1.0 for x in jr_raw]
    else:
        jr_axis = jr_raw[:]

    # buttons: convert to probabilities if they look like logits
    buttons_are_logits = _detect_logits_like(buttons_raw)
    if buttons_are_logits:
        buttons_prob = [_sigmoid(x) for x in buttons_raw]
    else:
        # already probs (or close enough)
        buttons_prob = [max(0.0, min(1.0, float(x))) for x in buttons_raw]

    out: Dict[str, Any] = {
        "axes": {
            "AXIS_LEFTX": float(jl_axis[0]),
            "AXIS_LEFTY": float(jl_axis[1]),
            "AXIS_RIGHTX": float(jr_axis[0]),
            "AXIS_RIGHTY": float(jr_axis[1]),
        },
        "sticks_raw": {
            "j_left": [float(jl_raw[0]), float(jl_raw[1])],
            "j_right": [float(jr_raw[0]), float(jr_raw[1])],
            "sticks_are_01": bool(jl01 and jr01),
        },
        "buttons_raw": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_raw)},
        "buttons_prob": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_prob)},
        "meta": {
            "buttons_are_logits": bool(buttons_are_logits),
            "layout": "old_layout" if old_layout else "new_layout_buttons_first",
        },
    }
    return out


class ModelEngine:
    def __init__(self):
        self.policy = None
        self.loaded = False

        self.action_horizon = 18
        self.vision_horizon = 1
        self.action_offset = int(os.getenv("ACTION_OFFSET", "0"))

        # Canonical order (must match training+precache)
        self.button_tokens = BUTTON_TOKENS[:]

        # Tokenizer layout details (filled after load if possible)
        self.tokenizer_old_layout = False

        if NgNitroGenPolicy and load_ng_checkpoint and os.path.exists(CHECKPOINT_PATH):
            try:
                print(f"🧠 Loading model from {CHECKPOINT_PATH}...")
                loaded = load_ng_checkpoint(CHECKPOINT_PATH, device=torch.device(DEVICE))
                self.policy = NgNitroGenPolicy(loaded).to(DEVICE).eval()
                self.loaded = True

                tok = getattr(self.policy, "tokenizer", None)
                if tok is not None:
                    ah = getattr(tok, "action_horizon", None)
                    vh = getattr(tok, "vision_horizon", None)
                    old_layout = getattr(tok, "old_layout", None)

                    if isinstance(ah, int) and ah > 0:
                        self.action_horizon = ah
                    if isinstance(vh, int) and vh > 0:
                        self.vision_horizon = vh
                    if isinstance(old_layout, bool):
                        self.tokenizer_old_layout = old_layout

                print(
                    "✅ Model loaded.\n"
                    f"  action_horizon={self.action_horizon}\n"
                    f"  vision_horizon={self.vision_horizon}\n"
                    f"  action_offset={self.action_offset}\n"
                    f"  tokenizer_old_layout={self.tokenizer_old_layout}\n"
                    f"  button_tokens={self.button_tokens}"
                )

            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.loaded = False

    def infer_seq(self, frames_tensor: torch.Tensor) -> Optional[List[List[float]]]:
        """
        frames_tensor must be float in [-1,1], shape [V,3,H,W] or [1,3,H,W]
        """
        if not self.loaded:
            return None

        x = frames_tensor
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1,3,H,W]

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [V,3,H,W], got {tuple(x.shape)}")

        x = x.to(dtype=torch.float32)
        x = x.unsqueeze(0).to(DEVICE, non_blocking=True)  # [B,V,3,H,W]

        with torch.inference_mode():
            action_seq = self.policy(
                x,
                take_step=0,
                return_continuous=True,
                return_sequence=True,
            )

        # Ensure [T,25]
        return action_seq.squeeze(0).detach().cpu().tolist()


engine = ModelEngine()


@app.route("/")
def index():
    replays = []
    if os.path.exists(DATASET_DIR):
        replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        replays.sort()

    cached = []
    if os.path.exists(CACHE_DIR):
        cached = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt")]
        cached.sort()

    return render_template("index.html", replays=replays, cached=cached)


@app.route("/view/<path:replay_name>")
def view_replay(replay_name):
    return render_template("view.html", replay_name=replay_name)


@app.route("/video/<path:replay_name>")
def serve_video(replay_name):
    return send_from_directory(os.path.join(DATASET_DIR, replay_name), "video.mp4")


@app.route("/inputs/<path:replay_name>")
def serve_inputs(replay_name):
    replay_path = os.path.join(DATASET_DIR, replay_name)
    jsonl_path = os.path.join(replay_path, "actions.jsonl")
    static_path = os.path.join(replay_path, "static_data.json")
    response: Dict[str, Any] = {"frames": [], "static": None}

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        response["frames"].append(json.loads(line))
                    except Exception:
                        continue

    if os.path.exists(static_path):
        try:
            with open(static_path, "r") as f:
                response["static"] = json.load(f)
        except Exception:
            pass

    return jsonify(response)


@app.route("/inspect/<path:filename>")
def inspect_cache(filename):
    return render_template("inspect_cache.html", filename=filename)


@app.route("/api/cache_meta/<path:filename>")
def cache_meta(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return jsonify({"count": int(data["frames"].shape[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache_frame/<path:filename>/<int:idx>")
def cache_frame(filename, idx):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            data = torch.load(path, map_location="cpu", weights_only=True)

        n = int(data["actions"].shape[0])
        idx = _clamp_index(idx, n)

        # --- Display frame (PNG) ---
        frame_uint8 = data["frames"][idx]             # [3,H,W] uint8
        img_np = frame_uint8.permute(1, 2, 0).numpy() # [H,W,3] uint8
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # --- Model input frames: V = vision_horizon, ending at idx ---
        V = int(getattr(engine, "vision_horizon", 1))
        seq_uint8 = _get_frame_window_uint8(data["frames"], idx, V)  # [V,3,H,W]
        # IMPORTANT: training uses [-1,1]
        seq_float = seq_uint8.float().div(255.0).mul(2.0).sub(1.0)   # [-1,1]

        # --- Ground truth: plan starting at idx + ACTION_OFFSET ---
        ACTION_OFFSET_LOCAL = int(os.getenv("ACTION_OFFSET", "0"))
        T = int(getattr(engine, "action_horizon", 18))

        gt_seq: List[Dict[str, float]] = []
        for s in range(T):
            v = data["actions"][_clamp_index(idx + ACTION_OFFSET_LOCAL + s, n)].tolist()
            gt_seq.append(_vec_cache_to_gt_dict(v))

        truth_dict = gt_seq[0] if gt_seq else _vec_cache_to_gt_dict(data["actions"][idx].tolist())

        # --- Prediction ---
        pred_seq: Optional[List[Dict[str, Any]]] = None
        pred_dict: Optional[Dict[str, Any]] = None
        pred_debug: Optional[Dict[str, Any]] = None

        if engine.loaded:
            pred_vecs = engine.infer_seq(seq_float)
            if pred_vecs:
                pred_seq = [
                    _policy_vec_to_display(v, old_layout=engine.tokenizer_old_layout)
                    for v in pred_vecs
                ]
                pred_dict = pred_seq[0]

                # debug: top-5 by probability for step 0
                prob0 = pred_dict.get("buttons_prob", {})
                pairs = [(k, float(prob0.get(k, 0.0))) for k in BUTTON_TOKENS]
                pairs.sort(key=lambda kv: kv[1], reverse=True)
                pred_debug = {
                    "top5_prob_buttons": pairs[:5],
                    "pred_meta": pred_dict.get("meta", {}),
                    "sticks_raw": pred_dict.get("sticks_raw", {}),
                }

        return jsonify({
            "image": "data:image/png;base64," + b64_img,
            "ground_truth": truth_dict,
            "ground_truth_seq": gt_seq,

            "prediction": pred_dict,
            "prediction_seq": pred_seq,

            "idx": idx,
            "horizon": T,
            "vision_horizon": V,
            "action_offset": ACTION_OFFSET_LOCAL,

            "button_tokens": BUTTON_TOKENS,
            "ui_buttons": UI_BUTTONS,

            "pred_debug": pred_debug,
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Viewer running at http://127.0.0.1:5011")
    app.run(debug=True, port=5011, host="0.0.0.0")
