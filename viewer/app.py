# viewer/app.py (python portion)
import os
import sys
import json
import math
import torch
import io
import base64
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify
from typing import Any, Optional

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
CHECKPOINT_PATH = os.path.join(parent_dir, "weights/ng.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UI buttons: ONLY the GBA-relevant set we want to display
UI_BUTTONS = list(GBA_UI_BUTTONS)

ACTION_OFFSET = int(os.getenv("ACTION_OFFSET", "0"))  # keep in sync with training


class ModelEngine:
    def __init__(self):
        self.policy = None
        self.loaded = False

        self.action_horizon = 18
        self.vision_horizon = 1
        self.action_offset = int(os.getenv("ACTION_OFFSET", "0"))

        # Canonical order (must match training+precache)
        self.button_tokens = BUTTON_TOKENS[:]

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
                    if isinstance(ah, int) and ah > 0:
                        self.action_horizon = ah
                    if isinstance(vh, int) and vh > 0:
                        self.vision_horizon = vh

                print(
                    "✅ Model loaded.\n"
                    f"  action_horizon={self.action_horizon}\n"
                    f"  vision_horizon={self.vision_horizon}\n"
                    f"  action_offset={self.action_offset}\n"
                    f"  button_tokens={self.button_tokens}"
                )

            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.loaded = False

    def infer_seq(self, frames_tensor: torch.Tensor) -> Optional[list[list[float]]]:
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

        return action_seq.squeeze(0).detach().cpu().tolist()


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
    return torch.stack(out, dim=0)


def _vec25_to_gt_dict(v25: list[float]) -> dict[str, float]:
    d: dict[str, float] = {
        "AXIS_LEFTX": float(v25[0]),
        "AXIS_LEFTY": float(v25[1]),
        "AXIS_RIGHTX": float(v25[2]),
        "AXIS_RIGHTY": float(v25[3]),
    }
    btn = v25[4:4 + len(BUTTON_TOKENS)]
    for name, value in zip(BUTTON_TOKENS, btn):
        d[name] = float(value)
    return d


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _vec25_to_pred_display(v25: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "axes": {
            "AXIS_LEFTX": float(v25[0]),
            "AXIS_LEFTY": float(v25[1]),
            "AXIS_RIGHTX": float(v25[2]),
            "AXIS_RIGHTY": float(v25[3]),
        },
        "buttons_raw": {},
        "buttons_prob": {},
    }

    btn = v25[4:4 + len(BUTTON_TOKENS)]
    for name, value in zip(BUTTON_TOKENS, btn):
        raw = float(value)
        out["buttons_raw"][name] = raw
        out["buttons_prob"][name] = float(_sigmoid(raw))
    return out


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
    response: dict[str, Any] = {"frames": [], "static": None}

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

        # --- Display frame (PNG) ---
        frame_uint8 = data["frames"][idx]             # [3,H,W] uint8
        img_np = frame_uint8.permute(1, 2, 0).numpy() # [H,W,3] uint8
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        def clamp_i(i: int) -> int:
            n = int(data["actions"].shape[0])
            return max(0, min(i, n - 1))

        # --- Model input frames: V = vision_horizon, ending at idx ---
        V = int(getattr(engine, "vision_horizon", 1))
        seq_uint8 = _get_frame_window_uint8(data["frames"], idx, V)  # [V,3,H,W]
        seq_float = seq_uint8.float().div(255.0)                     # [0,1]

        # --- Ground truth: plan starting at idx + ACTION_OFFSET ---
        ACTION_OFFSET_LOCAL = int(os.getenv("ACTION_OFFSET", "0"))
        T = int(getattr(engine, "action_horizon", 18))

        truth_dict = _vec25_to_gt_dict(data["actions"][clamp_i(idx + ACTION_OFFSET_LOCAL)].tolist())

        gt_seq = []
        for s in range(T):
            v = data["actions"][clamp_i(idx + ACTION_OFFSET_LOCAL + s)].tolist()
            gt_seq.append(_vec25_to_gt_dict(v))

        # --- Prediction ---
        pred_vecs: Optional[list[list[float]]] = None
        pred_seq = None
        pred_dict = None
        pred_debug = None

        if engine.loaded:
            pred_vecs = engine.infer_seq(seq_float)
            if pred_vecs:
                pred_seq = [_vec25_to_pred_display(v) for v in pred_vecs]
                pred_dict = pred_seq[0]

                # debug top-5 by raw logit (step 0)
                raw0 = pred_vecs[0][4:4 + len(BUTTON_TOKENS)]
                pairs = list(zip(BUTTON_TOKENS, [float(x) for x in raw0]))
                pairs.sort(key=lambda kv: kv[1], reverse=True)
                pred_debug = {"top5_raw_buttons": pairs[:5]}

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
