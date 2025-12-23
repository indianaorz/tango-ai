import os
import sys
import json
import torch
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify, request

# --- PATH HACK: Look 1 folder up for ng_policy ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # ../
sys.path.append(parent_dir)
# -------------------------------------------------

# --- IMPORT POLICY ---
try:
    from ng_policy import NgNitroGenPolicy, load_ng_checkpoint
except ImportError as e:
    print(f"⚠️ Warning: ng_policy.py not found in {parent_dir}. Inference disabled.")
    print(f"Error details: {e}")
    NgNitroGenPolicy = None

app = Flask(__name__)

# --- CONFIG ---
# Since app.py is in /viewer, we look up one level for data
DATASET_DIR = os.path.join(parent_dir, "data/dataset")
CACHE_DIR = os.path.join(parent_dir, "data/dataset_cached")
# CHECKPOINT_PATH = os.path.join(parent_dir, "checkpoints/step_10000.pt")#os.path.join(parent_dir, "weights/ng.pt")
CHECKPOINT_PATH = os.getenv("NG_CKPT_PATH", os.path.join(parent_dir, "weights", "ng.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE', 
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER', 
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST', 
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP' 
]

# --- MODEL SINGLETON ---
class ModelEngine:
    def __init__(self):
        self.policy = None
        self.loaded = False

        self.action_horizon = 18
        if self.loaded and getattr(self.policy, "tokenizer", None) is not None:
            ah = getattr(self.policy.tokenizer, "action_horizon", None)
            if isinstance(ah, int) and ah > 0:
                self.action_horizon = ah
        if NgNitroGenPolicy and os.path.exists(CHECKPOINT_PATH):
            try:
                print(f"🧠 Loading model from {CHECKPOINT_PATH}...")
                loaded = load_ng_checkpoint(CHECKPOINT_PATH, device=torch.device(DEVICE))
                self.policy = NgNitroGenPolicy(loaded).to(DEVICE).eval()
                self.loaded = True
                print("✅ Model loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")

    def infer(self, frames_tensor: torch.Tensor):
        """
        frames_tensor:
          - [3,H,W] or [T,3,H,W] in [0,1] float
        Returns:
          - action vector (25 floats) for the last step in the provided window
        """
        if not self.loaded:
            return None
        try:
            x = frames_tensor
            if x.ndim == 3:
                x = x.unsqueeze(0)  # [1,3,H,W] -> treat as T=1

            if x.ndim != 4 or x.shape[1] != 3:
                raise ValueError(f"Expected [T,3,H,W], got {tuple(x.shape)}")

            x = x.float().to(DEVICE, non_blocking=True)
            x = x.unsqueeze(0)  # [1,T,3,H,W] (batch=1)

            take_step = int(x.shape[1] - 1)  # action for last frame in window

            with torch.inference_mode():
                action_vec = self.policy(x, take_step=take_step, return_continuous=True)

            return action_vec.squeeze(0).detach().cpu().tolist()
        except Exception as e:
            print(f"Inference Error: {e}")
            return None



def _get_frame_window_uint8(frames_uint8: torch.Tensor, idx: int, T: int) -> torch.Tensor:
    """
    frames_uint8: [N, 3, H, W] uint8
    Returns:      [T, 3, H, W] uint8 window ending at idx (inclusive), padded by clamping.
    """
    n = int(frames_uint8.shape[0])
    if n <= 0:
        raise ValueError("Empty frames tensor")

    idx = max(0, min(idx, n - 1))
    start = idx - (T - 1)

    out = []
    for t in range(T):
        src_i = start + t
        if src_i < 0:
            src_i = 0
        elif src_i >= n:
            src_i = n - 1
        out.append(frames_uint8[src_i])
    return torch.stack(out, dim=0)



engine = ModelEngine()

@app.route('/')
def index():
    replays = []
    if os.path.exists(DATASET_DIR):
        replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        replays.sort()
    
    cached = []
    if os.path.exists(CACHE_DIR):
        cached = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
        cached.sort()

    return render_template('index.html', replays=replays, cached=cached)

@app.route('/view/<path:replay_name>')
def view_replay(replay_name):
    return render_template('view.html', replay_name=replay_name)

@app.route('/video/<path:replay_name>')
def serve_video(replay_name):
    return send_from_directory(os.path.join(DATASET_DIR, replay_name), 'video.mp4')

@app.route('/inputs/<path:replay_name>')
def serve_inputs(replay_name):
    replay_path = os.path.join(DATASET_DIR, replay_name)
    jsonl_path = os.path.join(replay_path, 'actions.jsonl')
    static_path = os.path.join(replay_path, 'static_data.json')
    response = {"frames": [], "static": None}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try: response["frames"].append(json.loads(line))
                    except: continue
    if os.path.exists(static_path):
        try:
            with open(static_path, 'r') as f: response["static"] = json.load(f)
        except: pass
    return jsonify(response)

# --- CACHED DATA INSPECTOR ---
@app.route('/inspect/<path:filename>')
def inspect_cache(filename):
    return render_template('inspect_cache.html', filename=filename)

@app.route('/api/cache_meta/<path:filename>')
def cache_meta(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path): return jsonify({"error": "File not found"}), 404
    try:
        data = torch.load(path, map_location='cpu', weights_only=True)
        return jsonify({"count": data["frames"].shape[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache_frame/<path:filename>/<int:idx>')
def cache_frame(filename, idx):
    path = os.path.join(CACHE_DIR, filename)
    try:
        # Load File
        try: data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except TypeError: data = torch.load(path, map_location='cpu', weights_only=True)

        # Single frame for display
        frame_uint8 = data["frames"][idx]                 # [3,H,W] uint8
        frame_float = frame_uint8.float() / 255.0         # [3,H,W] in [0,1]

        # Sequence window for model (base model/tokenizer expects T == action_horizon)
        T = getattr(engine, "action_horizon", 18)
        seq_uint8 = _get_frame_window_uint8(data["frames"], idx, T)     # [T,3,H,W] uint8
        seq_float = seq_uint8.float() / 255.0                           # [T,3,H,W] in [0,1]

        
        # Prepare PNG
        img_np = (frame_float * 255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 2. Get Ground Truth Action
        act_vec = data["actions"][idx].tolist()
        
        def vec_to_dict(v):
            d = { "AXIS_LEFTX": v[0], "AXIS_LEFTY": v[1], "AXIS_RIGHTX": v[2], "AXIS_RIGHTY": v[3] }
            for i, name in enumerate(BUTTON_TOKENS):
                d[name] = v[4 + i]
            return d

        truth_dict = vec_to_dict(act_vec)

        # 3. Run Inference (Optional)
        pred_dict = None
        if engine.loaded:
            pred_vec = engine.infer(seq_float)  # pass [T,3,H,W]
            if pred_vec:
                pred_dict = vec_to_dict(pred_vec)

        return jsonify({
            "image": "data:image/png;base64," + b64_img,
            "ground_truth": truth_dict,
            "prediction": pred_dict,
            "idx": idx
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Viewer running at http://127.0.0.1:5011")
    app.run(debug=True, port=5011, host='0.0.0.0')