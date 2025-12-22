import os
import json
import torch
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify, request

app = Flask(__name__)

# --- CONFIG ---
# Paths relative to where you run the script (usually root)
DATASET_DIR = os.path.abspath("data/dataset")
CACHE_DIR = os.path.abspath("data/dataset_cached")

# Mapping for the Cached Vector back to Dictionary for the UI
BUTTON_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE', 
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_SHOULDER', 
    'RIGHT_THUMB', 'RIGHT_TRIGGER', 'SOUTH', 'START', 'WEST', 
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP' 
]

@app.route('/')
def index():
    # List Raw Replays
    replays = []
    if os.path.exists(DATASET_DIR):
        replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        replays.sort()
    
    # List Cached Tensors
    cached = []
    if os.path.exists(CACHE_DIR):
        cached = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
        cached.sort()

    return render_template('index.html', replays=replays, cached=cached)

# --- RAW DATASET ROUTES ---
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
                    try:
                        response["frames"].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if os.path.exists(static_path):
        try:
            with open(static_path, 'r') as f:
                response["static"] = json.load(f)
        except: pass

    return jsonify(response)

# --- CACHED DATA ROUTES ---

@app.route('/inspect/<path:filename>')
def inspect_cache(filename):
    return render_template('inspect_cache.html', filename=filename)

@app.route('/api/cache_meta/<path:filename>')
def cache_meta(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path): return jsonify({"error": "File not found"}), 404
    
    # Load just the structure (map_location cpu)
    try:
        # We load lightly just to get length
        data = torch.load(path, map_location='cpu', weights_only=True)
        count = data["frames"].shape[0]
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache_frame/<path:filename>/<int:idx>')
def cache_frame(filename, idx):
    path = os.path.join(CACHE_DIR, filename)
    
    try:
        # Load Tensor (Memory mapping would be better for prod, but direct load is fine for localhost single user)
        # Using mmap=True if available (PyTorch 2.1+)
        try:
            data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except TypeError:
            data = torch.load(path, map_location='cpu', weights_only=True)

        # 1. Get Frame Image
        # [T, 3, H, W] -> [3, H, W]
        frame_tensor = data["frames"][idx]
        # [3, H, W] -> [H, W, 3] -> Numpy
        img_np = frame_tensor.permute(1, 2, 0).numpy()
        
        # Convert to PNG Base64
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 2. Get Action Data
        # [LeftX, LeftY, RightX, RightY, ...Buttons]
        act_vec = data["actions"][idx].tolist()
        
        action_dict = {
            "AXIS_LEFTX": [act_vec[0]],
            "AXIS_LEFTY": [act_vec[1]],
            "AXIS_RIGHTX": [act_vec[2]],
            "AXIS_RIGHTY": [act_vec[3]]
        }
        
        for i, btn_name in enumerate(BUTTON_TOKENS):
            # Buttons start at index 4
            action_dict[btn_name] = act_vec[4 + i]

        return jsonify({
            "image": "data:image/png;base64," + b64_img,
            "action": action_dict,
            "idx": idx
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Viewer running at http://127.0.0.1:5011")
    app.run(debug=True, port=5011)