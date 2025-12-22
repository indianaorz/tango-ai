import os
import json
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

# Config: Point this to your actual dataset folder
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))

@app.route('/')
def index():
    if not os.path.exists(DATASET_DIR):
        return f"Error: Dataset directory not found at {DATASET_DIR}"
    replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    replays.sort()
    return render_template('index.html', replays=replays)

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
    
    response = {
        "frames": [],
        "static": None
    }

    # 1. Load Frames
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        response["frames"].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # 2. Load Static Data (Folder, NaviCust)
    if os.path.exists(static_path):
        try:
            with open(static_path, 'r') as f:
                response["static"] = json.load(f)
        except Exception as e:
            print(f"Error loading static data: {e}")

    return jsonify(response)

if __name__ == '__main__':
    print(f"Serving Data from: {DATASET_DIR}")
    app.run(debug=True, port=5011)