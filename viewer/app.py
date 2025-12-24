# viewer/app.py
import os
import sys
import json
import math
import torch
import io
import base64
from PIL import Image
from flask import Flask, render_template, send_from_directory, jsonify, request
from typing import Any, Optional, Dict, List, Tuple

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # ../
sys.path.append(parent_dir)

# Add 'strategy' folder to path so we can import the model class
sys.path.append(os.path.join(parent_dir, "strategy"))

app = Flask(__name__)

# --- CONFIG ---
DATASET_DIR = os.path.join(parent_dir, "data/dataset")
CACHE_DIR = os.path.join(parent_dir, "data/dataset_cached")
ASSETS_DIR = os.path.join(parent_dir, "data/assets")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
CHIPS_JSON_PATHS = [os.path.join(ASSETS_DIR, "chips.json")]

STRATEGY_DB_PATH = os.path.join(parent_dir, "data/chipwindows/strategy.jsonl")
STRATEGY_MODEL_PATH = os.path.join(parent_dir, "checkpoints_strategy_rl/strategy_model_rl.pt")
RL_WEIGHTS_PATH = os.path.join(parent_dir, "data/nitrogen_rl/frame_weights.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- NEW: RL DATA LOADER ---
RL_DATA_CACHE = []

# --- IMPORT POLICIES ---
def load_rl_data():
    """Loads the lightweight JSONL file into memory for the viewer."""
    global RL_DATA_CACHE
    if os.path.exists(RL_WEIGHTS_PATH):
        print(f"⚖️  Loading RL Weights for Viewer from {RL_WEIGHTS_PATH}...")
        temp_list = []
        try:
            with open(RL_WEIGHTS_PATH, 'r') as f:
                for line in f:
                    if line.strip():
                        # Line format: {"key": "replay/frame", "val": 3.0}
                        obj = json.loads(line)
                        parts = obj['key'].split('/')
                        replay_name = parts[0]
                        frame_idx = int(parts[1])
                        
                        temp_list.append({
                            "replay": replay_name,
                            "frame": frame_idx,
                            "weight": obj['val']
                        })
            
            # Sort by weight descending (Most interesting first)
            RL_DATA_CACHE = sorted(temp_list, key=lambda x: x['weight'], reverse=True)
            print(f"✅ Loaded {len(RL_DATA_CACHE)} RL samples.")
        except Exception as e:
            print(f"❌ Error loading RL weights: {e}")

# Load immediately
load_rl_data()

# 1. NITROGEN (Battle Policy)
try:
    from ng_policy import NgNitroGenPolicy, load_ng_checkpoint
    from action_schema import BUTTON_TOKENS, GBA_UI_BUTTONS
    
    # Nitrogen Config
    _CKPT_DIR = os.path.join(parent_dir, "checkpoints")
    
    def _get_latest_checkpoint(ckpt_dir: str, default: str = "ng.pt") -> str:
        if not os.path.exists(ckpt_dir): return os.path.join(ckpt_dir, default)
        max_step = -1
        best_ckpt = default
        for fname in os.listdir(ckpt_dir):
            if fname.startswith("step_") and fname.endswith(".pt"):
                try:
                    step = int(fname.split("_")[1].split(".")[0])
                    if step > max_step:
                        max_step = step
                        best_ckpt = fname
                except: continue
        return os.path.join(ckpt_dir, best_ckpt)

    CHECKPOINT_PATH = os.getenv("NG_CKPT_PATH", _get_latest_checkpoint(_CKPT_DIR, "step_20000.pt"))
    UI_BUTTONS = list(GBA_UI_BUTTONS)
except ImportError:
    print("⚠️ Nitrogen policy not found.")
    NgNitroGenPolicy = None
    BUTTON_TOKENS = []
    UI_BUTTONS = []

# 2. STRATEGY (Chip Selection Policy)
strategy_model = None
meta_proc = None

try:
    from strategy_model import StrategyTransformer
    from train_strategy_bc import ChipMetaProcessor, META_DIM
    
    if os.path.exists(STRATEGY_MODEL_PATH) and os.path.exists(CHIPS_JSON_PATHS[0]):
        print(f"🧠 Loading Strategy Model from {STRATEGY_MODEL_PATH}...")
        try:
            meta_proc = ChipMetaProcessor(CHIPS_JSON_PATHS[0])
            strategy_model = StrategyTransformer(
                num_chip_ids=512, num_codes=32, d_model=128, meta_dim=META_DIM
            )
            state_dict = torch.load(STRATEGY_MODEL_PATH, map_location=DEVICE)
            strategy_model.load_state_dict(state_dict)
            strategy_model.to(DEVICE).eval()
            print("✅ Strategy Model Loaded!")
        except Exception as e:
            print(f"❌ Failed to load Strategy Model: {e}")
except ImportError as e:
    print(f"⚠️ Strategy modules not found: {e}")


# --- CHIP DATABASE (For Visualization) ---
CHIP_DB = {}
def load_chip_db():
    global CHIP_DB
    print(f"📂 Scanning chip data in: {ASSETS_DIR}")
    for path in CHIPS_JSON_PATHS:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for chip in data:
                        if chip.get('SId'): CHIP_DB[f"S{chip['SId']}"] = chip
                        if chip.get('MId'): CHIP_DB[f"M{chip['MId']}"] = chip
                print(f"  ✅ Loaded chips from {os.path.basename(path)}")
            except Exception as e:
                print(f"  ❌ Error loading {path}: {e}")
load_chip_db()

# --- HELPERS ---
CODE_INDEXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ*"

def resolve_chip_info(raw_id, raw_code):
    if raw_id == 255: return None 
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
        "image_url": f"/assets/images/{image_file}" if image_file else None
    }

def _sigmoid(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    else: return math.exp(x) / (1.0 + math.exp(x))

def _get_frame_window_uint8(frames_uint8, idx, V):
    n = int(frames_uint8.shape[0])
    idx = max(0, min(idx, n - 1))
    start = idx - (V - 1)
    out = []
    for t in range(V):
        src_i = max(0, min(start + t, n - 1))
        out.append(frames_uint8[src_i])
    return torch.stack(out, dim=0)

def _clamp_index(i, n): return max(0, min(int(i), int(n) - 1))

def _vec_cache_to_gt_dict(v_cache):
    d = {"AXIS_LEFTX": float(v_cache[0]), "AXIS_LEFTY": float(v_cache[1]), "AXIS_RIGHTX": float(v_cache[2]), "AXIS_RIGHTY": float(v_cache[3])}
    btn = v_cache[4:4 + len(BUTTON_TOKENS)]
    for name, value in zip(BUTTON_TOKENS, btn): d[name] = float(value)
    return d

def _detect_logits_like(values: List[float]) -> bool:
    if not values: return False
    mn = min(values)
    mx = max(values)
    return (mn < -0.05) or (mx > 1.05)

def _detect_sticks_01_like(stick2: List[float]) -> bool:
    mn = min(stick2)
    mx = max(stick2)
    return (mn >= -0.05) and (mx <= 1.05)

def _split_policy_action_layout(v25, *, old_layout):
    if len(v25) != 25: raise ValueError(f"Expected 25, got {len(v25)}")
    if old_layout:
        jl = [float(v25[0]), float(v25[1])]
        jr = [float(v25[2]), float(v25[3])]
        buttons = [float(x) for x in v25[4:]]
    else:
        buttons = [float(x) for x in v25[:-4]]
        jl = [float(v25[-4]), float(v25[-3])]
        jr = [float(v25[-2]), float(v25[-1])]
    return buttons, jl, jr

def _policy_vec_to_display(v25, *, old_layout):
    buttons_raw, jl_raw, jr_raw = _split_policy_action_layout(v25, old_layout=old_layout)
    jl01 = _detect_sticks_01_like(jl_raw)
    jr01 = _detect_sticks_01_like(jr_raw)
    jl_axis = [x * 2.0 - 1.0 for x in jl_raw] if jl01 else jl_raw[:]
    jr_axis = [x * 2.0 - 1.0 for x in jr_raw] if jr01 else jr_raw[:]
    
    buttons_are_logits = _detect_logits_like(buttons_raw)
    buttons_prob = [_sigmoid(x) for x in buttons_raw] if buttons_are_logits else [max(0.0, min(1.0, float(x))) for x in buttons_raw]

    return {
        "axes": {"AXIS_LEFTX": float(jl_axis[0]), "AXIS_LEFTY": float(jl_axis[1]), "AXIS_RIGHTX": float(jr_axis[0]), "AXIS_RIGHTY": float(jr_axis[1])},
        "sticks_raw": {"j_left": jl_raw, "j_right": jr_raw},
        "buttons_raw": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_raw)},
        "buttons_prob": {name: float(val) for name, val in zip(BUTTON_TOKENS, buttons_prob)},
        "meta": {"buttons_are_logits": buttons_are_logits}
    }

# --- ADD THIS HELPER FUNCTION ---
def _json_action_to_display(json_row: dict) -> dict:
    """Converts a raw JSONL row (flat dict) into the nested structure the viewer expects."""
    # 1. Extract Axes
    axes = {
        "AXIS_LEFTX": float(json_row.get("AXIS_LEFTX", 0.0)),
        "AXIS_LEFTY": float(json_row.get("AXIS_LEFTY", 0.0)),
        "AXIS_RIGHTX": float(json_row.get("AXIS_RIGHTX", 0.0)),
        "AXIS_RIGHTY": float(json_row.get("AXIS_RIGHTY", 0.0))
    }
    
    # 2. Extract Buttons
    buttons_raw = {}
    for btn in BUTTON_TOKENS:
        # JSONL stores 1.0/0.0
        val = float(json_row.get(btn, 0.0))
        buttons_raw[btn] = val
        
    return {
        "axes": axes,
        "buttons_raw": buttons_raw,
        "buttons_prob": buttons_raw, # For JSONL, raw 1.0 IS the probability
        "meta": {"source": "jsonl"}
    }

# --- NITROGEN ENGINE ---
class ModelEngine:
    def __init__(self):
        self.policy = None
        self.loaded = False
        self.action_horizon = 18
        self.vision_horizon = 1
        self.tokenizer_old_layout = False

        if NgNitroGenPolicy and load_ng_checkpoint and os.path.exists(CHECKPOINT_PATH):
            try:
                print(f"🧠 Loading Nitrogen from {CHECKPOINT_PATH}...")
                loaded = load_ng_checkpoint(CHECKPOINT_PATH, device=torch.device(DEVICE))
                self.policy = NgNitroGenPolicy(loaded).to(DEVICE).eval()
                self.loaded = True
                
                tok = getattr(self.policy, "tokenizer", None)
                if tok:
                    self.action_horizon = getattr(tok, "action_horizon", 18)
                    self.vision_horizon = getattr(tok, "vision_horizon", 1)
                    self.tokenizer_old_layout = getattr(tok, "old_layout", False)
                print("✅ Nitrogen loaded.")
            except Exception as e:
                print(f"❌ Nitrogen load failed: {e}")

    def infer_seq(self, frames_tensor):
        if not self.loaded: return None
        if frames_tensor.ndim == 3: frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.to(dtype=torch.float32).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.inference_mode():
            action_seq = self.policy(frames_tensor, take_step=0, return_continuous=True, return_sequence=True)
        return action_seq.squeeze(0).detach().cpu().tolist()

engine = ModelEngine()

# --- ROUTES ---

@app.route("/")
def index():
    replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    replays.sort()
    cached = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt")]
    cached.sort()
    has_strategy = os.path.exists(STRATEGY_DB_PATH)
    return render_template("index.html", replays=replays, cached=cached, has_strategy=has_strategy)

@app.route("/view/<path:replay_name>")
def view_replay(replay_name):
    return render_template("view.html", replay_name=replay_name)

@app.route("/video/<path:replay_name>")
def serve_video(replay_name):
    return send_from_directory(os.path.join(DATASET_DIR, replay_name), "video.mp4")

@app.route("/assets/images/<path:filename>")
def serve_chip_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/strategy")
def view_strategy():
    turns = []
    if os.path.exists(STRATEGY_DB_PATH):
        try:
            with open(STRATEGY_DB_PATH, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    raw_turn = json.loads(line)
                    
                    # 1. Process Hand for Display
                    processed_hand = []
                    hand_slots = raw_turn.get('hand_slots', [])
                    hand_codes = raw_turn.get('hand_codes', [])
                    for i, chip_id in enumerate(hand_slots):
                        chip_code = hand_codes[i]
                        info = resolve_chip_info(chip_id, chip_code)
                        processed_hand.append(info)
                    raw_turn['rich_hand'] = processed_hand
                    
                    # 2. AI Inference
                    if strategy_model and meta_proc:
                        try:
                            with torch.no_grad():
                                # A. Inputs
                                h_ids = torch.tensor([hand_slots], dtype=torch.long).to(DEVICE)
                                san_codes = [min(c, 31) for c in hand_codes]
                                h_codes = torch.tensor([san_codes], dtype=torch.long).to(DEVICE)
                                
                                # B. Meta
                                meta_list = [meta_proc.get_meta(cid) for cid in hand_slots]
                                h_meta = torch.stack(meta_list).unsqueeze(0).to(DEVICE)
                                
                                # C. Context (HP + Used Crosses)
                                hp_ctx = torch.tensor([raw_turn['p_hp_start']/1000.0, raw_turn['e_hp_start']/2000.0], dtype=torch.float32)
                                used_vec = torch.zeros(6, dtype=torch.float32)
                                for c in raw_turn.get('used_crosses', []):
                                    if 1 <= c <= 6: used_vec[c-1] = 1.0
                                full_ctx = torch.cat([hp_ctx, used_vec]).unsqueeze(0).to(DEVICE)
                                
                                # D. Forward
                                B = 1
                                pos = torch.arange(10, device=DEVICE).unsqueeze(0)
                                src = (strategy_model.chip_embedding(h_ids) + 
                                       strategy_model.code_embedding(h_codes) + 
                                       strategy_model.meta_proj(h_meta) + 
                                       strategy_model.pos_embedding(pos))
                                src = src + strategy_model.context_proj(full_ctx).unsqueeze(1)
                                
                                memory = strategy_model.encoder(src)
                                cross_logits = strategy_model.cross_head(memory.mean(dim=1))
                                _, seq_tokens = strategy_model.inference(memory, cross_logits)
                                
                                # E. Parse
                                raw_turn['ai_indices'] = [t for t in seq_tokens[0].tolist() if t < 10]
                                raw_turn['ai_cross'] = torch.argmax(cross_logits, dim=1).item()
                        except Exception as e:
                            print(f"Inference error: {e}")

                    turns.append(raw_turn)
        except Exception as e:
            print(f"Error reading strategy DB: {e}")
            
    return render_template("view_strategy.html", turns=turns[::-1])

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
                    except: continue
    if os.path.exists(static_path):
        try:
            with open(static_path, "r") as f:
                response["static"] = json.load(f)
        except: pass
    return jsonify(response)

@app.route("/api/cache_meta/<path:filename>")
def cache_meta(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path): return jsonify({"error": "File not found"}), 404
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return jsonify({"count": int(data["frames"].shape[0])})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/cache_frame/<path:filename>/<int:idx>")
def cache_frame(filename, idx):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path): return jsonify({"error": "File not found"}), 404
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        n = int(data["actions"].shape[0])
        idx = _clamp_index(idx, n)
        
        # Frame
        frame_uint8 = data["frames"][idx]
        img_np = frame_uint8.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Inference Prep
        V = int(getattr(engine, "vision_horizon", 1))
        seq_uint8 = _get_frame_window_uint8(data["frames"], idx, V)
        seq_float = seq_uint8.float().div(255.0).mul(2.0).sub(1.0)
        
        # Ground Truth
        ACTION_OFFSET_LOCAL = int(os.getenv("ACTION_OFFSET", "0"))
        T = int(getattr(engine, "action_horizon", 18))
        gt_seq = []
        for s in range(T):
            v = data["actions"][_clamp_index(idx + ACTION_OFFSET_LOCAL + s, n)].tolist()
            gt_seq.append(_vec_cache_to_gt_dict(v))
        truth_dict = gt_seq[0] if gt_seq else {}
        
        # Pred
        pred_seq, pred_dict, pred_debug = None, None, None
        if engine.loaded:
            pred_vecs = engine.infer_seq(seq_float)
            if pred_vecs:
                pred_seq = [_policy_vec_to_display(v, old_layout=engine.tokenizer_old_layout) for v in pred_vecs]
                pred_dict = pred_seq[0]
                
        return jsonify({
            "image": "data:image/png;base64," + b64_img,
            "ground_truth": truth_dict,
            "ground_truth_seq": gt_seq,
            "prediction": pred_dict,
            "prediction_seq": pred_seq,
            "idx": idx, "horizon": T, "vision_horizon": V
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/inspect/<path:filename>")
def inspect_cache(filename):
    return render_template("inspect_cache.html", filename=filename)



# --- NEW ROUTE: RL INSPECTOR UI ---
@app.route("/rl")
def view_rl_inspector():
    # Pass metadata so we can show counts
    stats = {
        "total": len(RL_DATA_CACHE),
        "high_reward": len([x for x in RL_DATA_CACHE if x['weight'] > 1.0]),
        "punishment": len([x for x in RL_DATA_CACHE if x['weight'] < 1.0])
    }
    return render_template("rl_inspector.html", stats=stats)

# --- NEW ROUTE: RL SAMPLE API ---
@app.route("/api/rl_list")
def api_rl_list():
    """Returns the lightweight list for the sidebar."""
    # Optional filtering
    filter_type = request.args.get('filter', 'all') # all, high, low
    
    data = RL_DATA_CACHE
    if filter_type == 'high':
        data = [x for x in RL_DATA_CACHE if x['weight'] > 1.0]
    elif filter_type == 'low':
        data = [x for x in RL_DATA_CACHE if x['weight'] < 1.0]
        
    # Limit to first 1000 to prevent browser lag if list is huge
    return jsonify(data[:2000])


# --- REPLACE THE api_rl_detail ROUTE ---
@app.route("/api/rl_detail/<int:list_idx>")
def api_rl_detail(list_idx):
    # 1. Validate Index
    if list_idx < 0 or list_idx >= len(RL_DATA_CACHE):
        return jsonify({"error": "Index out of bounds"}), 404
        
    sample = RL_DATA_CACHE[list_idx]
    replay_name = sample['replay']
    frame_idx = sample['frame']
    
    # Response Data Containers
    b64_img = ""
    action_seq = []
    data_found = False
    
    # 2. STRATEGY A: TRY CACHE (.PT)
    pt_path = os.path.join(CACHE_DIR, f"{replay_name}.pt")
    
    if os.path.exists(pt_path):
        try:
            # CPU map avoids VRAM usage
            data = torch.load(pt_path, map_location='cpu')
            if frame_idx < data['frames'].shape[0]:
                # Extract Image
                frame_uint8 = data['frames'][frame_idx]
                img_np = frame_uint8.permute(1, 2, 0).numpy()
                pil_img = Image.fromarray(img_np)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                # Extract Actions
                horizon = 18
                end_idx = min(frame_idx + horizon, data['actions'].shape[0])
                actions_tensor = data['actions'][frame_idx : end_idx]
                
                for t in range(actions_tensor.shape[0]):
                    vec = actions_tensor[t].tolist()
                    disp = _policy_vec_to_display(vec, old_layout=False)
                    action_seq.append(disp)
                
                data_found = True
        except Exception as e:
            print(f"⚠️ Error reading PT file: {e}")

    # 3. STRATEGY B: FALLBACK TO RAW JSONL (If PT failed)
    if not data_found:
        jsonl_path = os.path.join(DATASET_DIR, replay_name, "actions.jsonl")
        if os.path.exists(jsonl_path):
            try:
                # Read specific window of lines
                # (Reading all lines is fine for <50MB files)
                with open(jsonl_path, 'r') as f:
                    lines = f.readlines()
                
                if frame_idx < len(lines):
                    # No Tensor Image available in JSONL
                    b64_img = "" 
                    
                    # Extract Actions
                    horizon = 18
                    end_idx = min(frame_idx + horizon, len(lines))
                    
                    for i in range(frame_idx, end_idx):
                        try:
                            row = json.loads(lines[i])
                            disp = _json_action_to_display(row)
                            action_seq.append(disp)
                        except: pass
                    
                    data_found = True
            except Exception as e:
                print(f"❌ Error reading JSONL: {e}")

    if not data_found:
        return jsonify({"error": f"Could not find data for {replay_name} frame {frame_idx}"}), 404

    return jsonify({
        "replay": replay_name,
        "frame": frame_idx,
        "weight": sample['weight'],
        "image": "data:image/png;base64," + b64_img if b64_img else None,
        "actions": action_seq,
        "timestamp": float(frame_idx) / 60.0,
        "source_type": "pt" if b64_img else "jsonl"
    })


if __name__ == "__main__":
    print("🚀 Viewer running at http://127.0.0.1:5011")
    app.run(debug=True, port=5011, host="0.0.0.0")