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
import torchvision.transforms.functional as TF
from typing import Any, Optional, Dict, List, Tuple

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # ../
sys.path.append(parent_dir)

# Add 'strategy' folder to path so we can import the model class
sys.path.append(os.path.join(parent_dir, "strategy"))

# --- IMPORT ACTION SCHEMA ---
try:
    from action_schema import BUTTON_TOKENS, GBA_UI_BUTTONS
except ImportError:
    print("⚠️ action_schema.py not found. Using defaults.")
    BUTTON_TOKENS = []
    GBA_UI_BUTTONS = []

app = Flask(__name__)

# --- CONFIG & DIRECTORIES ---
DATASET_DIR = os.path.join(parent_dir, "data/dataset")
ASSETS_DIR = os.path.join(parent_dir, "data/assets")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
CHIPS_JSON_PATHS = [os.path.join(ASSETS_DIR, "chips.json")]
MASK_PATH = os.path.join(parent_dir, "chip_window_mask.png")

# Cache Locations
CACHE_DIRS = {
    "Legacy": os.path.join(parent_dir, "data/dataset_cached"),
    "Plan": os.path.join(parent_dir, "data/planning_cache"),
    "Battle": os.path.join(parent_dir, "data/battle_cache"),
}

# Checkpoint Locations
CKPT_ROOT = ""#os.path.join(parent_dir, "checkpoints")
PLANNING_CKPT_DIR = ""#os.path.join(CKPT_ROOT, "planning")
BATTLE_CKPT_DIR = ""#os.path.join(CKPT_ROOT, "battle")

# Strategy & RL Paths
STRATEGY_DB_PATH = os.path.join(parent_dir, "data/chipwindows/strategy.jsonl")
STRATEGY_MODEL_PATH = os.path.join(parent_dir, "checkpoints_strategy/strategy_model.pt")
RL_WEIGHTS_PATH = os.path.join(parent_dir, "data/nitrogen_rl/frame_weights.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER: FIND CACHE ---
def find_cache_info(filename: str, hint_label: str = None) -> Tuple[Optional[str], str]:
    """
    Returns (full_path, cache_type_label).
    If hint_label is provided, checks that specific folder first.
    """
    # 1. If a hint is provided, check that specific folder first
    if hint_label and hint_label in CACHE_DIRS:
        candidate = os.path.join(CACHE_DIRS[hint_label], filename)
        if os.path.exists(candidate):
            return candidate, hint_label

    # 2. Fallback: Search all folders (existing logic)
    for label, path in CACHE_DIRS.items():
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate, label
            
    return None, "Unknown"

# --- HELPER: MATH ---
def _sigmoid(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    else: return math.exp(x) / (1.0 + math.exp(x))

def _detect_logits_like(values: List[float]) -> bool:
    if not values: return False
    mn, mx = min(values), max(values)
    return (mn < -0.05) or (mx > 1.05)

def _detect_sticks_01_like(stick2: List[float]) -> bool:
    mn, mx = min(stick2), max(stick2)
    return (mn >= -0.05) and (mx <= 1.05)

def _clamp_index(i, n): return max(0, min(int(i), int(n) - 1))

# --- HELPER: MASK LOADING ---
_CACHED_MASK_TENSOR = None
def get_mask_tensor():
    global _CACHED_MASK_TENSOR
    if _CACHED_MASK_TENSOR is None:
        if os.path.exists(MASK_PATH):
            try:
                img = Image.open(MASK_PATH).convert("RGBA").resize((256, 256), Image.NEAREST)
                alpha = TF.to_tensor(img)[3, :, :]
                # True = Keep (Alpha=0), False = Mask (Alpha>0)
                _CACHED_MASK_TENSOR = (alpha == 0.0).bool()
            except Exception as e:
                print(f"Failed to load mask: {e}")
    return _CACHED_MASK_TENSOR

# --- NITROGEN ENGINE (DUAL MODEL SUPPORT) ---
class DualModelEngine:
    def __init__(self):
        self.models = { "Battle": None, "Plan": None }
        self.configs = {
            "Battle": {"action_horizon": 18, "vision_horizon": 1, "old_layout": False},
            "Plan": {"action_horizon": 18, "vision_horizon": 1, "old_layout": False},
        }

        try:
            from ng_policy import NgNitroGenPolicy, load_ng_checkpoint
            self.PolicyClass = NgNitroGenPolicy
            self.loader_func = load_ng_checkpoint
        except ImportError:
            print("⚠️ Nitrogen policy code not found.")
            return

        # Load Models
        self._load_best_ckpt(BATTLE_CKPT_DIR, "Battle")
        self._load_best_ckpt(PLANNING_CKPT_DIR, "Plan")

    def _load_best_ckpt(self, ckpt_dir, key):
        if not os.path.exists(ckpt_dir): return
        max_step = -1
        best_file = None
        for f in os.listdir(ckpt_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                try:
                    step = int(f.split("_")[1].split(".")[0])
                    if step > max_step:
                        max_step = step
                        best_file = f
                except: continue
        
        if best_file:
            path = os.path.join(ckpt_dir, best_file)
            print(f"🧠 Loading {key} Model: {best_file}...")
            try:
                loaded = self.loader_func(path, device=torch.device(DEVICE))
                policy = self.PolicyClass(loaded).to(DEVICE).eval()
                self.models[key] = policy
                
                tok = getattr(policy, "tokenizer", None)
                if tok:
                    self.configs[key]["action_horizon"] = getattr(tok, "action_horizon", 18)
                    self.configs[key]["vision_horizon"] = getattr(tok, "vision_horizon", 1)
                    self.configs[key]["old_layout"] = getattr(tok, "old_layout", False)
                print(f"✅ {key} Model Loaded.")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")

    def infer(self, frames_tensor, model_key="Battle"):
        # Map Legacy/Unknown to Battle by default
        if model_key not in self.models: model_key = "Battle"
        
        policy = self.models.get(model_key)
        if policy is None: return None

        if frames_tensor.ndim == 3: frames_tensor = frames_tensor.unsqueeze(0)
        frames_batch = frames_tensor.unsqueeze(0).to(DEVICE, non_blocking=True)
        
        with torch.inference_mode():
            action_seq = policy(frames_batch, take_step=0, return_continuous=True, return_sequence=True)
        
        return action_seq.squeeze(0).detach().cpu().tolist()

engine = DualModelEngine()

# --- RL DATA LOADER ---
RL_DATA_CACHE = []
def load_rl_data():
    """Loads the lightweight JSONL file into memory for the viewer."""
    global RL_DATA_CACHE
    if os.path.exists(RL_WEIGHTS_PATH):
        print(f"⚖️  Loading RL Weights from {RL_WEIGHTS_PATH}...")
        temp_list = []
        try:
            with open(RL_WEIGHTS_PATH, 'r') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        parts = obj['key'].split('/')
                        if len(parts) >= 2:
                            temp_list.append({
                                "replay": parts[0],
                                "frame": int(parts[1]),
                                "weight": obj['val']
                            })
            # Sort by weight descending (Most interesting first)
            RL_DATA_CACHE = sorted(temp_list, key=lambda x: x['weight'], reverse=True)
            print(f"✅ Loaded {len(RL_DATA_CACHE)} RL samples.")
        except Exception as e: print(f"❌ Error loading RL weights: {e}")
load_rl_data()

# --- STRATEGY MODEL ---
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

# --- CHIP DATABASE ---
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
            except Exception as e:
                print(f"  ❌ Error loading {path}: {e}")
load_chip_db()

# --- FORMATTING HELPERS ---
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

def _split_policy_action_layout(v25, *, old_layout):
    if len(v25) != 25: 
        # Fallback/Error state
        return [0]*21, [0,0], [0,0]
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

def _cache_vec_to_gt(v):
    # Cache format: [LX, LY, RX, RY, BTNS...]
    d = {"AXIS_LEFTX": float(v[0]), "AXIS_LEFTY": float(v[1]), "AXIS_RIGHTX": float(v[2]), "AXIS_RIGHTY": float(v[3])}
    btn_vals = v[4:]
    for name, val in zip(BUTTON_TOKENS, btn_vals): d[name] = float(val)
    return d

def _json_action_to_display(json_row: dict) -> dict:
    """Converts a raw JSONL row (flat dict) into the nested structure."""
    axes = {
        "AXIS_LEFTX": float(json_row.get("AXIS_LEFTX", 0.0)),
        "AXIS_LEFTY": float(json_row.get("AXIS_LEFTY", 0.0)),
        "AXIS_RIGHTX": float(json_row.get("AXIS_RIGHTX", 0.0)),
        "AXIS_RIGHTY": float(json_row.get("AXIS_RIGHTY", 0.0))
    }
    
    buttons_raw = {}
    for btn in BUTTON_TOKENS:
        val = float(json_row.get(btn, 0.0))
        buttons_raw[btn] = val
        
    return {
        "axes": axes,
        "buttons_raw": buttons_raw,
        "buttons_prob": buttons_raw, 
        "meta": {"source": "jsonl"}
    }

# --- ROUTES ---

@app.route("/")
def index():
    replays = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    replays.sort()
    
    # List files for all cache types
    cached_files = {}
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
    return send_from_directory(os.path.join(DATASET_DIR, replay_name), "video.mp4")

@app.route("/assets/images/<path:filename>")
def serve_chip_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/api/chip_library")
def api_chip_library():
    """Returns a simplified version of CHIP_DB for frontend visualization."""
    return jsonify(CHIP_DB)

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
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            response["frames"].append(json.loads(line))
                        except: continue
        except: pass
    if os.path.exists(static_path):
        try:
            with open(static_path, "r") as f:
                response["static"] = json.load(f)
        except: pass
    return jsonify(response)

@app.route("/api/cache_meta/<path:filename>")
def cache_meta(filename):
    # Get type from query string
    req_type = request.args.get('type')
    
    path, _ = find_cache_info(filename, hint_label=req_type)
    
    if not path: return jsonify({"error": "File not found"}), 404
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return jsonify({"count": int(data["frames"].shape[0])})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/cache_frame/<path:filename>/<int:idx>")
def cache_frame(filename, idx):
    # Get type from query string
    req_type = request.args.get('type')
    apply_mask = request.args.get('mask') == '1'
    
    path, cache_type = find_cache_info(filename, hint_label=req_type)
    
    if not path: return jsonify({"error": "File not found"}), 404
    
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        actions_tensor = data["actions"]
        n_frames = int(data["frames"].shape[0])
        idx = max(0, min(idx, n_frames - 1))
        
        # Check if actions are baked (3D) or flat (2D)
        is_baked_3d = (actions_tensor.ndim == 3)
        
        # 1. Image
        frame_uint8 = data["frames"][idx]

        # --- MASK APPLICATION ---
        if apply_mask:
            mask = get_mask_tensor()
            if mask is not None:
                # Expand mask to [3, 256, 256] matching frame
                mask_3ch = mask.unsqueeze(0).expand_as(frame_uint8)
                # Apply mask (0 out opaque areas)
                frame_uint8 = frame_uint8.masked_fill(~mask_3ch, 0)

        img_np = frame_uint8.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 2. Config & Input Window
        # Determine which model config to use based on folder name
        model_key = cache_type if cache_type in ["Plan", "Battle"] else "Battle"
        cfg = engine.configs.get(model_key, engine.configs["Battle"])
        V = cfg["vision_horizon"]
        
        # Get frame window
        indices = [max(0, idx - (V - 1) + i) for i in range(V)]
        seq_uint8 = data["frames"][indices]

        if apply_mask:
            mask = get_mask_tensor()
            if mask is not None:
                mask_seq = mask.unsqueeze(0).unsqueeze(0).expand_as(seq_uint8)
                seq_uint8 = seq_uint8.masked_fill(~mask_seq, 0)

        seq_float = seq_uint8.float().div(255.0).mul(2.0).sub(1.0)

        # 3. Ground Truth (Baked vs Legacy)

        # 3. Ground Truth (Baked vs Legacy)
        gt_seq_display = []
        T = 18
        if is_baked_3d:
            # Baked: [N, 18, Dim]. Current frame has its whole future baked in.
            baked_seq = actions_tensor[idx] 
            limit = min(T, baked_seq.shape[0])
            for t in range(limit):
                gt_seq_display.append(_cache_vec_to_gt(baked_seq[t].tolist()))
        else:
            # Legacy: [N, Dim]. We must look ahead in the big tensor.
            for t in range(T):
                target_idx = min(idx + t, actions_tensor.shape[0] - 1)
                gt_seq_display.append(_cache_vec_to_gt(actions_tensor[target_idx].tolist()))

        # 4. Inference
        pred_seq_display = None
        pred_meta = {}
        pred_vecs = engine.infer(seq_float, model_key=model_key)
        
        if pred_vecs:
            pred_seq_display = [_policy_vec_to_display(v, old_layout=cfg["old_layout"]) for v in pred_vecs]
            pred_meta = {"model_used": model_key, "horizon": len(pred_vecs)}
        else:
            pred_meta = {"error": f"Model {model_key} not loaded"}

        return jsonify({
            "image": "data:image/png;base64," + b64_img,
            "ground_truth_seq": gt_seq_display,
            "ground_truth": gt_seq_display[0] if gt_seq_display else {},
            "prediction_seq": pred_seq_display,
            "prediction": pred_seq_display[0] if pred_seq_display else {},
            "pred_meta": pred_meta,
            "idx": idx, "is_baked": is_baked_3d
        })

    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/inspect/<path:filename>")
def inspect_cache(filename):
    return render_template("inspect_cache.html", filename=filename)

@app.route("/rl")
def view_rl_inspector():
    stats = {
        "total": len(RL_DATA_CACHE),
        "high_reward": len([x for x in RL_DATA_CACHE if x['weight'] > 1.0]),
        "punishment": len([x for x in RL_DATA_CACHE if x['weight'] < 1.0])
    }
    # Pass ui_buttons if needed for the template
    return render_template("rl_inspector.html", stats=stats, ui_buttons=GBA_UI_BUTTONS)

@app.route("/api/rl_list")
def api_rl_list():
    """Returns the lightweight list for the sidebar."""
    filter_type = request.args.get('filter', 'all') # all, high, low
    
    data = RL_DATA_CACHE
    if filter_type == 'high':
        data = [x for x in RL_DATA_CACHE if x['weight'] > 1.0]
    elif filter_type == 'low':
        data = [x for x in RL_DATA_CACHE if x['weight'] < 1.0]
        
    return jsonify(data[:2000])

@app.route("/api/rl_detail/<int:list_idx>")
def api_rl_detail(list_idx):
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
    # Use find_cache_info to locate the file in any cache dir
    pt_path, _ = find_cache_info(f"{replay_name}.pt")
    
    if pt_path:
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
                
                # Extract Actions (Handle Baked vs Legacy)
                acts = data['actions']
                is_baked = (acts.ndim == 3)
                
                if is_baked:
                    # Baked: 18 frames are pre-packaged at this index
                    baked = acts[frame_idx]
                    limit = min(18, baked.shape[0])
                    for t in range(limit):
                        action_seq.append(_policy_vec_to_display(baked[t].tolist(), old_layout=False))
                else:
                    # Legacy: Look ahead
                    end_idx = min(frame_idx + 18, acts.shape[0])
                    for t in range(frame_idx, end_idx):
                        action_seq.append(_policy_vec_to_display(acts[t].tolist(), old_layout=False))
                
                data_found = True
        except Exception as e:
            print(f"⚠️ Error reading PT file: {e}")

    # 3. STRATEGY B: FALLBACK TO RAW JSONL (If PT failed)
    if not data_found:
        jsonl_path = os.path.join(DATASET_DIR, replay_name, "actions.jsonl")
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, 'r') as f:
                    lines = f.readlines()
                
                if frame_idx < len(lines):
                    # No Tensor Image available in JSONL
                    b64_img = "" 
                    
                    end_idx = min(frame_idx + 18, len(lines))
                    
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