# ── Begin: config.py ──
# config.py
# -----------------------------------------------------------------------------
# Tango AI – Unified configuration
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
from typing import List, Dict
import torch

STRAT_DRL       = "drl"
STRAT_STAND     = "scripted_stand"
STRAT_WANDER    = "scripted_wander"
STRAT_CHARGE    = "scripted_charge"

# =============================================================================
# Paths
# =============================================================================

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
APP_PATH     = os.path.join(PROJECT_ROOT, "dist", "tango-x86_64-linux.AppImage")

CRITIC_CKPT_PATH os.path.join(PROJECT_ROOT, "checkpoints", "critic_rl", "tdlam_complete_v5_final", "last.pt")
# =============================================================================
# Instance orchestration
# =============================================================================

NUM_GAME_PAIRS: int = int(os.getenv("NUM_GAME_PAIRS", 1))
BASE_PORT:      int = int(os.getenv("BASE_PORT", 12350))

SAVE_PATH_TEMPLATE = "/home/lee/Documents/Tango/saves/BN6 Gregar {idx}.sav"
ROM_PATH_DEFAULT   = "bn6,0"
INIT_CODE_DEFAULT  = "arena_drl"
ADDRESS_DEFAULT    = "127.0.0.1"

EARLY_STOP_ENABLED = bool(int(os.getenv("EARLY_STOP_ENABLED", "1")))
EARLY_STOP_HP      = int(os.getenv("EARLY_STOP_HP", "900"))
EARLY_STOP_WHOM    = os.getenv("EARLY_STOP_WHOM", "player").lower()
EARLY_STOP_SCOPE   = os.getenv("EARLY_STOP_SCOPE", "pair").lower()

def generate_instances(n_pairs: int, base_port: int = BASE_PORT) -> List[Dict]:
    inst: List[Dict] = []
    for i in range(n_pairs):
        learner_port  = base_port + i * 2
        opponent_port = learner_port + 1
        # save_path     = SAVE_PATH_TEMPLATE.format(idx=i + 1)
        save_path     = "/home/lee/Documents/Tango/saves/BN6_JCode.sav"

        # Default: learner uses DRL; opponent uses curriculum
        if i % 2 == 0:
            opp_strategy = STRAT_DRL   # Game1 Opponent
        elif i % 2 == 1:
            opp_strategy = STRAT_DRL   # Game2 Opponent
        else:
            opp_strategy = STRAT_DRL      # remaining pairs: self-play

        inst.extend([
            {
                "address": ADDRESS_DEFAULT,
                "port": learner_port,
                "rom_path": ROM_PATH_DEFAULT,
                "save_path": save_path,
                "name": f"Game{i+1} Learner",
                "init_link_code": f"{INIT_CODE_DEFAULT}-{i}",
                "strategy": STRAT_DRL,
            },
            {
                "address": ADDRESS_DEFAULT,
                "port": opponent_port,
                "rom_path": ROM_PATH_DEFAULT,
                "save_path": save_path,
                "name": f"Game{i+1} Opponent",
                "init_link_code": f"{INIT_CODE_DEFAULT}-{i}",
                "strategy": opp_strategy,
            },
        ])
    return inst

INSTANCES: List[Dict] = generate_instances(NUM_GAME_PAIRS)

# =============================================================================
# Input mapping
# =============================================================================

KEY_BIT_POSITIONS = {
    "A": 8, "DOWN": 7, "UP": 6, "LEFT": 5, "RIGHT": 4,
    "RETURN": 3, "X": 1, "Z": 0,
}
BUTTON_ALIAS = { "A": "Z", "B": "X", "START": "RETURN" }

DISCRETE_ACTIONS = [
    "NO_OP", "UP", "DOWN", "LEFT", "RIGHT",
    "A", "B", "START",
    "UP_A", "DOWN_A", "LEFT_A", "RIGHT_A",
    "UP_B", "DOWN_B", "LEFT_B", "RIGHT_B",
]
RANDOM_ACTION_KEYS_FOR_SKIP_STRATEGY = ["B", "DOWN", "UP", "LEFT", "RIGHT", "A"]

_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
_BATTLE_DIR = os.path.join(_CKPT_DIR, "nitrogen_battle_critic")
_PLAN_DIR = os.path.join(_CKPT_DIR, "planning")

def _get_latest_checkpoint(ckpt_dir: str, default: str = "ng.pt") -> str:
    """Finds the checkpoint with the highest step count in the directory."""
    if not os.path.exists(ckpt_dir):
        # Fallback to root weights if subfolder missing
        return os.path.join(PROJECT_ROOT, "weights", default)
    
    max_step = -1
    best_ckpt = default
    found = False
    
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("step_") and fname.endswith(".pt"):
            try:
                step = int(fname.split("_")[1].split(".")[0])
                if step > max_step:
                    max_step = step
                    best_ckpt = fname
                    found = True
            except: continue
            
    if not found:
         return os.path.join(PROJECT_ROOT, "weights", default)

    return os.path.join(ckpt_dir, best_ckpt)

# Define paths for both models
BATTLE_CKPT_PATH = os.getenv("BATTLE_CKPT_PATH", _get_latest_checkpoint(_BATTLE_DIR, "step_150000.pt"))
# PLAN_CKPT_PATH = os.getenv("PLAN_CKPT_PATH", _get_latest_checkpoint(_PLAN_DIR, "step_150000.pt"))

# NEW: Point to the Strategy Transformer
PLAN_CKPT_PATH = os.getenv("PLAN_CKPT_PATH", "checkpoints_strategy/strategy_model.pt") 
# NEW: Point to the Chips Database
CHIPS_DB_PATH = os.getenv("CHIPS_DB_PATH", "data/assets/chips.json")

# Keep NG_CKPT_PATH for backward compat if needed, but point it to battle
NG_CKPT_PATH = BATTLE_CKPT_PATH
# NG_CKPT_PATH = os.getenv("NG_CKPT_PATH", os.path.join(PROJECT_ROOT, "weights", "ng.pt"))
USE_NG_POLICY = 1
NG_DEVICE = os.getenv("NG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# NEW: Use FP16 for speed on TensorCores
NG_USE_FP16 = bool(int(os.getenv("NG_USE_FP16", "1"))) 

# =============================================================================
# Environment variables
# =============================================================================
ENV_COMMON = os.environ.copy()
ENV_COMMON["INIT_LINK_CODE"] = "valuesearch"
ENV_COMMON["AI_MODEL_PATH"]  = "ai_model"

# =============================================================================
# Timing / networking
# =============================================================================
INFERENCE_FPS           = 60
CONNECTION_RETRY_DELAY  = 5
INSTANCE_INIT_WAIT_TIME = 7
INSTANCE_STAGGER_TIME   = 0.1
GAME_RESTART_DELAY      = 5

# =============================================================================
# Device & Model Settings
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_IMAGES = bool(int(os.getenv("USE_IMAGES", "1")))

FRAME_HEIGHT   = 256
FRAME_WIDTH    = 256
FRAME_CHANNELS = 3

# === CRITICAL PERFORMANCE FIX ===
# Reduced from 48 to 1 to prevent massive token sequence generation.
# This forces the model to look at the current frame only (Stateless).
SEQ_LEN_FRAMES = 1 

# Legacy params
NUM_FRAMES_STACKED = FRAME_CHANNELS
RNN_HIDDEN   = int(os.getenv("RNN_HIDDEN", 128))
RNN_LAYERS   = int(os.getenv("RNN_LAYERS", 2))
RNN_TYPE     = os.getenv("RNN_TYPE", "gru")
FUSE_DIM     = int(os.getenv("FUSE_DIM", 0))
DROPOUT_RNN  = float(os.getenv("DROPOUT_RNN", 0.0))
CNN_CHANNELS = tuple(int(x) for x in os.getenv("CNN_CHANNELS", "64,128,256,256").split(","))
CNN_KERNELS  = tuple(int(x) for x in os.getenv("CNN_KERNELS",  "8,4,3,3").split(","))
CNN_STRIDES  = tuple(int(x) for x in os.getenv("CNN_STRIDES",  "4,2,1,1").split(","))
USE_AMP      = bool(int(os.getenv("USE_AMP", "1")))
EXPERIENCE_ON_DEVICE = os.getenv("EXPERIENCE_ON_DEVICE", "cpu")

# =============================================================================
# PPO / RL hyperparameters
# =============================================================================
LEARNING_RATE      = 3e-4
GAMMA              = 0.99
GAE_LAMBDA         = 0.95
PPO_CLIP_EPSILON   = 0.2
PPO_EPOCHS         = 4
MINI_BATCH_SIZE    = 32
NUM_STEPS_PER_COLLECT = 1024
EXPERIENCE_BUFFER_SIZE    = NUM_STEPS_PER_COLLECT
EXPERIENCE_BUFFER_SECONDS = NUM_STEPS_PER_COLLECT // INFERENCE_FPS
MAX_TRAINING_STEPS = 1_000_000_000_000
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5
MAX_HEALTH         = 2000.0
MAX_CHARGE_LEVEL   = 2.0
MAX_CUST_GAUGE_VALUE = 255.0

# =============================================================================
# Logging / checkpoints / Rewards
# =============================================================================
TENSORBOARD_LOG_DIR   = os.path.join(PROJECT_ROOT, "runs")
MODEL_SAVE_DIR        = os.path.join(PROJECT_ROOT, "trained_models")
MODEL_SAVE_FREQUENCY  = 1
REWARD_DAMAGE_DEALT_MULTIPLIER = +1.0
REWARD_DAMAGE_TAKEN_MULTIPLIER = -0.20
REWARD_WIN_GAME  = 0
REWARD_LOSE_GAME = 0
REWARD_TIME_PENALTY_STEP = -0.0005
REWARD_CHARGE_GAIN_COEF      = 0.00
CHARGE_MAX_LEVEL             = 3
REWARD_CHARGE_RELEASE_BONUS = 5.0

def summary() -> str:
    return (
        "Tango AI Config\n"
        f" • Device: {DEVICE} (NG: {NG_DEVICE}, FP16={NG_USE_FP16})\n"
        f" • Game pairs: {NUM_GAME_PAIRS} starting at {BASE_PORT}\n"
        f" • Obs: {SEQ_LEN_FRAMES} frames (stateless optimized)\n"
    )
# ── End: config.py ──