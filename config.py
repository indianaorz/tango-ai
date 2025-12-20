# config.py
# -----------------------------------------------------------------------------
# Tango AI – Unified configuration
# Centralizes runtime, environment, DRL, model, and reward settings.
# Keep public names consistent with other modules (network_handler, strategy, etc.)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
from typing import List, Dict
import torch



STRAT_DRL      = "drl"
STRAT_STAND    = "scripted_stand"
STRAT_WANDER   = "scripted_wander"
STRAT_CHARGE   = "scripted_charge"


# =============================================================================
# Paths
# =============================================================================

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
APP_PATH     = os.path.join(PROJECT_ROOT, "dist", "tango-x86_64-linux.AppImage")

# =============================================================================
# Instance orchestration (window/process management)
# =============================================================================

NUM_GAME_PAIRS: int = int(os.getenv("NUM_GAME_PAIRS", 1))   # number of learner/opponent pairs
BASE_PORT:      int = int(os.getenv("BASE_PORT", 12350))    # first TCP port to use

SAVE_PATH_TEMPLATE = "/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav"
ROM_PATH_DEFAULT   = "bn6,0"
INIT_CODE_DEFAULT  = "arena_drl"
ADDRESS_DEFAULT    = "127.0.0.1"



EARLY_STOP_ENABLED = bool(int(os.getenv("EARLY_STOP_ENABLED", "1")))
EARLY_STOP_HP      = int(os.getenv("EARLY_STOP_HP", "900"))
# "player"  → only this window’s player HP
# "either"  → either side’s HP (player OR enemy) triggers early stop
EARLY_STOP_WHOM    = os.getenv("EARLY_STOP_WHOM", "player").lower()  # "player" | "either"

# What to kill when triggered
EARLY_STOP_SCOPE   = os.getenv("EARLY_STOP_SCOPE", "pair").lower()   # "self" | "pair"


def generate_instances(n_pairs: int, base_port: int = BASE_PORT) -> List[Dict]:
    """Build a flat list of instance configs (learner/opponent alternating)."""
    inst: List[Dict] = []
    for i in range(n_pairs):
        learner_port  = base_port + i * 2
        opponent_port = learner_port + 1
        save_path     = SAVE_PATH_TEMPLATE.format(idx=i + 1)

        # Default: learner uses DRL; opponent uses curriculum for first two pairs
        if i % 2 == 0:
            opp_strategy = STRAT_CHARGE   # Game1 Opponent: charge & release
            # opp_strategy = STRAT_STAND     # Game1 Opponent: stand & shoot
        elif i % 2 == 1:
            opp_strategy = STRAT_WANDER    # Game2 Opponent: random walk
        else:
            opp_strategy = STRAT_DRL       # remaining pairs: self-play

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

# Export full instance plan (consumed by various modules)
INSTANCES: List[Dict] = generate_instances(NUM_GAME_PAIRS)

# =============================================================================
# Input mapping / discrete action space
# =============================================================================

KEY_BIT_POSITIONS = {
    # NOTE: Actual in-game bits we send are X/Z/RETURN + directions.
    # We keep A/B as logical names via DISCRETE_ACTIONS mapping below.
    "A": 8, "DOWN": 7, "UP": 6, "LEFT": 5, "RIGHT": 4,
    "RETURN": 3, "X": 1, "Z": 0,
}

# Logical buttons:
#   chip(A)  -> Z
#   shoot(B) -> X
BUTTON_ALIAS = {
    "A": "Z",
    "B": "X",
    "START": "RETURN",
}

# Minimal discrete actions for BN control
DISCRETE_ACTIONS = [
    "NO_OP",
    "UP", "DOWN", "LEFT", "RIGHT",
    "A", "B",
    "START",
    "UP_A", "DOWN_A", "LEFT_A", "RIGHT_A",
    "UP_B", "DOWN_B", "LEFT_B", "RIGHT_B",
]

# For SkipAndRandomStrategy / scripted randomness
RANDOM_ACTION_KEYS_FOR_SKIP_STRATEGY = ["B", "DOWN", "UP", "LEFT", "RIGHT", "A"]

# =============================================================================
# Nitrogen (ng.pt) policy config
# =============================================================================
NG_CKPT_PATH = os.getenv("NG_CKPT_PATH", os.path.join(PROJECT_ROOT, "weights", "ng.pt"))
USE_NG_POLICY = 1#bool(int(os.getenv("USE_NG_POLICY", "1")))  # 1 = use ng.pt inference for learner DRL policy
NG_DEVICE = os.getenv("NG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================
# Environment variables to pass to game processes
# =============================================================================

ENV_COMMON = os.environ.copy()
ENV_COMMON["INIT_LINK_CODE"] = "valuesearch"
ENV_COMMON["AI_MODEL_PATH"]  = "ai_model"

# =============================================================================
# Timing / networking
# =============================================================================

INFERENCE_FPS             = 60     # model decision rate per env (game can run faster)
CONNECTION_RETRY_DELAY    = 5
INSTANCE_INIT_WAIT_TIME   = 7
INSTANCE_STAGGER_TIME     = 0.1
GAME_RESTART_DELAY        = 5

# =============================================================================
# Device
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Model / observation settings
# =============================================================================
USE_IMAGES = bool(int(os.getenv("USE_IMAGES", "1")))  # 1 = use frames, 0 = state-only

# Image frames (per observation sequence)
FRAME_HEIGHT   = 84
FRAME_WIDTH    = 84
FRAME_CHANNELS = 3

# Temporal window (sequence length)
SEQ_LEN_FRAMES = 48

# Legacy name still referenced in some places (equals channels per frame)
NUM_FRAMES_STACKED = FRAME_CHANNELS

# ==== Model capacity knobs ====
RNN_HIDDEN   = int(os.getenv("RNN_HIDDEN", 128))   # was 768
RNN_LAYERS   = int(os.getenv("RNN_LAYERS", 2))      # new: multi-layer GRU/LSTM
RNN_TYPE     = os.getenv("RNN_TYPE", "gru")         # "gru" or "lstm"
FUSE_DIM     = int(os.getenv("FUSE_DIM", 0))     # linear proj before RNN (0 to disable)
DROPOUT_RNN  = float(os.getenv("DROPOUT_RNN", 0.0)) # dropout between RNN layers

# CNN layout is now configurable
CNN_CHANNELS = tuple(int(x) for x in os.getenv("CNN_CHANNELS", "64,128,256,256").split(","))
CNN_KERNELS  = tuple(int(x) for x in os.getenv("CNN_KERNELS",  "8,4,3,3").split(","))
CNN_STRIDES  = tuple(int(x) for x in os.getenv("CNN_STRIDES",  "4,2,1,1").split(","))

# AMP + buffer placement (to free VRAM for a bigger model)
USE_AMP              = bool(int(os.getenv("USE_AMP", "1")))   # autocast + GradScaler
EXPERIENCE_ON_DEVICE = os.getenv("EXPERIENCE_ON_DEVICE", "cpu")  # "cpu" or "cuda"


# =============================================================================
# PPO / RL hyperparameters
# =============================================================================

LEARNING_RATE      = 3e-4
GAMMA              = 0.99
GAE_LAMBDA         = 0.95
PPO_CLIP_EPSILON   = 0.2
PPO_EPOCHS         = 4
MINI_BATCH_SIZE    = 32

# Steps collected across all envs before each update
NUM_STEPS_PER_COLLECT = 1024

# Experience buffer sizing (kept compatible with ExperienceBuffer expectations)
EXPERIENCE_BUFFER_SIZE    = NUM_STEPS_PER_COLLECT
EXPERIENCE_BUFFER_SECONDS = NUM_STEPS_PER_COLLECT // INFERENCE_FPS  # derived convenience

# Training schedule
MAX_TRAINING_STEPS = 1_000_000_000_000
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5

# =============================================================================
# Normalization constants for scalar features
# =============================================================================

MAX_HEALTH         = 2000.0
MAX_CHARGE_LEVEL   = 2.0
MAX_CUST_GAUGE_VALUE = 255.0

# =============================================================================
# Logging / checkpoints
# =============================================================================

TENSORBOARD_LOG_DIR   = os.path.join(PROJECT_ROOT, "runs")
MODEL_SAVE_DIR        = os.path.join(PROJECT_ROOT, "trained_models")
MODEL_SAVE_FREQUENCY  = 1  # save every N updates

# =============================================================================
# Reward shaping
# =============================================================================

# Damage
REWARD_DAMAGE_DEALT_MULTIPLIER = +1.0   # per HP dealt
REWARD_DAMAGE_TAKEN_MULTIPLIER = -0.20   # per HP taken

# Episode result
REWARD_WIN_GAME  = 0
REWARD_LOSE_GAME = 0

# Time pressure
REWARD_TIME_PENALTY_STEP = -0.0005

# Charge behavior
REWARD_CHARGE_GAIN_COEF     = 0.00
CHARGE_MAX_LEVEL            = 3
REWARD_CHARGE_RELEASE_BONUS = 5.0

# =============================================================================
# Optional: small helper for human-readable summary (not used by runtime)
# =============================================================================

def summary() -> str:
    return (
        "Tango AI Config\n"
        f" • Device: {DEVICE}\n"
        f" • Game pairs: {NUM_GAME_PAIRS} (windows: {NUM_GAME_PAIRS*2}) starting at {BASE_PORT}\n"
        f" • Obs: {SEQ_LEN_FRAMES}×({FRAME_CHANNELS}×{FRAME_HEIGHT}×{FRAME_WIDTH})  |  RNN_HIDDEN={RNN_HIDDEN}\n"
        f" • PPO: steps/collect={NUM_STEPS_PER_COLLECT}, minibatch={MINI_BATCH_SIZE}, epochs={PPO_EPOCHS}\n"
        f" • Buffer: size={EXPERIENCE_BUFFER_SIZE} (~{EXPERIENCE_BUFFER_SECONDS}s @ {INFERENCE_FPS} fps)\n"
    )
