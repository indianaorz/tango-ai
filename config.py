# config.py
import os
import torch
from typing import List, Dict

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
APP_PATH = os.path.join(PROJECT_ROOT, "dist/tango-x86_64-linux.AppImage")

# ── Dynamic instance generation  ──────────────────────────────────────────────
NUM_GAME_PAIRS: int = int(os.getenv("NUM_GAME_PAIRS", 2))   # ← set env var or edit
BASE_PORT:      int = int(os.getenv("BASE_PORT", 12340))    # first port to use

SAVE_PATH_TEMPLATE = "/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav"
ROM_PATH_DEFAULT   = "bn6,0"
INIT_CODE_DEFAULT  = "arena_drl"
ADDRESS_DEFAULT    = "127.0.0.1"

def generate_instances(n_pairs: int,
                       base_port: int = BASE_PORT) -> List[Dict]:
    """Return a list of dicts compatible with GameManager for n learner/opponent pairs."""
    inst: List[Dict] = []
    for i in range(n_pairs):
        learner_port   = base_port + i * 2
        opponent_port  = learner_port + 1
        save_path      = SAVE_PATH_TEMPLATE.format(idx=i + 1)

        inst.extend([
            {
                "address": ADDRESS_DEFAULT,
                "port": learner_port,
                "rom_path": ROM_PATH_DEFAULT,
                "save_path": save_path,
                "name": f"Game{i+1} Learner",
                "init_link_code": f"{INIT_CODE_DEFAULT}-{i}",
            },
            {
                "address": ADDRESS_DEFAULT,
                "port": opponent_port,
                "rom_path": ROM_PATH_DEFAULT,
                "save_path": save_path,
                "name": f"Game{i+1} Opponent",
                "init_link_code": f"{INIT_CODE_DEFAULT}-{i}",
            },
        ])
    return inst

INSTANCES: List[Dict] = generate_instances(NUM_GAME_PAIRS)


# --- Key Mappings (same as before) ---
KEY_BIT_POSITIONS = {
    'A': 8, 'DOWN': 7, 'UP': 6, 'LEFT': 5, 'RIGHT': 4,
    'RETURN': 3, 'X': 1, 'Z': 0,
}
# For DRL, we define a fixed discrete action space the AI will choose from
# These are conceptual actions the AI outputs; mapping to button presses happens later.
# Order matters for the model's output layer.
DISCRETE_ACTIONS = [
    "NO_OP",        # 0000000000000000
    "UP",           # 0000000001000000
    "DOWN",         # 0000000010000000
    "LEFT",         # 0000000000100000
    "RIGHT",        # 0000000000010000
    "X",            # 0000000000000010 (Buster / Primary action)
    "Z",            # 0000000000000001 (Chip use / Secondary action)
    "LEFT_X",       # Left + Buster
    "RIGHT_X",      # Right + Buster
    "UP_X",         # Up + Buster
    "DOWN_X",       # Down + Buster
    "HOLD_X",
    "HOLD_Z",      
    "RELEASE_X","RELEASE_Z", # Holding X or Z (Buster or Chip) for a longer time
    # Add more combinations or special moves as the action space evolves
    # e.g., "SHIELD" (BACK + X) might be a specific action if desired.
]
# For SkipAndRandomStrategy
RANDOM_ACTION_KEYS_FOR_SKIP_STRATEGY = ['Z', 'DOWN', 'UP', 'LEFT', 'RIGHT', 'X']


ENV_COMMON = os.environ.copy()
ENV_COMMON["INIT_LINK_CODE"] = "valuesearch"
ENV_COMMON["AI_MODEL_PATH"] = "ai_model"

# --- Timing and Network (mostly same) ---
INFERENCE_FPS = 20 # AI might not need to react at 60fps initially. Game runs faster.
CONNECTION_RETRY_DELAY = 5
INSTANCE_INIT_WAIT_TIME = 7 # Increased slightly for stability
INSTANCE_STAGGER_TIME = 0.1
GAME_RESTART_DELAY = 5 # Seconds to wait before restarting game instances after they all terminate


# --- Experience‑buffer sizing (NEW) ------------------------------------------
# How many seconds of gameplay each environment should retain before we start
# an update.  We derive a per‑env step count from INFERENCE_FPS so the time
# window scales automatically if you speed the bot up or slow it down.
# EXPERIENCE_BUFFER_SECONDS = int(os.getenv("EXPERIENCE_BUFFER_SECONDS", 10))  # default 10 s

# # Steps per *environment* for that time window
# EXPERIENCE_STEPS_PER_ENV = EXPERIENCE_BUFFER_SECONDS * INFERENCE_FPS          # e.g. 10 s × 20 fps = 200

# # Total capacity for ExperienceBuffer (all envs combined)
# EXPERIENCE_BUFFER_SIZE = EXPERIENCE_STEPS_PER_ENV * len(INSTANCES)



# print(f"Experience buffer size: {EXPERIENCE_BUFFER_SIZE} steps ({EXPERIENCE_BUFFER_SECONDS}s at {INFERENCE_FPS} FPS)")

# --- DRL Hyperparameters (NEW) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor for future rewards
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
PPO_CLIP_EPSILON = 0.2 # PPO clipping parameter
PPO_EPOCHS = 10 # Number of epochs to train on a batch of experience
MINI_BATCH_SIZE = 64
NUM_STEPS_PER_COLLECT = 2048 # Steps collected from all envs before an update

EXPERIENCE_BUFFER_SIZE = NUM_STEPS_PER_COLLECT
# or, if you really want secs × FPS:
EXPERIENCE_BUFFER_SECONDS = NUM_STEPS_PER_COLLECT // INFERENCE_FPS


MAX_TRAINING_STEPS = 1_000_000_000 # Total training steps for the experiment
ENTROPY_COEF = 0.01 # Coefficient for entropy bonus (encourages exploration)
VALUE_LOSS_COEF = 0.5 # Coefficient for value function loss
# --- Normalization Constants for DRL Agent features ---
MAX_HEALTH = 2000.0  # Typical max HP in BN6, adjust as needed
MAX_CHARGE_LEVEL = 3.0 # Max value for player_charge / enemy_charge
MAX_CUST_GAUGE_VALUE = 255.0 # Max value for cust_gage

# --- Image / temporal window -----------------------------------------------
FRAME_HEIGHT        = 84
FRAME_WIDTH         = 84

SEQ_LEN_FRAMES      = 32        # 40 frames ≈ 2.0 s at 20 FPS
# FRAME_CHANNELS      = 1         # we convert to grayscale
FRAME_CHANNELS = 3
# keep this for legacy code that still references NUM_FRAMES_STACKED,
# but it’s now “frames per element in the sequence”, not the whole history
NUM_FRAMES_STACKED  = FRAME_CHANNELS
# --- Image Preprocessing (NEW) ---
# --- Logging & Saving (NEW) ---
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, "runs")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "trained_models")
MODEL_SAVE_FREQUENCY = 1 # Save model every N updates

# --- Reward Coefficients (charging + buster) --------------------------------
# Encourage: land damage; penalize getting hit; small time pressure;
# charge gradually; big pop on a released max-charge that actually hits.
REWARD_DAMAGE_DEALT_MULTIPLIER = 0.20   # +0.20 per HP dealt
REWARD_DAMAGE_TAKEN_MULTIPLIER = -0.20  # -0.20 per HP taken
REWARD_WIN_GAME = +1.0
REWARD_LOSE_GAME = -1.0
REWARD_TIME_PENALTY_STEP = -0.002

REWARD_CHARGE_GAIN_COEF     = 0.02   # slow drip while bar fills
CHARGE_MAX_LEVEL            = 3
REWARD_CHARGE_RELEASE_BONUS = 5.0    # big payoff for releasing a full shot that hits
