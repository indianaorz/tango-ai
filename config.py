# config.py
import os
import torch

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
APP_PATH = os.path.join(PROJECT_ROOT, "dist/tango-x86_64-linux.AppImage")

# --- Instance Configuration (same as before) ---
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav', # Replace
        'name': 'Instance 1 (Learner)',
        'init_link_code': 'arena_drl',
    },
    {
        'address': '127.0.0.1',
        'port': 12345,
        'rom_path': 'bn6,0',
        # 'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 2.sav', # Replace
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav', # Replace
        'name': 'Instance 2 (Opponent)',
        'init_link_code': 'arena_drl',
    },
]

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
    "A",            # 0000000100000000 (Chip use / Secondary action)
    "LEFT_X",       # Left + Buster
    "RIGHT_X",      # Right + Buster
    "UP_X",         # Up + Buster
    "DOWN_X",       # Down + Buster
    # Add more combinations or special moves as the action space evolves
    # e.g., "SHIELD" (BACK + X) might be a specific action if desired.
]
# For SkipAndRandomStrategy
RANDOM_ACTION_KEYS_FOR_SKIP_STRATEGY = ['A', 'DOWN', 'UP', 'LEFT', 'RIGHT', 'X']


ENV_COMMON = os.environ.copy()
ENV_COMMON["INIT_LINK_CODE"] = "valuesearch"
ENV_COMMON["AI_MODEL_PATH"] = "ai_model"

# --- Timing and Network (mostly same) ---
INFERENCE_FPS = 20 # AI might not need to react at 60fps initially. Game runs faster.
CONNECTION_RETRY_DELAY = 5
INSTANCE_INIT_WAIT_TIME = 7 # Increased slightly for stability
INSTANCE_STAGGER_TIME = 1.5
GAME_RESTART_DELAY = 5 # Seconds to wait before restarting game instances after they all terminate

# --- DRL Hyperparameters (NEW) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor for future rewards
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
PPO_CLIP_EPSILON = 0.2 # PPO clipping parameter
PPO_EPOCHS = 10 # Number of epochs to train on a batch of experience
MINI_BATCH_SIZE = 64
NUM_STEPS_PER_COLLECT = 2048 # Steps collected from all envs before an update
MAX_TRAINING_STEPS = 1_000_000 # Total training steps for the experiment
ENTROPY_COEF = 0.01 # Coefficient for entropy bonus (encourages exploration)
VALUE_LOSS_COEF = 0.5 # Coefficient for value function loss
# --- Normalization Constants for DRL Agent features ---
MAX_HEALTH = 2000.0  # Typical max HP in BN6, adjust as needed
MAX_CHARGE_LEVEL = 3.0 # Max value for player_charge / enemy_charge
MAX_CUST_GAUGE_VALUE = 255.0 # Max value for cust_gage

# --- Image Preprocessing (NEW) ---
FRAME_HEIGHT = 84 # Standard for DRL from pixels
FRAME_WIDTH = 84
NUM_FRAMES_STACKED = 4 # Number of recent frames to stack for temporal context

# --- Logging & Saving (NEW) ---
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, "runs")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "trained_models")
MODEL_SAVE_FREQUENCY = 1 # Save model every N updates

# --- Reward Coefficients (NEW - for reward shaping) ---
# These are initial suggestions, will need tuning.
REWARD_DAMAGE_DEALT_MULTIPLIER = 0.1  # Per point of damage
REWARD_DAMAGE_TAKEN_MULTIPLIER = -0.1 # Per point of damage
REWARD_WIN_GAME = 10.0
REWARD_LOSE_GAME = -10.0
REWARD_TIME_PENALTY_STEP = -0.001 # Small penalty per step to encourage faster games (optional)
# Add more for charge shots, shields etc. if you can detect them