from __future__ import annotations
import os
from typing import List, Dict

def get_project_root() -> str:
    import os as _os
    return _os.path.dirname(_os.path.abspath(__file__))

PROJECT_ROOT = os.path.join(get_project_root(), '..')
APP_PATH     = os.path.join(PROJECT_ROOT, "dist", "tango-x86_64-linux.AppImage")

NUM_GAME_PAIRS: int = int(os.getenv("NUM_GAME_PAIRS", 1))
BASE_PORT:      int = int(os.getenv("BASE_PORT", 12340))

SAVE_PATH_TEMPLATE = "/home/lee/Documents/Tango/saves/BN6 Gregar {idx}.sav"
ROM_PATH_DEFAULT   = "bn6,0"
INIT_CODE_DEFAULT  = "arena_runtime"
ADDRESS_DEFAULT    = "127.0.0.1"

def generate_instances(n_pairs: int, base_port: int = BASE_PORT) -> List[Dict]:
    inst: List[Dict] = []
    for i in range(n_pairs):
        learner_port  = base_port + i * 2
        opponent_port = learner_port + 1
        save_path     = SAVE_PATH_TEMPLATE.format(idx=i + 1)
        inst.extend([
            {"address": ADDRESS_DEFAULT, "port": learner_port,  "rom_path": ROM_PATH_DEFAULT,
             "save_path": save_path, "name": f"Game{i+1} Learner",  "init_link_code": f"{INIT_CODE_DEFAULT}-{i}"},
            {"address": ADDRESS_DEFAULT, "port": opponent_port, "rom_path": ROM_PATH_DEFAULT,
             "save_path": save_path, "name": f"Game{i+1} Opponent", "init_link_code": f"{INIT_CODE_DEFAULT}-{i}"},
        ])
    return inst

INSTANCES: List[Dict] = generate_instances(NUM_GAME_PAIRS)

ENV_COMMON = os.environ.copy()
ENV_COMMON["INIT_LINK_CODE"] = "valuesearch"
ENV_COMMON["AI_MODEL_PATH"]  = "ai_model"

CONTROL_FPS            = int(os.getenv("CONTROL_FPS", "20"))
CONNECTION_RETRY_DELAY = 5
INSTANCE_INIT_WAIT_TIME= 7
INSTANCE_STAGGER_TIME  = 0.1
GAME_RESTART_DELAY     = 5

FRAME_HEIGHT   = 96
FRAME_WIDTH    = 96
FRAME_CHANNELS = 3
SEQ_LEN_FRAMES = 48

LOG_DIR = os.path.join(PROJECT_ROOT, "runs")
