# train_agent.py

import subprocess
import os
import time
import asyncio
from multiprocessing import Process
from stable_baselines3 import PPO
from game_env import GameEnv
from event_listener import event_listener
from capture_utils import find_game_window, get_window_geometry, send_input_command

# Path to the Tango AppImage
APP_PATH = ".././dist/tango-x86_64-linux.AppImage"

# Common environment variables
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "your_link_code"
env_common["AI_MODEL_PATH"] = "ai_model"
env_common["MATCHMAKING_ID"] = "your_matchmaking_id"

# Define server addresses and ports for each instance
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12345,
        'rom_path': 'bn6,0',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar.sav',
        'name': 'Instance 1',
    },
    {
        'address': '127.0.0.1',
        'port': 12346,
        'rom_path': 'bn6,1',
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Falzar 1.sav',
        'name': 'Instance 2',
    },
]

def run_instance(rom_path, save_path, port):
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["PORT"] = str(port)

    print(f"Running instance with ROM_PATH: {rom_path}, SAVE_PATH: {save_path}, PORT: {port}")
    subprocess.Popen([APP_PATH], env=env)

def start_instances():
    for instance in INSTANCES:
        run_instance(instance['rom_path'], instance['save_path'], instance['port'])
        time.sleep(0.5)

def get_unique_event_port():
    # Return a unique port for event communication; you can increment based on instance['port']
    return 16000 + random.randint(0, 1000)

def start_training(instance):
    env = GameEnv(instance)
    instance['env'] = env
    model = PPO('CnnPolicy', env, verbose=1, device='cuda')
    model.learn(total_timesteps=100000)
    model.save(f"game_agent_{instance['port']}")
    env.close()

async def main():
    start_instances()
    print("Both instances are running.")
    await asyncio.sleep(5)

    # Find windows and get geometries
    for instance in INSTANCES:
        window_id = find_game_window(instance['port'])
        instance['window_id'] = window_id
        instance['geometry'] = get_window_geometry(window_id)
        instance['event_port'] = get_unique_event_port()

    # Start event listeners
    event_tasks = [event_listener(instance) for instance in INSTANCES]

    # Start training processes
    processes = []
    for instance in INSTANCES:
        p = Process(target=start_training, args=(instance,))
        p.start()
        processes.append(p)

    # Run event listeners
    await asyncio.gather(*event_tasks)

    # Wait for training processes to finish
    for p in processes:
        p.join()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
