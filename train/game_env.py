# game_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import asyncio
from capture_utils import capture_window, preprocess_image, send_input_command

class GameEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, instance):
        super(GameEnv, self).__init__()
        self.instance = instance
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(7)  # Number of possible actions
        self.current_observation = None
        self.done = False
        self.info = {}
        self.reward = 0
        self.previous_health = None  # To track damage dealt and received
        self.damage_dealt = 0
        self.damage_received = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.reward = 0
        self.damage_dealt = 0
        self.damage_received = 0
        # Capture the initial observation
        self.current_observation = self.get_observation()
        return self.current_observation, self.info
    
    def step(self, action):
        # Send the action to the game instance
        asyncio.run(self.send_action(action))
        
        # Wait for the next frame
        asyncio.run(asyncio.sleep(0.1))
        
        # Get the new observation
        self.current_observation = self.get_observation()
        
        # Compute the reward
        self.reward = self.compute_reward()
        
        # Check if the game is over
        self.done = self.check_done()
        
        return self.current_observation, self.reward, self.done, False, self.info
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    async def send_action(self, action):
        action_command = self.map_action(action)
        await send_input_command(self.instance['writer'], action_command)
    
    def get_observation(self):
        image = capture_window(self.instance['geometry'])
        return preprocess_image(image)
    
    def compute_reward(self):
        reward = self.damage_dealt - self.damage_received
        self.damage_dealt = 0
        self.damage_received = 0
        return reward
    
    def check_done(self):
        return self.done
    
    def map_action(self, action_index):
        action_mapping = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'A']
        action = action_mapping[action_index]
        return {'type': 'key_press', 'key': action}
