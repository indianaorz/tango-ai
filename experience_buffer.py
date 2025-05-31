# experience_buffer.py
import torch
import numpy as np
import random

class ExperienceBuffer:
    def __init__(self, buffer_size, mini_batch_size, num_game_features, frame_shape, gamma, gae_lambda, device):
        self.buffer_size = buffer_size # Typically num_steps_per_collect * num_envs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate memory for efficiency
        self.stacked_frames = torch.zeros((buffer_size, *frame_shape), dtype=torch.float32, device=device)
        self.game_features = torch.zeros((buffer_size, num_game_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size), dtype=torch.long, device=device) # Assuming discrete actions
        self.log_probs = torch.zeros((buffer_size), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size), dtype=torch.float32, device=device) # State values from critic
        self.dones = torch.zeros((buffer_size), dtype=torch.bool, device=device)    # Done flags

        self.ptr = 0 # Current position in buffer
        self.path_start_idx = 0 # For GAE calculation, marks start of current trajectory/episode segment
        self.is_full = False

    def add(self, stacked_frame, game_feature, action, log_prob, reward, value, done):
        if not isinstance(stacked_frame, torch.Tensor): stacked_frame = torch.tensor(stacked_frame, dtype=torch.float32)
        if not isinstance(game_feature, torch.Tensor): game_feature = torch.tensor(game_feature, dtype=torch.float32)
        if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.long)
        if not isinstance(log_prob, torch.Tensor): log_prob = torch.tensor(log_prob, dtype=torch.float32)
        if not isinstance(reward, (int, float, torch.Tensor)): reward = torch.tensor(reward, dtype=torch.float32)
        if not isinstance(value, torch.Tensor): value = torch.tensor(value, dtype=torch.float32)
        if not isinstance(done, (bool, torch.Tensor)): done = torch.tensor(done, dtype=torch.bool)

        self.stacked_frames[self.ptr] = stacked_frame.to(self.device)
        self.game_features[self.ptr] = game_feature.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards[self.ptr] = reward.to(self.device) if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = value.to(self.device)
        self.dones[self.ptr] = done.to(self.device) if isinstance(done, torch.Tensor) else torch.tensor(done, dtype=torch.bool, device=self.device)

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0 and not self.is_full:
            self.is_full = True # Buffer has been filled once
            print("Experience buffer is now full.")
            
    def _calculate_advantages_gae(self, last_value, last_done):
        """
        Computes advantages using Generalized Advantage Estimation (GAE).
        This should be called when a trajectory/collection phase ends.
        'last_value' is the critic's value estimate of the state *after* the last collected step.
        'last_done' indicates if that state was terminal.
        """
        num_samples = self.buffer_size if self.is_full else self.ptr
        advantages = torch.zeros(num_samples, dtype=torch.float32, device=self.device)
        gae = 0.0
        
        # If the buffer isn't full, we operate on the filled part
        # This logic assumes buffer is filled sequentially before GAE is computed on it.
        # For simplicity in PPO, usually GAE is computed over NUM_STEPS_PER_COLLECT.
        
        # Slicing the relevant parts of the buffer
        # Note: This assumes `finish_path` is called when `ptr` reaches `buffer_size` or an episode ends within it.
        rewards_segment = self.rewards[:num_samples]
        values_segment = self.values[:num_samples]
        dones_segment = self.dones[:num_samples]

        next_value = last_value
        next_done = last_done

        for t in reversed(range(num_samples)):
            if t == num_samples - 1: # Last step in the segment
                delta = rewards_segment[t] + self.gamma * next_value * (1.0 - next_done.float()) - values_segment[t]
            else:
                delta = rewards_segment[t] + self.gamma * values_segment[t+1] * (1.0 - dones_segment[t].float()) - values_segment[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_segment[t].float()) * gae
            advantages[t] = gae
            
        returns = advantages + values_segment # Q-values (target for value function)
        return advantages, returns

    def get_batches(self, last_value_for_gae, last_done_for_gae):
        """
        Calculates advantages and returns an iterator for mini-batches.
        Should be called after collecting `buffer_size` (or `num_steps_per_collect`) experiences.
        """
        num_samples = self.buffer_size if self.is_full else self.ptr
        if num_samples == 0:
            return iter([])

        advantages, returns = self._calculate_advantages_gae(last_value_for_gae, last_done_for_gae)
        
        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(num_samples)
        np.random.shuffle(indices) # Shuffle for training

        for start_idx in range(0, num_samples, self.mini_batch_size):
            batch_indices = indices[start_idx : start_idx + self.mini_batch_size]
            if len(batch_indices) < self.mini_batch_size and start_idx > 0 : # Drop last partial batch unless it's the only one
                continue

            yield (
                self.stacked_frames[batch_indices],
                self.game_features[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices], # Old log_probs from behavior policy
                advantages[batch_indices],
                returns[batch_indices],        # Targets for value function
                self.values[batch_indices]     # Old values from behavior policy
            )
        
        # After yielding all batches, reset pointer for next collection phase
        self.ptr = 0
        self.is_full = False # Buffer is now considered empty for new data