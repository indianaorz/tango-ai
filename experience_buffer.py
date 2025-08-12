import torch
import numpy as np

class ExperienceBuffer:
    """
    Stores (frames-seq, features-seq, action, log_prob, reward, value, done)
    for PPO.

    Each entry:
        stacked_frames : [T, C, H, W]   (sequence of frames for s_t)
        game_features  : [T, F]         (sequence of features for s_t)
        action         : int            (a_t sampled under behavior policy)
        log_prob       : float          (log π_old(a_t|s_t))
        reward         : float          (r_t)
        value          : float          (V_old(s_t))
        done           : bool           (episode terminal at s_{t+1})
    """
    def __init__(self, buffer_size, mini_batch_size, num_game_features,
                 frame_shape, gamma, gae_lambda, device):

        self.buffer_size      = buffer_size
        self.mini_batch_size  = mini_batch_size
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.device           = device

        T, C, H, W            = frame_shape  # NOTE: T == seq_len

        # pre-allocate contiguous tensors on device
        self.stacked_frames = torch.zeros(
            (buffer_size, T, C, H, W), dtype=torch.float32, device=device)
        self.game_features  = torch.zeros(
            (buffer_size, T, num_game_features), dtype=torch.float32, device=device)

        self.actions   = torch.zeros(buffer_size, dtype=torch.long,    device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards   = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values    = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones     = torch.zeros(buffer_size, dtype=torch.bool,    device=device)

        self.ptr = 0
        self.is_full = False

    def add(self, stacked_frame, game_feature, action, log_prob, reward, value, done):
        """
        stacked_frame : [1, T, C, H, W] or [T, C, H, W]
        game_feature  : [1, T, F]       or [T, F]
        """
        sf = stacked_frame.squeeze(0) if isinstance(stacked_frame, torch.Tensor) \
             else torch.tensor(stacked_frame, dtype=torch.float32, device=self.device).squeeze(0)
        gf = game_feature.squeeze(0) if isinstance(game_feature, torch.Tensor) \
             else torch.tensor(game_feature, dtype=torch.float32, device=self.device).squeeze(0)

        self.stacked_frames[self.ptr] = sf.to(self.device, non_blocking=True)
        self.game_features [self.ptr] = gf.to(self.device, non_blocking=True)

        self.actions  [self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards  [self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values   [self.ptr] = value.to(self.device)
        self.dones    [self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0 and not self.is_full:
            self.is_full = True
            print("Experience buffer is now full.")

    def _calculate_advantages_gae(self, last_value_bootstrap: torch.Tensor):
        """
        GAE(λ) with correct s_{t+1} bootstrap:
          δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
          Adv_t computed backwards; Returns_t = Adv_t + V_t

        `last_value_bootstrap` is V(s_{N}) where N is one step *after* the
        last collected transition in the slice. This is used only for the
        final step that doesn’t have V_{t+1} inside the slice.
        """
        n = self.buffer_size if self.is_full else self.ptr
        rewards = self.rewards[:n]
        values  = self.values[:n]
        dones   = self.dones [:n].float()

        # V_{t+1}: shift left and append the provided bootstrap
        next_values = torch.empty_like(values)
        if n > 1:
            next_values[:-1] = values[1:]
        next_values[-1] = last_value_bootstrap

        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in reversed(range(n)):
            nonterminal = (1.0 - dones[t])
            gae = deltas[t] + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return n, advantages, returns

    def get_batches(self, last_value_bootstrap: torch.Tensor):
        """
        Compute GAE using a *single* bootstrap value for the end of the slice.
        This is correct if:
          • `dones` is accurate per-step, and
          • last_value_bootstrap ≈ V(s_{t_last+1}) of the slice.
        """
        n, advantages, returns = self._calculate_advantages_gae(last_value_bootstrap)

        # Normalize adv for PPO stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, self.mini_batch_size):
            batch_idx = indices[start:start + self.mini_batch_size]
            if len(batch_idx) < self.mini_batch_size and start > 0:
                continue

            yield (
                self.stacked_frames[batch_idx],
                self.game_features [batch_idx],
                self.actions       [batch_idx],
                self.log_probs     [batch_idx],
                advantages         [batch_idx],
                returns            [batch_idx],
                self.values        [batch_idx],
            )

        # reset for next collection
        self.ptr = 0
        self.is_full = False
