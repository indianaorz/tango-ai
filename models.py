# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticCNN(nn.Module):
    def __init__(self, num_stacked_frames, num_game_features, num_actions, frame_height, frame_width):
        super(ActorCriticCNN, self).__init__()
        self.num_stacked_frames = num_stacked_frames
        self.num_game_features = num_game_features # Health, positions, charge, etc.
        self.num_actions = num_actions

        # CNN for processing stacked frames
        self.cnn_base = nn.Sequential(
            nn.Conv2d(num_stacked_frames, 32, kernel_size=8, stride=4), # Input: [B, C, H, W] -> [B, 4, 84, 84]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() # Flatten the output of conv layers
        )
        
        # Calculate the flattened CNN output size
        # To do this properly, run a dummy tensor through cnn_base once
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_stacked_frames, frame_height, frame_width)
            cnn_out_features = self.cnn_base(dummy_input).shape[1]

        # Fully connected layers for combined features
        # The input size is cnn_out_features + num_game_features
        self.fc_shared = nn.Sequential(
            nn.Linear(cnn_out_features + num_game_features, 512),
            nn.ReLU()
        )

        # Actor head (outputs action logits)
        self.actor_head = nn.Linear(512, num_actions)

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, stacked_frames_tensor, game_features_tensor):
        """
        Args:
            stacked_frames_tensor: Tensor of shape [batch_size, num_stacked_frames, height, width]
            game_features_tensor: Tensor of shape [batch_size, num_game_features]
        Returns:
            action_logits: Tensor of shape [batch_size, num_actions]
            state_value: Tensor of shape [batch_size, 1]
        """
        cnn_output = self.cnn_base(stacked_frames_tensor)
        
        # Concatenate CNN output with game features
        combined_features = torch.cat((cnn_output, game_features_tensor), dim=1)
        
        shared_output = self.fc_shared(combined_features)
        
        action_logits = self.actor_head(shared_output)
        state_value = self.critic_head(shared_output)
        
        return action_logits, state_value

    def get_action_and_value(self, stacked_frames_tensor, game_features_tensor, action_idx=None):
        """
        Gets action probabilities, samples an action, and gets state value.
        Also calculates log probability of the action.
        """
        action_logits, state_value = self.forward(stacked_frames_tensor, game_features_tensor)
        
        probs = F.softmax(action_logits, dim=-1)
        
        if action_idx is None: # During inference/acting
            action_distribution = torch.distributions.Categorical(probs)
            action_idx = action_distribution.sample() # Sample an action
            log_prob = action_distribution.log_prob(action_idx)
            entropy = action_distribution.entropy().mean() # For entropy bonus in loss
        else: # During training, action is given
            # Create distribution to get log_prob for the given action
            action_distribution = torch.distributions.Categorical(probs)
            log_prob = action_distribution.log_prob(action_idx)
            entropy = action_distribution.entropy().mean()

        return action_idx, log_prob, entropy, state_value