# ppo_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, actor_critic_model, learning_rate, ppo_clip_epsilon, ppo_epochs,
                 value_loss_coef, entropy_coef, device):
        self.model = actor_critic_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.clip_epsilon = ppo_clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device

    def train_step(self, stacked_frames_batch, game_features_batch, actions_batch,
                   old_log_probs_batch, advantages_batch, returns_batch, old_values_batch):
        """Performs a single PPO training step on a batch of data."""
        
        # PPO requires multiple epochs over the same batch of data
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            # Get new log_probs, entropy, and values from the current model
            # Note: actions_batch are the actions taken by the old policy
            _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                stacked_frames_batch, game_features_batch, actions_batch
            )
            new_values = new_values.squeeze(-1) # Ensure correct shape [batch_size]

            # Calculate policy loss (PPO-Clip objective)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss (clipped or unclipped)
            # Clipped value loss (optional, sometimes helps)
            # values_pred_clipped = old_values_batch + torch.clamp(new_values - old_values_batch, -self.clip_epsilon, self.clip_epsilon)
            # vf_loss1 = F.mse_loss(new_values, returns_batch)
            # vf_loss2 = F.mse_loss(values_pred_clipped, returns_batch)
            # value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            value_loss = F.mse_loss(new_values, returns_batch).mean()


            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Gradient clipping
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")

        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_value_loss = total_value_loss / self.ppo_epochs
        avg_entropy = total_entropy / self.ppo_epochs
        
        return avg_policy_loss, avg_value_loss, avg_entropy