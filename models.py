import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticCNNRNN(nn.Module):
    """
    CNN encoder → GRU → actor / critic heads.
    Now supports an arbitrary number of colour channels (default 3).
    """

    def __init__(self,
                 seq_len_frames: int,
                 num_game_features: int,
                 num_actions: int,
                 frame_height: int,
                 frame_width: int,
                 frame_channels: int = 3,
                 rnn_hidden: int = 512):
        super().__init__()

        self.seq_len_frames   = seq_len_frames
        self.num_game_features= num_game_features
        self.num_actions      = num_actions
        self.rnn_hidden       = rnn_hidden
        self.C                = frame_channels

        # ── 1 ▸ CNN encoder ───────────────────────────────────────────
        self.cnn_base = nn.Sequential(
            nn.Conv2d(self.C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),     nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),     nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.C, frame_height, frame_width)
            self.frame_latent_dim = self.cnn_base(dummy).shape[1]

        # ── 2 ▸ Temporal GRU ──────────────────────────────────────────
        self.rnn = nn.GRU(
            input_size  = self.frame_latent_dim + num_game_features,
            hidden_size = rnn_hidden,
            num_layers  = 1,
            batch_first = True
        )

        # ── 3 ▸ Heads ────────────────────────────────────────────────
        self.actor_head  = nn.Linear(rnn_hidden, num_actions)
        self.critic_head = nn.Linear(rnn_hidden, 1)

    # ------------------------------------------------------------------
    def forward(self, frame_seq, feat_seq, h0=None):
        """
        frame_seq : [B, T, C, H, W]
        feat_seq  : [B, T, F]
        """
        B, T, C, H, W = frame_seq.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"

        flat_frames = frame_seq.reshape(B*T, C, H, W)      # (B·T, C, H, W)
        latents     = self.cnn_base(flat_frames).view(B, T, -1)
        rnn_in      = torch.cat([latents, feat_seq], dim=-1)
        rnn_out, h_n  = self.rnn(rnn_in, h0)               # (B, T, H)
        last        = rnn_out[:, -1]                       # most‑recent step
        return self.actor_head(last), self.critic_head(last).squeeze(-1), h_n

    # identical external API
    def get_action_and_value(self, frame_seq, feat_seq, action_idx=None, h0=None):
        logits, value, h_n = self.forward(frame_seq, feat_seq, h0)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)

        if action_idx is None:
            action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy  = dist.entropy().mean()

        return action_idx, log_prob, entropy, value, h_n
