import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticCNNRNN(nn.Module):
    """
    CNN (optional) -> optional linear fuse -> {GRU/LSTM}^{layers} -> actor/critic heads

    Set frame_channels=0 to disable the CNN path and run state-only (features-only).
    """
    def __init__(self,
                 seq_len_frames: int,
                 num_game_features: int,
                 num_actions: int,
                 frame_height: int,
                 frame_width: int,
                 frame_channels: int = 3,
                 # Configurable CNN
                 cnn_channels=(64, 128, 256, 256),
                 cnn_kernels=(8, 4, 3, 3),
                 cnn_strides=(4, 2, 1, 1),
                 fuse_dim: int = 1024,
                 rnn_type: str = "gru",
                 rnn_hidden: int = 2048,
                 rnn_layers: int = 2,
                 rnn_dropout: float = 0.0):
        super().__init__()
        self.T  = seq_len_frames
        self.F  = num_game_features
        self.A  = num_actions
        self.C  = frame_channels

        # ---- CNN backbone (optional) ----
        self.cnn_base = None
        flat = 0
        if self.C > 0:
            layers = []
            in_ch = self.C
            for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
                layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s), nn.ReLU(inplace=True)]
                in_ch = out_ch
            layers += [nn.Flatten()]
            self.cnn_base = nn.Sequential(*layers)

            with torch.no_grad():
                dummy = torch.zeros(1, self.C, frame_height, frame_width)
                flat = self.cnn_base(dummy).shape[1]  # per-frame latent

        # ---- fuse image latent + scalar features ----
        fuse_in = flat + self.F
        if fuse_dim and fuse_dim > 0:
            self.fuse = nn.Sequential(
                nn.Linear(fuse_in, fuse_dim),
                nn.ReLU(inplace=True)
            )
            rnn_input_size = fuse_dim
        else:
            self.fuse = None
            rnn_input_size = fuse_in

        # ---- RNN ----
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=(rnn_dropout if rnn_layers > 1 else 0.0),
            batch_first=True,
        )

        # ---- Heads ----
        self.actor_head  = nn.Linear(rnn_hidden, self.A)
        self.critic_head = nn.Linear(rnn_hidden, 1)

    def forward(self, frame_seq, feat_seq, h0=None):
        """
        frame_seq: [B,T,C,H,W] (C may be 0) ; feat_seq: [B,T,F]
        """
        B, T, C, H, W = frame_seq.shape

        # 🔧 Use runtime C, not self.C
        if C > 0 and self.cnn_base is not None:
            flat_frames = self.cnn_base(frame_seq.reshape(B * T, C, H, W)).view(B, T, -1)
        else:
            # No-vision path
            flat_frames = feat_seq.new_zeros(B, T, 0)

        fused = torch.cat([flat_frames, feat_seq], dim=-1)
        if self.fuse is not None:
            fused = self.fuse(fused)

        rnn_out, h_n = self.rnn(fused, h0)
        last = rnn_out[:, -1]
        logits = self.actor_head(last)
        value  = self.critic_head(last).squeeze(-1)
        return logits, value, h_n

    def get_action_and_value(self, frame_seq, feat_seq, action_idx=None, h0=None):
        logits, value, h_n = self.forward(frame_seq, feat_seq, h0)
        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        if action_idx is None:
            action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy  = dist.entropy().mean()
        return action_idx, log_prob, entropy, value, h_n
