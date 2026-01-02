import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.
    The 'cond' vector determines the Scale and Shift for the normalization.
    This effectively tells the network "Which Mode" to be in.
    """
    def __init__(self, embedding_dim, chunk_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(chunk_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, cond):
        emb = self.linear(self.silu(cond))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, cond_size, heads):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_size, cond_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=heads, batch_first=True)
        self.norm2 = AdaLayerNorm(hidden_size, cond_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        
        h = self.norm2(x, cond)
        mlp_out = self.mlp(h)
        x = x + mlp_out
        return x

class ActionFlowDiT(nn.Module):
    def __init__(
        self, 
        feat_dim=47, 
        act_dim=10, 
        hist_len=256, 
        seq_len=18, 
        embed_dim=384, 
        depth=6, 
        heads=6
    ):
        super().__init__()
        
        # History Encoder (1D CNN) -> Compresses 256 frames into 'embed_dim'
        self.history_enc = nn.Sequential(
            nn.Conv1d(feat_dim, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(256, embed_dim, 3, padding=1), nn.GroupNorm(16, embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        
        self.act_emb = nn.Linear(act_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.ret_embedder = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        
        # The Director's Signal: Time + Return + History
        self.cond_dim = embed_dim * 3 
        
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, self.cond_dim, heads) for _ in range(depth)
        ])
        
        self.final_norm = AdaLayerNorm(embed_dim, self.cond_dim)
        self.head = nn.Linear(embed_dim, act_dim)

    def forward(self, x_t, t, history, returns):
        # Embed inputs
        x = self.act_emb(x_t) + self.pos_emb
        
        # Embed conditions (The "Director" context)
        t_emb = self.t_embedder(t)
        h_emb = self.history_enc(history.permute(0, 2, 1))
        r_emb = self.ret_embedder(returns)
        
        # Fuse conditions
        c = torch.cat([t_emb, r_emb, h_emb], dim=1)
        
        # Pass through AdaLN blocks
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_norm(x, c)
        return self.head(x)