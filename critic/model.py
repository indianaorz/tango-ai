# critic/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def _clamp_int(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


class HashEmbedding(nn.Module):
    """
    Deterministic hashed embedding for large sparse integer IDs (chip ids, folder ids, etc.)
    Avoids huge vocab tables.
    """
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.emb = nn.Embedding(self.num_buckets, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (...,) int64
        idx = torch.remainder(ids, self.num_buckets)
        return self.emb(idx)


class SmallTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=src_key_padding_mask)


@dataclass(frozen=True)
class CriticConfig:
    # Embedding sizes
    d_model: int = 96
    id_buckets: int = 8192
    id_dim: int = 64
    code_vocab: int = 64
    code_dim: int = 16

    tile_vocab: int = 32     # supports null->0 + tile ids up to ~31 (adjust if needed)
    tile_dim: int = 24
    owner_vocab: int = 3     # 0=p1,1=p2,2=neutral/unknown
    owner_dim: int = 8

    # Transformers
    grid_layers: int = 2
    hand_layers: int = 2
    folder_layers: int = 2
    held_layers: int = 1

    nhead: int = 4
    ff_mult: int = 4
    dropout: float = 0.05

    # Scalar MLP
    scalar_dim: int = 32
    scalar_hidden: int = 128

    # Fusion
    fusion_hidden: int = 256


class HPDeltaCritic(nn.Module):
    """
    RL value critic:
      Predicts discounted return of hp_delta advantage from current state.

    Target (in dataset):
      reward r_t = (hp_delta[t+1] - hp_delta[t])
      return G_t = sum_k gamma^k r_{t+k}

    Output:
      out: (B,) float32
    """
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg

        # --- Embeddings ---
        self.id_emb = HashEmbedding(cfg.id_buckets, cfg.id_dim)

        self.code_emb = nn.Embedding(cfg.code_vocab, cfg.code_dim)
        nn.init.normal_(self.code_emb.weight, mean=0.0, std=0.02)

        self.tile_emb = nn.Embedding(cfg.tile_vocab, cfg.tile_dim)
        self.owner_emb = nn.Embedding(cfg.owner_vocab, cfg.owner_dim)
        nn.init.normal_(self.tile_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.owner_emb.weight, mean=0.0, std=0.02)

        # Positional embeddings
        self.grid_pos = nn.Parameter(torch.zeros(1, 18, cfg.d_model))
        self.hand_pos = nn.Parameter(torch.zeros(1, 10, cfg.d_model))
        self.folder_pos = nn.Parameter(torch.zeros(1, 30, cfg.d_model))
        self.held_pos = nn.Parameter(torch.zeros(1, 5, cfg.d_model))

        # Token projections to common d_model
        self.grid_tok_proj = nn.Linear(cfg.tile_dim + cfg.owner_dim, cfg.d_model)
        self.hand_tok_proj = nn.Linear(cfg.id_dim + cfg.code_dim + 1, cfg.d_model)   # +visible flag
        self.folder_tok_proj = nn.Linear(cfg.id_dim + 1, cfg.d_model)                # +used mask
        self.held_tok_proj = nn.Linear(cfg.id_dim + cfg.code_dim, cfg.d_model)

        # Encoders
        ff = cfg.d_model * cfg.ff_mult
        self.grid_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.grid_layers, ff, cfg.dropout)
        self.hand_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.hand_layers, ff, cfg.dropout)
        self.folder_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.folder_layers, ff, cfg.dropout)
        self.held_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.held_layers, ff, cfg.dropout)

        # Scalar tower
        self.scalar_mlp = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_hidden),
            nn.GELU(),
            nn.Linear(cfg.scalar_hidden, cfg.scalar_hidden),
            nn.GELU(),
            nn.Linear(cfg.scalar_hidden, cfg.d_model),
        )

        # Cross/beast tower
        # used_cross_mask len=11 for each side => 22 floats
        # active_cross_idx (0..10) + unknown=11 for each side
        self.cross_idx_emb = nn.Embedding(12, 16)  # 0..10 plus 11=unknown
        nn.init.normal_(self.cross_idx_emb.weight, mean=0.0, std=0.02)

        self.cross_mlp = nn.Sequential(
            nn.Linear(22 + (16 * 2) + 6, cfg.d_model),  # used masks + 2 idx embeddings + beast/turns_since feats
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Fuse player+enemy folder pooled vectors -> d_model (REGISTERED, trained)
        self.folder_fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)
        nn.init.normal_(self.folder_fuse.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.folder_fuse.bias)

        # Fusion head
        # grid pooled + hand pooled + folder pooled + held pooled + scalar + cross
        fused_in = cfg.d_model * 6
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden, 1),
        )

    @staticmethod
    def _pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,T,D), mask: (B,T) True for valid
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)  # (B,T,1)
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        return (x * m).sum(dim=1) / denom

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns: (B,) float32 predicted discounted return of hp_delta advantage.
        Expects keys produced by critic/dataset.py::tensorize_sample().
        """
        cfg = self.cfg

        # --- Grid ---
        tiles = _clamp_int(batch["grid_tile"], 0, cfg.tile_vocab - 1)
        owners = _clamp_int(batch["grid_owner"], 0, cfg.owner_vocab - 1)
        g = torch.cat([self.tile_emb(tiles), self.owner_emb(owners)], dim=-1)  # (B,18,td+od)
        g = self.grid_tok_proj(g) + self.grid_pos
        g = self.grid_enc(g)
        grid_vec = self._pool(g)

        # --- Hand ---
        hid = batch["hand_id"]
        hcode = _clamp_int(batch["hand_code"], 0, cfg.code_vocab - 1)
        hvis = batch["hand_vis"].unsqueeze(-1)
        h = torch.cat([self.id_emb(hid), self.code_emb(hcode), hvis], dim=-1)
        h = self.hand_tok_proj(h) + self.hand_pos
        h = self.hand_enc(h)
        hand_vec = self._pool(h)

        # --- Folders (player + enemy) ---
        def folder_tower(prefix: str) -> torch.Tensor:
            fid = batch[f"folder_id_{prefix}"]
            fused = torch.cat([self.id_emb(fid), batch[f"folder_used_{prefix}"].unsqueeze(-1)], dim=-1)
            fused = self.folder_tok_proj(fused) + self.folder_pos
            fused = self.folder_enc(fused)
            return self._pool(fused, mask=batch[f"folder_mask_{prefix}"])

        folder_p = folder_tower("p")
        folder_e = folder_tower("e")
        folder_vec = self.folder_fuse(torch.cat([folder_p, folder_e], dim=-1))

        # --- Held chips (player only; 5 slots) ---
        held_id = batch["held_id"]         # (B,5)
        held_code = _clamp_int(batch["held_code"], 0, cfg.code_vocab - 1)
        held_mask = batch["held_mask"]     # (B,5) bool
        hc = torch.cat([self.id_emb(held_id), self.code_emb(held_code)], dim=-1)
        hc = self.held_tok_proj(hc) + self.held_pos
        hc = self.held_enc(hc)
        held_vec = self._pool(hc, mask=held_mask)

        # --- Scalars ---
        scal = batch["scalars"]            # (B,scalar_dim)
        scalar_vec = self.scalar_mlp(scal)

        # --- Cross/Beast ---
        used = batch["used_cross"]         # (B,22)
        p_idx = _clamp_int(batch["active_cross_idx_p"], 0, 11)
        e_idx = _clamp_int(batch["active_cross_idx_e"], 0, 11)
        p_emb = self.cross_idx_emb(p_idx)
        e_emb = self.cross_idx_emb(e_idx)

        cross_in = torch.cat([used, p_emb, e_emb, batch["beast_feats"]], dim=-1)
        cross_vec = self.cross_mlp(cross_in)

        # --- Fusion ---
        fused = torch.cat([grid_vec, hand_vec, folder_vec, held_vec, scalar_vec, cross_vec], dim=-1)
        out = self.fusion(fused).squeeze(-1)
        return out
