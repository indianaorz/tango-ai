# critic_minimal/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MinimalQConfig:
    max_seq_len: int = 192

    # Transformer
    d_model: int = 256
    time_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.10

    # Input dims (must match cache)
    action_dim: int = 10
    scalar_dim: int = 5

    # Vocab sizes (clamped; increase if you want fewer collisions)
    emotion_vocab: int = 256
    chip_vocab: int = 512
    tile_vocab: int = 128
    owner_vocab: int = 3
    grid_idx_vocab: int = 18
    rel_idx_vocab: int = 55

    # Embedding dims
    emb_emotion: int = 16
    emb_chip: int = 16
    emb_pos: int = 16
    emb_rel: int = 16
    emb_tile: int = 16
    emb_owner: int = 8

    # Projection dims before transformer
    proj_action: int = 32
    proj_scalar: int = 32


def _clamp_ids(x: torch.Tensor, vocab: int) -> torch.Tensor:
    # x may contain garbage; keep it safe.
    if x.dtype != torch.long:
        x = x.long()
    if vocab <= 1:
        return torch.zeros_like(x)
    return torch.clamp(x, 0, vocab - 1)


class _GridEncoder(nn.Module):
    """
    Encodes (grid_tile, grid_owner) of shape [B,T,18] each into a per-timestep vector.

    We add a learned cell-position embedding to tile embeddings so the mean-pool
    still preserves some spatial identity.
    """

    def __init__(self, cfg: MinimalQConfig):
        super().__init__()
        self.tile_emb = nn.Embedding(cfg.tile_vocab, cfg.emb_tile)
        self.owner_emb = nn.Embedding(cfg.owner_vocab, cfg.emb_owner)

        # Cell positional embedding for 18 cells (added to tile embeddings)
        self.cell_pos = nn.Embedding(18, cfg.emb_tile)

        self.out_dim = cfg.emb_tile + cfg.emb_owner

    def forward(self, grid_tile: torch.Tensor, grid_owner: torch.Tensor) -> torch.Tensor:
        """
        grid_tile/grid_owner: [B,T,18] int64
        returns: [B,T,out_dim]
        """
        B, T, C = grid_tile.shape
        assert C == 18, f"expected 18 cells, got {C}"

        tile = _clamp_ids(grid_tile, self.tile_emb.num_embeddings)
        owner = _clamp_ids(grid_owner, self.owner_emb.num_embeddings)

        tile_e = self.tile_emb(tile)  # [B,T,18,emb_tile]
        owner_e = self.owner_emb(owner)  # [B,T,18,emb_owner]

        cell_ids = torch.arange(18, device=grid_tile.device, dtype=torch.long)  # [18]
        pos_e = self.cell_pos(cell_ids).view(1, 1, 18, -1)  # [1,1,18,emb_tile]
        tile_e = tile_e + pos_e

        tile_pool = tile_e.mean(dim=2)  # [B,T,emb_tile]
        owner_pool = owner_e.mean(dim=2)  # [B,T,emb_owner]

        return torch.cat([tile_pool, owner_pool], dim=-1)  # [B,T,out_dim]


class MinimalQCritic(nn.Module):
    """
    Predicts Q(s_t, a_t) (actually "value-like" return) for each timestep.

    IMPORTANT: We train on y_norm = tanh(y_raw / norm_factor).
               So the head output is in [-1,1].
    """

    def __init__(self, cfg: MinimalQConfig):
        super().__init__()
        self.cfg = cfg

        # Projections for float inputs
        self.action_proj = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.proj_action),
            nn.SiLU(),
            nn.Linear(cfg.proj_action, cfg.proj_action),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.proj_scalar),
            nn.SiLU(),
            nn.Linear(cfg.proj_scalar, cfg.proj_scalar),
        )

        # Embeddings for discrete inputs
        self.pemo_emb = nn.Embedding(cfg.emotion_vocab, cfg.emb_emotion)
        self.eemo_emb = nn.Embedding(cfg.emotion_vocab, cfg.emb_emotion)

        self.chip_emb = nn.Embedding(cfg.chip_vocab, cfg.emb_chip)

        self.pidx_emb = nn.Embedding(cfg.grid_idx_vocab, cfg.emb_pos)
        self.eidx_emb = nn.Embedding(cfg.grid_idx_vocab, cfg.emb_pos)
        self.rel_emb = nn.Embedding(cfg.rel_idx_vocab, cfg.emb_rel)

        self.grid_enc = _GridEncoder(cfg)

        # Fuse per-timestep features -> d_model
        fuse_in = (
            cfg.proj_action
            + cfg.proj_scalar
            + 2 * cfg.emb_emotion
            + cfg.emb_chip
            + 2 * cfg.emb_pos
            + cfg.emb_rel
            + self.grid_enc.out_dim
        )
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Learned time positional embedding
        self.time_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.time_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x keys (all are [B,T,...]):
          - scalars: [B,T,5] float
          - action: [B,T,10] float
          - p_emotion_id: [B,T] long
          - e_emotion_id: [B,T] long
          - player_chip: [B,T] long
          - p_grid_idx: [B,T] long
          - e_grid_idx: [B,T] long
          - rel_pe_idx: [B,T] long
          - grid_tile: [B,T,18] long
          - grid_owner: [B,T,18] long

        returns:
          - q_norm: [B,T] float (tanh-space target)
        """
        scalars = x["scalars"].to(torch.float32)
        action = x["action"].to(torch.float32)

        pemo = _clamp_ids(x["p_emotion_id"], self.pemo_emb.num_embeddings)
        eemo = _clamp_ids(x["e_emotion_id"], self.eemo_emb.num_embeddings)
        chip = _clamp_ids(x["player_chip"], self.chip_emb.num_embeddings)

        pidx = _clamp_ids(x["p_grid_idx"], self.pidx_emb.num_embeddings)
        eidx = _clamp_ids(x["e_grid_idx"], self.eidx_emb.num_embeddings)
        rel = _clamp_ids(x["rel_pe_idx"], self.rel_emb.num_embeddings)

        grid_feat = self.grid_enc(x["grid_tile"], x["grid_owner"])  # [B,T,G]

        feats = torch.cat(
            [
                self.action_proj(action),
                self.scalar_proj(scalars),
                self.pemo_emb(pemo),
                self.eemo_emb(eemo),
                self.chip_emb(chip),
                self.pidx_emb(pidx),
                self.eidx_emb(eidx),
                self.rel_emb(rel),
                grid_feat,
            ],
            dim=-1,
        )

        h = self.fuse(feats)  # [B,T,d_model]

        B, T, _ = h.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds cfg.max_seq_len {self.cfg.max_seq_len}")

        tpos = self.time_pos(torch.arange(T, device=h.device, dtype=torch.long)).view(1, T, -1)
        h = h + tpos

        h = self.encoder(h)  # [B,T,d_model]
        q = self.head(h).squeeze(-1)  # [B,T]

        # Head is unconstrained; caller trains against tanh target.
        return q


__all__ = ["MinimalQConfig", "MinimalQCritic"]
