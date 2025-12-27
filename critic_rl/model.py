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
class CriticRLConfig:
    # Embedding sizes
    d_model: int = 128
    id_buckets: int = 8192
    id_dim: int = 64
    code_vocab: int = 64
    code_dim: int = 16

    tile_vocab: int = 32
    tile_dim: int = 24
    owner_vocab: int = 3
    owner_dim: int = 8

    # Per-state token encoders
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

    # Cross/beast tower
    cross_idx_vocab: int = 12
    cross_idx_dim: int = 16

    # Temporal encoder (sequence over time)
    time_layers: int = 2
    time_nhead: int = 4
    time_ff_mult: int = 4
    max_seq_len: int = 32

    # Fusion / head
    fusion_hidden: int = 256


class HPDeltaTDLambdaCritic(nn.Module):
    """
    TD(λ) value critic over short sequences.
    Produces V(s_t) for each timestep in a sequence.

    Input batch tensors are shaped:
      - per-token inputs: [B, T, ...]
    Output:
      - values: [B, T]
    """
    def __init__(self, cfg: CriticRLConfig, *, folder_len: int = 30):
        super().__init__()
        self.cfg = cfg
        self.folder_len = int(folder_len)

        # --- Embeddings ---
        self.id_emb = HashEmbedding(cfg.id_buckets, cfg.id_dim)

        self.code_emb = nn.Embedding(cfg.code_vocab, cfg.code_dim)
        nn.init.normal_(self.code_emb.weight, mean=0.0, std=0.02)

        self.tile_emb = nn.Embedding(cfg.tile_vocab, cfg.tile_dim)
        self.owner_emb = nn.Embedding(cfg.owner_vocab, cfg.owner_dim)
        nn.init.normal_(self.tile_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.owner_emb.weight, mean=0.0, std=0.02)

        # Positional embeddings for within-state token sequences
        self.grid_pos = nn.Parameter(torch.zeros(1, 18, cfg.d_model))
        self.hand_pos = nn.Parameter(torch.zeros(1, 10, cfg.d_model))
        self.folder_pos = nn.Parameter(torch.zeros(1, self.folder_len, cfg.d_model))
        self.held_pos = nn.Parameter(torch.zeros(1, 5, cfg.d_model))

        # Token projections to common d_model
        self.grid_tok_proj = nn.Linear(cfg.tile_dim + cfg.owner_dim, cfg.d_model)
        self.hand_tok_proj = nn.Linear(cfg.id_dim + cfg.code_dim + 1, cfg.d_model)  # +visible flag
        self.folder_tok_proj = nn.Linear(cfg.id_dim + 1, cfg.d_model)               # +used mask
        self.held_tok_proj = nn.Linear(cfg.id_dim + cfg.code_dim, cfg.d_model)

        # Per-state token encoders
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
        self.cross_idx_emb = nn.Embedding(cfg.cross_idx_vocab, cfg.cross_idx_dim)
        nn.init.normal_(self.cross_idx_emb.weight, mean=0.0, std=0.02)

        self.cross_mlp = nn.Sequential(
            nn.Linear(22 + (cfg.cross_idx_dim * 2) + 6, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Fuse player+enemy folder pooled vectors -> d_model
        self.folder_fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)
        nn.init.normal_(self.folder_fuse.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.folder_fuse.bias)

        # --- Temporal encoder over state embeddings ---
        tff = cfg.d_model * cfg.time_ff_mult
        self.time_pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.time_enc = SmallTransformerEncoder(cfg.d_model, cfg.time_nhead, cfg.time_layers, tff, cfg.dropout)

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, 1),
        )

    @staticmethod
    def _pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (N,T,D), mask: (N,T) True for valid
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        return (x * m).sum(dim=1) / denom

    def _encode_state_flat(self, batch_flat: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of states (flattened B*T -> N) into state vectors (N, D).
        Expects keys shaped like the original per-frame tensors (no time dim).
        """
        cfg = self.cfg

        # --- Grid ---
        tiles = _clamp_int(batch_flat["grid_tile"], 0, cfg.tile_vocab - 1)
        owners = _clamp_int(batch_flat["grid_owner"], 0, cfg.owner_vocab - 1)
        g = torch.cat([self.tile_emb(tiles), self.owner_emb(owners)], dim=-1)  # (N,18,td+od)
        g = self.grid_tok_proj(g) + self.grid_pos
        g = self.grid_enc(g)
        grid_vec = self._pool(g)

        # --- Hand ---
        hid = batch_flat["hand_id"]
        hcode = _clamp_int(batch_flat["hand_code"], 0, cfg.code_vocab - 1)
        hvis = batch_flat["hand_vis"].unsqueeze(-1)
        h = torch.cat([self.id_emb(hid), self.code_emb(hcode), hvis], dim=-1)
        h = self.hand_tok_proj(h) + self.hand_pos
        h = self.hand_enc(h)
        hand_vec = self._pool(h)

        # --- Folders ---
        def folder_tower(prefix: str) -> torch.Tensor:
            fid = batch_flat[f"folder_id_{prefix}"]
            fused = torch.cat([self.id_emb(fid), batch_flat[f"folder_used_{prefix}"].unsqueeze(-1)], dim=-1)
            fused = self.folder_tok_proj(fused) + self.folder_pos
            fused = self.folder_enc(fused)
            return self._pool(fused, mask=batch_flat[f"folder_mask_{prefix}"])

        folder_p = folder_tower("p")
        folder_e = folder_tower("e")
        folder_vec = self.folder_fuse(torch.cat([folder_p, folder_e], dim=-1))

        # --- Held ---
        held_id = batch_flat["held_id"]
        held_code = _clamp_int(batch_flat["held_code"], 0, cfg.code_vocab - 1)
        held_mask = batch_flat["held_mask"]
        hc = torch.cat([self.id_emb(held_id), self.code_emb(held_code)], dim=-1)
        hc = self.held_tok_proj(hc) + self.held_pos
        hc = self.held_enc(hc)
        held_vec = self._pool(hc, mask=held_mask)

        # --- Scalars ---
        scal = batch_flat["scalars"]

        # Charge is currently cached as (charge / 100), but charge is actually in {0,1,2}.
        # Convert cached scale -> normalized [0,1] without rebuilding the cache:
        #   (charge/100) * 50 == charge/2
        # player_charge is scal[...,2], enemy_charge is scal[...,3]
        if scal.shape[-1] >= 4:
            scal = scal.clone()
            scal[..., 2:4] = torch.clamp(scal[..., 2:4] * 50.0, 0.0, 1.0)

        scalar_vec = self.scalar_mlp(scal)


        # --- Cross/Beast ---
        used = batch_flat["used_cross"]
        p_idx = _clamp_int(batch_flat["active_cross_idx_p"], 0, cfg.cross_idx_vocab - 1)
        e_idx = _clamp_int(batch_flat["active_cross_idx_e"], 0, cfg.cross_idx_vocab - 1)
        p_emb = self.cross_idx_emb(p_idx)
        e_emb = self.cross_idx_emb(e_idx)

        cross_in = torch.cat([used, p_emb, e_emb, batch_flat["beast_feats"]], dim=-1)
        cross_vec = self.cross_mlp(cross_in)

        # --- Fuse all into one state vector ---
        # Keep it simple: sum of towers after a linear mix
        fused = torch.stack([grid_vec, hand_vec, folder_vec, held_vec, scalar_vec, cross_vec], dim=0).mean(dim=0)
        return fused  # (N,D)

    @staticmethod
    def _flatten_time(batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int, int]:
        # batch tensors are [B,T,...] -> flatten to [N,...]
        any_k = next(iter(batch.keys()))
        B = int(batch[any_k].shape[0])
        T = int(batch[any_k].shape[1])

        flat: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if v.ndim < 2:
                raise ValueError(f"expected time-major tensors [B,T,...] for key '{k}', got shape {tuple(v.shape)}")
            flat[k] = v.reshape(B * T, *v.shape[2:])
        return flat, B, T

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: dict of tensors shaped [B, T, ...]
        returns: values [B, T]
        """
        flat, B, T = self._flatten_time(batch)
        state_vec = self._encode_state_flat(flat)  # [B*T, D]
        state_seq = state_vec.reshape(B, T, self.cfg.d_model)

        # temporal encoding
        if T > self.cfg.max_seq_len:
            # fail fast: caller should set max_seq_len >= dataset seq_len
            raise ValueError(f"sequence length T={T} exceeds max_seq_len={self.cfg.max_seq_len}")

        state_seq = state_seq + self.time_pos[:, :T, :]
        state_seq = self.time_enc(state_seq)  # [B,T,D]

        values = self.value_head(state_seq).squeeze(-1)  # [B,T]
        return values
