# critic_rl/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _clamp_int(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _infer_N(batch_flat: Dict[str, torch.Tensor]) -> int:
    if "scalars" in batch_flat:
        return int(batch_flat["scalars"].shape[0])
    k = next(iter(batch_flat.keys()))
    return int(batch_flat[k].shape[0])


def _zeros_like_batch(
    batch_flat: Dict[str, torch.Tensor],
    *,
    shape_tail: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    dev = next(iter(batch_flat.values())).device
    N = _infer_N(batch_flat)
    return torch.zeros((N, *shape_tail), dtype=dtype, device=dev)


def _get_or_zeros(
    batch_flat: Dict[str, torch.Tensor],
    key: str,
    *,
    shape_tail: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    v = batch_flat.get(key, None)
    if v is None:
        return _zeros_like_batch(batch_flat, shape_tail=shape_tail, dtype=dtype)
    return v


def _causal_attn_mask(T: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    TransformerEncoder src_mask: shape [T, T]
    Masked positions should be -inf, allowed positions 0.
    """
    # disallow attending to future: mask upper triangle (j > i)
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m


class HashEmbedding(nn.Module):
    """
    Deterministic hashed embedding for large sparse integer IDs (chip ids, folder ids, etc.).
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # PyTorch supports either attn_mask or is_causal (varies by version).
        # We pass both safely; unsupported args will raise, so keep to common signature.
        try:
            return self.enc(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)  # type: ignore[arg-type]
        except TypeError:
            # Older PyTorch: no is_causal
            return self.enc(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CriticRLConfig:
    # Core width
    d_model: int = 256
    nhead: int = 8

    # Token tower depths
    grid_layers: int = 2
    hand_layers: int = 2
    folder_layers: int = 2
    held_layers: int = 1

    # Temporal encoder
    max_seq_len: int = 64
    time_layers: int = 4
    time_nhead: int = 8
    time_ff_mult: int = 4

    # Heads / fusion
    fusion_hidden: int = 512
    ff_mult: int = 4
    dropout: float = 0.05

    # Sparse IDs / codes
    id_buckets: int = 8192
    id_dim: int = 64
    code_vocab: int = 64
    code_dim: int = 16

    # Grid (tile/owner)
    tile_vocab: int = 32
    tile_dim: int = 24
    owner_vocab: int = 3
    owner_dim: int = 8

    # Scalars (STATE ONLY)
    scalar_dim: int = 16
    scalar_hidden: int = 128

    # Action vector (buttons) at time t
    action_dim: int = 10
    action_hidden: int = 128

    # Cross/beast
    cross_idx_vocab: int = 12
    cross_idx_dim: int = 16

    # --- Option A/C (discrete spatial) ---
    grid_idx_vocab: int = 18
    grid_idx_dim: int = 16
    rel_pe_vocab: int = 55
    rel_pe_dim: int = 16

    # Occupancy features (Option A)
    occ_dim: int = 18  # per-side
    occ_hidden: int = 64

    # --- Option B (CNN-ish board) ---
    board_channels: int = 4
    board_h: int = 3
    board_w: int = 6
    board_cnn_hidden: int = 64


class HPDeltaTDLambdaCritic(nn.Module):
    """
    Action-conditioned TD(λ) critic.

    It predicts Q(s_t, a_t) for each timestep in a behavior sequence.
    Training uses SARSA-style TD(λ) targets:
      y_t = r_t + γ[(1-λ) Q(s_{t+1}, a_{t+1}) + λ y_{t+1}]
    IMPORTANT:
      The temporal encoder is CAUSAL, so Q(s_t,a_t) cannot attend to future timesteps.
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

        # “Last Used” and “On Deck”
        self.last_used_proj = nn.Linear(cfg.id_dim, cfg.d_model)
        self.on_deck_proj = nn.Linear(cfg.id_dim, cfg.d_model)

        # Per-state token encoders (bidirectional is fine inside-state)
        ff = cfg.d_model * cfg.ff_mult
        self.grid_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.grid_layers, ff, cfg.dropout)
        self.hand_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.hand_layers, ff, cfg.dropout)
        self.folder_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.folder_layers, ff, cfg.dropout)
        self.held_enc = SmallTransformerEncoder(cfg.d_model, cfg.nhead, cfg.held_layers, ff, cfg.dropout)

        # Scalar tower (STATE ONLY)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.scalar_hidden),
            nn.GELU(),
            nn.Linear(cfg.scalar_hidden, cfg.scalar_hidden),
            nn.GELU(),
            nn.Linear(cfg.scalar_hidden, cfg.d_model),
        )

        # Action tower
        self.action_mlp = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.action_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_hidden, cfg.action_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_hidden, cfg.d_model),
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

        # Option A/C towers
        self.grid_idx_emb = nn.Embedding(cfg.grid_idx_vocab, cfg.grid_idx_dim)
        self.rel_pe_emb = nn.Embedding(cfg.rel_pe_vocab, cfg.rel_pe_dim)
        nn.init.normal_(self.grid_idx_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_pe_emb.weight, mean=0.0, std=0.02)

        self.spatial_mlp = nn.Sequential(
            nn.Linear((cfg.grid_idx_dim * 2) + cfg.rel_pe_dim + (cfg.occ_dim * 2), cfg.occ_hidden),
            nn.GELU(),
            nn.Linear(cfg.occ_hidden, cfg.d_model),
        )

        # Option B tower
        self.board_cnn = nn.Sequential(
            nn.Conv2d(cfg.board_channels, cfg.board_cnn_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.board_cnn_hidden, cfg.board_cnn_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.board_proj = nn.Linear(cfg.board_cnn_hidden, cfg.d_model)

        # Learned fusion
        # towers: grid, hand, folder, held, scalar, action, cross, last_p, deck_p, deck_e, spatial+board => 11
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 11, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.d_model),
        )

        # Temporal encoder (CAUSAL)
        tff = cfg.d_model * cfg.time_ff_mult
        self.time_pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.time_enc = SmallTransformerEncoder(cfg.d_model, cfg.time_nhead, cfg.time_layers, tff, cfg.dropout)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, 1),
        )
        # --- INITIAL BELIEF HINT ---
        # We nudge the first layer of the scalar MLP to start with the "correct" 
        # intuition: My HP (index 0) is Good, Enemy HP (index 1) is Bad.
        with torch.no_grad():
            # self.scalar_mlp[0] is the first nn.Linear(cfg.scalar_dim, cfg.scalar_hidden)
            first_linear = self.scalar_mlp[0]
            
            # Reset the HP-related weights to a specific polarity
            # Weight shape is [hidden, input_dim]
            # We set the first hidden neuron to be a pure "Advantage" detector
            first_linear.weight.data[0, 0] = 1.0  # Player HP -> Positive
            first_linear.weight.data[0, 1] = -1.0 # Enemy HP  -> Negative
            
            # Zero out the other scalar weights for that specific neuron 
            # so it starts focused purely on HP advantage
            first_linear.weight.data[0, 2:] = 0.0
            
            print("[init] Applied HP-Advantage hint to scalar_mlp.")

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
        Encode flattened timesteps (B*T -> N) into vectors (N, D).
        This is "single-step" encoding; temporal mixing happens later (causally).
        """
        cfg = self.cfg

        # Grid tokens (18)
        tiles = _clamp_int(batch_flat["grid_tile"], 0, cfg.tile_vocab - 1)
        owners = _clamp_int(batch_flat["grid_owner"], 0, cfg.owner_vocab - 1)
        g = torch.cat([self.tile_emb(tiles), self.owner_emb(owners)], dim=-1)
        g = self.grid_tok_proj(g) + self.grid_pos
        g = self.grid_enc(g)
        grid_vec = self._pool(g)

        # Hand tokens (10)
        hid = batch_flat["hand_id"]
        hcode = _clamp_int(batch_flat["hand_code"], 0, cfg.code_vocab - 1)
        hvis = batch_flat["hand_vis"].unsqueeze(-1)
        h = torch.cat([self.id_emb(hid), self.code_emb(hcode), hvis], dim=-1)
        h = self.hand_tok_proj(h) + self.hand_pos
        h = self.hand_enc(h)
        hand_vec = self._pool(h)

        # Folders (player/enemy)
        def folder_tower(prefix: str) -> torch.Tensor:
            fid = batch_flat[f"folder_id_{prefix}"]
            fused = torch.cat([self.id_emb(fid), batch_flat[f"folder_used_{prefix}"].unsqueeze(-1)], dim=-1)
            fused = self.folder_tok_proj(fused) + self.folder_pos
            fused = self.folder_enc(fused)
            return self._pool(fused, mask=batch_flat[f"folder_mask_{prefix}"])

        folder_p = folder_tower("p")
        folder_e = folder_tower("e")
        folder_vec = self.folder_fuse(torch.cat([folder_p, folder_e], dim=-1))

        # Held
        held_id = batch_flat["held_id"]
        held_code = _clamp_int(batch_flat["held_code"], 0, cfg.code_vocab - 1)
        held_mask = batch_flat["held_mask"]
        hc = torch.cat([self.id_emb(held_id), self.code_emb(held_code)], dim=-1)
        hc = self.held_tok_proj(hc) + self.held_pos
        hc = self.held_enc(hc)
        held_vec = self._pool(hc, mask=held_mask)

        # Scalars (state-only)
        scalar_vec = self.scalar_mlp(batch_flat["scalars"])

        # Action
        action = _get_or_zeros(batch_flat, "action", shape_tail=(cfg.action_dim,), dtype=torch.float32)
        action_vec = self.action_mlp(action)

        # Cross / Beast
        used = batch_flat["used_cross"]
        p_idx = _clamp_int(batch_flat["active_cross_idx_p"], 0, cfg.cross_idx_vocab - 1)
        e_idx = _clamp_int(batch_flat["active_cross_idx_e"], 0, cfg.cross_idx_vocab - 1)
        p_emb = self.cross_idx_emb(p_idx)
        e_emb = self.cross_idx_emb(e_idx)
        cross_in = torch.cat([used, p_emb, e_emb, batch_flat["beast_feats"]], dim=-1)
        cross_vec = self.cross_mlp(cross_in)

        # Last Used / On Deck
        last_p = self.id_emb(batch_flat["last_used_id_p"])
        last_p_vec = self.last_used_proj(last_p)

        deck_p = self.id_emb(batch_flat["current_chip_p"])
        deck_p_vec = self.on_deck_proj(deck_p)

        deck_e = self.id_emb(batch_flat["current_chip_e"])
        deck_e_vec = self.on_deck_proj(deck_e)

        # Option A/C
        p_grid_idx = _clamp_int(_get_or_zeros(batch_flat, "p_grid_idx", shape_tail=(), dtype=torch.int64), 0, cfg.grid_idx_vocab - 1)
        e_grid_idx = _clamp_int(_get_or_zeros(batch_flat, "e_grid_idx", shape_tail=(), dtype=torch.int64), 0, cfg.grid_idx_vocab - 1)
        rel_pe_idx = _clamp_int(_get_or_zeros(batch_flat, "rel_pe_idx", shape_tail=(), dtype=torch.int64), 0, cfg.rel_pe_vocab - 1)

        p_idx_emb = self.grid_idx_emb(p_grid_idx)
        e_idx_emb = self.grid_idx_emb(e_grid_idx)
        rel_emb = self.rel_pe_emb(rel_pe_idx)

        p_occ = _get_or_zeros(batch_flat, "grid_p_occ", shape_tail=(cfg.occ_dim,), dtype=torch.float32)
        e_occ = _get_or_zeros(batch_flat, "grid_e_occ", shape_tail=(cfg.occ_dim,), dtype=torch.float32)

        spatial_in = torch.cat([p_idx_emb, e_idx_emb, rel_emb, p_occ, e_occ], dim=-1)
        spatial_vec = self.spatial_mlp(spatial_in)

        # Option B
        board_feat = _get_or_zeros(
            batch_flat,
            "board_feat",
            shape_tail=(cfg.board_channels, cfg.board_h, cfg.board_w),
            dtype=torch.float32,
        )
        b = self.board_cnn(board_feat).flatten(1)
        board_vec = self.board_proj(b)

        fused = self.fusion(
            torch.cat(
                [
                    grid_vec,
                    hand_vec,
                    folder_vec,
                    held_vec,
                    scalar_vec,
                    action_vec,
                    cross_vec,
                    last_p_vec,
                    deck_p_vec,
                    deck_e_vec,
                    spatial_vec + board_vec,
                ],
                dim=-1,
            )
        )
        return fused  # (N, D)

    @staticmethod
    def _flatten_time(batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int, int]:
        any_k = next(iter(batch.keys()))
        B = int(batch[any_k].shape[0])
        T = int(batch[any_k].shape[1])

        flat: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if v.ndim < 2:
                continue
            flat[k] = v.reshape(B * T, *v.shape[2:])
        return flat, B, T

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: dict of tensors shaped [B, T, ...]
        returns: Q values [B, T]
        """
        flat, B, T = self._flatten_time(batch)
        step_vec = self._encode_state_flat(flat)  # [B*T, D]
        seq = step_vec.reshape(B, T, self.cfg.d_model)

        if T > self.cfg.max_seq_len:
            seq = seq[:, : self.cfg.max_seq_len, :]
            T = self.cfg.max_seq_len

        seq = seq + self.time_pos[:, :T, :]

        # CAUSAL temporal mixing (no future leakage)
        attn_mask = _causal_attn_mask(T, device=seq.device, dtype=seq.dtype)
        seq = self.time_enc(seq, attn_mask=attn_mask, is_causal=True)

        values = self.value_head(seq).squeeze(-1)  # [B,T]
        return values


__all__ = ["CriticRLConfig", "HPDeltaTDLambdaCritic"]
