# planning_critic/model.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PlanningConfig:
    # Sizes
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3

    # Dropout (increase for generalization)
    # - transformer dropout affects attention/ffn
    # - feature dropout applied to pooled vectors
    # - head dropout applied in MLP head
    dropout_tf: float = 0.25
    dropout_feat: float = 0.35
    dropout_head: float = 0.35

    # Vocabs
    chip_buckets: int = 8192
    code_vocab: int = 128
    cross_vocab: int = 64
    tile_vocab: int = 32
    owner_vocab: int = 3  # keep tight (0,1,2) unless you *really* have others

    # Dims
    chip_dim: int = 64
    code_dim: int = 16
    tile_dim: int = 16
    owner_dim: int = 8

    # Sequence lengths
    n_held: int = 5
    n_draw: int = 10
    n_hand: int = 15  # held + draw
    n_selected: int = 5


class HashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.emb = nn.Embedding(self.num_buckets, int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: long
        return self.emb(x % self.num_buckets)


class PlanningCritic(nn.Module):
    def __init__(self, cfg: PlanningConfig):
        super().__init__()
        self.cfg = cfg

        # --- Shared chip embedding (id + code -> token) ---
        self.chip_emb = HashEmbedding(cfg.chip_buckets, cfg.chip_dim)
        self.code_emb = nn.Embedding(cfg.code_vocab, cfg.code_dim)
        self.chip_proj = nn.Linear(cfg.chip_dim + cfg.code_dim, cfg.d_model)

        # --- Context tower ---
        self.tile_emb = nn.Embedding(cfg.tile_vocab, cfg.tile_dim)
        self.owner_emb = nn.Embedding(cfg.owner_vocab, cfg.owner_dim)
        self.grid_proj = nn.Linear(18 * (cfg.tile_dim + cfg.owner_dim), cfg.d_model)

        self.scalar_proj = nn.Linear(3, cfg.d_model)

        # 30 + 30 + 11 + 11 = 82 floats
        self.mask_proj = nn.Linear(82, cfg.d_model)

        # --- Hand encoder (15 tokens) ---
        self.hand_pos = nn.Parameter(torch.randn(1, cfg.n_hand, cfg.d_model) * 0.02)
        self.hand_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout_tf,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=cfg.num_layers,
        )
        self.hand_pool_dropout = nn.Dropout(cfg.dropout_feat)

        # --- Action encoder (5 tokens) ---
        self.action_pos = nn.Parameter(torch.randn(1, cfg.n_selected, cfg.d_model) * 0.02)
        self.cross_emb = nn.Embedding(cfg.cross_vocab, cfg.d_model)
        self.beast_emb = nn.Embedding(2, cfg.d_model)

        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout_tf,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=cfg.num_layers,
        )
        self.action_pool_dropout = nn.Dropout(cfg.dropout_feat)

        # --- Fusion head ---
        self.fusion_norm = nn.LayerNorm(cfg.d_model * 5)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 5, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout_head),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout_head),
            nn.Linear(256, 1),
        )

    def _embed_chips(self, ids: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # ids/codes: [B, T]
        c = self.chip_emb(ids)
        o = self.code_emb(codes)
        x = torch.cat([c, o], dim=-1)
        return self.chip_proj(x)

    def forward(self, batch: dict) -> torch.Tensor:
        # -------- Context --------
        g_t = self.tile_emb(batch["grid_tile"])          # [B, 18, tile_dim]
        g_o = self.owner_emb(batch["grid_owner"])        # [B, 18, owner_dim]
        grid_flat = torch.cat([g_t, g_o], dim=-1).flatten(1)  # [B, 18*(tile+owner)]
        vec_grid = self.grid_proj(grid_flat)             # [B, d]

        scalars = torch.stack([batch["p_hp"], batch["e_hp"], batch["turn_idx"]], dim=-1)  # [B, 3]
        vec_scalar = self.scalar_proj(scalars)           # [B, d]

        masks = torch.cat(
            [batch["folder_p"], batch["folder_e"], batch["cross_hist_p"], batch["cross_hist_e"]],
            dim=-1,
        )                                                # [B, 82]
        vec_mask = self.mask_proj(masks)                 # [B, d]

        # -------- Hand --------
        hand_ids = torch.cat([batch["held_id"], batch["draw_id"]], dim=1)        # [B, 15]
        hand_codes = torch.cat([batch["held_code"], batch["draw_code"]], dim=1) # [B, 15]
        hand = self._embed_chips(hand_ids, hand_codes) + self.hand_pos          # [B, 15, d]
        hand_out = self.hand_encoder(hand)                                     # [B, 15, d]
        vec_hand = self.hand_pool_dropout(hand_out.mean(dim=1))                # [B, d]

        # -------- Action --------
        sel = self._embed_chips(batch["sel_id"], batch["sel_code"]) + self.action_pos  # [B, 5, d]
        sel_out = self.action_encoder(sel)                                         # [B, 5, d]
        vec_sel = self.action_pool_dropout(sel_out.mean(dim=1))                    # [B, d]

        vec_cross = self.cross_emb(batch["sel_cross"].long())                      # [B, d]
        vec_beast = self.beast_emb(batch["sel_beast"].long())                      # [B, d]
        vec_action = vec_sel + vec_cross + vec_beast                               # [B, d]

        # -------- Fusion / Head --------
        fusion = torch.cat([vec_grid, vec_scalar, vec_mask, vec_hand, vec_action], dim=-1)  # [B, 5d]
        fusion = self.fusion_norm(fusion)
        return self.head(fusion).squeeze(-1)
