from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class PlanningConfig:
    # Sizes
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    
    # Vocabs (INCREASED FOR SAFETY)
    chip_buckets: int = 8192  
    code_vocab: int = 128     # Was 64, data has 63. Bump to 128 to be safe.
    cross_vocab: int = 64     # Was 24, data has 24 (index 25). Bump to 64.
    tile_vocab: int = 32
    owner_vocab: int = 16     # Was 3. Data is fine, but cheap to raise.
    
    # Dims
    chip_dim: int = 64
    code_dim: int = 16
    
    # Action Sequence
    max_selected: int = 5

class HashEmbedding(nn.Module):
    def __init__(self, num_buckets, dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, dim)
    def forward(self, x):
        return self.emb(x % self.num_buckets)

class PlanningCritic(nn.Module):
    def __init__(self, cfg: PlanningConfig):
        super().__init__()
        self.cfg = cfg
        
        # --- Shared Embeddings ---
        self.chip_emb = HashEmbedding(cfg.chip_buckets, cfg.chip_dim)
        self.code_emb = nn.Embedding(cfg.code_vocab, cfg.code_dim)
        # Project chip+code -> d_model
        self.chip_proj = nn.Linear(cfg.chip_dim + cfg.code_dim, cfg.d_model)
        
        # --- Context Tower (State) ---
        # Scalars: P_HP, E_HP, Turn (3 floats)
        # Grid: 18 tiles * (TileEmb + OwnerEmb)
        self.tile_emb = nn.Embedding(cfg.tile_vocab, 16)
        self.owner_emb = nn.Embedding(cfg.owner_vocab, 8)
        self.grid_proj = nn.Linear(18 * (16 + 8), cfg.d_model)
        
        self.scalar_proj = nn.Linear(3, cfg.d_model)
        
        # Masks (Folder + Cross History)
        # 30 (P Folder) + 30 (E Folder) + 11 (P Cross) + 11 (E Cross) = 82 bits
        self.mask_proj = nn.Linear(82, cfg.d_model)
        
        # --- Hand Tower (Available Options) ---
        self.hand_pos = nn.Parameter(torch.randn(1, 15, cfg.d_model))
        self.hand_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.d_model*4, cfg.dropout, batch_first=True),
            num_layers=cfg.num_layers
        )
        
        # --- Action Tower (The Decision) ---
        self.action_pos = nn.Parameter(torch.randn(1, 5, cfg.d_model))
        self.cross_emb = nn.Embedding(cfg.cross_vocab, cfg.d_model)
        self.beast_emb = nn.Embedding(2, cfg.d_model)
        
        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.d_model*4, cfg.dropout, batch_first=True),
            num_layers=cfg.num_layers
        )
        
        # --- Fusion Head ---
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 5, 512), 
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1) 
        )

    def _embed_chips(self, ids, codes):
        c = self.chip_emb(ids)
        o = self.code_emb(codes)
        return self.chip_proj(torch.cat([c, o], dim=-1))

    def forward(self, batch):
        B = batch['p_hp'].shape[0]
        
        # 1. Context Features
        g_t = self.tile_emb(batch['grid_tile']) 
        g_o = self.owner_emb(batch['grid_owner']) 
        grid_flat = torch.cat([g_t, g_o], dim=-1).flatten(1) 
        vec_grid = self.grid_proj(grid_flat)
        
        scalars = torch.stack([batch['p_hp'], batch['e_hp'], batch['turn_idx']], dim=-1)
        vec_scalar = self.scalar_proj(scalars)
        
        masks = torch.cat([
            batch['folder_p'], batch['folder_e'], 
            batch['cross_hist_p'], batch['cross_hist_e']
        ], dim=-1)
        vec_mask = self.mask_proj(masks)
        
        # 2. Hand Encoder
        raw_hand_ids = torch.cat([batch['held_id'], batch['draw_id']], dim=1)
        raw_hand_codes = torch.cat([batch['held_code'], batch['draw_code']], dim=1)
        
        hand_embeds = self._embed_chips(raw_hand_ids, raw_hand_codes)
        hand_embeds = hand_embeds + self.hand_pos
        hand_out = self.hand_encoder(hand_embeds)
        vec_hand = hand_out.mean(dim=1) 
        
        # 3. Action Encoder
        sel_embeds = self._embed_chips(batch['sel_id'], batch['sel_code'])
        sel_embeds = sel_embeds + self.action_pos
        
        sel_out = self.action_encoder(sel_embeds)
        vec_sel_chips = sel_out.mean(dim=1) 
        
        vec_cross = self.cross_emb(batch['sel_cross'])
        vec_beast = self.beast_emb(batch['sel_beast'].long())
        
        vec_action = vec_sel_chips + vec_cross + vec_beast
        
        # 4. Fusion
        fusion = torch.cat([vec_grid, vec_scalar, vec_mask, vec_hand, vec_action], dim=-1)
        return self.head(fusion).squeeze(-1)