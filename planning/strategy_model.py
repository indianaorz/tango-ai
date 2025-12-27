# planning/strategy_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default Meta Config
ELEMENTS = ["", "Fire", "Aqua", "Elec", "Wood", "Break", "Sword", "Wind", "Cursor", "Plus", "Obstacle", "Invis"]
TYPES = ["Standard", "Mega", "Giga"]
META_DIM = 1 + len(ELEMENTS) + len(TYPES) 

class StrategyTransformer(nn.Module):
    def __init__(self, 
                 num_chip_ids=512, 
                 num_codes=32, 
                 d_model=512, 
                 nhead=8, 
                 num_layers=6, 
                 num_crosses=6,
                 meta_dim=META_DIM,
                 dropout=0.3): 
        super().__init__()
        
        # Embeddings
        self.chip_embedding = nn.Embedding(num_chip_ids, d_model)
        self.code_embedding = nn.Embedding(num_codes, d_model)
        self.pos_embedding = nn.Embedding(10, d_model)
        self.meta_proj = nn.Linear(meta_dim, d_model)
        
        # 🚀 FIX: Context Dimension = 36
        # 2(HP) + 6(History) + 6(Current) + 1(Beast) + 18(Grid) + 1(Sync) + 1(Area) + 1(ECol)
        self.context_dim = 36 
        self.context_proj = nn.Linear(self.context_dim, d_model) 
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=d_model*4, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.token_embedding = nn.Embedding(14, d_model) 
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=d_model*4, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Heads
        self.cross_head = nn.Linear(d_model, num_crosses)
        self.action_head = nn.Linear(d_model, 14)

    def forward(self, hand_ids, hand_codes, hand_meta, context_vec, target_seq=None):
        B = hand_ids.shape[0]
        positions = torch.arange(10, device=hand_ids.device).unsqueeze(0).expand(B, 10)
        
        src = (self.chip_embedding(hand_ids) + 
               self.code_embedding(hand_codes) + 
               self.meta_proj(hand_meta) + 
               self.pos_embedding(positions))
        
        src = src + self.context_proj(context_vec).unsqueeze(1)
        src = self.dropout(src)
        
        memory = self.encoder(src)
        cross_logits = self.cross_head(memory.mean(dim=1))
        
        if target_seq is not None:
            tgt_len = target_seq.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(hand_ids.device)
            tgt_emb = self.token_embedding(target_seq)
            tgt_emb = self.dropout(tgt_emb)
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            return cross_logits, self.action_head(output)
        else:
            return self.inference(memory, cross_logits)

    def inference(self, memory, cross_logits, max_len=6, max_chip_index=10):
        B = memory.shape[0]
        device = memory.device
        curr_token = torch.full((B, 1), 12, dtype=torch.long, device=device) # SOS=12
        generated_indices = []
        
        for _ in range(max_len):
            tgt_emb = self.token_embedding(curr_token)
            output = self.decoder(tgt_emb, memory)
            last_token_logits = self.action_head(output[:, -1, :])
            
            if max_chip_index < 10:
                last_token_logits[:, max_chip_index:10] = float('-inf')
            
            last_token_logits[:, 12] = float('-inf')

            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
            generated_indices.append(next_token)
            curr_token = torch.cat([curr_token, next_token], dim=1)
            
        return cross_logits, torch.cat(generated_indices, dim=1)