# strategy/strategy_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyTransformer(nn.Module):
    def __init__(self, 
                 num_chip_ids=512, 
                 num_codes=32, 
                 d_model=128, 
                 nhead=4, 
                 num_layers=2, 
                 num_crosses=6,
                 meta_dim=16): # <--- ADDED THIS ARGUMENT
        super().__init__()
        
        # Embeddings
        self.chip_embedding = nn.Embedding(num_chip_ids, d_model)
        self.code_embedding = nn.Embedding(num_codes, d_model)
        self.pos_embedding = nn.Embedding(10, d_model)
        
        # New: Metadata Projection Layer
        self.meta_proj = nn.Linear(meta_dim, d_model)
        
        # Context (HP + Used Crosses)
        self.context_dim = 2 + 6 
        self.context_proj = nn.Linear(self.context_dim, d_model) 
        
        # Transformer Architecture
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.token_embedding = nn.Embedding(12, d_model) 
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output Heads
        self.cross_head = nn.Linear(d_model, num_crosses)
        self.action_head = nn.Linear(d_model, 12)

    def forward(self, hand_ids, hand_codes, hand_meta, context_vec, target_seq=None):
        """
        hand_meta: [Batch, 10, META_DIM]
        """
        B = hand_ids.shape[0]
        
        # 1. ENCODE HAND
        positions = torch.arange(10, device=hand_ids.device).unsqueeze(0).expand(B, 10)
        
        # Sum all features: ID + Code + Meta + Pos
        src = (self.chip_embedding(hand_ids) + 
               self.code_embedding(hand_codes) + 
               self.meta_proj(hand_meta) +   # <--- INJECT METADATA
               self.pos_embedding(positions))
        
        # 2. INJECT CONTEXT
        ctx_emb = self.context_proj(context_vec).unsqueeze(1)
        src = src + ctx_emb
        
        memory = self.encoder(src)
        
        # 3. PREDICT
        cross_logits = self.cross_head(memory.mean(dim=1))
        
        if target_seq is not None:
            tgt_len = target_seq.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(hand_ids.device)
            output = self.decoder(self.token_embedding(target_seq), memory, tgt_mask=tgt_mask)
            return cross_logits, self.action_head(output)
        else:
            return self.inference(memory, cross_logits)

    def inference(self, memory, cross_logits, max_len=6):
        B = memory.shape[0]
        device = memory.device
        curr_token = torch.full((B, 1), 10, dtype=torch.long, device=device)
        generated_indices = []
        
        for _ in range(max_len):
            tgt_emb = self.token_embedding(curr_token)
            output = self.decoder(tgt_emb, memory)
            last_token_logits = self.action_head(output[:, -1, :])
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
            generated_indices.append(next_token)
            curr_token = torch.cat([curr_token, next_token], dim=1)
            
        return cross_logits, torch.cat(generated_indices, dim=1)
    
