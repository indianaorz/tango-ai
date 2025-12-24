# strategy/train_strategy_bc.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random

# --- CONFIG ---
DATA_PATH = "data/chipwindows/strategy.jsonl"
CHIPS_DB_PATH = "data/assets/chips.json" # <--- POINT THIS TO YOUR JSON
BATCH_SIZE = 32
EPOCHS = 5000
LR = 0.001
VAL_SIZE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Meta Config
ELEMENTS = ["", "Fire", "Aqua", "Elec", "Wood", "Break", "Sword", "Wind", "Cursor", "Plus", "Obstacle", "Invis"]
TYPES = ["Standard", "Mega", "Giga"]
META_DIM = 1 + len(ELEMENTS) + len(TYPES) # 1 (Dmg) + 12 (Elem) + 3 (Type) = 16

# --- 1. CHIP METADATA PROCESSOR ---
class ChipMetaProcessor:
    def __init__(self, json_path):
        self.db = {} # Map ID -> Vector
        self.load_db(json_path)
        
    def load_db(self, path):
        if not os.path.exists(path):
            print(f"⚠️ Warning: Chips DB not found at {path}. Metadata will be empty.")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"✅ Loaded metadata for {len(data)} chips.")
        
        for chip in data:
            vec = self._build_vector(chip)
            
            # Safe Setter Helper
            def set_id(raw_val):
                if raw_val is None: return
                s_val = str(raw_val).strip()
                if not s_val: return # Skip empty strings
                try:
                    self.db[int(s_val)] = vec
                except ValueError:
                    pass # Skip non-integers

            set_id(chip.get('SId'))
            set_id(chip.get('MId'))
                
    def _build_vector(self, chip):
        # 1. Damage (Normalize / 500)
        dmg_str = str(chip.get('Damage', '0'))
        if '-' in dmg_str: dmg_str = dmg_str.split('-')[1] # "10-20" -> 20
        try:
            # Handle "X" or "?" or empty
            if not dmg_str.strip() or not dmg_str[0].isdigit():
                dmg_val = 0.0
            else:
                dmg_val = float(dmg_str) / 500.0
        except:
            dmg_val = 0.0
            
        # 2. Element (One-Hot)
        elem_vec = [0.0] * len(ELEMENTS)
        c_elem = chip.get('Element', "")
        if c_elem in ELEMENTS:
            elem_vec[ELEMENTS.index(c_elem)] = 1.0
        else:
            elem_vec[0] = 1.0 # Default to None
            
        # 3. Type (One-Hot)
        type_vec = [0.0] * 3
        
        # Safe MB Parsing
        try:
            mb_str = str(chip.get('MB', '0')).strip()
            if not mb_str: 
                mb_val = 0
            else:
                mb_val = int(float(mb_str)) # Handle "10.0" strings just in case
        except:
            mb_val = 0

        # Heuristic for Type
        if chip.get('MId') is not None and str(chip.get('MId')).strip():
            if "Giga" in chip.get('Version', '') or mb_val > 80:
                type_vec[2] = 1.0 # Giga
            else:
                type_vec[1] = 1.0 # Mega
        else:
            type_vec[0] = 1.0 # Standard
            
        return torch.tensor([dmg_val] + elem_vec + type_vec, dtype=torch.float32)

    def get_meta(self, chip_id):
        if chip_id in self.db:
            return self.db[chip_id]
        return torch.zeros(META_DIM, dtype=torch.float32)
# Global Processor
meta_proc = ChipMetaProcessor(CHIPS_DB_PATH)


# --- 2. UPDATED MODEL ---
class StrategyTransformer(nn.Module):
    def __init__(self, 
                 num_chip_ids=512, 
                 num_codes=32, 
                 d_model=128, 
                 nhead=4, 
                 num_layers=2, 
                 num_crosses=6,
                 meta_dim=META_DIM): 
        super().__init__()
        
        # Embeddings
        self.chip_embedding = nn.Embedding(num_chip_ids, d_model)
        self.code_embedding = nn.Embedding(num_codes, d_model)
        self.pos_embedding = nn.Embedding(10, d_model)
        
        # New: Metadata Projection
        self.meta_proj = nn.Linear(meta_dim, d_model)
        
        # Context
        self.context_dim = 2 + 6 # HP + UsedCrosses
        self.context_proj = nn.Linear(self.context_dim, d_model) 
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.token_embedding = nn.Embedding(12, d_model) 
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Heads
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
               self.meta_proj(hand_meta) +   # <--- INJECTED HERE
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


# --- 3. UPDATED DATASET ---
class StrategyDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 1. Load Raw Data
        raw_ids = row['hand_slots']
        raw_codes = [min(c, 31) for c in row['hand_codes']]
        target_indices = row['selected_indices']
        
        # 2. DATA AUGMENTATION: Shuffle Only Valid Chips
        perm = torch.arange(10)
        
        if self.augment:
            # Count how many actual chips we have (ignore 255/Empty)
            # We assume chips are left-aligned, so we just count non-255s
            valid_count = 0
            for x in raw_ids:
                if x != 255: valid_count += 1
                else: break # Stop at first empty
            
            # Shuffle ONLY the valid range [0, valid_count)
            if valid_count > 1:
                valid_perm = torch.randperm(valid_count)
                # Remainder stays fixed [valid_count, 10)
                remainder = torch.arange(valid_count, 10)
                perm = torch.cat([valid_perm, remainder])

        # Apply permutation (Same as before)
        hand_ids = torch.tensor([raw_ids[i] for i in perm], dtype=torch.long)
        hand_codes = torch.tensor([raw_codes[i] for i in perm], dtype=torch.long)
        
        # --- NEW: Build Metadata Tensor ---
        meta_list = []
        for chip_id in hand_ids:
            # Look up the ID in our processor
            meta_vec = meta_proc.get_meta(chip_id.item())
            meta_list.append(meta_vec)
        hand_meta = torch.stack(meta_list) # [10, META_DIM]

        # Target Remapping
        new_target_indices = []
        for old_idx in target_indices:
            if old_idx < 0 or old_idx > 9: continue
            matches = (perm == old_idx).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                new_target_indices.append(matches.item())
                
        # Context
        hp_ctx = torch.tensor([row['p_hp_start']/1000.0, row['e_hp_start']/2000.0], dtype=torch.float32)
        
        used_cross_vec = torch.zeros(6, dtype=torch.float32)
        for c in row.get('used_crosses', []):
            if 1 <= c <= 6: used_cross_vec[c-1] = 1.0
            
        full_ctx = torch.cat([hp_ctx, used_cross_vec])
        
        # Sequences
        seq_in = [10] + new_target_indices
        seq_tgt = new_target_indices + [11]
        
        MAX_LEN = 6
        seq_in = seq_in + [11] * (MAX_LEN - len(seq_in)) 
        seq_tgt = seq_tgt + [11] * (MAX_LEN - len(seq_tgt))
        
        return {
            'hand_ids': hand_ids,
            'hand_codes': hand_codes,
            'hand_meta': hand_meta, # <--- NEW INPUT
            'context_vec': full_ctx,
            'seq_in': torch.tensor(seq_in[:MAX_LEN], dtype=torch.long),
            'seq_tgt': torch.tensor(seq_tgt[:MAX_LEN], dtype=torch.long),
            'cross_tgt': torch.tensor(row.get('selected_cross', 0), dtype=torch.long)
        }


def load_data():
    dataset = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    random.seed(42)
    random.shuffle(dataset)
    return dataset[VAL_SIZE:], dataset[:VAL_SIZE]

def train():
    train_data, val_data = load_data()
    print(f"Training on {len(train_data)}, Val on {len(val_data)}")
    
    train_loader = DataLoader(StrategyDataset(train_data, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StrategyDataset(val_data, augment=False), batch_size=BATCH_SIZE)
    
    model = StrategyTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    seq_criterion = nn.CrossEntropyLoss()
    cross_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            h_ids = batch['hand_ids'].to(DEVICE)
            h_codes = batch['hand_codes'].to(DEVICE)
            h_meta = batch['hand_meta'].to(DEVICE) # <--- NEW
            ctx_vec = batch['context_vec'].to(DEVICE)
            seq_in = batch['seq_in'].to(DEVICE)
            seq_tgt = batch['seq_tgt'].to(DEVICE)
            cross_tgt = batch['cross_tgt'].to(DEVICE)
            
            cross_pred, seq_pred = model(h_ids, h_codes, h_meta, ctx_vec, seq_in)
            
            loss = seq_criterion(seq_pred.reshape(-1, 12), seq_tgt.reshape(-1)) + cross_criterion(cross_pred, cross_tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            val_acc_seq = 0
            val_acc_cross = 0
            total_tokens = 0
            
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    h_ids = batch['hand_ids'].to(DEVICE)
                    h_codes = batch['hand_codes'].to(DEVICE)
                    h_meta = batch['hand_meta'].to(DEVICE)
                    ctx_vec = batch['context_vec'].to(DEVICE)
                    cross_tgt = batch['cross_tgt'].to(DEVICE)
                    seq_tgt = batch['seq_tgt'].to(DEVICE)
                    
                    # Inference
                    B = h_ids.shape[0]
                    pos = torch.arange(10, device=DEVICE).unsqueeze(0).expand(B, 10)
                    
                    # Manual embedding construction for inference
                    src = (model.chip_embedding(h_ids) + 
                           model.code_embedding(h_codes) + 
                           model.meta_proj(h_meta) + 
                           model.pos_embedding(pos))
                    src = src + model.context_proj(ctx_vec).unsqueeze(1)
                    
                    memory = model.encoder(src)
                    cross_logits, seq_tokens = model.inference(memory, model.cross_head(memory.mean(dim=1)))
                    
                    # Acc
                    val_acc_cross += (torch.argmax(cross_logits, 1) == cross_tgt).sum().item()
                    limit = min(seq_tokens.shape[1], seq_tgt.shape[1])
                    val_acc_seq += (seq_tokens[:, :limit] == seq_tgt[:, :limit]).sum().item()
                    total_tokens += (seq_tokens.shape[0] * limit)
                    
                    if i == 0:
                        print(f"\n--- Ep {epoch+1} Debug ---")
                        # Show raw IDs so we can see what the model is looking at
                        print(f"IDs: {h_ids[0].tolist()}")
                        # Show Target (filter out EOS)
                        tgt_clean = [x for x in seq_tgt[0].tolist() if x != 11]
                        print(f"TGT: {tgt_clean}")
                        # Show Pred (filter out EOS)
                        pred_clean = []
                        for x in seq_tokens[0].tolist():
                            if x == 11: break
                            pred_clean.append(x)
                        print(f"PRD: {pred_clean}")

            # Safe Division
            avg_loss = total_loss / len(train_loader)
            cross_acc = val_acc_cross / len(val_data)
            seq_acc = val_acc_seq / total_tokens if total_tokens > 0 else 0
            
            print(f"Ep {epoch+1}: Loss {avg_loss:.4f} | Cross {cross_acc:.2f} | Seq {seq_acc:.2f}")

    # Save Model
    save_dir = "checkpoints_strategy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "strategy_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()