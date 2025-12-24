import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import numpy as np

# Import your existing architecture and processor
# Ensure this script is in the 'strategy' folder or adjust imports
try:
    from strategy_model import StrategyTransformer
    from train_strategy_bc import ChipMetaProcessor, META_DIM, CHIPS_DB_PATH
except ImportError:
    print("❌ Error: Could not import model/processor. Run this from the parent directory or fix paths.")
    exit()

# --- CONFIG ---
DATA_PATH = "data/chipwindows/strategy.jsonl"
CHECKPOINT_PATH = "checkpoints_strategy/strategy_model.pt" # Start from BC checkpoint
SAVE_DIR = "checkpoints_strategy_rl"
BATCH_SIZE = 32
EPOCHS = 100 # RL fine-tuning usually takes fewer epochs
LR = 0.0001 # Lower LR for fine-tuning
VAL_SIZE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET WITH REWARDS ---
class RLStrategyDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
        
        # Calculate Reward Statistics for Normalization
        yields = [d.get('net_yield', 0) for d in data]
        self.rew_mean = np.mean(yields)
        self.rew_std = np.std(yields) + 1e-6 # Avoid div/0
        print(f"Dataset Rewards: Mean={self.rew_mean:.2f}, Std={self.rew_std:.2f}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 1. Inputs (Same as BC)
        raw_ids = row['hand_slots']
        raw_codes = [min(c, 31) for c in row['hand_codes']]
        target_indices = row['selected_indices']
        
        # 2. SHUFFLING (Keep this!)
        perm = torch.arange(10)
        if self.augment:
            valid_count = 0
            for x in raw_ids:
                if x != 255: valid_count += 1
                else: break
            
            if valid_count > 1:
                valid_perm = torch.randperm(valid_count)
                remainder = torch.arange(valid_count, 10)
                perm = torch.cat([valid_perm, remainder])

        hand_ids = torch.tensor([raw_ids[i] for i in perm], dtype=torch.long)
        hand_codes = torch.tensor([raw_codes[i] for i in perm], dtype=torch.long)
        
        # 3. Metadata Injection
        meta_list = [meta_proc.get_meta(cid.item()) for cid in hand_ids]
        hand_meta = torch.stack(meta_list)

        # 4. Target Remapping
        new_target_indices = []
        for old_idx in target_indices:
            if old_idx < 0 or old_idx > 9: continue
            matches = (perm == old_idx).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                new_target_indices.append(matches.item())
                
        # 5. Context
        hp_ctx = torch.tensor([row['p_hp_start']/1000.0, row['e_hp_start']/2000.0], dtype=torch.float32)
        used_vec = torch.zeros(6, dtype=torch.float32)
        for c in row.get('used_crosses', []):
            if 1 <= c <= 6: used_vec[c-1] = 1.0
        full_ctx = torch.cat([hp_ctx, used_vec])
        
        # 6. Sequences
        seq_in = [10] + new_target_indices
        seq_tgt = new_target_indices + [11]
        MAX_LEN = 6
        seq_in = seq_in + [11] * (MAX_LEN - len(seq_in)) 
        seq_tgt = seq_tgt + [11] * (MAX_LEN - len(seq_tgt))
        
        # --- 7. REWARD CALCULATION ---
        # Net Yield = Damage Dealt - Damage Taken
        raw_reward = row.get('net_yield', 0)
        
        # Normalize (Z-Score)
        norm_reward = (raw_reward - self.rew_mean) / self.rew_std
        
        # Advantage Weighting: exp(reward) ensures positive weights
        # High reward -> Weight > 1.0
        # Low reward  -> Weight < 1.0 (but > 0)
        weight = np.exp(norm_reward) 
        
        # Clamp to prevent exploding gradients from outlier lucky turns
        weight = max(0.1, min(weight, 5.0))

        return {
            'hand_ids': hand_ids,
            'hand_codes': hand_codes,
            'hand_meta': hand_meta,
            'context_vec': full_ctx,
            'seq_in': torch.tensor(seq_in[:MAX_LEN], dtype=torch.long),
            'seq_tgt': torch.tensor(seq_tgt[:MAX_LEN], dtype=torch.long),
            'cross_tgt': torch.tensor(row.get('selected_cross', 0), dtype=torch.long),
            'weight': torch.tensor(weight, dtype=torch.float32) # <--- THE KEY
        }

# --- GLOBAL PROCESSOR ---
meta_proc = ChipMetaProcessor(CHIPS_DB_PATH)

def load_data():
    dataset = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    else:
        print(f"ERROR: {DATA_PATH} not found.")
        return [], []
    
    random.seed(42)
    random.shuffle(dataset)
    return dataset[VAL_SIZE:], dataset[:VAL_SIZE]

def train():
    train_data, val_data = load_data()
    print(f"RL Fine-Tuning: {len(train_data)} Training, {len(val_data)} Validation")
    
    train_loader = DataLoader(RLStrategyDataset(train_data, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RLStrategyDataset(val_data, augment=False), batch_size=BATCH_SIZE)
    
    # Init Model
    model = StrategyTransformer(
        num_chip_ids=512, num_codes=32, d_model=128, meta_dim=META_DIM
    ).to(DEVICE)
    
    # Load Pre-trained Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Loading checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("⚠️ No checkpoint found! Starting from scratch (Not recommended for RL).")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Weighted Loss Functions (Reduction='none' to apply per-sample weights)
    seq_criterion = nn.CrossEntropyLoss(reduction='none') 
    cross_criterion = nn.CrossEntropyLoss(reduction='none')
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_reward_weight = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move data
            h_ids = batch['hand_ids'].to(DEVICE)
            h_codes = batch['hand_codes'].to(DEVICE)
            h_meta = batch['hand_meta'].to(DEVICE)
            ctx_vec = batch['context_vec'].to(DEVICE)
            seq_in = batch['seq_in'].to(DEVICE)
            seq_tgt = batch['seq_tgt'].to(DEVICE)
            cross_tgt = batch['cross_tgt'].to(DEVICE)
            weights = batch['weight'].to(DEVICE) # [Batch]
            
            # Forward
            cross_pred, seq_pred = model(h_ids, h_codes, h_meta, ctx_vec, seq_in)
            
            # Calculate Raw Losses [Batch]
            # Seq: [B, T, V] -> [B, T] -> Sum/Mean over T? Let's mean over T first.
            loss_seq_per_sample = seq_criterion(seq_pred.permute(0, 2, 1), seq_tgt).mean(dim=1)
            loss_cross_per_sample = cross_criterion(cross_pred, cross_tgt)
            
            # Apply RL Weights
            # "Important" samples get multiplied by >1.0
            # "Bad" samples get multiplied by <1.0
            weighted_loss = ((loss_seq_per_sample + loss_cross_per_sample) * weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            total_reward_weight += weights.mean().item()
            
        # Logging
        if (epoch+1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_weight = total_reward_weight / len(train_loader)
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Avg Weight {avg_weight:.2f}")
            
            # Validation (Just check accuracy to ensure we aren't breaking the model)
            val_acc_seq = 0
            total_tokens = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    # ... [Inference setup same as BC] ...
                    h_ids = batch['hand_ids'].to(DEVICE)
                    h_codes = batch['hand_codes'].to(DEVICE)
                    h_meta = batch['hand_meta'].to(DEVICE)
                    ctx_vec = batch['context_vec'].to(DEVICE)
                    seq_tgt = batch['seq_tgt'].to(DEVICE)
                    
                    B = h_ids.shape[0]
                    pos = torch.arange(10, device=DEVICE).unsqueeze(0).expand(B, 10)
                    src = (model.chip_embedding(h_ids) + model.code_embedding(h_codes) + 
                           model.meta_proj(h_meta) + model.pos_embedding(pos)) + \
                           model.context_proj(ctx_vec).unsqueeze(1)
                    
                    memory = model.encoder(src)
                    cross_logits, seq_tokens = model.inference(memory, model.cross_head(memory.mean(dim=1)))
                    
                    limit = min(seq_tokens.shape[1], seq_tgt.shape[1])
                    val_acc_seq += (seq_tokens[:, :limit] == seq_tgt[:, :limit]).sum().item()
                    total_tokens += (seq_tokens.shape[0] * limit)
            
            acc = val_acc_seq / total_tokens if total_tokens > 0 else 0
            print(f"   Val Seq Acc: {acc:.2f}")

    # Save RL-Tuned Model
    save_path = os.path.join(SAVE_DIR, "strategy_model_rl.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved RL-tuned model to {save_path}")

if __name__ == "__main__":
    train()