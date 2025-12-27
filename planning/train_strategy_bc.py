import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import time

from strategy_model import StrategyTransformer, META_DIM, ELEMENTS

# --- CONFIG ---
DATA_PATH = "data/chipwindows/strategy.jsonl"
CHIPS_DB_PATH = "data/assets/chips.json"
BATCH_SIZE = 2048         
EPOCHS = 5000
LR = 0.0005               
VAL_SIZE = 50
CHECKPOINT_EVERY = 50
NUM_WORKERS = 8           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

class ChipMetaProcessor:
    def __init__(self, json_path):
        self.db = {} 
        self.load_db(json_path)
    def load_db(self, path):
        if not os.path.exists(path): return
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        for chip in data:
            vec = self._build_vector(chip)
            def set_id(raw_val):
                if raw_val is None: return
                try: self.db[int(str(raw_val).strip())] = vec
                except: pass 
            set_id(chip.get('SId'))
            set_id(chip.get('MId'))
    def _build_vector(self, chip):
        try: dmg = float(str(chip.get('Damage', '0')).split('-')[1] if '-' in str(chip.get('Damage', '0')) else str(chip.get('Damage', '0'))) / 500.0
        except: dmg = 0.0
        elem = [0.0]*len(ELEMENTS)
        if chip.get('Element') in ELEMENTS: elem[ELEMENTS.index(chip.get('Element'))] = 1.0
        else: elem[0] = 1.0
        ctype = [0.0]*3
        ctype[0] = 1.0 
        return torch.tensor([dmg] + elem + ctype, dtype=torch.float32)
    def get_meta(self, chip_id):
        return self.db.get(chip_id, torch.zeros(META_DIM, dtype=torch.float32))

meta_proc = ChipMetaProcessor(CHIPS_DB_PATH)

class StrategyDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # --- A. LOAD & SHUFFLE HAND ---
        raw_ids = row['hand_slots']
        raw_codes = [min(c, 31) for c in row['hand_codes']]
        target_indices = row['selected_indices']
        
        perm = torch.arange(10)
        
        if self.augment:
            non_empty_count = 0
            for x in raw_ids:
                if x != 255: non_empty_count += 1
                else: break 
            visible_limit = int(row.get('chip_visible_count', 10))
            shuffle_limit = min(non_empty_count, visible_limit)
            if shuffle_limit > 1:
                valid_perm = torch.randperm(shuffle_limit)
                remainder = torch.arange(shuffle_limit, 10)
                perm = torch.cat([valid_perm, remainder])

        hand_ids = torch.tensor([raw_ids[i] for i in perm], dtype=torch.long)
        hand_codes = torch.tensor([raw_codes[i] for i in perm], dtype=torch.long)
        
        visible_limit = int(row.get('chip_visible_count', 10))
        if visible_limit < 10:
            hand_ids[visible_limit:] = 255
            hand_codes[visible_limit:] = 0

        meta_list = [meta_proc.get_meta(cid.item()) for cid in hand_ids]
        hand_meta = torch.stack(meta_list)

        new_target_indices = []
        for old_idx in target_indices:
            if old_idx < 0 or old_idx > 9: continue
            matches = (perm == old_idx).nonzero(as_tuple=True)[0]
            if len(matches) > 0: new_target_indices.append(matches.item())
                
        # --- B. BUILD EXTENDED CONTEXT (36 Dims) ---
        # 1. HP (2)
        hp_ctx = [row['p_hp_start']/1000.0, row['e_hp_start']/2000.0]
        
        # 2. Used Crosses History (6)
        used_cross_vec = [0.0] * 6
        used_list = row.get('player_used_crosses', [])
        for c in used_list:
            if 1 <= c <= 6: used_cross_vec[c-1] = 1.0
            
        # 3. Current Active Cross (6) - One Hot
        # If dataset doesn't have explicit 'current_cross', try to infer or default to 0
        current_cross_id = row.get('current_cross', 0) 
        # Fallback heuristic: If we used a cross, maybe we are still in it? (Imperfect but okay)
        if current_cross_id == 0 and len(used_list) > 0:
             # This is a weak guess, but better than nothing if field missing
             pass 
             
        current_cross_vec = [0.0] * 6
        if 0 <= current_cross_id <= 5:
            current_cross_vec[current_cross_id] = 1.0
        else:
            current_cross_vec[0] = 1.0 # Default Normal Form

        # 4. Beast Mode (1)
        beast_val = 1.0 if row.get('beast_mode', 0) > 0 else 0.0

        # 5. Grid State (18)
        grid_raw = row.get('grid_state', [0]*18)
        grid_vec = [x / 10.0 for x in grid_raw]
        
        # 6. Full Synchro (1)
        is_full_sync = 1.0 if row.get('player_emotion', 0) == 1 else 0.0
        
        # 7. Area Advantage (1)
        owner_state = row.get('grid_owner_state', [0]*18)
        enemy_panels = sum(owner_state)
        area_adv = ((18 - enemy_panels) - 9.0) / 9.0
        
        # 8. Enemy Column (1)
        e_pos = row.get('enemy_pos', [0, 0])
        e_col = max(0, min(5, int((e_pos[0] - 40) / 40)))
        e_col_norm = e_col / 5.0
        
        # [HP(2) + Hist(6) + Curr(6) + Beast(1) + Grid(18) + Sync(1) + Area(1) + ECol(1)] = 36
        full_ctx_list = hp_ctx + used_cross_vec + current_cross_vec + [beast_val] + grid_vec + [is_full_sync, area_adv, e_col_norm]
        full_ctx = torch.tensor(full_ctx_list, dtype=torch.float32)
        
        MAX_LEN = 6
        SOS, EOS = 12, 13
        seq_in = [SOS] + new_target_indices
        seq_tgt = new_target_indices + [EOS]
        seq_in = seq_in + [EOS] * (MAX_LEN - len(seq_in)) 
        seq_tgt = seq_tgt + [EOS] * (MAX_LEN - len(seq_tgt))
        
        return {
            'hand_ids': hand_ids,
            'hand_codes': hand_codes,
            'hand_meta': hand_meta,
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
                if line.strip(): dataset.append(json.loads(line))
    random.seed(42)
    random.shuffle(dataset)
    return dataset[VAL_SIZE:], dataset[:VAL_SIZE]

def train():
    train_data, val_data = load_data()
    print(f"Dataset: {len(train_data)} Train, {len(val_data)} Val")
    
    train_loader = DataLoader(
        StrategyDataset(train_data, augment=True), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        StrategyDataset(val_data, augment=False), 
        batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
    )
    
    model = StrategyTransformer().to(DEVICE)
    try: model = torch.compile(model)
    except: pass
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    seq_criterion = nn.CrossEntropyLoss()
    cross_criterion = nn.CrossEntropyLoss()
    
    save_dir = "checkpoints_strategy"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    print(f"🔥 Starting Training (Batch {BATCH_SIZE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            h_ids = batch['hand_ids'].to(DEVICE, non_blocking=True)
            h_codes = batch['hand_codes'].to(DEVICE, non_blocking=True)
            h_meta = batch['hand_meta'].to(DEVICE, non_blocking=True)
            ctx_vec = batch['context_vec'].to(DEVICE, non_blocking=True)
            seq_in = batch['seq_in'].to(DEVICE, non_blocking=True)
            seq_tgt = batch['seq_tgt'].to(DEVICE, non_blocking=True)
            cross_tgt = batch['cross_tgt'].to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                cross_pred, seq_pred = model(h_ids, h_codes, h_meta, ctx_vec, seq_in)
                loss = seq_criterion(seq_pred.reshape(-1, 14), seq_tgt.reshape(-1)) + \
                       cross_criterion(cross_pred, cross_tgt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            val_acc_seq = 0
            val_acc_cross = 0
            total_tokens = 0
            model.eval()
            with torch.no_grad():
                debug_printed = False
                for batch in val_loader:
                    h_ids = batch['hand_ids'].to(DEVICE, non_blocking=True)
                    h_codes = batch['hand_codes'].to(DEVICE, non_blocking=True)
                    h_meta = batch['hand_meta'].to(DEVICE, non_blocking=True)
                    ctx_vec = batch['context_vec'].to(DEVICE, non_blocking=True)
                    cross_tgt = batch['cross_tgt'].to(DEVICE, non_blocking=True)
                    seq_tgt = batch['seq_tgt'].to(DEVICE, non_blocking=True)
                    
                    B = h_ids.shape[0]
                    pos = torch.arange(10, device=DEVICE).unsqueeze(0).expand(B, 10)
                    with torch.cuda.amp.autocast():
                        src = (model.chip_embedding(h_ids) + model.code_embedding(h_codes) + 
                               model.meta_proj(h_meta) + model.pos_embedding(pos))
                        src = src + model.context_proj(ctx_vec).unsqueeze(1)
                        memory = model.encoder(src)
                        cross_logits, seq_tokens = model.inference(memory, model.cross_head(memory.mean(dim=1)))
                    
                    val_acc_cross += (torch.argmax(cross_logits, 1) == cross_tgt).sum().item()
                    limit = min(seq_tokens.shape[1], seq_tgt.shape[1])
                    val_acc_seq += (seq_tokens[:, :limit] == seq_tgt[:, :limit]).sum().item()
                    total_tokens += (seq_tokens.shape[0] * limit)

                    if not debug_printed:
                        debug_printed = True
                        print(f"\n--- DEBUG EPOCH {epoch+1} ---")
                        print(f"Hand IDs: {h_ids[0].cpu().numpy().tolist()}")
                        print(f"TGT Seq : {seq_tgt[0].cpu().numpy().tolist()}") 
                        print(f"PRED Seq: {seq_tokens[0].cpu().numpy().tolist()}")

            epoch_dur = time.time() - start_time
            print(f"Ep {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Cross {val_acc_cross/len(val_data):.2f} | Seq {val_acc_seq/total_tokens:.2f} | {len(train_data)/epoch_dur:.0f} samp/s")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"strategy_model_ep{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(save_dir, "strategy_model.pt"))
    print("🔥 Done.")

if __name__ == "__main__":
    train()