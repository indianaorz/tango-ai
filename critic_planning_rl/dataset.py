import torch
import json
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path

class PlanningDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None, shuffle_hand=False):
        self.data = []
        self.shuffle_hand = shuffle_hand  # Data Augmentation flag
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                self.data.append(json.loads(line))
                if max_samples and len(self.data) >= max_samples: break
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # --- SCALARS ---
        p_hp = float(row.get('p_hp_open', 0)) / 1000.0
        e_hp = float(row.get('e_hp_open', 0)) / 1000.0
        turn = float(row.get('turn_index_open', 0)) / 50.0
        
        # --- GRID ---
        grid_tile = torch.tensor(row.get('grid_tile_open', [0]*18), dtype=torch.long).clamp(0, 31)
        grid_owner = torch.tensor(row.get('grid_owner_open', [2]*18), dtype=torch.long).clamp(0, 2)
        
        # --- MASKS ---
        def to_float(k, n):
            lst = row.get(k, [])
            return torch.tensor([1.0 if x else 0.0 for x in lst] + [0.0]*max(0, n-len(lst)))[:n]
            
        folder_p = to_float('folder_used_mask_p_open', 30)
        folder_e = to_float('folder_used_mask_e_open', 30)
        cross_p = to_float('used_cross_mask_p_open', 11)
        cross_e = to_float('used_cross_mask_e_open', 11)
        
        # --- HAND (CONTEXT) ---
        held_id = torch.tensor(row.get('held_before_id', [0]*5), dtype=torch.long)
        held_code = torch.tensor(row.get('held_before_code', [0]*5), dtype=torch.long).clamp(0, 127)
        
        # Draw Logic
        draw_id_raw = row.get('window_hand_id', [0]*10)
        draw_code_raw = row.get('window_hand_code', [0]*10)
        vis = row.get('window_hand_vis', [0.0]*10)
        
        # Extract only visible chips
        visible_chips = []
        for i in range(10):
            if vis[i] > 0.5:
                visible_chips.append((draw_id_raw[i], draw_code_raw[i]))
        
        # DATA AUGMENTATION: Shuffle the visible chips order
        # This teaches the model that "Cannon in Slot 1" == "Cannon in Slot 5"
        if self.shuffle_hand:
            random.shuffle(visible_chips)
            
        # Reconstruct padded list (10 slots)
        draw_id = []
        draw_code = []
        for i in range(10):
            if i < len(visible_chips):
                draw_id.append(visible_chips[i][0])
                draw_code.append(visible_chips[i][1])
            else:
                draw_id.append(0)
                draw_code.append(0)
                
        draw_id = torch.tensor(draw_id, dtype=torch.long)
        draw_code = torch.tensor(draw_code, dtype=torch.long).clamp(0, 127)
        
        # --- ACTION ---
        sel_id = torch.tensor(row.get('selected_chips_id', [0]*5), dtype=torch.long)
        sel_code = torch.tensor(row.get('selected_chips_code', [0]*5), dtype=torch.long).clamp(0, 127)
        
        sel_cross_val = int(row.get('selected_cross', 0))
        sel_cross = torch.tensor(max(0, min(sel_cross_val, 63)), dtype=torch.long)
        
        sel_beast = torch.tensor(1 if row.get('beast_selected') else 0, dtype=torch.long)
        
        # --- TARGET ---
        ny = float(row.get('net_yield', 0))
        target = np.sign(ny) * np.log1p(np.abs(ny))
        
        return {
            'p_hp': torch.tensor(p_hp, dtype=torch.float32),
            'e_hp': torch.tensor(e_hp, dtype=torch.float32),
            'turn_idx': torch.tensor(turn, dtype=torch.float32),
            'grid_tile': grid_tile,
            'grid_owner': grid_owner,
            'folder_p': folder_p,
            'folder_e': folder_e,
            'cross_hist_p': cross_p,
            'cross_hist_e': cross_e,
            'held_id': held_id,
            'held_code': held_code,
            'draw_id': draw_id,
            'draw_code': draw_code,
            'sel_id': sel_id,
            'sel_code': sel_code,
            'sel_cross': sel_cross,
            'sel_beast': sel_beast,
            'target': torch.tensor(target, dtype=torch.float32)
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}