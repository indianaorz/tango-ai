from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import model definition from your training package
from critic_planning_rl.model import PlanningCritic, PlanningConfig

class PlanningCriticRunner:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load Checkpoint
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Planning Critic ckpt not found: {ckpt_path}")
            
        print(f"[PlanningCritic] Loading {ckpt_path} on {self.device}...")
        state = torch.load(ckpt_path, map_location="cpu")
        
        # Init Model
        self.model = PlanningCritic(PlanningConfig())
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        print("[PlanningCritic] Loaded successfully.")

    def infer_rows(self, rows: List[Dict[str, Any]]) -> List[Optional[float]]:
        """
        Runs inference on a list of strategy_v2 dictionaries.
        Returns a list of predicted Net Yields (Real HP, not SymLog).
        """
        if not rows:
            return []

        # 1. Tensorize Batch (Matches dataset.py logic exactly)
        batch = self._tensorize_batch(rows)
        
        # 2. Inference
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            symlog_preds = self.model(batch) # [B]
            
        # 3. Inverse SymLog Transform
        # y = sign(x) * (exp(|x|) - 1)
        preds = symlog_preds.cpu().numpy()
        real_yields = np.sign(preds) * (np.expm1(np.abs(preds)))
        
        return [float(x) for x in real_yields]

    def _tensorize_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        p_hp_list, e_hp_list, turn_list = [], [], []
        grid_t_list, grid_o_list = [], []
        fp_list, fe_list, cp_list, ce_list = [], [], [], []
        held_id_list, held_code_list = [], []
        draw_id_list, draw_code_list = [], []
        sel_id_list, sel_code_list, sel_cross_list, sel_beast_list = [], [], [], []

        for row in rows:
            # --- SCALARS ---
            p_hp_list.append(float(row.get('p_hp_open', 0)) / 1000.0)
            e_hp_list.append(float(row.get('e_hp_open', 0)) / 1000.0)
            turn_list.append(float(row.get('turn_index_open', 0)) / 50.0)

            # --- GRID ---
            grid_t_list.append(torch.tensor(row.get('grid_tile_open', [0]*18), dtype=torch.long).clamp(0, 31))
            grid_o_list.append(torch.tensor(row.get('grid_owner_open', [2]*18), dtype=torch.long).clamp(0, 2))

            # --- MASKS ---
            def to_float(k, n):
                lst = row.get(k, [])
                return torch.tensor([1.0 if x else 0.0 for x in lst] + [0.0]*max(0, n-len(lst)))[:n]

            fp_list.append(to_float('folder_used_mask_p_open', 30))
            fe_list.append(to_float('folder_used_mask_e_open', 30))
            cp_list.append(to_float('used_cross_mask_p_open', 11))
            ce_list.append(to_float('used_cross_mask_e_open', 11))

            # --- HAND (CONTEXT) ---
            held_id_list.append(torch.tensor(row.get('held_before_id', [0]*5), dtype=torch.long))
            held_code_list.append(torch.tensor(row.get('held_before_code', [0]*5), dtype=torch.long).clamp(0, 127))

            # Draw Logic (Apply Visibility Mask)
            draw_id_raw = row.get('window_hand_id', [0]*10)
            draw_code_raw = row.get('window_hand_code', [0]*10)
            vis = row.get('window_hand_vis', [0.0]*10)
            
            d_ids, d_codes = [], []
            for i in range(10):
                is_vis = (vis[i] > 0.5)
                d_ids.append(draw_id_raw[i] if is_vis else 0)
                d_codes.append(draw_code_raw[i] if is_vis else 0)
            
            draw_id_list.append(torch.tensor(d_ids, dtype=torch.long))
            draw_code_list.append(torch.tensor(d_codes, dtype=torch.long).clamp(0, 127))

            # --- ACTION ---
            sel_id_list.append(torch.tensor(row.get('selected_chips_id', [0]*5), dtype=torch.long))
            sel_code_list.append(torch.tensor(row.get('selected_chips_code', [0]*5), dtype=torch.long).clamp(0, 127))
            
            # Cross/Beast
            sc = int(row.get('selected_cross', 0))
            sel_cross_list.append(torch.tensor(max(0, min(sc, 63)), dtype=torch.long))
            sel_beast_list.append(torch.tensor(1 if row.get('beast_selected') else 0, dtype=torch.long))

        # Stack
        return {
            'p_hp': torch.tensor(p_hp_list, dtype=torch.float32),
            'e_hp': torch.tensor(e_hp_list, dtype=torch.float32),
            'turn_idx': torch.tensor(turn_list, dtype=torch.float32),
            'grid_tile': torch.stack(grid_t_list),
            'grid_owner': torch.stack(grid_o_list),
            'folder_p': torch.stack(fp_list),
            'folder_e': torch.stack(fe_list),
            'cross_hist_p': torch.stack(cp_list),
            'cross_hist_e': torch.stack(ce_list),
            'held_id': torch.stack(held_id_list),
            'held_code': torch.stack(held_code_list),
            'draw_id': torch.stack(draw_id_list),
            'draw_code': torch.stack(draw_code_list),
            'sel_id': torch.stack(sel_id_list),
            'sel_code': torch.stack(sel_code_list),
            'sel_cross': torch.stack(sel_cross_list),
            'sel_beast': torch.stack(sel_beast_list),
        }