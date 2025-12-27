#planning/planning_model.py
import torch
import torch.nn as nn
import json
import os
from collections import deque
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path so we can import 'strategy'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from planning.strategy_model import StrategyTransformer, META_DIM, ELEMENTS

# =============================================================================
# 1. METADATA PROCESSOR
# =============================================================================

class ChipMetaProcessor:
    def __init__(self, json_path):
        self.db = {} 
        self.load_db(json_path)
        
    def load_db(self, path):
        if not os.path.exists(path):
            print(f"⚠️ [Plan] Warning: Chips DB not found at {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for chip in data:
            vec = self._build_vector(chip)
            def set_id(raw_val):
                if raw_val is None: return
                s_val = str(raw_val).strip()
                if not s_val: return 
                try: self.db[int(s_val)] = vec
                except: pass 
            set_id(chip.get('SId'))
            set_id(chip.get('MId'))
                
    def _build_vector(self, chip):
        dmg_str = str(chip.get('Damage', '0'))
        if '-' in dmg_str: dmg_str = dmg_str.split('-')[1]
        try:
            if not dmg_str.strip() or not dmg_str[0].isdigit(): dmg_val = 0.0
            else: dmg_val = float(dmg_str) / 500.0
        except: dmg_val = 0.0
        elem_vec = [0.0] * len(ELEMENTS)
        c_elem = chip.get('Element', "")
        if c_elem in ELEMENTS: elem_vec[ELEMENTS.index(c_elem)] = 1.0
        else: elem_vec[0] = 1.0 
        type_vec = [0.0] * 3
        try:
            mb_str = str(chip.get('MB', '0')).strip()
            mb_val = int(float(mb_str)) if mb_str else 0
        except: mb_val = 0
        if chip.get('MId') is not None and str(chip.get('MId')).strip():
            if "Giga" in chip.get('Version', '') or mb_val > 80: type_vec[2] = 1.0
            else: type_vec[1] = 1.0
        else: type_vec[0] = 1.0
        return torch.tensor([dmg_val] + elem_vec + type_vec, dtype=torch.float32)

    def get_meta(self, chip_id):
        if chip_id in self.db: return self.db[chip_id]
        return torch.zeros(META_DIM, dtype=torch.float32)

# =============================================================================
# 2. PLANNING STRATEGY (Execution Logic)
# =============================================================================

class PlanningAgentStrategy:
    def __init__(self, 
                 model_path: str, 
                 chips_db_path: str,
                 device: torch.device, 
                 key_bit_positions: Dict[str, int]):
        
        self.device = device
        self.key_bits = key_bit_positions
        self.EOS_TOKEN = 13
        
        # Load Model
        self.model = StrategyTransformer(
            num_chip_ids=512, 
            num_codes=32, 
            d_model=512, 
            nhead=8, 
            num_layers=6,
            dropout=0.0
        )
        
        try:
            # Checkpoint Sanitization
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.to(device)
            self.model.eval()
            print(f"[Plan] Model loaded from {model_path} on {device}")
        except Exception as e:
            print(f"[Plan] ❌ Failed to load planning model: {e}")
            
        self.meta_proc = ChipMetaProcessor(chips_db_path)
        self.action_queue = deque() 
        self.current_targets = deque() 
        
        # WARMUP TIMER (180 frames = ~3.0s)
        self.frames_in_window = 0
        self.WINDOW_WARMUP_FRAMES = 180 
        self.plan_generated = False
        
    def reset_state(self, port: int):
        self.action_queue.clear()
        self.current_targets.clear()
        self.frames_in_window = 0
        self.plan_generated = False

    def _int_to_bin16(self, mask: int) -> str:
        return format(int(mask) & 0xFFFF, "016b")

    def _get_key_mask(self, btn_name: str) -> int:
        ALIASES = {
            'A': 'Z', 'B': 'X', 'L': 'A', 'R': 'S',
            'START': 'RETURN', 'SELECT': 'BACKSPACE',
            'UP': 'Up', 'DOWN': 'Down', 'LEFT': 'Left', 'RIGHT': 'Right'
        }
        physical_name = ALIASES.get(btn_name, btn_name)
        if physical_name in self.key_bits: return (1 << int(self.key_bits[physical_name]))
        for k, v in self.key_bits.items():
            if k.upper() == physical_name.upper(): return (1 << int(v))
        for k, v in self.key_bits.items():
            if k.upper() == btn_name.upper(): return (1 << int(v))
        return 0

    def decide_action(self, port: int, game_state: dict) -> dict:
        inside_window = bool(float(game_state.get("inside_window", 0)))
        
        # 1. Reset logic if we leave the window
        if not inside_window:
            self.frames_in_window = 0
            self.action_queue.clear()
            self.current_targets.clear()
            self.plan_generated = False
            return self._no_op()

        # 2. Increment timer
        self.frames_in_window += 1

        # 3. WARMUP CHECK
        if self.frames_in_window < self.WINDOW_WARMUP_FRAMES:
            return self._no_op()

        # 4. Execute Queued Actions
        if self.action_queue:
            mask = self.action_queue.popleft()
            return self._create_response(mask, debug_msg="executing_queue")

        # 5. Navigate to Targets
        if self.current_targets:
            return self._navigate_to_next_target(game_state)

        # 6. Stop if Done
        if self.plan_generated:
            return self._no_op()

        # 7. Generate Plan
        self.plan_generated = True 
        return self._run_inference_and_plan(game_state)

    def _run_inference_and_plan(self, game_state: dict):
        try:
            # --- A. PREPARE HAND ---
            chip_slots = game_state.get("chip_slots", [])[:10]
            chip_codes = game_state.get("chip_codes", [])[:10]
            visible_count = int(game_state.get("chip_visible_count", 5))
            
            while len(chip_slots) < 10: chip_slots.append(255)
            while len(chip_codes) < 10: chip_codes.append(0)

            for i in range(visible_count, 10):
                chip_slots[i] = 255
                chip_codes[i] = 0

            chip_slots = [max(0, min(int(x), 511)) for x in chip_slots]
            chip_codes = [max(0, min(int(x), 31)) for x in chip_codes]

            h_ids = torch.tensor([chip_slots], dtype=torch.long).to(self.device)
            h_codes = torch.tensor([chip_codes], dtype=torch.long).to(self.device)
            meta_list = [self.meta_proc.get_meta(cid) for cid in chip_slots]
            h_meta = torch.stack(meta_list).unsqueeze(0).to(self.device)
            
            # --- B. PREPARE CONTEXT (36 Dims) ---
            p_hp = float(game_state.get("player_health", 500)) / 1000.0
            e_hp = float(game_state.get("enemy_health", 500)) / 2000.0
            
            used_crosses = game_state.get("player_used_crosses_list", [])
            used_cross_vec = [0.0] * 6
            for c in used_crosses:
                if 1 <= c <= 6: used_cross_vec[c-1] = 1.0
            
            curr_cross_id = int(float(game_state.get("player_cross_id", 0)))
            curr_cross_vec = [0.0] * 6
            if 0 <= curr_cross_id <= 5: curr_cross_vec[curr_cross_id] = 1.0
            else: curr_cross_vec[0] = 1.0
                
            beast_val = 1.0 if int(float(game_state.get("beast_mode", 0))) > 0 else 0.0
            
            grid_raw = game_state.get("grid_state", [0]*18)
            grid_vec = [x / 10.0 for x in grid_raw]
            if len(grid_vec) != 18: grid_vec = [0.0]*18
            
            p_emo = int(float(game_state.get("player_emotion", 0)))
            is_full_sync = 1.0 if p_emo == 1 else 0.0
            
            owner_state = game_state.get("grid_owner_state", [0]*18)
            if len(owner_state) != 18: owner_state = [0]*9 + [1]*9
            area_adv = ((18 - sum(owner_state)) - 9.0) / 9.0
            
            e_pos = game_state.get("enemy_pos", [0, 0])
            e_col = max(0, min(5, int((e_pos[0] - 40) / 40)))
            e_col_norm = e_col / 5.0
            
            ctx_list = [p_hp, e_hp] + used_cross_vec + curr_cross_vec + [beast_val] + grid_vec + [is_full_sync, area_adv, e_col_norm]
            ctx_vec = torch.tensor([ctx_list], dtype=torch.float32).to(self.device)
            
            # --- C. INFERENCE ---
            B = h_ids.shape[0]
            pos = torch.arange(10, device=self.device).unsqueeze(0).expand(B, 10)
            
            with torch.no_grad():
                src = (self.model.chip_embedding(h_ids) + 
                       self.model.code_embedding(h_codes) + 
                       self.model.meta_proj(h_meta) + 
                       self.model.pos_embedding(pos))
                src = src + self.model.context_proj(ctx_vec).unsqueeze(1)
                memory = self.model.encoder(src)
                
                cross_logits, seq_tokens = self.model.inference(
                    memory, 
                    self.model.cross_head(memory.mean(dim=1)),
                    max_chip_index=visible_count
                )

            # --- D. DECODE PLAN ---
            # Cross output MUST be interpreted as a *menu index* in the cross window.
            # The menu dynamically shrinks as crosses are used (removed from the list).
            # If the model outputs an index that doesn't exist anymore, we ignore it.
            selected_cross_raw = int(torch.argmax(cross_logits, dim=1).item())

            used_list = game_state.get("player_used_crosses_list", []) or []

            # Cross menu contains 5 crosses (not "Normal"). IDs are 1..5 in our planner convention.
            available_cross_ids = [cid for cid in (1, 2, 3, 4, 5) if cid not in used_list]
            avail_n = len(available_cross_ids)

            def _as_valid_menu_index(raw: int) -> Optional[int]:
                """
                Support both training conventions:
                - raw in [0..4]  : already a menu index
                - raw in [1..5]  : menu index is raw-1 (common off-by-one if 0 reserved)
                Anything else => invalid / 'no cross'.
                """
                if 0 <= raw < avail_n:
                    return raw
                if 1 <= raw <= 5 and 0 <= (raw - 1) < avail_n:
                    return raw - 1
                return None

            cross_menu_idx = _as_valid_menu_index(selected_cross_raw)
            if cross_menu_idx is not None:
                self.current_targets.append({"type": "cross", "val": cross_menu_idx})

                
            tokens = seq_tokens[0].tolist()
            for t in tokens:
                if t == self.EOS_TOKEN:
                    self.current_targets.append({"type": "ok", "val": 0})
                    break 
                elif t == 11:
                    self.current_targets.append({"type": "beast", "val": 0})
                elif t < 10:
                    if t < visible_count:
                        self.current_targets.append({"type": "chip", "val": t})
            
            print(f"[Plan] Generated: {self.current_targets}")
            return self._navigate_to_next_target(game_state)
            
        except Exception as e:
            print(f"[Plan] Error: {e}")
            import traceback; traceback.print_exc()
            self.action_queue.append(self._get_key_mask("START"))
            return self._create_response(0, debug_msg="inference_failed")

    def _navigate_to_next_target(self, game_state):
        if not self.current_targets: return self._no_op()
        target = self.current_targets[0]
        
        curr_idx = game_state.get("selected_menu_index")
        curr_cross_idx = int(game_state.get("selected_cross_index", 0))
        try: inside_cross_window = bool(float(game_state.get("inside_cross_window", 0)))
        except: inside_cross_window = False

        if curr_idx is None: return self._no_op()
        curr_idx = int(curr_idx)
        visible_count = int(game_state.get("chip_visible_count", 5))

        # 1. OK
        if target['type'] == 'ok':
            if curr_idx == 10:
                self.current_targets.popleft()
                return self._press("A")
            if inside_cross_window: return self._press("DOWN")
            return self._press("START")

        # 2. BEAST
        if target['type'] == 'beast':
            if inside_cross_window: return self._press("DOWN")
            if curr_idx == 11:
                print("[Plan] Beast Out Selected")
                self.current_targets.popleft()
                return self._press("A")
            if curr_idx == 10: return self._press("RIGHT")
            return self._press("START")

        # 3. CROSS (Dynamic) — MENU-INDEX BASED
        if target['type'] == 'cross':
            desired_menu_idx = int(target['val'])

            used_list = game_state.get("player_used_crosses_list", []) or []

            # Cross menu contains 5 crosses (IDs 1..5). Used ones are removed.
            available_cross_ids = [cid for cid in (1, 2, 3, 4, 5) if cid not in used_list]
            avail_n = len(available_cross_ids)

            # If the model asks for an index that no longer exists (e.g. 4 when only 0..3),
            # we must IGNORE it (skip this target) — otherwise we can never reach it.
            if desired_menu_idx < 0 or desired_menu_idx >= avail_n:
                print(f"[Plan] ⚠️ Cross menu idx {desired_menu_idx} out of range (0..{max(avail_n-1, 0)}). Ignoring.")
                self.current_targets.popleft()
                return self._no_op()

            # Optional: for logging only (which actual cross ID sits at that menu index right now)
            tgt_cross_id = available_cross_ids[desired_menu_idx]

            if not inside_cross_window:
                if curr_idx == 11: return self._press("UP")
                if curr_idx == 10: return self._press("LEFT")
                return self._press("UP")

            if curr_cross_idx == desired_menu_idx:
                print(f"[Plan] Cross menu idx {desired_menu_idx} (id={tgt_cross_id}) Selected")
                self.current_targets.popleft()
                return self._press("A")

            diff = desired_menu_idx - curr_cross_idx
            if diff > 0: return self._press("DOWN")
            if diff < 0: return self._press("UP")

        # 4. CHIP
        if target['type'] == 'chip':
            tgt_idx = target['val']
            if inside_cross_window: return self._press("DOWN")
            
            if curr_idx == 11: return self._press("UP")
            if curr_idx == 10: return self._press("LEFT")

            if curr_idx == tgt_idx:
                print(f"[Plan] Chip {tgt_idx} Selected")
                self.current_targets.popleft()
                return self._press("A")

            curr_x, curr_y = curr_idx % 5, curr_idx // 5
            tgt_x, tgt_y = tgt_idx % 5, tgt_idx // 5
            
            if curr_y != tgt_y:
                direction = "DOWN" if tgt_y > curr_y else "UP"
                dest_idx = curr_idx + (5 if direction == "DOWN" else -5)
                if dest_idx < visible_count: return self._press(direction)

            if curr_x != tgt_x:
                direction = "RIGHT" if tgt_x > curr_x else "LEFT"
                dest_idx = curr_idx + (1 if direction == "RIGHT" else -1)
                if dest_idx < visible_count: return self._press(direction)
            
            return self._press("LEFT")

        return self._no_op()

    def _press(self, key_name: str):
        mask = self._get_key_mask(key_name)
        
        # 🚀 UPDATE: "Decisive Press" Logic
        # 1. HOLD the button for 5 frames (ensure registration)
        for _ in range(5): 
            self.action_queue.append(mask)
            
        # 2. WAIT for 25 frames (allow UI animation/state update)
        for _ in range(25): 
            self.action_queue.append(0)
            
        # Total: 30 frames per action
        return self._create_response(mask, f"press_{key_name}")

    def _create_response(self, mask: int, debug_msg: str):
        return {"button_command": {"type": "key_press", "key": self._int_to_bin16(mask)}, "ng_key_bin": "", "debug": {"plan_action": debug_msg}}

    def _no_op(self):
        return self._create_response(0, "no_op")