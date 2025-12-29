import torch
import numpy as np
import os
import sys
import copy
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

# Ensure viewer/ is in python path to access critic_infer
sys.path.append(os.getcwd())

try:
    from viewer.critic_infer import CriticRunner, _tensorize_frame_v5
except ImportError:
    # Fallback if running from a different root
    sys.path.append(os.path.join(os.getcwd(), 'viewer'))
    from critic_infer import CriticRunner, _tensorize_frame_v5

# =============================================================================
# MMBN6 CONSTANTS
# =============================================================================
WILDCARD_CODE = 26  # * Code (Wildcard)
MAX_SLOTS = 5       # Max total slots allowed (Chips + Beast)
NAVI_CUST_MAX = 50  # MB limit (simplified, not strictly enforced here)

# =============================================================================
# CRITIC PLANNER STRATEGY
# =============================================================================

class PlanningAgentStrategy:
    def __init__(self, 
                 model_path: str, 
                 device: torch.device, 
                 key_bit_positions: Dict[str, int],
                 folder_len: int = 30,
                 **kwargs):
        
        self.device = device
        self.key_bits = key_bit_positions
        self.folder_len = folder_len
        
        print(f"[Planner] Initializing Critic-Based Planner using: {model_path}")
        
        # Load the Critic Model (The "Brain")
        # We use batch_seqs=2048 to ensure we can eval ALL combos in one massive batch
        try:
            self.critic = CriticRunner(
                ckpt_path=model_path, 
                device=str(device), 
                use_amp=True, 
                batch_seqs=2048
            )
            print("[Planner] Critic loaded successfully.")
        except Exception as e:
            print(f"[Planner] ❌ FATAL: Failed to load critic: {e}")
            self.critic = None

        # State management
        self.action_queue = deque()      # Buffer for button mashing (holding A)
        self.current_targets = deque()   # High level steps (Navigate -> Select -> OK)
        self.frames_in_window = 0
        self.plan_generated = False
        
        # Config
        self.WINDOW_WARMUP_FRAMES = 30   # Wait 0.5s for menu to settle
        self.BUTTON_HOLD_FRAMES = 4
        self.BUTTON_WAIT_FRAMES = 12

    def reset_state(self, port: int):
        """Called when entering Battle Mode or Resetting."""
        self.action_queue.clear()
        self.current_targets.clear()
        self.frames_in_window = 0
        self.plan_generated = False

    def decide_action(self, port: int, game_state: dict) -> dict:
        """
        Main Loop:
        1. If Window Closed -> Reset & NoOp.
        2. If executing a plan -> return next button press.
        3. If no plan -> Brute force best hand -> Generate plan.
        """
        inside_window = bool(float(game_state.get("inside_window", 0)))
        
        # 1. Reset logic if we leave the window
        if not inside_window:
            if self.frames_in_window > 0:
                print(f"[Planner] Window closed. Resetting.")
            self.reset_state(port)
            return self._no_op()

        # 2. Increment timer
        self.frames_in_window += 1

        # 3. Warmup (Wait for UI to fade in)
        if self.frames_in_window < self.WINDOW_WARMUP_FRAMES:
            return self._no_op()

        # 4. Execute Low-Level Action Queue (Button Holds/Waits)
        if self.action_queue:
            mask = self.action_queue.popleft()
            return self._create_response(mask, debug_msg="executing_queue")

        # 5. Navigate to High-Level Targets
        if self.current_targets:
            return self._navigate_to_next_target(game_state)

        # 6. Stop if Done (Waiting for battle to start)
        if self.plan_generated:
            return self._no_op()

        # 7. Generate Plan (The "Brute Force" Step)
        self.plan_generated = True
        return self._run_brute_force_planning(game_state)

    # -------------------------------------------------------------------------
    # CORE PLANNING LOGIC
    # -------------------------------------------------------------------------

    def _run_brute_force_planning(self, game_state: dict):
        """
        Generates all valid hands, feeds them to Critic, picks best.
        """
        if not self.critic:
            return self._no_op()

        try:
            # A. Parse Context
            static = {} # Static data usually mostly relevant for folder ID mapping, 
                        # but Critic needs it. If missing, we rely on tensorizer defaults.
            
            # Reconstruct 'derived' dict from game_state for the tensorizer
            derived = self._reconstruct_derived(game_state)
            
            hand_slots = [int(x) for x in game_state.get("chip_slots", [])]
            hand_codes = [int(x) for x in game_state.get("chip_codes", [])]
            
            # B. Generate Candidates
            # Returns list of dicts: {'chain': [idxs], 'cross': int, 'beast': bool}
            candidates = self._generate_all_valid_moves(hand_slots, hand_codes, derived)
            
            if not candidates:
                print("[Planner] No valid moves found. Pressing OK.")
                self.current_targets.append({"type": "ok"})
                return self._no_op()

            # C. Batch Evaluation
            # We construct a massive batch of "Hallucinated Futures"
            batch_tensors = []
            
            for cand in candidates:
                # Create the future state (Window Closed, Chips in Queue)
                t_dict = self._create_hallucinated_state(
                    cand, game_state, static, derived, hand_slots, hand_codes
                )
                batch_tensors.append(t_dict)

            # Stack for inference
            xb = {}
            first_keys = batch_tensors[0].keys()
            for k in first_keys:
                # Stack [N, ...] -> [B, N, ...] -> [B, 1, ...]
                t_list = [d[k] for d in batch_tensors]
                stacked = torch.stack(t_list, dim=0).unsqueeze(1)
                xb[k] = stacked.to(self.critic.device, non_blocking=True)

            # D. Inference
            with torch.no_grad():
                # [B, 1] output
                values = self.critic.model(xb).squeeze(-1).squeeze(-1).cpu().numpy()

            # E. Selection
            best_idx = np.argmax(values)
            best_move = candidates[best_idx]
            best_val = values[best_idx]

            print(f"[Planner] Evaluated {len(candidates)} hands. Best V(s): {best_val:.3f}")
            print(f"   -> Chips: {best_move['chain']} | Cross: {best_move['cross_idx']} | Beast: {best_move['do_beast']}")

            # F. Build Execution Path
            self._build_execution_path(best_move)
            
            # Start executing immediately
            return self._navigate_to_next_target(game_state)

        except Exception as e:
            print(f"[Planner] ❌ Error in planning: {e}")
            import traceback; traceback.print_exc()
            # Fallback: Just press OK
            self.current_targets.append({"type": "ok"})
            return self._no_op()

    def _generate_all_valid_moves(self, hand_slots, hand_codes, derived) -> List[Dict]:
        """
        Combinatorics:
        1. Valid Chip Chains (DFS) - Enforcing Same ID *OR* Same Code logic.
        2. Valid Form Changes (Current Cross -> New Cross / Beast).
        """
        # --- 1. Generate Chains ---
        valid_indices = [i for i, x in enumerate(hand_slots) if x != 255 and x != 65535]
        chains = self._generate_chip_chains(hand_slots, hand_codes, valid_indices)
        
        # --- 2. Form Options ---
        curr_cross = derived.get("active_cross", {}).get("idx", 0)
        is_beast = derived.get("beast", {}).get("active", False)
        
        form_opts = []
        # Option A: Stay
        form_opts.append((curr_cross, is_beast))
        
        # Option B: Beast Out (if not active)
        if not is_beast:
            form_opts.append((curr_cross, True))
            
        # Option C: Change Cross (Not implemented in this version to keep logic simple)
        # Would require knowing unlocked crosses.
        
        # --- 3. Cartesian Product & Filter ---
        candidates = []
        for ch in chains:
            for (cr, b) in form_opts:
                
                # RULE: Beast Out consumes 1 chip slot.
                # If Beast=True, max chips = MAX_SLOTS - 1
                max_allowed = MAX_SLOTS - 1 if b else MAX_SLOTS
                
                if len(ch) <= max_allowed:
                    candidates.append({
                        "chain": ch,
                        "cross_idx": cr,
                        "do_beast": b
                    })
                
        return candidates

    def _generate_chip_chains(self, ids: List[int], codes: List[int], valid_indices: List[int]) -> List[List[int]]:
        """
        Generates valid MMBN chip selection chains.
        
        RULES:
        1. Ascending hand indices (Sequential).
        2. The ENTIRE chain must satisfy ONE of these Global Constraints:
           a) All chips have the same ID (Code doesn't matter).
           b) All chips have compatible Codes (One specific code + Wildcards).
        """
        chains = [[]] # Option: Empty hand
        
        # Recursive State: (is_same_id_possible, required_id, is_same_code_possible, required_code)
        
        def extend_chain(current_chain, state):
            # Max depth check is handled by caller/filters, but good to have safety
            if len(current_chain) >= MAX_SLOTS: return

            last_idx = current_chain[-1]
            # Search strictly forward
            start_search = last_idx + 1
            
            same_id_ok, req_id, same_code_ok, req_code = state

            for i in valid_indices:
                if i < start_search: continue
                
                c = codes[i]
                idn = ids[i]
                
                # Check 1: Can we continue "Same ID" mode?
                next_same_id_ok = same_id_ok and (idn == req_id)
                
                # Check 2: Can we continue "Same Code" mode?
                next_same_code_ok = same_code_ok
                next_req_code = req_code
                
                if same_code_ok:
                    if c != WILDCARD_CODE:
                        if req_code is None:
                            # We just locked into a code
                            next_req_code = c
                        elif c != req_code:
                            # Code mismatch
                            next_same_code_ok = False
                
                # If EITHER mode is still alive, this chip is valid
                if next_same_id_ok or next_same_code_ok:
                    new_chain = current_chain + [i]
                    chains.append(new_chain)
                    
                    new_state = (next_same_id_ok, req_id, next_same_code_ok, next_req_code)
                    extend_chain(new_chain, new_state)

        # Start chains from every single chip
        for i in valid_indices:
            chain = [i]
            chains.append(chain)
            
            # Initial Constraint State for a chain of length 1
            c = codes[i]
            req_c = c if c != WILDCARD_CODE else None
            
            # (ID Mode Alive, ID[i], Code Mode Alive, Code[i] or None)
            init_state = (True, ids[i], True, req_c)
            extend_chain(chain, init_state)
            
        return chains

    def _create_hallucinated_state(self, cand, frame, static, derived, hand_ids, hand_codes):
        """
        Creates the tensor dict for the Critic, imagining the window is closed.
        """
        chain_idxs = cand["chain"]
        
        # 1. Hallucinate Held Chips
        sim_ids = [hand_ids[i] for i in chain_idxs]
        sim_codes = [hand_codes[i] for i in chain_idxs]
        
        # On Deck is the first chip
        sim_on_deck = sim_ids[0] if sim_ids else 0
        
        # 2. Hallucinate Form
        sim_derived = copy.deepcopy(derived)
        sim_derived["active_cross"]["idx"] = cand["cross_idx"]
        sim_derived["beast"]["active"] = cand["do_beast"]
        sim_derived["held_chips"] = [{"id": i, "code": c} for i, c in zip(sim_ids, sim_codes)]
        
        # 3. Hallucinate Frame
        sim_frame = frame.copy()
        sim_frame["inside_window"] = False
        sim_frame["player_chip"] = sim_on_deck # The critical causal link
        
        # 4. Tensorize
        return _tensorize_frame_v5(sim_frame, sim_derived, static, folder_len=self.folder_len)

    def _reconstruct_derived(self, game_state):
        """
        Helper to rebuild 'derived' dict structure from flat game_state.
        """
        return {
            "active_cross": {"idx": int(game_state.get("player_cross_id", 0))},
            "beast": {"active": bool(game_state.get("beast_mode", 0))},
            "held_chips": [], # Empty in window
            "folder_used_mask": game_state.get("folder_used_mask", []),
            "used_cross_mask": game_state.get("player_used_crosses_list", []),
            "used_chip_id": 0 # Reset
        }

    # -------------------------------------------------------------------------
    # NAVIGATION & EXECUTION
    # -------------------------------------------------------------------------

    def _build_execution_path(self, best_move):
        """
        Converts the Best Move into a queue of targets.
        """
        # 1. Beast Out?
        if best_move['do_beast']:
             # Assume 'R' button or similar triggers Beast in the menu.
             self.current_targets.append({"type": "beast"})

        # 2. Chips
        for idx in best_move['chain']:
            self.current_targets.append({"type": "chip", "idx": idx})
            
        # 3. OK
        self.current_targets.append({"type": "ok"})

    def _navigate_to_next_target(self, game_state):
        """
        Moves the cursor to the target index and presses A.
        """
        if not self.current_targets:
            return self._no_op()
            
        target = self.current_targets[0]
        curr_idx = int(game_state.get("selected_menu_index", 0))
        visible_count = int(game_state.get("chip_visible_count", 5))
        
        # --- EXECUTE: BEAST ---
        if target['type'] == 'beast':
            print("[Planner] Executing Beast Out")
            self.current_targets.popleft()
            return self._press("R") # Standard Beast Button

        # --- EXECUTE: OK ---
        if target['type'] == 'ok':
            # Shortcut: Press START to OK immediately?
            # Standard MMBN behavior usually allows START to confirm.
            if curr_idx == 10:
                print("[Planner] Pressing OK (A)")
                self.current_targets.popleft()
                return self._press("A")
            
            print("[Planner] Pressing OK (Start Shortcut)")
            self.current_targets.popleft()
            return self._press("START") 

        # --- EXECUTE: CHIP SELECT ---
        if target['type'] == 'chip':
            tgt_idx = target['idx']
            
            if curr_idx == tgt_idx:
                print(f"[Planner] Selecting Chip {tgt_idx}")
                self.current_targets.popleft()
                return self._press("A")
            
            # Grid Navigation (2 rows of 5)
            # 0 1 2 3 4
            # 5 6 7 8 9
            # 10 (OK)
            
            # If we are at OK (10), go back up
            if curr_idx == 10:
                return self._press("UP")
                
            curr_x, curr_y = curr_idx % 5, curr_idx // 5
            tgt_x, tgt_y = tgt_idx % 5, tgt_idx // 5
            
            if curr_y < tgt_y: return self._press("DOWN")
            if curr_y > tgt_y: return self._press("UP")
            if curr_x < tgt_x: return self._press("RIGHT")
            if curr_x > tgt_x: return self._press("LEFT")

        return self._no_op()

    def _press(self, key_name: str):
        mask = self._get_key_mask(key_name)
        
        # Hold for a few frames
        for _ in range(self.BUTTON_HOLD_FRAMES):
            self.action_queue.append(mask)
        # Release and wait
        for _ in range(self.BUTTON_WAIT_FRAMES):
            self.action_queue.append(0)
            
        return self._create_response(mask, f"press_{key_name}")

    def _get_key_mask(self, btn_name: str) -> int:
        # Standard alias mapping
        ALIAS = {'A': 'Z', 'B': 'X', 'L': 'A', 'R': 'S', 'START': 'RETURN'}
        name = ALIAS.get(btn_name, btn_name)
        if name in self.key_bits:
            return (1 << int(self.key_bits[name]))
        return 0

    def _create_response(self, mask: int, debug_msg: str):
        bin_str = format(int(mask) & 0xFFFF, "016b")
        return {
            "button_command": {"type": "key_press", "key": bin_str}, 
            "debug": {"plan_action": debug_msg}
        }

    def _no_op(self):
        return self._create_response(0, "no_op")