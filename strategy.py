import time
from collections import defaultdict, deque
import torch
import torch.nn.functional as F # For softmax
from PIL import Image
from io import BytesIO
import base64

class ActionStrategy:
    """Base class for all action decision strategies."""
    def __init__(self, key_bit_positions_map, discrete_actions_map, util_functions):
        self.key_bit_positions = key_bit_positions_map
        self.discrete_actions = discrete_actions_map
        self.utils_int_to_binary_string = util_functions['int_to_binary_string']
        self.utils_map_discrete_action_to_buttons = util_functions['map_discrete_action_to_buttons']
        self.utils_preprocess_frame = util_functions.get('preprocess_frame')
        self.utils_generate_random_action = util_functions.get('generate_random_action_for_skip_strategy')

    def decide_action(self, port, game_state_dict):
        raise NotImplementedError("Subclasses must implement the decide_action method.")

    def reset_state(self, port):
        pass # Optional for subclasses to implement


class SkipAndRandomStrategy(ActionStrategy):
    """Implements the MVP skip logic and random actions for non-DRL scenarios."""
    def __init__(self, key_bit_positions_map, discrete_actions_map, util_functions, random_battle_action_keys):
        super().__init__(key_bit_positions_map, discrete_actions_map, util_functions)
        self.random_battle_action_keys = random_battle_action_keys
        # Note: This strategy still uses the tuple-based state for its skip logic.
        # If it needs the same alternating Z behavior, its skip logic and state would also need updating.
        self._skip_logic_state = defaultdict(lambda: (0, 0.0)) # (state_code, timestamp)

    def decide_action(self, port, game_state_dict):
        inside_window_flag = game_state_dict.get('inside_window', False)
        action_to_send_str = '0000000000000000' # Default NO_OP

        if inside_window_flag:
            current_state_code, timestamp = self._skip_logic_state[port]
            if current_state_code == 0: # Just entered window, start initial wait
                self._skip_logic_state[port] = (10, time.time())
                # print(f"Port {port} (SR): Entered window. State 0->10. Initial 0.5s wait.")
            elif current_state_code == 10: # Initial 0.5s wait
                if time.time() - timestamp >= 0.5:
                    self._skip_logic_state[port] = (11, time.time())
                    # print(f"Port {port} (SR): State 10->11. Initial wait over. Sending RETURN.")
                    action_to_send_str = self.utils_int_to_binary_string(1 << self.key_bit_positions['RETURN'])
            elif current_state_code == 11: # First RETURN sent, waiting 1s for Z.
                # Original SkipAndRandom logic: spam RETURN, then final Z.
                # If this strategy also needs the "alternate Z and NO_OP" logic,
                # this section needs to be updated similar to DRLAgentStrategy.
                if time.time() - timestamp >= 1.0: # Time for final Z
                    self._skip_logic_state[port] = (12, 0.0)
                    # print(f"Port {port} (SR): State 11->12. Sending Z.")
                    action_to_send_str = self.utils_int_to_binary_string(1 << self.key_bit_positions['Z'])
                else: # Still waiting, keep pressing RETURN
                    # print(f"Port {port} (SR): State 11. Spamming RETURN before final Z.")
                    action_to_send_str = self.utils_int_to_binary_string(1 << self.key_bit_positions['RETURN'])
            # elif current_state_code == 12: # Z sent, sequence complete, send NO_OP
            #     pass # Default NO_OP is fine
            
            return {
                'button_command': {'type': 'key_press', 'key': action_to_send_str},
                'action_idx': None, 'log_prob': None, 'value': None,
                'stacked_frames_tensor': None, 'game_features_tensor': None
            }
        else: # Not in window
            if self._skip_logic_state[port][0] != 0: # If was in a skipping state
                self.reset_state(port)
            
            action_key_str = self.utils_generate_random_action(self.key_bit_positions, self.random_battle_action_keys)
            return {
                'button_command': {'type': 'key_press', 'key': action_key_str},
                'action_idx': None, 'log_prob': None, 'value': None,
                'stacked_frames_tensor': None, 'game_features_tensor': None
            }

    def reset_state(self, port):
        # print(f"Port {port} (SR): Resetting skip logic state.")
        self._skip_logic_state[port] = (0, 0.0)


class DRLAgentStrategy(ActionStrategy):
    def __init__(self, key_bit_positions_map, discrete_actions_map, util_functions,
                 actor_critic_model, device, frame_height, frame_width, num_frames_stacked,
                 max_health_config, max_charge_config, max_cust_gauge_config):
        super().__init__(key_bit_positions_map, discrete_actions_map, util_functions)
        self.model = actor_critic_model
        self.device = device
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames_stacked = num_frames_stacked

        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.num_frames_stacked))
        
        self.max_health = max_health_config
        self.max_charge = max_charge_config
        self.max_cust_gauge = max_cust_gauge_config
        
        # ** CORRECTED INITIALIZATION FOR _skip_logic_state_internal **
        self._skip_logic_state_internal = defaultdict(lambda: {'code': 0, 'timestamp': 0.0, 'next_alternating_action': 'Z'})


    def _preprocess_state(self, port, game_state_dict):
        # 1. Image Data
        b64_image = game_state_dict.get('image')
        pil_image = None
        if b64_image:
            try:
                pil_image = Image.open(BytesIO(base64.b64decode(b64_image)))
            except Exception as e:
                # print(f"Port {port} (DRL): Error decoding image: {e}")
                pass 
        
        frame_tensor = self.utils_preprocess_frame(pil_image, self.frame_height, self.frame_width)

        buffer = self.frame_buffers[port]
        if pil_image is None and len(buffer) > 0: 
            buffer.append(buffer[-1].clone()) 
        else:
            buffer.append(frame_tensor)
        
        stacked_frames_list = list(buffer)
        while len(stacked_frames_list) < self.num_frames_stacked:
            stacked_frames_list.insert(0, torch.zeros_like(frame_tensor))

        stacked_frames_tensor = torch.cat(stacked_frames_list, dim=0).unsqueeze(0).to(self.device)

        # 2. Numerical Game Features
        features = []
        features.append(float(game_state_dict.get('player_health', 0)) / self.max_health)
        features.append(float(game_state_dict.get('enemy_health', 0)) / self.max_health)
        player_grid_pos = game_state_dict.get('player_grid_position', [1, 1]) 
        enemy_grid_pos = game_state_dict.get('enemy_grid_position', [4, 1])   
        features.append((player_grid_pos[0] - 1) / 5.0) 
        features.append((player_grid_pos[1] - 1) / 2.0)  
        features.append((enemy_grid_pos[0] - 1) / 5.0)
        features.append((enemy_grid_pos[1] - 1) / 2.0)
        features.append(float(game_state_dict.get('player_charge', 0)) / self.max_charge)
        features.append(float(game_state_dict.get('enemy_charge', 0)) / self.max_charge)
        features.append(float(game_state_dict.get('cust_gage', 0)) / self.max_cust_gauge)
        features.append(1.0 if game_state_dict.get('is_player_beasted_out', False) else 0.0)
        features.append(1.0 if game_state_dict.get('is_enemy_beasted_out', False) else 0.0)
        
        game_features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        return stacked_frames_tensor, game_features_tensor

    def decide_action(self, port, game_state_dict):
        inside_window_flag = game_state_dict.get('inside_window', False)
        action_to_send_str = '0000000000000000' # Default NO_OP

        if inside_window_flag:
            state_info = self._skip_logic_state_internal[port]
            current_state_code = state_info['code']
            
            if current_state_code == 0: 
                state_info['code'] = 1
                state_info['timestamp'] = time.time()
                action_to_send_str = '0000000000000000' 
            elif current_state_code == 1: 
                if time.time() - state_info['timestamp'] >= 0.5:
                    state_info['code'] = 2
                    state_info['next_alternating_action'] = 'Z' 
                    action_to_send_str = self.utils_int_to_binary_string(1 << self.key_bit_positions['RETURN'])
                else:
                    action_to_send_str = '0000000000000000' 
            elif current_state_code == 2: 
                if state_info['next_alternating_action'] == 'Z':
                    action_to_send_str = self.utils_int_to_binary_string(1 << self.key_bit_positions['Z'])
                    state_info['next_alternating_action'] = 'NO_OP'
                else: 
                    action_to_send_str = '0000000000000000' 
                    state_info['next_alternating_action'] = 'Z'
            
            num_gf = self.model.num_game_features if hasattr(self.model, 'num_game_features') else 11 
            dummy_sf = torch.zeros(1, self.num_frames_stacked, self.frame_height, self.frame_width, device=self.device)
            dummy_gf = torch.zeros(1, num_gf, device=self.device)
            
            action_idx_val = self.discrete_actions.index("NO_OP") 
            if action_to_send_str == self.utils_int_to_binary_string(1 << self.key_bit_positions['RETURN']):
                if "RETURN" in self.discrete_actions: action_idx_val = self.discrete_actions.index("RETURN")
            elif action_to_send_str == self.utils_int_to_binary_string(1 << self.key_bit_positions['Z']):
                 if "Z" in self.discrete_actions: action_idx_val = self.discrete_actions.index("Z")

            return {
                'button_command': {'type': 'key_press', 'key': action_to_send_str},
                'action_idx': torch.tensor(action_idx_val, device=self.device, dtype=torch.long),
                'log_prob': torch.tensor(0.0, device=self.device), 
                'value': None, 
                'stacked_frames_tensor': dummy_sf,
                'game_features_tensor': dummy_gf
            }
        else: 
            state_info = self._skip_logic_state_internal[port]
            if state_info['code'] != 0: 
                state_info['code'] = 0
                state_info['timestamp'] = 0.0 
                state_info['next_alternating_action'] = 'Z' 

            stacked_frames, game_features = self._preprocess_state(port, game_state_dict)

            self.model.eval() 
            with torch.no_grad():
                action_idx_tensor, log_prob_tensor, _, value_tensor = \
                    self.model.get_action_and_value(stacked_frames, game_features)
            
            action_idx_item = action_idx_tensor.item() 
            button_command_str = self.utils_map_discrete_action_to_buttons(
                action_idx_item, self.discrete_actions, self.key_bit_positions
            )
            
            return {
                'button_command': {'type': 'key_press', 'key': button_command_str},
                'action_idx': action_idx_tensor,
                'log_prob': log_prob_tensor,
                'value': value_tensor,
                'stacked_frames_tensor': stacked_frames,
                'game_features_tensor': game_features
            }
        
    def reset_state(self, port):
        # print(f"Port {port} (DRL): Resetting strategy state (frame buffer & skip logic).")
        self.frame_buffers[port].clear()
        # ** CORRECTED RESET FOR _skip_logic_state_internal **
        self._skip_logic_state_internal[port] = {'code': 0, 'timestamp': 0.0, 'next_alternating_action': 'Z'}