# network_handler.py
import asyncio
import json
import traceback
from collections import deque # Though DRLAgentStrategy handles its own frame deque now
import torch # For an empty tensor placeholder if needed

class ConnectionHandler:
    def __init__(self, instance_config, active_strategy, inference_fps,
                 experience_buffer, shared_episode_data, config_module, utils_module):
        self.instance_config_data = instance_config # Specific config for this instance
        self.strategy = active_strategy
        self.inference_interval = 1.0 / inference_fps if inference_fps > 0 else 0.033
        self.experience_buffer = experience_buffer # Shared experience buffer
        self.shared_episode_data = shared_episode_data # For tracking global episode rewards/lengths

        self.port = instance_config['port']
        self.address = instance_config['address']
        self.name = instance_config['name']
        
        self.cfg = config_module # Full config module (config.py)
        self.utils = utils_module # Utils module (utils.py)

        self.reader = None
        self.writer = None
        self.message_processor_task = None # Task for _message_processor_loop
        self._is_running = False # Flag to control main loops

        # Per-instance cache for data not in every screen_data or derived over time
        self.instance_game_data_cache = {
            'player_chips_in_window': [], 'current_hand_selected': [], # For custom window logic
            'player_used_crosses_list': [], 'enemy_used_crosses_list': [], # For game progression
            'is_player_beasted_out': False, 'is_enemy_beasted_out': False,
            'is_player_beasted_over': False, 'is_enemy_beasted_over': False,
            'cached_player_folder': [], 'cached_enemy_folder': [], # Initial folder data
            'player_active_chip': 0, 'enemy_active_chip': 0, # Chip currently in use (visual effect)
            'player_chip_timer_task': None, 'enemy_chip_timer_task': None, # Async tasks for chip timers
            'previous_player_chip_from_server': 0, # To detect when a new chip is used by player
            'previous_enemy_chip_from_server': 0,  # To detect when a new chip is used by enemy
        }

        # State for constructing (s_t, a_t, r_t, s_t+1, d_t) experience tuples for DRL
        self.prev_raw_game_state_for_reward = None # s_t (raw, for reward calculation)
        self.prev_processed_stacked_frames = None  # s_t (processed frame part for buffer)
        self.prev_processed_game_features = None   # s_t (processed feature part for buffer)
        self.prev_action_info_for_buffer = None    # Dict from strategy: {action_idx, log_prob, value, ...} for a_t

        # For reward calculation and per-episode stats for this handler
        self.ep_last_player_health = None
        self.ep_last_enemy_health = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    async def _send_command_internal(self, command_dict):
        """Internal method to send a JSON command to the game instance."""
        if self.writer and not self.writer.is_closing():
            try:
                command_json = json.dumps(command_dict)
                self.writer.write(command_json.encode() + b'\n')
                await self.writer.drain()
                return True
            except (ConnectionResetError, BrokenPipeError):
                print(f"Port {self.port} ({self.name}): Connection closed while sending.")
            except Exception as e:
                print(f"Port {self.port} ({self.name}): Failed to send command: {e}")
        else:
            print(f"Port {self.port} ({self.name}): Writer not available or closing.")
        self._is_running = False # Crucial: Stop loops if send fails, indicating connection issue
        return False

    async def _request_screen_image(self):
        """Sends a request to the game instance for a new screen image and game state."""
        return await self._send_command_internal({'type': 'request_screen', 'key': ''})

    async def _handle_active_chip_timer(self, chip_owner_type):
        """Resets the 'active_chip' in cache after a delay (visual effect duration)."""
        await asyncio.sleep(1.0) # Duration for which the chip is considered "active"
        cache_key_active_chip = f'{chip_owner_type}_active_chip'
        cache_key_timer_task = f'{chip_owner_type}_chip_timer_task'
        
        self.instance_game_data_cache[cache_key_active_chip] = 0 # Reset active chip ID
        if self.instance_game_data_cache.get(cache_key_timer_task):
            self.instance_game_data_cache[cache_key_timer_task] = None # Clear the task reference

    def _update_instance_game_data_cache(self, screen_data_dict):
        """
        Updates the persistent per-instance cache based on fresh screen_data from the server.
        This handles derived states like beast status, active chip detection, and one-time data like folders.
        """
        cache = self.instance_game_data_cache

        # Detect player chip usage
        new_player_chip_id = screen_data_dict.get("player_chip", 0)
        if new_player_chip_id != cache['previous_player_chip_from_server'] and cache['previous_player_chip_from_server'] != 0:
            cache['player_active_chip'] = cache['previous_player_chip_from_server'] # The chip that was just used
            if cache['player_chip_timer_task']:
                cache['player_chip_timer_task'].cancel() # Cancel existing timer
            cache['player_chip_timer_task'] = asyncio.create_task(self._handle_active_chip_timer('player'))
        cache['previous_player_chip_from_server'] = new_player_chip_id

        # Detect enemy chip usage
        new_enemy_chip_id = screen_data_dict.get("enemy_chip", 0)
        if new_enemy_chip_id != cache['previous_enemy_chip_from_server'] and cache['previous_enemy_chip_from_server'] != 0:
            cache['enemy_active_chip'] = cache['previous_enemy_chip_from_server']
            if cache['enemy_chip_timer_task']:
                cache['enemy_chip_timer_task'].cancel()
            cache['enemy_chip_timer_task'] = asyncio.create_task(self._handle_active_chip_timer('enemy'))
        cache['previous_enemy_chip_from_server'] = new_enemy_chip_id

        # Cumulative beast status
        player_game_emotion = screen_data_dict.get("player_game_emotion", 0)
        enemy_game_emotion = screen_data_dict.get("enemy_game_emotion", 0)
        if player_game_emotion >= 11: cache['is_player_beasted_out'] = True # 11-22 is normal beast
        if player_game_emotion >= 23: cache['is_player_beasted_over'] = True # 23+ indicates Beast Over
        if enemy_game_emotion >= 11: cache['is_enemy_beasted_out'] = True
        if enemy_game_emotion >= 23: cache['is_enemy_beasted_over'] = True
        
        # Cache initial folder data (if not already cached)
        if not cache['cached_player_folder'] and "player_chip_folder" in screen_data_dict:
            cache['cached_player_folder'] = screen_data_dict.get("player_chip_folder", [])
        if not cache['cached_enemy_folder'] and "enemy_chip_folder" in screen_data_dict:
            cache['cached_enemy_folder'] = screen_data_dict.get("enemy_chip_folder", [])
            
        # Cache chips visible in the custom window (if inside window)
        if bool(float(screen_data_dict.get("inside_window", 0))):
            chip_slots = screen_data_dict.get("chip_slots", [])
            chip_codes = screen_data_dict.get("chip_codes", [])
            chips_visible_count = screen_data_dict.get("chip_visible_count", 0)
            visible_chips_in_window = []
            for i in range(min(len(chip_slots), chips_visible_count)):
                # Storing raw slot/code; strategy can interpret if needed
                visible_chips_in_window.append({'slot': chip_slots[i], 'code': chip_codes[i], 'id_in_window': i})
            cache['player_chips_in_window'] = visible_chips_in_window
        else:
            if cache['player_chips_in_window']: # Clear if not in window anymore
                cache['player_chips_in_window'] = []
        
        # `current_hand_selected` would be populated by a strategy that makes selections,
        # or from `selected_chip_indices` if that directly maps to a usable hand.
        # For now, it's initialized as empty in the cache.

    def _calculate_reward_and_done(self, current_raw_state_dict):
        """Calculates reward and done flag based on state changes."""
        reward = 0.0
        done = False

        # Initialize healths at the start of an episode for this handler
        if self.ep_last_player_health is None:
            self.ep_last_player_health = current_raw_state_dict.get('player_health', 0)
        if self.ep_last_enemy_health is None:
            self.ep_last_enemy_health = current_raw_state_dict.get('enemy_health', 0)

        current_player_hp = current_raw_state_dict.get('player_health', self.ep_last_player_health)
        current_enemy_hp = current_raw_state_dict.get('enemy_health', self.ep_last_enemy_health)

        # Calculate damage dealt and taken
        damage_dealt_to_enemy = self.ep_last_enemy_health - current_enemy_hp
        damage_taken_by_player = self.ep_last_player_health - current_player_hp

        if damage_dealt_to_enemy > 0:
            reward += damage_dealt_to_enemy * self.cfg.REWARD_DAMAGE_DEALT_MULTIPLIER
        if damage_taken_by_player > 0:
            # REWARD_DAMAGE_TAKEN_MULTIPLIER should be negative in config.py
            reward += damage_taken_by_player * self.cfg.REWARD_DAMAGE_TAKEN_MULTIPLIER 

        # Update healths for the next step's calculation
        self.ep_last_player_health = current_player_hp
        self.ep_last_enemy_health = current_enemy_hp
        
        # Apply time penalty per step (optional, encourages efficiency)
        reward += self.cfg.REWARD_TIME_PENALTY_STEP

        # Check for terminal states (win/loss based on HP)
        if current_player_hp <= 0:
            reward += self.cfg.REWARD_LOSE_GAME
            done = True
            # print(f"Port {self.port} ({self.name}): Player Lost (HP <= 0).")
        elif current_enemy_hp <= 0:
            reward += self.cfg.REWARD_WIN_GAME
            done = True
            # print(f"Port {self.port} ({self.name}): Player Won (Enemy HP <= 0).")
        
        return reward, done

    async def _message_processor_loop(self):
        """Continuously processes messages from the game instance."""
        json_buffer = ""
        try:
            while self._is_running:
                if not self.reader or self.reader.at_eof():
                    print(f"Port {self.port} ({self.name}): Reader closed or EOF. Exiting processor loop.")
                    break
                
                try:
                    # Read data with a timeout to prevent blocking indefinitely
                    # Timeout should be longer than inference_interval to allow game to respond
                    data_chunk = await asyncio.wait_for(self.reader.read(8192), timeout=max(0.2, self.inference_interval * 5))
                except asyncio.TimeoutError:
                    # print(f"Port {self.port} ({self.name}): Timeout reading from game. Will retry on next cycle.")
                    continue # Allow outer loop to request screen again if needed

                if not data_chunk: # Empty read typically means connection closed by peer
                    print(f"Port {self.port} ({self.name}): Connection closed by peer (received no data). Exiting processor loop.")
                    break 
                
                json_buffer += data_chunk.decode(errors='ignore') # Ignore decoding errors for robustness
                
                # Process all complete JSON messages in the buffer
                while "\n" in json_buffer and self._is_running:
                    message_str, json_buffer = json_buffer.split("\n", 1)
                    message_str = message_str.strip()
                    if not message_str: continue

                    try:
                        parsed_message = json.loads(message_str)
                    except json.JSONDecodeError:
                        print(f"Port {self.port} ({self.name}): Failed to parse JSON from game: {message_str[:100]}...")
                        continue # Skip malformed message

                    event = parsed_message.get("event", "Unknown")
                    # Ensure details_str is a string for json.loads, or use details directly if already dict
                    details_payload = parsed_message.get("details", {}) 
                    if not isinstance(details_payload, (str, dict)):
                        details_payload = str(details_payload) # Fallback if unexpected type

                    if event == "screen_image":
                        current_raw_game_data_from_server = {}
                        try:
                            # If details_payload is a string, it's JSON that needs parsing.
                            # If it's already a dict (some servers might send it pre-parsed), use it directly.
                            current_raw_game_data_from_server = json.loads(details_payload) if isinstance(details_payload, str) else details_payload
                        except json.JSONDecodeError:
                            print(f"Port {self.port} ({self.name}): Failed to parse screen_image 'details' JSON: {str(details_payload)[:100]}...")
                            continue # Skip this corrupted screen_image's details
                        
                        # 1. Update persistent instance cache based on new raw data
                        self._update_instance_game_data_cache(current_raw_game_data_from_server)

                        # 2. Construct comprehensive game_state for the strategy by merging raw and cached data
                        comprehensive_game_state_for_strategy = {
                            **current_raw_game_data_from_server,    # All raw fields directly from server
                            **self.instance_game_data_cache      # All cached/derived fields for this instance
                        }
                        # Ensure 'inside_window' is a proper boolean for the strategy
                        comprehensive_game_state_for_strategy['inside_window'] = \
                            bool(float(current_raw_game_data_from_server.get("inside_window", 0)))

                        # --- DRL Experience Collection ---
                        # This block executes if we have a previous state (s_t) and action (a_t)
                        # current_raw_game_data_from_server represents the state s_{t+1} resulting from a_t
                        if self.prev_raw_game_state_for_reward and self.prev_action_info_for_buffer:
                            reward_t, done_t = self._calculate_reward_and_done(current_raw_game_data_from_server)
                            self.current_episode_reward += reward_t
                            self.current_episode_length += 1

                            # Add to experience buffer only if it was a DRL decision (value is not None)
                            # The value stored is V(s_t) from prev_action_info_for_buffer
                            if self.prev_action_info_for_buffer.get('value') is not None and \
                               self.prev_processed_stacked_frames is not None and \
                               self.prev_processed_game_features is not None:
                                self.experience_buffer.add(
                                    self.prev_processed_stacked_frames,  # s_t (frames)
                                    self.prev_processed_game_features,   # s_t (features)
                                    self.prev_action_info_for_buffer['action_idx'], # a_t
                                    self.prev_action_info_for_buffer['log_prob'],   # log_prob(a_t|s_t)
                                    reward_t,                                       # r_t
                                    self.prev_action_info_for_buffer['value'],      # V(s_t)
                                    done_t                                          # d_t
                                )
                                self.shared_episode_data['steps_collected_since_last_train'] += 1
                            
                            if done_t:
                                print(f"Port {self.port} ({self.name}): EPISODE END. Reward: {self.current_episode_reward:.2f}, Len: {self.current_episode_length}")
                                self.shared_episode_data['episode_rewards'].append(self.current_episode_reward)
                                self.shared_episode_data['episode_lengths'].append(self.current_episode_length)
                                # Reset episode-specific states for this handler
                                self.current_episode_reward = 0.0
                                self.current_episode_length = 0
                                self.ep_last_player_health = None 
                                self.ep_last_enemy_health = None
                                self.strategy.reset_state(self.port) # Reset strategy's internal state (frame stack, skip logic)
                                
                                # Clear previous state storage to prevent forming an invalid tuple across episodes
                                self.prev_raw_game_state_for_reward = None
                                self.prev_action_info_for_buffer = None 
                                self.prev_processed_stacked_frames = None
                                self.prev_processed_game_features = None
                                # Note: Game might need explicit reset actions. For now, assumes next state is usable.

                        # 4. Decide and take action for the current_raw_game_state (s_{t+1} or s_0)
                        # The strategy uses the comprehensive_game_state_for_strategy
                        action_info_from_strategy = self.strategy.decide_action(self.port, comprehensive_game_state_for_strategy)
                        
                        button_command_to_send = action_info_from_strategy['button_command']
                        if not await self._send_command_internal(button_command_to_send):
                            break # Send failed, stop processing for this instance

                        # 5. Store results of *this current step* to become the *previous step* in the next iteration
                        self.prev_raw_game_state_for_reward = comprehensive_game_state_for_strategy # This is s_t for next cycle
                        
                        # Store DRL-specific info only if the strategy provided valid DRL outputs (value is not None)
                        if action_info_from_strategy.get('value') is not None:
                            self.prev_processed_stacked_frames = action_info_from_strategy.get('stacked_frames_tensor')
                            self.prev_processed_game_features = action_info_from_strategy.get('game_features_tensor')
                            self.prev_action_info_for_buffer = {
                                'action_idx': action_info_from_strategy['action_idx'],
                                'log_prob': action_info_from_strategy['log_prob'],
                                'value': action_info_from_strategy['value'] # This is V(current_state)
                            }
                        else: # Strategy handled it as non-DRL (e.g. menu skip), or returned placeholder DRL values
                            self.prev_action_info_for_buffer = None # Don't store for DRL buffer
                            self.prev_processed_stacked_frames = None
                            self.prev_processed_game_features = None
        
        except (ConnectionResetError, BrokenPipeError):
            print(f"Port {self.port} ({self.name}): Connection reset/broken in message processor loop.")
        except asyncio.CancelledError:
            print(f"Port {self.port} ({self.name}): Message processor task was cancelled.")
        except Exception as e:
            print(f"Port {self.port} ({self.name}): Unexpected error in _message_processor_loop: {e}")
            print(traceback.format_exc())
        finally:
            self._is_running = False # Ensure flag is set if loop exits for any reason
            print(f"Port {self.port} ({self.name}): Message processor loop definitively ended.")

    async def start_handling(self):
        """Connects to the game instance and starts the send/receive loops."""
        self._is_running = True
        # Reset per-connection/episode state trackers
        self.ep_last_player_health = None 
        self.ep_last_enemy_health = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.prev_raw_game_state_for_reward = None # Reset S_t storage
        self.prev_action_info_for_buffer = None    # Reset A_t storage
        self.prev_processed_stacked_frames = None
        self.prev_processed_game_features = None
        
        # Reset strategy state (e.g., frame buffers, internal skip logic state) for this port
        if self.strategy:
            self.strategy.reset_state(self.port)

        print(f"Attempting to connect to {self.name} at {self.address}:{self.port}")
        try:
            self.reader, self.writer = await asyncio.open_connection(self.address, self.port)
            print(f"Successfully connected to {self.name} (Port {self.port})")

            # Start the message processing loop as a background task
            self.message_processor_task = asyncio.create_task(self._message_processor_loop())

            # Main loop for sending screen requests
            while self._is_running:
                if not await self._request_screen_image(): # This sets self._is_running=False on critical send failure
                    break 
                await asyncio.sleep(self.inference_interval) # Control request frequency
            
            # print(f"Port {self.port} ({self.name}): Screen request loop ended (is_running: {self._is_running}).")

        except ConnectionRefusedError:
            print(f"Port {self.port} ({self.name}): Connection refused. Is the game instance running and listening?")
        except asyncio.CancelledError:
            print(f"Port {self.port} ({self.name}): ConnectionHandler task for {self.name} was cancelled externally.")
        except Exception as e:
            print(f"Port {self.port} ({self.name}): Unhandled error in ConnectionHandler.start_handling: {e}")
            print(traceback.format_exc())
        finally:
            self._is_running = False # Ensure the flag is false before cleanup
            
            # Clean up the message processor task
            if self.message_processor_task and not self.message_processor_task.done():
                # print(f"Port {self.port} ({self.name}): Cancelling message processor task...")
                self.message_processor_task.cancel()
                try: await self.message_processor_task 
                except asyncio.CancelledError: pass # Expected upon cancellation
            
            # Clean up any active chip timer tasks for this instance
            for timer_key in ['player_chip_timer_task', 'enemy_chip_timer_task']:
                task = self.instance_game_data_cache.get(timer_key)
                if task and not task.done():
                    task.cancel()
                    try: await task
                    except asyncio.CancelledError: pass
            
            # Close the network writer
            if self.writer:
                try:
                    if not self.writer.is_closing():
                        self.writer.close()
                    await self.writer.wait_closed()
                except Exception as e: 
                    print(f"Port {self.port} ({self.name}): Error closing writer: {e}")
            # Reader is implicitly closed when writer is closed or connection drops.
            
            print(f"Connection handler for {self.name} (Port {self.port}) has fully shut down.")