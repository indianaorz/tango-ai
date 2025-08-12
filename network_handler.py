# network_handler.py
import asyncio
import json
import traceback
from collections import deque # Though DRLAgentStrategy handles its own frame deque now
import torch # For an empty tensor placeholder if needed
import sys


def _c(text, code):              # _c("txt", 32) -> green txt
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

class ConnectionHandler:
    def __init__(self, instance_config, active_strategy, inference_fps,
                 experience_buffer, shared_episode_data, config_module, utils_module):

        # ── existing assignments ────────────────────────────────────────
        self.instance_config_data = instance_config
        self.strategy            = active_strategy
        self.inference_interval  = 1.0 / inference_fps if inference_fps > 0 else 0.033
        self.experience_buffer   = experience_buffer
        self.shared_episode_data = shared_episode_data

        self.port    = instance_config['port']
        self.address = instance_config['address']
        self.name    = instance_config['name']

        self.cfg   = config_module
        self.utils = utils_module

        # NEW: flag controlled by ManualInputRouter
        self.manual_override = False         # ←────────────── add this line

        # ── rest of original __init__ (reader, writer, caches, etc.) ──
        self.reader = None
        self.writer = None
        self.message_processor_task = None
        self._is_running = False
        self.ep_last_player_charge = None   # NEW – tracks previous charge level


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

    def _charge_shaping(self, prev_raw, curr_raw):
        prev_lvl = prev_raw.get('player_charge', 0)
        curr_lvl = curr_raw.get('player_charge', 0)

        reward = 0.0
        # tiny drip‑feed while bar is filling
        if curr_lvl > prev_lvl:
            reward += (curr_lvl - prev_lvl) * self.cfg.REWARD_CHARGE_GAIN_COEF

        # big bonus the frame you let go (curr went to 0, prev > 0)
        if prev_lvl > 0 and curr_lvl == 0:
            reward += self.cfg.REWARD_CHARGE_RELEASE_BONUS

        return reward

    
    # ------------------------------------------------------------------
    def _calculate_reward_and_done(self, curr_raw):
        """
        Compute the shaped reward **and** a breakdown of its components.

        Returns
        -------
        reward : float          summed reward_t
        done   : bool           terminal flag
        comps  : dict[str,float] keys:
                 damage_dealt, damage_taken, charge_gain,
                 charge_shot, time_penalty, win, loss
        """
        comps = {
            "damage_dealt":  0.0,
            "damage_taken":  0.0,
            "charge_gain":   0.0,
            "charge_shot":   0.0,
            "time_penalty":  self.cfg.REWARD_TIME_PENALTY_STEP,
            "win":           0.0,
            "loss":          0.0,
        }

        # ── bootstrap trackers ───────────────────────────────────────
        if self.ep_last_player_health is None:
            self.ep_last_player_health = curr_raw.get("player_health", 0)
        if self.ep_last_enemy_health is None:
            self.ep_last_enemy_health  = curr_raw.get("enemy_health", 0)
        if self.ep_last_player_charge is None:
            self.ep_last_player_charge = curr_raw.get("player_charge", 0)

        # ── current values ───────────────────────────────────────────
        hp_p  = curr_raw.get("player_health", self.ep_last_player_health)
        hp_e  = curr_raw.get("enemy_health",  self.ep_last_enemy_health)
        ch    = curr_raw.get("player_charge", self.ep_last_player_charge)

        dmg_e      = self.ep_last_enemy_health  - hp_e
        dmg_p      = self.ep_last_player_health - hp_p
        ch_gain    = ch - self.ep_last_player_charge
        released   = (
            self.ep_last_player_charge >= self.cfg.CHARGE_MAX_LEVEL
            and ch == 0 and dmg_e > 0
        )

        # ── component calculations ───────────────────────────────────
        if dmg_e > 0:
            comps["damage_dealt"] = (
                dmg_e * self.cfg.REWARD_DAMAGE_DEALT_MULTIPLIER
            )
        if dmg_p > 0:
            comps["damage_taken"] = (
                dmg_p * self.cfg.REWARD_DAMAGE_TAKEN_MULTIPLIER
            )
        if ch_gain > 0:
            comps["charge_gain"]  = (
                ch_gain * self.cfg.REWARD_CHARGE_GAIN_COEF
            )
        if released:
            comps["charge_shot"]  = self.cfg.REWARD_CHARGE_RELEASE_BONUS

        # win / loss bonus
        done = False
        if hp_p <= 0:
            comps["loss"] = self.cfg.REWARD_LOSE_GAME
            done = True
        elif hp_e <= 0:
            comps["win"]  = self.cfg.REWARD_WIN_GAME
            done = True

        reward = sum(comps.values())

        # ── debug print (when meaningful) ────────────────────────────
        noteworthy = abs(reward) >= 0.01 or released
        if noteworthy:
            parts = []
            if dmg_e   > 0: parts.append(_c(f"-{dmg_e} HP enemy", 32))
            if dmg_p   > 0: parts.append(_c(f"+{dmg_p} HP self", 31))
            if ch_gain > 0: parts.append(_c(f"+{ch_gain} charge", 36))
            if released:    parts.append(_c("FULL‑SHOT!", 35))
            if done and hp_e <= 0: parts.append(_c("WIN", 92))
            if done and hp_p <= 0: parts.append(_c("LOSS", 91))
            print(
                f"Port {self.port} ({self.name}): "
                f"{' | '.join(parts)}  →  r={reward:+.2f}"
            )

        # ── carry forward ────────────────────────────────────────────
        self.ep_last_player_health  = hp_p
        self.ep_last_enemy_health   = hp_e
        self.ep_last_player_charge  = ch

        return reward, done, comps

    # ------------------------------------------------------------------
    async def _message_processor_loop(self):
        """Continuously processes messages from the game instance."""
        json_buffer = ""
        try:
            while self._is_running:
                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                # 1. read from socket (non‑blocking with timeout)
                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                if not self.reader or self.reader.at_eof():
                    print(f"Port {self.port} ({self.name}): reader closed/EOF – leaving loop.")
                    break

                try:
                    data_chunk = await asyncio.wait_for(
                        self.reader.read(8192),
                        timeout=max(0.2, self.inference_interval * 5),
                    )
                except asyncio.TimeoutError:
                    continue  # just try again

                if not data_chunk:          # peer closed connection
                    print(f"Port {self.port} ({self.name}): connection closed by peer.")
                    break

                json_buffer += data_chunk.decode(errors="ignore")

                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                # 2. process all complete JSON lines
                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                while "\n" in json_buffer and self._is_running:
                    msg_str, json_buffer = json_buffer.split("\n", 1)
                    msg_str = msg_str.strip()
                    if not msg_str:
                        continue

                    try:
                        parsed = json.loads(msg_str)
                    except json.JSONDecodeError:
                        print(f"Port {self.port}: bad JSON → {msg_str[:80]}…")
                        continue

                    event  = parsed.get("event", "Unknown")
                    detail = parsed.get("details", {})
                    if not isinstance(detail, (str, dict)):
                        detail = str(detail)

                    # ============================================================== #
                    #  ░ screen_image  →  main RL update path
                    # ============================================================== #
                    if event != "screen_image":
                        continue

                    # 2‑a. raw payload ------------------------------------------------
                    try:
                        current_raw = (
                            json.loads(detail) if isinstance(detail, str) else detail
                        )
                    except json.JSONDecodeError:
                        print(f"Port {self.port}: bad detail JSON → {str(detail)[:80]}…")
                        continue

                    # 2‑b. update long‑lived cache -----------------------------------
                    self._update_instance_game_data_cache(current_raw)

                    # 2‑c. build comprehensive state ---------------------------------
                    comp_state = {**current_raw, **self.instance_game_data_cache}
                    comp_state["inside_window"] = bool(
                        float(current_raw.get("inside_window", 0))
                    )

                    # 2‑d. reward + buffer insertion ----------------------------------
                    if (
                        self.prev_raw_game_state_for_reward
                        and self.prev_action_info_for_buffer
                    ):
                        r_t, done_t, comps_t = self._calculate_reward_and_done(current_raw)
                        self.current_episode_reward += r_t
                        self.current_episode_length += 1

                        # push transition to buffer if previous action came from DRL
                        if (
                            self.prev_action_info_for_buffer.get("value") is not None
                            and self.prev_processed_stacked_frames is not None
                            and self.prev_processed_game_features is not None
                        ):
                            self.experience_buffer.add(
                                self.prev_processed_stacked_frames,
                                self.prev_processed_game_features,
                                self.prev_action_info_for_buffer["action_idx"],
                                self.prev_action_info_for_buffer["log_prob"],
                                r_t,
                                self.prev_action_info_for_buffer["value"],
                                done_t,
                            )
                            self.shared_episode_data[
                                "steps_collected_since_last_train"
                            ] += 1

                        # ── NEW ▸ accumulate per‑component rewards
                        for k, v in comps_t.items():
                            self.shared_episode_data["reward_batch"][k]      += v
                            self.shared_episode_data["reward_cumulative"][k] += v

                        # episode end housekeeping
                        if done_t:
                            print(
                                f"Port {self.port} ({self.name}): EPISODE END. "
                                f"Reward {self.current_episode_reward:.2f}  "
                                f"Len {self.current_episode_length}"
                            )
                            self.shared_episode_data["episode_rewards"].append(
                                self.current_episode_reward
                            )
                            self.shared_episode_data["episode_lengths"].append(
                                self.current_episode_length
                            )
                            self.current_episode_reward = 0.0
                            self.current_episode_length = 0
                            self.ep_last_player_health  = None
                            self.ep_last_enemy_health   = None
                            self.strategy.reset_state(self.port)

                            # clear previous‑step caches
                            self.prev_raw_game_state_for_reward = None
                            self.prev_action_info_for_buffer    = None
                            self.prev_processed_stacked_frames  = None
                            self.prev_processed_game_features   = None

                    # 2‑e. decide next action -----------------------------------------
                    if not self.manual_override:
                        act_info = self.strategy.decide_action(self.port, comp_state)
                        if not await self._send_command_internal(act_info["button_command"]):
                            break  # send failed – drop out

                        self.prev_processed_stacked_frames = act_info.get("stacked_frames_tensor")
                        self.prev_processed_game_features  = act_info.get("game_features_tensor")
                        if act_info.get("value") is not None:
                            self.prev_action_info_for_buffer = {
                                "action_idx": act_info["action_idx"],
                                "log_prob":   act_info["log_prob"],
                                "value":      act_info["value"],
                            }
                        else:
                            self.prev_action_info_for_buffer = None
                    else:
                        # manual override – do not store DRL transition
                        self.prev_processed_stacked_frames = None
                        self.prev_processed_game_features  = None
                        self.prev_action_info_for_buffer   = None

                    # 2‑f. save raw state for next reward computation -----------------
                    self.prev_raw_game_state_for_reward = comp_state

        except (ConnectionResetError, BrokenPipeError):
            print(f"Port {self.port} ({self.name}): connection reset/broken.")
        except asyncio.CancelledError:
            print(f"Port {self.port} ({self.name}): processor task cancelled.")
        except Exception as e:
            print(f"Port {self.port}: unexpected error → {e}\n{traceback.format_exc()}")
        finally:
            self._is_running = False
            print(f"Port {self.port} ({self.name}): processor loop ended.")

        # ────────────────────────────────────────────────────────────────────────
    async def start_handling(self, max_retries: int = 0,  # 0 = infinite
                             retry_base_delay: float = 0.5,
                             retry_max_delay:  float = 5.0):
        """
        Connect to the game instance and start send/receive loops.
        Will keep retrying until it succeeds **or** the global shutdown sets
        ``self._is_running = False`` (e.g. in the training‑loop’s ``finally``).
        """
        self._is_running = True          # master flag checked by both loops
        attempt, delay = 0, retry_base_delay

        # ── keep trying to open the socket ───────────────────────────
        while self._is_running:
            try:
                print(f"Attempting to connect to {self.name} at "
                      f"{self.address}:{self.port} (try {attempt + 1})")
                self.reader, self.writer = await asyncio.open_connection(
                    self.address, self.port)
                print(f"Successfully connected to {self.name} "
                      f"(Port {self.port})")
                break                                    # ↲ success
            except ConnectionRefusedError:
                print(f"Port {self.port} ({self.name}): connection refused.")
            except Exception as e:
                print(f"Port {self.port} ({self.name}): unexpected error "
                      f"while connecting: {e}")

            attempt += 1
            if max_retries and attempt >= max_retries:
                print(f"Port {self.port} ({self.name}): exceeded "
                      f"{max_retries} retries; abandoning handler.")
                self._is_running = False
                return

            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_max_delay)     # exponential back‑off

        if not self._is_running:        # e.g. global shutdown during retries
            return

        # ── reset per‑episode state and start background tasks ───────
        self.ep_last_player_health = None
        self.ep_last_enemy_health  = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.prev_raw_game_state_for_reward = None
        self.prev_action_info_for_buffer   = None
        self.prev_processed_stacked_frames = None
        self.prev_processed_game_features  = None

        if self.strategy:
            self.strategy.reset_state(self.port)

        # background task for inbound messages
        self.message_processor_task = asyncio.create_task(
            self._message_processor_loop())

        # ── main send‑request loop ───────────────────────────────────
        try:
            while self._is_running:
                if not await self._request_screen_image():
                    break
                await asyncio.sleep(self.inference_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Port {self.port} ({self.name}): unhandled error in "
                  f"request loop: {e}\n{traceback.format_exc()}")
        finally:
            # graceful shutdown (same logic as before) ----------------
            self._is_running = False
            if self.message_processor_task and not self.message_processor_task.done():
                self.message_processor_task.cancel()
                try:
                    await self.message_processor_task
                except asyncio.CancelledError:
                    pass

            for timer_key in ('player_chip_timer_task', 'enemy_chip_timer_task'):
                task = self.instance_game_data_cache.get(timer_key)
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            if self.writer:
                try:
                    if not self.writer.is_closing():
                        self.writer.close()
                    await self.writer.wait_closed()
                except Exception:
                    pass

            print(f"Connection handler for {self.name} "
                  f"(Port {self.port}) has fully shut down.")
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