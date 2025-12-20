# ── Begin: network_handler.py ──
# network_handler.py
import asyncio
import json
import traceback
from collections import deque
from strategy import DRLAgentStrategy
import torch 
import sys
import os
from typing import Optional
import socket


def _c(text, code):             # _c("txt", 32) -> green txt
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

class ConnectionHandler:
    def __init__(
        self,
        instance_config,
        active_strategy,
        inference_fps,
        experience_buffer: Optional[object] = None,
        shared_episode_data: Optional[dict] = None,
        config_module=None,
        utils_module=None,
        policy_eval_strategy=None,
    ):
        # ── existing assignments ────────────────────────────────────────
        self.instance_config_data = instance_config
        self.strategy            = active_strategy
        self.inference_interval  = 1.0 / inference_fps if inference_fps > 0 else 0.0

        # Optional (training/analytics)
        self.experience_buffer   = experience_buffer
        self.shared_episode_data = shared_episode_data

        # Optional (evaluate scripted actions under a policy)
        self.policy_eval = policy_eval_strategy

        self.port    = instance_config['port']
        self.address = instance_config['address']
        self.name    = instance_config['name']

        self.cfg   = config_module
        self.utils = utils_module

        # NEW: flag controlled by ManualInputRouter
        self.manual_override = False

        # ── rest of original __init__ (reader, writer, caches, etc.) ──
        self.reader = None
        self.writer = None
        self.message_processor_task = None
        self._is_running = False
        self.ep_last_player_charge = None   # NEW – tracks previous charge level
        
        # ── NEW: Synchronization event for lock-step FPS ────────────────
        self._inference_done_event = asyncio.Event()
        # Pre-set so the first loop iteration doesn't block indefinitely
        self._inference_done_event.set()

        # ── NEW: logging gate ─────────────────────────────────────────
        self.only_learner_logs = bool(int(os.getenv("LOG_ONLY_LEARNER", "1")))
        self.is_learner = ("Learner" in self.name)

        def _should_log():
            return (not self.only_learner_logs) or self.is_learner

        # bind as methods so we can use in other funcs
        self._should_log = _should_log
        self._log = lambda msg: (print(msg) if self._should_log() else None)


        # Per-instance cache for data not in every screen_data or derived over time
        self.instance_game_data_cache = {
            'player_chips_in_window': [], 'current_hand_selected': [], 
            'player_used_crosses_list': [], 'enemy_used_crosses_list': [], 
            'is_player_beasted_out': False, 'is_enemy_beasted_out': False,
            'is_player_beasted_over': False, 'is_enemy_beasted_over': False,
            'cached_player_folder': [], 'cached_enemy_folder': [], 
            'player_active_chip': 0, 'enemy_active_chip': 0, 
            'player_chip_timer_task': None, 'enemy_chip_timer_task': None, 
            'previous_player_chip_from_server': 0, 
            'previous_enemy_chip_from_server': 0,  
        }

        # State for constructing (s_t, a_t, r_t, s_t+1, d_t) experience tuples for DRL
        self.prev_raw_game_state_for_reward = None 
        self.prev_processed_stacked_frames = None  
        self.prev_processed_game_features = None   
        self.prev_action_info_for_buffer = None    

        # For reward calculation and per-episode stats for this handler
        self.ep_last_player_health = None
        self.ep_last_enemy_health = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.collect_experience = isinstance(active_strategy, DRLAgentStrategy) and ("Learner" in self.name)


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
        self._is_running = False 
        return False

    async def _request_screen_image(self):
        """Sends a request to the game instance for a new screen image and game state."""
        return await self._send_command_internal({'type': 'request_screen', 'key': ''})

    async def _handle_active_chip_timer(self, chip_owner_type):
        """Resets the 'active_chip' in cache after a delay (visual effect duration)."""
        await asyncio.sleep(1.0) 
        cache_key_active_chip = f'{chip_owner_type}_active_chip'
        cache_key_timer_task = f'{chip_owner_type}_chip_timer_task'
        
        self.instance_game_data_cache[cache_key_active_chip] = 0 
        if self.instance_game_data_cache.get(cache_key_timer_task):
            self.instance_game_data_cache[cache_key_timer_task] = None 

    def _update_instance_game_data_cache(self, screen_data_dict):
        """
        Updates the persistent per-instance cache based on fresh screen_data from the server.
        """
        cache = self.instance_game_data_cache

        # Detect player chip usage
        new_player_chip_id = screen_data_dict.get("player_chip", 0)
        if new_player_chip_id != cache['previous_player_chip_from_server'] and cache['previous_player_chip_from_server'] != 0:
            cache['player_active_chip'] = cache['previous_player_chip_from_server'] 
            if cache['player_chip_timer_task']:
                cache['player_chip_timer_task'].cancel() 
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
        if player_game_emotion >= 11: cache['is_player_beasted_out'] = True 
        if player_game_emotion >= 23: cache['is_player_beasted_over'] = True 
        if enemy_game_emotion >= 11: cache['is_enemy_beasted_out'] = True
        if enemy_game_emotion >= 23: cache['is_enemy_beasted_over'] = True
        
        # Cache initial folder data 
        if not cache['cached_player_folder'] and "player_chip_folder" in screen_data_dict:
            cache['cached_player_folder'] = screen_data_dict.get("player_chip_folder", [])
        if not cache['cached_enemy_folder'] and "enemy_chip_folder" in screen_data_dict:
            cache['cached_enemy_folder'] = screen_data_dict.get("enemy_chip_folder", [])
            
        # Cache chips visible in the custom window
        if bool(float(screen_data_dict.get("inside_window", 0))):
            chip_slots = screen_data_dict.get("chip_slots", [])
            chip_codes = screen_data_dict.get("chip_codes", [])
            chips_visible_count = screen_data_dict.get("chip_visible_count", 0)
            visible_chips_in_window = []
            for i in range(min(len(chip_slots), chips_visible_count)):
                visible_chips_in_window.append({'slot': chip_slots[i], 'code': chip_codes[i], 'id_in_window': i})
            cache['player_chips_in_window'] = visible_chips_in_window
        else:
            if cache['player_chips_in_window']: 
                cache['player_chips_in_window'] = []
        

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

        dmg_e       = self.ep_last_enemy_health  - hp_e
        dmg_p       = self.ep_last_player_health - hp_p
        ch_gain     = ch - self.ep_last_player_charge
        released    = (
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
        if noteworthy and self._should_log():
            parts = []
            if dmg_e   > 0: parts.append(_c(f"-{dmg_e} HP enemy", 32))
            if dmg_p   > 0: parts.append(_c(f"+{dmg_p} HP self", 31))
            if ch_gain > 0: parts.append(_c(f"+{ch_gain} charge", 36))
            if released:    parts.append(_c("FULL‑SHOT!", 35))
            if done and hp_e <= 0: parts.append(_c("WIN", 92))
            if done and hp_p <= 0: parts.append(_c("LOSS", 91))
            self._log(
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
                # 1) read from socket (non-blocking with timeout)
                if not self.reader or self.reader.at_eof():
                    print(f"Port {self.port} ({self.name}): reader closed/EOF – leaving loop.")
                    break

                try:
                    data_chunk = await asyncio.wait_for(
                        self.reader.read(8192),
                        timeout=max(0.5, self.inference_interval * 20), # generous timeout
                    )
                except asyncio.TimeoutError:
                    continue  # try again

                if not data_chunk:  # peer closed connection
                    print(f"Port {self.port} ({self.name}): connection closed by peer.")
                    break

                json_buffer += data_chunk.decode(errors="ignore")

                # 2) process all complete JSON lines
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

                    # ─────────────────────────────────────────────────────────────
                    # Only handle screen frames here
                    # ─────────────────────────────────────────────────────────────
                    if event != "screen_image":
                        continue

                    # 2-a) parse raw payload
                    try:
                        current_raw = json.loads(detail) if isinstance(detail, str) else detail
                    except json.JSONDecodeError:
                        print(f"Port {self.port}: bad detail JSON → {str(detail)[:80]}…")
                        continue

                    # 2-b) update long-lived cache
                    self._update_instance_game_data_cache(current_raw)

                    # 2-c) build comprehensive state
                    comp_state = {**current_raw, **self.instance_game_data_cache}
                    comp_state["inside_window"] = bool(float(current_raw.get("inside_window", 0)))

                    # 2-d) reward + buffer insertion (for the *previous* action)
                    if self.prev_raw_game_state_for_reward and self.prev_action_info_for_buffer:
                        r_t, done_t, comps_t = self._calculate_reward_and_done(current_raw)
                        self.current_episode_reward += r_t
                        self.current_episode_length += 1

                        if (
                            self.prev_action_info_for_buffer.get("value") is not None
                            and self.prev_processed_stacked_frames is not None
                            and self.prev_processed_game_features is not None
                            and self.collect_experience
                        ):
                            if self.experience_buffer is not None:
                                self.experience_buffer.add(
                                    self.prev_processed_stacked_frames,
                                    self.prev_processed_game_features,
                                    self.prev_action_info_for_buffer["action_idx"],
                                    self.prev_action_info_for_buffer["log_prob"],
                                    r_t,
                                    self.prev_action_info_for_buffer["value"],
                                    done_t,
                                )

                            if self.shared_episode_data is not None:
                                self.shared_episode_data["steps_collected_since_last_train"] = (
                                    self.shared_episode_data.get("steps_collected_since_last_train", 0) + 1
                                )

                        # accumulate per-component rewards
                        if self.shared_episode_data is not None:
                            rb = self.shared_episode_data.setdefault("reward_batch", {})
                            rc = self.shared_episode_data.setdefault("reward_cumulative", {})
                            for k, v in comps_t.items():
                                rb[k] = rb.get(k, 0.0) + v
                                rc[k] = rc.get(k, 0.0) + v


                        # episode end housekeeping
                        if done_t:
                            if self._should_log():
                                self._log(
                                    f"Port {self.port} ({self.name}): EPISODE END. "
                                    f"Reward {self.current_episode_reward:.2f}  "
                                    f"Len {self.current_episode_length}"
                                )

                            if self.shared_episode_data is not None:
                                self.shared_episode_data.setdefault("episode_rewards", []).append(self.current_episode_reward)
                                self.shared_episode_data.setdefault("episode_lengths", []).append(self.current_episode_length)

                            self.current_episode_reward = 0.0
                            self.current_episode_length = 0
                            self.ep_last_player_health  = None
                            self.ep_last_enemy_health   = None
                            self.strategy.reset_state(self.port)

                            # clear previous-step caches
                            self.prev_raw_game_state_for_reward = None
                            self.prev_action_info_for_buffer    = None
                            self.prev_processed_stacked_frames  = None
                            self.prev_processed_game_features   = None

                    # 2-e) decide next action (or skip if manual override)
                    if not self.manual_override:
                        # --------------------------------------------------------------
                        # CRITICAL SPEED FIX: 
                        # Run the strategy (decoding+inference) in a separate thread so
                        # we don't block the asyncio event loop for other handlers.
                        # --------------------------------------------------------------
                        act_info = await asyncio.to_thread(self.strategy.decide_action, self.port, comp_state)
                        
                        if not await self._send_command_internal(act_info["button_command"]):
                            break  # send failed – drop out

                        # Prefer tensors/logprob/value from the acting strategy (DRL path)
                        seq_f = act_info.get("stacked_frames_tensor")
                        seq_g = act_info.get("game_features_tensor")

                        if act_info.get("value") is not None and seq_f is not None and seq_g is not None:
                            # Pure DRL step: we already have everything
                            self.prev_processed_stacked_frames = seq_f
                            self.prev_processed_game_features  = seq_g
                            self.prev_action_info_for_buffer   = {
                                "action_idx": act_info["action_idx"],
                                "log_prob":   act_info["log_prob"],
                                "value":      act_info["value"],
                            }
                        else:
                            # Scripted (or non-DRL) step: evaluate the *taken* action under the DRL policy
                            self.prev_action_info_for_buffer = None
                            self.prev_processed_stacked_frames = None
                            self.prev_processed_game_features  = None

                            try:
                                # Build policy inputs if missing
                                if (seq_f is None or seq_g is None) and self.policy_eval is not None:
                                    seq_f, seq_g = self.policy_eval.encode_for_policy(self.port, comp_state)

                                if self.policy_eval is not None and seq_f is not None and seq_g is not None:
                                    # Decode pressed mask → discrete action index
                                    key_str  = act_info["button_command"]["key"]         # "0100...”
                                    mask_int = int(key_str, 2)
                                    a_idx_int = self.utils.bitmask_to_action_index(mask_int)

                                    # Evaluate that action under current policy (to get old_log_prob & V)
                                    with torch.no_grad():
                                        _, log_p_demo, _, v_demo, _ = self.policy_eval.model.get_action_and_value(
                                            seq_f.to(self.policy_eval.dev),
                                            seq_g.to(self.policy_eval.dev),
                                            torch.tensor(a_idx_int, device=self.policy_eval.dev),
                                        )

                                    # Store for next tick’s reward insert
                                    self.prev_processed_stacked_frames = seq_f
                                    self.prev_processed_game_features  = seq_g
                                    self.prev_action_info_for_buffer   = {
                                        "action_idx": torch.tensor(a_idx_int, device=self.policy_eval.dev),
                                        "log_prob":   log_p_demo.detach(),
                                        "value":      v_demo.detach(),
                                    }
                            except Exception as _e:
                                # Couldn’t evaluate scripted action — skip learning for this step
                                self.prev_action_info_for_buffer   = None
                                self.prev_processed_stacked_frames = None
                                self.prev_processed_game_features  = None
                    else:
                        # manual override – do not store DRL transition (human path handled elsewhere)
                        self.prev_processed_stacked_frames = None
                        self.prev_processed_game_features  = None
                        self.prev_action_info_for_buffer   = None

                    # 2-f) save raw state for next reward computation
                    self.prev_raw_game_state_for_reward = comp_state
                    
                    # 2-g) Signal that we are ready for the next frame
                    self._inference_done_event.set()

        except (ConnectionResetError, BrokenPipeError):
            print(f"Port {self.port} ({self.name}): connection reset/broken.")
        except asyncio.CancelledError:
            print(f"Port {self.port} ({self.name}): processor task cancelled.")
        except Exception as e:
            print(f"Port {self.port}: unexpected error → {e}\n{traceback.format_exc()}")
        finally:
            self._is_running = False
            self._inference_done_event.set() # Ensure waiters wake up to exit
            print(f"Port {self.port} ({self.name}): processor loop ended.")

    
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
                print(f"Attempting to connect to {self.name} at {self.address}:{self.port} ...")
                self.reader, self.writer = await asyncio.open_connection(self.address, self.port)
                
                # [FIX] Disable Nagle's algorithm on the Python side too
                sock = self.writer.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                print(f"Successfully connected to {self.name} (Port {self.port})")
                break
            except ConnectionRefusedError:
                print(f"Port {self.port} ({self.name}): connection refused.")
            except Exception as e:
                print(f"Port {self.port} ({self.name}): unexpected error "
                      f"while connecting: {e}")

            attempt += 1
            if max_retries and attempt >= max_retries:
                print(f"Port {self.port} ({self.name}): exceeded "
                      f"{max_retries} retries; abandoning handler.")
                self._is_running = False
                return

            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_max_delay)      # exponential back‑off

        if not self._is_running:        # e.g. global shutdown during retries
            return

        # ── reset per‑episode state and start background tasks ───────
        self.ep_last_player_health = None
        self.ep_last_enemy_health  = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.prev_raw_game_state_for_reward = None
        self.prev_action_info_for_buffer    = None
        self.prev_processed_stacked_frames = None
        self.prev_processed_game_features  = None
        self._inference_done_event.set() # Start open

        if self.strategy:
            self.strategy.reset_state(self.port)

        # background task for inbound messages
        self.message_processor_task = asyncio.create_task(
            self._message_processor_loop())

        # ── main send‑request loop ───────────────────────────────────
        try:
            while self._is_running:
                # 1. Wait for previous inference cycle to finish (Lock-Step)
                #    This prevents queue buildup and ensures we process at max speed the system allows.
                await self._inference_done_event.wait()
                if not self._is_running: break
                
                self._inference_done_event.clear()

                # 2. Request next frame
                if not await self._request_screen_image():
                    break
                
                # 3. Optional throttle (if inference_interval > 0, we can wait, otherwise run full speed)
                #    Since we are using lock-step, this is just a minimum frame time.
                if self.inference_interval > 0:
                    await asyncio.sleep(self.inference_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Port {self.port} ({self.name}): unhandled error in "
                  f"request loop: {e}\n{traceback.format_exc()}")
        finally:
            # graceful shutdown (same logic as before) ----------------
            self._is_running = False
            self._inference_done_event.set() # wake up if stuck
            
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
                  f"(Port {self.port}) has fully shut down.")
# ── End: network_handler.py ──