# network_handler.py
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import traceback
import time as _time
from collections import deque
from typing import Any, Dict, Optional

import torch

from strategy import DRLAgentStrategy


def _c(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


class ConnectionHandler:
    def __init__(
        self,
        instance_config: Dict[str, Any],
        active_strategy: Any,
        inference_fps: float,
        experience_buffer: Optional[object] = None,
        shared_episode_data: Optional[dict] = None,
        config_module: Any = None,
        utils_module: Any = None,
        policy_eval_strategy: Any = None,
    ):
        # ---- core wiring ----
        self.instance_config_data = instance_config
        self.strategy = active_strategy
        self.inference_interval = 1.0 / float(inference_fps) if inference_fps and float(inference_fps) > 0 else 0.0

        self.port = int(instance_config["port"])
        self.address = str(instance_config["address"])
        self.name = str(instance_config["name"])

        self.cfg = config_module
        self.utils = utils_module

        self.experience_buffer = experience_buffer
        self.shared_episode_data = shared_episode_data
        self.policy_eval = policy_eval_strategy

        self.manual_override = False

        # ---- networking state ----
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self._action_task: Optional[asyncio.Task] = None
        self._is_running = False

        # CRITICAL: serialize all writes to the socket (request loop + playback loop)
        self._send_lock = asyncio.Lock()

        # ---- lock-step sync ----
        self._inference_done_event = asyncio.Event()
        self._inference_done_event.set()

        # ---- action playback ----
        self.action_fps = float(os.getenv("NG_ACTION_FPS", "60").strip() or "60")
        self._action_dt = 1.0 / self.action_fps if self.action_fps > 0 else 0.0
        self._action_plan = deque()
        self._action_last_key = "0" * 16
        self._action_lock = asyncio.Lock()

        # ---- logging gate ----
        self.only_learner_logs = bool(int(os.getenv("LOG_ONLY_LEARNER", "1")))
        self.is_learner = ("Learner" in self.name)

        def _should_log() -> bool:
            return (not self.only_learner_logs) or self.is_learner

        self._should_log = _should_log
        self._log = lambda msg: (print(msg) if self._should_log() else None)

        # ---- debug counters ----
        self._rx_lines = 0
        self._rx_events: Dict[str, int] = {}
        self._rx_first_lines_left = int(os.getenv("DBG_FIRST_LINES", "5"))
        self._tx_request_screen = 0

        # ---- per-instance cache ----
        self.instance_game_data_cache = {
            "player_chips_in_window": [],
            "current_hand_selected": [],
            "player_used_crosses_list": [],
            "enemy_used_crosses_list": [],
            "is_player_beasted_out": False,
            "is_enemy_beasted_out": False,
            "is_player_beasted_over": False,
            "is_enemy_beasted_over": False,
            "cached_player_folder": [],
            "cached_enemy_folder": [],
            "player_active_chip": 0,
            "enemy_active_chip": 0,
            "player_chip_timer_task": None,
            "enemy_chip_timer_task": None,
            "previous_player_chip_from_server": 0,
            "previous_enemy_chip_from_server": 0,
        }

        # ---- DRL step caches ----
        self.prev_raw_game_state_for_reward = None
        self.prev_processed_stacked_frames = None
        self.prev_processed_game_features = None
        self.prev_action_info_for_buffer = None

        # ---- per-episode reward trackers ----
        self.ep_last_player_health = None
        self.ep_last_enemy_health = None
        self.ep_last_player_charge = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.collect_experience = isinstance(active_strategy, DRLAgentStrategy) and ("Learner" in self.name)

    # ---------------------------------------------------------------------
    # Socket send helpers (SERIALIZED!)
    # ---------------------------------------------------------------------
    async def _send_command_internal(self, command_dict: Dict[str, Any]) -> bool:
        """
        CRITICAL INVARIANT:
          Only one coroutine may write to self.writer at a time.
        """
        if not self.writer or self.writer.is_closing():
            print(f"Port {self.port} ({self.name}): writer not available or closing.")
            self._is_running = False
            return False

        try:
            payload = json.dumps(command_dict)
            data = payload.encode() + b"\n"

            async with self._send_lock:
                # Re-check inside lock, in case we got closed while waiting
                if not self.writer or self.writer.is_closing():
                    self._is_running = False
                    return False
                self.writer.write(data)
                await self.writer.drain()

            return True
        except (ConnectionResetError, BrokenPipeError):
            print(f"Port {self.port} ({self.name}): connection closed while sending.")
        except Exception as e:
            print(f"Port {self.port} ({self.name}): failed to send command: {e}")

        self._is_running = False
        return False

    async def _request_screen_image(self) -> bool:
        self._tx_request_screen += 1
        if self._tx_request_screen <= 5 and self._should_log():
            self._log(f"Port {self.port} ({self.name}): TX request_screen #{self._tx_request_screen}")
        return await self._send_command_internal({"type": "request_screen", "key": ""})

    # ---------------------------------------------------------------------
    # Cache updates
    # ---------------------------------------------------------------------
    async def _handle_active_chip_timer(self, chip_owner_type: str) -> None:
        await asyncio.sleep(1.0)
        cache_key_active_chip = f"{chip_owner_type}_active_chip"
        cache_key_timer_task = f"{chip_owner_type}_chip_timer_task"
        self.instance_game_data_cache[cache_key_active_chip] = 0
        if self.instance_game_data_cache.get(cache_key_timer_task):
            self.instance_game_data_cache[cache_key_timer_task] = None

    def _update_instance_game_data_cache(self, screen_data_dict: Dict[str, Any]) -> None:
        cache = self.instance_game_data_cache

        new_player_chip_id = screen_data_dict.get("player_chip", 0)
        if new_player_chip_id != cache["previous_player_chip_from_server"] and cache["previous_player_chip_from_server"] != 0:
            cache["player_active_chip"] = cache["previous_player_chip_from_server"]
            if cache["player_chip_timer_task"]:
                cache["player_chip_timer_task"].cancel()
            cache["player_chip_timer_task"] = asyncio.create_task(self._handle_active_chip_timer("player"))
        cache["previous_player_chip_from_server"] = new_player_chip_id

        new_enemy_chip_id = screen_data_dict.get("enemy_chip", 0)
        if new_enemy_chip_id != cache["previous_enemy_chip_from_server"] and cache["previous_enemy_chip_from_server"] != 0:
            cache["enemy_active_chip"] = cache["previous_enemy_chip_from_server"]
            if cache["enemy_chip_timer_task"]:
                cache["enemy_chip_timer_task"].cancel()
            cache["enemy_chip_timer_task"] = asyncio.create_task(self._handle_active_chip_timer("enemy"))
        cache["previous_enemy_chip_from_server"] = new_enemy_chip_id

        player_game_emotion = screen_data_dict.get("player_game_emotion", 0)
        enemy_game_emotion = screen_data_dict.get("enemy_game_emotion", 0)
        if player_game_emotion >= 11:
            cache["is_player_beasted_out"] = True
        if player_game_emotion >= 23:
            cache["is_player_beasted_over"] = True
        if enemy_game_emotion >= 11:
            cache["is_enemy_beasted_out"] = True
        if enemy_game_emotion >= 23:
            cache["is_enemy_beasted_over"] = True

        if not cache["cached_player_folder"] and "player_chip_folder" in screen_data_dict:
            cache["cached_player_folder"] = screen_data_dict.get("player_chip_folder", [])
        if not cache["cached_enemy_folder"] and "enemy_chip_folder" in screen_data_dict:
            cache["cached_enemy_folder"] = screen_data_dict.get("enemy_chip_folder", [])

        if bool(float(screen_data_dict.get("inside_window", 0))):
            chip_slots = screen_data_dict.get("chip_slots", [])
            chip_codes = screen_data_dict.get("chip_codes", [])
            chips_visible_count = screen_data_dict.get("chip_visible_count", 0)
            visible_chips_in_window = []
            for i in range(min(len(chip_slots), chips_visible_count)):
                visible_chips_in_window.append({"slot": chip_slots[i], "code": chip_codes[i], "id_in_window": i})
            cache["player_chips_in_window"] = visible_chips_in_window
        else:
            if cache["player_chips_in_window"]:
                cache["player_chips_in_window"] = []

    # ---------------------------------------------------------------------
    # Reward & done (unchanged)
    # ---------------------------------------------------------------------
    def _calculate_reward_and_done(self, curr_raw: Dict[str, Any]):
        comps = {
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "charge_gain": 0.0,
            "charge_shot": 0.0,
            "time_penalty": self.cfg.REWARD_TIME_PENALTY_STEP,
            "win": 0.0,
            "loss": 0.0,
        }

        if self.ep_last_player_health is None:
            self.ep_last_player_health = curr_raw.get("player_health", 0)
        if self.ep_last_enemy_health is None:
            self.ep_last_enemy_health = curr_raw.get("enemy_health", 0)
        if self.ep_last_player_charge is None:
            self.ep_last_player_charge = curr_raw.get("player_charge", 0)

        hp_p = curr_raw.get("player_health", self.ep_last_player_health)
        hp_e = curr_raw.get("enemy_health", self.ep_last_enemy_health)
        ch = curr_raw.get("player_charge", self.ep_last_player_charge)

        dmg_e = self.ep_last_enemy_health - hp_e
        dmg_p = self.ep_last_player_health - hp_p
        ch_gain = ch - self.ep_last_player_charge
        released = (
            self.ep_last_player_charge >= self.cfg.CHARGE_MAX_LEVEL
            and ch == 0
            and dmg_e > 0
        )

        if dmg_e > 0:
            comps["damage_dealt"] = dmg_e * self.cfg.REWARD_DAMAGE_DEALT_MULTIPLIER
        if dmg_p > 0:
            comps["damage_taken"] = dmg_p * self.cfg.REWARD_DAMAGE_TAKEN_MULTIPLIER
        if ch_gain > 0:
            comps["charge_gain"] = ch_gain * self.cfg.REWARD_CHARGE_GAIN_COEF
        if released:
            comps["charge_shot"] = self.cfg.REWARD_CHARGE_RELEASE_BONUS

        done = False
        if hp_p <= 0:
            comps["loss"] = self.cfg.REWARD_LOSE_GAME
            done = True
        elif hp_e <= 0:
            comps["win"] = self.cfg.REWARD_WIN_GAME
            done = True

        reward = sum(comps.values())

        noteworthy = abs(reward) >= 0.01 or released
        if noteworthy and self._should_log():
            parts = []
            if dmg_e > 0:
                parts.append(_c(f"-{dmg_e} HP enemy", 32))
            if dmg_p > 0:
                parts.append(_c(f"+{dmg_p} HP self", 31))
            if ch_gain > 0:
                parts.append(_c(f"+{ch_gain} charge", 36))
            if released:
                parts.append(_c("FULL-SHOT!", 35))
            if done and hp_e <= 0:
                parts.append(_c("WIN", 92))
            if done and hp_p <= 0:
                parts.append(_c("LOSS", 91))

            self._log(
                f"Port {self.port} ({self.name}): "
                f"{' | '.join(parts)}  →  r={reward:+.2f}"
            )

        self.ep_last_player_health = hp_p
        self.ep_last_enemy_health = hp_e
        self.ep_last_player_charge = ch
        return reward, done, comps

    # ---------------------------------------------------------------------
    # Action plan playback
    # ---------------------------------------------------------------------
    async def _action_playback_loop(self):
        if self._action_dt <= 0:
            return

        next_t = _time.perf_counter()
        try:
            while self._is_running:
                now = _time.perf_counter()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                    continue

                next_t += self._action_dt
                if (now - next_t) > (self._action_dt * 5):
                    next_t = now + self._action_dt

                async with self._action_lock:
                    if self._action_plan:
                        key_bin = str(self._action_plan.popleft())
                        self._action_last_key = key_bin
                    else:
                        key_bin = self._action_last_key   # <— repeats forever when plan runs out

                ok = await self._send_command_internal({"type": "key_press", "key": key_bin})
                if not ok:
                    break
        except asyncio.CancelledError:
            pass

    async def _set_plan_from_act_info(self, act_info: Dict[str, Any]) -> None:
        plan = act_info.get("action_plan_keys")
        if isinstance(plan, list) and len(plan) > 0:
            async with self._action_lock:
                self._action_plan.clear()
                self._action_plan.extend(str(k) for k in plan)
                self._action_last_key = str(plan[-1])
            return

        bc = act_info.get("button_command") or {}
        k = bc.get("key")
        if isinstance(k, str) and len(k) == 16 and all(ch in "01" for ch in k):
            async with self._action_lock:
                self._action_plan.clear()
                self._action_plan.append(k)
                self._action_last_key = k

    # ---------------------------------------------------------------------
    # Incoming message processing
    # ---------------------------------------------------------------------
    async def _message_processor_loop(self):
        json_buffer = ""
        try:
            while self._is_running:
                if not self.reader or self.reader.at_eof():
                    print(f"Port {self.port} ({self.name}): reader closed/EOF – leaving loop.")
                    break

                try:
                    data_chunk = await asyncio.wait_for(
                        self.reader.read(8192),
                        timeout=max(0.5, self.inference_interval * 20),
                    )
                except asyncio.TimeoutError:
                    continue

                if not data_chunk:
                    print(f"Port {self.port} ({self.name}): connection closed by peer.")
                    break

                json_buffer += data_chunk.decode(errors="ignore")

                while "\n" in json_buffer and self._is_running:
                    msg_str, json_buffer = json_buffer.split("\n", 1)
                    msg_str = msg_str.strip()
                    if not msg_str:
                        continue

                    try:
                        parsed = json.loads(msg_str)
                    except json.JSONDecodeError:
                        print(f"Port {self.port}: bad JSON → {msg_str[:120]}…")
                        continue

                    self._rx_lines += 1
                    event = parsed.get("event", "Unknown")
                    detail = parsed.get("details", {})

                    self._rx_events[event] = self._rx_events.get(event, 0) + 1

                    # Show the first few non-ack lines (avoid burning DBG_FIRST_LINES on key_press acks)
                    if self._rx_first_lines_left > 0 and self._should_log():
                        if event not in ("command_received",):
                            self._rx_first_lines_left -= 1
                            self._log(f"Port {self.port} ({self.name}): RX line #{self._rx_lines}: {msg_str[:220]}")

                    # Only process screen frames
                    if event not in ("screen_image", "screen", "screen_data"):
                        continue

                    # Parse details robustly
                    if isinstance(detail, dict):
                        current_raw = detail
                    elif isinstance(detail, str):
                        s = detail.strip()
                        if not s:
                            current_raw = {}
                        else:
                            try:
                                current_raw = json.loads(s)
                            except json.JSONDecodeError:
                                current_raw = {"image": s}
                    else:
                        current_raw = {"detail": str(detail)}

                    if not isinstance(current_raw, dict):
                        continue

                    self._update_instance_game_data_cache(current_raw)
                    comp_state = {**current_raw, **self.instance_game_data_cache}
                    comp_state["inside_window"] = bool(float(current_raw.get("inside_window", 0)))
                    if "image" not in comp_state and "screen_image" in comp_state:
                        comp_state["image"] = comp_state["screen_image"]

                    # Reward insert for previous action
                    if self.prev_raw_game_state_for_reward and self.prev_action_info_for_buffer:
                        r_t, done_t, comps_t = self._calculate_reward_and_done(comp_state)
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

                        if self.shared_episode_data is not None:
                            rb = self.shared_episode_data.setdefault("reward_batch", {})
                            rc = self.shared_episode_data.setdefault("reward_cumulative", {})
                            for k, v in comps_t.items():
                                rb[k] = rb.get(k, 0.0) + v
                                rc[k] = rc.get(k, 0.0) + v

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
                            self.ep_last_player_health = None
                            self.ep_last_enemy_health = None
                            self.ep_last_player_charge = None

                            if self.strategy:
                                self.strategy.reset_state(self.port)

                            self.prev_raw_game_state_for_reward = None
                            self.prev_action_info_for_buffer = None
                            self.prev_processed_stacked_frames = None
                            self.prev_processed_game_features = None

                    # Decide next action (unless manual override)
                    if not self.manual_override and self.strategy:
                        act_info = await asyncio.to_thread(self.strategy.decide_action, self.port, comp_state)
                        await self._set_plan_from_act_info(act_info)

                        seq_f = act_info.get("stacked_frames_tensor")
                        seq_g = act_info.get("game_features_tensor")

                        if act_info.get("value") is not None and seq_f is not None and seq_g is not None:
                            self.prev_processed_stacked_frames = seq_f
                            self.prev_processed_game_features = seq_g
                            self.prev_action_info_for_buffer = {
                                "action_idx": act_info["action_idx"],
                                "log_prob": act_info["log_prob"],
                                "value": act_info["value"],
                            }
                        else:
                            self.prev_action_info_for_buffer = None
                            self.prev_processed_stacked_frames = None
                            self.prev_processed_game_features = None

                            try:
                                if (seq_f is None or seq_g is None) and self.policy_eval is not None:
                                    seq_f, seq_g = self.policy_eval.encode_for_policy(self.port, comp_state)

                                if self.policy_eval is not None and seq_f is not None and seq_g is not None:
                                    key_str = (act_info.get("button_command") or {}).get("key", "0" * 16)
                                    mask_int = int(key_str, 2)
                                    a_idx_int = self.utils.bitmask_to_action_index(mask_int)

                                    with torch.no_grad():
                                        _, log_p_demo, _, v_demo, _ = self.policy_eval.model.get_action_and_value(
                                            seq_f.to(self.policy_eval.dev),
                                            seq_g.to(self.policy_eval.dev),
                                            torch.tensor(a_idx_int, device=self.policy_eval.dev),
                                        )

                                    self.prev_processed_stacked_frames = seq_f
                                    self.prev_processed_game_features = seq_g
                                    self.prev_action_info_for_buffer = {
                                        "action_idx": torch.tensor(a_idx_int, device=self.policy_eval.dev),
                                        "log_prob": log_p_demo.detach(),
                                        "value": v_demo.detach(),
                                    }
                            except Exception:
                                self.prev_action_info_for_buffer = None
                                self.prev_processed_stacked_frames = None
                                self.prev_processed_game_features = None
                    else:
                        self.prev_processed_stacked_frames = None
                        self.prev_processed_game_features = None
                        self.prev_action_info_for_buffer = None

                    self.prev_raw_game_state_for_reward = comp_state
                    self._inference_done_event.set()

        except (ConnectionResetError, BrokenPipeError):
            print(f"Port {self.port} ({self.name}): connection reset/broken.")
        except asyncio.CancelledError:
            print(f"Port {self.port} ({self.name}): processor task cancelled.")
        except Exception as e:
            print(f"Port {self.port}: unexpected error → {e}\n{traceback.format_exc()}")
        finally:
            self._is_running = False
            self._inference_done_event.set()
            print(f"Port {self.port} ({self.name}): processor loop ended.")

    # ---------------------------------------------------------------------
    # Connection lifecycle
    # ---------------------------------------------------------------------
    async def start_handling(
        self,
        max_retries: int = 0,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 5.0,
    ):
        self._is_running = True
        attempt, delay = 0, retry_base_delay

        while self._is_running:
            try:
                print(f"Attempting to connect to {self.name} at {self.address}:{self.port} ...")
                self.reader, self.writer = await asyncio.open_connection(self.address, self.port)

                sock = self.writer.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                print(f"Successfully connected to {self.name} (Port {self.port})")
                break
            except ConnectionRefusedError:
                print(f"Port {self.port} ({self.name}): connection refused.")
            except Exception as e:
                print(f"Port {self.port} ({self.name}): unexpected error while connecting: {e}")

            attempt += 1
            if max_retries and attempt >= max_retries:
                print(f"Port {self.port} ({self.name}): exceeded {max_retries} retries; abandoning handler.")
                self._is_running = False
                return

            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_max_delay)

        if not self._is_running:
            return

        self.ep_last_player_health = None
        self.ep_last_enemy_health = None
        self.ep_last_player_charge = None
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.prev_raw_game_state_for_reward = None
        self.prev_action_info_for_buffer = None
        self.prev_processed_stacked_frames = None
        self.prev_processed_game_features = None
        self._inference_done_event.set()

        if self.strategy:
            self.strategy.reset_state(self.port)

        self.message_processor_task = asyncio.create_task(self._message_processor_loop())
        self._action_task = asyncio.create_task(self._action_playback_loop())

        try:
            while self._is_running:
                await self._inference_done_event.wait()
                if not self._is_running:
                    break

                self._inference_done_event.clear()

                if not await self._request_screen_image():
                    break

                if self.inference_interval > 0:
                    await asyncio.sleep(self.inference_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Port {self.port} ({self.name}): unhandled error in request loop: {e}\n{traceback.format_exc()}")
        finally:
            self._is_running = False
            self._inference_done_event.set()

            if self.message_processor_task and not self.message_processor_task.done():
                self.message_processor_task.cancel()
                try:
                    await self.message_processor_task
                except asyncio.CancelledError:
                    pass

            if self._action_task and not self._action_task.done():
                self._action_task.cancel()
                try:
                    await self._action_task
                except asyncio.CancelledError:
                    pass

            for timer_key in ("player_chip_timer_task", "enemy_chip_timer_task"):
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

            print(f"Connection handler for {self.name} (Port {self.port}) has fully shut down.")
