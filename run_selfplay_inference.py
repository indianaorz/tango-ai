# ── Begin: run_selfplay_inference.py ──
#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import os
import signal
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

import config
import utils
from game_manager import GameManager
from network_handler import ConnectionHandler
from strategy import (
    ScriptedStandAndShootStrategy,
    ScriptedWanderStrategy,
    ScriptedChargeAndRelease,
)
from strategy_ng import NGAgentStrategy

from selfplay_debug_ui import DebugState, start_debug_ui

import logging

# Silence Werkzeug request logs (the "127.0.0.1 - - ..." lines)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


# ----------------------------
# Debug helpers
# ----------------------------

def _decode_pil_from_b64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


class DebugStrategyWrapper:
    """
    Wrap any strategy and record:
      - image it received (game_state["image"])
      - key press decision mask + derived pressed buttons (via DebugState)
    Also prints a rate-limited console line per port.
    """

    def __init__(
        self,
        inner: Any,
        debug_state: DebugState,
        *,
        print_hz: float = 4.0,
    ):
        self._inner = inner
        self._dbg = debug_state
        self._min_print_dt = (1.0 / print_hz) if print_hz > 0 else 0.0
        self._last_print_ts_by_port: Dict[int, float] = {}

    def reset_state(self, port: int):
        if hasattr(self._inner, "reset_state"):
            self._inner.reset_state(port)

    def decide_action(self, port: int, game_state: dict) -> dict:
        decision = self._inner.decide_action(port, game_state)

        # Decode and push to DebugState immediately. 
        # DebugState now uses Condition variables to wake up the UI stream instantly.
        pil = _decode_pil_from_b64(game_state.get("image"))
        self._dbg.update(port=port, game_state=game_state, decision=decision, pil_img=pil)

        now = time.time()
        last = self._last_print_ts_by_port.get(port, 0.0)
        if self._min_print_dt <= 0 or (now - last) >= self._min_print_dt:
            self._last_print_ts_by_port[port] = now
            payload = self._dbg.to_json()
            snap = (payload.get("ports") or {}).get(str(port), {})

            inside = bool(snap.get("inside_window", False))

            # "mapped" = what we actually send to the game
            mapped_key_int = int(snap.get("mapped_key_int") or 0)
            mapped_pressed = snap.get("mapped_pressed_buttons") or []
            mapped_key_bin = snap.get("mapped_key_bin") or ""

            # "ng" = raw model intent (only present if your strategy provides ng_key_bin)
            ng_key_int = int(snap.get("ng_key_int") or 0)
            ng_pressed = snap.get("ng_pressed_buttons") or []
            ng_key_bin = snap.get("ng_key_bin") or ""

            action_type = snap.get("action_type") or "—"
            #only print port 12350
            # if port == 12350:
            #     print(
            #         f"[port {port}] inside_window={inside} action={action_type} "
            #         f"mapped_int={mapped_key_int} mapped={mapped_pressed} mapped_bin={mapped_key_bin} "
            #         f"ng_int={ng_key_int} ng={ng_pressed} ng_bin={ng_key_bin}"
            #     )


        return decision


# ----------------------------
# Strategy selection
# ----------------------------

def _make_util_funcs() -> dict:
    return {
        "int_to_binary_string": utils.int_to_binary_string,
        "map_discrete_action_to_buttons": utils.map_discrete_action_to_buttons,
        "preprocess_frame": utils.preprocess_frame,
        "generate_random_action_for_skip_strategy": utils.generate_random_action_for_skip_strategy,
    }


def _make_strategy(cfg: dict, util_funcs: dict):
    """
    Strategy selection is driven by config.INSTANCES[i]["strategy"].

    Learner uses NG inference by default (config.USE_NG_POLICY=1).
    Opponent uses scripted based on config.generate_instances().
    """
    s = (cfg.get("strategy") or "").strip().lower()

    if s == config.STRAT_DRL:
        if config.USE_NG_POLICY:
            return NGAgentStrategy(
                ckpt_path=config.NG_CKPT_PATH,
                device=torch.device(config.NG_DEVICE),
                key_bit_positions=config.KEY_BIT_POSITIONS,
                discrete_actions=config.DISCRETE_ACTIONS,
                util_fns=util_funcs,
                frame_h=config.FRAME_HEIGHT,
                frame_w=config.FRAME_WIDTH,
                seq_len_frames=config.SEQ_LEN_FRAMES,
                use_images=True,  # NG is vision-based
                allow_actions_in_window=True,
                forbid_actions_in_battle=[]#["START"],
            )

        raise RuntimeError(
            "USE_NG_POLICY=0 but run_selfplay_inference.py is configured for inference-only.\n"
            "Either set USE_NG_POLICY=1, or extend this runner to load ActorCritic weights and use DRLAgentStrategy."
        )

    if s == config.STRAT_STAND:
        return ScriptedStandAndShootStrategy(
            key_bit_positions=config.KEY_BIT_POSITIONS,
            discrete_actions=config.DISCRETE_ACTIONS,
            util_fns=util_funcs,
            fire_every_ms=500,
            seed=123,
        )

    if s == config.STRAT_WANDER:
        return ScriptedWanderStrategy(
            key_bit_positions=config.KEY_BIT_POSITIONS,
            discrete_actions=config.DISCRETE_ACTIONS,
            util_fns=util_funcs,
            hold_ms=350,
            noop_chance=0.15,
            seed=1337,
        )

    if s == config.STRAT_CHARGE:
        return ScriptedChargeAndRelease(
            key_bit_positions=config.KEY_BIT_POSITIONS,
            discrete_actions=config.DISCRETE_ACTIONS,
            util_fns=util_funcs,
            move_mode="strafe",
            charge_level=config.CHARGE_MAX_LEVEL,
            align_release=True,
            max_charge_ms=3500,
            strafe_ms=500,
            seed=7,
        )

    # Default fallback (safe)
    return ScriptedWanderStrategy(
        key_bit_positions=config.KEY_BIT_POSITIONS,
        discrete_actions=config.DISCRETE_ACTIONS,
        util_fns=util_funcs,
        seed=1337,
    )


# ----------------------------
# Main runner
# ----------------------------

async def _run() -> None:
    print(config.summary())
    util_funcs = _make_util_funcs()

    # Debug UI settings (override via env)
    debug_enabled = os.environ.get("SELFPLAY_DEBUG_UI", "1").strip() not in ("0", "false", "False")
    debug_host = os.environ.get("SELFPLAY_DEBUG_HOST", "0.0.0.0").strip()
    debug_port = int(os.environ.get("SELFPLAY_DEBUG_PORT", "5010").strip())
    debug_print_hz = float(os.environ.get("SELFPLAY_DEBUG_PRINT_HZ", "4").strip() or "4")

    debug_state = DebugState(key_bit_positions=config.KEY_BIT_POSITIONS)
    if debug_enabled:
        start_debug_ui(debug_state, host=debug_host, port=debug_port)
        print(f"Debug UI: http://{debug_host}:{debug_port}  (host=0.0.0.0 means use your LAN IP in browser)")

    # 1) Spawn windows from the plan (2 windows per pair)
    gm = GameManager(
        app_path=config.APP_PATH,
        env_common=config.ENV_COMMON,
        instance_stagger_time=config.INSTANCE_STAGGER_TIME,
        base_port=config.BASE_PORT,
    )

    plan = list(config.INSTANCES)
    if len(plan) < 2:
        raise RuntimeError("config.INSTANCES must contain at least 2 entries (learner + opponent).")

    gm.start_instances_from_plan(plan)

    # 2) Create per-window handlers
    handlers: List[ConnectionHandler] = []
    for inst_cfg in plan:
        strat = _make_strategy(inst_cfg, util_funcs)

        # Wrap strategy with debug capture + console output
        strat = DebugStrategyWrapper(
            inner=strat,
            debug_state=debug_state,
            print_hz=debug_print_hz,
        )

        h = ConnectionHandler(
            instance_config=inst_cfg,
            active_strategy=strat,
            inference_fps=config.INFERENCE_FPS,
            experience_buffer=None,           # inference-only
            shared_episode_data=None,         # inference-only
            config_module=config,
            utils_module=utils,
            policy_eval_strategy=None,
        )
        handlers.append(h)

    # 3) Graceful shutdown on Ctrl+C
    stop_event = asyncio.Event()

    def _request_stop(*_args):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_a: _request_stop())

    # 4) Run all handlers concurrently
    tasks = [asyncio.create_task(h.start_handling(max_retries=0)) for h in handlers]

    print("Self-play inference running. Ctrl+C to stop.")
    await stop_event.wait()

    # 5) Stop handlers
    for h in handlers:
        h._is_running = False

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # 6) Close windows
    gm.terminate_all_instances()
    print("Shutdown complete.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
# ── End: run_selfplay_inference.py ──