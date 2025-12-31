# ── Begin: run_selfplay_inference.py ──
#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import logging
import os
import signal
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

import config
import utils
from game_manager import GameManager
from network_handler import ConnectionHandler
from selfplay_debug_ui import DebugState, start_debug_ui
from strategy import (
    ScriptedChargeAndRelease,
    ScriptedStandAndShootStrategy,
    ScriptedWanderStrategy,
)
from strategy_critic_minimal import MinimalCriticBattleStrategy
from strategy_ng import NGAgentStrategy

from planning.planning_model import PlanningAgentStrategy

# Silence Werkzeug request logs (the "127.0.0.1 - - ..." lines)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Debug/trace toggles (env)
# -----------------------------------------------------------------------------
_TRACE = os.environ.get("SELFPLAY_TRACE", "0").strip().lower() not in ("0", "false", "no", "")
_TRACE_HZ = float(os.environ.get("SELFPLAY_TRACE_HZ", "2").strip() or "2")
_USE_DUAL_POLICY = os.environ.get("USE_DUAL_POLICY", "1").strip().lower() not in ("0", "false", "no", "")


# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------
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

def _handler_cfg(h: Any) -> dict:
    cfg = getattr(h, "instance_config", None)
    if isinstance(cfg, dict):
        return cfg
    cfg = getattr(h, "instance_config_data", None)
    if isinstance(cfg, dict):
        return cfg
    return {}

# -----------------------------------------------------------------------------
# Strategy wrappers
# -----------------------------------------------------------------------------
class DualModelStrategy:
    """
    Holds two specialized strategies (Battle and Planning).
    Switches between them based on game state.
    """
    def __init__(self, battle_strat: Any, plan_strat: Any):
        self.battle = battle_strat
        self.plan = plan_strat

        # This dual policy does NOT require vision for gating (battle is features-only).
        # The planner may require image or not depending on your impl, but we treat dual
        # as "doesn't require image" so SafeHybridStrategy won't block it.
        self.requires_image = False

        self.current_mode = "battle"

    def reset_state(self, port: int):
        if hasattr(self.battle, "reset_state"):
            self.battle.reset_state(port)
        if hasattr(self.plan, "reset_state"):
            self.plan.reset_state(port)
        self.current_mode = "battle"

    def decide_action(self, port: int, game_state: dict) -> dict:
        inside_window = bool(float(game_state.get("inside_window", 0)))
        cust_gauge = int(float(game_state.get("cust_gauge", 0)))

        # Plan mode only when chip window is open AND gauge is 0 (start-of-turn plan).
        if inside_window and cust_gauge == 0:
            mode = "plan"
            active = self.plan
        else:
            mode = "battle"
            active = self.battle
            # Ensure planner is reset while fighting so it doesn't get stuck.
            if hasattr(self.plan, "reset_state"):
                self.plan.reset_state(port)

        self.current_mode = mode
        decision = active.decide_action(port, game_state)

        if "debug" not in decision:
            decision["debug"] = {}
        decision["debug"]["active_model"] = mode

        return decision


class SafeHybridStrategy:
    """
    Wraps a primary strategy and a fallback bootstrap strategy.

    If primary.requires_image is True, and no image is present, we use fallback.
    If primary.requires_image is False, we never gate on missing image.
    """
    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback

    def reset_state(self, port: int):
        if hasattr(self.primary, "reset_state"):
            self.primary.reset_state(port)
        if hasattr(self.fallback, "reset_state"):
            self.fallback.reset_state(port)

    def decide_action(self, port: int, game_state: dict) -> dict:
        requires_image = bool(getattr(self.primary, "requires_image", True))
        image_data = game_state.get("image")

        if requires_image and not image_data:
            return self.fallback.decide_action(port, game_state)

        return self.primary.decide_action(port, game_state)

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


class StartSpamStrategy:
    """
    Minimal bootstrap strategy:
      - presses START every call (or every N calls) so menus advance
      - never requires images
    """
    def __init__(self, *, key_bit_positions: Dict[str, int], every_n: int = 1):
        self.key_bits = dict(key_bit_positions or {})
        self.every_n = max(1, int(every_n))
        self._ctr = 0
        self.requires_image = False

    def reset_state(self, port: int):
        self._ctr = 0

    def decide_action(self, port: int, game_state: dict) -> dict:
        self._ctr += 1
        press = (self._ctr % self.every_n) == 0

        bit = self.key_bits.get("RETURN", None)
        mask = 0
        if press and bit is not None:
            mask |= (1 << int(bit))

        key_bin16 = format(int(mask) & 0xFFFF, "016b")
        return {
            "button_command": {"type": "key_press", "key": key_bin16},
            "ng_key_bin": "",
            "debug": {"bootstrap": True, "press_start": bool(press), "active_model": "bootstrap"},
        }


class SwappableStrategy:
    """
    Thread-safe wrapper that delegates to an inner strategy that can be swapped at runtime.
    """
    def __init__(self, initial: Any):
        self._lock = threading.Lock()
        self._inner = initial
        self.requires_image = bool(getattr(initial, "requires_image", False))

    def swap(self, new_inner: Any) -> None:
        with self._lock:
            self._inner = new_inner
            self.requires_image = bool(getattr(new_inner, "requires_image", True))

    def reset_state(self, port: int):
        with self._lock:
            inner = self._inner
        if hasattr(inner, "reset_state"):
            inner.reset_state(port)

    def decide_action(self, port: int, game_state: dict) -> dict:
        with self._lock:
            inner = self._inner
        return inner.decide_action(port, game_state)


class DebugStrategyWrapper:
    """
    Wrap any strategy and record:
      - image it received (game_state["image"])
      - key press decision mask + derived pressed buttons (via DebugState)
    Also optionally prints trace lines.
    """
    def __init__(self, inner: Any, debug_state: DebugState, *, print_hz: float = 4.0):
        self._inner = inner
        self._dbg = debug_state
        self._min_print_dt = (1.0 / print_hz) if print_hz > 0 else 0.0
        self._last_print_ts_by_port: Dict[int, float] = {}

    def reset_state(self, port: int):
        if hasattr(self._inner, "reset_state"):
            self._inner.reset_state(port)

    def decide_action(self, port: int, game_state: dict) -> dict:
        decision = self._inner.decide_action(port, game_state)

        pil = _decode_pil_from_b64(game_state.get("image"))
        self._dbg.update(port=port, game_state=game_state, decision=decision, pil_img=pil)

        if _TRACE:
            now = time.time()
            last = self._last_print_ts_by_port.get(port, 0.0)
            if self._min_print_dt <= 0 or (now - last) >= self._min_print_dt:
                self._last_print_ts_by_port[port] = now
                dbg = decision.get("debug") or {}
                mode = dbg.get("active_model", "?")
                note = dbg.get("note", "")
                cust = int(float(game_state.get("cust_gauge", 0) or 0))
                inside = bool(float(game_state.get("inside_window", 0) or 0))
                img_ok = bool(game_state.get("image"))
                # show whether this looks like real inference or holding
                top0 = None
                try:
                    top = dbg.get("top") or []
                    if isinstance(top, list) and top:
                        top0 = top[0]
                except Exception:
                    top0 = None

                print(
                    f"[trace][port {int(port)}] mode={mode} inside_window={inside} cust={cust} image={img_ok} "
                    f"note={note!r} top0={top0}"
                )

        return decision


# -----------------------------------------------------------------------------
# Strategy selection
# -----------------------------------------------------------------------------
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
    """
    s = (cfg.get("strategy") or "").strip().lower()

    if s == config.STRAT_DRL:
        # NG policy (vision) — only if explicitly enabled.
        if bool(getattr(config, "USE_NG_POLICY", False)):
            return NGAgentStrategy(
                ckpt_path=config.NG_CKPT_PATH,
                device=torch.device(config.NG_DEVICE),
                key_bit_positions=config.KEY_BIT_POSITIONS,
                discrete_actions=config.DISCRETE_ACTIONS,
                util_fns=util_funcs,
                frame_h=config.FRAME_HEIGHT,
                frame_w=config.FRAME_WIDTH,
                seq_len_frames=config.SEQ_LEN_FRAMES,
                use_images=True,
                allow_actions_in_window=True,
                forbid_actions_in_battle=[],
            )

        # If DRL requested but NG disabled, fall back safely.
        return ScriptedWanderStrategy(
            key_bit_positions=config.KEY_BIT_POSITIONS,
            discrete_actions=config.DISCRETE_ACTIONS,
            util_fns=util_funcs,
            seed=1337,
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


def _should_use_dual_policy(inst_cfg: dict) -> bool:
    """
    Use dual only for the Learner instance(s) with DRL strategy.
    This intentionally does NOT depend on config.USE_NG_POLICY, because dual != NG.
    """
    return True
    if not _USE_DUAL_POLICY:
        return False
    name = str(inst_cfg.get("name", ""))
    is_learner = ("Learner" in name)
    s = (inst_cfg.get("strategy") or "").strip().lower()
    is_drl = (s == config.STRAT_DRL)
    return bool(is_learner and is_drl)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
async def _run() -> None:
    print(config.summary())
    util_funcs = _make_util_funcs()

    # Debug UI settings (override via env)
    debug_enabled = os.environ.get("SELFPLAY_DEBUG_UI", "1").strip() not in ("0", "false", "False")
    debug_host = os.environ.get("SELFPLAY_DEBUG_HOST", "0.0.0.0").strip()
    debug_port = int(os.environ.get("SELFPLAY_DEBUG_PORT", "5010").strip())
    debug_print_hz = float(os.environ.get("SELFPLAY_DEBUG_PRINT_HZ", str(_TRACE_HZ)).strip() or str(_TRACE_HZ))

    debug_state = DebugState(key_bit_positions=config.KEY_BIT_POSITIONS)
    if debug_enabled:
        start_debug_ui(debug_state, host=debug_host, port=debug_port)
        print(f"Debug UI: http://{debug_host}:{debug_port}  (host=0.0.0.0 means use your LAN IP in browser)")

    # 1) Spawn windows from the plan
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

    # 2) Build per-window handlers (ALL of them)
    handlers: List[ConnectionHandler] = []
    ng_loader_threads: List[threading.Thread] = []

    # Print plan summary up front (helps catch “wrong instance got NG” immediately)
    print("Instance plan:")
    for inst in plan:
        print(
            f" • {inst.get('name')} port={inst.get('port')} strat={inst.get('strategy')} "
            f"dual={_should_use_dual_policy(inst)}"
        )

    for inst_cfg in plan:
        # Choose base strategy object
        if _should_use_dual_policy(inst_cfg):
            # Start with bootstrap; swap in dual after heavy models load.
            bootstrap = StartSpamStrategy(key_bit_positions=config.KEY_BIT_POSITIONS, every_n=1)
            sw = SwappableStrategy(initial=bootstrap)
            strat_obj: Any = sw

            def _load_dual_models(swappable: SwappableStrategy, fallback_strat: Any):
                try:
                    print(f"[Dual] Loading BATTLE model (CriticMinimalPolicy) from {config.BATTLE_CKPT_PATH}...")
                    battle_strat = MinimalCriticBattleStrategy(
                        ckpt_path=config.BATTLE_CKPT_PATH,
                        device=torch.device(config.NG_DEVICE),
                        key_bit_positions=config.KEY_BIT_POSITIONS,
                        hold=4,
                        seq_len=192,
                        start_stride=1,
                        require_cust_gt0=True,
                        use_amp=True,
                    )

                    print(f"[Dual] Loading PLANNING model from {config.PLAN_CKPT_PATH}...")
                    plan_strat = PlanningAgentStrategy(
                        model_path=config.PLAN_CKPT_PATH,
                        chips_db_path=config.CHIPS_DB_PATH,
                        device=torch.device(config.NG_DEVICE),
                        key_bit_positions=config.KEY_BIT_POSITIONS,
                    )

                    dual = DualModelStrategy(battle_strat, plan_strat)
                    safe = SafeHybridStrategy(primary=dual, fallback=fallback_strat)

                    swappable.swap(safe)
                    print("[Dual] Ready. Swapped in DUAL (Battle/Plan) strategy.")
                except Exception as e:
                    print(f"[Dual] Failed to load models: {e!r}")
                    import traceback
                    traceback.print_exc()
                    # Keep bootstrap running; do NOT crash process.

            t = threading.Thread(target=_load_dual_models, args=(sw, bootstrap), daemon=True)
            t.start()
            ng_loader_threads.append(t)

        else:
            # Opponent (and any non-dual instance) uses normal selection.
            # If config.USE_NG_POLICY=0, DRL instances become scripted wander instead of crashing.
            try:
                strat_obj = _make_strategy(inst_cfg, util_funcs)
            except Exception as e:
                print(f"[warn] Failed to build strategy for {inst_cfg.get('name')} (port={inst_cfg.get('port')}): {e!r}")
                strat_obj = ScriptedWanderStrategy(
                    key_bit_positions=config.KEY_BIT_POSITIONS,
                    discrete_actions=config.DISCRETE_ACTIONS,
                    util_fns=util_funcs,
                    seed=1337,
                )

        # Wrap for UI + trace
        strat_obj = DebugStrategyWrapper(
            inner=strat_obj,
            debug_state=debug_state,
            print_hz=debug_print_hz,
        )

        handlers.append(
            ConnectionHandler(
                instance_config=inst_cfg,
                active_strategy=strat_obj,
                inference_fps=config.INFERENCE_FPS,
                experience_buffer=None,
                shared_episode_data=None,
                config_module=config,
                utils_module=utils,
                policy_eval_strategy=None,
            )
        )

    # 3) Graceful shutdown on Ctrl+C (ONCE)
    stop_event = asyncio.Event()

    def _request_stop(*_args):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_a: _request_stop())

    # 4) Run ALL handlers concurrently (ONCE)
    tasks = [asyncio.create_task(h.start_handling(max_retries=0)) for h in handlers]
    ports = [int(_handler_cfg(h).get("port", -1)) for h in handlers]
    print(f"Self-play inference running for ports={ports}. Ctrl+C to stop.")


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
