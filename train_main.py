# train_main.py
import asyncio
import os
import random
import re
import time
import traceback
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from experience_buffer import ExperienceBuffer
from game_manager import GameManager
from models import ActorCriticCNNRNN
from network_handler import ConnectionHandler
from ppo_trainer import PPOTrainer
from strategy import (
    DRLAgentStrategy,
    ScriptedStandAndShootStrategy,
    ScriptedWanderStrategy,
    ScriptedChargeAndRelease
)

# ------------------------------------------------------------
# Utils already in your file; kept as-is
# ------------------------------------------------------------
def find_latest_model_and_steps(model_dir, model_prefix="mmbn_ppo_model_"):
    if not os.path.isdir(model_dir):
        return None, 0
    latest_model_path = None
    max_steps = -1
    pattern = re.compile(rf"{re.escape(model_prefix)}(\d+)\.pth")
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_model_path = os.path.join(model_dir, filename)
    if latest_model_path:
        return latest_model_path, max_steps
    return None, 0


class _RepeatingTimer:
    def __init__(self):
        self._t = time.time()

    def every(self, seconds: float) -> bool:
        now = time.time()
        if now - self._t >= seconds:
            self._t = now
            return True
        return False


# --- NEW: helper to train in background on a snapshot buffer -----------
async def _ppo_update_async(snapshot_buf, handlers, model, trainer, writer, shared, cfg):
    # Get a bootstrap value without blocking the event loop (short, but do it here anyway)
    def _bootstrap_value():
        vals = []
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                for hnd in handlers:
                    sf = getattr(hnd, "prev_processed_stacked_frames", None)
                    gf = getattr(hnd, "prev_processed_game_features", None)
                    if sf is None or gf is None:
                        continue
                    _, _, _, v, _ = model.get_action_and_value(sf.to(cfg.DEVICE), gf.to(cfg.DEVICE))
                    vals.append(v.squeeze())
            return (torch.stack(vals).mean().to(cfg.DEVICE) if vals
                    else torch.tensor(0.0, device=cfg.DEVICE))
        finally:
            model.train(mode=was_training)

    # Synchronous trainer work (runs in a worker thread)
    def _train_sync():
        model.train()
        last_val = _bootstrap_value()
        tot_pol = tot_val = tot_ent = 0.0
        n_batches = 0
        for batch in snapshot_buf.get_batches(last_val):
            s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs = batch
            p_loss, v_loss, ent = trainer.train_step(
                s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs
            )
            tot_pol += p_loss; tot_val += v_loss; tot_ent += ent
            n_batches += 1
        if n_batches == 0:
            return 0.0, 0.0, 0.0
        return tot_pol/n_batches, tot_val/n_batches, tot_ent/n_batches

    # Actually run training in a background thread so the loop keeps ticking
    avg_pol, avg_val, avg_ent = await asyncio.to_thread(_train_sync)
    model.eval()

    cur_steps = shared["total_steps_trained"]
    writer.add_scalar("Losses/PolicyLoss", avg_pol, cur_steps)
    writer.add_scalar("Losses/ValueLoss",  avg_val, cur_steps)
    writer.add_scalar("Charts/Entropy",     avg_ent, cur_steps)
    writer.flush()


# =================================================================
#  ░ Complete training loop ░
# =================================================================
async def main_drl_training_loop() -> None:
    print(f"Starting DRL Training Script using device: {config.DEVICE}")
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    writer = SummaryWriter(
        log_dir=os.path.join(
            config.TENSORBOARD_LOG_DIR, f"mmbn_ppo_{int(time.time())}"
        )
    )

    # -----------------------------------------------------------------
    # constants (short names so the code reads cleanly)
    # -----------------------------------------------------------------
    WINDOWS_TARGET = config.NUM_GAME_PAIRS * 2
    ROM_PATH = config.ROM_PATH_DEFAULT
    SAVE_TMPL = config.SAVE_PATH_TEMPLATE
    INIT_CODE_BASE = config.INIT_CODE_DEFAULT
    ADDRESS = config.ADDRESS_DEFAULT
    NUM_GAME_FEATURES_FOR_MODEL = 11  # <- keep in sync with model
    # 🔧 Derive channels from config.USE_IMAGES
    chan = (config.FRAME_CHANNELS if config.USE_IMAGES else 0)

    # -----------------------------------------------------------------
    # build Actor-Critic
    # -----------------------------------------------------------------
    actor_critic_model = ActorCriticCNNRNN(
        seq_len_frames=config.SEQ_LEN_FRAMES,
        num_game_features=NUM_GAME_FEATURES_FOR_MODEL,
        num_actions=len(config.DISCRETE_ACTIONS),
        frame_height=config.FRAME_HEIGHT,
        frame_width=config.FRAME_WIDTH,
        frame_channels=chan, 
        cnn_channels=config.CNN_CHANNELS,
        cnn_kernels=config.CNN_KERNELS,
        cnn_strides=config.CNN_STRIDES,
        fuse_dim=config.FUSE_DIM,
        rnn_type=config.RNN_TYPE,
        rnn_hidden=config.RNN_HIDDEN,
        rnn_layers=config.RNN_LAYERS,
        rnn_dropout=config.DROPOUT_RNN,
    ).to(config.DEVICE)
    print("ActorCritic model instantiated.")

    # resume checkpoint ------------------------------------------------
    initial_total_steps_trained = 0
    ckpt_file, ckpt_steps = find_latest_model_and_steps(config.MODEL_SAVE_DIR)
    if ckpt_file:
        try:
            actor_critic_model.load_state_dict(
                torch.load(ckpt_file, map_location=config.DEVICE)
            )
            initial_total_steps_trained = ckpt_steps
            print(f"Loaded {ckpt_file} ({ckpt_steps} steps).")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")

    # -----------------------------------------------------------------
    # helpers (buffer, trainer, strategies)
    # -----------------------------------------------------------------
    experience_buffer = ExperienceBuffer(
        buffer_size=config.EXPERIENCE_BUFFER_SIZE,
        mini_batch_size=config.MINI_BATCH_SIZE,
        num_game_features=NUM_GAME_FEATURES_FOR_MODEL,
        frame_shape=(                                   # 🔧 keep buffer shape in sync
            config.SEQ_LEN_FRAMES,
            chan,                                       # <-- was config.FRAME_CHANNELS
            config.FRAME_HEIGHT,
            config.FRAME_WIDTH,
        ),
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        device=config.DEVICE,
    )

    ppo_trainer = PPOTrainer(
        actor_critic_model=actor_critic_model,
        learning_rate=config.LEARNING_RATE,
        ppo_clip_epsilon=config.PPO_CLIP_EPSILON,
        ppo_epochs=config.PPO_EPOCHS,
        value_loss_coef=config.VALUE_LOSS_COEF,
        entropy_coef=config.ENTROPY_COEF,
        device=config.DEVICE,
    )

    util_funcs = {
        "int_to_binary_string": utils.int_to_binary_string,
        "map_discrete_action_to_buttons": utils.map_discrete_action_to_buttons,
        "preprocess_frame": utils.preprocess_frame,
    }

    # We keep a single shared learner strategy (it holds the model ref & buffers)
    drl_strategy_shared = DRLAgentStrategy(
        config.KEY_BIT_POSITIONS,
        config.DISCRETE_ACTIONS,
        util_funcs,
        actor_critic_model,
        config.DEVICE,
        config.FRAME_HEIGHT,
        config.FRAME_WIDTH,
        config.SEQ_LEN_FRAMES,
        max_health_config=config.MAX_HEALTH,
        max_charge_config=config.MAX_CHARGE_LEVEL,
        max_cust_gauge_config=config.MAX_CUST_GAUGE_VALUE,
        use_images=config.USE_IMAGES,
    )

    shared = {
        "episode_rewards": deque(maxlen=100),
        "episode_lengths": deque(maxlen=100),
        "steps_collected_since_last_train": 0,
        "total_steps_trained": initial_total_steps_trained,
        "reward_batch": {
            k: 0.0
            for k in (
                "damage_dealt",
                "damage_taken",
                "charge_gain",
                "charge_shot",
                "time_penalty",
                "win",
                "loss",
            )
        },
        "reward_cumulative": {
            k: 0.0
            for k in (
                "damage_dealt",
                "damage_taken",
                "charge_gain",
                "charge_shot",
                "time_penalty",
                "win",
                "loss",
            )
        },
    }

    # -----------------------------------------------------------------
    # Strategy plan (from config)
    # Build a name->strategy_name and port->strategy_name map
    # -----------------------------------------------------------------
    _plan_by_name = {}
    _plan_by_port = {}
    for cfg in getattr(config, "INSTANCES", []):
        sname = cfg.get("strategy", "").strip().lower()
        nm = cfg.get("name")
        prt = cfg.get("port")
        if nm:
            _plan_by_name[nm] = sname
        if prt is not None:
            _plan_by_port[int(prt)] = sname

    def _parse_pair_index_from_name(name: str) -> int | None:
        try:
            head = name.split()[0]  # "Game1"
            if head.startswith("Game"):
                return int(head[4:])
        except Exception:
            pass
        return None

    # Factory that instantiates a concrete strategy for a given cfg
    def _make_strategy_from_plan(cfg_dict):
        name = cfg_dict.get("name", "")
        port = int(cfg_dict.get("port", -1))

        # Prefer explicit config match (port, then name)
        s = _plan_by_port.get(port) or _plan_by_name.get(name) or ""

        # Fallback heuristics if not specified (e.g., respawns beyond plan):
        if not s:
            if "Learner" in name:
                s = "drl"
            elif "Opponent" in name:
                pair_idx = _parse_pair_index_from_name(name) or 0
                if pair_idx == 1:
                    s = "scripted_charge"
                elif pair_idx == 2:
                    s = "scripted_charge"
                elif pair_idx == 3:
                    s = "scripted_charge"
                else:
                    s = "drl"  # default to self-play if plan missing

        s = s.lower()
        if s in ("drl", "learner", "policy"):
            return drl_strategy_shared

        if s in ("scripted_stand", "stand", "stand_and_shoot"):
            return ScriptedStandAndShootStrategy(
                config.KEY_BIT_POSITIONS,
                config.DISCRETE_ACTIONS,
                util_funcs,
                fire_every_ms=500,
                seed=123,
            )

        if s in ("scripted_wander", "wander", "random_walk"):
            return ScriptedWanderStrategy(
                config.KEY_BIT_POSITIONS,
                config.DISCRETE_ACTIONS,
                util_funcs,
                hold_ms=350,
                noop_chance=0.15,
                seed=1337,
            )
        if s in ("scripted_charge", "charge_and_release", "charge"):
            return ScriptedChargeAndRelease(
                config.KEY_BIT_POSITIONS,
                config.DISCRETE_ACTIONS,
                util_funcs,
                move_mode="stand",      # "stand" == "still"
                charge_level=3,
                align_release=True,
                max_charge_ms=3500,     # optional
                strafe_ms=500,          # ignored in "stand" mode
                )

        # Safe default
        return drl_strategy_shared

    # -----------------------------------------------------------------
    # Game-process orchestration
    # -----------------------------------------------------------------
    game_mgr = GameManager(
        config.APP_PATH, config.ENV_COMMON, config.INSTANCE_STAGGER_TIME, base_port=config.BASE_PORT
    )

    # Launch initial windows based on NUM_GAME_PAIRS; window names/ports match plan
    # instance_cfgs = game_mgr.start_initial_pairs(
    #     num_pairs=config.NUM_GAME_PAIRS,
    #     rom_path=ROM_PATH,
    #     save_path_template=SAVE_TMPL,
    #     init_code_base=INIT_CODE_BASE,
    #     address=ADDRESS,
    # )

    instance_cfgs = game_mgr.start_instances_from_plan(config.INSTANCES)

    # Build first batch of handlers
    handlers: list[ConnectionHandler] = []
    port_to_handler: dict[int, ConnectionHandler] = {}
    pid_to_port: dict[int, int] = {}

    def _make_strategy_from_cfg(cfg_dict):
        s = (cfg_dict.get("strategy") or "").lower()
        if s in ("drl", "learner", "policy"):
            return drl_strategy_shared
        if s in ("scripted_stand", "stand", "stand_and_shoot"):
            return ScriptedStandAndShootStrategy(
                config.KEY_BIT_POSITIONS, config.DISCRETE_ACTIONS, util_funcs,
                fire_every_ms=500, seed=123
            )
        if s in ("scripted_wander", "wander", "random_walk"):
            return ScriptedWanderStrategy(
                config.KEY_BIT_POSITIONS, config.DISCRETE_ACTIONS, util_funcs,
                hold_ms=350, noop_chance=0.15, seed=1337
            )
        
        if s in ("scripted_charge", "charge_and_release", "charge"):
            return ScriptedChargeAndRelease(
                config.KEY_BIT_POSITIONS,
                config.DISCRETE_ACTIONS,
                util_funcs,
                move_mode="stand",      # "stand" == "still"
                charge_level=3,
                align_release=True,
                max_charge_ms=3500,     # optional
                strafe_ms=500,          # ignored in "stand" mode
                )
        # safe default
        return drl_strategy_shared

    def _attach_handler(cfg):
        strat = _make_strategy_from_cfg(cfg)
        h = ConnectionHandler(
            cfg, strat, config.INFERENCE_FPS,
            experience_buffer, shared, config, utils,
            policy_eval_strategy=drl_strategy_shared, 
        )
        h.collect_experience = (cfg["strategy"].lower() == "drl") and ("Learner" in cfg.get("name",""))
        if isinstance(strat, DRLAgentStrategy):
            drl_strategy_shared.set_cross_select(h.port, 0)
            # drl_strategy_shared.set_cross_select(h.port, random.randint(1, 5))

        handlers.append(h)
        port_to_handler[h.port] = h
        # map PID→port once GameManager has registered the Popen obj
        for pinfo in game_mgr.processes:
            if pinfo["port"] == h.port:
                pid_to_port[pinfo["process"].pid] = h.port
                break
        return h

    for cfg in instance_cfgs:
        _attach_handler(cfg)

    # start manual-input router
    from manual_input_router import ManualInputRouter
    manual_router = ManualInputRouter(
        port_to_handler, pid_to_port,
        config.KEY_BIT_POSITIONS, utils.int_to_binary_string,
    )
    # manual_router.start()

    # schedule handler tasks ------------------------------------------
    handler_tasks = [asyncio.create_task(h.start_handling()) for h in handlers]

    # maintenance timers 
    maint_timer   = _RepeatingTimer()
    log_timer     = _RepeatingTimer()

    def _bootstrap_value_avg_over_handlers(model, handlers_list, device):
        """
        Compute an average V(s_{t+1}) over currently alive handlers for the
        final bootstrap in GAE. Restores the model train/eval mode afterwards.
        """
        vals = []
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                for hnd in handlers_list:
                    sf = getattr(hnd, "prev_processed_stacked_frames", None)
                    gf = getattr(hnd, "prev_processed_game_features", None)
                    if sf is None or gf is None:
                        continue
                    _, _, _, v, _ = model.get_action_and_value(sf.to(device), gf.to(device))
                    vals.append(v.squeeze())
            if vals:
                return torch.stack(vals).mean().to(device)
            return torch.tensor(0.0, device=device)
        finally:
            model.train(mode=was_training)

    # =================================================================
    #  main event-loop
    # =================================================================
    try:
        while shared["total_steps_trained"] < config.MAX_TRAINING_STEPS:
            # ----------------------------------------------------------
            #  1. replace crashed / finished windows
            # ----------------------------------------------------------
            if maint_timer.every(2.0):
                # plan-aware healing: respawn exact plan entries by port
                new_cfgs = game_mgr.maintain_from_plan(config.INSTANCES)
                for cfg in new_cfgs:
                    task = asyncio.create_task(_attach_handler(cfg).start_handling())
                    handler_tasks.append(task)
                if new_cfgs:
                    print(f"🆕  Re-attached {len(new_cfgs)} handler(s) from plan.")

            # ----------------------------------------------------------
            #  2. PPO update when buffer full
            # ----------------------------------------------------------
            if shared["steps_collected_since_last_train"] >= config.NUM_STEPS_PER_COLLECT:
                print(f"\n--- Collected {shared['steps_collected_since_last_train']} steps. Starting PPO update ---")

                # 1) Snapshot the buffer & immediately swap a fresh one so handlers keep collecting
                snapshot_buf = experience_buffer
                experience_buffer = ExperienceBuffer(
                    buffer_size=config.EXPERIENCE_BUFFER_SIZE,
                    mini_batch_size=config.MINI_BATCH_SIZE,
                    num_game_features=NUM_GAME_FEATURES_FOR_MODEL,
                    frame_shape=(config.SEQ_LEN_FRAMES, chan,   # 🔧 here too
                                 config.FRAME_HEIGHT, config.FRAME_WIDTH),                    
                    gamma=config.GAMMA,
                    gae_lambda=config.GAE_LAMBDA,
                    device=config.DEVICE,
                )
                # re-point all handlers to the fresh buffer
                for h in handlers:
                    h.experience_buffer = experience_buffer

                # housekeeping counters
                shared["total_steps_trained"] += shared["steps_collected_since_last_train"]
                shared["steps_collected_since_last_train"] = 0
                cur_steps = shared["total_steps_trained"]
                update_idx = cur_steps // config.NUM_STEPS_PER_COLLECT

                # 2) Train on the snapshot in the background
                await _ppo_update_async(snapshot_buf, handlers, actor_critic_model, ppo_trainer, writer, shared, config)

                # 3) (optional) save model
                if update_idx > 0 and update_idx % config.MODEL_SAVE_FREQUENCY == 0:
                    ckpt = os.path.join(config.MODEL_SAVE_DIR, f"mmbn_ppo_model_{cur_steps}.pth")
                    torch.save(actor_critic_model.state_dict(), ckpt)
                    print(f"✅  Model checkpoint saved to {ckpt}")

            # ----------------------------------------------------------
            #  3. occasional status log
            # ----------------------------------------------------------
            if log_timer.every(30.0):
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"windows: {len(game_mgr.processes)}/{WINDOWS_TARGET}  |  "
                    f"steps: {shared['total_steps_trained']}"
                )

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"❌  Unhandled error in training loop: {e}")
        print(traceback.format_exc())
    finally:
        print("Closing writer & killing game windows …")
        writer.close()
        game_mgr.terminate_all_instances()


if __name__ == "__main__":
    try:
        asyncio.run(main_drl_training_loop())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Initiating shutdown sequence in main...")
    except Exception as e:
        print(f"Unhandled error in __main__: {e}")
        print(traceback.format_exc())
    finally:
        print("DRL Training Program Exiting from __main__.")
