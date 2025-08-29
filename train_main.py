# train_main.py

import asyncio

import time

import traceback

import os

import random

import re # For parsing filenames

import torch

from torch.utils.tensorboard import SummaryWriter

from collections import deque

from models import ActorCriticCNNRNN

# Import modular components

import config

import utils

from game_manager import GameManager

# from models import ActorCriticCNN

from experience_buffer import ExperienceBuffer

from ppo_trainer import PPOTrainer

from strategy import DRLAgentStrategy, SkipAndRandomStrategy

from network_handler import ConnectionHandler



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
    def __init__(self):           # start “now”
        self._t = time.time()
    def every(self, seconds: float) -> bool:
        """Returns True once every <seconds> seconds."""
        now = time.time()
        if now - self._t >= seconds:
            self._t = now
            return True
        return False
    



# ---------------------------------------------------------------------
#  main training loop – auto‑heals missing windows
# ---------------------------------------------------------------------
async def main_drl_training_loop() -> None:
    print(f"Starting DRL Training Script using device: {config.DEVICE}")
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR,     exist_ok=True)

    writer = SummaryWriter(
        log_dir=os.path.join(
            config.TENSORBOARD_LOG_DIR, f"mmbn_ppo_{int(time.time())}"
        )
    )

    # -----------------------------------------------------------------
    # constants (short names so the code reads cleanly)
    # -----------------------------------------------------------------
    WINDOWS_TARGET   = config.NUM_GAME_PAIRS * 2
    ROM_PATH         = config.ROM_PATH_DEFAULT
    SAVE_TMPL        = config.SAVE_PATH_TEMPLATE
    INIT_CODE_BASE   = config.INIT_CODE_DEFAULT
    ADDRESS          = config.ADDRESS_DEFAULT
    NUM_GAME_FEATURES_FOR_MODEL = 11      # <- keep in sync with model

    # -----------------------------------------------------------------
    # build Actor‑Critic
    # -----------------------------------------------------------------
    actor_critic_model = ActorCriticCNNRNN(
        seq_len_frames    = config.SEQ_LEN_FRAMES,
        num_game_features = NUM_GAME_FEATURES_FOR_MODEL,
        num_actions       = len(config.DISCRETE_ACTIONS),
        frame_height      = config.FRAME_HEIGHT,
        frame_width       = config.FRAME_WIDTH,
        frame_channels    = config.FRAME_CHANNELS,
        cnn_channels      = config.CNN_CHANNELS,
        cnn_kernels       = config.CNN_KERNELS,
        cnn_strides       = config.CNN_STRIDES,
        fuse_dim          = config.FUSE_DIM,
        rnn_type          = config.RNN_TYPE,
        rnn_hidden        = config.RNN_HIDDEN,
        rnn_layers        = config.RNN_LAYERS,
        rnn_dropout       = config.DROPOUT_RNN,
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
            print(f"Loaded {ckpt_file} ({ckpt_steps} steps).")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")

    # -----------------------------------------------------------------
    # helpers (buffer, trainer, strategy)
    # -----------------------------------------------------------------
    experience_buffer = ExperienceBuffer(
        buffer_size       = config.EXPERIENCE_BUFFER_SIZE,
        mini_batch_size   = config.MINI_BATCH_SIZE,
        num_game_features = NUM_GAME_FEATURES_FOR_MODEL,
        frame_shape = (
            config.SEQ_LEN_FRAMES,
            config.FRAME_CHANNELS,
            config.FRAME_HEIGHT,
            config.FRAME_WIDTH,
        ),
        gamma      = config.GAMMA,
        gae_lambda = config.GAE_LAMBDA,
        device     = config.DEVICE,
    )

    ppo_trainer = PPOTrainer(
        actor_critic_model = actor_critic_model,
        learning_rate      = config.LEARNING_RATE,
        ppo_clip_epsilon   = config.PPO_CLIP_EPSILON,
        ppo_epochs         = config.PPO_EPOCHS,
        value_loss_coef    = config.VALUE_LOSS_COEF,
        entropy_coef       = config.ENTROPY_COEF,
        device             = config.DEVICE,
    )

    util_funcs = {
        "int_to_binary_string": utils.int_to_binary_string,
        "map_discrete_action_to_buttons": utils.map_discrete_action_to_buttons,
        "preprocess_frame": utils.preprocess_frame,
    }

    drl_strategy = DRLAgentStrategy(
        config.KEY_BIT_POSITIONS, config.DISCRETE_ACTIONS, util_funcs,
        actor_critic_model, config.DEVICE,
        config.FRAME_HEIGHT, config.FRAME_WIDTH, config.SEQ_LEN_FRAMES,
        max_health_config   = config.MAX_HEALTH,
        max_charge_config   = config.MAX_CHARGE_LEVEL,
        max_cust_gauge_config = config.MAX_CUST_GAUGE_VALUE,
    )
    opponent_strategy = drl_strategy   # currently self‑play

    shared = {
        "episode_rewards":   deque(maxlen=100),
        "episode_lengths":   deque(maxlen=100),
        "steps_collected_since_last_train": 0,
        "total_steps_trained":             initial_total_steps_trained,
        "reward_batch":      {k: 0.0 for k in (
            "damage_dealt","damage_taken","charge_gain",
            "charge_shot","time_penalty","win","loss")},
        "reward_cumulative": {k: 0.0 for k in (
            "damage_dealt","damage_taken","charge_gain",
            "charge_shot","time_penalty","win","loss")},
    }



    # -----------------------------------------------------------------
    # Game‑process orchestration
    # -----------------------------------------------------------------
    game_mgr = GameManager(
        config.APP_PATH, config.ENV_COMMON, config.INSTANCE_STAGGER_TIME,
        base_port = config.BASE_PORT,
    )

    # 1️⃣ launch initial windows --------------------------------------
    instance_cfgs = game_mgr.start_initial_pairs(
        num_pairs          = config.NUM_GAME_PAIRS,
        rom_path           = ROM_PATH,
        save_path_template = SAVE_TMPL,
        init_code_base     = INIT_CODE_BASE,
        address            = ADDRESS,
    )

    # build first batch of handlers -----------------------------------
    handlers        : list[ConnectionHandler] = []
    port_to_handler : dict[int, ConnectionHandler] = {}
    pid_to_port     : dict[int, int] = {}

    def _attach_handler(cfg):
        strat = drl_strategy   # could be per‑role; single strat for now
        h = ConnectionHandler(
            cfg, strat, config.INFERENCE_FPS,
            experience_buffer, shared, config, utils,
        )
        #pick a random number from 0 - 5 inclusive
        mode = random.randint(0, 5)
        # mode = int(cfg.get("cross_select", 1))  # add this to your instance_cfgs however you like
        drl_strategy.set_cross_select(h.port, mode)

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

    # start manual‑input router (dicts are mutable — updates propagate)
    from manual_input_router import ManualInputRouter
    manual_router = ManualInputRouter(
        port_to_handler, pid_to_port,
        config.KEY_BIT_POSITIONS, utils.int_to_binary_string,
    )
    manual_router.start()

    # schedule handler tasks ------------------------------------------
    handler_tasks = [asyncio.create_task(h.start_handling()) for h in handlers]

    # maintenance timers ----------------------------------------------
    maint_timer   = _RepeatingTimer()   # every 2 s
    log_timer     = _RepeatingTimer()   # every 30 s
    
    def _bootstrap_value_avg_over_handlers(model, handlers, device):
        """
        Compute an average V(s_{t+1}) over currently alive handlers to use as the
        final bootstrap for GAE. Restores the model's train/eval mode afterwards.
        """
        vals = []
        was_training = model.training
        try:
            # Temporarily evaluate *without* building a grad graph
            model.eval()
            with torch.no_grad():
                for h in handlers:
                    sf = getattr(h, "prev_processed_stacked_frames", None)
                    gf = getattr(h, "prev_processed_game_features",  None)
                    if sf is None or gf is None:
                        continue
                    _, _, _, v, _ = model.get_action_and_value(
                        sf.to(device), gf.to(device)
                    )
                    vals.append(v.squeeze())
            if vals:
                return torch.stack(vals).mean().to(device)
            return torch.tensor(0.0, device=device)
        finally:
            # Restore prior mode so PPO can safely run backward through GRU
            if was_training:
                model.train()
            else:
                model.eval()



    # =================================================================
    #  ░ main event‑loop ░
    # =================================================================
    try:
        while shared["total_steps_trained"] < config.MAX_TRAINING_STEPS:
            # ----------------------------------------------------------
            #  1. replace crashed / finished windows
            # ----------------------------------------------------------
            if maint_timer.every(2.0):
                new_cfgs = game_mgr.maintain_window_count(
                    WINDOWS_TARGET, ROM_PATH, SAVE_TMPL, INIT_CODE_BASE, ADDRESS
                )
                for cfg in new_cfgs:
                    task = asyncio.create_task(_attach_handler(cfg).start_handling())
                    handler_tasks.append(task)
                if new_cfgs:
                    print(f"🆕  Attached {len(new_cfgs)} fresh handler(s).")

            # ----------------------------------------------------------
            #  2. PPO update when buffer full
            # ----------------------------------------------------------
            if shared["steps_collected_since_last_train"] >= config.NUM_STEPS_PER_COLLECT:
                print(f"\n--- Collected {shared['steps_collected_since_last_train']} steps. Starting PPO update ---")
                actor_critic_model.train()

                # --- compute bootstrap V(s_{t+1}) over current env states -----------
                last_val_for_gae = _bootstrap_value_avg_over_handlers(
                    actor_critic_model, handlers, config.DEVICE
                )

                # --- iterate mini-batches -------------------------------------------
                tot_pol = tot_val = tot_ent = 0.0
                n_batches = 0
                for batch in experience_buffer.get_batches(last_val_for_gae):
                    s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs = batch
                    p_loss, v_loss, ent = ppo_trainer.train_step(
                        s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs
                    )
                    tot_pol += p_loss; tot_val += v_loss; tot_ent += ent; n_batches += 1

                avg_pol = tot_pol / max(1, n_batches)
                avg_val = tot_val / max(1, n_batches)
                avg_ent = tot_ent / max(1, n_batches)

                # --- bookkeeping ------------------------------------------------------
                shared["total_steps_trained"] += shared["steps_collected_since_last_train"]
                cur_steps = shared["total_steps_trained"]
                shared["steps_collected_since_last_train"] = 0
                update_idx = cur_steps // config.NUM_STEPS_PER_COLLECT

                # --- TensorBoard logs -------------------------------------------------
                if writer and shared["episode_rewards"]:
                    avg_r = sum(shared["episode_rewards"]) / len(shared["episode_rewards"])
                    avg_l = sum(shared["episode_lengths"]) / len(shared["episode_lengths"])
                    writer.add_scalar("Charts/AverageEpisodeReward", avg_r, cur_steps)
                    writer.add_scalar("Charts/AverageEpisodeLength",  avg_l, cur_steps)

                writer.add_scalar("Losses/PolicyLoss", avg_pol, cur_steps)
                writer.add_scalar("Losses/ValueLoss",  avg_val, cur_steps)
                writer.add_scalar("Charts/Entropy",    avg_ent, cur_steps)

                # reward breakdowns
                for name, val in shared["reward_batch"].items():
                    writer.add_scalar(f"Rewards/{name}", val, cur_steps)
                    shared["reward_batch"][name] = 0.0
                for name, cum_val in shared["reward_cumulative"].items():
                    writer.add_scalar(f"RewardsCumulative/{name}", cum_val, cur_steps)

                writer.flush()

                # --- checkpoint -------------------------------------------------------
                if update_idx > 0 and update_idx % config.MODEL_SAVE_FREQUENCY == 0:
                    ckpt = os.path.join(config.MODEL_SAVE_DIR, f"mmbn_ppo_model_{cur_steps}.pth")
                    torch.save(actor_critic_model.state_dict(), ckpt)
                    print(f"✅  Model checkpoint saved to {ckpt}")


            # ----------------------------------------------------------
            #  3. occasional status log
            # ----------------------------------------------------------
            if log_timer.every(30.0):
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"windows: {len(game_mgr.processes)}/{WINDOWS_TARGET}  |  "
                      f"steps: {shared['total_steps_trained']}")

            # ----------------------------------------------------------
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"❌  Unhandled error in training loop: {e}")
        print(traceback.format_exc())
    finally:
        print("Closing writer & killing game windows …")
        writer.close()
        game_mgr.terminate_all_instances()

if __name__ == '__main__':

  # game_manager_instance is no longer assigned from asyncio.run directly for final cleanup,

  # as cleanup is handled inside main_drl_training_loop's finally block.

  try:

    asyncio.run(main_drl_training_loop())

  except KeyboardInterrupt:

    print("\nKeyboard interrupt. Initiating shutdown sequence in main...")

    # The finally block in main_drl_training_loop should handle graceful shutdown.

  except Exception as e:

    print(f"Unhandled error in __main__: {e}")

    print(traceback.format_exc())

  finally:

    print("DRL Training Program Exiting from __main__.")