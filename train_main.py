# train_main.py

import asyncio

import time

import traceback

import os

import re # For parsing filenames

import torch

from torch.utils.tensorboard import SummaryWriter

from collections import deque



# Import modular components

import config

import utils

from game_manager import GameManager

from models import ActorCriticCNN

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



async def main_drl_training_loop():

  print(f"Starting DRL Training Script using device: {config.DEVICE}")

  os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)

  os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

 

  writer = SummaryWriter(log_dir=os.path.join(config.TENSORBOARD_LOG_DIR, f"mmbn_ppo_{int(time.time())}"))

 

  NUM_GAME_FEATURES_FOR_MODEL = 11



  actor_critic_model = ActorCriticCNN(

    num_stacked_frames=config.NUM_FRAMES_STACKED,

    num_game_features=NUM_GAME_FEATURES_FOR_MODEL,

    num_actions=len(config.DISCRETE_ACTIONS),

    frame_height=config.FRAME_HEIGHT,

    frame_width=config.FRAME_WIDTH

  ).to(config.DEVICE)

  print(f"ActorCritic model initialized with {NUM_GAME_FEATURES_FOR_MODEL} game features.")



  initial_total_steps_trained = 0

  latest_model_file, loaded_steps = find_latest_model_and_steps(config.MODEL_SAVE_DIR)



  if latest_model_file:

    try:

      print(f"Loading model from: {latest_model_file} with {loaded_steps} steps.")

      actor_critic_model.load_state_dict(torch.load(latest_model_file, map_location=config.DEVICE))

      initial_total_steps_trained = loaded_steps

      print(f"Successfully loaded model. Resuming from {initial_total_steps_trained} steps.")

    except Exception as e:

      print(f"Error loading model from {latest_model_file}: {e}. Starting from scratch.")

      initial_total_steps_trained = 0

  else:

    print("No saved models found. Starting training from scratch.")



  experience_buffer = ExperienceBuffer(

    buffer_size=config.NUM_STEPS_PER_COLLECT * len(config.INSTANCES),

    mini_batch_size=config.MINI_BATCH_SIZE,

    num_game_features=NUM_GAME_FEATURES_FOR_MODEL,

    frame_shape=(config.NUM_FRAMES_STACKED, config.FRAME_HEIGHT, config.FRAME_WIDTH),

    gamma=config.GAMMA,

    gae_lambda=config.GAE_LAMBDA,

    device=config.DEVICE

  )



  ppo_trainer = PPOTrainer(

    actor_critic_model=actor_critic_model,

    learning_rate=config.LEARNING_RATE,

    ppo_clip_epsilon=config.PPO_CLIP_EPSILON,

    ppo_epochs=config.PPO_EPOCHS,

    value_loss_coef=config.VALUE_LOSS_COEF,

    entropy_coef=config.ENTROPY_COEF,

    device=config.DEVICE

  )



  util_funcs = {

    'int_to_binary_string': utils.int_to_binary_string,

    'map_discrete_action_to_buttons': utils.map_discrete_action_to_buttons,

    'preprocess_frame': utils.preprocess_frame

  }

  drl_strategy = DRLAgentStrategy(

    config.KEY_BIT_POSITIONS, config.DISCRETE_ACTIONS, util_funcs,

    actor_critic_model, config.DEVICE,

    config.FRAME_HEIGHT, config.FRAME_WIDTH, config.NUM_FRAMES_STACKED,

    max_health_config=config.MAX_HEALTH,

    max_charge_config=config.MAX_CHARGE_LEVEL,

    max_cust_gauge_config=config.MAX_CUST_GAUGE_VALUE

  )

  opponent_strategy = drl_strategy



  shared_episode_data = {

    'episode_rewards': deque(maxlen=100), 'episode_lengths': deque(maxlen=100),

    'steps_collected_since_last_train': 0,

    'total_steps_trained': initial_total_steps_trained,

  }



  # GameManager is created once and reused

  game_mgr = GameManager(config.APP_PATH, config.ENV_COMMON, config.INSTANCE_STAGGER_TIME)

  first_session = True



  try:

    while shared_episode_data['total_steps_trained'] < config.MAX_TRAINING_STEPS:

      if not first_session:

        print(f"--- All game instances terminated. Attempting to restart in {config.GAME_RESTART_DELAY}s... ---")

        await asyncio.sleep(config.GAME_RESTART_DELAY)

      first_session = False

     

      print("--- Starting new training session / Restarting game instances ---")

      game_mgr.terminate_all_instances() # Clean up any previous processes



      if not game_mgr.start_all_instances(config.INSTANCES):

        print("CRITICAL: Failed to start game instances. Exiting training loop.")

        break

     

      print(f"Instances launched. Waiting {config.INSTANCE_INIT_WAIT_TIME}s for initialization...")

      await asyncio.sleep(config.INSTANCE_INIT_WAIT_TIME)



      current_session_handlers = []

      for i, instance_cfg in enumerate(config.INSTANCES):

        strat = drl_strategy if i == 0 else opponent_strategy

        # Each new session gets new ConnectionHandler instances

        # Strategy objects are reused, but their internal per-port state (like frame buffers)

        # should be reset by ConnectionHandler's start_handling calling strategy.reset_state()

        handler = ConnectionHandler(

          instance_cfg, strat, config.INFERENCE_FPS,

          experience_buffer, shared_episode_data, config, utils

        )

        current_session_handlers.append(handler)



      if not current_session_handlers: # Should not happen if config.INSTANCES is populated

        print("CRITICAL: No handlers created, instances might not have started. Exiting.")

        break



      current_session_tasks = [asyncio.create_task(h.start_handling()) for h in current_session_handlers]

      print(f"Started {len(current_session_tasks)} connection handlers for this session.")

     

      session_active = True

      while session_active and shared_episode_data['total_steps_trained'] < config.MAX_TRAINING_STEPS:

        # Check if all handlers for the current session have stopped

        if not any(not task.done() for task in current_session_tasks):

          print("All connection handlers for the current session have completed or failed.")

          session_active = False # Will break outer session loop after this iteration



        # --- Data Collection & PPO Update ---

        if shared_episode_data['steps_collected_since_last_train'] >= config.NUM_STEPS_PER_COLLECT:

          print(f"\n--- Collected {shared_episode_data['steps_collected_since_last_train']} steps. Starting PPO Update ---")

          actor_critic_model.train()



          if experience_buffer.ptr == 0 and not experience_buffer.is_full:

            last_val_for_gae = torch.tensor(0.0, device=config.DEVICE)

            last_done_for_gae = torch.tensor(False, device=config.DEVICE)

          else:

            last_idx_in_buffer = (experience_buffer.ptr - 1 + experience_buffer.buffer_size) % experience_buffer.buffer_size

            if experience_buffer.dones[last_idx_in_buffer]:

              last_val_for_gae = torch.tensor(0.0, device=config.DEVICE)

            else:

              last_val_for_gae = experience_buffer.values[last_idx_in_buffer]

            last_done_for_gae = experience_buffer.dones[last_idx_in_buffer]

         

          total_policy_loss_epoch, total_value_loss_epoch, total_entropy_epoch = 0,0,0

          num_batches = 0



          for batch_data in experience_buffer.get_batches(last_val_for_gae, last_done_for_gae):

            s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs = batch_data

            policy_loss, value_loss, entropy = ppo_trainer.train_step(

              s_frames, s_feats, acts, old_log_ps, advs, rets, old_vs

            )

            total_policy_loss_epoch += policy_loss

            total_value_loss_epoch += value_loss

            total_entropy_epoch += entropy

            num_batches +=1

         

          avg_policy_loss = total_policy_loss_epoch / num_batches if num_batches > 0 else 0

          avg_value_loss = total_value_loss_epoch / num_batches if num_batches > 0 else 0

          avg_entropy = total_entropy_epoch / num_batches if num_batches > 0 else 0





          shared_episode_data['total_steps_trained'] += shared_episode_data['steps_collected_since_last_train']

          current_total_steps = shared_episode_data['total_steps_trained'] # For logging and saving this update cycle

          shared_episode_data['steps_collected_since_last_train'] = 0



          # Logging

          current_update_num = current_total_steps // config.NUM_STEPS_PER_COLLECT if config.NUM_STEPS_PER_COLLECT > 0 else 0

          if writer and len(shared_episode_data['episode_rewards']) > 0:

            avg_reward = sum(shared_episode_data['episode_rewards']) / len(shared_episode_data['episode_rewards'])

            avg_length = sum(shared_episode_data['episode_lengths']) / len(shared_episode_data['episode_lengths'])

            writer.add_scalar('Charts/AverageEpisodeReward', avg_reward, current_total_steps)

            writer.add_scalar('Charts/AverageEpisodeLength', avg_length, current_total_steps)

            print(f"Update {current_update_num}, Total Steps: {current_total_steps}/{config.MAX_TRAINING_STEPS}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")

         

          writer.add_scalar('Losses/PolicyLoss', avg_policy_loss, current_total_steps)

          writer.add_scalar('Losses/ValueLoss', avg_value_loss, current_total_steps)

          writer.add_scalar('Charts/Entropy', avg_entropy, current_total_steps)

          writer.flush()



          # Model Saving

          if current_update_num > 0 and current_update_num % config.MODEL_SAVE_FREQUENCY == 0:

            save_path = os.path.join(config.MODEL_SAVE_DIR, f"mmbn_ppo_model_{current_total_steps}.pth")

            torch.save(actor_critic_model.state_dict(), save_path)

            print(f"Model saved to {save_path}")

       

        if not session_active: # If handlers died, break inner loop after potential update

          break

       

        await asyncio.sleep(0.1) # Yield control, check conditions periodically



      # --- End of current session's inner loop ---

      print(f"Session ended. Cleaning up {len(current_session_tasks)} tasks for this session...")

      for task in current_session_tasks: # Cancel any still running tasks from this session

        if not task.done():

          task.cancel()

      await asyncio.gather(*current_session_tasks, return_exceptions=True) # Wait for tasks to actually finish/cancel

      print("Session tasks cleaned up.")



      if shared_episode_data['total_steps_trained'] >= config.MAX_TRAINING_STEPS:

        print("Maximum training steps reached.")

        break # Break the outer training loop



    # --- End of main training loop (while total_steps < MAX_TRAINING_STEPS) ---

    print("Training loop finished or MAX_TRAINING_STEPS reached.")



  except asyncio.CancelledError:

    print("Main training loop cancelled.")

  except Exception as e:

    print(f"Error in main DRL training loop: {e}")

    print(traceback.format_exc())

  finally:

    print("Closing TensorBoard writer...")

    if writer: writer.close()

    print("Terminating any remaining game instances...")

    game_mgr.terminate_all_instances() # Final cleanup

    print("Game instances terminated.")

  # No need to return game_mgr from here as it's managed within the loop now



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