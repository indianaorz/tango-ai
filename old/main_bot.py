# main_bot.py
import asyncio
import time
import traceback
import os # For potential path adjustments if needed, though config handles most

# Import modular components
import config  # Loads all configurations from config.py
import utils   # Utility functions from utils.py
from game_manager import GameManager
from strategy import SkipAndRandomStrategy, AIStrategy # Import available strategies
from network_handler import ConnectionHandler

async def main_async_orchestrator():
    print("Starting Modular Bot Script...")

    # Initialize GameManager
    # It takes app_path, env_common from config, and instance_stagger_time
    game_mgr = GameManager(
        config.APP_PATH, 
        config.ENV_COMMON, 
        config.INSTANCE_STAGGER_TIME # Defined in config.py
    )

    if not game_mgr.start_all_instances(config.INSTANCES):
        print("Failed to start all game instances. Exiting.")
        return game_mgr # Return for potential cleanup in finally block

    print(f"Instances launched. Waiting {config.INSTANCE_INIT_WAIT_TIME}s for them to initialize...")
    await asyncio.sleep(config.INSTANCE_INIT_WAIT_TIME) # Defined in config.py

    # --- Strategy Selection ---
    # Prepare utility functions to be passed to the strategy instance.
    # This makes the strategy's dependencies explicit.
    util_functions_for_strategy = {
        'int_to_binary_string': utils.int_to_binary_string,
        'generate_random_action': utils.generate_random_action
    }

    # Instantiate the desired strategy. For MVP behavior, this is SkipAndRandomStrategy.
    # It requires key mappings, random action keys, and the utility functions.
    chosen_strategy = SkipAndRandomStrategy(
        config.KEY_BIT_POSITIONS, 
        config.RANDOM_ACTION_KEYS,
        util_functions_for_strategy
    )
    
    # --- Example: To use an AIStrategy in the future (uncomment and configure) ---
    # Ensure AIStrategy is defined in strategy.py and model path is correct.
    # ai_model_path = os.path.join(config.PROJECT_ROOT, "models", "your_ai_model.pth") # Example path
    # chosen_strategy = AIStrategy(
    #     config.KEY_BIT_POSITIONS, 
    #     config.RANDOM_ACTION_KEYS,
    #     util_functions_for_strategy,
    #     model_path=ai_model_path 
    # )
    # --- End Strategy Selection Example ---

    print(f"Using strategy: {chosen_strategy.__class__.__name__}")

    # Create and start ConnectionHandler for each game instance
    connection_handlers = []
    for instance_cfg in config.INSTANCES:
        handler = ConnectionHandler(
            instance_cfg,          # Configuration for this specific instance
            chosen_strategy,       # The selected strategy object (shared or new per instance)
            config.INFERENCE_FPS   # How often to interact with the game
        )
        connection_handlers.append(handler)

    # Create asyncio tasks for each connection handler to run its main loop (start_handling)
    handler_tasks = [asyncio.create_task(handler.start_handling()) for handler in connection_handlers]
    
    try:
        if handler_tasks:
            # Wait for all connection handlers to complete (they run until game/connection ends or error)
            await asyncio.gather(*handler_tasks)
        else:
            print("No connection handler tasks were created to run.")
    except Exception as e: 
        # This catch is a fallback; individual handlers should manage their exceptions.
        print(f"Critical error during asyncio.gather of connection handlers: {e}")
        print(traceback.format_exc())

    print("All instance connection handlers have completed their execution.")
    return game_mgr # Return GameManager for cleanup in the finally block

if __name__ == '__main__':
    game_manager_instance = None
    try:
        # Run the main asynchronous orchestrator
        game_manager_instance = asyncio.run(main_async_orchestrator())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Initiating shutdown...")
        # Note: Individual connection handlers will also see cancellation via asyncio context.
    except Exception as e:
        print(f"An unhandled error occurred in the main execution block: {e}")
        print(traceback.format_exc())
    finally:
        # Ensure game instances are terminated if the GameManager was successfully initialized
        if game_manager_instance:
            print("Main execution finished or interrupted. Terminating game instances...")
            game_manager_instance.terminate_all_instances()
        else:
            print("GameManager was not initialized; no instances to terminate from main block.")
        print("Program exiting.")