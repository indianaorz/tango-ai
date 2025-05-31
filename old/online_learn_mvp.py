import traceback
import subprocess
import os
import time
import asyncio
import json
import random
from collections import defaultdict

# --- Global Configuration & Constants ---

def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))

APP_PATH = os.path.join(get_root_dir(), "dist/tango-x86_64-linux.AppImage") # Ensure this path is correct

# Simplified INSTANCES configuration for two players in the same battle
INSTANCES = [
    {
        'address': '127.0.0.1',
        'port': 12344,
        'rom_path': 'bn6,0', # Example: Gregar
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 1.sav', # Replace with actual path
        'name': 'Instance 1 (Port 12344)',
        'init_link_code': 'arena_mvp', # Must be same for both instances to connect
    },
    {
        'address': '127.0.0.1',
        'port': 12345,
        'rom_path': 'bn6,0', # Example: Gregar
        'save_path': '/home/lee/Documents/Tango/saves/BN6 Gregar 2.sav', # Replace with actual path (use a different save or copy)
        'name': 'Instance 2 (Port 12345)',
        'init_link_code': 'arena_mvp', # Must be same for both instances to connect
    },
]

# Key mappings for random actions and skip logic
KEY_BIT_POSITIONS = {
    'A': 8,     # Game's B button (often cancel or special chip use)
    'DOWN': 7,
    'UP': 6,
    'LEFT': 5,
    'RIGHT': 4,
    'RETURN': 3, # Start button / Confirm in some menus
    'X': 1,     # Game's A button (buster/chip use)
    'Z': 0,     # Game's L-shoulder (often Confirm in menus like chip selection)
}

# For random action generation (excluding RETURN and Z to avoid menu conflicts during battle)
RANDOM_ACTION_KEYS = ['A', 'DOWN', 'UP', 'LEFT', 'RIGHT', 'X']


# --- State Tracking for Window Skipping ---
# skip_logic_state stores {port: (state_code, timestamp)}
# state_code:
#   0: Initial / Ready to press RETURN
#   1: RETURN pressed, waiting for 1 sec. timestamp = time_return_pressed
#   2: Z pressed, sequence complete.
skip_logic_state = defaultdict(lambda: (0, 0.0))


# Environment variables for launching instances
env_common = os.environ.copy()
env_common["INIT_LINK_CODE"] = "valuesearch" # Default, overridden by instance spec
env_common["AI_MODEL_PATH"] = "ai_model" # Not used by this MVP, but tango might expect it

# --- Utility Functions ---

def int_to_binary_string(value):
    """Converts an integer to a 16-bit binary string."""
    return format(value, '016b')

def generate_random_action():
    """Generates a random key press combination as a 16-bit binary string."""
    binary_command = 0
    # Press 0 to 2 random keys (more likely to press fewer)
    num_keys_to_press = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2], k=1)[0]

    if num_keys_to_press > 0:
        selected_keys = random.sample(RANDOM_ACTION_KEYS, num_keys_to_press)
        for key_name in selected_keys:
            if key_name in KEY_BIT_POSITIONS:
                binary_command |= (1 << KEY_BIT_POSITIONS[key_name])
    
    return int_to_binary_string(binary_command)

# --- Core Logic for Action Decision ---

def decide_action(port, inside_window_flag):
    """
    Decides the next action based on whether the game is in the custom window.
    Includes an initial 0.5s delay, then skips the window with a 1-second delay 
    (pressing RETURN during the 1s wait), or sends random actions.
    """
    global skip_logic_state

    if inside_window_flag:
        # State codes for skip_logic_state[port] = (state_code, timestamp):
        # 0: Default/Reset state (technically, when not inside window, or before first detection)
        # 10: Just entered window, initial 0.5s wait. `timestamp` is time of entry.
        # 11: Initial 0.5s wait over, first RETURN sent. Waiting 1s to send Z. `timestamp` is time first RETURN was sent.
        # 12: Z sent. Sequence complete for this window. `timestamp` is irrelevant.

        current_state_code, timestamp = skip_logic_state[port] # defaultdict provides (0, 0.0) if new or reset

        if current_state_code == 0: # Just detected inside_window (or was reset)
            # Start initial 0.5s wait period
            skip_logic_state[port] = (10, time.time()) # Transition to State 10: Initial_Wait
            print(f"Port {port}: Entered window. State 0 -> 10: Initial 0.5s wait started.")
            return {'type': 'key_press', 'key': '0000000000000000'} # Send NO-OP during initial wait

        elif current_state_code == 10: # State 10: Initial_Wait
            elapsed_initial_wait = time.time() - timestamp
            if elapsed_initial_wait >= 0.5:
                # Initial 0.5s wait is over, send the first RETURN
                # Transition to State 11, recording the time this RETURN is sent for the *next* 1s delay phase
                skip_logic_state[port] = (11, time.time())
                print(f"Port {port}: Skipping window (State 10 -> 11: Initial wait {elapsed_initial_wait:.2f}s over. Sending FIRST RETURN, starting 1s Z-timer)")
                return {'type': 'key_press', 'key': int_to_binary_string(1 << KEY_BIT_POSITIONS['RETURN'])}
            else:
                # Still in initial 0.5s wait
                # print(f"Port {port}: Skipping window (State 10: Initial wait {elapsed_initial_wait:.2f}s)")
                return {'type': 'key_press', 'key': '0000000000000000'} # Send NO-OP

        elif current_state_code == 11: # State 11: First_Return_Sent, now waiting 1 second for Z
            elapsed_z_wait = time.time() - timestamp
            if elapsed_z_wait >= 1.0:
                # 1s wait for Z is over, send Z
                skip_logic_state[port] = (12, 0.0) # Transition to State 12: Z_Sent_Sequence_Complete
                print(f"Port {port}: Skipping window (State 11 -> 12: Sending Z after {elapsed_z_wait:.2f}s Z-wait)")
                return {'type': 'key_press', 'key': int_to_binary_string(1 << KEY_BIT_POSITIONS['Z'])}
            else:
                # Still in 1s Z-wait, keep pressing RETURN
                # print(f"Port {port}: Skipping window (State 11: Z-wait {elapsed_z_wait:.2f}s, pressing RETURN)")
                return {'type': 'key_press', 'key': int_to_binary_string(1 << KEY_BIT_POSITIONS['RETURN'])}

        elif current_state_code == 12: # State 12: Z_Sent_Sequence_Complete
            # Sequence is done, send no-op until out of window
            # print(f"Port {port}: Skipping window (State 12: Sequence complete, waiting for window exit)")
            return {'type': 'key_press', 'key': '0000000000000000'}
        
        # Fallback, should ideally not be reached if states are managed correctly
        return {'type': 'key_press', 'key': '0000000000000000'}

    else: # Not in the chip selection window
        # Reset skip logic state to 0 if it was actively in a skipping state
        if skip_logic_state[port][0] != 0:
            print(f"Port {port}: Exited window, resetting skip logic to State 0.")
            skip_logic_state[port] = (0, 0.0) # Reset to initial state (0)
        
        # In battle: send random actions
        random_key_string = generate_random_action()
        # print(f"Port {port}: In battle, sending random action: {random_key_string}")
        return {'type': 'key_press', 'key': random_key_string}

# --- Asynchronous Network Communication ---

async def send_input_command(writer, command, port=0):
    """Sends a command to the game instance."""
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        print(f"Port {port}: Connection closed while sending. Stopping for this instance.")
        raise 
    except Exception as e:
        print(f"Port {port}: Failed to send command: {e}")
        raise

async def request_screen_image(writer, port):
    """Requests a screen update from the game instance."""
    try:
        command = {'type': 'request_screen', 'key': ''}
        await send_input_command(writer, command, port)
    except Exception:
        raise

async def receive_messages(reader, writer, port):
    """Receives and processes messages from the game instance."""
    buffer = ""
    try:
        while True:
            data = await reader.read(8192)
            if not data:
                print(f"Port {port}: Connection closed by peer.")
                break
            
            buffer += data.decode()
            
            while "\n" in buffer:
                message, buffer = buffer.split("\n", 1)
                message = message.strip()
                if not message:
                    continue

                try:
                    parsed_message = json.loads(message)
                except json.JSONDecodeError:
                    print(f"Port {port}: Failed to parse JSON: {message[:100]}...")
                    continue

                event = parsed_message.get("event", "Unknown")
                details = parsed_message.get("details", {})

                if event == "screen_image":
                    try:
                        screen_data = json.loads(details) if isinstance(details, str) else details
                        inside_window_flag = bool(float(screen_data.get("inside_window", 0)))
                        
                        action_command = decide_action(port, inside_window_flag)
                        
                        if writer and not writer.is_closing():
                           await send_input_command(writer, action_command, port)
                        else:
                            print(f"Port {port}: Writer closed, cannot send command.")
                            return 

                    except json.JSONDecodeError:
                        print(f"Port {port}: Failed to parse screen_image details: {details[:100]}...")
                    except Exception as e:
                        print(f"Port {port}: Error processing screen_image: {e}")
                        print(traceback.format_exc())

    except (ConnectionResetError, BrokenPipeError):
        print(f"Port {port}: Connection was reset/broken in receiver.")
    except Exception as e:
        print(f"Port {port}: Error in receive_messages: {e}")
        print(traceback.format_exc())
    finally:
        print(f"Port {port}: receive_messages loop ended.")


async def handle_connection(instance_config):
    """Manages the connection lifecycle for a single game instance."""
    port = instance_config['port']
    writer = None
    receive_task = None
    print(f"Attempting to connect to {instance_config['name']} at {instance_config['address']}:{port}")

    try:
        reader, writer = await asyncio.open_connection(instance_config['address'], port)
        print(f"Connected to {instance_config['name']}")

        receive_task = asyncio.create_task(receive_messages(reader, writer, port))

        inference_interval = 1 / 30  # Approx 30 FPS for requests
        while not reader.at_eof() and writer and not writer.is_closing():
            try:
                await request_screen_image(writer, port)
                await asyncio.sleep(inference_interval)
            except (ConnectionResetError, BrokenPipeError):
                print(f"Port {port}: Connection lost in send loop.")
                break 
            except Exception as e:
                print(f"Port {port}: Error in periodic screen request: {e}")
                break
        
        print(f"Port {port}: Send loop ended.")

    except ConnectionRefusedError:
        print(f"Port {port}: Connection refused for {instance_config['name']}.")
    except Exception as e:
        print(f"Port {port}: Error in handle_connection for {instance_config['name']}: {e}")
        print(traceback.format_exc())
    finally:
        if receive_task and not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                print(f"Port {port}: receive_task cancelled.")
        if writer:
            try:
                if not writer.is_closing():
                    writer.close()
                await writer.wait_closed()
            except Exception as e:
                print(f"Port {port}: Error closing writer for {instance_config['name']}: {e}")
        print(f"Connection to {instance_config['name']} (Port {port}) fully closed.")


# --- Instance Management ---

def run_instance(rom_path, save_path, port, init_link_code, name):
    """Runs a single game instance."""
    env = env_common.copy()
    env["ROM_PATH"] = rom_path
    env["SAVE_PATH"] = save_path
    env["INIT_LINK_CODE"] = init_link_code
    env["PORT"] = str(port)
    env["INSTANCE_NAME"] = str(port) 
    
    print(f"Starting instance '{name}' on Port {port} with ROM: {rom_path}, Save: {save_path}")
    try:
        subprocess.Popen([APP_PATH], env=env)
    except FileNotFoundError:
        print(f"ERROR: AppImage not found at {APP_PATH}. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"Failed to start instance '{name}' on port {port}: {e}")

def start_instances():
    """Starts all configured game instances."""
    if not os.path.exists(APP_PATH):
        print(f"ERROR: Tango AppImage not found at {APP_PATH}")
        print("Please ensure the APP_PATH variable is correct and the AppImage is executable.")
        return False
        
    for instance_config in INSTANCES:
        if not os.path.exists(instance_config['save_path']):
            print(f"Warning: Save file not found for instance {instance_config['name']}: {instance_config['save_path']}")
            print("The game might create a new save or fail to load.")

        run_instance(
            instance_config['rom_path'],
            instance_config['save_path'],
            instance_config['port'],
            instance_config['init_link_code'],
            instance_config['name']
        )
        time.sleep(1.0) 
    return True

# --- Main Execution ---

async def main():
    print("Starting MVP script...")
    if not start_instances():
        print("Failed to start instances. Exiting.")
        return

    print("Instances launched. Waiting a bit for them to initialize...")
    await asyncio.sleep(5) 

    connection_tasks = [asyncio.create_task(handle_connection(inst)) for inst in INSTANCES]
    
    try:
        await asyncio.gather(*connection_tasks)
    except Exception as e:
        print(f"Error during asyncio.gather: {e}")

    print("All instance handlers have completed. Program finished.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"An unhandled error occurred in main: {e}")
        print(traceback.format_exc())
    finally:
        print("Exiting program.")