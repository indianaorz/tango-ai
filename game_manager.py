# game_manager.py
import subprocess
import time
import os

class GameManager:
    def __init__(self, app_path, env_common, instance_stagger_time=1.0):
        self.app_path = app_path
        self.env_common = env_common
        self.instance_stagger_time = instance_stagger_time
        self.processes = []

    def _run_single_instance(self, rom_path, save_path, port, init_link_code, name):
        env = self.env_common.copy()
        env["ROM_PATH"] = rom_path
        env["SAVE_PATH"] = save_path
        env["INIT_LINK_CODE"] = init_link_code
        env["PORT"] = str(port)
        env["INSTANCE_NAME"] = str(port) 
        
        print(f"Starting instance '{name}' on Port {port}...")
        try:
            # For Linux, ensure AppImage is executable. Consider pre-exec or error handling.
            proc = subprocess.Popen([self.app_path], env=env)
            self.processes.append({"process": proc, "port": port, "name": name})
        except FileNotFoundError:
            print(f"ERROR: AppImage not found at {self.app_path}.")
            raise
        except Exception as e:
            print(f"Failed to start instance '{name}' on port {port}: {e}")
            raise

    def start_all_instances(self, instances_config):
        if not os.path.exists(self.app_path):
            print(f"ERROR: Tango AppImage not found at {self.app_path}")
            return False
            
        for config_item in instances_config:
            if not os.path.exists(config_item['save_path']):
                print(f"Warning: Save file for {config_item['name']} not found: {config_item['save_path']}")
            try:
                self._run_single_instance(
                    config_item['rom_path'],
                    config_item['save_path'],
                    config_item['port'],
                    config_item['init_link_code'],
                    config_item['name']
                )
                time.sleep(self.instance_stagger_time) 
            except Exception as e:
                print(f"Could not start instance {config_item['name']}. Error: {e}")
        return True

    def terminate_all_instances(self):
        print("Terminating all game instances...")
        for proc_info in self.processes:
            proc = proc_info["process"]
            if proc.poll() is None: # Check if process is still running
                print(f"Terminating instance {proc_info['name']} (PID: {proc.pid})...")
                try:
                    proc.terminate() # Ask nicely first
                    proc.wait(timeout=3) # Wait for graceful termination
                except subprocess.TimeoutExpired:
                    print(f"Instance {proc_info['name']} (PID: {proc.pid}) did not terminate gracefully, killing...")
                    proc.kill() # Force kill
                    proc.wait() # Ensure it's killed
                except Exception as e:
                    print(f"Error terminating process {proc.pid} for {proc_info['name']}: {e}")
        self.processes = []
        print("All tracked game instances processed for termination.")