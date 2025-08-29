import subprocess
import time
from typing import Dict, List

class GameManager:
    def __init__(self, app_path: str, env_common: Dict[str, str],
                 instance_stagger_time: float = 0.2, base_port: int = 12340):
        self.app_path = app_path
        self.env_common = env_common
        self.instance_stagger_time = instance_stagger_time
        self.processes: List[Dict] = []
        self._next_port = base_port
        self._next_pair_idx = 1

    def _run_single_instance(self, rom_path: str, save_path: str, port: int,
                             init_link_code: str, name: str) -> None:
        env = self.env_common.copy()
        env.update(
            ROM_PATH=rom_path,
            SAVE_PATH=save_path,
            INIT_LINK_CODE=init_link_code,
            PORT=str(port),
            INSTANCE_NAME=str(port),
        )
        print(f"Starting instance «{name}» on port {port} …")
        proc = subprocess.Popen([self.app_path], env=env)
        self.processes.append({"process": proc, "port": port, "name": name})

    def _spawn_game_pair(self, rom_path: str, save_path_template: str,
                         init_code_base: str, address: str = "127.0.0.1") -> List[Dict]:
        learner_port = self._next_port
        opponent_port = learner_port + 1
        self._next_port += 2
        pair_idx = self._next_pair_idx
        self._next_pair_idx += 1
        save_path = save_path_template.format(idx=pair_idx)

        inst_cfgs = [
            dict(address=address, port=learner_port, rom_path=rom_path,
                 save_path=save_path, name=f"Game{pair_idx} Learner",
                 init_link_code=f"{init_code_base}-{pair_idx-1}"),
            dict(address=address, port=opponent_port, rom_path=rom_path,
                 save_path=save_path, name=f"Game{pair_idx} Opponent",
                 init_link_code=f"{init_code_base}-{pair_idx-1}"),
        ]
        for cfg in inst_cfgs:
            self._run_single_instance(cfg["rom_path"], cfg["save_path"], cfg["port"],
                                      cfg["init_link_code"], cfg["name"])
        time.sleep(self.instance_stagger_time)
        return inst_cfgs

    def _purge_zombies(self) -> int:
        before = len(self.processes)
        self.processes = [p for p in self.processes if p["process"].poll() is None]
        return before - len(self.processes)

    def start_initial_pairs(self, num_pairs: int, rom_path: str,
                            save_path_template: str, init_code_base: str,
                            address: str = "127.0.0.1") -> List[Dict]:
        all_cfgs: List[Dict] = []
        for _ in range(num_pairs):
            all_cfgs.extend(self._spawn_game_pair(rom_path, save_path_template, init_code_base, address))
        return all_cfgs

    def maintain_window_count(self, target_windows: int, rom_path: str,
                              save_path_template: str, init_code_base: str,
                              address: str = "127.0.0.1") -> List[Dict]:
        zombies = self._purge_zombies()
        if zombies:
            print(f"🗑️  Cleaned up {zombies} finished window(s).")
        new_cfgs: List[Dict] = []
        while len(self.processes) < target_windows:
            print(f"Window-count {len(self.processes)}/{target_windows}. Spawning replacement pair …")
            new_cfgs.extend(self._spawn_game_pair(rom_path, save_path_template, init_code_base, address))
        return new_cfgs

    def terminate_all_instances(self) -> None:
        print("Terminating all game instances …")
        for info in self.processes:
            proc = info["process"]
            if proc.poll() is None:
                print(f" • killing «{info['name']}» (PID {proc.pid})")
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.processes.clear()
        print("All windows closed.")
