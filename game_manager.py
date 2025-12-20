# game_manager.py
import subprocess
import time
import os
from typing import Dict, List, Optional

class GameManager:
    """
    Starts, tracks and (optionally) re‑spawns Tango game instances.

    One “instance” == one **window / TCP port**.
    A “game pair” == (learner window, opponent window).
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        app_path: str,
        env_common: Dict[str, str],
        instance_stagger_time: float = 1.0,
        base_port: int = 12350,
    ):
        self.app_path             = app_path
        self.env_common           = env_common
        self.instance_stagger_time= instance_stagger_time

        self.processes: List[Dict] = []          # [{process, port, name}]
        self._next_port            = base_port   # first free TCP port
        self._next_pair_idx        = 1           # for save‑file template

    # ──────────────────────────────────────────────────────────────────
    # low‑level helpers
    # ──────────────────────────────────────────────────────────────────
    def _run_single_instance(
        self,
        rom_path: str,
        save_path: str,
        port: int,
        init_link_code: str,
        name: str,
    ):
        env = self.env_common.copy()
        env.update(
            ROM_PATH       = rom_path,
            SAVE_PATH      = save_path,
            INIT_LINK_CODE = init_link_code,
            PORT           = str(port),
            INSTANCE_NAME  = str(port),
        )

        print(f"Starting instance «{name}» on port {port} …")
        proc = subprocess.Popen([self.app_path], env=env)
        self.processes.append({"process": proc, "port": port, "name": name})

    def _spawn_game_pair(
        self,
        rom_path: str,
        save_path_template: str,
        init_code_base: str,
        address: str = "127.0.0.1",
    ) -> List[Dict]:
        """
        Starts **two** windows (learner/opponent) with consecutive ports and
        returns their full instance‑configs (so the caller can attach new
        ConnectionHandlers).
        """
        learner_port  = self._next_port
        opponent_port = learner_port + 1
        self._next_port += 2

        pair_idx   = self._next_pair_idx
        self._next_pair_idx += 1
        save_path  = save_path_template.format(idx=pair_idx)

        inst_cfgs = [
            dict(
                address = address,     port = learner_port,
                rom_path = rom_path,   save_path = save_path,
                name = f"Game{pair_idx} Learner",
                init_link_code = f"{init_code_base}-{pair_idx-1}",
            ),
            dict(
                address = address,     port = opponent_port,
                rom_path = rom_path,   save_path = save_path,
                name = f"Game{pair_idx} Opponent",
                init_link_code = f"{init_code_base}-{pair_idx-1}",
            ),
        ]

        for cfg in inst_cfgs:
             self._run_single_instance(
                rom_path       = cfg["rom_path"],
                save_path      = cfg["save_path"],
                port           = cfg["port"],
                init_link_code = cfg["init_link_code"],
                name           = cfg["name"],
            )

        # tiny stagger so Tango’s networking init doesn’t collide
        time.sleep(self.instance_stagger_time)
        return inst_cfgs

    def _purge_zombies(self):
        """Drop finished processes from self.processes and return the number removed."""
        before = len(self.processes)
        self.processes = [p for p in self.processes if p["process"].poll() is None]
        return before - len(self.processes)

    # ──────────────────────────────────────────────────────────────────
    # plan-aware helpers
    # ──────────────────────────────────────────────────────────────────
    def _run_instance_from_cfg(self, cfg: Dict):
        """Spawn one window exactly as described by the plan entry."""
        env = self.env_common.copy()
        env.update(
            ROM_PATH       = cfg["rom_path"],
            SAVE_PATH      = cfg["save_path"],
            INIT_LINK_CODE = cfg["init_link_code"],
            PORT           = str(cfg["port"]),
            INSTANCE_NAME  = str(cfg.get("name", cfg["port"])),
        )
        name = cfg.get("name", f"Port {cfg['port']}")
        print(f"Starting instance «{name}» on port {cfg['port']} …")
        proc = subprocess.Popen([self.app_path], env=env)
        self.processes.append({"process": proc, "port": cfg["port"], "name": name})

    def start_instances_from_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Spawn all instances provided by the plan. Returns the list of cfgs
        actually launched (same dicts you passed in).
        """
        launched = []
        for cfg in plan:
            self._run_instance_from_cfg(cfg)
            launched.append(cfg)
            time.sleep(self.instance_stagger_time)  # small stagger
        return launched

    def maintain_from_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Ensure that EXACTLY the set of plan ports are alive. If any plan entry
        (by port) is missing, respawn it using that same cfg.
        Returns a list of cfgs that were newly launched.
        """
        zombies = self._purge_zombies()
        if zombies:
            print(f"🗑️  Cleaned up {zombies} finished window(s).")

        running_ports = {p["port"] for p in self.processes if p["process"].poll() is None}
        new_cfgs: List[Dict] = []
        for cfg in plan:
            if cfg["port"] not in running_ports:
                print(f"Window for port {cfg['port']} missing; respawning from plan …")
                self._run_instance_from_cfg(cfg)
                new_cfgs.append(cfg)
                time.sleep(self.instance_stagger_time)
        return new_cfgs
    
    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    def start_initial_pairs(
        self,
        num_pairs: int,
        rom_path: str,
        save_path_template: str,
        init_code_base: str,
        address: str = "127.0.0.1",
    ) -> List[Dict]:
        """Launch exactly num_pairs learner/opponent pairs."""
        all_cfgs: List[Dict] = []
        for _ in range(num_pairs):
            cfgs = self._spawn_game_pair(
                rom_path, save_path_template, init_code_base, address
            )
            all_cfgs.extend(cfgs)
        return all_cfgs

    # ------------------------------------------------------------------
    def maintain_window_count(
        self,
        target_windows: int,
        rom_path: str,
        save_path_template: str,
        init_code_base: str,
        address: str = "127.0.0.1",
    ) -> List[Dict]:
        """
        • Cleans up finished Popen objects  
        • Spawns fresh game pairs until `len(running_windows) == target_windows`  
        Returns a **list of newly launched instance‑configs** (empty if nothing
        had to be spawned).
        """
        zombies = self._purge_zombies()
        if zombies:
            print(f"🗑️  Cleaned up {zombies} finished window(s).")

        new_cfgs: List[Dict] = []
        while len(self.processes) < target_windows:
            print(f"Window‑count {len(self.processes)}/{target_windows}. Spawning replacement pair …")
            cfgs = self._spawn_game_pair(
                rom_path, save_path_template, init_code_base, address
            )
            new_cfgs.extend(cfgs)
        return new_cfgs

    # ------------------------------------------------------------------
    def terminate_all_instances(self):
        print("Terminating all game instances …")
        for info in self.processes:
            proc = info["process"]
            if proc.poll() is None:
                print(f" • killing «{info['name']}» (PID {proc.pid})")
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.processes.clear()
        print("All windows closed.")
