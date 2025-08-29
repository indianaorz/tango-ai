#!/usr/bin/env python3
import asyncio
import time
import os
from typing import List

import config
from game_manager import GameManager
from controllers.random_controller import RandomController
from network_handler import ConnectionHandler

class _Ticker:
    def __init__(self): self._t = time.time()
    def every(self, seconds: float) -> bool:
        now = time.time()
        if now - self._t >= seconds:
            self._t = now; return True
        return False

async def main() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    controller = RandomController()
    windows_target = config.NUM_GAME_PAIRS * 2
    gm = GameManager(config.APP_PATH, config.ENV_COMMON, config.INSTANCE_STAGGER_TIME, base_port=config.BASE_PORT)

    inst_cfgs = gm.start_initial_pairs(
        num_pairs=config.NUM_GAME_PAIRS,
        rom_path=config.ROM_PATH_DEFAULT,
        save_path_template=config.SAVE_PATH_TEMPLATE,
        init_code_base=config.INIT_CODE_DEFAULT,
        address=config.ADDRESS_DEFAULT,
    )

    handlers: List[ConnectionHandler] = []
    for cfg in inst_cfgs:
        handlers.append(ConnectionHandler(cfg, controller, control_fps=config.CONTROL_FPS))

    tasks = [asyncio.create_task(h.start()) for h in handlers]

    maintain_tick = _Ticker()
    log_tick = _Ticker()

    try:
        while True:
            if maintain_tick.every(2.0):
                new_cfgs = gm.maintain_window_count(
                    windows_target, config.ROM_PATH_DEFAULT, config.SAVE_PATH_TEMPLATE,
                    config.INIT_CODE_DEFAULT, config.ADDRESS_DEFAULT
                )
                for cfg in new_cfgs:
                    tasks.append(asyncio.create_task(ConnectionHandler(cfg, controller, control_fps=config.CONTROL_FPS).start()))
                if new_cfgs:
                    print(f"🆕 Attached {len(new_cfgs)} new handler(s).")
            if log_tick.every(30.0):
                print(f"[{time.strftime('%H:%M:%S')}] windows: {len(gm.processes)}/{windows_target}")
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down…")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        gm.terminate_all_instances()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited by user.")
