from __future__ import annotations
import random, time
from typing import Dict, Any, Optional
from collections import defaultdict

from PIL import Image

from actions import KEY_BIT_POSITIONS, DISCRETE_ACTIONS, map_discrete_to_mask

class RandomController:
    '''
    A simple, non-learning controller that:
      - in menus (inside_window==True): occasionally presses RETURN/Z to 'advance'
      - in combat: randomly moves and fires; supports sticky hold on X/Z with release prob
    Tunable via constructor arguments.
    '''

    def __init__(self,
                 move_prob: float = 0.55,
                 fire_prob: float = 0.35,
                 hold_prob: float = 0.10,
                 release_prob: float = 0.20,
                 menu_press_interval: float = 0.6):
        self.move_prob       = move_prob
        self.fire_prob       = fire_prob
        self.hold_prob       = hold_prob
        self.release_prob    = release_prob
        self.menu_press_interval = menu_press_interval

        # per-port sticky holds
        self._held_mask = defaultdict(int)  # int bitmask

        # menu timers
        self._menu_ts = defaultdict(float)

    # ------------------------------------------------------------------
    def reset_state(self, port: int) -> None:
        self._held_mask[port] = 0
        self._menu_ts[port]   = 0.0

    # ------------------------------------------------------------------
    def _maybe_menu_press(self, port: int) -> int:
        '''Occasionally press RETURN or Z while in menus, with pauses between taps.'''
        now = time.time()
        last = self._menu_ts[port]
        if last == 0.0 or (now - last) >= self.menu_press_interval:
            self._menu_ts[port] = now
            # 70% RETURN, 30% Z
            if random.random() < 0.7:
                return 1 << KEY_BIT_POSITIONS['RETURN']
            return 1 << KEY_BIT_POSITIONS['Z']
        return 0

    # ------------------------------------------------------------------
    def decide_action(self, port: int, game_state: Dict[str, Any],
                      pil_image: Optional[Image.Image]) -> int:
        inside_menu = bool(game_state.get('inside_window', False))
        if inside_menu:
            # Clear combat holds while in menus
            self._held_mask[port] = 0
            return self._maybe_menu_press(port)

        # --- Combat choice -------------------------------------------------
        mask = 0

        # Random movement (choose at most one direction per frame)
        if random.random() < self.move_prob:
            dir_name = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            mask |= 1 << KEY_BIT_POSITIONS[dir_name]

        # Random fire (X or Z tap this frame)
        if random.random() < self.fire_prob:
            mask |= 1 << KEY_BIT_POSITIONS[random.choice(['X', 'Z'])]

        # Sticky holds: may set a hold bit
        if random.random() < self.hold_prob:
            bit = KEY_BIT_POSITIONS[random.choice(['X', 'Z'])]
            self._held_mask[port] |= (1 << bit)

        # Randomly release any held bit
        if self._held_mask[port] and random.random() < self.release_prob:
            held_bits = [b for b in [KEY_BIT_POSITIONS['X'], KEY_BIT_POSITIONS['Z']]
                         if self._held_mask[port] & (1 << b)]
            if held_bits:
                to_clear = random.choice(held_bits)
                self._held_mask[port] &= ~(1 << to_clear)

        return mask | self._held_mask[port]
