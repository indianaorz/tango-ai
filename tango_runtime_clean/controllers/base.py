from __future__ import annotations
from typing import Protocol, Dict, Any, Optional

class Controller(Protocol):
    '''Interface all controllers must implement.'''
    def reset_state(self, port: int) -> None: ...
    def decide_action(self, port: int, game_state: Dict[str, Any],
                      pil_image: Optional['Image.Image']) -> int:
        '''
        Return an integer 16-bit mask of buttons to press this frame.
        The runtime will convert it to a binary string for the server.
        '''
        ...
