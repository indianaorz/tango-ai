# manual_input_router.py
"""
Manual‑override input router for Tango‑AI self‑play
────────────────────────────────────────────────────
• Poll the X11 window under the mouse every POLL_MS ms
• Resolve that window → Tango port via
      – window‑title “… 12340 …”   (fast & reliable with nested Wine windows)
      – or owning PID  → port      (fallback if title has no digits)
• Toggle ConnectionHandler.manual_override so the AI stays silent
• Forward human key‑events to the selected handler
• Store human demonstrations in ExperienceBuffer (action_idx = -1)
"""

from __future__ import annotations

import re
import shlex
import subprocess
import asyncio
import threading
import time
from typing import Dict, Optional, Callable

import torch
from pynput import keyboard

from strategy import DRLAgentStrategy
import utils

POLL_MS = 40
HUMAN_ACTION_IDX = -1
PORT_RE = re.compile(r"\b(\d{5})\b")          # match any 5‑digit number

# ── shell helpers ────────────────────────────────────────────────────────────
def _cmd_out(cmd: str) -> str:
    try:
        return subprocess.check_output(
            shlex.split(cmd), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def window_under_pointer() -> Optional[int]:
    out = _cmd_out("xdotool getmouselocation --shell")
    for ln in out.splitlines():
        if ln.startswith("WINDOW="):
            try:
                return int(ln.split("=", 1)[1], 0)
            except ValueError:
                pass
    return None


def window_pid(xid: int | None) -> Optional[int]:
    if not xid:
        return None
    out = _cmd_out(f"xdotool getwindowpid {xid}")
    try:
        return int(out.strip())
    except ValueError:
        return None


def window_title(xid: int | None) -> str:
    if not xid:
        return ""
    for prop in ("_NET_WM_NAME", "WM_NAME"):
        out = _cmd_out(f"xprop -id {xid} {prop}")
        if "=" in out:
            return out.split("=", 1)[1].strip().strip('"')
    return ""

# ── main class ───────────────────────────────────────────────────────────────
class ManualInputRouter:
    """
    Args
    ----
    port_to_handler : {port:int → ConnectionHandler}
    pid_to_port     : {pid:int  → port:int}
    key_to_bitpos   : KEY_BIT_POSITIONS dict from config
    util_bin16      : utils.int_to_binary_string
    """

    def __init__(
        self,
        port_to_handler: Dict[int, "ConnectionHandler"],
        pid_to_port: Dict[int, int],
        key_to_bitpos: Dict[str, int],
        util_bin16: Callable[[int], str],
    ):
        self.port_to_handler = port_to_handler
        self.pid_to_port = pid_to_port
        self.key_bits = key_to_bitpos
        self.bin16 = util_bin16

        self._sel_port: Optional[int] = None
        self._pressed_mask: int = 0
        self._lock = threading.Lock()

        # capture the running asyncio loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None  # should not happen in normal use

    # --------------------------------------------------------------------- #
    def start(self) -> None:
        threading.Thread(target=self._pointer_poll_loop, daemon=True).start()
        keyboard.Listener(on_press=self._on_press,
                          on_release=self._on_release).start()

    # --------------------------------------------------------------------- #
    # mouse‑pointer polling thread
    # --------------------------------------------------------------------- #
    def _pointer_poll_loop(self) -> None:
        while True:
            xid = window_under_pointer()
            title = window_title(xid)
            port: Optional[int] = None

            # 1️⃣ fast path — any 5‑digit number in the title
            m = PORT_RE.search(title)
            if m:
                try:
                    port = int(m.group(1))
                except ValueError:
                    port = None

            # 2️⃣ fallback — PID → port mapping
            if port is None:
                pid = window_pid(xid)
                port = self.pid_to_port.get(pid) if pid else None

            # debug
            # print(f"Mouse over window {xid} «{title}» → port {port}")

            with self._lock:
                if port != self._sel_port:
                    # disable previous
                    if self._sel_port in self.port_to_handler:
                        self.port_to_handler[self._sel_port].manual_override = False
                    # enable new
                    if port in self.port_to_handler:
                        self.port_to_handler[port].manual_override = True
                    self._sel_port = port
            time.sleep(POLL_MS / 1000)

    # --------------------------------------------------------------------- #
    # keyboard callbacks (run in pynput threads)
    # --------------------------------------------------------------------- #
    def _bit_for(self, key) -> Optional[int]:
        """
        Convert a pynput key event into the bit‑position defined in
        KEY_BIT_POSITIONS.

        Human‑override mapping
        ----------------------
        ← → ↑ ↓ : LEFT / RIGHT / UP / DOWN
        Z       : Z  (chip / confirm)
        X       : X  (buster / cancel)
        ENTER   : RETURN (start/pause)

        Returns None when the key is not part of the control scheme so it is
        simply ignored.
        """
        name = None

        # alphanumeric keys -------------------------------------------------
        if isinstance(key, keyboard.KeyCode):
            ch = (key.char or "").lower()

            # map physical key -> logical button
            #  z  = A‑button (chip)
            #  x  = X‑button (buster)
            mapping = {"z": "Z", "x": "X"}
            name = mapping.get(ch)

        # arrow & special keys ---------------------------------------------
        elif key == keyboard.Key.up:
            name = "UP"
        elif key == keyboard.Key.down:
            name = "DOWN"
        elif key == keyboard.Key.left:
            name = "LEFT"
        elif key == keyboard.Key.right:
            name = "RIGHT"
        elif key == keyboard.Key.enter:
            name = "RETURN"

        # convert logical button → bit position
        return self.key_bits.get(name)


    def _on_press(self, key):
        bit = self._bit_for(key)
        if bit is not None:
            with self._lock:
                self._pressed_mask |= 1 << bit
            self._forward()

    def _on_release(self, key):
        bit = self._bit_for(key)
        if bit is not None:
            with self._lock:
                self._pressed_mask &= ~(1 << bit)
            self._forward()

    # --------------------------------------------------------------------- #
    # send to ConnectionHandler + store demo
    # --------------------------------------------------------------------- #
    def _forward(self) -> None:
        with self._lock:
            port = self._sel_port
            bits = self._pressed_mask

        if port is None or self._loop is None:
            return

        handler = self.port_to_handler.get(port)
        if not handler or not handler._is_running:
            return

        # 1) schedule key‑press to game
        coro = handler._send_command_internal(
            {"type": "key_press", "key": self.bin16(bits)}
        )
        asyncio.run_coroutine_threadsafe(coro, self._loop)

        # ─────────────────────────────────────────────────────────────
        # 2) store human demonstration  (now as a *valid* PPO sample)
        # ─────────────────────────────────────────────────────────────
        if bits and handler.prev_processed_stacked_frames is not None:
            # 1. Translate pressed keys -> discrete action index
            a_idx_int = utils.bitmask_to_action_index(bits)

            # 2. Get logπ(a|s) & V(s) from the current policy so that the
            #    PPO ratio term is well‑behaved
            if isinstance(handler.strategy, DRLAgentStrategy):
                with torch.no_grad():
                    _, log_p_demo, _, v_demo, _ = handler.strategy.model.get_action_and_value(
                        handler.prev_processed_stacked_frames.to(handler.strategy.dev),
                        handler.prev_processed_game_features.to(handler.strategy.dev),
                        torch.tensor(a_idx_int, device=handler.strategy.dev),
                    )
            else:
                log_p_demo = torch.tensor(0.0)
                v_demo     = torch.tensor(0.0)

            # 3. Push into the experience buffer
            handler.experience_buffer.add(
                handler.prev_processed_stacked_frames,
                handler.prev_processed_game_features,
                torch.tensor(a_idx_int),
                log_p_demo,
                0.0,          # reward will be computed on the next frame as usual
                v_demo,
                torch.tensor(False),
            )

