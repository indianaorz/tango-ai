import asyncio
import json
import traceback
from typing import Dict, Any, Optional
from io import BytesIO
import base64

from PIL import Image

from actions import int_to_binary_string
from utils import preprocess_frame

class ConnectionHandler:
    '''
    Clean runtime-only handler:
      - Opens TCP to one game instance
      - Requests frames at CONTROL_FPS
      - Parses screen_image payloads
      - Calls a pluggable Controller to produce a key mask
      - Sends key_press back to the instance
    No training/inference here.
    '''
    def __init__(self, instance_config: Dict[str, Any], controller,
                 control_fps: int):
        self.cfg = instance_config
        self.controller = controller
        self.interval = 1.0 / max(1, control_fps)

        self.port    = instance_config['port']
        self.address = instance_config['address']
        self.name    = instance_config['name']

        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def _send(self, command: Dict[str, Any]) -> bool:
        if self.writer and not self.writer.is_closing():
            try:
                self.writer.write(json.dumps(command).encode() + b'\n')
                await self.writer.drain()
                return True
            except (ConnectionResetError, BrokenPipeError):
                print(f'Port {self.port} ({self.name}): Connection closed while sending.')
            except Exception as e:
                print(f'Port {self.port} ({self.name}): Failed to send command: {e}')
        else:
            print(f'Port {self.port} ({self.name}): Writer not available or closing.')
        self._is_running = False
        return False

    async def _request_screen(self) -> bool:
        return await self._send({'type': 'request_screen', 'key': ''})

    async def _message_processor_loop(self):
        json_buffer = ''
        try:
            while self._is_running:
                if not self.reader or self.reader.at_eof():
                    print(f'Port {self.port} ({self.name}): reader closed/EOF – leaving loop.')
                    break
                try:
                    data_chunk = await asyncio.wait_for(self.reader.read(8192), timeout=max(0.2, self.interval * 5))
                except asyncio.TimeoutError:
                    continue
                if not data_chunk:
                    print(f'Port {self.port} ({self.name}): connection closed by peer.')
                    break
                json_buffer += data_chunk.decode(errors='ignore')

                while '\n' in json_buffer and self._is_running:
                    msg_str, json_buffer = json_buffer.split('\n', 1)
                    msg_str = msg_str.strip()
                    if not msg_str:
                        continue
                    try:
                        parsed = json.loads(msg_str)
                    except json.JSONDecodeError:
                        print(f'Port {self.port}: bad JSON → {msg_str[:120]}…')
                        continue
                    if parsed.get('event') != 'screen_image':
                        continue

                    details = parsed.get('details')
                    if isinstance(details, str):
                        try:
                            data = json.loads(details)
                        except json.JSONDecodeError:
                            print(f'Port {self.port}: bad detail JSON → {details[:120]}…')
                            continue
                    elif isinstance(details, dict):
                        data = details
                    else:
                        continue

                    # Extract image (optional) and pass-through game data
                    pil_img = None
                    b64 = data.get('image')
                    if b64:
                        try:
                            pil_img = Image.open(BytesIO(base64.b64decode(b64)))
                        except Exception:
                            pil_img = None

                    # Controller decides the key mask for this frame
                    try:
                        mask_int = self.controller.decide_action(self.port, data, pil_img)
                    except Exception as e:
                        print(f'Port {self.port}: controller error → {e}')
                        mask_int = 0

                    # Send command
                    if not await self._send({'type':'key_press', 'key': int_to_binary_string(mask_int)}):
                        break

        except (ConnectionResetError, BrokenPipeError):
            print(f'Port {self.port} ({self.name}): connection reset/broken.')
        except asyncio.CancelledError:
            print(f'Port {self.port} ({self.name}): processor task cancelled.')
        except Exception as e:
            print(f'Port {self.port}: unexpected error → {e}\n{traceback.format_exc()}')
        finally:
            self._is_running = False
            print(f'Port {self.port} ({self.name}): processor loop ended.')

    async def start(self, max_retries: int = 0, retry_base_delay: float = 0.5, retry_max_delay: float = 5.0):
        self._is_running = True
        self.controller.reset_state(self.port)
        attempt, delay = 0, retry_base_delay

        # Connect loop
        while self._is_running:
            try:
                print(f'Connecting to {self.name} at {self.address}:{self.port} (try {attempt+1})')
                self.reader, self.writer = await asyncio.open_connection(self.address, self.port)
                print(f'Connected to {self.name} (Port {self.port})')
                break
            except Exception as e:
                print(f'Port {self.port} ({self.name}): connect failed: {e}')
                attempt += 1
                if max_retries and attempt >= max_retries:
                    self._is_running = False
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, retry_max_delay)

        if not self._is_running:
            return

        # Start background processor
        self.message_processor_task = asyncio.create_task(self._message_processor_loop())

        # Request frames at CONTROL_FPS
        try:
            while self._is_running:
                if not await self._request_screen():
                    break
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._is_running = False
            if self.message_processor_task and not self.message_processor_task.done():
                self.message_processor_task.cancel()
                try:
                    await self.message_processor_task
                except asyncio.CancelledError:
                    pass
            if self.writer:
                try:
                    if not self.writer.is_closing():
                        self.writer.close()
                    await self.writer.wait_closed()
                except Exception:
                    pass
            print(f'Connection handler for {self.name} (Port {self.port}) shut down.')
