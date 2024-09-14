# capture_utils.py

import subprocess
import mss
from PIL import Image
import torchvision.transforms as T
import re

transform = T.Compose([
    T.Resize((84, 84)),
    T.ToTensor(),
])

def preprocess_image(image):
    image = transform(image)
    return image.numpy()

def capture_window(geometry):
    with mss.mss() as sct:
        sct_img = sct.grab(geometry)
        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return image

def find_game_window(port):
    result = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if str(port) in line:
            window_id = line.split()[0]
            return window_id
    raise Exception(f"No window found with title containing '{port}'")

def get_window_geometry(window_id):
    result = subprocess.run(['xwininfo', '-id', window_id], stdout=subprocess.PIPE, text=True)
    x = y = width = height = 0
    for line in result.stdout.splitlines():
        if "Absolute upper-left X" in line:
            x = int(re.search(r'\d+', line).group())
        elif "Absolute upper-left Y" in line:
            y = int(re.search(r'\d+', line).group())
        elif "Width" in line:
            width = int(re.search(r'\d+', line).group())
        elif "Height" in line:
            height = int(re.search(r'\d+', line).group())
    return {"left": x, "top": y, "width": width, "height": height}

async def send_input_command(writer, command):
    try:
        command_json = json.dumps(command)
        writer.write(command_json.encode() + b'\n')
        await writer.drain()
        print(f"Sent command: {command}")
    except Exception as e:
        print(f"Failed to send command: {e}")
