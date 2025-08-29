from __future__ import annotations
from typing import Optional
from PIL import Image
import torchvision.transforms.functional as TF
import torch

def int_to_binary_string(value: int) -> str:
    return format(value, "016b")

def preprocess_frame(frame_pil_image: Optional[Image.Image],
                     height: int, width: int) -> torch.Tensor:
    if frame_pil_image is None:
        return torch.zeros((3, height, width), dtype=torch.float32)
    img = frame_pil_image.convert("RGB")
    img = TF.resize(img, [height, width], antialias=True)
    return TF.to_tensor(img)
