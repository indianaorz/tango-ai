import torch

ckpt = torch.load("weights/ng.pt", map_location="cpu")
if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

# Look for vision encoder dims (e.g., first layer norm)
for key in state_dict:
    if "vision_encoder.encoder.layers.0.layer_norm1.weight" in key:
        print(f"Vision hidden size: {state_dict[key].shape[0]}")
        break