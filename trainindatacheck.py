import torch
import glob
from torchvision.utils import save_image

# Load a cached file
files = glob.glob("data/dataset_cached/*.pt")
if files:
    data = torch.load(files[0])
    # Get the first frame [3, 256, 256]
    img_tensor = data["frames"][300].float().div(255.0)
    save_image(img_tensor, "debug_training_sample.png")
    print("📸 Saved debug_training_sample.png")
else:
    print("No cached files found.")