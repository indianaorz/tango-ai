# inference.py
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
import os
from train import GameInputPredictor  # Import the model class


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py path_to_image checkpoint_path")
        sys.exit(1)

    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    image_memory = 1  # Adjust if needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GameInputPredictor(image_memory=image_memory).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    image_tensor = load_image(image_path, image_memory=image_memory)

    probs = predict(model, image_tensor, device)

    # Print the probabilities
    print("Probabilities for each input (length 16):")
    print(probs)
    # Optionally, threshold probabilities to get binary outputs
    threshold = 0.5
    predicted_inputs = (probs >= threshold).astype(int)
    print("Predicted inputs (0 or 1):")
    print(predicted_inputs)

if __name__ == '__main__':
    main()
