# GridStateEvaluator.py
import torch.nn.functional as F
import torch.nn as nn
import torch

class GridStateEvaluator(nn.Module):
    def __init__(self):
        super(GridStateEvaluator, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 16, 256)  # Adjust based on the input size after conv layers
        self.fc2 = nn.Linear(256, 1)  # Output single float

    def forward(self, x):
        # Apply convolutional layers with activation and pooling
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 16, H, W)
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)      # Shape: (batch_size, 16, ceil(H/2), ceil(W/2))
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 32, ceil(H/2), ceil(W/2))
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)      # Shape: (batch_size, 32, ceil(H/4), ceil(W/4))
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 64, ceil(H/4), ceil(W/4))
        x = F.relu(self.conv4(x))  # Shape: (batch_size, 128, ceil(H/4), ceil(W/4))
        
        # Adaptive pooling to ensure consistent input size for FC layers
        x = F.adaptive_avg_pool2d(x, (3, 16))  # Adjust based on your grid size
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 128*3*16)
        x = F.relu(self.fc1(x))    # Shape: (batch_size, 256)
        x = torch.sigmoid(self.fc2(x))  # Shape: (batch_size, 1), values between 0 and 1
        return x

    def get_confidence_score(self, current_data_point):
        """
        Extracts the grid tensor from current_data_point, preprocesses it,
        and returns the confidence score from the model.
        
        Parameters:
            current_data_point (dict): A dictionary containing the 'grid' key with tensor data.
        
        Returns:
            float: The confidence score between 0 and 1.
            None: If grid data is unavailable or invalid.
        """
        grid_tensor = current_data_point.get('grid', None)
        
        if grid_tensor is None:
            print("No grid data available in current_data_point.")
            return None

        # Ensure the grid tensor is a PyTorch tensor
        if not isinstance(grid_tensor, torch.Tensor):
            try:
                grid_tensor = torch.tensor(grid_tensor, dtype=torch.float32)
            except Exception as e:
                print(f"Error converting grid data to tensor: {e}")
                return None

        # Move tensor to the appropriate device
        device = next(self.parameters()).device  # Automatically get the device of model parameters
        grid_tensor = grid_tensor.to(device)

        # Reshape the tensor to match model's expected input
        # Expected shape: (batch_size, channels, height, width)
        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)  # Add batch dimension
        elif grid_tensor.dim() != 4:
            print(f"Invalid grid tensor dimensions: {grid_tensor.dim()}. Expected 3 or 4 dimensions.")
            return None

        # Forward pass through the model to get confidence score
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            confidence = self.forward(grid_tensor).item()  # Get scalar value

        return confidence
