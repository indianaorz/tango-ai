# GridStateEvaluator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GridStateEvaluator(nn.Module):
    def __init__(self, learning_rate=1e-3, feature_scale=2.0):
        """
        Initializes the GridStateEvaluator model.

        Parameters:
            learning_rate (float): Learning rate for the optimizer.
            feature_scale (float): Scaling factor to amplify additional features.
        """
        super(GridStateEvaluator, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Optional: Batch Normalization for convolutional layers
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        
        # Define fully connected layers for convolutional features
        self.fc_conv = nn.Linear(256 * 1 * 2, 256)
        
        # Define separate fully connected layers for additional features
        self.fc_additional = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Optional: Dropout for regularization
        # self.dropout = nn.Dropout(p=0.5)
        
        # Define fully connected layers after merging features
        self.fc1 = nn.Linear(256 + 64, 256)  # 256 from conv, 64 from additional
        self.fc2 = nn.Linear(256, 1)         # Output single float (logits)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Feature scaling factor
        self.feature_scale = feature_scale

    def forward(self, x, additional_features):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Grid tensor of shape (batch_size, 16, 3, 6)
            additional_features (torch.Tensor): Tensor of shape (batch_size, 4)

        Returns:
            torch.Tensor: Logits (before sigmoid)
        """
        # Apply convolutional layers with activation and pooling
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 32, 3, 6)
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)  # Shape: (batch_size, 32, 2, 3)
        
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 64, 2, 3)
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)  # Shape: (batch_size, 64, 1, 2)
        
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 128, 1, 2)
        x = F.relu(self.conv4(x))  # Shape: (batch_size, 256, 1, 2)
        
        # Optional: Apply Batch Normalization
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten the convolutional output
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 256*1*2) = (batch_size, 512)
        
        # Process convolutional features
        conv_features = F.relu(self.fc_conv(x))  # Shape: (batch_size, 256)
        
        # Scale additional features
        additional_features = additional_features * self.feature_scale
        
        # Process additional features through dedicated layers
        additional_processed = self.fc_additional(additional_features)  # Shape: (batch_size, 64)
        
        # Optional: Apply Dropout
        # conv_features = self.dropout(conv_features)
        # additional_processed = self.dropout(additional_processed)
        
        # Concatenate all features
        combined = torch.cat([conv_features, additional_processed], dim=1)  # Shape: (batch_size, 256 + 64) = (batch_size, 320)
        
        # Fully connected layers with activation
        x = F.relu(self.fc1(combined))  # Shape: (batch_size, 256)
        logits = self.fc2(x)            # Shape: (batch_size, 1), raw logits
        
        return logits

    def get_confidence_score(self, current_data_point):
        """
        Extracts the grid tensor and additional features from current_data_point,
        preprocesses them by scaling additional features, and returns the confidence score from the model.
        
        Parameters:
            current_data_point (dict): A dictionary containing the 'grid' key with tensor data,
                                       'player_shoot_button', 'player_chip_button',
                                       'player_charge', and 'enemy_charge'.
        
        Returns:
            float: The confidence score between 0 and 1.
            None: If grid data or additional features are unavailable or invalid.
        """
        self.eval()  # Set model to evaluation mode
        grid_tensor = current_data_point.get('grid', None)
        player_shoot_button = current_data_point.get('player_shoot_button', 0.0)
        player_chip_button = current_data_point.get('player_chip_button', 0.0)
        player_charge = current_data_point.get('player_charge', 0.0)
        enemy_charge = current_data_point.get('enemy_charge', 0.0)
        
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

        # Scale additional features
        try:
            additional_features = torch.tensor([
                float(player_shoot_button) * self.feature_scale,  # Scale shoot_button
                float(player_chip_button) * self.feature_scale,   # Scale chip_button
                float(player_charge) * self.feature_scale,        # Scale charge
                float(enemy_charge) * self.feature_scale          # Scale enemy_charge
            ], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        except Exception as e:
            print(f"Error processing additional features: {e}")
            return None

        # Move tensors to the appropriate device
        device = next(self.parameters()).device  # Automatically get the device of model parameters
        grid_tensor = grid_tensor.to(device)
        additional_features = additional_features.to(device)

        # Reshape the grid tensor to match model's expected input
        # Expected shape: (batch_size, channels, height, width)
        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)  # Add batch dimension: (1, 16, 3, 6)
        elif grid_tensor.dim() != 4:
            print(f"Invalid grid tensor dimensions: {grid_tensor.dim()}. Expected 3 or 4 dimensions.")
            return None

        # Forward pass through the model to get logits
        with torch.no_grad():
            logits = self.forward(grid_tensor, additional_features)  # Shape: (1, 1)
            confidence = torch.sigmoid(logits).item()  # Apply sigmoid to get probability

        return confidence

    def train_batch(self, grid_experiences, max_grad_norm=1.0):
        """
        Trains the model on a batch of experiences with weighted loss.

        Parameters:
            grid_experiences (list of lists): Each sublist contains [data_point, reward_value].
            max_grad_norm (float): Maximum allowed norm of the gradients (for gradient clipping).

        Returns:
            float: The average loss over the batch.
            int: Number of valid experiences trained on.
            None: If no valid experiences are provided.
        """
        if not grid_experiences:
            print("No experiences provided for training.")
            return None, 0

        # Set model to training mode
        self.train()

        # Lists to hold processed tensors and targets
        grid_tensors = []
        additional_features_list = []
        target_tensors = []
        sample_weights = []
        valid_experiences = 0

        for experience in grid_experiences:
            if len(experience) != 2:
                print(f"Invalid experience format: {experience}. Expected [data_point, reward_value]. Skipping.")
                continue

            data_point, reward = experience
            # Clamp the reward value between -1 and 1
            reward = max(-1, min(reward, 1))
            grid_tensor = data_point.get('grid', None)
            player_shoot_button = data_point.get('player_shoot_button', 0.0)
            player_chip_button = data_point.get('player_chip_button', 0.0)
            player_charge = data_point.get('player_charge', 0.0)
            enemy_charge = data_point.get('enemy_charge', 0.0)

            if grid_tensor is None:
                print("An experience has no grid data. Skipping.")
                continue

            # Ensure the grid tensor is a PyTorch tensor
            if not isinstance(grid_tensor, torch.Tensor):
                try:
                    grid_tensor = torch.tensor(grid_tensor, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting grid data to tensor: {e}. Skipping.")
                    continue

            # Scale additional features
            try:
                additional_features = torch.tensor([
                    float(player_shoot_button) * self.feature_scale,  # Scale shoot_button
                    float(player_chip_button) * self.feature_scale,   # Scale chip_button
                    float(player_charge) * self.feature_scale,        # Scale charge
                    float(enemy_charge) * self.feature_scale          # Scale enemy_charge
                ], dtype=torch.float32)
            except Exception as e:
                print(f"Error processing additional features: {e}. Skipping.")
                continue

            # Move tensors to the appropriate device
            device = next(self.parameters()).device  # Automatically get the device of model parameters
            grid_tensor = grid_tensor.to(device)
            additional_features = additional_features.to(device)

            # Reshape the grid tensor to match model's expected input
            # Expected shape: (batch_size, channels, height, width)
            if grid_tensor.dim() == 3:
                grid_tensor = grid_tensor.unsqueeze(0)  # Add batch dimension: (1, 16, 3, 6)
            elif grid_tensor.dim() != 4:
                print(f"Invalid grid tensor dimensions: {grid_tensor.dim()}. Expected 3 or 4 dimensions. Skipping.")
                continue

            grid_tensors.append(grid_tensor)
            additional_features_list.append(additional_features)
            # Convert reward to target
            target = (reward + 1) / 2
            target_tensors.append(target)

            # Assign higher weights to experiences where shooting or charging is active
            is_shoot_active = float(player_shoot_button) > 0.5  # Threshold as needed
            is_charge_active = float(player_charge) > 0.5        # Threshold as needed
            weight = 2.0 if (is_shoot_active or is_charge_active) else 1.0
            sample_weights.append(weight)

            valid_experiences += 1

        if not grid_tensors:
            print("No valid experiences to train on.")
            return None, 0

        # Stack all grid tensors into a single batch tensor
        input_batch = torch.cat(grid_tensors, dim=0)  # Shape: (batch_size, 16, 3, 6)

        # Stack all additional features
        additional_features_batch = torch.stack(additional_features_list, dim=0)  # Shape: (batch_size, 4)

        # Convert targets to a tensor
        target_batch = torch.tensor(target_tensors, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Convert sample weights to a tensor
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Forward pass
        output = self.forward(input_batch, additional_features_batch)  # Shape: (batch_size, 1)

        # Compute loss with sample weights
        loss = self.loss_fn(output, target_batch)  # BCEWithLogitsLoss expects raw logits
        weighted_loss = loss * sample_weights_tensor  # Apply weights
        loss = weighted_loss.mean()  # Average the weighted loss

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        self.optimizer.step()

        # Calculate average loss
        average_loss = loss.item()

        return average_loss, valid_experiences

    def save_model(self, filepath):
        """
        Saves the model's state dictionary to the specified filepath.

        Parameters:
            filepath (str): Path to save the model.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Loads the model's state dictionary from the specified filepath.

        Parameters:
            filepath (str): Path from which to load the model.
        """
        try:
            self.load_state_dict(torch.load(filepath))
            self.eval()
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model from {filepath}: {e}")
