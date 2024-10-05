# GridStateEvaluator.py
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import get_image_memory
config = {
        # 'include_screen_image': True,
        # 'include_cust_gage': True,
        # 'include_health': True,
        'include_player_beasted_out': True,
        'include_enemy_beasted_out': True,
        'include_player_beasted_over': True,
        'include_enemy_beasted_over': True,
        'include_player_chip': True,
        'include_enemy_chip': True,
        # 'include_player_chip_hand': True,
        # 'player_chip_hand_size': 5,  # Example size
        # 'include_player_folder': True,
        # 'include_enemy_folder': True,
        # 'include_player_custom': True,
        # 'include_enemy_custom': True,
        'include_player_emotion_state': True,
        'include_enemy_emotion_state': True,
        'include_player_used_crosses': True,
        'include_enemy_used_crosses': True,
        'include_player_active_chip': True,
        'include_enemy_active_chip': True,
    }

class GridStateEvaluator(nn.Module):
    def __init__(self, learning_rate=1e-3, config=config):
        super(GridStateEvaluator, self).__init__()
        self.config = config if config is not None else {}

        # Define convolutional layers for grid
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Handle screen_image if included
        if self.config.get('include_screen_image', False):
            self.screen_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
            self.screen_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
            self.screen_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.screen_output_dim = self._compute_screen_output_dim()

        self.additional_features_size = self._compute_additional_features_size()
        fc1_input_dim = 256 * 1 * 2 + self.additional_features_size
        if self.config.get('include_screen_image', False):
            fc1_input_dim += self.screen_output_dim

        self.fc1 = nn.Linear(fc1_input_dim, 1024)  # Adjusted input dimension
        #hidden layers
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1)  # Output single float

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _compute_additional_features_size(self):
        size = 4  # The default 4 features: player_shoot_button, player_chip_button, player_charge, enemy_charge

        if self.config.get('include_cust_gage', False):
            size += 1  # cust_gage

        if self.config.get('include_health', False):
            size += 2  # player_health, enemy_health

        if self.config.get('include_player_beasted_out', False):
            size += 1  # player_beasted_out

        if self.config.get('include_enemy_beasted_out', False):
            size += 1  # enemy_beasted_out

        if self.config.get('include_player_beasted_over', False):
            size += 1  # player_beasted_over

        if self.config.get('include_enemy_beasted_over', False):
            size += 1  # enemy_beasted_over

        if self.config.get('include_player_chip', False):
            size += 401  # player_chip

        if self.config.get('include_enemy_chip', False):
            size += 401  # enemy_chip
            
        # Add sizes for active chips
        if self.config.get('include_player_active_chip', False):
            size += 401  # player_active_chip

        if self.config.get('include_enemy_active_chip', False):
            size += 401  # enemy_active_chip

        if self.config.get('include_player_chip_hand', False):
            size += self.config.get('player_chip_hand_size', 0)  # player_chip_hand_size specified in config

        if self.config.get('include_player_folder', False):
            size += 12930  # player_folder

        if self.config.get('include_enemy_folder', False):
            size += 12930  # enemy_folder

        if self.config.get('include_player_custom', False):
            size += 200  # player_custom

        if self.config.get('include_enemy_custom', False):
            size += 200  # enemy_custom

        if self.config.get('include_player_emotion_state', False):
            size += 27  # player_emotion_state

        if self.config.get('include_enemy_emotion_state', False):
            size += 27  # enemy_emotion_state

        if self.config.get('include_player_used_crosses', False):
            size += 10  # player_used_crosses

        if self.config.get('include_enemy_used_crosses', False):
            size += 10  # enemy_used_crosses

        return size

    def _compute_screen_output_dim(self):
        # Compute output dimensions after convolutions for screen_image
        # Assuming input size: (batch_size, 1, 160, 240)
        # After conv1 (kernel_size=5, stride=1, padding=0): (batch_size, 16, 156, 236)
        # After conv2 (kernel_size=5, stride=1, padding=0): (batch_size, 32, 152, 232)
        # After pooling: (batch_size, 32, 76, 116)
        conv1_out = (160 - 5 + 1)  # 156
        conv2_out = (conv1_out - 5 + 1)  # 152
        pool_out_h = conv2_out // 2  # 76
        pool_out_w = (240 - 5 + 1 - 5 + 1) // 2  # Simplified estimation: 116
        return 32 * pool_out_h * pool_out_w  # 32 * 76 * 116 = 281,  32 * 76 * 116 = 281,  32*76=2432, 2432*116= 282,112

    def forward(self, x, additional_features, screen_image=None):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Grid tensor of shape (batch_size, 16, 3, 6)
            additional_features (torch.Tensor): Tensor of shape (batch_size, additional_features_size)
            screen_image (torch.Tensor, optional): Tensor of shape (batch_size, 1, 160, 240)

        Returns:
            torch.Tensor: Confidence scores between 0 and 1
        """
        # Process grid through convolutional layers
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 32, 3, 6)
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)  # Shape: (batch_size, 32, 2, 3)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 64, 2, 3)
        x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)  # Shape: (batch_size, 64, 1, 2)
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 128, 1, 2)
        x = F.relu(self.conv4(x))  # Shape: (batch_size, 256, 1, 2)

        # Flatten grid features
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 512)

        # Process screen_image if included
        if self.config.get('include_screen_image', False) and screen_image is not None:
            screen_x = F.relu(self.screen_conv1(screen_image))  # Shape depends on conv parameters
            screen_x = F.relu(self.screen_conv2(screen_x))
            screen_x = self.screen_pool(screen_x)
            screen_x = screen_x.view(screen_x.size(0), -1)  # Flatten
            x = torch.cat([x, screen_x], dim=1)

        # Concatenate additional features
        x = torch.cat([x, additional_features], dim=1)  # Shape: (batch_size, fc1_input_dim)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        #hidden layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x

    def _extract_additional_features(self, current_data_point):
        features = []

        # Default features
        features.extend([
            float(current_data_point.get('player_shoot_button', 0.0)),
            float(current_data_point.get('player_chip_button', 0.0)),
            float(current_data_point.get('player_charge', 0.0)),
            float(current_data_point.get('enemy_charge', 0.0))
        ])

        # Optional features based on config
        if self.config.get('include_cust_gage', False):
            features.append(float(current_data_point.get('cust_gage', 0.0)))

        if self.config.get('include_health', False):
            features.extend([
                float(current_data_point.get('player_health', 0.0)),
                float(current_data_point.get('enemy_health', 0.0))
            ])

        if self.config.get('include_player_beasted_out', False):
            features.append(float(current_data_point.get('player_beasted_out', 0.0)))

        if self.config.get('include_enemy_beasted_out', False):
            features.append(float(current_data_point.get('enemy_beasted_out', 0.0)))

        if self.config.get('include_player_beasted_over', False):
            features.append(float(current_data_point.get('player_beasted_over', 0.0)))

        if self.config.get('include_enemy_beasted_over', False):
            features.append(float(current_data_point.get('enemy_beasted_over', 0.0)))

        if self.config.get('include_player_chip', False):
            player_chip = current_data_point.get('player_chip', None)
            if player_chip is not None:
                if isinstance(player_chip, torch.Tensor):
                    features.extend(player_chip.view(-1).tolist())
                else:
                    features.extend(list(player_chip))
            else:
                features.extend([0.0] * 401)  # Default padding if missing

        if self.config.get('include_enemy_chip', False):
            enemy_chip = current_data_point.get('enemy_chip', None)
            if enemy_chip is not None:
                if isinstance(enemy_chip, torch.Tensor):
                    features.extend(enemy_chip.view(-1).tolist())
                else:
                    features.extend(list(enemy_chip))
            else:
                features.extend([0.0] * 401)  # Default padding if missing

        if self.config.get('include_player_active_chip', False):
            player_active_chip = current_data_point.get('player_active_chip', None)
            if player_active_chip is not None:
                if isinstance(player_active_chip, torch.Tensor):
                    features.extend(player_active_chip.view(-1).tolist())
                else:
                    features.extend(list(player_active_chip))
            else:
                features.extend([0.0] * 401)
        
        if self.config.get('include_enemy_active_chip', False):
            enemy_active_chip = current_data_point.get('enemy_active_chip', None)
            if enemy_active_chip is not None:
                if isinstance(enemy_active_chip, torch.Tensor):
                    features.extend(enemy_active_chip.view(-1).tolist())
                else:
                    features.extend(list(enemy_active_chip))
            else:
                features.extend([0.0] * 401)

        if self.config.get('include_player_chip_hand', False):
            player_chip_hand = current_data_point.get('player_chip_hand', None)
            chip_hand_size = self.config.get('player_chip_hand_size', 0)
            if player_chip_hand is not None:
                if isinstance(player_chip_hand, torch.Tensor):
                    features.extend(player_chip_hand.view(-1).tolist())
                else:
                    features.extend(list(player_chip_hand))
            else:
                features.extend([0.0] * chip_hand_size)  # Default padding if missing

        if self.config.get('include_player_folder', False):
            player_folder = current_data_point.get('player_folder', None)
            if player_folder is not None:
                if isinstance(player_folder, torch.Tensor):
                    features.extend(player_folder.view(-1).tolist())
                else:
                    features.extend(list(player_folder))
            else:
                features.extend([0.0] * 12930)  # Default padding if missing

        if self.config.get('include_enemy_folder', False):
            enemy_folder = current_data_point.get('enemy_folder', None)
            if enemy_folder is not None:
                if isinstance(enemy_folder, torch.Tensor):
                    features.extend(enemy_folder.view(-1).tolist())
                else:
                    features.extend(list(enemy_folder))
            else:
                features.extend([0.0] * 12930)  # Default padding if missing

        if self.config.get('include_player_custom', False):
            player_custom = current_data_point.get('player_custom', None)
            if player_custom is not None:
                if isinstance(player_custom, torch.Tensor):
                    features.extend(player_custom.view(-1).tolist())
                else:
                    features.extend(list(player_custom))
            else:
                features.extend([0.0] * 200)  # Default padding if missing

        if self.config.get('include_enemy_custom', False):
            enemy_custom = current_data_point.get('enemy_custom', None)
            if enemy_custom is not None:
                if isinstance(enemy_custom, torch.Tensor):
                    features.extend(enemy_custom.view(-1).tolist())
                else:
                    features.extend(list(enemy_custom))
            else:
                features.extend([0.0] * 200)  # Default padding if missing

        if self.config.get('include_player_emotion_state', False):
            player_emotion_state = current_data_point.get('player_emotion_state', None)
            if player_emotion_state is not None:
                if isinstance(player_emotion_state, torch.Tensor):
                    features.extend(player_emotion_state.view(-1).tolist())
                else:
                    features.extend(list(player_emotion_state))
            else:
                features.extend([0.0] * 27)  # Default padding if missing

        if self.config.get('include_enemy_emotion_state', False):
            enemy_emotion_state = current_data_point.get('enemy_emotion_state', None)
            if enemy_emotion_state is not None:
                if isinstance(enemy_emotion_state, torch.Tensor):
                    features.extend(enemy_emotion_state.view(-1).tolist())
                else:
                    features.extend(list(enemy_emotion_state))
            else:
                features.extend([0.0] * 27)  # Default padding if missing

        if self.config.get('include_player_used_crosses', False):
            player_used_crosses = current_data_point.get('player_used_crosses', None)
            if player_used_crosses is not None:
                if isinstance(player_used_crosses, torch.Tensor):
                    features.extend(player_used_crosses.view(-1).tolist())
                else:
                    features.extend(list(player_used_crosses))
            else:
                features.extend([0.0] * 10)  # Default padding if missing

        if self.config.get('include_enemy_used_crosses', False):
            enemy_used_crosses = current_data_point.get('enemy_used_crosses', None)
            if enemy_used_crosses is not None:
                if isinstance(enemy_used_crosses, torch.Tensor):
                    features.extend(enemy_used_crosses.view(-1).tolist())
                else:
                    features.extend(list(enemy_used_crosses))
            else:
                features.extend([0.0] * 10)  # Default padding if missing

        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, additional_features_size)
        return features_tensor

    def get_confidence_score(self, current_data_point):
        """
        Extracts the grid tensor and additional features from current_data_point,
        preprocesses them, and returns the confidence score from the model.

        Parameters:
            current_data_point (dict): A dictionary containing various keys including 'grid'.

        Returns:
            float: The confidence score between 0 and 1.
            None: If grid data or required additional features are unavailable or invalid.
        """
        self.eval()  # Set model to evaluation mode
        grid_tensor = current_data_point.get('grid', None)
        if grid_tensor is None:
            print("No grid data available in current_data_point.")
            return None

        screen_image = None
        if self.config.get('include_screen_image', False):
            screen_image = current_data_point.get('screen_image', None)
            if screen_image is None:
                print("No screen_image data available in current_data_point.")
                return None

        # Ensure tensors are correct
        grid_tensor = grid_tensor if isinstance(grid_tensor, torch.Tensor) else torch.tensor(grid_tensor, dtype=torch.float32)
        if screen_image is not None:
            screen_image = screen_image if isinstance(screen_image, torch.Tensor) else torch.tensor(screen_image, dtype=torch.float32)

        additional_features = self._extract_additional_features(current_data_point)

        # Move to device
        device = next(self.parameters()).device
        grid_tensor = grid_tensor.to(device)
        additional_features = additional_features.to(device)
        if screen_image is not None:
            screen_image = screen_image.to(device)

        # Reshape tensors
        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)  # Shape: (1, 16, 3, 6)
        elif grid_tensor.dim() != 4:
            print(f"Invalid grid tensor dimensions: {grid_tensor.dim()}. Expected 3 or 4 dimensions.")
            return None

        if screen_image is not None and screen_image.dim() == 3:
            screen_image = screen_image.unsqueeze(0)  # Shape: (1, 1, 160, 240)

        # Forward pass
        with torch.no_grad():
            confidence = self.forward(grid_tensor, additional_features, screen_image).item()

        return confidence

    def train_batch(self, grid_experiences, max_grad_norm=1.0):
        """
        Trains the model on a batch of experiences.

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

        self.train()  # Set model to training mode

        grid_tensors = []
        additional_features_list = []
        screen_images = []
        target_tensors = []
        valid_experiences = 0

        for experience in grid_experiences:
            if len(experience) != 2:
                print(f"Invalid experience format: {experience}. Expected [data_point, reward_value]. Skipping.")
                continue

            data_point, reward = experience
            # Clamp the reward value between -1 and 1
            reward = max(-1, min(reward, 1))

            grid_tensor = data_point.get('grid', None)
            if grid_tensor is None:
                print("An experience has no grid data. Skipping.")
                continue

            screen_image = None
            if self.config.get('include_screen_image', False):
                screen_image = data_point.get('screen_image', None)
                if screen_image is None:
                    print("An experience has no screen_image data. Skipping.")
                    continue

            # Ensure grid_tensor is a tensor
            if not isinstance(grid_tensor, torch.Tensor):
                try:
                    grid_tensor = torch.tensor(grid_tensor, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting grid data to tensor: {e}. Skipping.")
                    continue

            # Ensure screen_image is a tensor if included
            if screen_image is not None and not isinstance(screen_image, torch.Tensor):
                try:
                    screen_image = torch.tensor(screen_image, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting screen_image to tensor: {e}. Skipping.")
                    continue

            # Extract additional features
            try:
                additional_features = self._extract_additional_features(data_point)
            except Exception as e:
                print(f"Error extracting additional features: {e}. Skipping.")
                continue

            # Move tensors to device
            device = next(self.parameters()).device
            grid_tensor = grid_tensor.to(device)
            additional_features = additional_features.to(device)
            if screen_image is not None:
                screen_image = screen_image.to(device)

            # Reshape tensors
            if grid_tensor.dim() == 3:
                grid_tensor = grid_tensor.unsqueeze(0)  # Shape: (1, 16, 3, 6)
            elif grid_tensor.dim() != 4:
                print(f"Invalid grid tensor dimensions: {grid_tensor.dim()}. Expected 3 or 4 dimensions. Skipping.")
                continue

            if screen_image is not None and screen_image.dim() == 3:
                screen_image = screen_image.unsqueeze(0)  # Shape: (1, 1, 160, 240)

            grid_tensors.append(grid_tensor)
            additional_features_list.append(additional_features)
            if screen_image is not None:
                screen_images.append(screen_image)
            # Convert reward to target (scale from [-1,1] to [0,1])
            target = (reward + 1) / 2
            target_tensors.append(target)

            valid_experiences += 1

        if not grid_tensors:
            print("No valid experiences to train on.")
            return None, 0

        # Stack all grid tensors into a single batch tensor
        input_batch = torch.cat(grid_tensors, dim=0)  # Shape: (batch_size, 16, 3, 6)

        # Stack all additional features
        additional_features_batch = torch.cat(additional_features_list, dim=0)  # Shape: (batch_size, additional_features_size)

        # Stack all screen images if included
        if self.config.get('include_screen_image', False):
            screen_image_batch = torch.cat(screen_images, dim=0)  # Shape: (batch_size, 1, 160, 240)
        else:
            screen_image_batch = None

        # Convert targets to a tensor
        target_batch = torch.tensor(target_tensors, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Forward pass
        output = self.forward(input_batch, additional_features_batch, screen_image_batch)

        # Compute loss
        loss = self.loss_fn(output, target_batch)

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        self.optimizer.step()

        # Calculate average loss
        average_loss = loss.item()

        return average_loss, valid_experiences
