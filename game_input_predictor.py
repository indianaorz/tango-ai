# game_input_predictor.py
import torch
import torch.nn as nn
import numpy as np

class GameInputPredictor(nn.Module):
    def __init__(self, image_memory=1):
        super(GameInputPredictor, self).__init__()
        self.image_memory = image_memory
        # Updated to use 3D convolutions to capture spatiotemporal features
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self._to_linear = None
        self._get_conv_output_shape((3, image_memory, 160, 240))  # Input shape based on (channels, depth, height, width)
        
        # LSTM layer to capture temporal dependencies in the flattened output
        self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=512, num_layers=1, batch_first=True)
        
        # Fully connected layers for image features
        self.image_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Fully connected layers for additional inputs
        self.additional_fc = nn.Sequential(
            nn.Linear(6*3 + 6*3 + 1, 128),  # Updated to 37 input features
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Final classification layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 + 128 + 2, 256),  # Combine image, additional inputs, and healths
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 16),  # Assuming 16 possible actions
        )

    def _get_conv_output_shape(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.conv_layers(input)
        self._to_linear = int(np.prod(output.shape[1:]))
        print(f"Convolution output size: {output.shape}")
        print(f"Flattened size: {self._to_linear}")

    def forward(self, x, player_grid, enemy_grid, inside_window, player_health, enemy_health):
        # Process image through convolutional layers
        x = self.conv_layers(x)  # Shape: (batch_size, 256, D_out, H_out, W_out)
        x = x.view(x.size(0), -1, self._to_linear)  # Shape: (batch_size, 1, _to_linear)
        x, _ = self.lstm(x)  # Shape: (batch_size, 1, 512)
        x = x[:, -1, :]  # Shape: (batch_size, 512)
        x = self.image_fc(x)  # Shape: (batch_size, 256)

        # Flatten player and enemy grids
        player_grid = player_grid.view(player_grid.size(0), -1)  # Shape: (batch_size, 18)
        enemy_grid = enemy_grid.view(enemy_grid.size(0), -1)    # Shape: (batch_size, 18)
        additional = torch.cat([player_grid, enemy_grid, inside_window], dim=1)  # Shape: (batch_size, 37)

        # Debugging: Verify shapes before passing through additional_fc
        # print(f"Player Grid Shape: {player_grid.shape}")
        # print(f"Enemy Grid Shape: {enemy_grid.shape}")
        # print(f"Inside Window Shape: {inside_window.shape}")
        # print(f"Additional Shape: {additional.shape}")

        additional = self.additional_fc(additional)  # Shape: (batch_size, 128)

        # Concatenate all features
        combined = torch.cat([x, additional, player_health, enemy_health], dim=1)  # Shape: (batch_size, 386)

        # Debugging: Verify shape before passing through fc_layers
        # print(f"Combined Shape: {combined.shape}")

        # Final classification
        output = self.fc_layers(combined)  # Shape: (batch_size, 16)

        return output

