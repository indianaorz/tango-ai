import torch
import torch.nn as nn
import numpy as np

class GameInputPredictor(nn.Module):
    def __init__(self, image_memory=1, config=None):
        super(GameInputPredictor, self).__init__()
        self.image_memory = image_memory
        
        # Extract configuration for input features
        input_features = config.get('input_features', {})
        self.include_image = input_features.get('include_image', True)
        self.include_position = input_features.get('include_position', True)
        self.position_type = input_features.get('position_type', 'grid')
        self.include_player_charge = input_features.get('include_player_charge', False)
        self.include_enemy_charge = input_features.get('include_enemy_charge', False)
        self.temporal_charge = input_features.get('temporal_charge', 0)
        self.include_temporal_charge = self.temporal_charge > 0
        self.input_memory = input_features.get('input_memory', 0)
        self.include_previous_inputs = self.input_memory > 0
        self.health_memory_size = input_features.get('health_memory', 0)
        self.include_health_memory = self.health_memory_size > 0

        
        # Image processing layers
        if self.include_image:
            self.conv_layers = nn.Sequential(
                nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(32, affine=True),
                nn.ReLU(),
                nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(64, affine=True),
                nn.ReLU(),
                nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(128, affine=True),
                nn.ReLU(),
                nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.InstanceNorm3d(256, affine=True),
                nn.ReLU(),
            )
            self._to_linear = None
            self._get_conv_output_shape((3, image_memory, 160, 240))  # Adjust the input size as needed
            
            # LSTM layer to capture temporal dependencies in the flattened output
            self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=512, num_layers=1, batch_first=True)
            
            # Fully connected layers for image features
            self.image_fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        
        # Additional inputs processing
        additional_input_size = 0
        
        if self.include_position:
            if self.position_type == 'float':
                additional_input_size += 4  # player_x, player_y, enemy_x, enemy_y
            elif self.position_type == 'grid':
                additional_input_size += 36  # 6x3 grids concatenated
            else:
                raise ValueError(f"Unknown position_type: {self.position_type}")
        
        if self.include_temporal_charge:
            additional_input_size += 64 * 2  # player and enemy temporal charge features

            # Additionally, include scalar player_charge and enemy_charge
            if self.include_player_charge:
                additional_input_size += 1

            if self.include_enemy_charge:
                additional_input_size += 1
        else:
            if self.include_player_charge:
                additional_input_size += 1

            if self.include_enemy_charge:
                additional_input_size += 1
        
        # Define separate LSTM layers for player and enemy temporal charges
        if self.include_temporal_charge:
            self.player_charge_lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
            self.enemy_charge_lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        
        # Process previous inputs
        if self.include_previous_inputs:
            previous_inputs_size = self.input_memory * 16
            additional_input_size += 128  # We'll reduce the size using an FC layer

            # Define a fully connected layer to reduce previous inputs
            self.previous_inputs_fc = nn.Sequential(
                nn.Linear(previous_inputs_size, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else:
            self.previous_inputs_fc = None

         # Process health memory
        if self.include_health_memory:
            health_memory_input_size = self.health_memory_size * 2  # player and enemy health
            additional_input_size += 128  # We'll reduce the size using an FC layer
            
            # Define a fully connected layer to reduce health memory
            self.health_memory_fc = nn.Sequential(
                nn.Linear(health_memory_input_size, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else:
            self.health_memory_fc = None

        # Fully connected layers for additional inputs
        if additional_input_size > 0:
            self.additional_fc = nn.Sequential(
                nn.Linear(additional_input_size, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else:
            self.additional_fc = None

        # Final classification layers
        final_input_size = 0
        if self.include_image:
            final_input_size += 256
        if self.additional_fc is not None:
            final_input_size += 128
        # If you plan to include other features directly, add their sizes here
        
        # Assuming 'input' is a bitstring of length 16
        self.fc_layers = nn.Sequential(
            nn.Linear(final_input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 16),  # Assuming 16 possible actions
        )
        print(f"Final input size: {final_input_size}")

    def _get_conv_output_shape(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.conv_layers(input)
        self._to_linear = int(np.prod(output.shape[1:]))
        print(f"Convolution output size: {output.shape}")
        print(f"Flattened size: {self._to_linear}")

    def forward(self, x, position=None, player_charge=None, enemy_charge=None,
                player_charge_temporal=None, enemy_charge_temporal=None, previous_inputs=None,
                health_memory=None):
        features = []
        
        if self.include_image:
            # Process image through convolutional layers
            x = self.conv_layers(x)  # Shape: (batch_size, 256, D_out, H_out, W_out)
            x = x.view(x.size(0), -1, self._to_linear)  # Shape: (batch_size, sequence_length, _to_linear)
            x, _ = self.lstm(x)  # Shape: (batch_size, sequence_length, 512)
            x = x[:, -1, :]  # Take the last time step
            x = self.image_fc(x)  # Shape: (batch_size, 256)
            features.append(x)
        
        if self.additional_fc is not None:
            additional = []
            if self.include_position:
                if self.position_type == 'float':
                    additional.append(position)  # Shape: (batch_size, 4)
                elif self.position_type == 'grid':
                    additional.append(position)  # Shape: (batch_size, 36)
            
            # Process temporal charge sequences
            if self.include_temporal_charge:
                if player_charge_temporal is not None:
                    player_charge_seq = player_charge_temporal.unsqueeze(-1)  # Shape: (batch_size, temporal_charge, 1)
                    _, (player_hidden, _) = self.player_charge_lstm(player_charge_seq)
                    player_charge_features = player_hidden[-1]  # Shape: (batch_size, 64)
                else:
                    player_charge_features = torch.zeros((x.size(0), 64), device=x.device)

                if enemy_charge_temporal is not None:
                    enemy_charge_seq = enemy_charge_temporal.unsqueeze(-1)  # Shape: (batch_size, temporal_charge, 1)
                    _, (enemy_hidden, _) = self.enemy_charge_lstm(enemy_charge_seq)
                    enemy_charge_features = enemy_hidden[-1]  # Shape: (batch_size, 64)
                else:
                    enemy_charge_features = torch.zeros((x.size(0), 64), device=x.device)

                temporal_charge_features = torch.cat((player_charge_features, enemy_charge_features), dim=1)  # Shape: (batch_size, 128)
                additional.append(temporal_charge_features)  # Add to additional features

                # Additionally, process scalar player_charge if included
                if self.include_player_charge and player_charge is not None:
                    if player_charge.dim() == 1:
                        player_charge = player_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                    additional.append(player_charge)

                if self.include_enemy_charge and enemy_charge is not None:
                    if enemy_charge.dim() == 1:
                        enemy_charge = enemy_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                    additional.append(enemy_charge)

            else:
                # Process scalar charges if temporal_charge is not included
                if self.include_player_charge and player_charge is not None:
                    if player_charge.dim() == 1:
                        player_charge = player_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                    additional.append(player_charge)

                if self.include_enemy_charge and enemy_charge is not None:
                    if enemy_charge.dim() == 1:
                        enemy_charge = enemy_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                    additional.append(enemy_charge)

            # Process previous inputs
            if self.include_previous_inputs and previous_inputs is not None:
                previous_inputs_features = self.previous_inputs_fc(previous_inputs)
                additional.append(previous_inputs_features)

            # Process health memory
            if self.include_health_memory and health_memory is not None:
                health_memory_features = self.health_memory_fc(health_memory)
                additional.append(health_memory_features)

            # Concatenate all additional features
            if additional:
                additional = torch.cat(additional, dim=1)
                additional = self.additional_fc(additional)
                features.append(additional)
        
        # Concatenate all features
        if features:
            combined = torch.cat(features, dim=1)
        else:
            raise ValueError("No features to combine. Check configuration.")
        
        # Final classification
        output = self.fc_layers(combined)  # Shape: (batch_size, 16)
        
        return output
