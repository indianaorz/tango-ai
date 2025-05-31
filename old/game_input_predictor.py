import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Helper Classes for Attention Mechanisms

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module to focus on important regions within each frame.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Forward pass for Spatial Attention.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, C, H, W)
        
        Returns:
            Tensor: Output tensor after applying spatial attention
        """
        batch, channel, height, width = x.size()
        
        # Apply first convolution
        f = self.conv1(x)  # Shape: (batch_size, C//8, H, W)
        
        # Apply second convolution
        f = self.conv2(f)  # Shape: (batch_size, 1, H, W)
        
        # Reshape for softmax: (batch_size, 1, H*W)
        f = f.view(batch, 1, -1)  # Shape: (batch_size, 1, H*W)
        
        # Apply softmax to compute attention weights
        attention_map = self.softmax(f)  # Shape: (batch_size, 1, H*W)
        
        # Reshape back to (batch_size, 1, H, W)
        attention_map = attention_map.view(batch, 1, height, width)  # Shape: (batch_size, 1, H, W)
        
        # Multiply input by attention map
        out = x * attention_map  # Shape: (batch_size, C, H, W)
        
        return out

class BilinearAttention(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, attention_dim):
        super(BilinearAttention, self).__init__()
        self.attention = nn.Bilinear(embed_dim1, embed_dim2, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))
    
    def forward(self, x1, x2):
        combined = self.attention(x1, x2)  # (batch_size, attention_dim)
        scores = torch.matmul(combined, self.context_vector)  # (batch_size)
        # If you have multiple attention "slots", adjust scores accordingly
        # For single scalar attention per example, consider using sigmoid
        attention_weights = torch.sigmoid(scores)  # (batch_size)
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1)
        return attention_weights

    
    
class GameInputPredictor(nn.Module):
    def __init__(self, image_memory=1, config=None, scale = 0.25):
        super(GameInputPredictor, self).__init__()
        self.image_memory = image_memory
        
        # Helper function to ensure dimensions are integers
        def scaled_int(x):
            return int(round(x * scale))
        
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
        self.include_player_chip = input_features.get('include_player_chip', False)
        self.include_enemy_chip = input_features.get('include_enemy_chip', False)
        
        # Image processing layers with Spatial Attention and Transformer Encoder
        if self.include_image:
            self.conv_layers = nn.Sequential(
                nn.Conv3d(3, scaled_int(64), kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(scaled_int(64), affine=True),
                nn.ReLU(),
                nn.Conv3d(scaled_int(64), scaled_int(128), kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(scaled_int(128), affine=True),
                nn.ReLU(),
                nn.Conv3d(scaled_int(128), scaled_int(256), kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.InstanceNorm3d(scaled_int(256), affine=True),
                nn.ReLU(),
                nn.Conv3d(scaled_int(256), scaled_int(512), kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.InstanceNorm3d(scaled_int(512), affine=True),
                nn.ReLU(),
                nn.Conv3d(scaled_int(512), scaled_int(1024), kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.InstanceNorm3d(scaled_int(1024), affine=True),
                nn.ReLU(),
            )

            self.positional_encoding = nn.Parameter(torch.randn(1, image_memory, scaled_int(512 )))


            self.spatial_attention = SpatialAttention(scaled_int(1024))  # Correct in_channels
            self._to_linear = None
            self._get_conv_output_shape((3, image_memory, 160, 240))  # Adjust the input size as needed
            
            # Linear projection to reduce feature dimensions before Transformer
            self.transformer_input_projection = nn.Linear(self._to_linear, scaled_int(512))
            
            # Transformer Encoder for image features
            encoder_layers = nn.TransformerEncoderLayer(d_model=scaled_int(512), nhead=16, dim_feedforward=4096)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=12)

            
            # Fully connected layers for image features
            self.image_fc = nn.Sequential(
                nn.Linear(scaled_int(512), scaled_int(512)),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        
        # Additional inputs processing
        additional_input_size = 0
        
        if self.include_position:
            if self.position_type == 'float':
                additional_input_size += 4  # player_x, player_y, enemy_x, enemy_y
            elif self.position_type == 'grid':
                additional_input_size += 36  # 6x6 grids concatenated
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
        
        # Define Transformer Encoders for player and enemy temporal charges
        if self.include_temporal_charge:
            self.player_charge_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=1, nhead=1, dim_feedforward=64),
                num_layers=1
            )
            self.enemy_charge_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=1, nhead=1, dim_feedforward=64),
                num_layers=1
            )
        
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

        # Process health memory with Transformer Encoder
        if self.include_health_memory:
            health_memory_input_size = self.health_memory_size * 2  # player and enemy health
            additional_input_size += 128  # We'll reduce the size using an FC layer
            
            # Transformer Encoder for health sequences
            self.health_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=2, nhead=1, dim_feedforward=64),
                num_layers=2
            )
            
            # Fully connected layer to process health features
            self.health_fc = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )
        else:
            self.health_transformer = None
            self.health_fc = None
        
        # Chip Embedding Layers
        if self.include_player_chip:
            self.player_chip_embedding = nn.Embedding(num_embeddings=400, embedding_dim=64)
            additional_input_size += 64  # Embedding dimension
        if self.include_enemy_chip:
            self.enemy_chip_embedding = nn.Embedding(num_embeddings=400, embedding_dim=64)
            additional_input_size += 64  # Embedding dimension
        
        # Bilinear Attention for Chip Interactions
        if self.include_player_chip and self.include_enemy_chip:
            self.chip_attention = BilinearAttention(embed_dim1=64, embed_dim2=64, attention_dim=128)
            additional_input_size += 1  # Attention weight
        else:
            self.chip_attention = None
        
        # Fully connected layers for additional inputs
        # Define additional_fc correctly
        self.additional_fc = nn.Sequential(
            nn.Linear(328, scaled_int(512)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(scaled_int(512), scaled_int(256)),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Update final_input_size accordingly if `additional_fc` output changes
        final_input_size = scaled_int(768) # Example: 256 from image features + 256 from additional features

        self.fc_layers = nn.Sequential(
            nn.Linear(final_input_size, scaled_int(512)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(scaled_int(512), scaled_int(256)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(scaled_int(256), 16),  # Final output layer
        )

        print(f"Final input size: {final_input_size}")


    def _get_conv_output_shape(self, shape):
        """
        Helper function to determine the output size after convolutional layers.
        
        Args:
            shape (tuple): Shape of the input tensor (C, D, H, W)
        """
        bs = 1
        input = torch.zeros(bs, *shape).to(next(self.conv_layers.parameters()).device)  # Ensure dummy input is on the correct device
        output = self.conv_layers(input)  # (1, 256, D_out, H_out, W_out)
        
        # Reshape to merge batch and temporal dimensions
        batch_size, channels, temporal, height, width = output.size()
        output = output.view(batch_size * temporal, channels, height, width)  # (batch_size * temporal, channels, H, W)
        
        # Do NOT apply spatial attention here. Instead, just flatten to determine _to_linear
        self._to_linear = int(np.prod(output.shape[1:]))  # channels * H * W
        print(f"Convolution output size: {output.shape}")
        print(f"Flattened size: {self._to_linear}")

    def forward(self, x, position=None, player_charge=None, enemy_charge=None,
                player_charge_temporal=None, enemy_charge_temporal=None, previous_inputs=None,
                health_memory=None, player_chip=None, enemy_chip=None):
        """
        Forward pass for the GameInputPredictor model.
        
        Args:
            x (Tensor): Images tensor of shape (batch_size, 3, image_memory, 160, 240)
            position (Tensor): Position features
            player_charge (Tensor): Player charge scalar
            enemy_charge (Tensor): Enemy charge scalar
            player_charge_temporal (Tensor): Player temporal charges
            enemy_charge_temporal (Tensor): Enemy temporal charges
            previous_inputs (Tensor): Previous inputs
            health_memory (Tensor): Health memory tensor of shape (batch_size, health_memory_size * 2)
            player_chip (Tensor): Player chip one-hot encoded tensor
            enemy_chip (Tensor): Enemy chip one-hot encoded tensor
        
        Returns:
            Tensor: Output tensor of shape (batch_size, 16)
        """
        features = []
        
        if self.include_image:
            # Process image through convolutional layers
            x = self.conv_layers(x)  # Shape: (batch_size, 256, D_out, H_out, W_out)
            # print(f"[Image Processing] After conv_layers: {x.shape}")
            
            # Apply spatial attention on each frame
            batch_size, channels, temporal, height, width = x.size()
            x = x.view(batch_size * temporal, channels, height, width)  # Merge batch and temporal
            # print(f"[Image Processing] After view (merge batch and temporal): {x.shape}")
            
            x = self.spatial_attention(x)  # Shape: (batch_size * temporal, channels, H, W)
            # print(f"[Image Processing] After spatial_attention: {x.shape}")
            
            x = x.view(batch_size, temporal, -1)  # Restore batch and temporal (batch_size, temporal, channels * H * W)
            # print(f"[Image Processing] After view (restore batch and temporal): {x.shape}")
            
            # Project to lower dimensions before Transformer
            x = self.transformer_input_projection(x)  # Shape: (batch_size, temporal, 256)
            # print(f"[Image Processing] After transformer_input_projection: {x.shape}")
            
              # Add positional encoding
            x = x + self.positional_encoding[:, :x.size(1), :]  # Ensure matching temporal length
            
            # Pass through Transformer Encoder for temporal attention
            # print(f"[Image Processing] Before permute for Transformer: {x.shape}")
            x = x.permute(1, 0, 2)  # Shape: (temporal, batch_size, 256)
            # print(f"[Image Processing] After permute for Transformer: {x.shape}")
            
            x = self.transformer_encoder(x)  # Shape: (temporal, batch_size, 256)
            # print(f"[Image Processing] After transformer_encoder: {x.shape}")
            
            x = x.mean(dim=0)  # Aggregate the sequence output (batch_size, 256)
            # print(f"[Image Processing] After mean pooling: {x.shape}")
            
            # Pass through fully connected layers
            x = self.image_fc(x)  # Shape: (batch_size, 256)
            # print(f"[Image Processing] After image_fc: {x.shape}")
            features.append(x)
        
        if self.additional_fc is not None:
            additional = []
            # Process position
            if self.include_position:
                if self.position_type == 'float':
                    additional.append(position)  # Shape: (batch_size, 4)
                    # print(f"[Additional Features] Position (float): {position.shape}")
                elif self.position_type == 'grid':
                    additional.append(position)  # Shape: (batch_size, 36)
                    # print(f"[Additional Features] Position (grid): {position.shape}")

            # Process chip selections with embeddings and attention
            chip_features = []
            player_embedded = None
            enemy_embedded = None
            if self.include_player_chip and player_chip is not None:
                # Assuming player_chip is one-hot encoded
                player_indices = player_chip.argmax(dim=1)  # Convert one-hot to indices
                player_embedded = self.player_chip_embedding(player_indices)  # Shape: (batch_size, 64)
                chip_features.append(player_embedded)
                # print(f"[Additional Features] Player Chip Embedded: {player_embedded.shape}")
            if self.include_enemy_chip and enemy_chip is not None:
                # Assuming enemy_chip is one-hot encoded
                enemy_indices = enemy_chip.argmax(dim=1)  # Convert one-hot to indices
                enemy_embedded256 = self.enemy_chip_embedding(enemy_indices)  # Shape: (batch_size, 64)
                chip_features.append(enemy_embedded)
                # print(f"[Additional Features] Enemy Chip Embedded: {enemy_embedded.shape}")
            
            if self.chip_attention and player_embedded is not None and enemy_embedded is not None:
                # Apply bilinear attention between player and enemy chip embeddings
                attention_weights = self.chip_attention(player_embedded, enemy_embedded)  # (batch_size, 1)
                # print(f"[Additional Features] Chip Attention Weights: {attention_weights.shape}")
                # Weighted sum (could be used to modulate features)
                chip_combined = attention_weights * player_embedded  # Shape: (batch_size, 64)
                additional.append(chip_combined)
                # print(f"[Additional Features] Chip Combined: {chip_combined.shape}")
            else:
                if player_embedded is not None:
                    additional.append(player_embedded)  # Shape: (batch_size, 64)
                    # print(f"[Additional Features] Player Embedded Added: {player_embedded.shape}")
                if enemy_embedded is not None:
                    additional.append(enemy_embedded)  # Shape: (batch_size, 64)
                    # print(f"[Additional Features] Enemy Embedded Added: {enemy_embedded.shape}")
            
            # Process temporal charge sequences with Transformer Encoders
            if self.include_temporal_charge:
                # Player charge temporal
                if player_charge_temporal is not None:
                    # print(f"[Additional Features] Player Charge Temporal Before Unsqueeze: {player_charge_temporal.shape}")
                    # Shape: (batch_size, temporal_charge)
                    player_charge_seq = player_charge_temporal.unsqueeze(-1)  # (batch_size, temporal_charge, 1)
                    # print(f"[Additional Features] Player Charge Temporal After Unsqueeze: {player_charge_seq.shape}")
                    # Permute for Transformer: (batch_size, temporal_charge, features) -> (temporal_charge, batch_size, features)
                    player_charge_seq = player_charge_seq.permute(1, 0, 2)  # (temporal_charge, batch_size, 1)
                    # print(f"[Additional Features] Player Charge Temporal After Permute: {player_charge_seq.shape}")
                    player_charge_encoded = self.player_charge_transformer(player_charge_seq)  # (temporal_charge, batch_size, 1)
                    # print(f"[Additional Features] Player Charge Encoded: {player_charge_encoded.shape}")
                    # Aggregate (mean pooling)
                    player_charge_features = player_charge_encoded.mean(dim=0)  # (batch_size, 1)
                    # print(f"[Additional Features] Player Charge Features: {player_charge_features.shape}")
                else:
                    player_charge_features = torch.zeros((x.size(0), 1), device=x.device)
                    # print(f"[Additional Features] Player Charge Temporal Missing: {player_charge_features.shape}")
                
                # Enemy charge temporal
                if enemy_charge_temporal is not None:
                    # print(f"[Additional Features] Enemy Charge Temporal Before Unsqueeze: {enemy_charge_temporal.shape}")
                    enemy_charge_seq = enemy_charge_temporal.unsqueeze(-1)  # (batch_size, temporal_charge, 1)
                    # print(f"[Additional Features] Enemy Charge Temporal After Unsqueeze: {enemy_charge_seq.shape}")
                    enemy_charge_seq = enemy_charge_seq.permute(1, 0, 2)  # (temporal_charge, batch_size, 1)
                    # print(f"[Additional Features] Enemy Charge Temporal After Permute: {enemy_charge_seq.shape}")
                    enemy_charge_encoded = self.enemy_charge_transformer(enemy_charge_seq)  # (temporal_charge, batch_size, 1)
                    # print(f"[Additional Features] Enemy Charge Encoded: {enemy_charge_encoded.shape}")
                    # Aggregate (mean pooling)
                    enemy_charge_features = enemy_charge_encoded.mean(dim=0)  # (batch_size, 1)
                    # print(f"[Additional Features] Enemy Charge Features: {enemy_charge_features.shape}")
                else:
                    enemy_charge_features = torch.zeros((x.size(0), 1), device=x.device)
                    # print(f"[Additional Features] Enemy Charge Temporal Missing: {enemy_charge_features.shape}")
                
                # Concatenate player and enemy charge features
                temporal_charge_features = torch.cat((player_charge_features, enemy_charge_features), dim=1)  # (batch_size, 2)
                additional.append(temporal_charge_features)  # Add to additional features
                # print(f"[Additional Features] Temporal Charge Features: {temporal_charge_features.shape}")
                
                # Additionally, process scalar player_charge if included
                if self.include_player_charge and player_charge is not None:
                    # print(f"[Additional Features] Player Charge Before Unsqueeze: {player_charge.shape}")
                    if player_charge.dim() == 1:
                        player_charge = player_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                        # print(f"[Additional Features] Player Charge After Unsqueeze: {player_charge.shape}")
                    additional.append(player_charge)  # (batch_size, 1)
                    # print(f"[Additional Features] Player Charge Added: {player_charge.shape}")
                
                # Additionally, process scalar enemy_charge if included
                if self.include_enemy_charge and enemy_charge is not None:
                    # print(f"[Additional Features] Enemy Charge Before Unsqueeze: {enemy_charge.shape}")
                    if enemy_charge.dim() == 1:
                        enemy_charge = enemy_charge.unsqueeze(1)  # Shape: (batch_size, 1)
                        # print(f"[Additional Features] Enemy Charge After Unsqueeze: {enemy_charge.shape}")
                    additional.append(enemy_charge)  # (batch_size, 1)
                    # print(f"[Additional Features] Enemy Charge Added: {enemy_charge.shape}")
            # else:
            #     print("[Additional Features] No temporal charge features to process.")
            
            # Process previous inputs
            if self.include_previous_inputs and previous_inputs is not None:
                # print(f"[Additional Features] Previous Inputs Before FC: {previous_inputs.shape}")
                # previous_inputs shape: (batch_size, input_memory, 16)
                previous_inputs_features = self.previous_inputs_fc(previous_inputs)  # (batch_size, 128)
                additional.append(previous_inputs_features)
                # print(f"[Additional Features] Previous Inputs After FC: {previous_inputs_features.shape}")
            
            # Process health memory with Transformer Encoder
            if self.include_health_memory and health_memory is not None:
                # print(f"[Additional Features] Health Memory Before Reshape: {health_memory.shape}")
                # Reshape health_memory from (batch_size, health_memory_size * 2) to (batch_size, health_memory_size, 2)
                batch_size = x.size(0)
                health_memory_size = health_memory.size(1) // 2
                health_memory = health_memory.view(batch_size, health_memory_size, 2)
                # print(f"[Additional Features] Health Memory After Reshape: {health_memory.shape}")

                # Permute for Transformer: (batch_size, health_memory_size, 2) -> (health_memory_size, batch_size, 2)
                health_memory_seq = health_memory.permute(1, 0, 2)  # (health_memory_size, batch_size, 2)
                # print(f"[Additional Features] Health Memory After Permute: {health_memory_seq.shape}")

                # print(f"[Additional Features] Health Memory After Permute: {health_memory_seq.shape}")
                health_encoded = self.health_transformer(health_memory_seq)  # (health_memory_size, batch_size, 2)
                # print(f"[Additional Features] Health Encoded: {health_encoded.shape}")
                # Aggregate (mean pooling)
                health_features = health_encoded.mean(dim=0)  # (batch_size, 2)
                # print(f"[Additional Features] Health Features After Mean Pooling: {health_features.shape}")
                # Pass through FC layers
                health_features = self.health_fc(health_features)  # (batch_size, 128)
                # print(f"[Additional Features] Health Features After FC: {health_features.shape}")
                additional.append(health_features)
            elif self.include_health_memory:
                # If health_memory is expected but not provided
                health_features = torch.zeros((batch_size, 128), device=x.device)
                additional.append(health_features)
                # print(f"[Additional Features] Health Memory Missing: {health_features.shape}")
            
            # Concatenate all additional features
            if additional:
                # print(f"[Additional Features] Concatenating {len(additional)} feature tensors.")
                additional = torch.cat(additional, dim=1)  # Shape: (batch_size, total_additional_features)
                # print(f"[Additional Features] After Concatenation: {additional.shape}")
                additional = self.additional_fc(additional)  # Shape: (batch_size, 128)
                # print(f"[Additional Features] After additional_fc: {additional.shape}")
                features.append(additional)
            # else:
                # print("[Additional Features] No additional features to concatenate.")
            
            # Concatenate all features
            if features:
                combined = torch.cat(features, dim=1)  # Shape: (batch_size, final_input_size)
                # print(f"[Combined Features] Shape after concatenation: {combined.shape}")
            # else:
            #     raise ValueError("No features to combine. Check configuration.")
            
            # Final classification
            output = self.fc_layers(combined)  # Shape: (batch_size, 16)
            # print(f"[Output] Final Output Shape: {output.shape}")
            
            return output
