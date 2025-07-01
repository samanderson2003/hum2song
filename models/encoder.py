"""
CNN encoder for mel spectrograms
"""

import torch
import torch.nn as nn

class SpectrogramEncoder(nn.Module):
    """CNN encoder for mel spectrograms"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 200, 40)
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 100, 20)
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 50, 10)
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, 400, 80)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        embeddings = self.fc_layers(features)
        
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
