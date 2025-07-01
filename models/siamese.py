"""
Siamese network for song-hum matching
"""

import torch
import torch.nn as nn
from .encoder import SpectrogramEncoder

class SiameseNetwork(nn.Module):
    """Siamese network for song-hum matching"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = SpectrogramEncoder(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, song, hum):
        song_embed = self.encoder(song)
        hum_embed = self.encoder(hum)
        
        # Compute element-wise product and pass through classifier
        combined = song_embed * hum_embed
        
        # Predict match probability
        prob = self.classifier(combined)
        
        return prob, song_embed, hum_embed
