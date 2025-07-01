"""
Dataset classes for loading and preprocessing spectrograms
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.samples = []
        
        # Load all .npy files and create pairs
        song_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and 'song' in f]
        hum_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and 'hum' in f]
        
        print(f"Found {len(song_files)} song files and {len(hum_files)} hum files")
        
        # Create positive pairs (song-hum of same melody)
        for song_file in song_files:
            song_name = song_file.split('_song')[0]
            # Find matching hum files
            matching_hums = [h for h in hum_files if song_name in h]
            
            for hum_file in matching_hums:
                self.samples.append({
                    'song': song_file,
                    'hum': hum_file,
                    'label': 1,  # Positive pair
                    'song_name': song_name
                })
        
        # Create negative pairs (song-hum of different melodies)
        for song_file in song_files:
            song_name = song_file.split('_song')[0]
            # Find non-matching hum files
            non_matching_hums = [h for h in hum_files if song_name not in h]
            
            # Add some negative samples (limit to avoid imbalance)
            for hum_file in non_matching_hums[:2]:  # Limit negative samples
                self.samples.append({
                    'song': song_file,
                    'hum': hum_file,
                    'label': 0,  # Negative pair
                    'song_name': song_name
                })
        
        print(f"Created {len(self.samples)} training pairs")
        positive_pairs = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Positive pairs: {positive_pairs}, Negative pairs: {len(self.samples) - positive_pairs}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load song and hum spectrograms
        song_path = os.path.join(self.data_dir, sample['song'])
        hum_path = os.path.join(self.data_dir, sample['hum'])
        
        song_spec = np.load(song_path).astype(np.float32)
        hum_spec = np.load(hum_path).astype(np.float32)
        
        # Normalize spectrograms
        song_spec = (song_spec - song_spec.mean()) / (song_spec.std() + 1e-8)
        hum_spec = (hum_spec - hum_spec.mean()) / (hum_spec.std() + 1e-8)
        
        # Pad or truncate to fixed size (400, 80)
        target_shape = (400, 80)
        song_spec = self._resize_spectrogram(song_spec, target_shape)
        hum_spec = self._resize_spectrogram(hum_spec, target_shape)
        
        # Convert to tensors and add channel dimension
        song_tensor = torch.from_numpy(song_spec).unsqueeze(0)  # (1, 400, 80)
        hum_tensor = torch.from_numpy(hum_spec).unsqueeze(0)    # (1, 400, 80)
        
        return song_tensor, hum_tensor, torch.tensor(sample['label'], dtype=torch.float32)
    
    def _resize_spectrogram(self, spec, target_shape):
        """Resize spectrogram to target shape"""
        h, w = spec.shape
        target_h, target_w = target_shape
        
        # Pad or truncate height
        if h < target_h:
            pad_h = target_h - h
            spec = np.pad(spec, ((0, pad_h), (0, 0)), mode='constant')
        elif h > target_h:
            spec = spec[:target_h, :]
        
        # Pad or truncate width
        if w < target_w:
            pad_w = target_w - w
            spec = np.pad(spec, ((0, 0), (0, pad_w)), mode='constant')
        elif w > target_w:
            spec = spec[:, :target_w]
        
        return spec
