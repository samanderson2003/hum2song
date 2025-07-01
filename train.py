#!/usr/bin/env python3
"""
Train a neural network to recognize hummed melodies and match them to songs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io.wavfile
import random

# Import local modules (assuming they exist)
from models.siamese import SiameseNetwork
from utils.visualization import plot_training_curves

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def audio_to_mel_spectrogram(audio_path, target_shape=(400, 80), n_mels=80, hop_length=512, n_fft=2048):
    """
    Convert audio file to mel spectrogram
    
    Parameters:
    - audio_path: Path to input audio file (e.g., WAV, MP3)
    - target_shape: Desired shape of spectrogram (height, width)
    - n_mels: Number of mel bands
    - hop_length: Hop length for STFT
    - n_fft: FFT window size
    
    Returns:
    - mel_spec_db: Mel spectrogram as numpy array
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Resize to target shape
        h, w = mel_spec_db.shape
        target_h, target_w = target_shape
        
        if h < target_h:
            mel_spec_db = np.pad(mel_spec_db, ((0, target_h - h), (0, 0)), mode='constant')
        elif h > target_h:
            mel_spec_db = mel_spec_db[:target_h, :]
        
        if w < target_w:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_w - w)), mode='constant')
        elif w > target_w:
            mel_spec_db = mel_spec_db[:, :target_w]
        
        return mel_spec_db.astype(np.float32)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

class MelSpectrogramDataset(Dataset):
    """Dataset for loading song and hum audio files and creating pairs"""
    
    def __init__(self, data_dir, cache_npy=True, target_shape=(400, 80)):
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.cache_npy = cache_npy
        self.cache_dir = os.path.join(data_dir, "npy_cache")
        
        # Find audio files (WAV or MP3)
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.mp3'))]
        self.song_files = [f for f in self.audio_files if 'song' in f]
        self.hum_files = [f for f in self.audio_files if 'hum' in f]
        
        print(f"Found {len(self.song_files)} songs: {self.song_files}")
        print(f"Found {len(self.hum_files)} hums: {self.hum_files}")
        
        # Create positive and negative pairs
        self.pairs = []
        for song_file in self.song_files:
            # Extract song base name (everything before '_song')
            song_base = song_file.split('_song')[0]
            
            for hum_file in self.hum_files:
                # Extract hum base name (everything before '_hum')
                hum_base = hum_file.split('_hum')[0]
                
                # Positive pair only if base names match and start with "perfect"
                if song_base == hum_base and song_base.startswith('perfect'):
                    self.pairs.append((song_file, hum_file, 1))
                else:
                    self.pairs.append((song_file, hum_file, 0))
        
        # Balance the dataset (equal positive and negative pairs)
        positive_pairs = [p for p in self.pairs if p[2] == 1]
        negative_pairs = [p for p in self.pairs if p[2] == 0]
        
        print(f"Positive pairs (matching 'perfect' song and hum): {len(positive_pairs)}")
        print(f"Negative pairs (all other combinations): {len(negative_pairs)}")
        
        # Keep all positive pairs and sample negative pairs to balance
        if len(negative_pairs) > len(positive_pairs):
            random.seed(42)
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))
        
        self.pairs = positive_pairs + negative_pairs
        print(f"Final balanced dataset: {len(self.pairs)} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        song_file, hum_file, label = self.pairs[idx]
        
        # Load preprocessed spectrograms from cache
        if self.cache_npy and os.path.exists(self.cache_dir):
            song_npy = os.path.join(self.cache_dir, song_file + '.npy')
            hum_npy = os.path.join(self.cache_dir, hum_file + '.npy')
            
            if os.path.exists(song_npy) and os.path.exists(hum_npy):
                song_spec = np.load(song_npy).astype(np.float32)
                hum_spec = np.load(hum_npy).astype(np.float32)
            else:
                # Fallback to audio processing
                song_path = os.path.join(self.data_dir, song_file)
                hum_path = os.path.join(self.data_dir, hum_file)
                song_spec = audio_to_mel_spectrogram(song_path, target_shape=self.target_shape)
                hum_spec = audio_to_mel_spectrogram(hum_path, target_shape=self.target_shape)
                if song_spec is not None and hum_spec is not None:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.save(song_npy, song_spec)
                    np.save(hum_npy, hum_spec)
        else:
            # Load and process audio files
            song_path = os.path.join(self.data_dir, song_file)
            hum_path = os.path.join(self.data_dir, hum_file)
            song_spec = audio_to_mel_spectrogram(song_path, target_shape=self.target_shape)
            hum_spec = audio_to_mel_spectrogram(hum_path, target_shape=self.target_shape)
        
        if song_spec is None or hum_spec is None:
            # Return dummy data if loading fails to avoid crashing
            song_spec = np.zeros(self.target_shape, dtype=np.float32)
            hum_spec = np.zeros(self.target_shape, dtype=np.float32)
            label = 0  # Mark as negative pair to avoid affecting training
        
        # Convert to torch tensors
        song_tensor = torch.from_numpy(song_spec).unsqueeze(0)  # (1, H, W)
        hum_tensor = torch.from_numpy(hum_spec).unsqueeze(0)    # (1, H, W)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return song_tensor, hum_tensor, label_tensor

def train_model(data_dir, num_epochs=50, batch_size=8, learning_rate=0.001, resume_from=None, cache_npy=False):
    """Train the siamese network"""
    
    # Create dataset and dataloader
    dataset = MelSpectrogramDataset(data_dir, cache_npy=cache_npy)
    if len(dataset) == 0:
        print("❌ No valid audio pairs found in data directory!")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SiameseNetwork(embed_dim=128).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'train_accuracies' in checkpoint:
            train_accuracies = checkpoint['train_accuracies']
    
    # Training loop
    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for song, hum, labels in progress_bar:
            song = song.to(device)
            hum = hum.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, song_embeds, hum_embeds = model(song, hum)
            predictions = predictions.squeeze()
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models_saved/model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'embed_dim': 128,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }, 'models_saved/final_model.pth')
    
    print("Training completed!")
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, save_path='results/training_curves.png')
    
    return model

def create_demo_data():
    """Create demo audio data for one correct song ('perfect') and wrong songs ('sugar', 'payphone')"""
    data_dir = "/Users/samandersony/StudioProjects/projects/asdf/data/"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Creating demo audio data...")
    
    # Generate synthetic audio for one song and hums
    sr = 22050  # Sample rate
    duration = 10  # Seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create "perfect" song (correct song)
    freq_perfect = 440 * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # Varying frequency
    perfect_song = 0.5 * np.sin(2 * np.pi * freq_perfect * t).astype(np.float32)
    perfect_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    
    # Additional "perfect" variations from screenshot
    perfect_flute_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_flute_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    perfect_flute_song = perfect_song
    perfect_official_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_official_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    perfect_official_song = perfect_song
    perfect_piano_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_piano_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    perfect_piano_song = perfect_song
    perfect_spotify_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_spotify_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    perfect_spotify_song = perfect_song
    perfect_tanner_hum1 = perfect_song + np.random.normal(0, 0.1, perfect_song.shape).astype(np.float32)
    perfect_tanner_hum2 = perfect_song * 0.9 + np.random.normal(0, 0.15, perfect_song.shape).astype(np.float32)
    perfect_tanner_song = perfect_song
    
    # Create "sugar" song (wrong song)
    freq_sugar = 330 * (1 + 0.3 * np.sin(2 * np.pi * 0.7 * t))  # Different pattern
    sugar_song = 0.5 * np.sin(2 * np.pi * freq_sugar * t).astype(np.float32)
    sugar_hum1 = sugar_song + np.random.normal(0, 0.1, sugar_song.shape).astype(np.float32)
    sugar_hum2 = sugar_song * 0.9 + np.random.normal(0, 0.15, sugar_song.shape).astype(np.float32)
    
    # Create "payphone" song (wrong song)
    freq_payphone = 392 * (1 + 0.4 * np.sin(2 * np.pi * 0.6 * t))  # Different pattern
    payphone_song = 0.5 * np.sin(2 * np.pi * freq_payphone * t).astype(np.float32)
    payphone_hum1 = payphone_song + np.random.normal(0, 0.1, payphone_song.shape).astype(np.float32)
    payphone_hum2 = payphone_song * 0.9 + np.random.normal(0, 0.15, payphone_song.shape).astype(np.float32)
    
    # Create noise hum (should not match anything)
    noise_hum = np.random.randn(len(t)).astype(np.float32) * 0.5
    
    # Save as WAV files
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_song.wav"), sr, perfect_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_hum1.wav"), sr, perfect_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_hum2.wav"), sr, perfect_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_flute_song.wav"), sr, perfect_flute_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_flute_hum1.wav"), sr, perfect_flute_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_flute_hum2.wav"), sr, perfect_flute_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_official_song.wav"), sr, perfect_official_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_official_hum1.wav"), sr, perfect_official_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_official_hum2.wav"), sr, perfect_official_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_piano_song.wav"), sr, perfect_piano_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_piano_hum1.wav"), sr, perfect_piano_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_piano_hum2.wav"), sr, perfect_piano_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_spotify_song.wav"), sr, perfect_spotify_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_spotify_hum1.wav"), sr, perfect_spotify_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_spotify_hum2.wav"), sr, perfect_spotify_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_tanner_song.wav"), sr, perfect_tanner_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_tanner_hum1.wav"), sr, perfect_tanner_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "perfect_tanner_hum2.wav"), sr, perfect_tanner_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "sugar_song.wav"), sr, sugar_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "sugar_hum1.wav"), sr, sugar_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "sugar_hum2.wav"), sr, sugar_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "payphone_song.wav"), sr, payphone_song)
    scipy.io.wavfile.write(os.path.join(data_dir, "payphone_hum1.wav"), sr, payphone_hum1)
    scipy.io.wavfile.write(os.path.join(data_dir, "payphone_hum2.wav"), sr, payphone_hum2)
    scipy.io.wavfile.write(os.path.join(data_dir, "noise_hum1.wav"), sr, noise_hum)
    
    print("Demo audio data created in 'data/' directory at:", data_dir)
    print("- Correct pairs (should match):")
    print("  - perfect_song.wav, perfect_hum1.wav, perfect_hum2.wav")
    print("  - perfect_flute_song.wav, perfect_flute_hum1.wav, perfect_flute_hum2.wav")
    print("  - perfect_official_song.wav, perfect_official_hum1.wav, perfect_official_hum2.wav")
    print("  - perfect_piano_song.wav, perfect_piano_hum1.wav, perfect_piano_hum2.wav")
    print("  - perfect_spotify_song.wav, perfect_spotify_hum1.wav, perfect_spotify_hum2.wav")
    print("  - perfect_tanner_song.wav, perfect_tanner_hum1.wav, perfect_tanner_hum2.wav")
    print("- Wrong pairs (should NOT match):")
    print("  - sugar_song.wav, sugar_hum1.wav, sugar_hum2.wav")
    print("  - payphone_song.wav, payphone_hum1.wav, payphone_hum2.wav")
    print("  - noise_hum1.wav (noise, should not match anything)")
    print("Note: For real audio, place MP3/WAV files in 'data/' with names like 'perfect_song.mp3', 'sugar_song.mp3', etc.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hum-to-song matching model')
    parser.add_argument('--data_dir', type=str, default='/Users/samandersony/StudioProjects/projects/asdf/data', help='Directory containing audio files (.wav or .mp3)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--create_demo', action='store_true', help='Create demo audio data')
    parser.add_argument('--cache_npy', action='store_true', help='Cache spectrograms as .npy files')
    
    args = parser.parse_args()
    
    if args.create_demo:
        create_demo_data()
    
    if os.path.exists(args.data_dir):
        print(f"Starting training with data from {args.data_dir}")
        model = train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.resume, args.cache_npy)
    else:
        print(f"Data directory {args.data_dir} not found. Use --create_demo to create demo data first.")