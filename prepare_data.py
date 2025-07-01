#!/usr/bin/env python3
"""
Prepare training data from MP3 files in assets folder
"""

import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

def audio_to_mel_spectrogram(audio_path, target_shape=(400, 80), n_mels=80, hop_length=512, n_fft=2048):
    """Convert audio file to mel spectrogram - same as in train.py"""
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

def create_synthetic_hum(audio_path, output_path, hum_style="simple"):
    """Create a synthetic hum from a song"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract melody using harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        if hum_style == "simple":
            # Simple approach: use harmonic component with reduced amplitude
            hum_audio = y_harmonic * 0.3
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.02, len(hum_audio))
            hum_audio = hum_audio + noise
            
        elif hum_style == "filtered":
            # More sophisticated: extract fundamental frequency
            # Apply pitch tracking and create sine wave
            pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
            
            # Create a simple sine wave hum based on dominant frequencies
            hum_audio = np.zeros_like(y_harmonic)
            hop_length = 512
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    # Create sine wave for this time frame
                    start_sample = t * hop_length
                    end_sample = min((t + 1) * hop_length, len(hum_audio))
                    time_samples = np.arange(start_sample, end_sample)
                    sine_wave = 0.3 * np.sin(2 * np.pi * pitch * time_samples / sr)
                    hum_audio[start_sample:end_sample] = sine_wave
            
            # Add vibrato and some noise for realism
            vibrato = 0.05 * np.sin(2 * np.pi * 5 * np.arange(len(hum_audio)) / sr)
            hum_audio = hum_audio * (1 + vibrato)
            noise = np.random.normal(0, 0.02, len(hum_audio))
            hum_audio = hum_audio + noise
        
        # Normalize
        hum_audio = hum_audio / np.max(np.abs(hum_audio)) * 0.8
        
        # Save as WAV
        sf.write(output_path, hum_audio, sr)
        return True
        
    except Exception as e:
        print(f"Error creating hum for {audio_path}: {e}")
        return False

def prepare_training_data():
    """Prepare training data from assets folder"""
    
    # Define paths
    assets_dir = "/Users/samandersony/StudioProjects/projects/asdf/.venv/assets"
    data_dir = "/Users/samandersony/StudioProjects/projects/asdf/data"
    npy_cache_dir = os.path.join(data_dir, "npy_cache")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(npy_cache_dir, exist_ok=True)
    
    # Define your song mappings
    song_mappings = {
        "Ed Sheeran - Perfect (Piano Cover by Riyandi Kusuma).mp3": "perfect_piano",
        "Ed Sheeran - Perfect (Official Music Video).mp3": "perfect_official", 
        "Perfect - Ed Sheeran Fluteフルート.mp3": "perfect_flute",
        "Maroon 5 - Payphone ft. Wiz Khalifa (Explicit) (Official Music Video).mp3": "payphone",
        "Maroon 5 - Sugar (Official Music Video).mp3": "sugar",
        "Perfect - Ed Sheeran (Wedding Version) [Lyric Video] Mild Nawin.mp3": "perfect_wedding",
        "Tanner Patrick - Perfect (Ed Sheeran Cover).mp3": "perfect_tanner",
        "ED SHEERAN - Perfect (Available in Spotify).mp3": "perfect_spotify"
    }
    
    # Process each song
    for original_filename, song_id in song_mappings.items():
        original_path = os.path.join(assets_dir, original_filename)
        
        if not os.path.exists(original_path):
            print(f"❌ File not found: {original_path}")
            continue
            
        print(f"📁 Processing: {original_filename}")
        
        # Define output paths
        song_wav = os.path.join(data_dir, f"{song_id}_song.wav")
        hum1_wav = os.path.join(data_dir, f"{song_id}_hum1.wav")
        hum2_wav = os.path.join(data_dir, f"{song_id}_hum2.wav")
        
        # Convert MP3 to WAV (for consistency)
        print(f"  Converting to WAV...")
        try:
            y, sr = librosa.load(original_path, sr=22050)
            sf.write(song_wav, y, sr)
            print(f"  ✅ Saved: {song_wav}")
        except Exception as e:
            print(f"  ❌ Error converting {original_filename}: {e}")
            continue
        
        # Create synthetic hums
        print(f"  Creating synthetic hums...")
        if create_synthetic_hum(song_wav, hum1_wav, "simple"):
            print(f"  ✅ Created hum1: {hum1_wav}")
        else:
            print(f"  ❌ Failed to create hum1")
            
        if create_synthetic_hum(song_wav, hum2_wav, "filtered"):
            print(f"  ✅ Created hum2: {hum2_wav}")
        else:
            print(f"  ❌ Failed to create hum2")
        
        # Preprocess to NPY files
        print(f"  Converting to spectrograms...")
        for audio_file in [song_wav, hum1_wav, hum2_wav]:
            if os.path.exists(audio_file):
                npy_path = os.path.join(npy_cache_dir, os.path.basename(audio_file) + '.npy')
                spec = audio_to_mel_spectrogram(audio_file)
                if spec is not None:
                    np.save(npy_path, spec)
                    print(f"    ✅ Saved spectrogram: {npy_path}")
                else:
                    print(f"    ❌ Failed to create spectrogram for {audio_file}")
    
    # Summary
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    npy_files = [f for f in os.listdir(npy_cache_dir) if f.endswith('.npy')]
    
    print(f"\n📊 PREPARATION COMPLETE!")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎵 WAV files created: {len(wav_files)}")
    print(f"📈 NPY files created: {len(npy_files)}")
    
    # Show file structure
    print(f"\n📂 Files created:")
    for f in sorted(wav_files):
        print(f"  🎵 {f}")
    
    print(f"\n📈 Spectrograms cached:")
    for f in sorted(npy_files):
        print(f"  📊 {f}")
    
    return data_dir

if __name__ == "__main__":
    prepare_training_data()