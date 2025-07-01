#!/usr/bin/env python3
"""
Check if a given audio file matches a 'perfect' song using the trained hum-to-song model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import scipy.io.wavfile
from pydub import AudioSegment

# Import model class
from models.siamese import SiameseNetwork

class PerfectMatchChecker:
    def __init__(self, model_path='/Users/samandersony/StudioProjects/projects/asdf/models_saved/final_model.pth', cache_npy=True, data_dir='/Users/samandersony/StudioProjects/projects/asdf/data'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_npy = cache_npy
        self.data_dir = data_dir
        print(f"Using device: {self.device}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        embed_dim = checkpoint.get('embed_dim', 128)
        
        self.model = SiameseNetwork(embed_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        
        # Filter only 'perfect' songs into database
        self.perfect_songs = {}
        self.perfect_embeddings = {}
        self._load_perfect_songs()
    
    def _load_perfect_songs(self):
        """Load only 'perfect' song files from data_dir into the database"""
        cache_dir = os.path.join(self.data_dir, "npy_cache") if self.cache_npy else None
        song_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.wav', '.mp3')) and 'song' in f and 'perfect' in f]
        for song_file in song_files:
            song_name = song_file.split('_song')[0].replace('_', ' ').title()
            song_path = os.path.join(self.data_dir, song_file)
            self.add_song_to_database(song_path, song_name, cache_dir)
        print(f"🎶 Loaded {len(song_files)} 'perfect' songs into database")
    
    def _preprocess_spectrogram(self, path, cache_dir=None):
        """Load and preprocess spectrogram from audio or .npy file"""
        if path.endswith('.npy'):
            spec = np.load(path).astype(np.float32)
        else:
            if self.cache_npy and cache_dir:
                npy_path = os.path.join(cache_dir, os.path.basename(path).replace('.mp3', '.wav') + '.npy')
                if os.path.exists(npy_path):
                    spec = np.load(npy_path).astype(np.float32)
                else:
                    if path.endswith('.mp3'):
                        wav_path = os.path.join(cache_dir, os.path.basename(path).replace('.mp3', '.wav'))
                        if not os.path.exists(wav_path):
                            audio = AudioSegment.from_mp3(path)
                            audio.export(wav_path, format="wav")
                        path = wav_path
                    spec = self.audio_to_mel_spectrogram(path)
                    if spec is not None:
                        os.makedirs(cache_dir, exist_ok=True)
                        np.save(npy_path, spec)
            else:
                if path.endswith('.mp3'):
                    wav_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace('.mp3', '.wav'))
                    if not os.path.exists(wav_path):
                        audio = AudioSegment.from_mp3(path)
                        audio.export(wav_path, format="wav")
                    path = wav_path
                spec = self.audio_to_mel_spectrogram(path)
            
            if spec is None:
                print(f"Error processing {path}, returning zero array")
                spec = np.zeros((400, 80), dtype=np.float32)
        
        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        # Resize to (400, 80)
        target_shape = (400, 80)
        h, w = spec.shape
        target_h, target_w = target_shape
        
        if h < target_h:
            pad_h = target_h - h
            spec = np.pad(spec, ((0, pad_h), (0, 0)), mode='constant')
        elif h > target_h:
            spec = spec[:target_h, :]
        
        if w < target_w:
            pad_w = target_w - w
            spec = np.pad(spec, ((0, 0), (0, pad_w)), mode='constant')
        elif w > target_w:
            spec = spec[:, :target_w]
        
        # Convert to tensor
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, 400, 80)
        return tensor.to(self.device)
    
    def audio_to_mel_spectrogram(self, audio_path, target_shape=(400, 80), n_mels=80, hop_length=512, n_fft=2048):
        """Convert audio file to mel spectrogram"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
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
    
    def add_song_to_database(self, song_path, song_name, cache_dir=None):
        """Add a 'perfect' song to the search database"""
        print(f"Adding {song_name} to database...")
        song_tensor = self._preprocess_spectrogram(song_path, cache_dir)
        
        with torch.no_grad():
            song_embedding = self.model.encoder(song_tensor)
        
        self.perfect_songs[song_name] = song_path
        self.perfect_embeddings[song_name] = song_embedding.cpu().numpy()
        print(f"✅ {song_name} added to database")
    
    def check_perfect_match(self, audio_path, threshold=0.5):
        """Check if the audio matches any 'perfect' song"""
        if not self.perfect_songs:
            print("❌ No 'perfect' songs in database! Please ensure data directory contains 'perfect' song files.")
            return None
        
        print(f"🔍 Checking if {audio_path} matches a 'perfect' song")
        cache_dir = os.path.join(self.data_dir, "npy_cache") if self.cache_npy else None
        hum_tensor = self._preprocess_spectrogram(audio_path, cache_dir)
        
        best_match = None
        best_confidence = -1
        best_cosine_sim = -1
        
        with torch.no_grad():
            for song_name, song_path in self.perfect_songs.items():
                song_tensor = self._preprocess_spectrogram(song_path, cache_dir)
                prob, song_embed, hum_embed = self.model(song_tensor, hum_tensor)
                confidence = prob.item()
                
                song_np = song_embed.cpu().numpy().flatten()
                hum_np = hum_embed.cpu().numpy().flatten()
                cosine_sim = np.dot(song_np, hum_np) / (np.linalg.norm(song_np) * np.linalg.norm(hum_np))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_cosine_sim = cosine_sim
                    best_match = song_name
        
        is_match = best_confidence > threshold
        
        return {
            'match': best_match,
            'confidence': best_confidence,
            'cosine_similarity': best_cosine_sim,
            'is_perfect_match': is_match
        }

def main():
    checker = PerfectMatchChecker()
    
    while True:
        print("\n" + "="*50)
        print("🎵 PERFECT MATCH CHECKER 🎵")
        print("Enter the path to an audio file (hum) to check if it matches a 'perfect' song.")
        print("Enter 'exit' to quit.")
        
        user_input = input("\nAudio file path: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not os.path.exists(user_input):
            print(f"❌ File not found: {user_input}")
            continue
        
        # Check match
        result = checker.check_perfect_match(user_input)
        
        if result:
            print("\n🎯 MATCH RESULTS:")
            print(f"Match: {result['match']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Cosine Similarity: {result['cosine_similarity']:.3f}")
            print(f"Is Perfect Match: {'✅ Yes' if result['is_perfect_match'] else '❌ No'}")
        else:
            print("❌ No results found")

if __name__ == "__main__":
    main()