#!/usr/bin/env python3
"""
Audio processing utilities for converting audio files to mel spectrograms
"""

import librosa
import numpy as np
import os
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self, 
                 sample_rate=22050, 
                 n_mels=80, 
                 n_fft=2048, 
                 hop_length=512,
                 duration=10.0):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel frequency bands
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            duration: Maximum duration in seconds to process
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        
    def load_audio(self, audio_path):
        """Load audio file and convert to target sample rate"""
        try:
            # Try loading with librosa first
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path} with librosa: {e}")
            try:
                # Fallback to pydub for other formats
                audio_segment = AudioSegment.from_file(audio_path)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                
                # Convert to numpy array
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / (2**15)  # Normalize to [-1, 1]
                
                # Truncate to duration
                max_samples = int(self.duration * self.sample_rate)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                    
                return audio, self.sample_rate
            except Exception as e2:
                print(f"Error loading {audio_path} with pydub: {e2}")
                raise e2
    
    def audio_to_spectrogram(self, audio_path):
        """Convert audio file to mel spectrogram"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, frequency) format
        mel_spec_db = mel_spec_db.T
        
        return mel_spec_db.astype(np.float32)
    
    def save_spectrogram(self, spectrogram, output_path):
        """Save spectrogram as numpy array"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, spectrogram)
        print(f"Spectrogram saved to {output_path}")
    
    def process_audio_file(self, audio_path, output_path):
        """Process single audio file to spectrogram"""
        try:
            spectrogram = self.audio_to_spectrogram(audio_path)
            self.save_spectrogram(spectrogram, output_path)
            return spectrogram
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_audio_directory(self, input_dir, output_dir, extensions=None):
        """Process all audio files in a directory"""
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(audio_files)} audio files")
        
        processed_count = 0
        for audio_file in audio_files:
            output_file = output_path / (audio_file.stem + '.npy')
            
            try:
                spectrogram = self.audio_to_spectrogram(str(audio_file))
                self.save_spectrogram(spectrogram, str(output_file))
                processed_count += 1
                print(f"Processed: {audio_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"Failed to process {audio_file.name}: {e}")
        
        print(f"Successfully processed {processed_count}/{len(audio_files)} files")
        return processed_count
    
    def create_song_hum_pairs(self, songs_dir, hums_dir, output_dir):
        """
        Process songs and hums directories and create properly named pairs
        
        Expected structure:
        songs_dir/
            song1.wav
            song2.mp3
            ...
        hums_dir/
            song1_hum1.wav
            song1_hum2.wav
            song2_hum1.wav
            ...
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        songs_path = Path(songs_dir)
        hums_path = Path(hums_dir)
        
        # Process songs
        song_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            song_files.extend(songs_path.glob(f'*{ext}'))
            song_files.extend(songs_path.glob(f'*{ext.upper()}'))
        
        print(f"Processing {len(song_files)} song files...")
        for song_file in song_files:
            song_name = song_file.stem
            output_file = output_path / f"{song_name}_song.npy"
            
            try:
                spectrogram = self.audio_to_spectrogram(str(song_file))
                self.save_spectrogram(spectrogram, str(output_file))
                print(f"Processed song: {song_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"Failed to process song {song_file.name}: {e}")
        
        # Process hums
        hum_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            hum_files.extend(hums_path.glob(f'*{ext}'))
            hum_files.extend(hums_path.glob(f'*{ext.upper()}'))
        
        print(f"Processing {len(hum_files)} hum files...")
        for hum_file in hum_files:
            hum_name = hum_file.stem
            output_file = output_path / f"{hum_name}.npy"
            
            try:
                spectrogram = self.audio_to_spectrogram(str(hum_file))
                self.save_spectrogram(spectrogram, str(output_file))
                print(f"Processed hum: {hum_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"Failed to process hum {hum_file.name}: {e}")
    
    def visualize_spectrogram(self, spectrogram, title="Mel Spectrogram", save_path=None):
        """Visualize a mel spectrogram"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            spectrogram.T,  # Transpose back for display
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectrogram visualization saved to {save_path}")
        
        plt.show()

def batch_convert_audio(input_dir, output_dir, sample_rate=22050, duration=10.0):
    """Convenience function for batch conversion"""
    processor = AudioProcessor(sample_rate=sample_rate, duration=duration)
    return processor.process_audio_directory(input_dir, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert audio files to mel spectrograms')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file or directory')
    parser.add_argument('--songs_dir', type=str, help='Directory containing song files')
    parser.add_argument('--hums_dir', type=str, help='Directory containing hum files')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Target sample rate')
    parser.add_argument('--duration', type=float, default=10.0, help='Max duration in seconds')
    parser.add_argument('--visualize', action='store_true', help='Visualize the spectrogram')
    
    args = parser.parse_args()
    
    processor = AudioProcessor(sample_rate=args.sample_rate, duration=args.duration)
    
    if args.songs_dir and args.hums_dir:
        # Process song-hum pairs
        processor.create_song_hum_pairs(args.songs_dir, args.hums_dir, args.output)
    elif os.path.isfile(args.input):
        # Process single file
        spectrogram = processor.process_audio_file(args.input, args.output)
        if spectrogram is not None and args.visualize:
            processor.visualize_spectrogram(spectrogram, f"Spectrogram: {args.input}")
    elif os.path.isdir(args.input):
        # Process directory
        processor.process_audio_directory(args.input, args.output)
    else:
        print(f"Input path {args.input} does not exist")
