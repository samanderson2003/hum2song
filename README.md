# Song Recognition AI Project

A neural network-based system for recognizing hummed melodies and matching them to songs using PyTorch and deep learning.

## Features

- **Siamese Neural Network**: Uses a CNN-based encoder to create embeddings for audio spectrograms
- **Song Database**: Build and search through a database of songs
- **Real-time Matching**: Match hummed melodies to songs in the database
- **Performance Analysis**: Comprehensive testing and visualization tools
- **Demo Data**: Built-in demo data generation for testing

## Project Structure

```
song-recognition/
├── train.py              # Train the neural network model
├── search.py             # Search and test the trained model
├── audio_processor.py    # Audio preprocessing utilities
├── models/               # Model architecture definitions
│   ├── __init__.py
│   ├── encoder.py        # CNN encoder
│   └── siamese.py        # Siamese network
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── dataset.py        # Dataset classes
│   └── visualization.py  # Plotting and visualization
├── data/                 # Training and test data
├── models_saved/         # Saved model checkpoints
├── results/              # Training results and plots
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create Demo Data
```bash
python train.py --create_demo
```

### 2. Train the Model
```bash
python train.py --data_dir data --epochs 50
```

### 3. Test Performance
```bash
python search.py --create_test_data
python search.py --test_performance
```

### 4. Interactive Demo
```bash
python search.py --interactive
```

## Usage

### Training a Model

```bash
# Basic training with default parameters
python train.py --data_dir data

# Custom training parameters
python train.py --data_dir data --epochs 100 --batch_size 16 --lr 0.0005

# Create demo data first
python train.py --create_demo
```

### Searching for Songs

```bash
# Test model performance
python search.py --test_performance

# Interactive search demo
python search.py --interactive

# Search for a specific hum file
python search.py --search_hum data/perfect_hum1.npy

# Create comprehensive test data
python search.py --create_test_data
```

## Model Architecture

The system uses a **Siamese Neural Network** with the following components:

### CNN Encoder
- 4 convolutional blocks with batch normalization
- Progressive channel increase: 1 → 32 → 64 → 128 → 256
- Adaptive average pooling for consistent output size
- Fully connected layers with dropout for regularization
- L2 normalization for embedding similarity

### Siamese Network
- Shared encoder for both song and hum spectrograms
- Similarity computation using dot product
- Classification head for match/no-match prediction

## Data Format

The system expects mel-spectrogram data in `.npy` format:
- **Shape**: (time_frames, mel_bins)
- **Naming Convention**: 
  - Songs: `{song_name}_song.npy`
  - Hums: `{song_name}_hum{number}.npy`

## Performance Metrics

The system tracks and visualizes:
- Accuracy, Precision, Recall, F1-Score
- ROC and Precision-Recall curves
- Confusion matrices
- Confidence score distributions
- Detailed false positive/negative analysis

## Audio Processing

To convert audio files to the required format, use the `audio_processor.py` utility:

```python
from audio_processor import AudioProcessor

processor = AudioProcessor()
spectrogram = processor.audio_to_spectrogram("song.wav")
processor.save_spectrogram(spectrogram, "song_spectrogram.npy")
```

## Advanced Usage

### Custom Dataset
1. Prepare your audio files (WAV, MP3, etc.)
2. Convert to mel-spectrograms using `audio_processor.py`
3. Follow the naming convention
4. Train the model with your data

### Fine-tuning
```bash
# Resume training from checkpoint
python train.py --resume model_epoch_20.pth --epochs 30
```

### Batch Processing
```python
from search import SongSearcher

searcher = SongSearcher('final_model.pth')
searcher.add_song_to_database('song1.npy', 'Song 1')
searcher.add_song_to_database('song2.npy', 'Song 2')

results = searcher.search_song('hum.npy', top_k=10)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

2. **Poor Performance**
   - Ensure data quality and proper normalization
   - Try different learning rates: `--lr 0.0001`
   - Increase training epochs: `--epochs 100`

3. **No Songs Found**
   - Check file naming convention
   - Ensure `.npy` files are in the correct directory
   - Verify spectrogram shapes match expected format

## Future Improvements

- [ ] Add real-time audio recording and processing
- [ ] Implement data augmentation techniques
- [ ] Support for different audio formats
- [ ] Web interface for easy interaction
- [ ] Mobile app integration
- [ ] Multi-language song recognition
- [ ] Emotion-based song matching

## Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- PyTorch team for the deep learning framework
- Librosa for audio processing capabilities
- The research community for Siamese network architectures
