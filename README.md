# Zeroshot_TTS

A zero-shot text-to-speech (TTS) system for Arabic language processing, featuring audio preprocessing, text normalization, and embedding generation.

## Project Overview

This project implements a zero-shot text-to-speech system that can process Arabic audio files and their transcriptions. It includes:

- Audio preprocessing (noise reduction, resampling, normalization)
- Text preprocessing for Arabic language
- Feature extraction (MFCC, mel spectrograms)
- Text tokenization and embedding generation
- Visualization tools for audio analysis

## Requirements

All required libraries are listed in `requirements.txt`. The main dependencies include:

- **Audio Processing**: librosa, noisereduce
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib
- **NLP and Text Processing**: pyarabic, num2words
- **Deep Learning Frameworks**: tensorflow, torch, sentence-transformers
- **Utilities**: tqdm
- **Evaluation**: jiwer
- **Jupyter**: jupyter, ipython

## System Requirements

- Python 3.8 or higher
- macOS with Apple Silicon (for MPS acceleration) or any system with CPU support
- At least 8GB of RAM recommended
- Sufficient disk space for audio files and processed data

## Installation

To install all required dependencies, run:

```bash
python3 -m pip install -r requirements.txt
```

### Notes for macOS Users

The project is optimized for Apple Silicon Macs using Metal Performance Shaders (MPS) for hardware acceleration. The requirements.txt file includes tensorflow-macos by default.

### Notes for Other Platforms

If you're not using macOS, edit the requirements.txt file to uncomment the standard tensorflow line and comment out the tensorflow-macos line.

## Project Structure

- `Audio_Preprocessing.ipynb`: Notebook for audio preprocessing and feature extraction
- `visualization_notebook.ipynb`: Notebook for visualizing audio features and transformations
- `Preprocessing_Script.py`: Script version of the preprocessing pipeline for batch processing
- `transcription_processing.py`: Module containing text preprocessing functions for Arabic text
- `transcription_embeddings.py`: Module for generating text embeddings using sentence-transformers
- `Vocab.py`: Module for text tokenization and vocabulary management
- `requirements.txt`: List of required Python packages



## Usage

1. Place your audio files in the `Dataset/data/` directory
2. Ensure you have a CSV file at `Dataset/index.csv` with columns for audio filenames and transcriptions
3. Run the preprocessing notebook to generate features and embeddings:
   ```bash
   jupyter notebook Audio_Preprocessing.ipynb
   ```
4. Alternatively, use the preprocessing script for batch processing:
   ```bash
   python Preprocessing_Script.py
   ```
5. Use the visualization notebook to explore the audio data:
   ```bash
   jupyter notebook visualization_notebook.ipynb
   ```

## Processing Pipeline

The project follows this processing pipeline:

1. **Data Loading**: Load audio files and their transcriptions from the dataset
2. **Audio Preprocessing**:
   - Filter audio files by length (3-10 seconds)
   - Apply noise reduction
   - Resample to 16kHz
   - Normalize audio signals
3. **Feature Extraction**:
   - Generate MFCC features from processed audio
4. **Text Preprocessing**:
   - Normalize Arabic text (remove diacritics, standardize characters)
   - Convert numbers to words
   - Handle mixed Arabic-English text
5. **Tokenization**:
   - Convert text to token sequences using character-level tokenization
   - Add special tokens (SOS, EOS, PAD)
6. **Embedding Generation**:
   - Generate text embeddings using sentence-transformers
7. **Output**:
   - Save processed data to CSV
   - Save processed audio files

## Features

- **Audio Preprocessing**: Noise reduction, resampling to 16kHz, and normalization
- **Feature Extraction**: MFCC and mel spectrogram generation
- **Text Preprocessing**: Arabic text normalization, number-to-word conversion, diacritics removal
- **Tokenization**: Character-level tokenization with special tokens (PAD, UNK, SOS, EOS)
- **Embedding Generation**: Using sentence-transformers (all-mpnet-base-v2) for text embedding
- **Hardware Acceleration**: Support for Apple MPS (Metal Performance Shaders) for faster processing on macOS
- **Batch Processing**: Efficient processing of multiple audio files with error handling

## Project Purpose

This project aims to develop a zero-shot text-to-speech system for the Arabic language, which can generate natural-sounding speech without requiring extensive training data for each new voice. The preprocessing pipeline established in this project creates the foundation for training TTS models by:

1. Cleaning and standardizing audio inputs
2. Normalizing and tokenizing Arabic text
3. Generating high-quality embeddings that capture semantic meaning
4. Creating a consistent dataset format for model training

## Future Work

- Implement encoder-decoder architecture for TTS generation
- Add support for voice cloning with few-shot learning
- Improve Arabic text normalization for dialectal variations
- Expand the dataset with more diverse speakers
- Evaluate and benchmark against existing TTS systems