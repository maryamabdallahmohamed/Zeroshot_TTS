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

## Installation

To install all required dependencies, run:

```bash
# Make the installation script executable
chmod +x install_requirements.sh

# Run the installation script
./install_requirements.sh
```

Or install directly using pip:

```bash
python -m pip install -r requirements.txt
```

## Project Structure

- `Audio_Preprocessing.ipynb`: Notebook for audio preprocessing and feature extraction
- `visualization_notebook.ipynb`: Notebook for visualizing audio features and transformations
- `transcription.py`: Module containing text preprocessing functions
- `requirements.txt`: List of required Python packages
- `install_requirements.sh`: Script to install all dependencies

## Usage

1. Place your audio files in the `Dataset/data/` directory
2. Ensure you have a CSV file at `Dataset/index.csv` with columns for audio filenames and transcriptions
3. Run the preprocessing notebook to generate features and embeddings
4. Use the visualization notebook to explore the audio data

## Features

- **Audio Preprocessing**: Noise reduction, resampling to 16kHz, and normalization
- **Feature Extraction**: MFCC and mel spectrogram generation
- **Text Preprocessing**: Arabic text normalization, number-to-word conversion
- **Tokenization**: Character-level tokenization with special tokens
- **Embedding Generation**: Using sentence-transformers for text embedding