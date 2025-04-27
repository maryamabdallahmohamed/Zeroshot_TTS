import librosa
import pandas as pd
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm  # Import tqdm for better progress bar display
import re
import pyarabic.araby as araby
from pyarabic.normalize import normalize_hamza as normalize_text
import string
from num2words import num2words
import torch
import transcription_processing
import Vocab
import transcription_embeddings



print("Loading dataset...")
df=pd.read_csv('Dataset/index.csv')
print(f"Dataset shape: {df.shape}")

print("Checking for mismatches between audio files and transcriptions...")
audio_files = set(os.listdir('Dataset/data/'))
csv_audio_ids = set(df['audio_file'].values)
audio_without_transcription = audio_files - csv_audio_ids
print("Audio files without transcriptions:", audio_without_transcription)
transcription_without_audio = csv_audio_ids - audio_files
print("Transcriptions without audio files:", transcription_without_audio)

print("Cleaning dataset...")
df.drop(df[df['audio_file']=='لشخصك ولأفكارك الـProgressive،"'].index,inplace = True)
df.drop(df[df['text'] == '[موسيقى]'].index,inplace = True)
df = df[~df['audio_file'].isin(transcription_without_audio)]
print(f"Dataset shape after cleaning: {df.shape}")


df.set_index('audio_file', inplace=True)
df['mfccs'] = None
df['cleaned_text'] = None
df['normalized_text'] = None
error_files = []
for audio in tqdm(df.index, desc="Processing Audio Files"):
    audio_path = 'Dataset/data/' + audio
    try:
        signal, rate = librosa.load(audio_path, sr=16000)
        df.at[audio, 'length'] = len(signal) / rate
        cleaned_audio = nr.reduce_noise(signal, rate)
        resampled_audio = librosa.resample(cleaned_audio, orig_sr=rate, target_sr=16000)
        normalized_audio = librosa.util.normalize(resampled_audio)
        mfccs = librosa.feature.mfcc(y=normalized_audio, sr=16000, n_mfcc=13)
        df.at[audio, 'mfccs'] = mfccs.tolist()

        raw_text = df.at[audio, 'text']

        cleaned_text = transcription_processing.transcription_preprocessing(raw_text)
        df.at[audio, 'cleaned_text'] = cleaned_text
    except Exception as e:
        error_message = str(e)
        tqdm.write(f"Error processing {audio}: {error_message}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        df.drop(audio, axis=0, inplace=True)
        error_files.append((audio, error_message))

if error_files:
    print("\nFiles with errors:")
    for file, error in error_files:
        print(f"- {file}: {error}")
    print(f"Total files with errors: {len(error_files)}/{len(df.index)}")
else:
    print("\nAll files processed successfully!")

print("\nAudio length statistics:")
print(df['length'].describe())
print("\nFiltering by audio length (2-10 seconds)...")
df=df[(df['length']<10) & (df['length']>2)]
print(f"Dataset shape after length filtering: {df.shape}")

print("\nTokenizing text...")

df['tokenized_text'] = [Vocab.tokenize_text(text, Vocab.char2idx, max_len=120)
                        for text in tqdm(df['cleaned_text'], desc="Tokenizing texts")]

print("\nGenerating text embeddings...")
model=transcription_embeddings.embedding_model(df)
df=model.generate_embeddings()

print("\nSaving preprocessed data to CSV...")
df.to_csv("Preprocessed_with_embeddings.csv")
print("Preprocessing complete! Output saved to 'Preprocessed_with_embeddings.csv'")