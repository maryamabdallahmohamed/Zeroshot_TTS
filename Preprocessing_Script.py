import librosa
import pandas as pd
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
import os
import soundfile as sf  
import zipfile  
from tqdm.auto import tqdm 
import Vocab
from transcription_processing import sentence_preprocessing
import transcription_embeddings 

output_audio_dir = "processed_audio"
os.makedirs(output_audio_dir, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('/kaggle/input/egyptian-arabic-lines/index.csv')
print(f"Dataset shape: {df.shape}")

print("Checking for mismatches between audio files and transcriptions...")
audio_files = set(os.listdir('/kaggle/input/egyptian-arabic-lines/data'))
csv_audio_ids = set(df['audio_file'].values)
audio_without_transcription = audio_files - csv_audio_ids
print("Audio files without transcriptions:", audio_without_transcription)
transcription_without_audio = csv_audio_ids - audio_files
print("Transcriptions without audio files:", transcription_without_audio)

print("Cleaning dataset...")
df.drop(df[df['audio_file'] == 'لشخصك ولأفكارك الـProgressive،"'].index, inplace=True)
df.drop(df[df['text'] == '[موسيقى]'].index, inplace=True)
df = df[~df['audio_file'].isin(transcription_without_audio)]
print(f"Dataset shape after cleaning: {df.shape}")
output_audio_dir = "processed_audio"
os.makedirs(output_audio_dir, exist_ok=True)

df.set_index('audio_file', inplace=True)
df['mfccs'] = None
df['cleaned_text'] = None
df['normalized_text'] = None
error_files = [] 
print(df.shape)
for audio in tqdm(df.index, desc="Processing Audio Files"):
    audio_path = 'Dataset/data/' + audio
    try:
        signal, rate = librosa.load(audio_path, sr=16000)
        length = len(signal) / rate
        df.at[audio, 'length'] = length
        
        if (length >=3  and length <= 10):
            cleaned_audio = nr.reduce_noise(signal, rate)
            resampled_audio = librosa.resample(cleaned_audio, orig_sr=rate, target_sr=16000)
            normalized_audio = librosa.util.normalize(resampled_audio)
            
            # Save processed audio file
            output_path = os.path.join(output_audio_dir, f"processed_{audio}")
            sf.write(output_path, normalized_audio, 16000)
            mfccs = librosa.feature.mfcc(y=normalized_audio, sr=16000, n_mfcc=13)
            df.at[audio, 'mfccs'] = mfccs.tolist()

            raw_text = df.at[audio, 'text']
            cleaned_text = sentence_preprocessing(raw_text)
            df.at[audio, 'cleaned_text'] = cleaned_text
        else:
            df.drop(audio, axis=0, inplace=True)
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

print(f"Dataset shape after length filtering: {df.shape}")

print("\nTokenizing text...")
df['tokenized_text'] = [Vocab.tokenize_text(text, Vocab.char2idx, max_len=120)
                       for text in tqdm(df['cleaned_text'], desc="Tokenizing texts")]

print("\nGenerating text embeddings...")
model = transcription_embeddings.embedding_model(df)
df = model.generate_embeddings()

print("\nSaving preprocessed data to CSV...")
df.to_csv("Preprocessed_with_embeddings.csv")
print("Preprocessing complete! Output saved to 'Preprocessed_with_embeddings.csv'")

zip_filename = "processed_audio_files.zip"
print(f"\nCreating zip archive: {zip_filename}")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_audio_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.join("processed_audio", file))

print(f"Processed audio files saved and zipped in {zip_filename}")