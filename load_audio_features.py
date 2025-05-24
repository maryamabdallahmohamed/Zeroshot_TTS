import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import glob
import librosa
import gc
import logging
from tqdm import tqdm

# ------------------- Configuration -------------------
AUDIO_DIR = 'phase2_data/subset_80k_audio'
MODEL_NAME = "openai/whisper-base"
CACHE_DIR = './cache'
BATCH_SIZE = 1  # Ensure output shape has batch_size = 1
MAX_LENGTH_AUDIO = 160000
MAX_LENGTH_FEATURES = 1500
RANDOM_SEED = 42
DEVICE = torch.device("mps")  # Use "cuda" if GPU is available

# ------------------- Setup -------------------
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w')
torch.manual_seed(RANDOM_SEED)

# ------------------- Dataset -------------------
class ArabicProcessedAudios(Dataset):
    def __init__(self, audio_path, max_length=160000):
        self.audio_files = glob.glob(os.path.join(audio_path, "*.mp3"))
        self.max_length = max_length

    def __getitem__(self, idx):
        file = self.audio_files[idx]
        try:
            audio, _ = librosa.load(file, sr=16000)
            audio = librosa.util.normalize(audio)
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            else:
                audio = audio[:self.max_length]
            return audio, os.path.basename(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            return None, None

    def __len__(self):
        return len(self.audio_files)

# ------------------- Feature Extraction -------------------
def extract_whisper_features(model, audio, processor, layer=-1, max_length=1500):
    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(inputs["input_features"])
            features = encoder_outputs.last_hidden_state if layer == -1 else encoder_outputs.hidden_states[layer]

        # features shape: (1, sequence_length, 512)
        sequence_length = features.shape[1]
        if sequence_length > max_length:
            features = features[:, :max_length, :]
        elif sequence_length < max_length:
            pad_len = max_length - sequence_length
            features = F.pad(features, (0, 0, 0, pad_len))  # Pad sequence_length dimension

        return features.cpu()
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return None

# ------------------- Main Script -------------------
def extract_features():
    print("Loading dataset...")
    dataset = ArabicProcessedAudios(AUDIO_DIR, max_length=MAX_LENGTH_AUDIO)
    print(f"Number of audio files: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading Whisper model and processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=os.path.join(CACHE_DIR, "processor"))
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=os.path.join(CACHE_DIR, "model"))
    model.to(DEVICE)

    print("Extracting audio features...")
    all_features = []
    for batch in tqdm(dataloader, desc="Processing Audio Files"):
        audios, filenames = batch
        for audio, filename in zip(audios, filenames):
            if audio is None:
                continue
            features = extract_whisper_features(model, audio.numpy(), processor, layer=-1, max_length=MAX_LENGTH_FEATURES)
            if features is not None and features.shape == (1, MAX_LENGTH_FEATURES, 512):
                all_features.append((filename, features))
            else:
                logging.warning(f"Feature shape mismatch or None for file: {filename}")
            del features, audio
            gc.collect()
            torch.mps.empty_cache() if torch.mps.is_available() else None

    print(f"Extraction complete. Total feature sets: {len(all_features)}")
    return all_features

