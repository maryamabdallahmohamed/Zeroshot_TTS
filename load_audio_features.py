import os, glob, gc, logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

AUDIO_DIR = "/kaggle/input/tts-dataset/FINAL_TTS_DATA"
MODEL_NAME= "openai/whisper-base"
CACHE_DIR = "./hf_cache"
BATCH_SIZE = 2
MAX_WAV_LEN = 160_000          
MEL_PAD_LEN = 3000           
FEATURE_SEQ_LEN = 1500         
DEVICE = (torch.device("mps")  if torch.backends.mps.is_available()  else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

logging.basicConfig(level=logging.INFO, filename="training.log", filemode="w")
torch.manual_seed(42)


class ArabicProcessedAudios(Dataset):
    def __init__(self, audio_path: str, max_length: int = MAX_WAV_LEN):
        self.audio_path = audio_path 
        all_files = glob.glob(os.path.join(self.audio_path, "*.mp3"))[:20000] 
        self.files = []
        self.maxlen = max_length

 
        for f in tqdm(all_files, desc="Validating audio files"):
            try:
                librosa.load(f, sr=16_000, duration=0.1) 
                self.files.append(f)
            except Exception as e:
                logging.warning(f"Skipping {f}: {e}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, _ = librosa.load(path, sr=16_000)
        audio = librosa.util.normalize(audio)

        if len(audio) < self.maxlen:
            audio = np.pad(audio, (0, self.maxlen - len(audio)))
        else:
            audio = audio[: self.maxlen]

        return torch.tensor(audio, dtype=torch.float32), os.path.basename(path)

def collate_fn(batch):
    audios, names = zip(*batch)
    return torch.stack(audios), names 


def whisper_features(model, processor, wav_batch: torch.Tensor,
                     out_len: int = FEATURE_SEQ_LEN, layer: int = -1) -> torch.Tensor:
    """
    wav_batch: (B, MAX_WAV_LEN) on CPU/MPS/GPU
    returns:   (B, out_len, 512) on CPU
    """
    wav_list = [wav.cpu().numpy() for wav in wav_batch]
    
    input_features_list = []
    for wav_array in wav_list:
        features = processor.feature_extractor(
            wav_array,
            sampling_rate=16_000,
            return_tensors="pt"
        )
        input_features_list.append(features["input_features"])
    
    input_features = torch.cat(input_features_list, dim=0)  
    
    input_features = input_features.to(DEVICE)

    with torch.no_grad():
        enc_out = model.get_encoder()(input_features)
        feats   = enc_out.last_hidden_state if layer == -1 else enc_out.hidden_states[layer]  

    if feats.shape[1] > out_len:
        feats = feats[:, :out_len, :]
    elif feats.shape[1] < out_len:
        pad = out_len - feats.shape[1]
        feats = F.pad(feats, (0, 0, 0, pad))

    return feats.cpu()   


def get_all_features() :
    ds = ArabicProcessedAudios(AUDIO_DIR) 
    print(f"✓ valid audio files found: {len(ds)}")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=0, collate_fn=collate_fn)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE).eval()

    out = [] 

    for wavs, names in tqdm(dl, desc="Extracting Whisper features"):
        feats = whisper_features(model, processor, wavs)  
        for i, n in enumerate(names):
            out.append((n, feats[i]))  
        # memory housekeeping
        del feats, wavs
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

    return out
