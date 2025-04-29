from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
class embedding_model:
    def __init__(self, df):
        self.df = df
        self.model =SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='mps',cache_folder='embedding_model_cache')  
 
        self.df['text_embedding'] = None
    def generate_embeddings(self):
        for idx in tqdm(self.df.index, desc="Generating embeddings"):
            text = self.df.at[idx, 'clean_text']
            embedding = self.model.encode(text)
            self.df.at[idx, 'text_embedding'] = embedding.tolist()
        print(f"Embedding shape: {np.array(self.df['text_embedding'].iloc[0]).shape}")
        return self.df