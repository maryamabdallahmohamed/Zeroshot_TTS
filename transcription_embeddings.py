from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm


class embedding_model:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', 
                 device='mps', cache_folder='embedding_model_cache'):
        # Initialize model
        self.model = SentenceTransformer(
            model_name,
            device=device,  # 'mps', 'cpu', or 'cuda' depending on system
            cache_folder=cache_folder
        )
        
    def process_file(self, input_file='processed.csv', output_file='processed_with_embeddings.csv', 
                    chunk_size=800, text_column='clean_text'):
        """Process a CSV file in chunks and add embeddings"""
        # Prepare output file
        first_chunk = True
        
        # Process CSV in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            print(f"Processing chunk with {len(chunk)} rows...")
            
            # Ensure text column exists
            if text_column not in chunk.columns:
                raise ValueError(f"'{text_column}' column not found in input CSV.")
                
            # Generate embeddings
            chunk = self.add_embeddings_to_df(chunk, text_column)
            
            # Append to output file
            if first_chunk:
                chunk.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk.to_csv(output_file, index=False, header=False, mode='a')
                
        print(f"All chunks processed. Saved to {output_file}.")
        
    def add_embeddings_to_df(self, df, text_column='clean_text', embedding_column='text_embedding'):
        """Add embeddings to a dataframe"""
        # Convert to list of texts
        texts = df[text_column].astype(str).tolist()
        
        # Generate embeddings in batch
        embeddings = self.model.encode(texts, batch_size=8, show_progress_bar=True)
        df[embedding_column] = [emb.tolist() for emb in embeddings]
        
        return df

