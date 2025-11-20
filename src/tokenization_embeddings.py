import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import joblib
from src.paths import WORKED_FOLDER, TOKENIZATION_DATA, CHUNKS_PATH, EMBEDDINGS_PATH


def chunk_text(text: str, enc: tiktoken.Encoding, chunk_size: int=200, chunk_overlap: int=30) -> list:
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i: i + chunk_size]
        if not chunk:
            continue
        chunks.append(enc.decode(chunk))

    return chunks

def create_chunks(df: pd.DataFrame, path=CHUNKS_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        print("📂 Loading existing chunks file...")
        chunks_df = pd.read_parquet(path)
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        
        rows = []
        chunk_id = 0
        df = df.reset_index(drop=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            text = row.combined_text

            chunk_list = chunk_text(
                text=text,
                enc=enc,
                chunk_size=200,
                chunk_overlap=30
            )

            for chunk in chunk_list:
                rows.append({
                    "doc_id": idx,
                    "chunk_id": chunk_id,
                    "domain": row["domain"],
                    "price": row["price"],
                    "average_rating": row["average_rating"],
                    "title": row["title"],
                    "categories": row["categories"],
                    "text": chunk
                })
                chunk_id += 1
        
        chunks_df = pd.DataFrame(rows)

        # Save metadata for later
        chunks_df.to_parquet(path, index=False)
        print("💾 chunks_parquet saved successfully.")
        

    return chunks_df

def create_embeddings(chunks_df: pd.DataFrame, model_name="BAAI/bge-base-en-v1.5", path=EMBEDDINGS_PATH, force_compute=False) -> tuple[np.ndarray, str]:    
    if os.path.exists(path) and not force_compute:
        print("📂 Loading existing embeddings...")
        embeddings = joblib.load(path)
        
    else:
        print("🔨 Creating new embeddings...")
        model = SentenceTransformer(model_name)   # uses CPU/GPU automatically
        BATCH_SIZE = 256                              # tune to VRAM

        # Encoding
        texts = chunks_df["text"].tolist()
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        joblib.dump(embeddings, path)
        print("💾 Embeddings saved successfully.")
    print("🔢 Embedding shape: ", embeddings.shape)

    # Save metadat for later
    return embeddings, model_name

def build_index(embeddings: np.ndarray, chunks_df: pd.DataFrame, model_name):
    dim = embeddings.shape[1]
    
    index_all = faiss.IndexFlatIP(dim)
    index_all.add(embeddings)                       # type: ignore

    # domain masks
    beauty_mask = chunks_df["domain"] == "Beauty"
    electronics_mask = chunks_df["domain"] == "Electronics"

    index_beauty = faiss.IndexFlatIP(dim)
    index_beauty.add(embeddings[beauty_mask])       # type: ignore

    index_elec = faiss.IndexFlatIP(dim)
    index_elec.add(embeddings[electronics_mask])    # type: ignore

    # Save 
    save_artifacts(index_all, index_beauty, beauty_mask, index_elec, electronics_mask, model_name, dim)
    

def save_artifacts(index_all: faiss.Index, index_beauty: faiss.Index, beauty_mask: pd.Series, index_elec: faiss.Index, 
                   electronics_mask: pd.Series, model_name: str, dim: int):
    faiss.write_index(index_all, os.path.join(TOKENIZATION_DATA, "faiss_all.index"))
    faiss.write_index(index_beauty, os.path.join(TOKENIZATION_DATA, "faiss_beauty.index"))
    faiss.write_index(index_elec, os.path.join(TOKENIZATION_DATA, "faiss_electronics.index"))

    # Map arrays
    meta = {
        "model_name": model_name,
        "dim": dim,
        "doc_table_path": CHUNKS_PATH,
        "beauty_indices": np.where(beauty_mask)[0].astype("int64"),
        "electronics_indices": np.where(electronics_mask)[0].astype("int64"),   
    }
    joblib.dump(meta, os.path.join(TOKENIZATION_DATA, "meta.joblib"))

    print("💾 Saved indices + metadata to ", TOKENIZATION_DATA)

def prepare_data():
    df = pd.read_parquet(os.path.join(WORKED_FOLDER, "cleaned_full_corpus.parquet"))
    
    df['doc_id'] = np.arange(len(df), dtype=np.int64)

    # 1) Chunking (token-based, overlap)
    chunks_df = create_chunks(df)

    # 2) Embeddings (SBERT)
    embeddings, model_name = create_embeddings(chunks_df)

    # 3) Build FAISS indices and save 
    #   - combined
    #   - per-domain (for balanced retrieval or filtering)
    build_index(embeddings, chunks_df, model_name)



def main():
    prepare_data()

if __name__ == '__main__':
    main()