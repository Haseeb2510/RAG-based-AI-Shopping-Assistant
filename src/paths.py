import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_BEAUTY = os.path.join(PROJECT_ROOT, 'data/raw_meta_All_Beauty')
os.makedirs(RAW_BEAUTY, exist_ok=True)

RAW_ELETRONICS = os.path.join(PROJECT_ROOT, 'data/raw_meta_Electronics')
os.makedirs(RAW_ELETRONICS, exist_ok=True)

WORKED_FOLDER = os.path.join(PROJECT_ROOT, 'data/worked')
os.makedirs(WORKED_FOLDER, exist_ok=True)

TOKENIZATION_DATA = os.path.join(PROJECT_ROOT, "data/tokenized")
os.makedirs(TOKENIZATION_DATA, exist_ok=True)

CHUNKS_PATH = os.path.join(TOKENIZATION_DATA, 'chunks.parquet')
EMBEDDINGS_PATH =  os.path.join(TOKENIZATION_DATA, "embeddings.joblib")
META_PATH = os.path.join(TOKENIZATION_DATA, "meta.joblib")
