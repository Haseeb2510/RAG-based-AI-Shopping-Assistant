import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from src.paths import WORKED_FOLDER, RAW_BEAUTY, RAW_ELETRONICS


def get_beauty_categories(title):
    title_lower = title.lower()
    categories = []
    
    category_keywords = {
        "Hair Care": ["hair", "shampoo", "conditioner", "scalp", "styling", "treatment"],
        "Skin Care": ["skin", "face", "serum", "cream", "moisturizer", "cleanser", "toner"],
        "Face Makeup": ["foundation", "concealer", "powder", "primer", "bb cream", "cc cream"],
        "Lip Products": ["lipstick", "lip gloss", "lip balm", "lip liner"],
        "Eye Makeup": ["mascara", "eyeliner", "eyeshadow", "eyebrow", "lash"],
        "Fragrance": ["perfume", "fragrance", "cologne", "scent"],
        "Sun Care": ["sunscreen", "spf", "sunblock", "uv protection"],
        "Face Masks": ["mask", "face mask", "sheet mask", "clay mask"],
        "Beauty Tools": ["brush", "sponge", "applicator", "beauty tool", "mirror"],
        "Bath & Body": ["body", "bath", "shower", "lotion", "soap", "scrub"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in title_lower for keyword in keywords):
            categories.append(category)
    
    return categories if categories else ["Other Beauty"]

def flatten_text(x):
    # Case 1: Empty or NaN
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    # Case 2: Real numpy array (even if empty)
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ''
        return ' '.join(map(str, x))
    # Case 3: Python list
    if isinstance(x, list):
        if len(x) == 0:
            return ''
        return ' '.join(map(str, x))
    # Case 4: Something else (string or object)
    return str(x)

def combine_text(row):
    parts = [
        f"Title: {row['title']}",
        f"Features: {row['features']}",
        f"Description: {row['description']}"
    ]
    return ". ".join(p for p in parts if p and p.strip())

def main():
    df1 = pd.read_parquet(os.path.join(RAW_BEAUTY, "full-00000-of-00001.parquet"))
    df1['domain'] = 'Beauty'
    df1["categories"] = df1["title"].apply(get_beauty_categories)
    df2 = pd.read_parquet(os.path.join(RAW_ELETRONICS, "full-00000-of-00010.parquet"))
    df2['domain'] = 'Electronics'
    df3 = pd.read_parquet(os.path.join(RAW_ELETRONICS, "full-00001-of-00010.parquet"))
    df3['domain'] = 'Electronics'

    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    print(df.head())
    print(df.info())
    print(df.describe().T)
    print(df.shape)

    columns_to_keep = [
        "main_category",       # domain label
        "title",               # product name
        "average_rating",      # numeric context
        "features",            # bullet/spec text
        "description",         # main product info
        "price",               # price
        "categories",          # product hierarchy
        "domain"               # product type
    ]
    df = df[columns_to_keep]
    print(df.columns)
    
    for col in columns_to_keep:
        df[col] = df[col].apply(flatten_text)


    df["combined_text"] = df.apply(combine_text, axis=1)

    df.to_parquet(os.path.join(WORKED_FOLDER, "Beauty_Electronics.parquet"), index=False)
    print("✅ Cleaned corpus saved with", len(df), "rows.")
    print(df.sample(5).T)

if __name__ == '__main__':
    main()