import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from src.paths import WORKED_FOLDER

def convert_price(price_str: str) -> float|None:
    if price_str == "None" or pd.isna(price_str):
        return np.nan
    try:
        # Remove EUR and any other currency symbols, spaces
        clean_price = price_str.replace("EUR", "").strip()
        return float(clean_price)
    except (ValueError, AttributeError):
        return np.nan

def main():
    df = pd.read_parquet(os.path.join(WORKED_FOLDER, 'Beauty_Electronics.parquet'))

    # Quick Structural Overview
    print("Shape: ", df.shape)
    print("\nColumns: ", df.columns)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("Info: ",df.info())
    
    # Domain Balance Check
    df['domain'].value_counts().plot(kind='bar', title='Domain Distribution', color=['pink', 'lightblue'])
    plt.show()

    # Text Lenght Distribution
    df['text_lenght'] = df['combined_text'].str.len()

    plt.hist(df['text_lenght'], bins=10, color='teal')
    plt.title('Distribution of combined_text length')
    plt.xlabel('Characthers')
    plt.ylabel('Frequency')
    plt.show()

    print(df['text_lenght'].describe())
    df = df[(df["text_lenght"] > 50) & (df["text_lenght"] < 5000)]


    # Average Rating & Price Analysis
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
    df['price'] = df['price'].apply(convert_price)
    df = df.dropna(subset=['price'])

    print(df.groupby('domain')[['average_rating', 'price']].describe().T)

    sns.boxplot(x='domain', y='average_rating', data=df)
    sns.boxplot(x='domain', y='price', data=df)
    plt.show()

    # Text Quality Inspection
    print("Duplicate combined_text: ", df['combined_text'].duplicated().sum())
    print("Empty combined_text: ", (df['combined_text'].str.strip()=="").sum())

    df = df.drop_duplicates(subset='combined_text').reset_index(drop=True)
    print("=====Columns=====")
    print(df.columns)
    
    # Keyword Snapshot
    beauty_text = " ".join(df[df['domain']=="Beauty"]['combined_text'].tolist())
    electronic_text = " ".join(df[df['domain']=="Electronics"]['combined_text'].tolist())

    plt.figure(figsize=(16, 8))
    wordcloud = WordCloud(width=1600, height=800, background_color='black').generate(beauty_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes
    plt.tight_layout(pad=0)
    plt.show()

    plt.figure(figsize=(16, 8))
    wordcloud = WordCloud(width=1600, height=800, background_color='black').generate(electronic_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes
    plt.tight_layout(pad=0)
    plt.show()

    df = df.drop(columns=["text_lenght"])
    df.to_parquet(os.path.join(WORKED_FOLDER, "cleaned_full_corpus.parquet"), index=False)
    print("💾 Cleaned corpus rich saved with", len(df), "rows.")


if __name__ == '__main__':
    main()