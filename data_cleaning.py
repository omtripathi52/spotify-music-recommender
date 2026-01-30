"""
Data Cleaning Script for Spotify Million Song Dataset
Downloads and cleans the dataset for the recommendation system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def download_dataset():
    """Download dataset from Kaggle using kagglehub"""
    import kagglehub
    
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")
    
    # Find CSV file in downloaded directory
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
    
    df = pd.read_csv(csv_files[0])
    print(f"Downloaded {len(df)} songs from Kaggle")
    
    # Save locally
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/Music info.csv", index=False)
    print("Saved to data/Music info.csv")
    
    return df

def load_data():
    """Load dataset - download if not found locally"""
    local_file = Path("data/Music info.csv")
    
    if local_file.exists():
        print(f"Loading local dataset from {local_file}")
        return pd.read_csv(local_file)
    else:
        print("Local dataset not found. Downloading from Kaggle...")
        return download_dataset()

def clean_data(df, max_songs=5000):
    """Clean and preprocess the dataset"""
    print(f"\nCleaning {len(df)} records...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop rows missing critical columns
    df = df.dropna(subset=['name', 'artist'])
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Normalize audio features to 0-1 range if needed
    feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                    'instrumentalness', 'liveness', 'speechiness']
    
    for col in feature_cols:
        if col in df.columns:
            if df[col].max() > 1:  # Not already normalized
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Limit dataset size to avoid memory issues
    # 50k songs would need 19GB RAM for similarity matrix!
    if len(df) > max_songs:
        print(f"Sampling {max_songs} songs (full dataset too large for similarity matrix)")
        df = df.sample(n=max_songs, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset: {len(df)} records")
    return df

def main():
    """Main function to run data cleaning pipeline"""
    print("=" * 60)
    print("SPOTIFY DATASET CLEANING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} records with columns: {list(df.columns)[:5]}...")
    
    # Clean data
    df = clean_data(df)
    
    # Save cleaned data
    output_path = "data/cleaned_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")
    
    # Show summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total songs: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    main()
