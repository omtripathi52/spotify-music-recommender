"""
Content-Based Filtering Recommender System
Recommends songs based on audio features and metadata similarity
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class ContentBasedRecommender:
    def __init__(self, data_path=None, df=None):
        """Initialize the content-based recommender"""
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self.similarity_matrix = None
    
    def fit(self, df=None):
        """Fit the recommender with data"""
        if df is not None:
            self.df = df
        self.prepare_features()
        self.compute_similarity()
        
    def prepare_features(self):
        """Prepare feature matrix from audio features"""
        print("Preparing features for content-based filtering...")
        
        # Select relevant audio features
        # Common features in Spotify dataset: danceability, energy, key, loudness,
        # mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo
        feature_columns = [col for col in self.df.columns if col in [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms', 'year'
        ]]
        
        if not feature_columns:
            # Fallback to numeric columns if specific features not found
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove id-like columns
            feature_columns = [col for col in feature_columns if not col.lower().endswith('_id')]
        
        print(f"Using features: {feature_columns}")
        
        # Extract and scale features
        self.feature_matrix = self.scaler.fit_transform(self.df[feature_columns])
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
    def compute_similarity(self):
        """Compute cosine similarity matrix"""
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        
    def get_recommendations(self, song_title, artist_name=None, n_recommendations=10):
        """Get song recommendations based on content similarity"""
        # Find the song index
        if artist_name:
            mask = (self.df['name'].str.lower() == song_title.lower()) & \
                   (self.df['artist'].str.lower() == artist_name.lower())
        else:
            mask = self.df['name'].str.lower() == song_title.lower()
        
        matching_songs = self.df[mask]
        
        if len(matching_songs) == 0:
            return f"Song '{song_title}' not found in database"
        
        song_idx = matching_songs.index[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[song_idx]))
        
        # Sort by similarity (excluding the song itself)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        # Get recommended songs
        recommended_indices = [i[0] for i in similarity_scores]
        recommendations = self.df.iloc[recommended_indices][['name', 'artist']]
        recommendations['similarity_score'] = [i[1] for i in similarity_scores]
        
        return recommendations
    
    def recommend_by_features(self, features_dict, n_recommendations=10):
        """Recommend songs based on desired features"""
        # Create feature vector from input
        feature_vector = np.array([features_dict.get(col, 0) for col in self.df.columns 
                                   if col in features_dict]).reshape(1, -1)
        
        # Scale the features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Compute similarity with all songs
        similarities = cosine_similarity(scaled_features, self.feature_matrix)[0]
        
        # Get top recommendations
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        recommendations = self.df.iloc[top_indices][['name', 'artist']]
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations

def main():
    # Initialize recommender
    recommender = ContentBasedRecommender("data/cleaned_data.csv")
    
    # Prepare features and compute similarity
    recommender.prepare_features()
    recommender.compute_similarity()
    
    # Example: Get recommendations for a song
    print("\n" + "="*50)
    print("CONTENT-BASED RECOMMENDATIONS")
    print("="*50)
    
    # Get a random song from the dataset
    sample_song = recommender.df.sample(1).iloc[0]
    print(f"\nSeed Song: {sample_song['name']} by {sample_song['artist']}")
    
    recommendations = recommender.get_recommendations(
        sample_song['name'], 
        sample_song['artist'], 
        n_recommendations=10
    )
    
    print("\nRecommended Songs:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
