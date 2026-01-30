"""
Hybrid Recommender System
Combines Content-Based and Popularity-Based approaches
"""

import pandas as pd
import numpy as np
from content_based_filtering import ContentBasedRecommender
from popularity_recommender import PopularityRecommender


class HybridRecommender:
    def __init__(self, df):
        """Initialize hybrid recommender"""
        self.df = df
        self.content_based = ContentBasedRecommender(df=df)
        self.content_based.fit()
        self.popularity = PopularityRecommender(df)
    
    def get_recommendations(self, song_title, artist_name=None, n_recommendations=10,
                           content_weight=0.7, popularity_weight=0.3):
        """
        Get hybrid recommendations combining content similarity and popularity
        
        Args:
            song_title: Name of the seed song
            artist_name: Artist name (optional)
            n_recommendations: Number of recommendations
            content_weight: Weight for content-based scores (0-1)
            popularity_weight: Weight for popularity scores (0-1)
        """
        # Normalize weights
        total = content_weight + popularity_weight
        content_weight = content_weight / total
        popularity_weight = popularity_weight / total
        
        # Get content-based recommendations (more than needed for re-ranking)
        content_recs = self.content_based.get_recommendations(
            song_title, artist_name, n_recommendations=n_recommendations * 3
        )
        
        if isinstance(content_recs, str):  # Error message
            return content_recs
        
        # Add popularity scores
        content_recs = content_recs.merge(
            self.df[['name', 'artist', 'genre']].drop_duplicates(),
            on=['name', 'artist'],
            how='left'
        )
        
        # Get popularity scores
        pop_scores = self.popularity.df[['name', 'artist', 'popularity_score']].drop_duplicates()
        content_recs = content_recs.merge(pop_scores, on=['name', 'artist'], how='left')
        content_recs['popularity_score'] = content_recs['popularity_score'].fillna(0.5)
        
        # Normalize similarity scores to 0-1
        max_sim = content_recs['similarity_score'].max()
        if max_sim > 0:
            content_recs['norm_similarity'] = content_recs['similarity_score'] / max_sim
        else:
            content_recs['norm_similarity'] = content_recs['similarity_score']
        
        # Calculate hybrid score
        content_recs['hybrid_score'] = (
            content_recs['norm_similarity'] * content_weight +
            content_recs['popularity_score'] * popularity_weight
        )
        
        # Sort and return top N
        result = content_recs.nlargest(n_recommendations, 'hybrid_score')
        
        return result[['name', 'artist', 'genre', 'similarity_score', 'popularity_score', 'hybrid_score']].reset_index(drop=True)


if __name__ == "__main__":
    # Test
    df = pd.read_csv("data/cleaned_data.csv")
    hybrid = HybridRecommender(df)
    
    print("Testing Hybrid Recommender...")
    recs = hybrid.get_recommendations("The Ghost of Tom Joad", n_recommendations=5)
    print(recs)
