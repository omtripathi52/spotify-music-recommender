"""
Popularity-Based Recommender
Recommends popular songs, optionally filtered by genre
"""

import pandas as pd
import numpy as np


class PopularityRecommender:
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.popularity_scores = None
        self._compute_popularity()
    
    def _compute_popularity(self):
        """Compute popularity scores based on available metrics"""
        # Create a composite popularity score
        # Using energy, danceability as proxies (higher = more popular typically)
        self.df = self.df.copy()
        
        if 'popularity' not in self.df.columns:
            # Create synthetic popularity from audio features
            self.df['popularity_score'] = (
                self.df['energy'] * 0.3 +
                self.df['danceability'] * 0.3 +
                self.df['valence'] * 0.2 +
                (1 - self.df['acousticness']) * 0.2
            )
        else:
            self.df['popularity_score'] = self.df['popularity'] / 100
    
    def get_recommendations(self, genre=None, n_recommendations=10, exclude_songs=None):
        """Get popular song recommendations"""
        df_filtered = self.df.copy()
        
        # Filter by genre if specified
        if genre and genre != "All":
            df_filtered = df_filtered[df_filtered['genre'].str.contains(genre, case=False, na=False)]
        
        # Exclude certain songs
        if exclude_songs:
            df_filtered = df_filtered[~df_filtered['name'].isin(exclude_songs)]
        
        # Sort by popularity and return top N
        recommendations = df_filtered.nlargest(n_recommendations, 'popularity_score')
        
        return recommendations[['name', 'artist', 'genre', 'popularity_score']].reset_index(drop=True)
    
    def get_genres(self):
        """Get unique genres"""
        genres = self.df['genre'].dropna().unique().tolist()
        return ["All"] + sorted(genres)
