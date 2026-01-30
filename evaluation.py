"""
Evaluation Metrics for Recommendation System
"""

import pandas as pd
import numpy as np
from content_based_filtering import ContentBasedRecommender


def evaluate_recommendations(df, recommender, n_samples=100, n_recommendations=10):
    """
    Evaluate recommendation quality using multiple metrics
    """
    results = {
        'intra_list_diversity': [],
        'genre_coverage': [],
        'artist_diversity': [],
        'avg_similarity': []
    }
    
    # Sample songs for evaluation
    sample_songs = df.sample(min(n_samples, len(df)))
    
    for _, song in sample_songs.iterrows():
        recs = recommender.get_recommendations(song['name'], n_recommendations=n_recommendations)
        
        if isinstance(recs, str):  # Error
            continue
        
        if len(recs) == 0:
            continue
        
        # 1. Intra-list Diversity (how different are recommendations from each other)
        unique_artists = recs['artist'].nunique()
        artist_diversity = unique_artists / len(recs)
        results['artist_diversity'].append(artist_diversity)
        
        # 2. Genre Coverage
        if 'genre' in recs.columns:
            unique_genres = recs['genre'].nunique()
            results['genre_coverage'].append(unique_genres)
        
        # 3. Average Similarity Score
        if 'similarity_score' in recs.columns:
            results['avg_similarity'].append(recs['similarity_score'].mean())
    
    # Calculate final metrics
    metrics = {
        'Artist Diversity': np.mean(results['artist_diversity']) if results['artist_diversity'] else 0,
        'Avg Genre Coverage': np.mean(results['genre_coverage']) if results['genre_coverage'] else 0,
        'Avg Similarity Score': np.mean(results['avg_similarity']) if results['avg_similarity'] else 0,
        'Songs Evaluated': len(results['artist_diversity'])
    }
    
    return metrics


def calculate_feature_importance(df, recommender):
    """
    Calculate which features contribute most to recommendations
    """
    feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                    'instrumentalness', 'liveness', 'speechiness', 'tempo']
    
    # Get feature variances (higher variance = more discriminative)
    available_features = [f for f in feature_cols if f in df.columns]
    
    importance = {}
    for feature in available_features:
        variance = df[feature].var()
        importance[feature] = variance
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    recommender = ContentBasedRecommender(df=df)
    recommender.fit()
    
    print("Evaluating Content-Based Recommender...")
    metrics = evaluate_recommendations(df, recommender, n_samples=50)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
