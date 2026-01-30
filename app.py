"""
Spotify Music Recommender System - Streamlit App
A content-based and hybrid music recommendation system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from content_based_filtering import ContentBasedRecommender
from popularity_recommender import PopularityRecommender
from hybrid_recommender import HybridRecommender
from evaluation import evaluate_recommendations, calculate_feature_importance

# Page config
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_and_models():
    """Load data and initialize models"""
    data_path = Path("data/cleaned_data.csv")
    
    if not data_path.exists():
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    
    # Initialize recommenders
    content_based = ContentBasedRecommender(df=df)
    content_based.fit()
    
    popularity = PopularityRecommender(df)
    hybrid = HybridRecommender(df)
    
    return df, content_based, popularity, hybrid


def create_radar_chart(song_features, title="Audio Features"):
    """Create a radar chart for song audio features"""
    categories = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                  'Instrumentalness', 'Liveness', 'Speechiness']
    
    values = [
        song_features.get('danceability', 0),
        song_features.get('energy', 0),
        song_features.get('valence', 0),
        song_features.get('acousticness', 0),
        song_features.get('instrumentalness', 0),
        song_features.get('liveness', 0),
        song_features.get('speechiness', 0)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        fillcolor='rgba(29, 185, 84, 0.3)',
        line_color='#1DB954'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_feature_comparison(seed_song, recommendations, df):
    """Create comparison chart between seed and recommendations"""
    features = ['danceability', 'energy', 'valence', 'acousticness']
    
    # Get seed song features
    seed_features = df[df['name'] == seed_song][features].iloc[0].values
    
    # Get average of recommendations
    rec_names = recommendations['name'].tolist()
    rec_df = df[df['name'].isin(rec_names)]
    rec_features = rec_df[features].mean().values
    
    fig = go.Figure()
    
    x = np.arange(len(features))
    width = 0.35
    
    fig.add_trace(go.Bar(
        x=features,
        y=seed_features,
        name='Seed Song',
        marker_color='#1DB954'
    ))
    
    fig.add_trace(go.Bar(
        x=features,
        y=rec_features,
        name='Recommendations Avg',
        marker_color='#535353'
    ))
    
    fig.update_layout(
        barmode='group',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def main():
    st.markdown('<h1 class="main-header">🎵 Spotify Music Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Discover your next favorite songs using AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Load data
    df, content_based, popularity, hybrid = load_data_and_models()
    
    if df is None:
        st.error("Data not found. Please run `python data_cleaning.py` first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎧 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "🎼 Content-Based", "🔀 Hybrid", "📊 Analytics", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home(df)
    elif page == "🎼 Content-Based":
        show_content_based(df, content_based)
    elif page == "🔀 Hybrid":
        show_hybrid(df, hybrid)
    elif page == "📊 Analytics":
        show_analytics(df, content_based)
    elif page == "ℹ️ About":
        show_about()


def show_home(df):
    """Home page with overview"""
    st.header("Welcome!")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎵 Total Songs", f"{len(df):,}")
    with col2:
        st.metric("🎤 Artists", f"{df['artist'].nunique():,}")
    with col3:
        st.metric("🎸 Genres", f"{df['genre'].nunique():,}")
    with col4:
        avg_energy = df['energy'].mean()
        st.metric("⚡ Avg Energy", f"{avg_energy:.2f}")
    
    st.markdown("---")
    
    # How it works
    st.subheader("How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎼 Content-Based Filtering
        Analyzes audio features like:
        - **Danceability** - How suitable for dancing
        - **Energy** - Intensity and activity
        - **Valence** - Musical positiveness
        - **Tempo** - Speed of the track
        
        Finds songs with similar characteristics using **cosine similarity**.
        """)
    
    with col2:
        st.markdown("""
        ### 🔀 Hybrid Approach
        Combines multiple signals:
        - **Content Similarity** - Similar audio features
        - **Popularity Score** - Trending tracks
        
        Adjustable weights let you balance discovery vs. popularity.
        """)
    
    st.markdown("---")
    
    # Sample songs
    st.subheader("🎵 Sample Songs from Dataset")
    sample = df[['name', 'artist', 'genre', 'year']].sample(10).reset_index(drop=True)
    st.dataframe(sample, hide_index=True)
    
    # Genre distribution
    st.subheader("📊 Genre Distribution")
    genre_counts = df['genre'].value_counts().head(10)
    fig = px.bar(
        x=genre_counts.index, 
        y=genre_counts.values,
        labels={'x': 'Genre', 'y': 'Number of Songs'},
        color=genre_counts.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_content_based(df, recommender):
    """Content-based recommendations page"""
    st.header("🎼 Content-Based Recommendations")
    st.write("Find songs with similar audio characteristics")
    
    # Song selection
    col1, col2 = st.columns(2)
    
    with col1:
        song_title = st.selectbox(
            "Select a song:",
            options=sorted(df['name'].unique()),
            index=0
        )
    
    with col2:
        artists = df[df['name'] == song_title]['artist'].unique()
        artist = st.selectbox("Select artist:", options=artists)
    
    n_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    # Get song info for display
    song_info = df[(df['name'] == song_title) & (df['artist'] == artist)].iloc[0]
    
    # Show seed song features
    st.subheader("📈 Seed Song Audio Profile")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**{song_title}**")
        st.write(f"by {artist}")
        st.write(f"Genre: {song_info.get('genre', 'Unknown')}")
        st.write(f"Year: {int(song_info.get('year', 0)) if pd.notna(song_info.get('year')) else 'Unknown'}")
    
    with col2:
        radar = create_radar_chart(song_info.to_dict())
        st.plotly_chart(radar)
    
    # Get recommendations button
    if st.button("🎵 Get Recommendations", type="primary"):
        with st.spinner("Finding similar songs..."):
            recommendations = recommender.get_recommendations(song_title, artist, n_recs)
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.success(f"Found {len(recommendations)} similar songs!")
            
            # Show comparison chart
            st.subheader("📊 Feature Comparison")
            comparison = create_feature_comparison(song_title, recommendations, df)
            st.plotly_chart(comparison)
            
            # Show recommendations
            st.subheader("🎵 Recommended Songs")
            
            for idx, row in recommendations.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{row['name']}**")
                with col2:
                    st.write(f"{row['artist']}")
                with col3:
                    score = row['similarity_score']
                    st.write(f"🎯 {score:.1%}")


def show_hybrid(df, hybrid):
    """Hybrid recommendations page"""
    st.header("🔀 Hybrid Recommendations")
    st.write("Combine content similarity with popularity")
    
    # Song selection
    col1, col2 = st.columns(2)
    
    with col1:
        song_title = st.selectbox(
            "Select a song:",
            options=sorted(df['name'].unique()),
            key="hybrid_song"
        )
    
    with col2:
        artists = df[df['name'] == song_title]['artist'].unique()
        artist = st.selectbox("Select artist:", options=artists, key="hybrid_artist")
    
    # Settings
    st.subheader("⚙️ Tuning")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_recs = st.slider("Recommendations:", 5, 20, 10, key="hybrid_n")
    with col2:
        content_weight = st.slider("Content Weight:", 0.0, 1.0, 0.7, 0.1)
    with col3:
        popularity_weight = st.slider("Popularity Weight:", 0.0, 1.0, 0.3, 0.1)
    
    if st.button("🔀 Get Hybrid Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = hybrid.get_recommendations(
                song_title, artist, n_recs, content_weight, popularity_weight
            )
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.success(f"Found {len(recommendations)} recommendations!")
            
            # Display with scores
            st.subheader("🎵 Recommended Songs")
            
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        st.write(f"**{row['name']}**")
                    with col2:
                        st.write(f"{row['artist']}")
                    with col3:
                        st.write(f"Sim: {row['similarity_score']:.1%}")
                    with col4:
                        st.write(f"Score: {row['hybrid_score']:.2f}")


def show_analytics(df, recommender):
    """Analytics and evaluation page"""
    st.header("📊 Analytics & Evaluation")
    
    # Feature distributions
    st.subheader("Audio Feature Distributions")
    
    features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
    
    tabs = st.tabs(features)
    
    for tab, feature in zip(tabs, features):
        with tab:
            fig = px.histogram(
                df, x=feature, nbins=50,
                color_discrete_sequence=['#1DB954']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig)
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = calculate_feature_importance(df, recommender)
    
    fig = px.bar(
        x=list(importance.keys()),
        y=list(importance.values()),
        labels={'x': 'Feature', 'y': 'Importance'},
        color=list(importance.values()),
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig)
    
    st.markdown("---")
    
    # Evaluation metrics
    st.subheader("📈 Model Evaluation")
    
    if st.button("Run Evaluation (may take a moment)"):
        with st.spinner("Evaluating recommendations..."):
            metrics = evaluate_recommendations(df, recommender, n_samples=50)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Artist Diversity", f"{metrics['Artist Diversity']:.1%}")
        with col2:
            st.metric("Avg Genre Coverage", f"{metrics['Avg Genre Coverage']:.1f}")
        with col3:
            st.metric("Avg Similarity", f"{metrics['Avg Similarity Score']:.1%}")
        
        st.info(f"Evaluated on {metrics['Songs Evaluated']} sample songs")


def show_about():
    """About page"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Spotify Music Recommender System
    
    A **content-based** and **hybrid** music recommendation system built with Python.
    
    ### 🛠️ Tech Stack
    - **Python** - Core programming language
    - **Pandas & NumPy** - Data manipulation
    - **Scikit-learn** - Machine learning (cosine similarity)
    - **Streamlit** - Web application framework
    - **Plotly** - Interactive visualizations
    
    ### 📊 Dataset
    Million Song Dataset from Kaggle containing 5,000 songs with audio features from Spotify.
    
    ### 🔬 How It Works
    
    **Content-Based Filtering:**
    1. Extract audio features (danceability, energy, tempo, etc.)
    2. Normalize features using StandardScaler
    3. Compute cosine similarity between all songs
    4. Rank by similarity to seed song
    
    **Hybrid Approach:**
    1. Get content-based recommendations
    2. Calculate popularity scores
    3. Combine with configurable weights
    
    ### 📁 Project Structure
    ```
    ├── app.py                     # Streamlit application
    ├── content_based_filtering.py # Content-based recommender
    ├── popularity_recommender.py  # Popularity-based recommender
    ├── hybrid_recommender.py      # Hybrid recommender
    ├── evaluation.py              # Evaluation metrics
    ├── data_cleaning.py           # Data preprocessing
    └── data/cleaned_data.csv      # Processed dataset
    ```
    
    ### 👤 Author
    Built as a portfolio project to demonstrate ML recommendation systems.
    
    ---
    *Data source: [Kaggle - Million Song Dataset](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)*
    """)


if __name__ == "__main__":
    main()
