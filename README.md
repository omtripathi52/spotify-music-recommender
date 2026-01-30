# 🎵 Spotify Music Recommender System

A content-based and hybrid music recommendation system that suggests songs based on audio features using machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Live Demo

**[Try the App →](https://your-app-name.streamlit.app)** *(Update after deployment)*

## ✨ Features

- **Content-Based Filtering**: Recommends songs with similar audio characteristics (danceability, energy, tempo, valence)
- **Hybrid Recommendations**: Combines content similarity with popularity scores
- **Interactive Visualizations**: Radar charts, feature comparisons, and distribution plots
- **Model Evaluation**: Built-in metrics to assess recommendation quality

## 📸 Screenshots

| Home | Recommendations | Analytics |
|------|-----------------|-----------|
| Overview & Stats | Similar Songs | Feature Analysis |

## 🛠️ Tech Stack

- **Python 3.10+** - Core language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Cosine similarity computation
- **Streamlit** - Web application
- **Plotly** - Interactive charts

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/spotify-music-recommender.git
cd spotify-music-recommender

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 How It Works

### Content-Based Filtering

1. **Feature Extraction**: Extract audio features (danceability, energy, valence, tempo, etc.)
2. **Normalization**: Scale features using StandardScaler
3. **Similarity Computation**: Calculate cosine similarity between songs
4. **Ranking**: Return top-N most similar songs

```
Similarity(A, B) = (A · B) / (||A|| × ||B||)
```

### Hybrid Approach

Combines content-based scores with popularity:

```
Hybrid Score = (Content Weight × Similarity) + (Popularity Weight × Popularity Score)
```

## 📁 Project Structure

```
spotify-music-recommender/
├── app.py                      # Streamlit web application
├── content_based_filtering.py  # Content-based recommender
├── popularity_recommender.py   # Popularity-based recommender
├── hybrid_recommender.py       # Hybrid recommender
├── evaluation.py               # Evaluation metrics
├── data_cleaning.py            # Data preprocessing script
├── data/
│   └── cleaned_data.csv        # Processed dataset (5,000 songs)
├── requirements.txt            # Python dependencies
└── README.md
```

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Artist Diversity** | Variety of artists in recommendations |
| **Genre Coverage** | Number of unique genres covered |
| **Avg Similarity** | Mean similarity score of recommendations |

## 🔮 Future Improvements

- [ ] Add collaborative filtering with real user data
- [ ] Implement matrix factorization (SVD)
- [ ] Add audio preview playback
- [ ] Integrate Spotify API for real-time data

## 📚 Dataset

[Million Song Dataset - Spotify & Last.fm](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm) from Kaggle.

**Features used:**
- danceability, energy, valence, acousticness
- instrumentalness, liveness, speechiness, tempo

## 📝 License

MIT License - feel free to use this project for learning and portfolio purposes.

---

⭐ If you found this helpful, please star the repository!
