<p align="center">
  <img src="https://img.icons8.com/fluency/96/spotify.png" alt="Spotify Logo" width="80"/>
</p>

<h1 align="center">🎵 Spotify Music Recommender System</h1>

<p align="center">
  <strong>AI-powered music recommendations using content-based filtering and hybrid algorithms</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## 🎯 Live Demo

> **[🚀 Try the App Live →](https://spotify-music-recommender-omtripathi52.streamlit.app)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎼 **Content-Based Filtering** | Finds songs with similar audio characteristics using cosine similarity |
| 🔀 **Hybrid Recommendations** | Combines content similarity + popularity with adjustable weights |
| 📊 **Interactive Visualizations** | Radar charts, bar graphs, and histograms powered by Plotly |
| 📈 **Model Evaluation** | Built-in metrics: Artist Diversity, Genre Coverage, Similarity Score |
| ⚡ **Optimized Performance** | 5,000 songs dataset (~200MB RAM) for fast recommendations |

---

## 🔬 How It Works

### Content-Based Filtering

The system analyzes **13 audio features** from each song:

```
┌─────────────────────────────────────────────────────────────┐
│  Audio Features                                             │
├─────────────────────────────────────────────────────────────┤
│  • Danceability    • Energy         • Valence              │
│  • Acousticness    • Instrumentalness • Liveness           │
│  • Speechiness     • Tempo          • Loudness             │
│  • Key             • Mode           • Duration             │
└─────────────────────────────────────────────────────────────┘
```

**Algorithm:**
1. Extract & normalize audio features using `StandardScaler`
2. Compute pairwise **cosine similarity** between all songs
3. For a given seed song, rank all others by similarity
4. Return top-N most similar songs

```
                    A · B
Similarity(A,B) = ─────────
                  ‖A‖ × ‖B‖
```

### Hybrid Approach

Combines multiple signals for better recommendations:

```
Hybrid Score = (α × Content Similarity) + (β × Popularity Score)

where α + β = 1.0 (configurable via UI sliders)
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/omtripathi52/spotify-music-recommender.git
cd spotify-music-recommender

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
spotify-music-recommender/
│
├── 📊 Data & Processing
│   ├── data/cleaned_data.csv      # Processed dataset (5,000 songs)
│   └── data_cleaning.py           # Data preprocessing pipeline
│
├── 🤖 Recommendation Engines
│   ├── content_based_filtering.py # Cosine similarity recommender
│   ├── popularity_recommender.py  # Popularity-based recommender
│   └── hybrid_recommender.py      # Combined hybrid approach
│
├── 📈 Evaluation
│   └── evaluation.py              # Metrics: diversity, coverage, similarity
│
├── 🎨 Web Application
│   └── app.py                     # Streamlit multi-page app
│
├── 📋 Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── .gitignore
│   └── LICENSE
│
└── 📖 README.md
```

---

## 🧰 Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **ML/Data** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **Artist Diversity** | Unique artists in recommendations | > 70% |
| **Genre Coverage** | Different genres represented | > 3 genres |
| **Avg Similarity** | How similar recommendations are to seed | 60-90% |

---

## 📚 Dataset

**Source:** [Million Song Dataset - Spotify & Last.fm](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)

| Attribute | Value |
|-----------|-------|
| Songs | 5,000 (sampled for performance) |
| Features | 13 audio characteristics |
| Genres | Multiple (Pop, Rock, Hip-Hop, etc.) |

---

## 🔮 Future Roadmap

- [ ] Integrate Spotify Web API for real-time data
- [ ] Add song preview playback
- [ ] Implement user-based collaborative filtering
- [ ] Deploy with Docker

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Om Tripathi**

[![GitHub](https://img.shields.io/badge/GitHub-omtripathi52-181717?style=flat-square&logo=github)](https://github.com/omtripathi52)

---

<p align="center">
  ⭐ Star this repo if you found it helpful!
</p>
