import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .movie-number {
        font-size: 2rem;
        font-weight: bold;
        opacity: 0.8;
    }
    .movie-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-left: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .stats-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
    <div class="header-container">
        <h1 style="margin: 0; font-size: 3rem;">🎬 Movie Recommendation System</h1>
        <p style="margin-top: 0.5rem; font-size: 1.2rem; opacity: 0.9;">
            Discover your next favorite movie based on what you love
        </p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview']]
    df['overview'] = df['overview'].fillna('')
    return df

movies = load_data()

# -----------------------------
# COMPUTE SIMILARITY
# -----------------------------
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(movies)

indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend_movies(movie_title):
    movie_title = movie_title.lower()
    
    if movie_title not in indices:
        return None
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices]

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📊 System Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="stats-box">
                <h2 style="margin: 0; color: #667eea;">{len(movies):,}</h2>
                <p style="margin: 0; color: #666;">Total Movies</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stats-box">
                <h2 style="margin: 0; color: #667eea;">5</h2>
                <p style="margin: 0; color: #666;">Recommendations</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("💡 How It Works")
    st.write("""
    1. **Enter** a movie title you enjoyed
    2. **Click** the Recommend button
    3. **Discover** similar movies based on plot descriptions
    """)
    
    st.markdown("---")
    
    st.subheader("🎯 Tips")
    st.info("Try popular movies like:\n- The Dark Knight\n- Avatar\n- Inception\n- Interstellar")

# -----------------------------
# MAIN CONTENT
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    user_input = st.text_input(
        "🔍 Enter a movie name:",
        placeholder="e.g., The Dark Knight, Avatar, Inception...",
        label_visibility="visible"
    )
    
    recommend_btn = st.button("🎬 Get Recommendations", use_container_width=True, type="primary")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# RECOMMENDATIONS DISPLAY
# -----------------------------
if recommend_btn:
    if not user_input:
        st.warning("⚠️ Please enter a movie name first!")
    else:
        with st.spinner("🔎 Finding similar movies..."):
            recommendations = recommend_movies(user_input)
        
        if recommendations is None:
            st.error(f"❌ Movie '{user_input}' not found. Please check the spelling or try another movie.")
            
            # Fuzzy matching suggestions
            user_lower = user_input.lower()
            similar = movies[movies['title'].str.lower().str.contains(user_lower[:3], na=False)]['title'].head(5)
            
            if not similar.empty:
                st.info("**Did you mean:**")
                for movie in similar:
                    st.write(f"• {movie}")
        else:
            st.success(f"✅ Based on **{user_input}**, here are your recommendations:")
            
            # Display recommendations in a nice grid
            for i, movie in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class="movie-card">
                        <span class="movie-number">#{i}</span>
                        <span class="movie-title">{movie}</span>
                    </div>
                """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <p style="text-align: center; color: #666;">
        Powered by TF-IDF & Cosine Similarity | Built with Streamlit
    </p>
""", unsafe_allow_html=True)
