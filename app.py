import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from newmrs import trending_songs  # Import trending songs list

# Load data and models
df = pd.read_csv("cleaned_spotify.csv")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

# Function to fetch artwork via iTunes API
def get_itunes_artwork(artist, song):
    try:
        query = f"{artist} {song}"
        query = re.sub(r"[^\w\s]", "", query)
        url = f"https://itunes.apple.com/search?term={requests.utils.quote(query)}&media=music&entity=song&limit=1"
        resp = requests.get(url, timeout=5).json()
        items = resp.get("results")
        if items:
            art = items[0].get("artworkUrl100")
            return art.replace("100x100bb", "300x300bb") if art else None
    except:
        pass
    return None

# Page setup
st.set_page_config(page_title="🎶 Song Lyrics Search Engine", layout="wide")
st.title("🎶 Song Lyrics Search Engine")
st.write("Type some **lyrics** or **keywords** to discover matching songs.")

# Search box
query = st.text_input("🔍 Enter lyrics or keywords:")

# Function to display songs in a row
def display_song_cards(songs):
    cols = st.columns(5)
    for i, (col, song_row) in enumerate(zip(cols * (len(songs)//5 + 1), songs), start=1):
        with col:
            st.markdown(f"**{song_row['song']}**")
            st.caption(f"by *{song_row['artist']}*")
            img_url = get_itunes_artwork(song_row['artist'], song_row['song'])
            st.image(img_url if img_url else "image.jpg", width=150)

# Search functionality
if query:
    qv = vectorizer.transform([query.lower()])
    sim = cosine_similarity(qv, tfidf_matrix).flatten()
    if sim.any():
        indices = sim.argsort()[-10:][::-1]
        songs = df.iloc[indices][['artist', 'song', 'clean_text']].to_dict(orient='records')
        st.subheader("🔍 Search Results")
        display_song_cards(songs)
    else:
        st.warning("⚠️ No matching songs found. Try different keywords.")
else:
    st.subheader("🔥 Trending Songs")
    display_song_cards(trending_songs)

# Footer
st.markdown(
    "<br><center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)
