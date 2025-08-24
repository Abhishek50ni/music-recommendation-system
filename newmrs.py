import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Download NLTK data (needed in cloud)
# ------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

print("📂 Loading Spotify data...")
df = pd.read_csv("cleaned_spotify.csv")
print(f"✅ Data loaded: {df.shape[0]} songs found.")

# ------------------------------
# Add Popularity Score for Trending
# ------------------------------
print("🎯 Generating random popularity scores...")
df['popularity'] = np.random.uniform(0, 1, df.shape[0])

def get_top_songs(dataframe, n=20):
    """Return the top n songs sorted by popularity."""
    return dataframe.sort_values(by='popularity', ascending=False).head(n)[['artist', 'song', 'link', 'popularity']]

# ------------------------------
# Text Cleaning
# ------------------------------
print("🧹 Cleaning lyrics, please wait...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\r|\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].astype(str).apply(clean_text)
print("✅ Lyrics cleaned successfully.")

# ------------------------------
# TF-IDF Vectorization
# ------------------------------
print("⚙️ Creating TF-IDF model...")
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
print("✅ TF-IDF model ready.")

# ------------------------------
# Song Search Function
# ------------------------------
def search_song(query, top_n=5):
    if not query.strip():
        return pd.DataFrame(columns=["artist", "song", "link", "snippet"])

    query_vector = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if not similarity.any():
        return pd.DataFrame(columns=["artist", "song", "link", "snippet"])

    top_indices = similarity.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['artist', 'song', 'link', 'clean_text']].copy()
    results['snippet'] = results['clean_text'].apply(lambda x: ' '.join(x.split()[:20]) + "...")
    return results[['artist', 'song', 'link', 'snippet']]

# ------------------------------
# Save Processed Data & Models
# ------------------------------
print("💾 Saving vectorizer, matrix, and cleaned data...")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
df.to_csv("cleaned_spotify.csv", index=False)
print("✅ All files saved successfully.")

# ------------------------------
# Trending Songs (Top 20)
# ------------------------------
print("\n🔥 Fetching Top 20 Trending Songs...")
trending_songs = get_top_songs(df, n=20).to_dict(orient='records')
print(pd.DataFrame(trending_songs))

# ------------------------------
# Test Search
# ------------------------------
print("\n🔍 Testing Search with Query: 'broken heart love'")
print(search_song("broken heart love"))

print("\n🔍 Testing Search with Query: 'party night'")
print(search_song("party night"))
