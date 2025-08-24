import pandas as pd
import re
import joblib
import nltk
import numpy as np  # NEW CODE: Import numpy for random number generation

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Ye downloads aap ek baar run karne ke baad comment out kar sakte hain ---
# nltk.download('stopwords')
# nltk.download('wordnet')

print("Loading Spotify data...")
df = pd.read_csv("cleaned_spotify.csv")
print(f"Data loaded: {df.shape[0]} songs selected.")

# --- NEW CODE STARTS HERE ---

# Step 1: Randomly generate a popularity score for each song
# Yeh har song ko 0 se 1 ke beech ek random float value dega.
print("Generating random popularity scores...")
df['popularity'] = np.random.uniform(0, 1, df.shape[0])

# Step 2: Create a function to get top N songs based on the new score
def get_top_songs(dataframe, n=20):
    """
    Sorts the dataframe by the 'popularity' column and returns the top n songs.
    """
    top_songs = dataframe.sort_values(by='popularity', ascending=False).head(n)
    return top_songs[['artist', 'song', 'link', 'popularity']]

# --- NEW CODE ENDS HERE ---


print("Cleaning lyrics, please wait...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\r|\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].astype(str).apply(clean_text)
print("Lyrics cleaned successfully.")

print("Creating TF-IDF model...")
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
print("TF-IDF model ready.")

def search_song(query, top_n=5):
    if not query.strip():
        return pd.DataFrame(columns=["artist", "song", "link", "snippet"])
    
    query_vector = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if not similarity.any():
        return pd.DataFrame(columns=["artist", "song", "link", "snippet"])
    
    top_indices = similarity.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['artist', 'song', 'link', 'clean_text']].copy()
    results['snippet'] = results['clean_text'].apply(
        lambda x: ' '.join(x.split()[:20]) + "..."
    )
    return results[['artist', 'song', 'link', 'snippet']]

print("Saving files for the app...")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
# Humne 'popularity' column add kiya hai, isliye updated df save karenge
df.to_csv("cleaned_spotify.csv", index=False)
print("All files saved successfully.")

# --- TESTING THE NEW FUNCTIONALITY ---
print("\n--- Testing Top 20 Randomly Generated Songs ---")
top_20_songs = get_top_songs(df, n=20)
print(top_20_songs)

print("\n--- Testing Search ---")
print(search_song("broken heart love"))
# Create a trending songs list (top 20 by popularity)
trending_songs = get_top_songs(df, n=20).to_dict(orient='records')


