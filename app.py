import streamlit as st
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity

# === STEP 1: Use your real credentials ===
client_id = "97e27d968cfc4e76bd40afc79853250f"
client_secret = "PASTE_YOUR_CLIENT_SECRET_HERE"  # Click "View client secret" to get it

# === STEP 2: Set up Spotify API client ===
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# === STEP 3: Function to extract song features ===
def get_track_features(track_id):
    features = sp.audio_features([track_id])[0]
    return {
        'danceability': features['danceability'],
        'energy': features['energy'],
        'tempo': features['tempo'],
        'valence': features['valence']
    }

# === STEP 4: Get recommendations ===
def get_recommendations(track_id):
    seed_features = get_track_features(track_id)
    input_vector = np.array(list(seed_features.values())).reshape(1, -1)

    recs = sp.recommendations(seed_tracks=[track_id], limit=10)
    all_data = []

    for track in recs['tracks']:
        f = get_track_features(track['id'])
        f['name'] = track['name']
        f['artist'] = track['artists'][0]['name']
        f['track_id'] = track['id']
        all_data.append(f)

    df = pd.DataFrame(all_data)
    track_vectors = df[['danceability', 'energy', 'tempo', 'valence']].values
    similarities = cosine_similarity(input_vector, track_vectors).flatten()
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False)

# === STEP 5: Streamlit UI ===
st.set_page_config(page_title="🎵 Real-Time Song Recommender", layout="centered")
st.title("🎧 Real-Time Spotify Song Recommendation System")

track_url = st.text_input("Paste a Spotify Track URL (e.g. https://open.spotify.com/track/...)")

if track_url:
    try:
        track_id = track_url.split("/")[-1].split("?")[0]
        st.success("Generating recommendations...")

        recs_df = get_recommendations(track_id)

        for _, row in recs_df.iterrows():
            st.markdown(f"**{row['name']}** by *{row['artist']}*")
            st.write(f"https://open.spotify.com/track/{row['track_id']}")
            st.write("---")

    except Exception as e:
        st.error(f"Error: {e}")
