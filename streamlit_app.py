import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa
from sklearn.metrics import pairwise

# Load the trained model
model = tf.keras.models.load_model('./models/cnn_genre_model.h5')

# Load the dataset to create genre mapping
df = pd.read_csv('./data/dataset.csv')

# Create genre mapping from the dataset
label_encoder = LabelEncoder()
df['track_genre'] = label_encoder.fit_transform(df['track_genre'])
genre_mapping = {index: genre for index, genre in enumerate(label_encoder.classes_)}

# Fit the scaler on the entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['danceability', 'energy', 'loudness', 'speechiness', 
                                     'acousticness', 'instrumentalness', 'liveness', 
                                     'valence', 'tempo']])

# Display model output shape
st.write("Model Output Shape:", model.output_shape)

# Streamlit App UI
st.title("Spotify Song Genre Classification")

# File uploader for MP3 files
uploaded_file = st.file_uploader("Upload your MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Load the audio file and extract features
    y, sr = librosa.load(uploaded_file, sr=None)
    
    # Extract relevant features using librosa
    features = {
        'danceability': np.mean(librosa.feature.tempogram(y=y, sr=sr)),
        'energy': np.mean(np.abs(librosa.feature.rms(y=y))),
        'loudness': np.mean(np.abs(librosa.amplitude_to_db(np.abs(librosa.stft(y))))),
        'speechiness': np.mean(np.abs(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'acousticness': np.mean(np.abs(librosa.feature.zero_crossing_rate(y=y))),
        'instrumentalness': np.mean(np.abs(librosa.feature.spectral_flatness(y=y))),
        'liveness': np.mean(np.abs(librosa.feature.chroma_stft(y=y, sr=sr))),
        'valence': np.mean(np.abs(librosa.feature.mfcc(y=y, sr=sr))),
        'tempo': np.abs(librosa.beat.tempo(y=y, sr=sr)[0]),
    }

    # Create a DataFrame for scaling
    feature_df = pd.DataFrame([features])
    
    # Scale the features using the fitted scaler
    X = scaler.transform(feature_df)

    # Predict the genre
    predictions = model.predict(X)
    
    # Get the genre index
    genre_index = np.argmax(predictions, axis=1)[0]
    genre_pred = genre_mapping.get(genre_index, "Unknown genre")

    # Display predicted genre
    st.write("Predicted genre:")
    st.write(genre_pred)

    # Find the closest match in the dataset based on features
    distances = pairwise.euclidean_distances(X_scaled, X)
    closest_index = np.argmin(distances)
    
    # Get the closest track genre from the dataset
    closest_genre = df.iloc[closest_index]['track_genre']

    # Display the closest match genre
    st.write("Closest matching genre from dataset:")
    st.write(label_encoder.inverse_transform([closest_genre])[0])
