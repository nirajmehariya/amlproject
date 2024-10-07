from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/cnn_genre_model.h5')

# Load feature scaler and label encoder
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Assuming you have a mapping of genres for encoding
# Update this based on your dataset
genre_mapping = {
    0: 'acoustic', 
    1: 'classical', 
    2: 'country', 
    3: 'dance', 
    4: 'hip-hop', 
    5: 'jazz', 
    6: 'pop', 
    7: 'rock'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the MP3 file uploaded
    file = request.files['file']
    
    # Load the audio file and extract features
    y, sr = librosa.load(file, sr=None)
    
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
    
    # Scale the features
    X = scaler.fit_transform(feature_df)

    # Predict the genre
    predictions = model.predict(X)
    genre_index = np.argmax(predictions, axis=1)[0]
    genre_pred = genre_mapping[genre_index]

    return f'Predicted genre: {genre_pred}'

if __name__ == '__main__':
    app.run(debug=True)
