import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Kaggle dataset (downloaded from Kaggle)
df = pd.read_csv('data/spotify_tracks.csv')

# Drop irrelevant columns (assume 'id' and 'name' are unnecessary)
df = df.drop(columns=['id', 'name', 'popularity', 'artists'])

# Check for missing data and drop missing rows
df = df.dropna()

# Select audio features for training
features = ['danceability', 'energy','loudness','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','track_genre']
X = df[features]

# Encode the 'genre' column into numeric labels
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['genre'])
y = df['genre_encoded']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed data as CSV
train_data = pd.DataFrame(X_train, columns=features)
train_data['track_genre'] = y_train
train_data.to_csv('data/train_data.csv', index=False)

test_data = pd.DataFrame(X_test, columns=features)
test_data['track_genre'] = y_test
test_data.to_csv('data/test_data.csv', index=False)

print("Data Preprocessing Complete. Training and test datasets saved.")
