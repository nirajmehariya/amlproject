import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('./data/dataset.csv')

# Define the features and target
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo']
target = 'track_genre'

# Encode the 'track_genre' column into numeric labels
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df[target])

# Split features and target
X = df[features]
y = df['genre_encoded']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

# Save the trained model
model.save('models/cnn_genre_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)

# Print classification report
print(classification_report(y_test, y_pred_labels))
