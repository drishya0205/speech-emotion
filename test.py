import tensorflow as tf
import librosa
import numpy as np
import argparse

# CLI Argument Parser
parser = argparse.ArgumentParser(description="Predict emotion from an EmoDB audio file.")

parser.add_argument("-f", "--filename", help="Path to the audio file. Example: -f happy.wav")
parser.add_argument("-m", "--model_path", help="Path to the saved model. Example: -m saved_model/emodb_model")
args = parser.parse_args()

# Default values if arguments are not provided
save_file_path = args.filename if args.filename else "disgusted.wav"
save_model_path = args.model_path if args.model_path else "saved_model.h5/10_trained_model.h5"

# EmoDB Emotion Mapping
EMOTION_DICT_EMODB = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

# Convert emotion labels to integer IDs
label_to_int = {key: i for i, key in enumerate(EMOTION_DICT_EMODB.keys())}
int_to_label = {val: key for key, val in label_to_int.items()}

def process_audio_clip(file_path):
    """Extracts features from an audio file for EmoDB classification."""
    audio, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)

    # Combine all features into one array
    extracted_features = np.concatenate([mfcc, mel, chroma, spectral, tonnetz], axis=0)

    return extracted_features

# Load the trained EmoDB model
Model = tf.keras.models.load_model(save_model_path)

# Process the given audio file
features = process_audio_clip(save_file_path)

# Reshape features for model input
features = tf.expand_dims(features, -1)  # Add channel dimension
features = tf.expand_dims(features, 0)   # Add batch dimension

# Predict emotion
prediction = Model.predict(features)
label = tf.math.argmax(prediction[0]).numpy()

# Convert label to emotion name
predicted_emotion = EMOTION_DICT_EMODB[int_to_label[label]]

print(f"Predicted Emotion: {predicted_emotion}")
