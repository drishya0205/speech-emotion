from logging import error
import tensorflow as tf
import librosa
import os
import numpy as np
from SpeechModel import SpeechModel

# EMODB Emotion Mapping
EMOTION_DICT_EMODB = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

# Reverse mapping for encoding labels as integers
LABEL_TO_INT = {key: i for i, key in enumerate(EMOTION_DICT_EMODB.keys())}

def process_audio_clip(file_path, label):
    file_path = file_path.numpy().decode("utf-8")
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)

    # Concatenate all features
    extracted_features = np.concatenate([mfcc, mel, chroma, spectral, tonnetz], axis=0)
    
    return extracted_features, label


def get_dataset(
    training_dir="./wav",
    validation_dir=None,
    val_split=0.2,
    batch_size=128,
    random_state=42,
    cache=False,
):

    def decompose_label(file_name):
        """Extracts the emotion label from the filename (6th character in EMODB)."""
        emotion_char = file_name[5]  # e.g., '03a01Fa.wav' -> 'F' (Happiness)
        return LABEL_TO_INT.get(emotion_char, -1)  # Default to -1 if not found

    def tf_wrapper_process_audio_clip(file_path, label):
        extracted_features, label = tf.py_function(
            process_audio_clip, [file_path, label], [tf.float32, tf.int32]
        )
        extracted_features.set_shape([193])
        label.set_shape([])
        extracted_features = tf.expand_dims(extracted_features, -1)
        return extracted_features, label

    file_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir)]
    labels = [decompose_label(os.path.basename(f)) for f in file_paths]

    # Remove any files with unrecognized labels (-1)
    valid_data = [(p, l) for p, l in zip(file_paths, labels) if l != -1]
    file_paths, labels = zip(*valid_data) if valid_data else ([], [])

    if validation_dir is None:
        if val_split > 0 and file_paths:
            from sklearn.model_selection import train_test_split
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                file_paths, labels, test_size=val_split, random_state=random_state
            )
        else:
            train_paths, train_labels = file_paths, labels
            val_paths, val_labels = [], []
    else:
        val_paths = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]
        val_labels = [decompose_label(os.path.basename(f)) for f in val_paths]
        val_paths, val_labels = zip(*[(p, l) for p, l in zip(val_paths, val_labels) if l != -1])

    train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))

    train_ds = train_ds.map(tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def create_model(num_output_classes):
    speechModel = SpeechModel(num_output_classes)
    model = speechModel.getEmoDB()  # Assuming SpeechModel has getEMODB() for EMODB
    return model
