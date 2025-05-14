import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    jitter = np.std(pitch_values) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
    
    features = np.hstack([mfccs, jitter])
    return features.reshape(1, -1)
