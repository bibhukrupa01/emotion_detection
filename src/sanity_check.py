import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import librosa
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "emotion_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")

# Load model
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model classes: ", model.classes_)

# 5 files from dataset, different emotions
files_to_test = [
    ("Happy", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-03-01-01-01-01.wav")),
    ("Sad", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-04-01-01-01-01.wav")),
    ("Angry", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-05-01-01-01-01.wav")),
    ("Fear", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-06-01-01-01-01.wav")),
    ("Disgust", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-07-01-01-01-01.wav")),
]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

for expected, fpath in files_to_test:
    if not os.path.exists(fpath):
        print(f"Missing {fpath}")
        continue
        
    features = extract_features(fpath)
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    
    pred = model.predict(features)[0]
    print(f"File ({expected}): Predicted {pred}")
