import os
import sys

# Add project root to path so we can import from src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from src.feature_extraction import extract_features

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "emotion_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")

# Load model
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model classes: ", model.classes_)

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# 5 files from dataset, different emotions
files_to_test = [
    ("Happy", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-03-01-01-01-01.wav")),
    ("Sad", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-04-01-01-01-01.wav")),
    ("Angry", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-05-01-01-01-01.wav")),
    ("Fearful", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-06-01-01-01-01.wav")),
    ("Disgust", os.path.join(project_root, "dataset", "RAVDESS", "Actor_01", "03-01-07-01-01-01-01.wav")),
]


for expected, fpath in files_to_test:
    if not os.path.exists(fpath):
        print(f"Missing {fpath}")
        continue
        
    features = extract_features(fpath)
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    
    pred = model.predict(features)[0]
    predicted_emotion = emotion_map.get(pred, "Unknown")
    
    match_status = "✅" if predicted_emotion == expected else "❌"
    print(f"File ({expected}): Predicted {predicted_emotion} {match_status}")
