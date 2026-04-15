import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
from src.feature_extraction import extract_features_from_audio
model = joblib.load("models/emotion_model.pkl")
scaler = joblib.load("models/scaler.pkl")

emotion_map = {
    "01": ("Neutral", "😐"),
    "02": ("Calm", "😌"),
    "03": ("Happy", "😄"),
    "04": ("Sad", "😢"),
    "05": ("Angry", "😠"),
    "06": ("Fearful", "😨"),
    "07": ("Disgust", "🤢"),
    "08": ("Surprised", "😲")
}

st.title("🎤 Real-Time Emotion Detection")

if st.button("Record Audio"):
    sr = 22050
    duration = 3

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = audio.flatten()
    features = extract_features_from_audio(audio, sr=sr)
    features = features.reshape(1, -1)
    
    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_id = prediction[0]
    
    emotion, emoji = emotion_map.get(predicted_id, ("Unknown", "❓"))

    st.success(f"Detected Emotion: {emotion} {emoji}")
