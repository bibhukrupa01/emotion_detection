import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib

model = joblib.load("models/emotion_model.pkl")

st.title("🎤 Real-Time Emotion Detection")

if st.button("Record Audio"):
    sr = 22050
    duration = 3

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = audio.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0)

    prediction = model.predict([features])

    st.success(f"Detected Emotion: {prediction[0]}")
