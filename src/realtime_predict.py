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


import sounddevice as sd
import numpy as np
import librosa
import joblib

model = joblib.load("../models/emotion_model.pkl")

def record_audio(duration=3, sr=22050):
    print("Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def extract_live_features(audio, sr=22050):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

while True:
    audio = record_audio()
    features = extract_live_features(audio)
    prediction = model.predict([features])

    emotion = emotion_map.get(prediction[0], "Unknown")
    print("Detected Emotion:", emotion)

scaler = joblib.load("../models/scaler.pkl")
features = scaler.transform([features])
