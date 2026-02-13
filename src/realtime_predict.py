import sounddevice as sd
import numpy as np
import librosa
import joblib

# Load model
model = joblib.load("../models/emotion_model.pkl")

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

def record_audio(duration=3, sr=22050):
    print("\n🎤 Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def is_silence(audio, threshold=0.01):
    volume = np.linalg.norm(audio)
    return volume < threshold

def extract_features(audio, sr=22050):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

while True:
    choice = input("\nPress ENTER to record (or type q to quit): ")

    if choice.lower() == "q":
        print("Exiting...")
        break

    audio = record_audio()

    # Silence detection
    if is_silence(audio):
        print("⚠️ No speech detected. Try speaking louder.")
        continue

    features = extract_features(audio)
    prediction = model.predict([features])[0]

    emotion = emotion_map.get(prediction, "Unknown")
    print("✅ Detected Emotion:", emotion)
