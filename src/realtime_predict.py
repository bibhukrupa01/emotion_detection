from colorama import init, Fore, Style
init()
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd
import numpy as np
import librosa
import joblib

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(project_root, "models", "emotion_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

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


def record_audio(duration=3, sr=22050):
    print("\n🎤 Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def is_silence(audio, threshold=0.01):
    rms_volume = np.sqrt(np.mean(audio**2))
    return rms_volume < threshold

from feature_extraction import extract_features_from_audio
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

    # Trim silence (strip background noise)
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)
    
    if len(trimmed_audio) < 22050 * 0.5: # Reject if less than 0.5s of actual speech
        print("⚠️ Audio too short after trimming silence. Please speak a full word.")
        continue

    # Extract features from the active speech only
    features = extract_features_from_audio(trimmed_audio, sr=22050)

    # Convert to proper shape (1 sample, 40 features)
    features = np.array(features).reshape(1, -1)

    # Scale features
    features = scaler.transform(features)

    # # Debug print
    # print("Feature sample:", features[0][:5])

    # Predict
    prediction = model.predict(features)[0]

    probs = model.predict_proba(features)[0]
    confidence = max(probs) * 100



    emotion, emoji = emotion_map.get(prediction, ("Unknown", "❓"))

    color_map = {
        "Happy": Fore.GREEN,
        "Sad": Fore.BLUE,
        "Angry": Fore.RED,
        "Fearful": Fore.MAGENTA,
        "Calm": Fore.CYAN,
        "Neutral": Fore.WHITE,
        "Disgust": Fore.YELLOW,
        "Surprised": Fore.LIGHTYELLOW_EX
    }

    color = color_map.get(emotion, Fore.WHITE)

    print("\n==============================")
    print(f"{color}✅ Detected Emotion: {emotion} {emoji}{Style.RESET_ALL}")
    print(f"{color}🔥 Confidence: {confidence:.1f}%{Style.RESET_ALL}")
    print("==============================")


