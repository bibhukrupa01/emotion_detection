from colorama import init, Fore, Style
init()
import sounddevice as sd
import numpy as np
import librosa
import joblib

# Load model
model = joblib.load("../models/emotion_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

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

    # Extract features
    features = extract_features(audio)

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
    print(color + f"✅ Detected Emotion: {emotion} {emoji}" + Style.RESET_ALL)
    print(color + f"🔥 Confidence: {confidence:.1f}%" + Style.RESET_ALL)
    print("==============================")


