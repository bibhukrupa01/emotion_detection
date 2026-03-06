import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.feature_extraction import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

print("Training started...")

X, y = [], []

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_root, "dataset", "RAVDESS")

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    for file in os.listdir(actor_path):
        file_path = os.path.join(actor_path, file)
        print("Processing:", file_path)

        emotion = file.split("-")[2]
        X.append(extract_features(file_path))
        y.append(emotion)
    

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Transform test subset as well

scaler_path = os.path.join(project_root, "models", "scaler.pkl")
joblib.dump(scaler, scaler_path)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)



print("Accuracy:", model.score(X_test, y_test))

model_path = os.path.join(project_root, "models", "emotion_model.pkl")
joblib.dump(model, model_path)
