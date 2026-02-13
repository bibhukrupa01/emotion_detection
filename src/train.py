print("Training started...")

import os
import numpy as np
from feature_extraction import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

X, y = [], []

dataset_path = "../dataset/RAVDESS"

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    for file in os.listdir(actor_path):
        file_path = os.path.join(actor_path, file)
        print("Processing:", file_path)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        emotion = file.split("-")[2]
        X.append(extract_features(file_path))
        y.append(emotion)
    

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)



print("Accuracy:", model.score(X_test, y_test))
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(model, "../models/emotion_model.pkl")
