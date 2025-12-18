import numpy as np
import joblib

from audio_cleaner import clean_audio
from feature_extractor import extract_features
from preprocess import preprocess
import os

# Load model + encoder
BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, "models", "xgboost_model.pkl"))
encoder = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))


def predict_insect(audio_path):
    """
    Full pipeline:
    - Clean audio
    - Extract features
    - Preprocess (scaling + reshape)
    - Predict using XGBoost
    - Decode label
    """

    # 1. Clean audio
    cleaned_path = clean_audio(audio_path)

    # 2. Extract features
    features = extract_features(cleaned_path)

    # 3. Preprocess (scaling + DF reconstruction)
    scaled = preprocess(features)

    # 4. Predict
    pred = model.predict(scaled)[0]

    # 5. Decode label
    label = encoder.inverse_transform([pred])[0]

    return label
