# music_genre_classifier.py

import numpy as np
import pandas as pd
import os
import pickle
from audio import AudioFeature
from model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_model():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("📦 Mô hình đã được nạp từ file.")
    return model

def predict_genre(model, mp3_path):
    audio = AudioFeature(mp3_path, genre=None)
    audio.extract_features("mfcc", "chroma", "zcr", "spectral_contrast", "rolloff", "tempo")
    X_new = model.best_estimator['scaler'].transform(audio.features.reshape(1, -1))
    pred_index = model.best_estimator['model'].predict(X_new)[0]
    genre = model.encoder.inverse_transform([pred_index])[0]
    print(f"\n🎵 File '{mp3_path}' được dự đoán là thể loại: {genre}")
    return genre

if __name__ == "__main__":
    print("🎧 HỆ THỐNG PHÂN LOẠI THỂ LOẠI NHẠC 🎶")
    
    try:
        model = load_model()
    except FileNotFoundError:
        print("❌ Không tìm thấy file mô hình trained_model.pkl")
        exit()

    mp3_path = input("🔍 Nhập đường dẫn đến file nhạc (.mp3): ")
    if not os.path.exists(mp3_path):
        print("❌ Không tìm thấy file mp3. Kiểm tra lại đường dẫn.")
    else:
        predict_genre(model, mp3_path)
