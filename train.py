import os
import numpy as np
import pandas as pd
from audio import AudioFeature
from model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

def get_audio_files(genres_dir):
    """Lấy danh sách các file audio từ thư mục genres"""
    audio_files = []
    for genre in os.listdir(genres_dir):
        genre_path = os.path.join(genres_dir, genre)
        if os.path.isdir(genre_path):
            for audio_file in os.listdir(genre_path):
                if audio_file.endswith('.wav'):
                    audio_files.append((os.path.join(genre_path, audio_file), genre))
    return audio_files

def extract_features(audio_files):
    """Trích xuất đặc trưng từ các file audio"""
    print("🔍 Đang trích xuất đặc trưng từ các file audio...")
    audio_features = []
    
    for path, genre in tqdm(audio_files):
        try:
            audio = AudioFeature(path, genre)
            # Trích xuất tất cả các đặc trưng có sẵn
            audio.extract_features(
                "mfcc",           # Mel-frequency cepstral coefficients
                "chroma",         # Chroma features
                "zcr",           # Zero crossing rate
                "spectral_contrast", # Spectral contrast
                "rolloff",       # Spectral rolloff
                "tempo"          # Tempo/BPM
            )
            audio_features.append(audio)
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý file {path}: {str(e)}")
            continue
    
    return audio_features

def prepare_training_data(audio_features):
    """Chuẩn bị dữ liệu cho việc huấn luyện"""
    print("📊 Đang chuẩn bị dữ liệu huấn luyện...")
    feature_matrix = np.vstack([audio.features for audio in audio_features])
    genre_labels = [audio.genre for audio in audio_features]
    return feature_matrix, genre_labels

def train_model(feature_matrix, genre_labels):
    """Huấn luyện mô hình với các tham số đã được tối ưu"""
    print("🎯 Đang huấn luyện mô hình...")
    
    model_cfg = dict(
        tt_test_dict=dict(shuffle=True, test_size=0.2),  # 20% cho test set
        tt_val_dict=dict(shuffle=True, test_size=0.2),   # 20% cho validation set
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Sử dụng tất cả CPU cores
            class_weight="balanced",
            n_estimators=500,  # Tăng số lượng cây
            bootstrap=True,
            max_depth=20,      # Giới hạn độ sâu của cây
            min_samples_split=5,
            min_samples_leaf=2
        ),
        param_grid=dict(
            model__criterion=["entropy", "gini"],
            model__max_features=["sqrt", "log2"],
            model__min_samples_leaf=[2, 3, 4],
            model__max_depth=[15, 20, 25]
        ),
        grid_dict=dict(
            n_jobs=-1,
            refit=True,
            scoring="balanced_accuracy",
            verbose=2
        ),
        kf_dict=dict(n_splits=5, random_state=42, shuffle=True)
    )

    model = Model(feature_matrix, genre_labels, model_cfg)
    model.train_kfold()
    
    # Đánh giá mô hình
    print("\n📈 Kết quả trên tập validation:")
    model.predict(holdout_type="val")
    print("\n📈 Kết quả trên tập test:")
    model.predict(holdout_type="test")
    
    return model

def save_model(model, output_path="trained_model.pkl"):
    """Lưu mô hình đã huấn luyện"""
    print(f"💾 Đang lưu mô hình vào {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Đã lưu mô hình thành công!")

def main():
    print("🎵 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI THỂ LOẠI NHẠC 🎵")
    
    # Đường dẫn đến thư mục chứa các thể loại nhạc
    genres_dir = "archive/Data/genres_original"
    
    # Lấy danh sách file audio
    audio_files = get_audio_files(genres_dir)
    print(f"📁 Tìm thấy {len(audio_files)} file audio")
    
    # Trích xuất đặc trưng
    audio_features = extract_features(audio_files)
    print(f"✨ Đã trích xuất đặc trưng từ {len(audio_features)} file")
    
    # Chuẩn bị dữ liệu
    feature_matrix, genre_labels = prepare_training_data(audio_features)
    
    # Huấn luyện mô hình
    model = train_model(feature_matrix, genre_labels)
    
    # Lưu mô hình
    save_model(model)
    
    print("\n🎉 Quá trình huấn luyện hoàn tất!")

if __name__ == "__main__":
    main() 