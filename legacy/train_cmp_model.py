"""
CMP 공정 모델 훈련 스크립트
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# ============================
# 설정
# ============================
PROCESS_NAME = 'semi_cmp_sensors'
DATA_FILE = 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv'
OUTPUT_DIR = f'models/{PROCESS_NAME}'
MODEL_VERSION = 'v1.0'

# Autoencoder 하이퍼파라미터
ENCODING_DIM = 4  # 인코딩 차원
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

print("="*70)
print("CMP 공정 모델 훈련")
print("="*70)

# ============================
# 1. 데이터 로드 및 전처리
# ============================
print(f"\n[1/7] 데이터 로드 중: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"✓ 원본 데이터: {df.shape[0]:,} 행 x {df.shape[1]} 열")

# 컬럼명 소문자 변환
df.columns = df.columns.str.lower()
print(f"✓ 컬럼: {list(df.columns)}")

# Timestamp 처리
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Equipment ID 생성
if 'equipment_id' not in df.columns:
    if 'sensor_id' in df.columns:
        df['equipment_id'] = df['sensor_id']
    elif 'chamber_id' in df.columns:
        df['equipment_id'] = df['chamber_id']

# ============================
# 2. Feature 선택
# ============================
print("\n[2/7] Feature 선택 중")

# 센서 컬럼만 선택
exclude_cols = ['timestamp', 'equipment_id', 'sensor_id', 'chamber_id', 'lot_no', 'pno']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"✓ 선택된 Features ({len(feature_cols)}개): {feature_cols}")

X = df[feature_cols].values

# 결측치 확인
null_counts = pd.DataFrame(X, columns=feature_cols).isnull().sum()
if null_counts.sum() > 0:
    print(f"⚠️  결측치 발견: {null_counts[null_counts > 0]}")
    print("   → 0으로 대체")

X = np.nan_to_num(X, nan=0.0)

print(f"✓ Feature Matrix: {X.shape}")

# ============================
# 3. 데이터 정규화
# ============================
print("\n[3/7] 데이터 정규화 중 (RobustScaler)")

# RobustScaler 사용 (이상치에 강건)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ 정규화 완료")
print(f"  - 평균: {np.mean(X_scaled, axis=0)[:3]}")
print(f"  - 표준편차: {np.std(X_scaled, axis=0)[:3]}")

# ============================
# 4. Autoencoder 모델 구성
# ============================
print(f"\n[4/7] Autoencoder 모델 구성 중")

input_dim = X_scaled.shape[1]

# Encoder
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
encoded = layers.Dense(8, activation='relu', name='encoder_layer1')(encoder_input)
encoded = layers.Dense(ENCODING_DIM, activation='relu', name='encoder_output')(encoded)

# Decoder
decoded = layers.Dense(8, activation='relu', name='decoder_layer1')(encoded)
decoded = layers.Dense(input_dim, activation='linear', name='decoder_output')(decoded)

# Autoencoder
autoencoder = keras.Model(encoder_input, decoded, name='cmp_autoencoder')

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mse']
)

print(f"✓ 모델 구조:")
autoencoder.summary()

# ============================
# 5. 모델 훈련
# ============================
print(f"\n[5/7] 모델 훈련 중...")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Validation Split: {VALIDATION_SPLIT}")

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    verbose=1
)

print(f"\n✓ 훈련 완료!")
print(f"  - Final Loss: {history.history['loss'][-1]:.6f}")
print(f"  - Final Val Loss: {history.history['val_loss'][-1]:.6f}")

# ============================
# 6. Threshold 계산
# ============================
print("\n[6/7] Threshold 계산 중...")

# 훈련 데이터로 reconstruction error 계산
reconstructed = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

# Threshold: 99 percentile
threshold = np.percentile(mse, 99)

print(f"✓ Reconstruction Error 통계:")
print(f"  - Mean: {np.mean(mse):.6f}")
print(f"  - Median: {np.median(mse):.6f}")
print(f"  - 95th percentile: {np.percentile(mse, 95):.6f}")
print(f"  - 99th percentile (threshold): {threshold:.6f}")

# ============================
# 7. 모델 저장
# ============================
print(f"\n[7/7] 모델 저장 중: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 저장
model_file = os.path.join(OUTPUT_DIR, 'autoencoder_model.h5')
autoencoder.save(model_file)
print(f"✓ 모델 저장: {model_file}")

# 스케일러 저장
scaler_file = os.path.join(OUTPUT_DIR, 'scaler.pkl')
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ 스케일러 저장: {scaler_file}")

# 메타데이터 저장
metadata = {
    'process_name': PROCESS_NAME,
    'model_version': MODEL_VERSION,
    'training_timestamp': datetime.now().isoformat(),
    'feature_columns': feature_cols,
    'input_dim': input_dim,
    'encoding_dim': ENCODING_DIM,
    'threshold': float(threshold),
    'training_data_info': {
        'data_file': DATA_FILE,
        'num_samples': len(X),
        'num_features': len(feature_cols),
    },
    'hyperparameters': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    },
    'performance_metrics': {
        'final_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'mse_mean': float(np.mean(mse)),
        'mse_std': float(np.std(mse)),
    }
}

metadata_file = os.path.join(OUTPUT_DIR, 'model_metadata.json')
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ 메타데이터 저장: {metadata_file}")

print("\n" + "="*70)
print("✓ CMP 모델 훈련 완료!")
print(f"  모델 디렉토리: {OUTPUT_DIR}")
print(f"  Feature 개수: {len(feature_cols)}")
print(f"  Threshold: {threshold:.6f}")
print("="*70)
