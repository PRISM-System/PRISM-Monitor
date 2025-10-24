"""
20개 CSV 파일 모델 일괄 훈련 스크립트

각 CSV 파일마다 독립적인 Autoencoder 모델을 훈련합니다.
"""

import os
# CPU만 사용 (안정성을 위해)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from glob import glob

print("TensorFlow 버전:", tf.__version__)
print("실행 모드: CPU (안정성)")

# 설정
BASE_DATA_DIR = "prism_monitor/test-scenarios/test_data"
BASE_MODEL_DIR = "models"
MODEL_VERSION = "v1.0"

# Autoencoder 하이퍼파라미터
ENCODING_DIM_RATIO = 0.5  # input_dim의 비율
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
THRESHOLD_PERCENTILE = 99  # 이상치 threshold

print("="*70)
print("20개 CSV 파일 모델 일괄 훈련")
print("="*70)

# 모든 CSV 파일 찾기
csv_files = []
for category in ['semiconductor', 'automotive', 'battery', 'chemical', 'steel']:
    category_path = os.path.join(BASE_DATA_DIR, category)
    if os.path.exists(category_path):
        files = glob(os.path.join(category_path, '*.csv'))
        csv_files.extend(files)

print(f"\n✓ 총 {len(csv_files)}개 CSV 파일 발견:")
for i, file in enumerate(csv_files, 1):
    print(f"  {i}. {os.path.basename(file)}")

# 각 파일별로 모델 훈련
successful_models = []
failed_models = []

for file_path in csv_files:
    filename = os.path.basename(file_path)
    file_identifier = filename.replace('.csv', '')

    print("\n" + "="*70)
    print(f"모델 훈련 중: {file_identifier}")
    print("="*70)

    try:
        # 1. 데이터 로드
        print(f"[1/7] 데이터 로드: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✓ 원본 데이터: {df.shape[0]:,} 행 x {df.shape[1]} 열")

        # 컬럼명 소문자 변환
        df.columns = df.columns.str.lower()

        # 2. Feature 선택
        print("[2/7] Feature 선택")
        exclude_cols = ['timestamp', 'equipment_id', 'sensor_id', 'chamber_id', 'lot_no', 'pno']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        print(f"✓ Features ({len(feature_cols)}개): {feature_cols}")

        features_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        dropped_cols = [col for col in features_df.columns if features_df[col].isna().all()]
        if dropped_cols:
            print(f"⚠️  숫자로 변환되지 않아 제외된 컬럼: {dropped_cols}")
            features_df = features_df.drop(columns=dropped_cols)

        if features_df.empty:
            print("✗ 사용 가능한 숫자 Feature가 없음, 스킵")
            failed_models.append((file_identifier, "no_numeric_features"))
            continue

        X = features_df.values
        X = np.nan_to_num(X, nan=0.0)

        if X.shape[0] < 100:
            print(f"⚠️  데이터가 너무 적음 ({X.shape[0]}개), 스킵")
            failed_models.append((file_identifier, "insufficient_data"))
            continue

        # 3. 데이터 정규화
        print("[3/7] 데이터 정규화 (RobustScaler)")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. Autoencoder 구성
        print("[4/7] Autoencoder 모델 구성")
        input_dim = X_scaled.shape[1]
        encoding_dim = max(2, int(input_dim * ENCODING_DIM_RATIO))  # 최소 2개

        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(max(8, encoding_dim * 2), activation='relu')(encoder_input)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

        decoded = layers.Dense(max(8, encoding_dim * 2), activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = keras.Model(encoder_input, decoded, name=f'{file_identifier}_autoencoder')

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mse']
        )

        print(f"✓ 모델 구조: input({input_dim}) → encode({encoding_dim}) → decode({input_dim})")

        # 5. 모델 훈련
        print(f"[5/7] 모델 훈련 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE})")
        history = autoencoder.fit(
            X_scaled, X_scaled,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            verbose=0  # 조용히
        )

        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"✓ Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")

        # 6. Threshold 계산
        print("[6/7] Threshold 계산")
        reconstructed = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
        print(f"✓ Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")

        # 7. 모델 저장
        print(f"[7/7] 모델 저장")
        output_dir = os.path.join(BASE_MODEL_DIR, file_identifier)
        os.makedirs(output_dir, exist_ok=True)

        # 모델 저장
        model_file = os.path.join(output_dir, 'autoencoder_model.h5')
        autoencoder.save(model_file)

        # 스케일러 저장
        scaler_file = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

        # 메타데이터 저장
        metadata = {
            'file_identifier': file_identifier,
            'model_version': MODEL_VERSION,
            'training_timestamp': datetime.now().isoformat(),
            'feature_columns': feature_cols,
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'threshold': float(threshold),
            'training_data_info': {
                'data_file': file_path,
                'num_samples': len(X),
                'num_features': len(feature_cols),
            },
            'hyperparameters': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'threshold_percentile': THRESHOLD_PERCENTILE,
            },
            'performance_metrics': {
                'final_loss': float(final_loss),
                'final_val_loss': float(final_val_loss),
                'mse_mean': float(np.mean(mse)),
                'mse_std': float(np.std(mse)),
            }
        }

        metadata_file = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ 저장 완료: {output_dir}")
        successful_models.append(file_identifier)

    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        failed_models.append((file_identifier, str(e)))

# 최종 요약
print("\n" + "="*70)
print("훈련 완료 요약")
print("="*70)
print(f"✓ 성공: {len(successful_models)}개")
for model in successful_models:
    print(f"  - {model}")

if failed_models:
    print(f"\n✗ 실패: {len(failed_models)}개")
    for model, reason in failed_models:
        print(f"  - {model}: {reason}")

print(f"\n총 {len(successful_models) + len(failed_models)}개 중 {len(successful_models)}개 성공")
print("="*70)

# 모델 디렉토리 구조 출력
print("\n모델 디렉토리 구조:")
print(f"{BASE_MODEL_DIR}/")
for model in successful_models[:5]:  # 처음 5개만
    print(f"├── {model}/")
    print(f"│   ├── autoencoder_model.h5")
    print(f"│   ├── scaler.pkl")
    print(f"│   └── model_metadata.json")
if len(successful_models) > 5:
    print(f"└── ... (외 {len(successful_models) - 5}개)")
