"""
빠른 테스트 스크립트 (설정 없이 즉시 실행)

CSV 파일을 직접 읽어서 이상 탐지 수행
"""

import pandas as pd
import numpy as np
from glob import glob
import os


def quick_test_with_csv():
    """
    CSV 파일로 직접 테스트 (DB 없이)
    """
    print("="*70)
    print("빠른 이상 탐지 테스트 (CSV 직접 로드)")
    print("="*70)

    # 1. 데이터 파일 찾기
    print("\n[1/5] 데이터 파일 검색 중...")
    data_paths = glob('prism_monitor/test-scenarios/test_data/semiconductor/*.csv')

    if not data_paths:
        data_paths = glob('prism_monitor/data/Industrial_DB_sample/*.csv')

    if not data_paths:
        print("✗ 데이터 파일을 찾을 수 없습니다.")
        return

    print(f"✓ 발견된 파일 ({len(data_paths)}개):")
    for i, path in enumerate(data_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")

    # 첫 번째 파일 선택
    selected_file = data_paths[0]
    print(f"\n선택된 파일: {os.path.basename(selected_file)}")

    # 2. 데이터 로드
    print("\n[2/5] 데이터 로드 중...")
    df = pd.read_csv(selected_file)
    print(f"✓ 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 열")

    # 컬럼명 정규화
    df.columns = df.columns.str.lower()

    # 3. Feature 추출
    print("\n[3/5] Feature 추출 중...")
    exclude_cols = ['timestamp', 'sensor_id', 'chamber_id', 'equipment_id', 'lot_no', 'pno']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    print(f"✓ Feature 컬럼 ({len(feature_cols)}개):")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    if len(feature_cols) > 10:
        print(f"  ... 외 {len(feature_cols) - 10}개")

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    # 4. 간단한 이상 탐지 (통계 기반)
    print("\n[4/5] 이상 탐지 수행 중 (Z-score 방법)...")

    # Z-score 계산
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # 0 방지

    z_scores = np.abs((X - mean) / std)

    # 이상치 판정 (Z-score > 3)
    threshold = 3
    anomaly_mask = np.any(z_scores > threshold, axis=1)

    anomaly_indices = np.where(anomaly_mask)[0]

    print(f"✓ 탐지 완료!")
    print(f"  - 총 샘플: {len(X):,}개")
    print(f"  - 이상치: {len(anomaly_indices):,}개")
    print(f"  - 이상 비율: {len(anomaly_indices) / len(X) * 100:.2f}%")

    # 5. 이상치 상세 정보
    print("\n[5/5] 이상치 상세 정보 (상위 5개):")

    if len(anomaly_indices) > 0:
        # 이상 점수 계산
        anomaly_scores = np.max(z_scores, axis=1)

        # 상위 5개 추출
        top_k = min(5, len(anomaly_indices))
        top_indices = anomaly_indices[np.argsort(-anomaly_scores[anomaly_indices])[:top_k]]

        for i, idx in enumerate(top_indices, 1):
            print(f"\n  이상치 {i} (행 {idx}):")

            # Timestamp 확인
            if 'timestamp' in df.columns:
                print(f"    - Timestamp: {df.iloc[idx]['timestamp']}")

            # 이상 점수
            print(f"    - 이상 점수 (Z-score): {anomaly_scores[idx]:.2f}")

            # 이상치 컬럼 확인
            row_z_scores = z_scores[idx]
            anomaly_col_indices = np.where(row_z_scores > threshold)[0]

            print(f"    - 이상 컬럼 ({len(anomaly_col_indices)}개):")
            for col_idx in anomaly_col_indices[:3]:  # 최대 3개
                col_name = feature_cols[col_idx]
                value = X[idx, col_idx]
                z_score = row_z_scores[col_idx]
                print(f"      · {col_name}: {value:.2f} (Z-score: {z_score:.2f})")

            if len(anomaly_col_indices) > 3:
                print(f"      ... 외 {len(anomaly_col_indices) - 3}개")

        # 결과 저장
        anomaly_df = df.iloc[anomaly_indices].copy()
        anomaly_df['anomaly_score'] = anomaly_scores[anomaly_indices]

        output_file = 'quick_test_anomalies.csv'
        anomaly_df.to_csv(output_file, index=False)
        print(f"\n✓ 이상치 데이터 저장: {output_file}")

    else:
        print("  ✓ 이상치가 탐지되지 않았습니다.")

    print("\n" + "="*70)
    print("✓ 빠른 테스트 완료!")
    print("="*70)


def test_with_autoencoder():
    """
    기존 Autoencoder 모델로 테스트
    """
    print("="*70)
    print("Autoencoder 모델 테스트")
    print("="*70)

    # 모델 확인
    model_file = 'models/autoencoder_model.h5'
    scaler_file = 'models/scaler.pkl'
    metadata_file = 'models/model_metadata.json'

    if not os.path.exists(model_file):
        print(f"\n✗ 모델 파일을 찾을 수 없습니다: {model_file}")
        print("   레거시 모드로 테스트하려면 모델이 필요합니다.")
        return

    print(f"\n[1/6] 모델 로드 중...")
    print(f"  - 모델: {model_file}")
    print(f"  - 스케일러: {scaler_file}")

    import pickle
    from tensorflow import keras

    model = keras.models.load_model(model_file)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    # 메타데이터 로드
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"✓ 모델 정보:")
    print(f"  - 버전: {metadata.get('model_version', 'unknown')}")
    print(f"  - Feature 개수: {len(metadata.get('feature_columns', []))}")
    print(f"  - Threshold: {metadata.get('threshold', 0):.6f}")

    # 데이터 로드
    print(f"\n[2/6] 데이터 로드 중...")
    data_paths = glob('prism_monitor/test-scenarios/test_data/semiconductor/*.csv')
    if not data_paths:
        data_paths = glob('prism_monitor/data/Industrial_DB_sample/*.csv')

    if not data_paths:
        print("✗ 데이터 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(data_paths[0])
    df.columns = df.columns.str.lower()
    print(f"✓ 데이터: {df.shape[0]:,} 행")

    # Feature 추출
    print(f"\n[3/6] Feature 추출 중...")
    feature_cols = metadata['feature_columns']

    # 데이터에 없는 컬럼은 0으로 채움
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    print(f"✓ Feature: {X.shape}")

    # 정규화
    print(f"\n[4/6] 데이터 정규화 중...")
    X_scaled = scaler.transform(X)

    # 예측
    print(f"\n[5/6] 이상 탐지 수행 중...")
    reconstructed = model.predict(X_scaled, verbose=0)

    # Reconstruction Error
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    # 이상치 판정
    threshold = metadata['threshold']
    anomalies = mse > threshold

    print(f"✓ 탐지 완료!")
    print(f"  - 총 샘플: {len(X):,}개")
    print(f"  - 이상치: {anomalies.sum():,}개")
    print(f"  - 이상 비율: {anomalies.sum() / len(X) * 100:.2f}%")

    # 상세 정보
    print(f"\n[6/6] 이상치 상세 정보 (상위 5개):")

    if anomalies.sum() > 0:
        anomaly_indices = np.where(anomalies)[0]
        top_k = min(5, len(anomaly_indices))
        top_indices = anomaly_indices[np.argsort(-mse[anomaly_indices])[:top_k]]

        for i, idx in enumerate(top_indices, 1):
            print(f"\n  이상치 {i} (행 {idx}):")
            if 'timestamp' in df.columns:
                print(f"    - Timestamp: {df.iloc[idx]['timestamp']}")
            print(f"    - Anomaly Score: {mse[idx]:.6f}")
            print(f"    - Threshold: {threshold:.6f}")

        # 결과 저장
        anomaly_df = df.iloc[anomaly_indices].copy()
        anomaly_df['anomaly_score'] = mse[anomaly_indices]

        output_file = 'autoencoder_test_anomalies.csv'
        anomaly_df.to_csv(output_file, index=False)
        print(f"\n✓ 이상치 데이터 저장: {output_file}")
    else:
        print("  ✓ 이상치가 탐지되지 않았습니다.")

    print("\n" + "="*70)
    print("✓ Autoencoder 테스트 완료!")
    print("="*70)


if __name__ == '__main__':
    import sys

    print("\n빠른 테스트 옵션:")
    print("1. 통계 기반 이상 탐지 (모델 없이)")
    print("2. Autoencoder 모델 테스트 (모델 필요)")

    if len(sys.argv) > 1 and sys.argv[1] == '2':
        test_with_autoencoder()
    else:
        quick_test_with_csv()
