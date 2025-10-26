# 멀티 공정 모델 훈련 가이드

이 문서는 PRISM Monitor의 공정별 Autoencoder 모델을 훈련하는 방법을 설명합니다.

## 목차
1. [개요](#개요)
2. [데이터 준비](#데이터-준비)
3. [모델 훈련](#모델-훈련)
4. [모델 검증](#모델-검증)
5. [배포](#배포)
6. [문제 해결](#문제-해결)

---

## 개요

### 아키텍처 변경 사항

**기존 방식 (통합 모델)**:
- 모든 공정 데이터를 하나의 모델로 학습
- 데이터 소스: `prism_monitor/data/Industrial_DB_sample/*.csv`
- 단일 Autoencoder 모델

**새로운 방식 (멀티 공정 모델)**:
- 각 공정별로 독립적인 모델 훈련
- 데이터 소스: `prism_monitor/test-scenarios/test_data/semiconductor/*.csv`
- 공정별 Autoencoder 모델:
  - `semi_cmp_sensors` (CMP 공정)
  - `semi_etch_sensors` (Etch 공정)
  - `semi_cvd_sensors` (CVD 공정)
  - `semi_ion_sensors` (Ion Implant 공정)
  - `semi_photo_sensors` (Photo 공정)

### 장점
- **특화된 이상 탐지**: 각 공정의 고유 특성을 반영
- **높은 정확도**: 공정별 정상 패턴 학습
- **확장성**: 새로운 공정 추가 용이
- **유지보수**: 공정별 독립적 업데이트 가능

---

## 데이터 준비

### 1. 데이터 위치 확인

공정별 데이터는 다음 경로에 위치해야 합니다:

```
prism_monitor/test-scenarios/test_data/semiconductor/
├── semiconductor_cmp_001.csv          (CMP 공정)
├── semiconductor_etch_002.csv         (Etch 공정)
├── semiconductor_deposition_003.csv   (CVD 공정)
├── semiconductor_ion_004.csv          (Ion Implant 공정)
└── semiconductor_photo_005.csv        (Photo 공정)
```

### 2. 데이터 형식

각 CSV 파일은 다음 구조를 가져야 합니다:

**공통 컬럼**:
- `TIMESTAMP`: ISO 8601 형식 (예: `2025-05-01T00:00:00Z`)
- `SENSOR_ID` 또는 `CHAMBER_ID` 또는 `EQUIPMENT_ID`: 장비 식별자

**공정별 센서 컬럼**:

#### CMP 센서 (semiconductor_cmp_001.csv)
```
MOTOR_CURRENT, SLURRY_FLOW_RATE, PRESSURE, HEAD_ROTATION,
PLATEN_ROTATION, PAD_TEMP, SLURRY_TEMP, REMOVAL_RATE,
HEAD_PRESSURE, RETAINER_PRESSURE, CONDITIONER_PRESSURE, ENDPOINT_SIGNAL
```

#### Etch 센서 (semiconductor_etch_002.csv)
```
RF_POWER_SOURCE, RF_POWER_BIAS, CHAMBER_PRESSURE, GAS_FLOW_CF4,
GAS_FLOW_O2, GAS_FLOW_AR, GAS_FLOW_CL2, ELECTRODE_TEMP,
CHAMBER_WALL_TEMP, HELIUM_PRESSURE, ENDPOINT_SIGNAL, PLASMA_DENSITY
```

#### CVD 센서 (semiconductor_deposition_003.csv)
```
SUSCEPTOR_TEMP, CHAMBER_PRESSURE, PRECURSOR_FLOW_TEOS,
PRECURSOR_FLOW_SILANE, PRECURSOR_FLOW_WF6, CARRIER_GAS_N2,
CARRIER_GAS_H2, SHOWERHEAD_TEMP, LINER_TEMP, DEPOSITION_RATE, FILM_STRESS
```

### 3. 데이터 품질 확인

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv')

# 필수 체크
print("데이터 크기:", df.shape)
print("컬럼 목록:", df.columns.tolist())
print("결측치 확인:\n", df.isnull().sum())
print("데이터 타입:\n", df.dtypes)

# Timestamp 유효성 확인
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
print("시간 범위:", df['TIMESTAMP'].min(), "~", df['TIMESTAMP'].max())
```

**권장 사항**:
- **최소 데이터 크기**: 각 공정당 최소 10,000개 행 (더 많을수록 좋음)
- **결측치**: 5% 미만
- **시계열 연속성**: Timestamp가 시간순으로 정렬되어 있어야 함
- **이상치**: 훈련 데이터는 가능한 정상 데이터만 사용

---

## 모델 훈련

### 1. 훈련 스크립트 작성

각 공정별로 모델을 훈련하는 스크립트 예시:

```python
# train_cmp_model.py

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
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
ENCODING_DIM = 8  # 인코딩 차원 (feature 수의 1/2 ~ 1/4 권장)
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# ============================
# 1. 데이터 로드 및 전처리
# ============================
print(f"[1/6] 데이터 로드 중: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# 컬럼명 소문자 변환
df.columns = df.columns.str.lower()

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
print("[2/6] Feature 선택 중")

# 센서 컬럼만 선택 (timestamp, equipment_id 등 제외)
exclude_cols = ['timestamp', 'equipment_id', 'sensor_id', 'chamber_id', 'lot_no', 'pno']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"선택된 Features ({len(feature_cols)}개): {feature_cols}")

X = df[feature_cols].values

# 결측치 처리
X = np.nan_to_num(X, nan=0.0)

# ============================
# 3. 데이터 정규화
# ============================
print("[3/6] 데이터 정규화 중")

# RobustScaler 사용 (이상치에 강건)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"정규화 완료: {X_scaled.shape}")

# ============================
# 4. Autoencoder 모델 구성
# ============================
print("[4/6] Autoencoder 모델 구성 중")

input_dim = X_scaled.shape[1]

# Encoder
encoder_input = layers.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(encoder_input)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(ENCODING_DIM, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

# Autoencoder
autoencoder = keras.Model(encoder_input, decoded)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mse']
)

print(autoencoder.summary())

# ============================
# 5. 모델 훈련
# ============================
print(f"[5/6] 모델 훈련 중 (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})")

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    verbose=1
)

# ============================
# 6. Threshold 계산
# ============================
print("[6/6] Threshold 계산 중")

# 훈련 데이터로 reconstruction error 계산
reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

# Threshold: 99 percentile
threshold = np.percentile(mse, 99)
print(f"계산된 Threshold (99th percentile): {threshold:.6f}")

# ============================
# 7. 모델 저장
# ============================
print(f"모델 저장 중: {OUTPUT_DIR}")
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
    }
}

metadata_file = os.path.join(OUTPUT_DIR, 'model_metadata.json')
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ 메타데이터 저장: {metadata_file}")

print("\n" + "="*50)
print("모델 훈련 완료!")
print(f"모델 디렉토리: {OUTPUT_DIR}")
print("="*50)
```

### 2. 모든 공정에 대해 훈련 실행

```bash
# CMP 모델 훈련
python train_cmp_model.py

# Etch 모델 훈련 (train_cmp_model.py를 복사하여 PROCESS_NAME, DATA_FILE 수정)
python train_etch_model.py

# CVD 모델 훈련
python train_cvd_model.py

# Ion Implant 모델 훈련
python train_ion_model.py

# Photo 모델 훈련
python train_photo_model.py
```

### 3. 훈련 결과 확인

각 공정별로 다음 파일들이 생성됩니다:

```
models/
├── semi_cmp_sensors/
│   ├── autoencoder_model.h5      # Keras 모델
│   ├── scaler.pkl                # StandardScaler/RobustScaler
│   └── model_metadata.json       # 메타데이터
├── semi_etch_sensors/
│   ├── autoencoder_model.h5
│   ├── scaler.pkl
│   └── model_metadata.json
├── semi_cvd_sensors/
│   ├── autoencoder_model.h5
│   ├── scaler.pkl
│   └── model_metadata.json
... (나머지 공정들)
```

---

## 모델 검증

### 1. ProcessModelManager로 모델 로드 테스트

```python
from prism_monitor.utils.process_model_manager import ProcessModelManager

# 모델 매니저 초기화
manager = ProcessModelManager(base_model_dir='models')

# 사용 가능한 공정 확인
available_processes = manager.list_available_processes()
print(f"사용 가능한 공정: {available_processes}")

# CMP 모델 로드
model, scaler, metadata = manager.get_model_for_process('semi_cmp_sensors')
print(f"모델 버전: {metadata['model_version']}")
print(f"Feature 개수: {len(metadata['feature_columns'])}")
print(f"Threshold: {metadata['threshold']}")
```

### 2. 이상 탐지 테스트

```python
import pandas as pd
import numpy as np

# 테스트 데이터 로드
test_data = pd.read_csv('prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv')
test_data.columns = test_data.columns.str.lower()

# Feature 추출
feature_cols = metadata['feature_columns']
X_test = test_data[feature_cols].values
X_test = np.nan_to_num(X_test, nan=0.0)

# 정규화
X_test_scaled = scaler.transform(X_test)

# 예측
reconstructed = model.predict(X_test_scaled)

# Reconstruction Error 계산
mse = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)

# 이상치 탐지
threshold = metadata['threshold']
anomalies = mse > threshold

print(f"총 샘플 수: {len(X_test)}")
print(f"이상치 개수: {anomalies.sum()}")
print(f"이상치 비율: {anomalies.sum() / len(X_test) * 100:.2f}%")
```

### 3. 성능 지표 확인

```python
# ROC Curve (정답 레이블이 있는 경우)
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# y_true: 실제 이상 여부 (0: 정상, 1: 이상)
# 예시: y_true = test_data['is_anomaly'].values

if 'is_anomaly' in test_data.columns:
    y_true = test_data['is_anomaly'].values

    # AUC 계산
    auc_score = roc_auc_score(y_true, mse)
    print(f"AUC Score: {auc_score:.4f}")

    # ROC Curve 그리기
    fpr, tpr, thresholds = roc_curve(y_true, mse)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'models/{PROCESS_NAME}/roc_curve.png')
    print("ROC Curve 저장 완료")
```

---

## 배포

### 1. 모델 디렉토리 구조 확인

배포 전에 다음 구조가 올바른지 확인:

```
models/
├── semi_cmp_sensors/
│   ├── autoencoder_model.h5
│   ├── scaler.pkl
│   └── model_metadata.json
├── semi_etch_sensors/
│   ├── autoencoder_model.h5
│   ├── scaler.pkl
│   └── model_metadata.json
...
```

### 2. 시스템 통합 테스트

```python
from prism_monitor.data.database import PrismCoreDataBase
from prism_monitor.modules.event.event_detect import detect_anomalies_realtime

# DB 연결
prism_core_db = PrismCoreDataBase(db_path='my_database.db')

# CMP 공정 이상 탐지
start_time = '2025-05-01T00:00:00Z'
end_time = '2025-05-01T01:00:00Z'

anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(
    prism_core_db,
    start=start_time,
    end=end_time,
    target_process='semi_cmp_sensors'  # 공정 지정
)

print(f"탐지된 이상: {len(anomalies)}건")
print(f"Drift 탐지: {len(drift_results)}건")
```

### 3. API 엔드포인트 테스트

```bash
# 모니터링 이벤트 감지 (공정 지정)
curl -X POST http://localhost:8000/api/monitoring/event/detect \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_001",
    "start": "2025-05-01T00:00:00Z",
    "end": "2025-05-01T01:00:00Z",
    "target_process": "semi_cmp_sensors"
  }'

# 결과 설명 (공정별 프롬프트 사용)
curl -X POST http://localhost:8000/api/monitoring/event/explain \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_001"
  }'
```

---

## 문제 해결

### Q1. 모델 로드 실패: "Model directory not found"

**원인**: 모델 디렉토리가 존재하지 않거나 경로가 잘못됨

**해결**:
```python
import os

# 모델 디렉토리 확인
model_dir = 'models/semi_cmp_sensors'
print(f"존재 여부: {os.path.exists(model_dir)}")

# 필요 파일 확인
for file in ['autoencoder_model.h5', 'scaler.pkl', 'model_metadata.json']:
    file_path = os.path.join(model_dir, file)
    print(f"{file}: {os.path.exists(file_path)}")
```

### Q2. 데이터 정규화 오류: "Feature mismatch"

**원인**: 훈련 시 사용한 feature와 추론 시 feature가 다름

**해결**:
```python
# 메타데이터에서 feature 목록 확인
with open('models/semi_cmp_sensors/model_metadata.json') as f:
    metadata = json.load(f)

required_features = metadata['feature_columns']
print(f"필요한 Features: {required_features}")

# 데이터 확인
df = pd.read_csv('your_data.csv')
df.columns = df.columns.str.lower()
missing_features = set(required_features) - set(df.columns)
print(f"누락된 Features: {missing_features}")
```

### Q3. Threshold 조정 필요

**원인**: False Positive 또는 False Negative가 많음

**해결**:
```python
# Threshold 재계산
import json

metadata_file = 'models/semi_cmp_sensors/model_metadata.json'
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# 현재 threshold 확인
current_threshold = metadata['threshold']
print(f"현재 Threshold: {current_threshold}")

# False Positive 줄이기: percentile 증가 (예: 99 → 99.5)
# False Negative 줄이기: percentile 감소 (예: 99 → 98)

# 재계산 후 메타데이터 업데이트
new_threshold = np.percentile(mse, 99.5)
metadata['threshold'] = float(new_threshold)

with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Q4. query_decompose 통합 시 오류

**원인**: query_decompose가 아직 classified_class를 반환하지 않음

**해결**:
```python
# monitoring.py에서 주석 해제가 필요함
# query_decompose.py가 다음과 같이 수정된 후:
# return timestamp_min, timestamp_max, result_df, classified_class

# monitoring.py (line 47-51) 주석 해제:
from prism_monitor.modules.query_decompose.query_decompose import query_decompose
timestamp_min, timestamp_max, result_df, classified_class = query_decompose(user_query)
classified_process = classified_class
print(f"query_decompose identified process: {classified_process}")
```

### Q5. 메모리 부족

**원인**: 대용량 데이터 또는 여러 모델 동시 로드

**해결**:
```python
# 모델 캐시 초기화
from prism_monitor.utils.process_model_manager import ProcessModelManager

manager = ProcessModelManager()
manager.clear_cache()

# 배치 크기 조정
BATCH_SIZE = 64  # 기본 128에서 감소
```

---

## 부록

### A. 권장 하이퍼파라미터

| 파라미터 | 권장 값 | 설명 |
|---------|--------|------|
| ENCODING_DIM | feature 수 / 2 ~ 4 | 인코딩 차원 (압축률) |
| LEARNING_RATE | 0.001 ~ 0.01 | Adam optimizer 학습률 |
| EPOCHS | 50 ~ 200 | 훈련 에폭 수 |
| BATCH_SIZE | 64 ~ 256 | 배치 크기 |
| VALIDATION_SPLIT | 0.1 ~ 0.2 | 검증 데이터 비율 |
| THRESHOLD_PERCENTILE | 95 ~ 99.5 | 이상치 판정 기준 |

### B. 참고 자료

- **Autoencoder 개념**: [Keras Autoencoder Tutorial](https://blog.keras.io/building-autoencoders-in-keras.html)
- **이상 탐지**: [Anomaly Detection with Autoencoders](https://towardsdatascience.com/anomaly-detection-with-autoencoders-7e4fe90f5e14)
- **반도체 공정**: 각 공정별 센서 데이터 특성 이해 필요

### C. 연락처

문의사항이나 이슈가 있을 경우:
- GitHub Issues: [PRISM-Monitor Issues](https://github.com/your-org/PRISM-Monitor/issues)
- 개발팀 이메일: prism-support@example.com

---

**문서 버전**: 1.0
**최종 수정일**: 2025-10-23
**작성자**: PRISM Monitor Team
