# Event Precursor Module

> **이상 징후 사전 예측 모듈**
> LSTM 기반 시계열 분석을 통한 제조 설비 이상 징후 예측 시스템

---

## 📋 목차

1. [개요](#개요)
2. [파일 구조](#파일-구조)
3. [주요 기능](#주요-기능)
4. [데이터 흐름](#데이터-흐름)
5. [함수 레퍼런스](#함수-레퍼런스)
6. [사용 방법](#사용-방법)
7. [입출력 형식](#입출력-형식)
8. [예제](#예제)
9. [파라미터 설정](#파라미터-설정)

---

## 개요

Event Precursor 모듈은 제조 공정의 센서 데이터를 분석하여 **미래의 이상 징후를 사전에 예측**하는 딥러닝 기반 시스템입니다.

### 핵심 특징

- **다중 산업 지원**: Automotive, Battery, Chemical, Semiconductor, Steel 등
- **LSTM 기반 시계열 예측**: PyTorch를 활용한 딥러닝 모델
- **3가지 예측 시나리오**:
  1. 단일 출력: 이상 징후 발생 확률 예측
  2. 다중 출력: 센서값 + 이상 징후 동시 예측
  3. 실시간 모니터링: 스트림 데이터 실시간 분석
- **자동 이상 레이블링**: Z-score 기반 비지도 학습
- **RUL 예측**: 잔여 유효 수명(Remaining Useful Life) 추정
- **경고 시스템**: 위험도별 알림 생성 (WARNING/CRITICAL)

### 기술 스택

```
- Python 3.11
- PyTorch (LSTM 모델)
- Pandas (데이터 처리)
- Scikit-learn (전처리, 분할)
- NumPy (수치 연산)
- Matplotlib (시각화)
```

---

## 파일 구조

```
event_precursor/
├── _precursor.py      # 핵심 로직 구현 (내부 모듈)
├── precursor.py       # 실행 인터페이스 (외부 API)
└── README.md          # 문서 (본 파일)
```

### `_precursor.py` (Core Module)

**역할**: 모든 핵심 기능 구현

**주요 컴포넌트**:
- 데이터 로딩 및 통합 함수
- 전처리 및 특성 엔지니어링
- LSTM 모델 정의
- 학습 및 예측 로직
- 시각화 유틸리티

**함수 구성**:
```python
# 데이터 처리
- load_and_explore_data()
- load_single_csv()
- integrate_sensor_data()
- create_unified_dataset()
- prepare_features()

# 시계열 데이터 생성
- create_time_series_data()
- create_multi_output_data()

# 모델 정의
- create_lstm_model()
- create_multi_output_lstm()

# 학습
- train_lstm_model()
- train_multi_output_model()

# 예측 및 평가
- predict_future_anomalies()
- calculate_remaining_useful_life()
- generate_alerts()

# 실시간 모니터링
- real_time_monitoring()
- create_mock_real_time_stream()

# 시나리오 실행
- run_single_output_scenario()
- run_multi_output_scenario()
- run_real_time_monitoring_scenario()

# 시각화
- visualize_predictions()
- plot_rul_distribution()
```

### `precursor.py` (Interface Module)

**역할**: 외부 인터페이스 제공

**주요 함수**:

#### 1. `main()`
- **목적**: 독립 실행 모드
- **데이터 소스**: 파일 시스템 (CSV 파일들)
- **경로**: `../../test-scenarios/test_data/`
- **사용 시나리오**: 개발, 테스트, 배치 처리

```python
def main():
    """
    파일 시스템에서 데이터를 로드하여 전체 파이프라인 실행

    Returns:
        dict: {
            'summary': {
                'predicted_value': float,
                'is_anomaly': str ('0', '1', '2')
            }
        }
    """
```

#### 2. `precursor(datasets)`
- **목적**: 프로그래밍 API
- **데이터 소스**: 인자로 전달받은 datasets (dict)
- **사용 시나리오**: 다른 모듈에서 호출, 통합 시스템

```python
def precursor(datasets):
    """
    외부에서 전달받은 데이터로 예측 수행

    Args:
        datasets (dict): {filename: DataFrame} 형태의 데이터셋

    Returns:
        dict: {
            'summary': {
                'predicted_value': float,
                'is_anomaly': str
            },
            'error': str (optional)
        }
    """
```

---

## 주요 기능

### 1️⃣ 데이터 로딩 및 통합

**다중 파일 처리**:
```python
# 디렉토리 구조 예시
test_data/
├── automotive/
│   ├── automotive_welding_001.csv
│   └── automotive_assembly_004.csv
├── battery/
│   └── battery_formation_001.csv
└── ...
```

**통합 프로세스**:
1. 모든 CSV 파일 로드
2. 센서 데이터를 Long Format으로 변환 (melt)
3. TIMESTAMP 기준으로 Pivot하여 Wide Format 생성
4. 결측치 처리 (forward fill → backward fill → 0)

**지원 ID 컬럼**:
- Primary: `SENSOR_ID`, `CHAMBER_ID`, `EQUIPMENT_ID`, `CELL_ID`, `LINE_ID`, `PRODUCTION_LINE`
- Fallback: `*_ID` 패턴 (예: `REACTOR_ID`, `PRESS_ID` 등)

### 2️⃣ 자동 이상 레이블링

**Z-Score 기반 비지도 학습**:

```python
# 각 특성별 Z-Score 계산
z_scores = abs((feature_data - mean) / std).mean(axis=1)

# 상위 10%를 이상으로 분류
threshold = np.percentile(z_scores, 90)
is_anomaly = z_scores > threshold
```

**특징**:
- 레이블 데이터 불필요
- Train 데이터의 통계만 사용 (데이터 누출 방지)
- Val/Test 데이터는 Train 통계로 평가

### 3️⃣ 시계열 데이터 생성

**Sliding Window 방식**:

```
[t-9, t-8, ..., t-1, t] → [t+1 ~ t+5에 이상 발생?]
    ↑                           ↑
 sequence_length=10      prediction_horizon=5
```

**예시**:
```python
# 입력: 과거 10 스텝의 센서값
X = [[센서1_t-9, 센서2_t-9, ...],
     [센서1_t-8, 센서2_t-8, ...],
     ...
     [센서1_t, 센서2_t, ...]]     # shape: (10, num_sensors)

# 출력: 미래 5 스텝 내 이상 발생 여부
y = 1  # 1: 이상 발생, 0: 정상
```

### 4️⃣ LSTM 모델

#### 단일 출력 모델 (LSTMPredictor)

```
Input (batch, seq_len, features)
    ↓
LSTM Layers (hidden_size=64, num_layers=2)
    ↓
Last Time Step Output
    ↓
FC Layer 1 (64 → 32)
    ↓
ReLU + Dropout
    ↓
FC Layer 2 (32 → 1)
    ↓
Sigmoid
    ↓
Anomaly Probability [0-1]
```

**특징**:
- Binary Classification (이상 발생 여부)
- Class Imbalance 처리 (Weighted BCE Loss)
- Learning Rate Scheduling (ReduceLROnPlateau)

#### 다중 출력 모델 (MultiOutputLSTM)

```
Input (batch, seq_len, features)
    ↓
Shared LSTM (hidden_size=128, num_layers=3)
    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
   Value Predictor      Anomaly Predictor
   (센서값 예측)         (이상 확률 예측)
         ↓                     ↓
   (batch, 5, features)   (batch, 5)
```

**특징**:
- 센서값과 이상 여부를 동시에 예측
- Multi-task Learning
- Loss = MSE(values) + BCE(anomalies)

### 5️⃣ 실시간 모니터링

**스트리밍 데이터 처리**:

```python
# 순환 버퍼 방식
buffer = [최신 sequence_length개 데이터]

for new_data in stream:
    buffer.append(new_data)
    if len(buffer) >= sequence_length:
        prediction = model(buffer[-sequence_length:])

        if prediction >= 0.7:
            alert("위험 경고")
        elif prediction >= 0.3:
            alert("주의")
```

**출력 상태 코드**:
- `'0'`: 정상 (확률 < 0.3)
- `'1'`: 주의 (0.3 ≤ 확률 < 0.7)
- `'2'`: 위험 (확률 ≥ 0.7)

### 6️⃣ RUL 예측 (Remaining Useful Life)

**열화 시뮬레이션 기반**:

```python
for horizon in range(1, max_horizon + 1):
    # 시간에 따른 가상 열화 적용
    degradation_factor = 1 + (horizon * 0.015)
    adjusted_prob = min(prob * degradation_factor, 1.0)

    if adjusted_prob >= failure_threshold:
        return horizon  # RUL
```

**활용**:
- 예방 정비 일정 계획
- 부품 교체 시기 예측
- 생산 중단 최소화

---

## 데이터 흐름

```
┌─────────────────────┐
│  CSV 파일들          │
│  (multiple files)   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ load_and_explore_   │
│ data()              │  ← 1. 데이터 로딩
└──────────┬──────────┘
           ↓
    ┌────────────┐
    │  datasets  │  (dict)
    └──────┬─────┘
           ↓
┌─────────────────────┐
│ integrate_sensor_   │
│ data()              │  ← 2. Long Format 변환
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ create_unified_     │
│ dataset()           │  ← 3. Pivot → Wide Format
└──────────┬──────────┘
           ↓
    ┌────────────┐
    │ unified_df │  (pivot table)
    └──────┬─────┘
           ↓
┌─────────────────────┐
│ train_test_split    │  ← 4. 데이터 분할
│ (시계열 순서 유지)   │     (Train:Val:Test = 81:9:10)
└──────────┬──────────┘
           ↓
    ┌──────────────────────────┐
    │ train_df │ val_df │ test_df
    └────┬─────┴────┬───┴───┬───┘
         ↓          ↓       ↓
┌─────────────────────┐
│ prepare_features()  │  ← 5. 전처리
│  - Z-score 이상 탐지│
│  - StandardScaler   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ create_time_series_ │  ← 6. 시계열 데이터 생성
│ data()              │     (Sliding Window)
└──────────┬──────────┘
           ↓
      ┌────────┐
      │ X, y   │  (sequences, labels)
      └───┬────┘
          ↓
┌─────────────────────┐
│ create_lstm_model() │  ← 7. 모델 생성
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ train_lstm_model()  │  ← 8. 학습
│  - BCELoss          │
│  - Adam Optimizer   │
│  - LR Scheduling    │
└──────────┬──────────┘
           ↓
    ┌──────────────┐
    │ trained_model│
    └──────┬───────┘
           ↓
┌─────────────────────────────┐
│ predict_future_anomalies()  │  ← 9. 예측
└──────────┬──────────────────┘
           ↓
    ┌──────────────┐
    │ anomaly_probs│  (probabilities)
    └──────┬───────┘
           ↓
┌─────────────────────┐
│ generate_alerts()   │  ← 10. 경고 생성
└──────────┬──────────┘
           ↓
      ┌────────┐
      │ alerts │  (list of alerts)
      └────────┘
```

---

## 함수 레퍼런스

### 데이터 처리 함수

#### `load_and_explore_data(data_base_path)`

**목적**: 디렉토리에서 모든 CSV 파일 로드

**인자**:
- `data_base_path` (str): 데이터 폴더 경로

**반환**:
- `dict`: `{filename: DataFrame}` 형태

**동작**:
```python
# 파일 직접 로드
if file.endswith('.csv'):
    datasets[filename] = pd.read_csv(file)

# 하위 디렉토리 탐색
for subdir in os.listdir(data_base_path):
    for file in os.listdir(subdir):
        if file.endswith('.csv'):
            datasets[filename] = pd.read_csv(file)
```

**예시**:
```python
datasets = load_and_explore_data('/path/to/test_data/')
# Returns:
# {
#   'automotive_welding_001': DataFrame(...),
#   'battery_formation_001': DataFrame(...),
#   ...
# }
```

---

#### `integrate_sensor_data(datasets)`

**목적**: 다중 데이터셋을 Long Format으로 통합

**인자**:
- `datasets` (dict): `{filename: DataFrame}` 형태

**반환**:
- `DataFrame`: Long format 통합 데이터
  - Columns: `['TIMESTAMP', equipment_col, 'sensor_table', 'sensor_type', 'sensor_value']`

**처리 과정**:
```python
# Wide Format (원본)
TIMESTAMP | SENSOR_ID | TEMP | PRESSURE | VOLTAGE
2025-01   | S001      | 25.0 | 1.2      | 220

# Long Format (변환 후)
TIMESTAMP | SENSOR_ID | sensor_table | sensor_type | sensor_value
2025-01   | S001      | file1        | TEMP        | 25.0
2025-01   | S001      | file1        | PRESSURE    | 1.2
2025-01   | S001      | file1        | VOLTAGE     | 220
```

---

#### `create_unified_dataset(datasets)`

**목적**: Long Format 데이터를 Pivot하여 Wide Format으로 변환

**인자**:
- `datasets` (dict): `{filename: DataFrame}`

**반환**:
- `DataFrame`: Wide format 통합 데이터
  - Index: `TIMESTAMP`
  - Columns: `sensor_TEMP`, `sensor_PRESSURE`, ... + `equipment_id` (optional)

**처리 과정**:
```python
# 1. integrate_sensor_data() 호출
# 2. pivot_table 생성
sensor_pivot = integrated.pivot_table(
    index='TIMESTAMP',
    columns='sensor_type',
    values='sensor_value',
    aggfunc='mean'
)

# 3. 결측치 처리
sensor_pivot = sensor_pivot.ffill().bfill().fillna(0)

# 4. equipment_id 병합 (있는 경우)
```

---

#### `prepare_features(df, train_stats=None)`

**목적**: 특성 전처리 및 이상 레이블링

**인자**:
- `df` (DataFrame): 입력 데이터
- `train_stats` (dict, optional): Train 데이터의 통계량
  - `None`: 새로 계산 (Train 데이터)
  - `dict`: 기존 통계 사용 (Val/Test 데이터)

**반환**:
- `df_processed` (DataFrame): 전처리된 데이터 + `is_anomaly` 컬럼
- `feature_cols` (list): 특성 컬럼 이름 리스트
- `scaler` (StandardScaler): 학습된 스케일러
- `train_stats` (dict): 통계량 `{'mean': ..., 'std': ...}`

**처리 과정**:
```python
# 1. 수치형 컬럼 추출
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 2. 이상 레이블링 (Z-score)
z_scores = abs((data - mean) / std).mean(axis=1)
threshold = np.percentile(z_scores, 90)
df['is_anomaly'] = z_scores > threshold

# 3. 정규화 (StandardScaler)
df[feature_cols] = scaler.fit_transform(df[feature_cols])
```

**주의사항**:
- Train 데이터는 `train_stats=None`으로 호출
- Val/Test 데이터는 Train의 `train_stats`를 전달하여 **데이터 누출 방지**

---

### 시계열 데이터 생성

#### `create_time_series_data(data, feature_cols, sequence_length=10, prediction_horizon=5)`

**목적**: Sliding Window 방식으로 시계열 시퀀스 생성

**인자**:
- `data` (DataFrame): 전처리된 데이터 (with `is_anomaly`)
- `feature_cols` (list): 특성 컬럼 리스트
- `sequence_length` (int): 입력 시퀀스 길이 (기본: 10)
- `prediction_horizon` (int): 예측 구간 (기본: 5)

**반환**:
- `X` (np.ndarray): 입력 시퀀스, shape: `(num_samples, sequence_length, num_features)`
- `y` (np.ndarray): 타겟 레이블, shape: `(num_samples,)`
  - `1`: 미래 구간에 이상 발생
  - `0`: 정상

**동작**:
```python
for i in range(sequence_length, len(data) - prediction_horizon):
    # 입력: 과거 데이터
    X.append(data[i-sequence_length : i])

    # 출력: 미래 구간 내 이상 발생 여부
    future_window = data['is_anomaly'][i+1 : i+1+prediction_horizon]
    y.append(1 if future_window.any() else 0)
```

**예시**:
```python
X, y = create_time_series_data(
    data=train_df,
    feature_cols=['sensor_TEMP', 'sensor_PRESSURE'],
    sequence_length=10,
    prediction_horizon=5
)
# X.shape: (4900, 10, 2)
# y.shape: (4900,)
```

---

#### `create_multi_output_data(data, feature_cols, sequence_length=10, prediction_steps=5)`

**목적**: 다중 출력 모델용 데이터 생성 (센서값 + 이상 여부)

**반환**:
- `X`: 입력 시퀀스
- `y_values`: 미래 센서값 (shape: `(N, prediction_steps, num_features)`)
- `y_anomalies`: 미래 이상 여부 (shape: `(N, prediction_steps)`)

---

### 모델 정의

#### `create_lstm_model(input_size, hidden_size=64, num_layers=2, dropout=0.2)`

**목적**: 단일 출력 LSTM 모델 생성

**인자**:
- `input_size` (int): 입력 특성 개수
- `hidden_size` (int): LSTM hidden state 크기
- `num_layers` (int): LSTM 레이어 수
- `dropout` (float): Dropout 비율

**반환**:
- `LSTMPredictor`: PyTorch 모델

**모델 구조**:
```python
class LSTMPredictor(nn.Module):
    - LSTM(input_size → hidden_size, num_layers)
    - Linear(hidden_size → 32)
    - ReLU + Dropout
    - Linear(32 → 1)
    - Sigmoid
```

---

#### `create_multi_output_lstm(input_size, hidden_size=128, num_layers=3, prediction_steps=5, num_features=None)`

**목적**: 다중 출력 LSTM 모델 생성

**반환**:
- `MultiOutputLSTM`: PyTorch 모델

**모델 구조**:
```python
class MultiOutputLSTM(nn.Module):
    - Shared LSTM
    - value_predictor: 센서값 예측 헤드
    - anomaly_predictor: 이상 확률 예측 헤드
```

---

### 학습 함수

#### `train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.001)`

**목적**: LSTM 모델 학습

**인자**:
- `model`: PyTorch 모델
- `X_train`, `y_train`: 학습 데이터
- `X_val`, `y_val`: 검증 데이터
- `epochs` (int): 에포크 수
- `batch_size` (int): 배치 크기
- `learning_rate` (float): 학습률

**반환**:
- `trained_model`: 학습된 모델
- `train_losses` (list): 에포크별 학습 손실
- `val_losses` (list): 에포크별 검증 손실

**주요 특징**:
```python
# Class Imbalance 처리
pos_weight = num_neg / num_pos
loss = F.binary_cross_entropy(pred, target, weight=batch_weights)

# Learning Rate Scheduling
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

**출력 예시**:
```
클래스 불균형 처리: pos_weight=9.23 (양성: 500, 음성: 4500)
모델 학습 시작...
Epoch [10/50] - Train Loss: 0.3421, Val Loss: 0.3298
Epoch [20/50] - Train Loss: 0.2156, Val Loss: 0.2089
...
```

---

### 예측 함수

#### `predict_future_anomalies(model, X_test, threshold=0.5)`

**목적**: 이상 확률 예측

**인자**:
- `model`: 학습된 PyTorch 모델
- `X_test`: 테스트 데이터
- `threshold` (float): 이상 판정 임계값

**반환**:
- `anomaly_probs` (np.ndarray): 이상 확률 [0-1]
- `anomaly_labels` (np.ndarray): 이진 레이블 {0, 1}

**예시**:
```python
probs, labels = predict_future_anomalies(model, X_test, threshold=0.5)
# probs: [0.23, 0.78, 0.45, ...]
# labels: [0, 1, 0, ...]
```

---

#### `generate_alerts(anomaly_probs, lot_numbers=None, alert_threshold=0.7, warning_threshold=0.5)`

**목적**: 경고 메시지 생성

**인자**:
- `anomaly_probs` (array): 이상 확률
- `lot_numbers` (list, optional): 샘플 식별자 (TIMESTAMP 또는 equipment_id)
- `alert_threshold` (float): 위험 경고 임계값 (기본: 0.7)
- `warning_threshold` (float): 주의 경고 임계값 (기본: 0.5)

**반환**:
- `alerts` (list of dict): 경고 목록

**경고 레벨**:
- `CRITICAL`: 확률 ≥ 0.7
- `WARNING`: 0.5 ≤ 확률 < 0.7
- (무시): 확률 < 0.5

**출력 형식**:
```python
[
    {
        'sample_id': 'sample_0042',
        'alert_level': 'CRITICAL',
        'probability': 0.85,
        'message': '위험! : Sample sample_0042 - 이상 발생 확률 85.0%',
        'action': '즉시 점검 필요',
        'timestamp': datetime.now()
    },
    ...
]
```

---

#### `calculate_remaining_useful_life(model, current_data_seq, max_horizon=100, failure_threshold=0.8)`

**목적**: 잔여 유효 수명(RUL) 예측

**인자**:
- `model`: 학습된 모델
- `current_data_seq`: 현재 시퀀스 데이터 (shape: `(sequence_length, num_features)`)
- `max_horizon` (int): 최대 예측 범위
- `failure_threshold` (float): 고장 판정 확률 (기본: 0.8)

**반환**:
- `rul` (int): 예상 잔여 수명 (스텝 수)

**동작**:
```python
for horizon in range(1, max_horizon + 1):
    prob = model(current_data)
    degradation_factor = 1 + (horizon * 0.015)  # 시간에 따른 열화
    adjusted_prob = prob * degradation_factor

    if adjusted_prob >= failure_threshold:
        return horizon  # RUL
```

---

### 실시간 모니터링

#### `real_time_monitoring(model, scaler, feature_cols, new_data_stream, sequence_length=10, update_interval=1)`

**목적**: 실시간 데이터 스트림 모니터링

**인자**:
- `model`: 학습된 모델
- `scaler`: StandardScaler 객체
- `feature_cols`: 특성 컬럼 리스트
- `new_data_stream`: Iterator (yields `(timestamp, data_row)`)
- `sequence_length`: 시퀀스 길이
- `update_interval`: 업데이트 주기 (초)

**반환**:
- `max_status` (str): 전체 모니터링 기간 중 최대 위험 레벨
  - `'0'`: 정상
  - `'1'`: 주의
  - `'2'`: 위험

**동작**:
```python
buffer = []  # 순환 버퍼

for timestamp, new_data in stream:
    buffer.append(preprocess(new_data))

    if len(buffer) >= sequence_length:
        prob = model(buffer[-sequence_length:])

        if prob >= 0.7:
            print(f"[{timestamp}] 위험 경고: 이상 확률 {prob:.1%}")
            max_status = '2'
        elif prob >= 0.3:
            print(f"[{timestamp}] 주의: 이상 징후 감지")
            if max_status < '1':
                max_status = '1'
```

---

### 시나리오 함수

#### `run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler)`

**목적**: 시나리오 1 - 단일 출력 이상 징후 예측 전체 파이프라인

**처리 과정**:
```
1. 시계열 데이터 생성 (sequence_length=2, prediction_horizon=1)
2. LSTM 모델 생성
3. 모델 학습 (10 epochs)
4. 테스트 데이터 예측
5. 경고 생성
6. RUL 예측 (첫 번째 샘플)
```

**반환**:
- `trained_model`: 학습된 모델
- `scaler`: 스케일러

---

#### `run_multi_output_scenario(train_df, val_df, test_df, feature_cols)`

**목적**: 시나리오 2 - 다중 출력 모델 (센서값 + 이상 징후)

**반환**:
- `pred_value` (np.ndarray): 예측된 미래 이상 확률 (첫 번째 샘플)

---

#### `run_real_time_monitoring_scenario(trained_model, scaler, feature_cols, test_df)`

**목적**: 시나리오 3 - 실시간 모니터링 시뮬레이션

**반환**:
- `anomaly_status` (str): 최종 상태 코드 (`'0'`, `'1'`, `'2'`)

---

## 사용 방법

### 방법 1: 독립 실행 (main 함수)

**시나리오**: CSV 파일들로부터 데이터를 로드하여 전체 파이프라인 실행

```bash
cd prism_monitor/modules/event_precursor
python precursor.py
```

**코드**:
```python
# precursor.py
if __name__ == "__main__":
    main()
```

**데이터 경로 설정**:
```python
# precursor.py의 main() 함수 내부
DATA_BASE_PATH = '../../test-scenarios/test_data/'
```

**출력 예시**:
```
============================================================
이상 징후 예측 모듈 시작
============================================================

[1/6] 데이터 로딩...
Loading: automotive/automotive_welding_001.csv
  - Shape: (5000, 11)
...
총 로드된 파일 수: 20

[2/6] 데이터 통합...
센서 데이터 통합...
  - automotive_welding_001: Sensor count: 9, Record count: 5000
...
통합 데이터셋 생성 완료: (100000, 181)

[3/6] 데이터 분할 (Train/Val/Test)...

[4/6] 특성 전처리 및 이상 레이블링...
  - Train 데이터로 통계 계산
전처리 완료: 180개 특성
이상 데이터 비율: 10.00%
  - Val/Test 데이터는 Train 통계 사용 (데이터 누출 방지)

데이터 분할 완료:
  - 학습 데이터: (72900, 182)
  - 검증 데이터: (8100, 182)
  - 테스트 데이터: (10000, 182)
  - 특성 개수: 180

[5/6] 모델 학습 및 예측...

>> 시나리오 1: 단일 출력 이상 징후 예측
1. 단일 출력 이상 징후 예측 모델
시계열 데이터 생성 완료: X shape=(72898, 2, 180), y shape=(72898,)
클래스 불균형 처리: pos_weight=9.12
모델 학습 시작...
Epoch [10/10] - Train Loss: 0.2345, Val Loss: 0.2198

테스트 데이터 예측 및 경고 생성
총 15개의 경고가 생성되었습니다.
  - 위험! : Sample 2025-05-01T12:34:00Z - 이상 발생 확률 78.5%
  - 경고: Sample 2025-05-01T14:22:00Z - 이상 징후 감지 (확률 62.3%)
  ...

RUL 예측 예시 (테스트 데이터 첫 번째 샘플)
첫 번째 테스트 샘플의 예측 RUL: 42 스텝

>> 시나리오 2: 다중 출력 센서값 및 이상 징후 동시 예측
...

[6/6] 실시간 모니터링...
실시간 모니터링 시작...
[2025-10-24T10:00:00] 안전: 이상 징후 발생 가능성 낮음 (확률 12.3%)
[2025-10-24T10:00:10] 주의: 이상 징후 감지 (확률 45.6%)
...

모니터링 완료: 최대 이상 확률 78.5%, 상태: 2

============================================================
이상 징후 예측 완료
============================================================
```

---

### 방법 2: 프로그래밍 API (precursor 함수)

**시나리오**: 다른 모듈에서 호출하여 사용

```python
from prism_monitor.modules.event_precursor.precursor import precursor

# 데이터 준비 (dict of DataFrames)
datasets = {
    'welding_data': pd.read_csv('welding.csv'),
    'battery_data': pd.read_csv('battery.csv'),
    ...
}

# Precursor 실행
result = precursor(datasets)

# 결과 확인
print(result)
# {
#     'summary': {
#         'predicted_value': 0.456,
#         'is_anomaly': '1'
#     }
# }
```

---

### 방법 3: 커스텀 파이프라인

**시나리오**: 세부 제어가 필요한 경우

```python
from prism_monitor.modules.event_precursor._precursor import (
    load_and_explore_data,
    create_unified_dataset,
    prepare_features,
    create_time_series_data,
    create_lstm_model,
    train_lstm_model,
    predict_future_anomalies,
    generate_alerts
)
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
datasets = load_and_explore_data('/path/to/data')

# 2. 통합
unified_df = create_unified_dataset(datasets)

# 3. 분할
train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

# 4. 전처리
train_df, feature_cols, scaler, train_stats = prepare_features(train_df)
val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

# 5. 시계열 데이터 생성
X_train, y_train = create_time_series_data(train_df, feature_cols,
                                           sequence_length=10,
                                           prediction_horizon=5)
X_val, y_val = create_time_series_data(val_df, feature_cols, 10, 5)
X_test, y_test = create_time_series_data(test_df, feature_cols, 10, 5)

# 6. 모델 생성 및 학습
model = create_lstm_model(input_size=X_train.shape[2],
                         hidden_size=128,
                         num_layers=3)

trained_model, train_losses, val_losses = train_lstm_model(
    model, X_train, y_train, X_val, y_val,
    epochs=100, batch_size=64, learning_rate=0.001
)

# 7. 예측
probs, labels = predict_future_anomalies(trained_model, X_test, threshold=0.5)

# 8. 경고 생성
timestamps = test_df['TIMESTAMP'].iloc[10:len(probs)+10].tolist()
alerts = generate_alerts(probs, lot_numbers=timestamps,
                        alert_threshold=0.8,
                        warning_threshold=0.6)

# 9. 결과 출력
for alert in alerts:
    print(f"[{alert['alert_level']}] {alert['message']}")
```

---

## 입출력 형식

### 입력 데이터 형식

#### CSV 파일 요구사항

**필수 컬럼**:
- `TIMESTAMP`: ISO 8601 형식 (예: `2025-05-01T00:00:00Z`)

**선택 컬럼** (하나 이상 권장):
- Equipment ID: `SENSOR_ID`, `CHAMBER_ID`, `LINE_ID`, `PRODUCTION_LINE` 등

**센서 데이터**:
- 수치형 컬럼 (float/int)
- 예: `TEMPERATURE`, `PRESSURE`, `VOLTAGE`, `CURRENT` 등

**예시 CSV**:
```csv
TIMESTAMP,SENSOR_ID,TEMPERATURE,PRESSURE,VOLTAGE,CURRENT
2025-05-01T00:00:00Z,SENSOR_001,25.3,1.21,220.5,15.2
2025-05-01T00:00:10Z,SENSOR_001,25.5,1.22,220.7,15.3
2025-05-01T00:00:20Z,SENSOR_001,25.8,1.20,221.0,15.1
...
```

#### datasets 딕셔너리 형식

```python
datasets = {
    'automotive_welding_001': pd.DataFrame({
        'TIMESTAMP': [...],
        'LINE_ID': [...],
        'WELD_CURRENT': [...],
        'WELD_VOLTAGE': [...],
        ...
    }),
    'battery_formation_001': pd.DataFrame({
        'TIMESTAMP': [...],
        'CELL_ID': [...],
        'VOLTAGE': [...],
        'CURRENT': [...],
        ...
    }),
    ...
}
```

---

### 출력 데이터 형식

#### `main()` 함수 출력

```python
{
    'summary': {
        'predicted_value': np.ndarray,  # 다중 출력 모델의 예측값
        'is_anomaly': str               # '0', '1', '2'
    }
}
```

#### `precursor(datasets)` 함수 출력

**정상 실행**:
```python
{
    'summary': {
        'predicted_value': 0.456,  # float
        'is_anomaly': '1'          # str: '0' (정상) | '1' (주의) | '2' (위험)
    }
}
```

**에러 발생**:
```python
{
    'summary': {
        'predicted_value': 0.0,
        'is_anomaly': '0'
    },
    'error': '데이터셋 통합 실패'
}
```

#### 경고 리스트 형식

```python
[
    {
        'sample_id': 'sample_0042',
        'alert_level': 'CRITICAL',
        'probability': 0.85,
        'message': '위험! : Sample sample_0042 - 이상 발생 확률 85.0%',
        'action': '즉시 점검 필요',
        'timestamp': datetime(2025, 10, 24, 14, 30, 0)
    },
    {
        'sample_id': 'sample_0067',
        'alert_level': 'WARNING',
        'probability': 0.62,
        'message': '경고: Sample sample_0067 - 이상 징후 감지 (확률 62.0%)',
        'action': '예방 점검 권장',
        'timestamp': datetime(2025, 10, 24, 14, 31, 0)
    }
]
```

---

## 예제

### 예제 1: 기본 사용 (main 함수)

```python
# precursor.py를 직접 실행
python precursor.py
```

---

### 예제 2: 외부 모듈에서 호출

```python
from prism_monitor.modules.event_precursor.precursor import precursor
import pandas as pd

# 데이터 준비
datasets = {
    'welding': pd.read_csv('welding.csv'),
    'battery': pd.read_csv('battery.csv')
}

# 실행
result = precursor(datasets)

# 결과 처리
if 'error' in result:
    print(f"에러 발생: {result['error']}")
else:
    predicted_value = result['summary']['predicted_value']
    status = result['summary']['is_anomaly']

    if status == '2':
        print(f"⚠️ 위험: 이상 확률 {predicted_value:.2%}")
        # 알림 전송, 점검 요청 등
    elif status == '1':
        print(f"⚡ 주의: 이상 징후 감지 ({predicted_value:.2%})")
        # 모니터링 강화
    else:
        print("✅ 정상")
```

---

### 예제 3: 커스텀 임계값 설정

```python
from prism_monitor.modules.event_precursor._precursor import (
    load_and_explore_data,
    create_unified_dataset,
    prepare_features,
    create_time_series_data,
    create_lstm_model,
    train_lstm_model,
    predict_future_anomalies,
    generate_alerts
)
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
datasets = load_and_explore_data('/path/to/data')
unified_df = create_unified_dataset(datasets)

# ... (중간 과정 생략)

# 예측
probs, _ = predict_future_anomalies(model, X_test)

# 커스텀 임계값으로 경고 생성
alerts = generate_alerts(
    probs,
    lot_numbers=timestamps,
    alert_threshold=0.85,    # 위험: 85% 이상
    warning_threshold=0.60   # 주의: 60% 이상
)

# 위험 경고만 필터링
critical_alerts = [a for a in alerts if a['alert_level'] == 'CRITICAL']
print(f"위험 경고: {len(critical_alerts)}건")
```

---

### 예제 4: RUL 예측 활용

```python
from prism_monitor.modules.event_precursor._precursor import (
    calculate_remaining_useful_life
)

# 현재 설비 상태 (최근 10 스텝)
current_sequence = X_test[0]  # shape: (10, num_features)

# RUL 예측
rul = calculate_remaining_useful_life(
    model,
    current_sequence,
    max_horizon=200,       # 최대 200 스텝
    failure_threshold=0.85  # 85% 이상을 고장으로 판단
)

print(f"예상 잔여 수명: {rul} 스텝")

# 예방 정비 계획
if rul < 50:
    print("⚠️ 긴급: 즉시 점검 필요")
elif rul < 100:
    print("⚡ 주의: 예방 정비 일정 수립")
else:
    print("✅ 정상: 정기 점검 유지")
```

---

### 예제 5: 실시간 모니터링 시뮬레이션

```python
from prism_monitor.modules.event_precursor._precursor import (
    real_time_monitoring,
    create_mock_real_time_stream
)

# Mock 데이터 스트림 생성
data_stream = create_mock_real_time_stream(
    test_df,
    feature_cols,
    num_samples=100
)

# 실시간 모니터링
status = real_time_monitoring(
    model=trained_model,
    scaler=scaler,
    feature_cols=feature_cols,
    new_data_stream=data_stream,
    sequence_length=10,
    update_interval=1
)

print(f"최종 상태: {status}")
# '0': 정상
# '1': 주의 발생
# '2': 위험 발생
```

---

## 파라미터 설정

### 시계열 파라미터

| 파라미터 | 기본값 | 설명 | 권장 범위 |
|---------|-------|------|----------|
| `sequence_length` | 10 | 입력 시퀀스 길이 (과거 데이터) | 5-50 |
| `prediction_horizon` | 5 | 예측 구간 길이 (미래 스텝) | 1-20 |
| `prediction_steps` | 5 | 다중 출력 모델의 예측 스텝 수 | 1-20 |

**선택 가이드**:
- **짧은 주기 공정** (초 단위): `sequence_length=5`, `prediction_horizon=3`
- **일반 공정** (분 단위): `sequence_length=10`, `prediction_horizon=5`
- **느린 공정** (시간 단위): `sequence_length=20`, `prediction_horizon=10`

---

### 모델 하이퍼파라미터

#### 단일 출력 모델

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `input_size` | auto | 특성 개수 (자동 계산) |
| `hidden_size` | 64 | LSTM hidden state 크기 |
| `num_layers` | 2 | LSTM 레이어 수 |
| `dropout` | 0.2 | Dropout 비율 |

#### 다중 출력 모델

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `hidden_size` | 128 | LSTM hidden state 크기 (더 큼) |
| `num_layers` | 3 | LSTM 레이어 수 (더 깊음) |

---

### 학습 파라미터

| 파라미터 | 기본값 | 설명 | 권장 값 |
|---------|-------|------|---------|
| `epochs` | 50 (단일) / 20 (다중) | 학습 에포크 수 | 10-100 |
| `batch_size` | 32 | 배치 크기 | 16-128 |
| `learning_rate` | 0.001 | 초기 학습률 | 0.0001-0.01 |

**튜닝 팁**:
- 데이터가 많으면: `batch_size` 증가 (64, 128)
- 과적합 발생 시: `dropout` 증가 (0.3, 0.4), `epochs` 감소
- 학습 느릴 때: `learning_rate` 증가 (0.005, 0.01)

---

### 임계값 파라미터

#### 이상 레이블링

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `percentile` | 90 | Z-score 기준 상위 N% |

**조정**:
- 더 엄격하게: `percentile=95` (상위 5%만 이상)
- 더 느슨하게: `percentile=85` (상위 15% 이상)

#### 경고 생성

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `alert_threshold` | 0.7 | CRITICAL 경고 임계값 |
| `warning_threshold` | 0.5 | WARNING 경고 임계값 |

**산업별 추천**:
- **안전 중요 (반도체, 화학)**: `alert=0.6`, `warning=0.4`
- **일반 제조**: `alert=0.7`, `warning=0.5`
- **비용 민감**: `alert=0.8`, `warning=0.6`

#### 실시간 모니터링

| 상태 | 기본 임계값 | 조정 가능 |
|-----|-----------|----------|
| 정상 (`'0'`) | < 0.3 | < `monitoring_warning` |
| 주의 (`'1'`) | 0.3 - 0.7 | `monitoring_warning` - `monitoring_critical` |
| 위험 (`'2'`) | ≥ 0.7 | ≥ `monitoring_critical` |

**코드에서 조정**:
```python
# real_time_monitoring 함수 내부 (608-646줄)
if anomaly_prob >= 0.7:  # monitoring_critical
    ...
elif anomaly_prob >= 0.3:  # monitoring_warning
    ...
```

---

### 데이터 분할 비율

**기본 설정**:
```python
train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)
```

**비율**:
- Train: 81% (0.9 × 0.9)
- Validation: 9% (0.9 × 0.1)
- Test: 10%

**주의**: `shuffle=False` 필수 (시계열 순서 유지)

---

## 참고 사항

### 성능 최적화

1. **GPU 사용**:
```python
# 자동으로 GPU 감지 및 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2. **배치 크기 조정**:
- GPU 메모리 충분: `batch_size=128`
- GPU 메모리 부족: `batch_size=32`
- CPU 전용: `batch_size=16`

3. **데이터 크기**:
- 메모리 부족 시: 데이터 샘플링 또는 분할 처리

---

### 주의사항

1. **데이터 누출 방지**:
   - Train 데이터로만 통계 계산
   - Val/Test는 Train 통계 사용

2. **시계열 순서 유지**:
   - `shuffle=False` 필수
   - 시간순 정렬 확인

3. **클래스 불균형**:
   - Weighted Loss 자동 적용
   - 필요 시 `percentile` 조정

4. **ID 컬럼 인식**:
   - 표준 ID 컬럼 사용 권장
   - 비표준 컬럼은 `possible_id_cols`에 추가

---

### 제한사항

1. **메모리 사용량**:
   - 대용량 데이터 (> 1GB): 분할 처리 필요

2. **학습 시간**:
   - 데이터 크기, 모델 복잡도에 비례
   - GPU 사용 강력 권장

3. **RUL 예측 정확도**:
   - 시뮬레이션 기반 (degradation_factor)
   - 실제 물리 모델 적용 시 정확도 향상 가능

---

### 확장 가능성

1. **다른 모델 적용**:
   - Transformer, GRU 등으로 교체 가능

2. **앙상블 모델**:
   - 여러 모델의 예측 결합

3. **온라인 학습**:
   - 실시간 데이터로 모델 업데이트

4. **설명 가능성**:
   - SHAP, LIME 등 적용 가능

---

## 문의 및 기여

### 버그 리포트
이슈 발견 시 다음 정보와 함께 보고:
- 데이터 형식 및 크기
- 에러 메시지 전문
- 실행 환경 (Python 버전, PyTorch 버전)

### 개선 제안
- 새로운 기능 요청
- 성능 개선 아이디어
- 문서 개선

---

**작성일**: 2025-10-24
**버전**: 1.0.0
**작성자**: PRISM-Monitor Development Team
