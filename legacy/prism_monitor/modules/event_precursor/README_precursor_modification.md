# Event Precursor Module - 수정 내역

## 📋 개요

이 문서는 Event Precursor 모듈의 주요 수정 사항을 정리한 문서입니다.
LSTM 기반 이상 징후 예측 모듈의 버그 수정 및 기능 개선 사항을 설명합니다.

**수정 일자**: 2025-10-24
**수정자**: Jonghak Jang

### 수정된 파일
- ✅ **_precursor.py**: 핵심 로직 (5가지 버그 수정)
- ✅ **precursor.py**: 메인 인터페이스 (사용자 경험 및 안정성 개선)

---

## 🎯 수정 목적

1. **데이터 호환성 개선**: 다양한 산업별 데이터셋 지원 (CELL_ID, LINE_ID 등)
2. **Critical 버그 수정**: 시계열 인덱싱 오류, 실시간 모니터링 조기 종료
3. **성능 개선**: 클래스 불균형 처리 및 데이터 누출 방지
4. **사용자 경험 개선**: 진행 상황 표시, 에러 처리 강화
5. **코드 품질**: Production-ready 코드로 개선

---

## 🔧 주요 수정 사항

총 **6가지** 수정 항목 (_precursor.py: 5개, precursor.py: 1개)

### 1. 장비 ID 컬럼 식별 로직 개선

**파일**: `_precursor.py`
**함수**: `integrate_sensor_data()`, `create_unified_dataset()`

#### 문제점
기존 코드는 `SENSOR_ID`, `CHAMBER_ID`, `EQUIPMENT_ID`만 지원하여 다음 데이터 타입을 처리하지 못함:
- Battery 데이터: `CELL_ID`
- Automotive 데이터: `LINE_ID`

#### 수정 전
```python
equipment_col = None
if 'SENSOR_ID' in df_copy.columns:
    equipment_col = 'SENSOR_ID'
elif 'CHAMBER_ID' in df_copy.columns:
    equipment_col = 'CHAMBER_ID'
elif 'EQUIPMENT_ID' in df_copy.columns:
    equipment_col = 'EQUIPMENT_ID'
```

#### 수정 후
```python
# 포괄적인 ID 컬럼 리스트
equipment_col = None
possible_id_cols = ['SENSOR_ID', 'CHAMBER_ID', 'EQUIPMENT_ID', 'CELL_ID', 'LINE_ID']
for col in possible_id_cols:
    if col in df_copy.columns:
        equipment_col = col
        break

# 패턴 매칭: _ID로 끝나는 컬럼 자동 탐지
if equipment_col is None:
    id_cols = [col for col in df_copy.columns if col.endswith('_ID')]
    if id_cols:
        equipment_col = id_cols[0]
```

#### 효과
✅ 모든 산업 카테고리 데이터 지원
✅ 확장성 향상 (새로운 ID 컬럼 자동 인식)

---

### 2. 시계열 레이블 인덱싱 오류 수정 (🔴 Critical)

**파일**: `_precursor.py`
**함수**: `create_time_series_data()`
**라인**: 263

#### 문제점
시계열 입력과 레이블의 인덱스 매핑이 잘못되어 학습 데이터가 손상됨

#### 수정 전
```python
for i in range(sequence_length, len(feature_data) - prediction_horizon):
    X.append(feature_data[i-sequence_length:i])
    y.append(future_anomalies[i-sequence_length])  # ❌ 잘못된 인덱싱
```

**문제**: `future_anomalies[i-sequence_length]`는 잘못된 미래 시점을 참조

#### 수정 후
```python
for i in range(sequence_length, len(feature_data) - prediction_horizon):
    X.append(feature_data[i-sequence_length:i])
    y.append(future_anomalies[i])  # ✅ 올바른 인덱싱
```

#### 효과
✅ 입력-출력 매핑이 정확해져 모델 학습 품질 향상
✅ 예측 정확도 개선

---

### 3. 실시간 모니터링 조기 종료 문제 수정 (🔴 Critical)

**파일**: `_precursor.py`
**함수**: `real_time_monitoring()`
**라인**: 663-712

#### 문제점
첫 번째 예측 후 즉시 `return`하여 실시간 모니터링이 작동하지 않음

#### 수정 전
```python
for timestamp, new_data in new_data_stream:
    # ... 예측 수행 ...

    if anomaly_prob >= 0.7:
        print(f"[{timestamp}] 위험 경고")
        return '2'  # ❌ 즉시 종료!
    elif anomaly_prob >= 0.3:
        return '1'  # ❌ 한 번만 예측
```

**문제**: 전체 데이터 스트림 중 첫 번째 샘플만 처리

#### 수정 후
```python
max_anomaly_prob = 0.0
max_status = '0'

for timestamp, new_data in new_data_stream:
    # ... 예측 수행 ...

    if anomaly_prob >= 0.7:
        print(f"[{timestamp}] 🚨 위험 경고")
        max_status = '2'
        max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)
    elif anomaly_prob >= 0.3:
        print(f"[{timestamp}] ⚠️ 주의")
        if max_status < '1':
            max_status = '1'
        max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)
    else:
        print(f"[{timestamp}] ✅ 안전")
        max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)

# 전체 스트림 처리 후 최고 위험도 반환
print(f"\n모니터링 완료: 최대 이상 확률 {max_anomaly_prob:.1%}, 상태: {max_status}")
return max_status
```

#### 효과
✅ 전체 데이터 스트림을 연속적으로 모니터링
✅ 최고 위험도를 추적하여 반환
✅ 실시간 모니터링 기능 정상 작동

---

### 4. Z-score 미래 정보 누출 방지 (🟡 Warning)

**파일**: `_precursor.py`, `precursor.py`
**함수**: `prepare_features()`
**라인**: 207-260

#### 문제점
전체 데이터(Train + Val + Test)의 통계를 사용하여 Z-score를 계산하면 미래 정보가 과거에 누출됨

#### 수정 전
```python
def prepare_features(df):
    # 전체 데이터의 평균/표준편차 사용
    z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
    # ... 이상 판정 ...
```

**문제**: Test 데이터의 정보가 Train 데이터 전처리에 영향

#### 수정 후
```python
def prepare_features(df, train_stats=None):
    """
    Args:
        df: 데이터프레임
        train_stats: 학습 데이터의 통계 (mean, std). None이면 현재 데이터로 계산
    """
    if train_stats is None:
        # Train 데이터인 경우: 통계 계산
        mean_vals = feature_data.mean()
        std_vals = feature_data.std() + 1e-8
        train_stats = {'mean': mean_vals, 'std': std_vals}
    else:
        # Val/Test 데이터인 경우: Train 통계 사용
        mean_vals = train_stats['mean']
        std_vals = train_stats['std']

    # Z-score 계산
    z_scores = np.abs((feature_data - mean_vals) / std_vals).mean(axis=1)
    # ...

    return df_processed, feature_cols, scaler, train_stats
```

**사용 예시** (`precursor.py`):
```python
# Train 데이터로 통계 계산
train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)

# Val/Test 데이터는 Train 통계 사용
val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)
```

#### 효과
✅ 데이터 누출 방지로 모델 성능 정확히 평가
✅ 실제 운영 환경과 동일한 조건으로 학습
✅ Overfitting 가능성 감소

---

### 5. 클래스 불균형 처리 (🟡 Warning)

**파일**: `_precursor.py`
**함수**: `train_lstm_model()`
**라인**: 391-422

#### 문제점
이상 데이터가 10%로 소수인데, 가중치 없이 학습하면 모델이 정상 데이터만 학습

#### 수정 전
```python
criterion = nn.BCELoss()  # 가중치 없음
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for batch_X, batch_y in train_loader:
    outputs = model(batch_X).squeeze(-1)
    loss = criterion(outputs, batch_y)  # ❌ 불균형 미처리
```

#### 수정 후
```python
# pos_weight 계산
num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
if num_pos > 0:
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    print(f"클래스 불균형 처리: pos_weight={pos_weight.item():.2f}")
else:
    pos_weight = torch.tensor([1.0]).to(device)

# 학습 루프에서 weighted loss 적용
for batch_X, batch_y in train_loader:
    outputs = model(batch_X).squeeze(-1)

    # 클래스 불균형 고려한 가중치
    batch_weights = torch.where(batch_y == 1, pos_weight.squeeze(), torch.tensor(1.0).to(device))
    loss = nn.functional.binary_cross_entropy(outputs, batch_y, weight=batch_weights)
```

**예시**: 정상 900개, 이상 100개인 경우
- `pos_weight = 900 / 100 = 9.0`
- 이상 샘플의 손실에 9배 가중치 적용

#### 효과
✅ 소수 클래스(이상 데이터)에 더 많은 가중치
✅ 이상 탐지 성능 향상
✅ False Negative 감소

---

### 6. precursor.py 인터페이스 개선

**파일**: `precursor.py`
**함수**: `main()`, `precursor()`
**전체 파일 리팩토링**

#### 개요
`precursor.py`는 `_precursor.py`의 함수들을 호출하는 메인 인터페이스입니다. `_precursor.py`의 수정사항(특히 `prepare_features`의 시그니처 변경)과 연동되도록 전면 수정되었습니다.

---

#### 6-1. main() 함수 개선

**목적**: 로컬 데이터로 전체 파이프라인 실행 및 테스트

#### 수정 전
```python
def main():
    DATA_BASE_PATH = '../../data/Industrial_DB_sample/'
    print("데이터 준비 및 전처리")
    all_datasets = load_and_explore_data(DATA_BASE_PATH)
    if not all_datasets:
        print("데이터 로딩 실패.")
        return

    unified_df = create_unified_dataset(all_datasets)
    if unified_df.empty:
        print("통합 데이터셋 생성 실패.")
        return

    processed_df, feature_cols, scaler = prepare_features(unified_df)  # ❌ 구 시그니처
    print("=" * 40, "\n")

    train_val_df, test_df = train_test_split(processed_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # ... 모델 실행 ...
```

**문제점:**
- ❌ `prepare_features` 호출이 데이터 분할 전에 실행 → 데이터 누출
- ❌ 진행 상황을 알 수 없음
- ❌ 에러 처리 부족
- ❌ 반환값 초기화 누락 (`anomaly_status` 미정의 가능)

#### 수정 후
```python
def main():
    """
    메인 실행 함수
    - 로컬 데이터 경로에서 데이터를 로드하여 전체 파이프라인 실행
    """
    DATA_BASE_PATH = '../../data/Industrial_DB_sample/'

    print("=" * 60)
    print("이상 징후 예측 모듈 시작")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/6] 데이터 로딩...")
    all_datasets = load_and_explore_data(DATA_BASE_PATH)
    if not all_datasets:
        print("❌ 데이터 로딩 실패.")
        return None  # ✅ 명시적 None 반환

    # 2. 데이터 통합
    print("\n[2/6] 데이터 통합...")
    unified_df = create_unified_dataset(all_datasets)
    if unified_df.empty:
        print("❌ 통합 데이터셋 생성 실패.")
        return None

    print(f"✅ 통합 데이터셋 생성 완료: {unified_df.shape}")

    # 3. 데이터 분할 (시계열 순서 유지)
    print("\n[3/6] 데이터 분할 (Train/Val/Test)...")
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # 4. 특성 전처리 (미래 정보 누출 방지)
    print("\n[4/6] 특성 전처리 및 이상 레이블링...")
    print("  - Train 데이터로 통계 계산")
    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)

    print("  - Val/Test 데이터는 Train 통계 사용 (데이터 누출 방지)")
    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

    print(f"\n✅ 데이터 분할 완료:")
    print(f"  - 학습 데이터: {train_df.shape}")
    print(f"  - 검증 데이터: {val_df.shape}")
    print(f"  - 테스트 데이터: {test_df.shape}")
    print(f"  - 특성 개수: {len(feature_cols)}")
    print("=" * 60)

    # 5. 모델 학습 및 예측
    print("\n[5/6] 모델 학습 및 예측...")
    # ... (시나리오 1, 2 실행)

    # 6. 실시간 모니터링
    print("\n[6/6] 실시간 모니터링...")
    anomaly_status = '0'  # ✅ 기본값 설정

    if trained_model is not None:
        print("\n>> 시나리오 3: 실시간 모니터링 시뮬레이션")
        anomaly_status = run_real_time_monitoring_scenario(
            trained_model, model_scaler, feature_cols, test_df
        )
    else:
        print("⚠️ 학습된 모델이 없어 실시간 모니터링을 건너뜁니다.")

    # 결과 반환
    print("\n" + "=" * 60)
    print("✅ 이상 징후 예측 완료")
    print("=" * 60)

    return {
        'summary': {
            'predicted_value': pred_value,
            'is_anomaly': anomaly_status
        }
    }
```

**개선 사항:**
- ✅ **6단계 진행 표시**: 사용자가 현재 진행 상황 파악 가능
- ✅ **데이터 분할 후 전처리**: 미래 정보 누출 방지
- ✅ **train_stats 전달**: Val/Test는 Train 통계 사용
- ✅ **명확한 로깅**: 각 단계별 상세 정보 출력
- ✅ **에러 처리 강화**: None 반환으로 실패 명시
- ✅ **기본값 설정**: anomaly_status 초기화로 안전성 향상

---

#### 6-2. precursor() 함수 개선

**목적**: 외부에서 호출 가능한 API 함수

#### 수정 전
```python
def precursor(datasets):
    unified_df = create_unified_dataset(datasets)
    processed_df, feature_cols, scaler = prepare_features(unified_df)  # ❌ 구 시그니처
    train_val_df, test_df = train_test_split(processed_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)
    trained_model, model_scaler = run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler)
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)
    anomaly_status = run_real_time_monitoring_scenario(trained_model, model_scaler, feature_cols, test_df)
    return {
        'summary': {
            'predicted_value': float(pred_value[0]),  # ❌ pred_value가 None일 수 있음
            'is_anomaly': anomaly_status
        }
    }
```

**문제점:**
- ❌ docstring 없음 (파라미터/반환값 설명 부족)
- ❌ 빈 데이터셋 처리 없음
- ❌ `pred_value`가 None인 경우 오류 발생
- ❌ 에러 발생 시 처리 부족

#### 수정 후
```python
def precursor(datasets):
    """
    외부에서 호출되는 이상 징후 예측 함수

    Args:
        datasets: dict - load_and_explore_data()로 로드된 데이터셋 딕셔너리
                  예: {'semiconductor_etch_002': DataFrame, ...}

    Returns:
        dict: 예측 결과
            {
                'summary': {
                    'predicted_value': float - 다중 출력 모델의 예측값,
                    'is_anomaly': str - '0': 정상, '1': 경고, '2': 위험
                }
            }
    """
    print("=" * 60)
    print("Precursor 모듈 실행 (외부 호출)")
    print("=" * 60)

    # 1. 데이터 통합
    print("\n[1/5] 데이터 통합...")
    unified_df = create_unified_dataset(datasets)

    if unified_df.empty:
        print("❌ 통합 데이터셋이 비어있습니다.")
        return {
            'summary': {
                'predicted_value': 0.0,
                'is_anomaly': '0'
            },
            'error': '데이터셋 통합 실패'  # ✅ 에러 정보 포함
        }

    print(f"✅ 통합 데이터셋: {unified_df.shape}")

    # 2. 데이터 분할 (시계열 순서 유지)
    print("\n[2/5] 데이터 분할...")
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # 3. 특성 전처리 (미래 정보 누출 방지)
    print("\n[3/5] 특성 전처리...")
    # Train 데이터로 통계 계산
    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)
    # Val/Test 데이터는 Train 통계 사용
    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

    print(f"✅ 특성 개수: {len(feature_cols)}")

    # 4. 모델 학습 및 예측
    print("\n[4/5] 모델 학습 및 예측...")
    print("  - 단일 출력 모델 학습 중...")
    trained_model, model_scaler = run_single_output_scenario(
        train_df, val_df, test_df, feature_cols, scaler
    )

    print("  - 다중 출력 모델 학습 중...")
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)

    # 5. 실시간 모니터링
    print("\n[5/5] 실시간 모니터링...")
    anomaly_status = '0'  # ✅ 기본값

    if trained_model is not None:
        anomaly_status = run_real_time_monitoring_scenario(
            trained_model, model_scaler, feature_cols, test_df
        )
    else:
        print("⚠️ 모델 학습 실패 - 기본값 반환")

    # 결과 반환
    print("\n" + "=" * 60)
    print(f"✅ Precursor 완료 - 이상 상태: {anomaly_status}")
    print("=" * 60)

    return {
        'summary': {
            'predicted_value': float(pred_value[0]) if pred_value is not None else 0.0,  # ✅ None 체크
            'is_anomaly': anomaly_status
        }
    }
```

**개선 사항:**
- ✅ **상세한 docstring**: 파라미터와 반환값 설명
- ✅ **빈 데이터셋 처리**: 에러 메시지 포함 응답 반환
- ✅ **None 안전성**: `pred_value is not None` 체크
- ✅ **5단계 진행 표시**: 외부 호출 시에도 명확한 진행 상황
- ✅ **기본값 설정**: 모델 학습 실패 시에도 안전한 반환
- ✅ **에러 정보 제공**: `error` 키로 실패 원인 전달

---

#### 6-3. 전체 실행 흐름 개선

**개선된 파이프라인:**
```
외부 호출: precursor(datasets)
    ↓
[1/5] 데이터 통합
    ├─ unified_df = create_unified_dataset()
    └─ 빈 데이터셋 체크 ✅
    ↓
[2/5] 데이터 분할 (shuffle=False)
    ├─ Train: 80%
    ├─ Val: 10%
    └─ Test: 10%
    ↓
[3/5] 특성 전처리 (데이터 누출 방지 ✅)
    ├─ Train: prepare_features(train_stats=None) ← 통계 학습
    ├─ Val: prepare_features(train_stats=...) ← Train 통계 사용
    └─ Test: prepare_features(train_stats=...) ← Train 통계 사용
    ↓
[4/5] 모델 학습
    ├─ 단일 출력 LSTM
    └─ 다중 출력 LSTM
    ↓
[5/5] 실시간 모니터링
    └─ 최고 위험도 반환 ('0', '1', '2')
    ↓
결과 반환 (에러 처리 포함 ✅)
```

---

#### 효과
✅ **_precursor.py와 완벽 연동**: `prepare_features` 시그니처 변경 반영
✅ **데이터 누출 방지**: Train 통계만 사용하도록 수정
✅ **사용자 경험 개선**: 6단계/5단계 진행 표시
✅ **안정성 향상**: None 체크, 기본값 설정, 에러 처리
✅ **문서화 개선**: 상세한 docstring 및 주석
✅ **Production-ready**: 외부 API로 사용 가능한 수준

---

## 📊 수정 전후 비교

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| **지원 데이터 타입** | 3개 (SENSOR_ID, CHAMBER_ID, EQUIPMENT_ID) | 5개+ (CELL_ID, LINE_ID 추가, 패턴 매칭) |
| **시계열 매핑** | ❌ 잘못된 인덱싱 | ✅ 정확한 입력-출력 매핑 |
| **실시간 모니터링** | ❌ 첫 샘플만 처리 | ✅ 전체 스트림 연속 모니터링 |
| **데이터 누출** | ❌ Train/Test 통계 혼용 | ✅ Train 통계만 사용 (prepare_features 개선) |
| **클래스 불균형** | ❌ 미처리 | ✅ Weighted Loss 적용 |
| **precursor.py** | ❌ 진행 상황 불명확, 에러 처리 부족 | ✅ 6단계/5단계 진행 표시, 에러 처리 강화 |
| **API 안정성** | ❌ None 체크 없음, docstring 부족 | ✅ None 안전성, 상세한 docstring |
| **코드 상태** | 🔴 Critical 버그 존재 | ✅ Production-ready |

---

---

## 📁 파일 구조

```
PRISM-Monitor/
├── prism_monitor/
│   ├── modules/
│   │   └── event_precursor/
│   │       ├── _precursor.py          # 핵심 로직 (✅ 수정됨)
│   │       ├── precursor.py           # 메인 인터페이스 (✅ 수정됨)
│   │       ├── README.md              # 📖 수정 내역 문서 (이 문서)
│   │       └── _precursor_save.py     # 백업 (수정 전)
│   └── test-scenarios/
│       └── test_data/
│           ├── semiconductor/         # 반도체 공정 데이터 (4개 CSV)
│           ├── battery/               # 배터리 제조 데이터 (4개 CSV)
│           ├── automotive/            # 자동차 조립 데이터 (4개 CSV)
│           ├── chemical/              # 화학 공정 데이터 (4개 CSV)
│           └── steel/                 # 철강 제조 데이터 (4개 CSV)
└── modification_test.py               # 🧪 테스트 스크립트 (신규)
```

### 주요 파일 설명

- **_precursor.py**: 데이터 로딩, 전처리, LSTM 모델 학습, 예측 등 핵심 로직 (5가지 주요 버그 수정)
- **precursor.py**: 외부에서 호출 가능한 메인 인터페이스 (main(), precursor() 함수)
- **modification_test.py**: 전체 파이프라인 테스트용 스크립트
- **README.md**: 모든 수정 내역과 사용법을 담은 문서

---

## 🐛 수정 항목 우선순위 분류

### 🔴 Critical (치명적) - _precursor.py
1. ✅ **시계열 레이블 인덱싱 오류** - 학습 데이터 손상
2. ✅ **실시간 모니터링 조기 종료** - 기능 미작동

### 🟡 Warning (중요) - _precursor.py
3. ✅ **Z-score 미래 정보 누출** - 과적합 가능성
4. ✅ **클래스 불균형 미처리** - 학습 품질 저하

### 🟢 Improvement (개선)
5. ✅ **장비 ID 컬럼 식별** (_precursor.py) - 데이터 호환성 향상
6. ✅ **precursor.py 인터페이스 개선** (precursor.py) - 사용자 경험 및 안정성 향상
   - main() 함수: 6단계 진행 표시, 에러 처리 강화
   - precursor() 함수: docstring 추가, None 안전성, 5단계 진행 표시

---

## 🔍 테스트 체크리스트

수정 후 다음 항목들이 정상 작동해야 합니다:

### _precursor.py 기능 테스트
- [ ] 모든 산업 카테고리 데이터 로드 (semiconductor, battery, automotive, chemical, steel)
- [ ] CELL_ID, LINE_ID 등 다양한 ID 컬럼 인식
- [ ] 시계열 데이터 생성 시 정확한 입력-출력 매핑
- [ ] Train/Val/Test 분리 후 통계 계산 (데이터 누출 방지)
- [ ] 클래스 불균형 처리 (pos_weight 계산 및 적용)
- [ ] 실시간 모니터링 전체 스트림 처리
- [ ] 최고 위험도 반환 ('0', '1', '2')

### precursor.py 인터페이스 테스트
- [ ] main() 함수: 6단계 진행 표시 정상 출력
- [ ] precursor() 함수: 5단계 진행 표시 정상 출력
- [ ] 빈 데이터셋 입력 시 에러 메시지 포함 응답 반환
- [ ] pred_value가 None일 때 0.0으로 안전하게 반환
- [ ] 모델 학습 실패 시 기본값 반환 ('0')
- [ ] 결과 딕셔너리 구조 검증: {'summary': {'predicted_value': float, 'is_anomaly': str}}

**테스트 방법**: `python modification_test.py` 실행

---

## 📖 참고 자료

### 이상 징후 예측 흐름
```
[데이터 로드]
    ↓ (다양한 ID 컬럼 지원)
[데이터 통합]
    ↓
[데이터 분할] (Train/Val/Test, shuffle=False)
    ↓
[특성 전처리] (Train 통계 계산, Val/Test는 Train 통계 사용)
    ├─ Z-score 기반 이상 레이블링
    └─ StandardScaler 정규화
    ↓
[시계열 시퀀스 생성] (정확한 인덱싱)
    ├─ 입력: [t-10 ~ t-1] 과거 10 스텝
    └─ 출력: [t+1 ~ t+5] 미래 5 스텝 내 이상 여부
    ↓
[LSTM 학습] (Weighted BCE Loss)
    ├─ 2-layer LSTM (hidden=64)
    └─ 클래스 불균형 처리
    ↓
[예측 및 경고]
    ├─ 확률 ≥ 0.7: 🚨 CRITICAL
    ├─ 확률 ≥ 0.3: ⚠️ WARNING
    └─ 확률 < 0.3: ✅ NORMAL
    ↓
[실시간 모니터링] (전체 스트림 처리)
    └─ 최고 위험도 반환
```

---


## 📊 전체 수정 요약

### _precursor.py (핵심 로직)
| 번호 | 수정 항목 | 우선순위 | 영향도 | 상태 |
|-----|----------|---------|-------|-----|
| 1 | 장비 ID 컬럼 식별 개선 | 🟢 Improvement | 데이터 호환성 | ✅ |
| 2 | 시계열 레이블 인덱싱 수정 | 🔴 Critical | 학습 품질 | ✅ |
| 3 | 실시간 모니터링 조기 종료 수정 | 🔴 Critical | 기능 작동 | ✅ |
| 4 | Z-score 데이터 누출 방지 | 🟡 Warning | 모델 성능 | ✅ |
| 5 | 클래스 불균형 처리 | 🟡 Warning | 예측 정확도 | ✅ |

### precursor.py (인터페이스)
| 번호 | 수정 항목 | 우선순위 | 영향도 | 상태 |
|-----|----------|---------|-------|-----|
| 6 | main() & precursor() 함수 개선 | 🟢 Improvement | UX & 안정성 | ✅ |

### 테스트
- ✅ **modification_test.py**: 전체 파이프라인 테스트 스크립트 생성