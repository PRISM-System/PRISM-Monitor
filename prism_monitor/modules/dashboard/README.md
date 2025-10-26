# Dashboard Module

PRISM Monitor의 대시보드 모듈 - 제조업 공정 모니터링 및 이상치 탐지를 위한 핵심 유틸리티와 API

## 개요

이 모듈은 다양한 제조업 공정 데이터를 실시간으로 모니터링하고, 학습된 AI 모델을 통해 이상치를 탐지하는 기능을 제공합니다.

### 주요 기능
- CSV 기반 공정 데이터 자동 로딩
- TensorFlow Autoencoder 기반 이상치 탐지
- 공정 상태(STATE) 자동 추론
- FastAPI 기반 REST API 서버
- GPU/CPU 자동 선택 및 폴백

## 폴더 구조

```
dashboard/
├── dashboard.py          # 핵심 유틸리티 (데이터 로딩, 모델 로딩, 이상치 탐지)
├── dashboard_api.py      # FastAPI 기반 REST API 서버
├── test_dashboard.py     # 전체 기능 테스트 스크립트
├── simple_test.py        # 간단한 예제 및 부분 테스트
└── README.md            # 본 문서
```

## 핵심 컴포넌트

### 1. dashboard.py
CSV 데이터 로딩, 모델 로딩, 이상치 탐지를 위한 핵심 함수 제공

**주요 함수:**
- `_iter_csv_datasets(test_data_dir)`: CSV 파일 자동 스캔 및 메타데이터 생성
- `build_anomaly_registry_from_root(models_root, datasets)`: 모델 디렉토리에서 모든 모델 로드
- `make_keras_autoencoder_anomaly_fn(model_dir)`: Autoencoder 기반 이상치 탐지 함수 생성
- `_tf_init_devices(device_pref)`: TensorFlow GPU/CPU 디바이스 초기화
- `default_state_fn(row)`: 기본 공정 상태 추론 함수

**데이터 구조:**
```python
dataset = {
    "key": "battery_formation_001.csv",
    "industry": "Battery",
    "process": "Formation",
    "line": "EQUIPMENT_ID",          # 라인 식별 컬럼
    "metric_cols": ["VOLTAGE", ...], # 메트릭 컬럼 목록
    "data": DataFrame,               # 실제 데이터
    "csv_path": "/path/to/file.csv"
}
```

### 2. dashboard_api.py
외부에서 접근 가능한 REST API 서버

**주요 엔드포인트:**
- `GET /`: 헬스체크
- `GET /api/dashboard`: 전체 대시보드 데이터 조회 (각 데이터셋의 랜덤 샘플)
- `GET /api/dashboard/{industry}`: 특정 산업군 데이터만 조회
- `GET /api/info`: 로드된 데이터셋 및 모델 정보

## 설치 및 설정

### 필수 패키지
```bash
pip install pandas numpy tensorflow joblib fastapi uvicorn
```

### 환경 변수 (선택사항)
```bash
export PRISM_TEST_DATA_DIR="/path/to/test_data"
export PRISM_MODELS_ROOT="/path/to/models"
export PRISM_DEVICE="auto"  # auto | cpu | gpu
```

설정하지 않으면 자동으로 프로젝트 구조에서 경로를 추론합니다.

## 테스트 방법

### 1. 빠른 테스트 (Simple Test)

`simple_test.py`는 원하는 기능만 골라서 테스트할 수 있는 예제 스크립트입니다.

```bash
cd prism_monitor/modules/dashboard
python simple_test.py
```

**포함된 예제:**
- CSV 데이터 로딩만 테스트
- 특정 산업 데이터 필터링
- 모델 로딩 및 이상치 탐지
- 특정 CSV 파일만 테스트

**커스터마이징:**
파일을 열어 원하는 부분만 실행하거나 수정하여 사용 가능합니다.

### 2. 전체 기능 테스트 (Comprehensive Test)

`test_dashboard.py`는 모든 핵심 기능을 체계적으로 테스트합니다.

```bash
cd prism_monitor/modules/dashboard
python test_dashboard.py
```

**테스트 항목:**
1. TensorFlow 디바이스 초기화
2. CSV 데이터셋 로딩
3. 이상치 탐지 모델 로딩
4. 이상치 탐지 실행
5. 공정 상태 추론
6. 데이터 변환 유틸리티

**성공 예시:**
```
✅ PASS - tf_device
✅ PASS - csv_loading
✅ PASS - model_loading
✅ PASS - anomaly_detection
✅ PASS - state_resolution
✅ PASS - data_conversion

총 6/6 테스트 통과
🎉 모든 테스트 통과!
```

### 3. API 서버 테스트

#### 서버 실행
```bash
# 기본 실행 (포트 8000)
python -m prism_monitor.modules.dashboard.dashboard_api

# 개발 모드 (자동 재시작)
python -m prism_monitor.modules.dashboard.dashboard_api --reload

# 커스텀 포트
python -m prism_monitor.modules.dashboard.dashboard_api --port 8080
```

또는 uvicorn으로 직접 실행:
```bash
uvicorn prism_monitor.modules.dashboard.dashboard_api:app --reload
```

#### API 테스트
서버가 실행되면 다음 URL로 접근 가능:

```bash
# 헬스체크
curl http://localhost:8000/

# 대시보드 데이터 조회
curl http://localhost:8000/api/dashboard

# 특정 산업군 데이터
curl http://localhost:8000/api/dashboard/battery

# 정보 조회
curl http://localhost:8000/api/info
```

**Swagger UI**: http://localhost:8000/docs

## 프로그래밍 방식 사용 예제

### 기본 사용법
```python
from prism_monitor.modules.dashboard.dashboard import (
    _iter_csv_datasets,
    _tf_init_devices,
    build_anomaly_registry_from_root,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_MODELS_ROOT,
)

# 1. TensorFlow 초기화
_tf_init_devices("auto")

# 2. CSV 데이터 로드
datasets = _iter_csv_datasets(DEFAULT_TEST_DATA_DIR)
print(f"로드된 데이터셋: {len(datasets)}개")

# 3. 모델 로드
anomaly_models = build_anomaly_registry_from_root(DEFAULT_MODELS_ROOT, datasets)
print(f"로드된 모델: {len(anomaly_models)}개")

# 4. 이상치 탐지 실행
for ds in datasets:
    csv_name = ds['csv_name']

    if csv_name not in anomaly_models:
        continue

    anomaly_fn = anomaly_models[csv_name]

    # 첫 번째 행으로 테스트
    row = ds['data'].iloc[0]
    result = anomaly_fn(row)

    print(f"\n{csv_name}:")
    print(f"  점수: {result['score']:.4f}")
    print(f"  임계값: {result['threshold']:.4f}")
    print(f"  이상치: {result['is_anomaly']}")
```

### 특정 산업만 필터링
```python
# Battery 산업 데이터만
battery_datasets = [d for d in datasets if d['industry'].lower() == 'battery']

for ds in battery_datasets:
    print(f"- {ds['csv_name']}: {len(ds['data'])} rows")
```

### 랜덤 샘플링 및 이상치 탐지
```python
import random

ds = datasets[0]
csv_name = ds['csv_name']

if csv_name in anomaly_models:
    anomaly_fn = anomaly_models[csv_name]

    # 랜덤 샘플 10개
    sample_size = min(10, len(ds['data']))
    sample_indices = random.sample(range(len(ds['data'])), sample_size)

    anomaly_count = 0
    for idx in sample_indices:
        row = ds['data'].iloc[idx]
        result = anomaly_fn(row)
        if result['is_anomaly']:
            anomaly_count += 1

    print(f"샘플 {sample_size}개 중 {anomaly_count}개 이상치 감지")
```

## API 응답 형식

### GET /api/dashboard
```json
[
  {
    "timestamp": "2025-10-26T12:34:56.789Z",
    "industry": "Battery",
    "process": "Formation",
    "line": "CELL_001",
    "state": "RUNNING",
    "anomaly": {
      "model": "battery_formation_001_autoencoder",
      "score": 0.0234,
      "threshold": 0.05,
      "is_anomaly": false,
      "details": {
        "error_metric": "mse",
        "features": ["VOLTAGE", "CURRENT", "TEMPERATURE"]
      }
    },
    "metrics": {
      "VOLTAGE": 3.65,
      "CURRENT": 2.1,
      "TEMPERATURE": 25.3,
      "SOC": 45.2
    }
  }
]
```

## 디렉토리 구조 요구사항

```
project_root/
├── prism_monitor/
│   ├── test-scenarios/
│   │   └── test_data/        # CSV 파일들이 여기에 위치
│   │       ├── battery/
│   │       ├── semiconductor/
│   │       └── ...
│   └── modules/
│       └── dashboard/         # 본 모듈
└── models/                    # 학습된 모델들
    ├── battery_formation_001/
    │   ├── autoencoder_model.h5
    │   ├── scaler.pkl
    │   └── model_metadata.json
    └── ...
```

## 문제 해결

### TensorFlow GPU 사용 불가
```
⚠️  GPU가 감지되었지만 사용할 수 없습니다. CPU 모드로 실행합니다.
```
→ 정상 동작입니다. cuDNN이 없거나 호환되지 않으면 자동으로 CPU로 폴백합니다.

### 데이터셋이 로드되지 않음
```
❌ 로드된 CSV 데이터셋이 없습니다
```
→ `DEFAULT_TEST_DATA_DIR` 경로를 확인하고, CSV 파일에 라인 식별 컬럼(EQUIPMENT_ID 등)이 있는지 확인하세요.

### 모델이 로드되지 않음
```
⚠️  매칭되는 CSV가 없어 모델 스킵
```
→ `model_metadata.json`의 `csv_name` 또는 `csv_glob` 필드와 실제 CSV 파일명이 일치하는지 확인하세요.

## 참고사항

- **GPU/CPU 자동 선택**: TensorFlow가 GPU를 감지하지만 사용할 수 없으면 자동으로 CPU로 폴백합니다
- **라인 컬럼 자동 추론**: EQUIPMENT_ID, PROCESS_ID, CHAMBER_ID 등 우선순위에 따라 자동 선택
- **상태 추론**: STATE 컬럼이 없으면 RPM 등의 메트릭으로 RUNNING/IDLE 상태 추론
- **모델 매칭**: 모델명과 CSV 파일명이 일치하지 않으면 glob 패턴 매칭 시도

## 문의

문제가 발생하거나 질문이 있으면 이슈를 생성하거나 개발팀에 문의하세요.
