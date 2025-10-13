# 제조업 AI 에이전트 명령어 분류 데이터셋

## 개요
반도체 제조업 현장에서 발생하는 자연어 쿼리를 5가지 의도로 분류하기 위한 테스트 데이터셋입니다.

## 데이터셋 구성
- **총 데이터 수**: 100건
- **의도 유형**: 5개 (ANOMALY_CHECK, OPTIMIZATION, CONTROL, PREDICTION, INFORMATION)
- **우선순위**: 3개 (HIGH, MEDIUM, LOW)

### 의도별 분포
- **ANOMALY_CHECK**: 30건 (30%) - 이상 감지 및 점검
- **OPTIMIZATION**: 25건 (25%) - 최적화 및 개선
- **CONTROL**: 15건 (15%) - 제어 및 조작
- **PREDICTION**: 15건 (15%) - 예측 및 분석
- **INFORMATION**: 15건 (15%) - 정보 조회

### 우선순위별 분포
- **HIGH**: 주로 ANOMALY_CHECK와 긴급 CONTROL
- **MEDIUM**: OPTIMIZATION, PREDICTION, 일반 CONTROL
- **LOW**: 대부분의 INFORMATION

## 파일 형식

### 1. JSON 형식 (권장)
```
manufacturing_dataset_100.json
```
- 메타데이터 포함
- 프로그래밍 작업에 최적
- UTF-8 인코딩

### 2. CSV 형식 (Excel 호환)
```
manufacturing_dataset_100.csv
```
- Excel에서 바로 열람 가능
- UTF-8 BOM 인코딩

### 3. Excel 형식 (분석용)
```
manufacturing_dataset_100.xlsx
```
- 다중 시트 구조
- 의도별 분리된 시트
- 통계 정보 시트 포함

## 데이터 구조

### JSON 형식
```json
{
  "metadata": {
    "title": "제조업 AI 에이전트 명령어 분류 데이터셋",
    "created_at": "2024-12-18T...",
    "total_samples": 100,
    "intent_types": {...},
    "priority_levels": {...}
  },
  "dataset": [
    {
      "id": 1,
      "query": "3번 엣칭 장비 압력이 이상해요",
      "expected_intent_type": "ANOMALY_CHECK",
      "expected_priority": "HIGH"
    }
  ]
}
```

## 사용법

### 1. Python에서 로드
```python
import json

with open('data/manufacturing_dataset_100.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = data['dataset']
```

### 2. 평가 시스템과 연동
```python
from enhanced_instruction_rf import EnhancedInstructionRefinementClient
from evaluation_script import InstructionRFEvaluator

# 클라이언트 초기화
client = EnhancedInstructionRefinementClient()

# 평가 실행
evaluator = InstructionRFEvaluator(client, dataset)
metrics = evaluator.run_evaluation()
```

## 목표 성능
- **Intent 분류 정확도**: 99% 이상
- **Priority 분류 정확도**: 99% 이상
- **전체 정확도**: 99% 이상

## 데이터 예시

### ANOMALY_CHECK (이상 감지)
- "3번 엣칭 장비 압력이 이상해요" (HIGH)
- "플라즈마 챔버 압력 센서 상태 확인해주세요" (MEDIUM)

### OPTIMIZATION (최적화)
- "생산 수율을 개선하고 싶어요" (MEDIUM)
- "에너지 효율성을 높이는 방법을 알려주세요" (MEDIUM)

### CONTROL (제어)
- "장비를 긴급 정지시켜주세요" (HIGH)
- "CVD 온도를 350도로 설정해주세요" (MEDIUM)

### PREDICTION (예측)
- "다음 주 생산량을 예측해주세요" (MEDIUM)
- "장비 교체 시기가 언제쯤 될까요?" (MEDIUM)

### INFORMATION (정보)
- "현재 생산량이 얼마나 되나요?" (LOW)
- "오늘 수율 현황을 알려주세요" (LOW)

## 버전 관리
- v1.0: 초기 100건 데이터셋
- 백업 파일: `data/backups/manufacturing_dataset_backup_YYYYMMDD_HHMMSS.json`

## 라이선스
내부 연구용 데이터셋
