# InstructionRF - 제조업 AI 에이전트 명령어 변환 시스템

자연어 쿼리를 구조화된 JSON 명령어로 변환하여 제조업 AI 에이전트 오케스트레이션을 지원하는 시스템입니다.

## 🚀 주요 기능

- **자연어 → JSON 변환**: 한글/영어 쿼리를 구조화된 명령어로 변환
- **의도 분류**: ANOMALY_CHECK, PREDICTION, CONTROL, INFORMATION, OPTIMIZATION
- **작업 분해**: 복잡한 요청을 순차적 하위 작업으로 분해
- **에이전트 연동**: 전문 에이전트를 통한 고품질 변환
- **Fallback 처리**: 서버 연결 실패 시 기본 구조 제공

## 📁 파일 구성

```
InstructionRF/
├── instruction_rf_client.py   # 메인 클라이언트 클래스
├── refine_prompt.md          # 프롬프트 템플릿
├── config.json               # 서버 설정 파일
├── test_instruction_rf.py    # 종합 테스트 스크립트
└── README.md                 # 이 파일
```

## ⚙️ 설정

### 1. 서버 URL 설정 (3가지 방법)

**방법 1: 환경변수**
```bash
export LLM_API_URL="http://your-server-ip/api/agents"
```

**방법 2: config.json 편집**
```json
{
  "server": {
    "llm_api_url": "http://your-server-ip/api/agents"
  }
}
```

**방법 3: 코드에서 직접 지정**
```python
client = InstructionRefinementClient(server_url="http://your-server-ip/api/agents")
```

## 🔧 사용법

### 기본 사용

```python
from instruction_rf_client import InstructionRefinementClient

# 클라이언트 초기화
client = InstructionRefinementClient()

# 에이전트 등록 (최초 1회)
client.setup_agent()

# 쿼리 변환
instruction = client.refine_instruction("3번 엣칭 장비 상태 확인해주세요")
print(instruction)
```

### 배치 처리

```python
queries = [
    "3번 엣칭 장비 압력이 이상해요",
    "생산 수율을 개선하고 싶어요",
    "CVD 장비 온도 센서 점검 필요"
]

instructions = client.batch_refine(queries)
for inst in instructions:
    print(f"ID: {inst['instruction_id']}, Type: {inst['intent_type']}")
```

## 🧪 테스트

### 전체 테스트 실행
```bash
cd InstructionRF
python3 test_instruction_rf.py
```

### API 연결만 테스트
```python
client = InstructionRefinementClient()
result = client.test_api_connection()
print(result)
```

## 📊 출력 JSON 구조

```json
{
  "instruction_id": "inst_20240814_001",
  "original_query": "사용자의 원본 쿼리",
  "intent_type": "ANOMALY_CHECK",
  "priority": "HIGH",
  "target": {
    "equipment_id": "Etching_Machine_#3",
    "parameter": "Pressure",
    "process": "Etching"
  },
  "tasks": [
    {
      "task_id": "task_001",
      "agent": "MONITORING",
      "action": "analyze_parameter_status",
      "parameters": {...},
      "dependencies": [],
      "expected_output": "anomaly_detection_report"
    }
  ],
  "context_requirements": {
    "historical_data": true,
    "real_time_data": true,
    "external_knowledge": false,
    "simulation_needed": true
  },
  "constraints": {
    "time_limit": "5_minutes",
    "safety_requirements": ["maintain_chamber_pressure_limits"],
    "regulatory_compliance": ["semiconductor_manufacturing_standards"]
  }
}
```

## 🔌 지원하는 API 엔드포인트

- `GET /`: 서비스 상태 확인
- `GET /docs`: Swagger UI 문서
- `POST /api/generate`: 직접 텍스트 생성
- `POST /api/agents`: 새 에이전트 등록
- `POST /api/agents/{agent_name}/invoke`: 특정 에이전트 실행

## 🛠️ 문제 해결

### 연결 오류
1. 서버 상태 확인: `curl http://your-server-ip/`
2. API 문서 확인: `curl http://your-server-ip/docs`
3. 설정 파일 확인: `config.json`의 URL이 올바른지 점검

### JSON 파싱 오류
- 작은 모델(Qwen 1.5B)의 특성상 완벽한 JSON을 생성하지 못할 수 있음
- Fallback 구조가 자동으로 생성됨

## 📝 의도 분류표

| 의도 타입 | 설명 | 예시 |
|-----------|------|------|
| ANOMALY_CHECK | 이상 탐지 | "장비 상태 확인", "압력이 이상해요" |
| PREDICTION | 예측/분석 | "수율 예측", "리스크 분석" |
| CONTROL | 제어/최적화 | "파라미터 조정", "공정 최적화" |
| INFORMATION | 정보 조회 | "데이터 확인", "현황 조회" |
| OPTIMIZATION | 성능 개선 | "효율성 향상", "품질 개선" |

## 🔒 보안

- 서버 IP는 설정 파일이나 환경변수로 관리
- 코드에 하드코딩된 민감 정보 없음
- HTTPS 사용 권장 (프로덕션 환경)