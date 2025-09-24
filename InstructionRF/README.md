```markdown
# InstructionRF

자연어 작업자 쿼리를 구조화된 JSON 명령어로 변환하는 의도(Intent) 분류 시스템입니다. vLLM 서버 자동 구동과 평가를 원클릭으로 수행할 수 있습니다.

## 🎯 주요 기능

- **지원 의도**: `ANOMALY_CHECK` | `PREDICTION` | `CONTROL` | `INFORMATION` | `OPTIMIZATION`
- **서버**: vLLM OpenAI 호환 API (`/v1`)
- **평가 지표**: Accuracy, Classification report, Confusion matrix, 에러 로그
- **원클릭 실행**: 서버 실행부터 평가까지 자동화

## 📝 의도 분류표

| 의도 타입 | 설명 | 예시 쿼리 |
|-----------|------|----------|
| **ANOMALY_CHECK** | 이상 탐지 및 점검 필요성 판단 | "Etcher 7번에서 금일 2런 연속 엔드포인트 SNR이 20% 낮아졌습니다. 야간 첫 런은 정상, 이후 런에서만 저하가 반복됩니다. RF 매칭 튜너 셋틀 시간과 가스 전환 타이밍 로그를 함께 확인해 설비 이상 여부와 즉시 점검 필요 여부를 판단해 주세요." |
| **PREDICTION** | 예측 및 분석 요청 | "PVD 7번 진공 회복 시간에 6시간 주기 패턴이 보입니다. 같은 패턴 유지 시 이틀 간 파티클 악화 확률과 생산 영향(UPH 손실)을 예측해 주세요." |
| **CONTROL** | 제어 및 조정 작업 | "Implanter 2번 도즈 균일도 개선을 위해 어퍼처/스캔 속도/온도 제어를 상호작용 고려하에 병행 조정합니다. 권장 순서·단계별 절차, 안전 한계, 검증/합격 기준, 실패 시 롤백 기준을 포함한 제어안을 작성해 주세요." |
| **INFORMATION** | 정보 조회 및 요약 | "Metrology 4번 CD_SEM 포커스 드리프트 보정 이력/보정 후 잔차를 표·주석 중심 요약 리포트로 정리해 주세요." |
| **OPTIMIZATION** | 성능 개선 및 최적화 | "Photo 5번에서 오버레이/포커스 품질을 유지하면서 보정 작업으로 인한 스루풋 저하를 최소화하는 최적 보정 전략을 제안해 주세요. 후보안과 수치 근거를 제시해 주세요." |

## 📁 프로젝트 구조

```
InstructionRF/
├─ run.sh                              # vLLM 서버 + 평가 원클릭 실행
├─ evaluate_intent_accuracy.py         # 평가 스크립트
├─ instruction_rf_client.py            # OpenAI 호환 클라이언트
├─ prompts.yaml                        # 프롬프트/스키마/예시 관리
└─ data/
    └─ Semiconductor_intent_dataset__preview_.csv  # 평가용 데이터셋
```

## 🛠️ 요구 사항

### 시스템 요구사항
- **Python**: 3.10+ (권장 3.11)
- **GPU**: NVIDIA GPU + CUDA 드라이버 (vLLM 실행용)

### 패키지 설치
```bash
# 기본 패키지
pip install pandas scikit-learn requests pyyaml vllm

# 4비트 양자화 지원 (선택사항)
pip install "bitsandbytes>=0.46.1"
```

## 📊 데이터셋 포맷

CSV 파일은 다음 형식을 따라야 합니다:

```csv
id,intent,query
Q001,ANOMALY_CHECK,"장비 A의 온도가 비정상적으로 높은데 즉시 점검이 필요한지 판단해 주세요"
Q002,PREDICTION,"다음 주 생산량을 예측해 주세요"
Q003,CONTROL,"공정 압력을 안전하게 조정하는 제어안을 제시해 주세요"
```

**중요 사항:**
- `intent` 값은 반드시 지원되는 5가지 의도 중 하나여야 합니다
- `query`는 실제 작업자 발화처럼 자연스러운 문장으로 작성

## 🚀 빠른 시작

### 원클릭 실행
```bash
bash run.sh
```

이 명령어는 다음을 자동으로 수행합니다:
1. vLLM 서버 백그라운드 실행
2. 서버 헬스체크
3. 의도 분류 평가 실행
4. 결과 출력

### 로그 파일 위치
- **서버 로그**: `/tmp/vllm_oneclick/vllm_server.log`
- **평가 로그**: `/tmp/vllm_oneclick/evaluate.log`
- **서버 PID**: `/tmp/vllm_oneclick/vllm_server.pid`

### 중단하기
`Ctrl+C`로 중단하면 모든 프로세스가 자동으로 정리됩니다.

## 🔧 모델 변경하기

`run.sh` 파일의 환경 변수만 수정하면 됩니다.

### A. 프리-양자화 모델 (권장)
```bash
# run.sh
MODEL="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
# --quantization 옵션 불필요 (자동 인식)
```

### B. 실시간 4bit 양자화
```bash
# run.sh
MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_EXTRA_ARGS="--quantization bitsandbytes"
```

## 📈 수동 평가 실행

평가만 별도로 실행하려면:

```bash
python evaluate_intent_accuracy.py \
  --base_url http://127.0.0.1:8000/v1 \
  --csv ./data/Semiconductor_intent_dataset__preview_.csv \
  --model "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit" \
  --out predictions.jsonl \
  --limit 100
```

### 주요 옵션
- `--base_url`: vLLM 서버 URL (기본값: `http://127.0.0.1:8000/v1`)
- `--csv`: 평가용 CSV 파일 경로
- `--out`: 예측 결과 저장 파일 (JSONL 형식)
- `--limit`: 평가할 샘플 수 제한 (스모크 테스트용)
- `--model`: 모델명 (로그에 기록됨)

## ⚙️ 프롬프트 커스터마이징

`prompts.yaml`에서 시스템 프롬프트, 스키마, Few-shot 예시를 관리합니다.

### 중요 규칙
1. **정확한 복사**: `output.original_query`는 `input`과 **문자 단위로 완전히 동일**해야 합니다
2. **의도별 키워드**: 각 의도를 명확히 구분할 수 있는 키워드 포함

### 의도별 키워드 가이드
```yaml
ANOMALY_CHECK: "이상 여부 판단", "즉시 점검 필요", "설비 이상"
PREDICTION: "예측해 주세요", "확률", "영향 분석", "손실 예측"
CONTROL: "제어안 제시", "조정", "절차", "롤백 기준"
INFORMATION: "요약", "리포트", "조회", "정리해 주세요"
OPTIMIZATION: "최적화", "개선", "효율", "후보안 제시"
```

## 📋 결과 해석

### 콘솔 출력
- **Accuracy**: 전체 정확도
- **Classification Report**: 클래스별 precision, recall, f1-score
- **Confusion Matrix**: 예측 혼동 행렬

### 상세 로그 (`predictions.jsonl`)
각 샘플별로 다음 정보가 저장됩니다:
```json
{
  "gold_intent": "ANOMALY_CHECK",
  "pred_intent": "ANOMALY_CHECK", 
  "query": "장비 온도 이상 여부 확인",
  "model_output": "..."
}
```

## 🧪 빠른 재현 체크리스트

1. **데이터 확인**: `data/*.csv` 헤더/라벨/인코딩(UTF-8)
2. **프롬프트 검증**: `prompts.yaml`의 `original_query == input`
3. **모델 설정**: `run.sh`의 `MODEL` 변수
4. **패키지 설치**: 필요시 `bitsandbytes` 추가
5. **실행**: `bash run.sh`
6. **결과 확인**: 콘솔, 로그 파일, `predictions.jsonl`
