# AI 워크플로우 및 모니터링 시스템

이 프로젝트는 시나리오 데이터 기반의 모니터링 대시보드와 복잡한 분석 워크플로우를 실행하는 AI 시스템입니다. 자연어 쿼리를 입력받아 다단계 분석을 수행하고, 실시간으로 시스템 상태를 예측합니다.

## 전체 예시 코드
### env 준비
```
#.env-local 파일에 작성합니다.
BIMATRIX_LLM_AGENT_INVOKE_URL=https://grnd.bimatrix.co.kr/django/agi/core/api/agents/Monitoring/invoke/
BIMATRIX_LLM_URL=https://grnd.bimatrix.co.kr/django/agi/llm-agent/
PLATFORM_URL=https://grnd.bimatrix.co.kr/django/agi/
ID=hanyang
PW=hanyang1234
USER_ID=user_2222
OPENROUTER_API_KEY= #임시 백업용 llm 키, 키 존재시 llm 호출 에러시에 openrouter의 무료 모델을 사용합니다
```
### 실행
```
#bash
docker-compose up --build
```
### API 요청
```
#python
import requests
#대시보드 조회
url = 'http://localhost:8002/api/v1/monitoring/dashboard'
res = requests.get(url)
print(res.json())
#이상탐지 수행
url = 'http://localhost:8002/api/v1/workflow/start'
data = {
    'taskId': 'example_task_id',
    'query': '어셈블리 생산라인 최근 이상탐지 해줘'
    'extra_key': '...'
}
res = requests.post(url, json=data)
print(res.json())
#{
#    'summary': '...', # <-를 사용하면 됩니다.
#    'result': '...',
#    'monitored_timeseries':{
#        'format': 'csv',
#        'description': '...',
#        'sample_data': '...'
#    }
#}
```

## 주요 기능 (Main Features)

### 1. 대시보드 (Dashboard)

시나리오 데이터를 기반으로 모든 시나리오의 현재와 미래 상태를 예측하여 제공합니다.

* **API:** `GET /api/v1/monitor/dashboard`
* **응답 예시 (Response Example):**
    ```json
    {
    "dashboard": {
        "semiconductor_cmp": {
            "predictions" : [
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "HEAD_ROTATION": 120.17138671875,
                    "MOTOR_CURRENT": 16.496299743652344,
                    "PAD_THICKNESS": 2.012294054031372,
                    "POLISH_TIME": 199.8785858154297,
                    "PRESSURE": 3.149606227874756,
                    "SLURRY_FLOW_RATE": 251.85693359375,
                    "SLURRY_TEMP": 25.056913375854492,
                    "TEMPERATURE": 25.00965118408203,
                    "VIBRATION": 0.986939549446106
                },
                {
                    ...
                }
            ],
            "current_data": [
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "HEAD_ROTATION": 120.17138671875,
                    "MOTOR_CURRENT": 16.496299743652344,
                    "PAD_THICKNESS": 2.012294054031372,
                    "POLISH_TIME": 199.8785858154297,
                    "PRESSURE": 3.149606227874756,
                    "SLURRY_FLOW_RATE": 251.85693359375,
                    "SLURRY_TEMP": 25.056913375854492,
                    "TEMPERATURE": 25.00965118408203,
                    "VIBRATION": 0.986939549446106
                },
                {
                    ...
                }
            ]
        },
        "semiconductor_etch": {
            "predictions" : [
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "...": ...,
                },
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "...": ...,
                },
            ],
            "current_data": [
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "...": ...,
                },
                {
                    "ANOMALY_SCORE": 0.09563945978879929,
                    "...": ...,
                },
            ]
        }
    }
    }
    ```

### 2. 워크플로우 (Workflow)

사용자의 자연어 쿼리(Query)를 받아 총 8단계의 복잡한 분석 파이프라인을 수행합니다.

* **API:** `POST /api/v1/workflow/start`
* **요청 예시 (Request Example):**
    ```json
    {
      "taskId": "TASK_0001",
      "query": "용접 라인의 WELD_CURRENT가 불안정합니다. 현재 용접 품질 상태를 분석해주세요",
      "extras": "..."
    }
    ```
* **응답 예시 (Response Example):**
    ```json
    {
        "summary": "...",
        "result": "...",
        "monitored_timeseries":{
            "format": "csv",
            "description": "...",
            "sample_data": "..."
        }
    }
    ```
* **워크플로우 단계:**
    1.  **query2sql**: 자연어 쿼리를 SQL로 변환합니다.
    2.  **이상치분석 (Anomaly Detection)**: 데이터에서 이상 징후를 탐지합니다.
    3.  **원인 설명 (Root Cause Analysis)**: 이상 현상의 근본 원인을 분석합니다.
    4.  **과업 후보군 생성 (Task Candidate Generation)**: 문제 해결을 위한 작업 후보를 생성합니다.
    5.  **향후 상태 예측 (Future State Prediction)**: 현재 상태 기반으로 미래 상태를 예측합니다.
    6.  **리스크 평가 (Risk Assessment)**: 잠재적 리스크를 평가합니다.
    7.  **향후 리스크 예측 (Future Risk Prediction)**: 미래에 발생 가능한 리스크를 예측합니다.
    8.  **종합 분석 결과 (Comprehensive Analysis Result)**: 모든 분석 결과를 요약하여 제공합니다.

    *관련 이미지: `image_808421.png`*

---

## 설치 (Installation)

본 프로젝트는 `pipenv`를 통해 패키지 및 가상 환경을 관리합니다.

1.  **pipenv 설치**
    ```bash
    pip install pipenv
    ```

2.  **가상 환경 활성화**
    ```bash
    pipenv shell
    ```

3.  **의존성 라이브러리 설치**
    `Pipfile` 및 `Pipfile.lock`에 명시된 라이브러리를 설치합니다.
    ```bash
    pipenv install
    ```

### 환경 변수 (Environment Variables)

* 프로젝트 루트 디렉터리에 `.env` 파일을 생성하여 필요한 환경 변수를 설정합니다.
* `pipenv shell` 명령어 실행 시 `.env` 파일의 변수들이 자동으로 환경에 주입됩니다.

---

## 실행 (Execution)

1.  **가상 환경 활성화**
    ```bash
    pipenv shell
    ```

2.  **서버 실행 (Uvicorn)**
    `uvicorn`을 사용하여 애플리케이션을 실행합니다. `--reload` 옵션은 코드 변경 시 서버를 자동으로 재시작합니다.
    ```bash
    uvicorn main:app --reload
    ```

### 초기 실행 참고사항

* **최초 실행 시:** 시나리오 데이터에 대한 모델 학습 및 저장을 수행합니다. 이 과정에서 시간이 다소 소요될 수 있습니다.
* **이후 실행 시:** 이미 저장된 모델이 있다면, 학습 과정 없이 바로 모델을 불러옵니다.
* **초기 설정:** 최초 실행 시 에이전트(Agents) 및 툴(Tools) 등록 과정을 수행합니다.

---

## Docker를 통한 배포

* Dockerfile을 통한 배포 시에도 `pipenv`를 설치하고 실행하는 과정이 포함되어 있습니다.
* Docker 배포 시에는 `.env-local` 파일의 값이 `.env` 파일로 복사되어 환경 변수로 주입됩니다. 배포 환경에 맞게 `.env-local` 파일을 수정하십시오.