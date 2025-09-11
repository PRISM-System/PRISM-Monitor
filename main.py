from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import time
import requests
import signal
import atexit
import logging

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from tinydb import TinyDB, Query
from fastapi import FastAPI, Query, Path
from typing import Union, Literal

from prism_monitor.data.database import PrismCoreDataBase
from prism_monitor.data.models import (
    DashboardResponse,
    StatusPendingResponse,
    EventOutputRequest, EventOutputResponse,
    MonitoringOutputRequest, MonitoringOutputResponse,
    EventDetectRequest, EventDetectResponse,
    EventExplainRequest, EventExplainResponse,
    CauseCandidatesRequest, CauseCandidatesResponse,
    PrecursorRequest, PrecursorResponse,
    EvaluateRiskRequest, EvaluateRiskResponse,
    DashboardUpdateRequest, DashboardUpdateResponse,
    WorkflowStartRequest, WorkflowStartResponse,
    RealTimeMonitoringResponse
)
from prism_monitor.modules.task import (
    monitoring_dashboard,
    monitoring_status_pending,
    monitoring_output,
    workflow_start
)
from prism_monitor.modules.monitoring import (
    monitoring_event_output, 
    monitoring_event_detect, 
    monitoring_event_explain, 
    monitoring_event_cause_candidates,
    monitoring_event_precursor,
    monitoring_event_evaluate_risk,
    monitoring_dashboard_update,
    monitoring_real_time
)

DATABASE_PATH="monitor_db.json"
LOCAL_FILE_DIR='prism_monitor/data/local'
LLM_URL=os.environ.get('LLM_URL', 'http://0.0.0.0:8001')
MONITOR_DB = TinyDB(DATABASE_PATH)
PRISM_CORE_DB = PrismCoreDataBase(os.environ['PRISM_CORE_DATABASE_URL'])

# vLLM 프로세스를 저장할 전역 변수
vllm_process = None

# Logger 설정
logger = logging.getLogger("prism_monitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI 앱 인스턴스 생성
app = FastAPI()

def is_vllm_running(port=8001):
    """vLLM 서버가 실행 중인지 확인"""
    try:
        response = requests.get(f"http://0.0.0.0:{port}/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_vllm_server():
    """vLLM 서버 시작"""
    global vllm_process
    
    # 이미 실행 중인지 확인
    if is_vllm_running():
        logger.info("vLLM server is already running")
        return
    
    logger.info("Starting vLLM server...")
    
    # vLLM 서버 시작
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-0.6B",
        "--port", "8001",
        "--host", "0.0.0.0"
    ]
    
    try:
        vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # 새로운 process group 생성
        )
        
        # vLLM 서버가 시작될 때까지 대기
        max_wait_time = 120  # 최대 2분 대기
        wait_interval = 5    # 5초마다 확인
        
        for i in range(0, max_wait_time, wait_interval):
            if is_vllm_running():
                logger.info(f"vLLM server started successfully (took {i}s)")
                return
            time.sleep(wait_interval)
            logger.info(f"Waiting for vLLM server to start... ({i}s)")
        
        logger.error("vLLM server failed to start within timeout")
        
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")

def stop_vllm_server():
    """vLLM 서버 종료"""
    global vllm_process
    
    if vllm_process:
        logger.info("Stopping vLLM server...")
        try:
            # process group 전체 종료
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
            vllm_process.wait(timeout=10)
        except:
            # 강제 종료
            try:
                os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
            except:
                pass
        finally:
            vllm_process = None
            logger.info("vLLM server stopped")

# 프로그램 종료 시 vLLM 서버도 함께 종료
atexit.register(stop_vllm_server)

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting PRISM Monitor application...")
    start_vllm_server()

# FastAPI shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down PRISM Monitor application...")
    stop_vllm_server()

# 기존 라우팅들...
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Hello World"}

@app.get(
    "/api/v1/task/{task_id}/monitoring/dashboard", 
    response_model=DashboardResponse,
    summary="실시간 대시보드 조회 (line or sensor)",
    tags=["Monitoring"]
)
def get_dashboard_data(
    task_id: int = Path(..., description="작업 ID"),
    type: Literal["LINE", "SENSOR"] = Query("LINE", description="요청 타입, LINE 또는 SENSOR"),
    field: int = Query(4, description="필터링할 필드, line_id 또는 sensor_id")
):
    logger.info(f"Dashboard requested: task_id={task_id}, type={type}, field={field}")
    res = monitoring_dashboard(task_id, type, field)
    return res

@app.get(
    "/api/v1/task/{task_id}/monitoring/status",
    response_model=StatusPendingResponse,
    summary="과업 처리 진행 상황 조회",
    tags=["Monitoring"]
)
def get_task_status(
    task_id: int = Path(..., description="작업 ID"),
    status: Literal["pending"] = Query("pending", description="과업 상태 필터")
):
    logger.info(f"Task status requested: task_id={task_id}, status={status}")
    res = monitoring_status_pending(task_id, status)
    return res

@app.post(
    "/api/v1/monitoring/event/output",
    response_model=EventOutputResponse,
    summary="이상 발생 알림",
    tags=["Monitoring"]
)
def receive_monitoring_event(body: EventOutputRequest):
    logger.info(f"Monitoring event received: {body}")
    res = monitoring_event_output(
        status=body.result.status,
        anomaly_detected=body.result.anomalyDetected,
        description=body.result.description
    )
    return res

@app.post(
    "/api/v1/task/{task_id}/monitoring/output",
    response_model=MonitoringOutputResponse,
    summary="과업지시 결과 오케스트레이션 전달",
    tags=["Monitoring"]
)
def receive_monitoring_output(
    task_id: int = Path(..., description="과업 ID"),
    body: MonitoringOutputRequest = ...
):
    logger.info(f"Monitoring output received: task_id={task_id}, body={body}")
    res = monitoring_output(
        task_id=task_id,
        status=body.result.status,
        anomaly_detected=body.result.anomalyDetected,
        description=body.result.description
    )
    return res

@app.post(
    "/api/v1/monitoring/event/detect",
    response_model=EventDetectResponse,
    summary="주어진 시간 구간에 이상 이벤트 탐지",
    tags=["Monitoring"]
)
def detect_anomaly_in_period(body: EventDetectRequest):
    logger.info(f"Anomaly detection requested: taskId={body.taskId}, start={body.start}, end={body.end}")
    res = monitoring_event_detect(
        task_id=body.taskId,
        start=body.start,
        end=body.end,
        prism_core_db=PRISM_CORE_DB,
        monitor_db=MONITOR_DB
    )
    print(res)
    return res

@app.post(
    "/api/v1/monitoring/event/explain",
    response_model=EventExplainResponse,
    summary="이상 이벤트의 원인 설명",
    tags=["Monitoring"]
)
def explain_anomaly_event(body: EventExplainRequest):
    logger.info(f"Anomaly explanation requested: {body}")
    res = monitoring_event_explain(
        llm_url=LLM_URL,
        monitor_db=MONITOR_DB,
        task_id=body.taskId
    )
    return res

@app.post(
    "/api/v1/monitoring/event/cause-candidates",
    response_model=CauseCandidatesResponse,
    summary="예측 분석을 위한 문제 원인 후보군 생성",
    tags=["Monitoring"]
)
def explain_anomaly_event(body: CauseCandidatesRequest):
    logger.info(f"Cause candidates requested: {body}")
    res = monitoring_event_cause_candidates(
        llm_url=LLM_URL,
        monitor_db=MONITOR_DB,
        task_id=body.taskId
    )
    return res

@app.post(
    "/api/v1/monitoring/event/precursor",
    response_model=PrecursorResponse,
    summary="현재 상태로부터 향후 이상징후 예측",
    tags=["Monitoring"]
)
def explain_anomaly_event(body: PrecursorRequest):
    logger.info(f"Precursor requested: taskId={body.taskId}, start={body.start}, end={body.end}")
    res = monitoring_event_precursor(
        monitor_db=MONITOR_DB,
        prism_core_db=PRISM_CORE_DB,
        start=body.start,
        end=body.end,
        task_id=body.taskId
    )
    return res

@app.post(
    "/api/v1/monitoring/event/evaluate-risk",
    response_model=EvaluateRiskResponse,
    summary="현재 상태를 기준으로 위험 여부 평가",
    tags=["Monitoring"]
)
def explain_anomaly_event(body: EvaluateRiskRequest):
    logger.info(f"Risk evaluation requested: taskId={body.taskId}, topk={body.topk}")
    res = monitoring_event_evaluate_risk(
        llm_url=LLM_URL,
        monitor_db=MONITOR_DB,
        task_id=body.taskId,
        topk=body.topk
    )
    return res

@app.post(
    "/api/v1/monitoring/dashboard/update",
    response_model=DashboardUpdateResponse,
    summary="대시보드 업데이트",
    tags=["Monitoring"]
)
def update_dashboard(body: DashboardUpdateRequest):
    logger.info(f"Dashboard update requested: {body}")
    res = monitoring_dashboard_update(
        field=body.field, 
        type=body.type,
        status=body.status,
        anomaly_detected=body.anomalyDetected,
        anomaly_type=body.anomalyType,
        updated_at=body.updatedAt,
    )
    return res

@app.post(
    "/api/v1/workflow/start",
    response_model=WorkflowStartResponse,
    summary="워크플로우 시작",
    tags=["Workflow"]
)
def start_workflow(body: WorkflowStartRequest):
    logger.info(f"Workflow start requested: {body}")
    res = workflow_start(
        llm_url=LLM_URL,
        monitor_db=MONITOR_DB,
        prism_core_db=PRISM_CORE_DB,
        task_id=body.taskId,
        query=body.query
    )
    return res

@app.get(
    "/api/v1/monitoring/real-time",
    summary="실시간 모니터링 데이터 조회",
    tags=["Monitoring"]
)
def get_real_time_monitoring_data():
    logger.info("Real-time monitoring data requested")
    return monitoring_real_time(PRISM_CORE_DB)

# vLLM 서버 상태 확인 엔드포인트 추가
@app.get("/api/v1/llm/status")
def get_llm_status():
    """vLLM 서버 상태 확인"""
    if is_vllm_running():
        return {"status": "running", "url": LLM_URL}
    else:
        return {"status": "stopped", "url": LLM_URL}

if __name__ == "__main__":
    import uvicorn
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        stop_vllm_server()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)  # reload=False로 설정