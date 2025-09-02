from dotenv import load_dotenv
load_dotenv()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import logging

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
LLM_URL=os.environ['LLM_URL']
MONITOR_DB = TinyDB(DATABASE_PATH)
PRISM_CORE_DB = PrismCoreDataBase(os.environ['PRISM_CORE_DATABASE_URL'])

# Logger 설정
logger = logging.getLogger("prism_monitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 라우팅 예시
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

#/api/v1/task/{task_id}/monitoring/status=pending
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

#/api/v1/monitoring/event/output
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

# workflow api post
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
        task_id=body.taskId
    )
    return res

# real time monitoring visualization
@app.get(
    "/api/v1/monitoring/real-time",
    summary="실시간 모니터링 데이터 조회",
    tags=["Monitoring"]
)
def get_real_time_monitoring_data():
    logger.info("Real-time monitoring data requested")
    return monitoring_real_time(PRISM_CORE_DB)  # 직접 반환

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)