import os
from dotenv import load_dotenv
load_dotenv()

import logging

from fastapi import FastAPI, Query, Path

from src.fastapi.data_model import (
    Query2SQLRequest, Query2SQLResponse,
    EventDetectRequest, EventDetectResponse,
    EventExplainRequest, EventExplainResponse,
    CauseCandidatesRequest, CauseCandidatesResponse,
    PrecursorRequest, PrecursorResponse,
    EvaluateRiskRequest, EvaluateRiskResponse,
    PredictionRiskRequest, PredictionRiskResponse,
    DashboardResponse,
    WorkflowStartRequest, WorkflowStartResponse,
)
from src.test_scenarios.modeling import TestScenarioModel
from src.agent.agent import MonitoringAgent
from src.modules.query2sql.query2sql import query2sql
from src.modules.event.event_detect import detect_anomalies
from src.modules.explanation.explanation import event_explain, event_cause_candidates
from src.modules.precursor.precursor import event_precursor
from src.modules.risk_assessment.risk_assessment import evaluate_event_risk, prediction_risk
from src.task.task import (
    workflow_start,
    get_dashboard
)

# Logger 설정
logger = logging.getLogger("prism_monitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

TEST_SCENARIO_MODEL = TestScenarioModel()
TEST_SCENARIO_MODEL.set_models()

AGENT = MonitoringAgent(prism_core_url=os.environ["PRISM_CORE_URL"], agent_name=os.environ["AGENT_NAME"])


app = FastAPI(
    title="PRISM Monitoring",
    description="자율 제조 구현을 위한 AI 에이전트 모니터링 모듈",
    version="1.0"
)


@app.get(
    "/api/v1/monitoring/dashboard",
    response_model=DashboardResponse,
    summary="모니터링 대시보드 조회",
    tags=["Workflow"]
)
def app_get_dashboard():
    logger.info("Dashboard data requested")
    res = get_dashboard()
    return res

# workflow api post
@app.post(
    "/api/v1/workflow/start",
    response_model=WorkflowStartResponse,
    summary="워크플로우 시작",
    tags=["Workflow"]
)
def app_start_workflow(body: WorkflowStartRequest):
    logger.info(f"Workflow start requested: {body}")
    res = workflow_start(
        task_id=body.taskId,
        query=body.query,
    )
    return res

#query2sql api post
@app.post(
    "/api/v1/monitoring/query2sql",
    response_model=Query2SQLResponse,
    summary="질의 SQL 변환",
    tags=["Monitoring"]
)
def app_query2sql(body: Query2SQLRequest):
    logger.info(f"Query to SQL requested: {body}")
    res = query2sql(
        user_query=body.query,
        serialize=True
    )
    return {'result': res}

# Event Detection API
@app.post(
    "/api/v1/monitoring/event/detect",
    response_model=EventDetectResponse,
    summary="이벤트 탐지",
    tags=["Monitoring"]
)
def app_event_detect(body: EventDetectRequest):
    logger.info(f"Event detection requested: {body}")
    res = detect_anomalies(
        model=TEST_SCENARIO_MODEL,
        target_process=body.targetProcess,
        start=body.start,
        end=body.end,
        serialize=True
    )
    return res

# Event Explanation API
@app.post(
    "/api/v1/monitoring/event/explain",
    response_model=EventExplainResponse,
    summary="이벤트 설명",
    tags=["Monitoring"]
)
def app_event_explain(body: EventExplainRequest):
    logger.info(f"Event explanation requested: {body}")
    res = event_explain(
        event_detect_analysis=body.eventDetectAnalysis,
        process_type=body.targetProcess,
    )
    return {'result': res}

# Cause Candidates API
@app.post(
    "/api/v1/monitoring/event/cause-candidates",
    response_model=CauseCandidatesResponse,
    summary="원인 후보 도출",
    tags=["Monitoring"]
)
def app_cause_candidates(body: CauseCandidatesRequest):
    logger.info(f"Cause candidates requested: {body}")
    res = event_cause_candidates(
        event_detect_analysis=body.eventDetectAnalysis,
        process_type=body.targetProcess,
    )
    return {'result': res}

# Precursor API
@app.post(
    "/api/v1/monitoring/event/precursor",
    response_model=PrecursorResponse,
    summary="선행 지표 분석",
    tags=["Monitoring"]
)
def app_event_precursor(body: PrecursorRequest):
    logger.info(f"Event precursor analysis requested: {body}")
    res = event_precursor(
        model=TEST_SCENARIO_MODEL,
        target_process=body.targetProcess,
        start=body.start,
        end=body.end,
        serialize=True
    )
    return {'result': res}

# Evaluate Risk API
@app.post(
    "/api/v1/monitoring/event/evaluate-risk",
    response_model=EvaluateRiskResponse,
    summary="위험도 평가",
    tags=["Monitoring"]
)
def app_evaluate_risk(body: EvaluateRiskRequest):
    logger.info(f"Evaluate risk requested: {body}")
    res = evaluate_event_risk(
        event_detect_analysis=body.eventDetectAnalysis
    )
    return {'result': res}

# Prediction Risk API
@app.post(
    "/api/v1/monitoring/event/prediction-risk",
    response_model=PredictionRiskResponse,
    summary="위험도 예측",
    tags=["Monitoring"]
)
def app_prediction_risk(body: PredictionRiskRequest):
    logger.info(f"Prediction risk requested: {body}")
    res = prediction_risk(
        task_instructions=body.taskInstructions
    )
    return {'result': res}