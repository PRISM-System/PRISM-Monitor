from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel
from factory_monitoring_agent import FactoryMonitoringAgent
import asyncio

app = FastAPI(title="PRISM-Monitor API", description="제조 공정 모니터링 에이전트 API", version="0.1.0")

# 목데이터
SENSOR_DATA = [
    {"sensor_id": 1, "type": "temperature", "value": 180, "unit": "°C", "timestamp": "2025-07-10T10:00:00"},
    {"sensor_id": 2, "type": "pressure", "value": 2.1, "unit": "bar", "timestamp": "2025-07-10T10:00:00"}
]

EVENTS = [
    {"event_id": 101, "type": "warning", "message": "온도 임계값 초과", "timestamp": "2025-07-10T10:01:00"}
]

ANOMALY = {"detected": True, "score": 0.92, "description": "온도 급상승 감지"}

RISK = {"level": "high", "reason": "온도 200°C 초과", "priority": 1}

class SensorData(BaseModel):
    sensor_id: int
    type: str
    value: float
    unit: str
    timestamp: str

class Event(BaseModel):
    event_id: int
    type: str
    message: str
    timestamp: str

@app.get("/status", summary="공정 상태 요약", tags=["실시간 모니터링"])
def get_status() -> Dict[str, Any]:
    return {"status": "RUNNING", "active_sensors": len(SENSOR_DATA), "last_event": EVENTS[-1]}

@app.get("/sensors", response_model=List[SensorData], summary="센서 데이터 조회", tags=["실시간 모니터링"])
def get_sensors() -> List[SensorData]:
    return SENSOR_DATA

@app.get("/events", response_model=List[Event], summary="이벤트 목록 조회", tags=["경고/이벤트"])
def get_events() -> List[Event]:
    return EVENTS

@app.get("/anomaly", summary="이상 탐지 결과", tags=["이상 탐지"])
def get_anomaly() -> Dict[str, Any]:
    return ANOMALY

@app.get("/risk", summary="위험 평가 결과", tags=["위험 평가"])
def get_risk() -> Dict[str, Any]:
    return RISK

@app.get("/monitoring/anomaly", summary="이상치 탐지", tags=["이상 탐지"])
async def detect_anomaly(line_id: str, sensor_id: str):
    base_url = "http://localhost:8000"
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.detect_anomaly(line_id, sensor_id)
        return result.__dict__

@app.get("/monitoring/explanation", summary="이상치 원인 설명", tags=["이상 탐지"])
async def explain_anomaly(task_id: int, line_id: str, sensor_id: str):
    base_url = "http://localhost:8000"
    anomaly_data = {"line_id": line_id, "sensor_id": sensor_id}
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.explain_anomaly(task_id, anomaly_data)
        return result.__dict__

@app.get("/monitoring/cause_candidates", summary="원인 후보군 생성", tags=["이상 탐지"])
async def generate_cause_candidates(task_id: int, line_id: str, sensor_id: str):
    base_url = "http://localhost:8000"
    anomaly_data = {"line_id": line_id, "sensor_id": sensor_id}
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.generate_cause_candidates(task_id, anomaly_data)
        return result.__dict__

@app.get("/monitoring/precursor", summary="이상징후 예측", tags=["이상 탐지"])
async def predict_precursor(task_id: int, line_id: str, sensor_id: str):
    base_url = "http://localhost:8000"
    anomaly_data = {"line_id": line_id, "sensor_id": sensor_id}
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.predict_precursor(task_id, anomaly_data)
        return result.__dict__

@app.get("/monitoring/risk_evaluation", summary="위험도 평가", tags=["위험 평가"])
async def evaluate_risk(task_id: int, line_id: str, sensor_id: str):
    base_url = "http://localhost:8000"
    anomaly_data = {"line_id": line_id, "sensor_id": sensor_id}
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.evaluate_risk(task_id, anomaly_data)
        return result.__dict__

@app.get("/monitoring/full_cycle", summary="전체 모니터링 사이클 실행", tags=["실시간 모니터링"])
async def run_full_monitoring_cycle(task_id: str = "112", line_id: str = "LINE_01", sensor_id: str = "TEMP_SENSOR_01"):
    base_url = "http://localhost:8000"
    async with FactoryMonitoringAgent(base_url) as agent:
        result = await agent.run_full_monitoring_cycle(task_id, line_id, sensor_id)
        return result

# uvicorn main:app --reload
