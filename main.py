from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel

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

# uvicorn main:app --reload
