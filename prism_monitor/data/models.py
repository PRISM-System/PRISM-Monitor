from pydantic import BaseModel
from typing import Optional, List, Union

class SensorData(BaseModel):
    sensor_id: int
    timestamp: str
    value: float

class Event(BaseModel):
    event_id: int
    description: str
    severity: str
    detected_at: str

# Request Models
class DashboardRequest(BaseModel):
    field: str  # line_id or sensor_id
    type: str   # LINE or SENSOR

class StatusPendingRequest(BaseModel):
    task_id: Optional[int]

class EventOutputRequest(BaseModel):
    result: dict

class MonitoringOutputRequest(BaseModel):
    taskId: int
    result: dict

class EventDetectRequest(BaseModel):
    start: str
    end: str

class EventExplainRequest(BaseModel):
    anomalyPeriods: dict

class CauseCandidatesRequest(BaseModel):
    anomalyPeriods: dict

class PrecursorRequest(BaseModel):
    lineId: int
    sensors: List[str]

class EvaluateRiskRequest(BaseModel):
    currentTemp: int

class DashboardUpdateRequest(BaseModel):
    field: str
    type: str
    status: str
    anomalyDetected: bool
    anomalyType: Optional[str]
    updatedAt: str

# Response Models
class DashboardResponseResult(BaseModel):
    status: str
    anomaly_detected: bool
    anomaly_type: Optional[str]
    updatedAt: str

class DashboardResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
    result: DashboardResponseResult

class StatusPendingTask(BaseModel):
    taskId: int
    instruction: str
    status: str

class StatusPendingResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
    tasks: List[StatusPendingTask]

class EventOutputResult(BaseModel):
    status: str
    anomalyDetected: bool
    description: str

class EventOutputResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
    result: EventOutputResult

class MonitoringOutputResult(BaseModel):
    status: str
    anomalyDetected: bool
    description: str

class MonitoringOutputResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
    taskId: int
    result: MonitoringOutputResult

class AnomalyPeriod(BaseModel):
    from_: str
    to: str
    class Config:
        fields = {'from_': 'from'}

class EventDetectResponse(BaseModel):
    anomalyPeriods: List[AnomalyPeriod]
    Value: Union[str, int]

class EventExplainResponse(BaseModel):
    explanation: str

class CauseCandidatesResponse(BaseModel):
    candidates: List[str]

class PrecursorResponse(BaseModel):
    precursor: str

class EvaluateRiskResponse(BaseModel):
    riskLevel: str
    message: str

class DashboardUpdateResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
