from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union, Any
from typing import Union, Literal
from typing_extensions import Annotated


class Query2SQLRequest(BaseModel):
    taskId: str = 'TASK_0001'
    query: str = Field(..., description="User query in natural language")

class Query2SQLResponse(BaseModel):
    result: dict = {}

#/api/v1/monitoring/event/detect
class EventDetectRequest(BaseModel):
    taskId: str = 'TASK_0001'
    targetProcess: str = Field("automotive_assembly", description="Target industrial process")
    start: str = Field("2025-05-01T12:00:00Z", description="Start time in ISO 8601 format")
    end: str = Field("2025-05-01T13:00:00Z", description="End time in ISO 8601 format")

class EventDetectResponse(BaseModel):
    result: dict = {}

#/api/v1/monitoring/event/explain
class EventExplainRequest(BaseModel):
    taskId: str = 'TASK_0001'
    targetProcess: str = Field("automotive_assembly", description="Target industrial process")
    eventDetectAnalysis: str = Field("Detected anomalies analysis data", description="Analysis data from event detection")
    targetProcess: str = Field("automotive_assembly", description="Target industrial process")

class EventExplainResponse(BaseModel):
    result: str = ''

#/api/v1/monitoring/event/cause-candidates
class CauseCandidatesRequest(BaseModel):
    taskId: str = 'TASK_0001'
    targetProcess: str = Field("automotive_assembly", description="Target industrial process")
    eventDetectAnalysis: str = Field("Detected anomalies analysis data", description="Analysis data from event detection")

class CauseCandidatesResponse(BaseModel):
    result: str = ''

#/api/v1/monitoring/event/precursor
class PrecursorRequest(BaseModel):
    taskId: str = 'TASK_0001'
    targetProcess: str = Field("automotive_assembly", description="Target industrial process")
    start: str = Field("2025-05-01T12:00:00Z", description="Start time in ISO 8601 format")
    end: str = Field("2025-05-01T13:00:00Z", description="End time in ISO 8601 format")

class PrecursorResponse(BaseModel):
    result: list = []

#/api/v1/monitoring/event/evaluate-risk
class EvaluateRiskRequest(BaseModel):
    taskId: str = 'TASK_0001'
    eventDetectAnalysis: str = Field("Detected anomalies analysis data", description="Analysis data from event detection")

class EvaluateRiskResponse(BaseModel):
    result: dict = {}

class PredictionRiskRequest(BaseModel):
    taskId: str = 'TASK_0001'
    taskInstructions: str = Field("Instructions related to risk prediction", description="Instructions for risk prediction")

class PredictionRiskResponse(BaseModel):
    result: dict = {}

class DashboardResponse(BaseModel):
    dashboard: dict = {}

class WorkflowStartResponse(BaseModel):
    summary: str = ''
    monitored_timeseries: dict = {}
    result: str = ''

class WorkflowStartRequest(BaseModel):
    taskId: str = 'TASK_0001'
    query: str = ''
    class Config:
        extra = "allow"  # 정의되지 않은 키도 허용

class RealTimeMonitoringResponse(BaseModel):
    visJson: dict = {}
