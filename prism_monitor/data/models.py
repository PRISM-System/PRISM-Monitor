from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union, Any
from typing import Union, Literal
from typing_extensions import Annotated

#/api/v1/task/{task_id}/monitoring/dashboard
class DashboardResponse(BaseModel):
    class Result(BaseModel):
        status: Literal["정상", "비정상"] = Field("비정상", description="Current status of the line or sensor")
        anomaly_detected: bool = Field(True, description="Whether an anomaly has been detected")
        anomaly_type: Literal[None,"temperature_spike", "temperature_drop", "sensor_failure"] = Field("temperature_spike", description="Type of anomaly detected")
        updatedAt: str = Field("2023-10-01T12:00:00Z", description="Last updated timestamp in ISO 8601 format")
    isSuccess: bool = True
    code: int = 200
    message: str = "대시보드 데이터 조회 성공"
    result: Result = Result()

#/api/v1/task/{task_id}/monitoring/status=pending
class StatusPendingResponse(BaseModel):
    class Task(BaseModel):
        taskId: int = 112
        instruction: str = "이상 탐지 실행 중"
        status: Literal["pending", "running", "completed"] = "pending"
    isSuccess: bool = True
    code: int = 200
    message: str = "이상 탐지 실행 중"
    tasks: List[Task] = [Task()]

#/api/v1/monitoring/event/output
class EventOutputRequest(BaseModel):
    class Result(BaseModel):
        status: Literal["complete", "failed"] = "complete"
        anomalyDetected: bool = True
        description: str = "라인2-5 온도 이상 감지"
    result: Result = Result()

class EventOutputResponse(BaseModel):
    isSuccess: bool = True
    code: int = 201
    message: str = "결과 전달 완료"

#/api/v1/task/{task_id}/monitoring/output
class MonitoringOutputRequest(BaseModel):
    class Result(BaseModel):
        status: Literal["complete", "failed"] = "complete"
        anomalyDetected: bool = True
        description: str = "라인2-5 온도 이상 감지"
    result: Result = Result()

class MonitoringOutputResponse(BaseModel):
    isSuccess: bool = True
    code: int = 201
    message: str = "결과 전달 완료" 

#/api/v1/monitoring/event/detect
class EventDetectRequest(BaseModel):
    taskId: str = 'TASK_0001'
    start: str = Field("2024-01-01T12:00:00Z", description="Start time in ISO 8601 format")
    end: str = Field("2024-02-01T12:30:00Z", description="End time in ISO 8601 format")

class EventDetectResponse(BaseModel):
    class Result(BaseModel):
        status: Literal["complete", "failed"] = "complete"
        anomalies: bool = True
        drift_detected: bool = False
    result: Result = Result()


#/api/v1/monitoring/event/explain
class EventExplainRequest(BaseModel):
    taskId: str = 'TASK_0001'

class EventExplainResponse(BaseModel):
    explain: str

#/api/v1/monitoring/event/cause-candidates
class CauseCandidatesRequest(BaseModel):
    taskId: str = 'TASK_0001'

class CauseCandidatesResponse(BaseModel):
    causeCandidates: str

#/api/v1/monitoring/event/precursor
class PrecursorRequest(BaseModel):
    taskId: str = 'TASK_0001'
    start: str = Field("2024-01-01T12:00:00Z", description="Start time in ISO 8601 format")
    end: str = Field("2024-02-01T12:30:00Z", description="End time in ISO 8601 format")

class PrecursorResponse(BaseModel):
    class Summary(BaseModel):
        predicted_value: float = 0.0
        is_anomaly: bool = False
    summary: Summary = Summary()

#/api/v1/monitoring/event/evaluate-risk
class EvaluateRiskRequest(BaseModel):
    taskId: str = 'TASK_0001'
    topk: int = 5

class EvaluateRiskResponse(BaseModel):
    # totalCandidates: int
    # passedCandidates: int
    # failedCandidates: int
    # riskLevel: Literal["HIGH", "LOW", "MEDIUM"]
    # complianceStatus: bool
    # class recommendedAction(BaseModel):
    #     actionName: str
    #     TotalScore: float
    # recommendedActions: List[recommendedAction]
    eventEvaluation: str
    predictionEvaluation: str

#/api/v1/monitoring/dashboard/update
class DashboardUpdateRequest(BaseModel):
    field: Literal["line_id", "sensor_id"] = "line_id"
    type: Literal["LINE", "SENSOR"] = "LINE"
    status: Literal["정상", "비정상"] = "비정상"
    anomalyDetected: bool = True
    anomalyType: Optional[Literal["temperature_spike", "temperature_drop", "sensor_failure"]] = "temperature_spike"
    updatedAt: str = "2023-10-01T12:00:00Z"

class DashboardUpdateResponse(BaseModel):
    isSuccess: bool = True
    code: int = 200
    message: str = "대시보드 업데이트 완료"

class WorkflowStartResponse(BaseModel):
    class Result(BaseModel):
        detectResult: dict = {}
        explainResult: dict = {}
        causeCandidatesResult: dict = {}
        precursorResult: dict = {}
        evaluateRiskResult: dict = {}
    result: Result = Result()

class WorkflowStartRequest(BaseModel):
    taskId: str = 'TASK_0001'
    query: str = ''

class RealTimeMonitoringResponse(BaseModel):
    visJson: dict = {}
