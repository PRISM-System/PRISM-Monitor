import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

class AnomalyType(Enum):
    TEMPERATURE_SPIKE = "temperature_spike"
    PRESSURE_ANOMALY = "pressure_anomaly"
    VIBRATION_ANOMALY = "vibration_anomaly"
    FLOW_RATE_ANOMALY = "flow_rate_anomaly"

@dataclass
class AnomalyResult:
    status: str
    anomaly_detected: bool
    anomaly_type: Optional[str] = None
    description: Optional[str] = None
    current_value: Optional[float] = None
    normal_range: Optional[tuple] = None
    updated_at: Optional[str] = None

@dataclass
class ExplanationResult:
    explanation: str
    confidence_score: float
    factors: List[str]

@dataclass
class CauseCandidateResult:
    candidates: List[Dict[str, Any]]
    primary_cause: str
    probability_scores: Dict[str, float]

@dataclass
class PrecursorResult:
    precursor_events: List[Dict[str, Any]]
    risk_level: str
    time_to_failure: Optional[int]

@dataclass
class RiskEvaluationResult:
    risk_score: float
    risk_level: str
    impact_assessment: str
    recommended_actions: List[str]

class FactoryMonitoringAgent:
    def __init__(self, base_url: str, session: Optional[aiohttp.ClientSession] = None):
        self.base_url = base_url.rstrip('/')
        self.session = session
        self._own_session = session is None
    
    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """API 요청을 수행하는 공통 메서드"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API 요청 실패: {method} {url} - {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            raise

    async def get_dashboard_data(self, task_id: str) -> Dict[str, Any]:
        """모니터링 대시보드 데이터 조회"""
        endpoint = f"/api/v1/task?{task_id}/monitoring/dashboard"
        
        logger.info(f"대시보드 데이터 조회: task_id={task_id}")
        return await self._make_request("GET", endpoint)

    async def get_pending_tasks(self, task_id: str) -> Dict[str, Any]:
        """대기 중인 태스크 조회"""
        endpoint = f"/api/v1/task?{task_id}/monitoring/status=pending"
        
        logger.info(f"대기 중인 태스크 조회: task_id={task_id}")
        return await self._make_request("GET", endpoint)

    async def detect_anomaly(self, line_id: str, sensor_id: str) -> AnomalyResult:
        """이상치 탐지 수행"""
        endpoint = "/api/v1/monitoring/event/detect"
        payload = {
            "field": f"{line_id}_or_{sensor_id}",
            "type": "LINE or SENSOR"
        }
        
        logger.info(f"이상치 탐지 시작: line_id={line_id}, sensor_id={sensor_id}")
        
        response = await self._make_request("POST", endpoint, json=payload)
        
        # 응답 데이터를 AnomalyResult 객체로 변환
        result_data = response.get("result", {})
        
        return AnomalyResult(
            status=result_data.get("status", "unknown"),
            anomaly_detected=result_data.get("anomalyDetected", False),
            anomaly_type=result_data.get("anomaly_type"),
            description=result_data.get("description"),
            current_value=result_data.get("current_value"),
            normal_range=(result_data.get("normal_range_min"), result_data.get("normal_range_max")),
            updated_at=result_data.get("updatedAt")
        )

    async def explain_anomaly(self, task_id: int, anomaly_data: Dict[str, Any]) -> ExplanationResult:
        """이상치 원인 설명 생성"""
        endpoint = "/api/v1/monitoring/event/explain"
        payload = {
            "taskId": task_id,
            "anomaly_data": anomaly_data
        }
        
        logger.info(f"이상치 원인 설명 생성: task_id={task_id}")
        
        response = await self._make_request("POST", endpoint, json=payload)
        result_data = response.get("result", {})
        
        return ExplanationResult(
            explanation=result_data.get("explanation", ""),
            confidence_score=result_data.get("confidence_score", 0.0),
            factors=result_data.get("contributing_factors", [])
        )

    async def generate_cause_candidates(self, task_id: int, anomaly_data: Dict[str, Any]) -> CauseCandidateResult:
        """원인 후보군 생성"""
        endpoint = "/api/v1/monitoring/event/cause-candidates"
        payload = {
            "taskId": task_id,
            "anomaly_data": anomaly_data
        }
        
        logger.info(f"원인 후보군 생성: task_id={task_id}")
        
        response = await self._make_request("POST", endpoint, json=payload)
        result_data = response.get("result", {})
        
        return CauseCandidateResult(
            candidates=result_data.get("candidates", []),
            primary_cause=result_data.get("primary_cause", ""),
            probability_scores=result_data.get("probability_scores", {})
        )

    async def predict_precursor(self, task_id: int, anomaly_data: Dict[str, Any]) -> PrecursorResult:
        """이상징후 예측"""
        endpoint = "/api/v1/monitoring/event/precursor"
        payload = {
            "taskId": task_id,
            "anomaly_data": anomaly_data
        }
        
        logger.info(f"이상징후 예측: task_id={task_id}")
        
        response = await self._make_request("POST", endpoint, json=payload)
        result_data = response.get("result", {})
        
        return PrecursorResult(
            precursor_events=result_data.get("precursor_events", []),
            risk_level=result_data.get("risk_level", "low"),
            time_to_failure=result_data.get("time_to_failure")
        )

    async def evaluate_risk(self, task_id: int, anomaly_data: Dict[str, Any]) -> RiskEvaluationResult:
        """위험도 평가"""
        endpoint = "/api/v1/monitoring/event/evaluate-risk"
        payload = {
            "taskId": task_id,
            "anomaly_data": anomaly_data
        }
        
        logger.info(f"위험도 평가: task_id={task_id}")
        
        response = await self._make_request("POST", endpoint, json=payload)
        result_data = response.get("result", {})
        
        return RiskEvaluationResult(
            risk_score=result_data.get("risk_score", 0.0),
            risk_level=result_data.get("risk_level", "low"),
            impact_assessment=result_data.get("impact_assessment", ""),
            recommended_actions=result_data.get("recommended_actions", [])
        )

    async def update_dashboard(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """대시보드 업데이트"""
        endpoint = "/api/v1/monitoring/dashboard/update"
        
        logger.info("대시보드 업데이트")
        return await self._make_request("POST", endpoint, json=dashboard_data)

    async def send_monitoring_output(self, task_id: str, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """모니터링 결과 전송"""
        endpoint = f"/api/v1/task?{task_id}/monitoring/output"
        
        logger.info(f"모니터링 결과 전송: task_id={task_id}")
        return await self._make_request("POST", endpoint, json=output_data)

    async def run_full_monitoring_cycle(self, task_id: str, line_id: str, sensor_id: str) -> Dict[str, Any]:
        """전체 모니터링 사이클 실행"""
        logger.info(f"전체 모니터링 사이클 시작: task_id={task_id}, line_id={line_id}, sensor_id={sensor_id}")
        
        results = {
            "task_id": task_id,
            "line_id": line_id,
            "sensor_id": sensor_id,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        try:
            # 1. 이상치 탐지
            logger.info("1단계: 이상치 탐지")
            anomaly_result = await self.detect_anomaly(line_id, sensor_id)
            results["steps"]["anomaly_detection"] = anomaly_result.__dict__
            
            if not anomaly_result.anomaly_detected:
                logger.info("이상치가 탐지되지 않음. 모니터링 종료")
                results["status"] = "normal"
                return results
            
            # 이상치가 탐지된 경우 계속 진행
            anomaly_data = {
                "anomaly_type": anomaly_result.anomaly_type,
                "description": anomaly_result.description,
                "current_value": anomaly_result.current_value,
                "normal_range": anomaly_result.normal_range,
                "line_id": line_id,
                "sensor_id": sensor_id
            }
            
            task_id_int = int(task_id) if task_id.isdigit() else 112  # 기본값
            
            # 2. 설명 생성
            logger.info("2단계: 이상치 원인 설명 생성")
            explanation_result = await self.explain_anomaly(task_id_int, anomaly_data)
            results["steps"]["explanation"] = explanation_result.__dict__
            
            # 3. 원인 후보군 생성
            logger.info("3단계: 원인 후보군 생성")
            cause_candidates_result = await self.generate_cause_candidates(task_id_int, anomaly_data)
            results["steps"]["cause_candidates"] = cause_candidates_result.__dict__
            
            # 4. 이상징후 예측
            logger.info("4단계: 이상징후 예측")
            precursor_result = await self.predict_precursor(task_id_int, anomaly_data)
            results["steps"]["precursor_prediction"] = precursor_result.__dict__
            
            # 5. 위험도 평가
            logger.info("5단계: 위험도 평가")
            risk_evaluation_result = await self.evaluate_risk(task_id_int, anomaly_data)
            results["steps"]["risk_evaluation"] = risk_evaluation_result.__dict__
            
            # 6. 대시보드 업데이트
            logger.info("6단계: 대시보드 업데이트")
            dashboard_data = {
                "task_id": task_id,
                "line_id": line_id,
                "sensor_id": sensor_id,
                "anomaly_detected": True,
                "risk_level": risk_evaluation_result.risk_level,
                "risk_score": risk_evaluation_result.risk_score,
                "primary_cause": cause_candidates_result.primary_cause,
                "recommended_actions": risk_evaluation_result.recommended_actions,
                "updated_at": datetime.now().isoformat()
            }
            
            dashboard_update_result = await self.update_dashboard(dashboard_data)
            results["steps"]["dashboard_update"] = dashboard_update_result
            
            # 7. 최종 결과 전송
            logger.info("7단계: 최종 결과 전송")
            final_output = {
                "is_anomaly": True,
                "anomaly_details": anomaly_result.__dict__,
                "normal_range": anomaly_result.normal_range,
                "related_data": {
                    "explanation": explanation_result.explanation,
                    "primary_cause": cause_candidates_result.primary_cause,
                    "risk_level": risk_evaluation_result.risk_level,
                    "recommended_actions": risk_evaluation_result.recommended_actions
                }
            }
            
            monitoring_output_result = await self.send_monitoring_output(task_id, final_output)
            results["steps"]["monitoring_output"] = monitoring_output_result
            
            results["status"] = "anomaly_detected"
            results["summary"] = {
                "anomaly_type": anomaly_result.anomaly_type,
                "risk_level": risk_evaluation_result.risk_level,
                "primary_cause": cause_candidates_result.primary_cause,
                "actions_required": len(risk_evaluation_result.recommended_actions) > 0
            }
            
            logger.info("전체 모니터링 사이클 완료")
            return results
            
        except Exception as e:
            logger.error(f"모니터링 사이클 중 오류 발생: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results

# 사용 예시
async def main():
    base_url = "http://your-api-server.com"  # 실제 API 서버 URL로 변경
    
    async with FactoryMonitoringAgent(base_url) as agent:
        # 전체 모니터링 사이클 실행
        result = await agent.run_full_monitoring_cycle(
            task_id="112",
            line_id="LINE_01", 
            sensor_id="TEMP_SENSOR_01"
        )
        
        print("모니터링 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 개별 단계별 실행도 가능
        # dashboard_data = await agent.get_dashboard_data("112")
        # pending_tasks = await agent.get_pending_tasks("112")
