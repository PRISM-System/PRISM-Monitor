
import os
import requests

from typing import Any, Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

from src.tools.monitoring_tool_setup import MonitoringToolSetup
from prism_core.core.agents import AgentManager, WorkflowManager
from prism_core.core.tools import ToolRegistry
from prism_core.core.llm import PrismLLMService
from prism_core.core.llm.schemas import Agent, AgentInvokeRequest, AgentResponse, LLMGenerationRequest

class MonitoringAgent:
    """
    제조 성능 분석 에이전트 - 수도 코드
    """
    
    def __init__(self):
        # 1. 기본 설정 로드
        self.load_configuration()
        
        # 2. 매니저 초기화
        self.initialize_managers()

        # 3. 에이전트 등록
        self.register_agent()
        
        # 4. 에이전트 도구 설정
        self.setup_agent_tools()
        
        # 5. 워크플로우 등록
        self.register_workflows()
        
        # 6. LLM 서비스 연결
        self.connect_llm_service()
    
    def load_configuration(self):
        """설정 파일 로드"""
        # .env-local에서 설정 로드
        # 데이터베이스 연결 정보
        # 모델 저장소 경로
        # 성능 임계값 등
        from dotenv import load_dotenv
        load_dotenv()
        self.agent_name = 'Monitoring-Agent'
        self.openai_base_url = os.getenv("LLM_URL")
        self.api_key = os.getenv("PRISM_CORE_SERVER_LLM_API_KEY", "EMPTY")
        self.prism_server_url = os.getenv("PRISM_CORE_SERVER_URL")
        self.prism_server_llm_url = os.getenv("PRISM_CORE_SERVER_LLM_URL")
        self.llm_model_name = os.getenv("LLM_MODEL_NAME")

        retry_strategy = Retry(
            total=2,  # 재시도 횟수 (즉, 최초 1회 + 재시도 2회 = 최대 3회 시도)
            backoff_factor=1,  # 지수 백오프 (1s, 2s, 4s)
            status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 상태 코드
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)


    
    def initialize_managers(self):
        """매니저 초기화"""
        # AgentManager 생성
        # WorkflowManager 생성
        # ToolRegistry 생성
        self.agent_manager = AgentManager()
        self.workflow_manager = WorkflowManager()
        self.monitoring_tool_setup = MonitoringToolSetup()
        self.tool_registry = self.monitoring_tool_setup.setup_tools()

    def setup_agent_tools(self):
        """에이전트 전용 도구 설정"""
        # 1. 데이터 조회 도구 (DataQueryTool)
        # 2. 모델 성능 측정 도구 (ModelPerformanceTool)
        # 3. 예측 도구 (PredictionTool)
        # 4. 이상 탐지 도구 (AnomalyDetectionTool)
        # 5. Compliance 검증 도구 (ComplianceTool)
        url = urljoin(self.prism_server_url, 'api/tools/register')
        for tool_name, tool in self.tool_registry.tools.items():
            self.session.post(url, json=tool.info())
        self.agent_manager.set_tool_registry(self.tool_registry)
        self.workflow_manager.set_tool_registry(self.tool_registry)

    def connect_llm_service(self):
        """LLM 서비스 연결"""
        # OpenAI 또는 기타 LLM 서비스 연결
        self.llm = PrismLLMService(
            model_name=self.llm_model_name,
            tool_registry=self.tool_registry,
            agent_name=self.agent_name,
            llm_service_url=self.prism_server_url,
            openai_base_url=self.prism_server_llm_url,
            api_key=self.api_key
        )

    def register_agent(self) -> None:
        """모니터링 에이전트를 등록합니다."""
        agent_config = {
            "name": "monitoring_agent",
            "description": "제조 공정 이상 탐지 및 모니터링 에이전트",
            "role_prompt": "당신은 제조 공정의 이상 탐지 및 모니터링을 담당하는 에이전트입니다. 실시간 데이터 분석과 이상 탐지를 통해 제조 공정의 안정성을 유지하는 것이 당신의 주요 임무입니다.",
            "tools": []  # 초기에는 빈 리스트
        }
        url = urljoin(self.prism_server_url, 'api/agents')
        self.session.post(url, json=agent_config)

        


    async def monitor(
        self, 
        prompt: str, 
        user_id: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        use_tools: bool = True,
        max_tool_calls: int = 3,
        extra_body: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        메인 오케스트레이션 메서드
        
        Args:
            prompt: 사용자 요청
            user_id: 사용자 ID (선택사항)
            max_tokens: 최대 토큰 수 (기본값: 1024)
            temperature: 생성 온도 (기본값: 0.7)
            stop: 중단 시퀀스 (기본값: None)
            use_tools: 도구 사용 여부 (기본값: True)
            max_tool_calls: 최대 도구 호출 수 (기본값: 3)
            extra_body: 추가 OpenAI 호환 옵션 (기본값: None)
            
        Returns:
            AgentResponse: 오케스트레이션 결과
        """
        # Ensure agent is registered
        if not getattr(self, '_agent', None):
            self.register_agent()

        # Prepare context
        context = {
            "user_id": user_id,
            "prompt": prompt,
            "timestamp": self._get_timestamp()
        }

        # Create agent invoke request with all parameters
        request = AgentInvokeRequest(
            prompt=prompt,
            user_id=user_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            use_tools=use_tools,
            max_tool_calls=max_tool_calls,
            extra_body=extra_body
        )

        # Invoke agent with tools using the correct method
        response = await self.llm.invoke_agent(self._agent, request)
        
        # Add context information to metadata
        if response.metadata is None:
            response.metadata = {}
        response.metadata.update(context)
        
        # Save conversation to memory if user_id is provided
        
        return response

        
    
    def register_workflows(self):
        """워크플로우 등록"""
        # 1. 데이터 분석 워크플로우
        # 2. 모델 평가 워크플로우
        # 3. 예측 워크플로우
        # 4. 이상 탐지 워크플로우
        # 5. Compliance 검증 워크플로우
    
    # === 핵심 비즈니스 로직 메서드들 ===
    
    def analyze_manufacturing_performance(self, time_range: dict, equipment_ids: list):
        """
        제조 성능 분석 메인 워크플로우
        """
        # 1. 데이터 조회
        data = self.query_manufacturing_data(time_range, equipment_ids)
        
        # 2. 모델 성능 측정
        model_performance = self.evaluate_model_performance(data)
        
        # 3. 최고 성능 모델 선택
        best_model = self.select_best_model(model_performance)
        
        # 4. 미래 예측
        predictions = self.predict_future_performance(best_model, data)
        
        # 5. 이상 탐지
        anomalies = self.detect_anomalies(predictions, data)
        
        # 6. Compliance 검증
        compliance_report = self.verify_compliance(anomalies, predictions)
        
        return {
            "data": data,
            "model_performance": model_performance,
            "best_model": best_model,
            "predictions": predictions,
            "anomalies": anomalies,
            "compliance_report": compliance_report
        }
    
    def query_manufacturing_data(self, time_range: dict, equipment_ids: list):
        """제조 데이터 조회"""
        # 1. 데이터베이스에서 센서 데이터 조회
        # 2. 설비 상태 데이터 조회
        # 3. 품질 측정 데이터 조회
        # 4. 환경 데이터 조회
        # 5. 데이터 전처리 및 정규화
    
    def evaluate_model_performance(self, data: dict):
        """모델 성능 평가"""
        # 1. 보유 모델 목록 조회
        # 2. 각 모델에 대해 성능 측정
        # 3. 정확도, 정밀도, 재현율 계산
        # 4. 교차 검증 수행
        # 5. 성능 점수 정렬
    
    def select_best_model(self, model_performance: dict):
        """최고 성능 모델 선택"""
        # 1. 성능 점수 기준으로 정렬
        # 2. 임계값 이상 모델 필터링
        # 3. 안정성 점수 고려
        # 4. 최종 모델 선택
    
    def predict_future_performance(self, model: dict, data: dict):
        """미래 성능 예측"""
        # 1. 모델 로드
        # 2. 예측 기간 설정
        # 3. 입력 데이터 준비
        # 4. 예측 실행
        # 5. 예측 결과 후처리
    
    def detect_anomalies(self, predictions: dict, historical_data: dict):
        """이상 탐지"""
        # 1. 예측값과 실제값 비교
        # 2. 통계적 이상 탐지
        # 3. 머신러닝 기반 이상 탐지
        # 4. 이상 구간 식별
        # 5. 위험도 점수 계산
    
    def verify_compliance(self, anomalies: dict, predictions: dict):
        """Compliance 검증"""
        # 1. 규정 준수 기준 로드
        # 2. 예측 결과 검증
        # 3. 이상 구간 규정 준수 확인
        # 4. 위험도 평가
        # 5. 권장사항 생성
    
    # === 워크플로우 실행 메서드들 ===
    
    def execute_data_analysis_workflow(self, parameters: dict):
        """데이터 분석 워크플로우 실행"""
        # 1. 워크플로우 정의
        # 2. 단계별 실행
        # 3. 결과 수집
        # 4. 오류 처리
    
    def execute_model_evaluation_workflow(self, parameters: dict):
        """모델 평가 워크플로우 실행"""
        # 1. 모델 목록 조회
        # 2. 성능 측정
        # 3. 결과 비교
        # 4. 최적 모델 선택
    
    def execute_prediction_workflow(self, parameters: dict):
        """예측 워크플로우 실행"""
        # 1. 모델 로드
        # 2. 데이터 전처리
        # 3. 예측 실행
        # 4. 결과 검증
    
    def execute_anomaly_detection_workflow(self, parameters: dict):
        """이상 탐지 워크플로우 실행"""
        # 1. 데이터 분석
        # 2. 이상 탐지 알고리즘 실행
        # 3. 결과 필터링
        # 4. 위험도 평가
    
    def execute_compliance_verification_workflow(self, parameters: dict):
        """Compliance 검증 워크플로우 실행"""
        # 1. 규정 기준 로드
        # 2. 데이터 검증
        # 3. 규정 준수 확인
        # 4. 보고서 생성

    def _get_timestamp(self) -> str:
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().isoformat()