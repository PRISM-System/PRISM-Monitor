import os
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from prism_core.core.tools import (
    create_rag_search_tool,
    create_compliance_tool,
    create_memory_search_tool,
    ToolRegistry
)

class MonitoringToolSetup:
    """
    MPA Agent 전용 도구 설정 클래스
    """
    
    def __init__(
            self,
            prism_server_url: str = None,
            openai_base_url: str = None,
            openai_api_key: str = "EMPTY",
        ):
        # MPA 전용 설정
        self.prism_server_url = prism_server_url
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key

        self.client_id = "monitoring"
        self.class_prefix = "Monitoring"
        
        # 도구 레지스트리
        self.tool_registry = ToolRegistry()
        
        self.data_query_tool = None
        self.model_performance_tool = None
        self.prediction_tool = None
        self.anomaly_detection_tool = None
        self.compliance_tool = None

        # 공용 세션 (재시도 전략 적용)
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
    
    def setup_tools(self) -> ToolRegistry:
        """MPA 전용 도구들을 설정하고 등록"""
        self.anomaly_detect_tool = self.create_anomaly_detect_tool()
        self.tool_registry.register_tool(self.anomaly_detect_tool)
        self.anomaly_database_tool = self.create_anomaly_database_tool()
        self.tool_registry.register_tool(self.anomaly_database_tool)
        self.data_view_tool = self.create_data_view_tool()
        self.tool_registry.register_tool(self.data_view_tool)
        
        return self.tool_registry
        
    
    def create_anomaly_detect_tool(self):
        """데이터 조회 도구 생성"""
        from src.tools.anomaly_detect_tool import AnomalyDetectTool
        return AnomalyDetectTool(
            client_id=self.client_id
        )
    
    def create_anomaly_database_tool(self):
        """이상치 탐지 데이터베이스 도구 생성"""
        from src.tools.anomaly_database_tool import AnomalyDataBaseTool
        return AnomalyDataBaseTool(
            database_url=self.prism_server_url,
            client_id=self.client_id
        )
    
    def create_data_view_tool(self):
        """쿼리 -> SQL 도구 생성"""
        from src.tools.data_view_tool import DataViewTool
        return DataViewTool(
            database_url=self.prism_server_url,
            client_id=self.client_id
        )
    
    def create_anomaly_detect_tool(self):
        """모델 성능 측정 도구 생성"""
        # 모델 로드
        # 성능 메트릭 계산
        # 결과 비교
    
    def create_prediction_tool(self):
        """예측 도구 생성"""
        # 모델 선택
        # 예측 실행
        # 결과 검증
    
    def create_anomaly_detection_tool(self):
        """이상 탐지 도구 생성"""
        # 이상 탐지 알고리즘
        # 임계값 설정
        # 결과 필터링
    
    def create_compliance_tool(self):
        """Compliance 검증 도구 생성"""
        # 규정 기준 로드
        # 데이터 검증
        # 보고서 생성