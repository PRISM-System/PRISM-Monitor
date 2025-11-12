import requests

DESCRIPTION = "제조 공정의 현재 상태를 모니터링하고 이상치를 탐지하는 모니터링 에이전트"
ROLE_PROMPT = """당신은 모니터링 에이전트입니다. 제조 공정의 현재 상태를 8단계 워크플로우로 종합 분석합니다: (1) 자연어 쿼리를 SQL로 변환, (2) 이상치 탐지 및 분류, (3) 근본 원인 분석, (4) 작업 후보 생성, (5) 미래 상태 예측, (6) 위험도 평가, (7) 미래 위험 예측, (8) 종합 분석 결과 제시. 대시보드 데이터를 기반으로 공정의 정상/비정상 상태를 판단하고, 예측 및 자율제어 에이전트와 협업하여 제조 공정의 품질과 안정성을 보장하세요."""

TOOLS = {
    "dashboard":{
        "name": "dashboard",
        "description": "Tool for retrieving monitoring dashboard data",
        "parameters_schema": {
            "type": "object",
            "properties": {
            "param1": {
                "type": "string"
            }
            }
        },
        "tool_type": "function"
    },
    "analysis":{
        "name": "analysis",
        "description": "Tool for analyzing industrial process data",
        "parameters_schema": {
            "type": "object",
            "properties": {
            "paramA": {
                "type": "string"
            }
            }
        },
        "tool_type": "function"
    }
}

class MonitoringAgent:
    def __init__(self, prism_core_url: str = None, agent_name: str="monitoring_agent"):
        if prism_core_url is None:
            raise ValueError("prism_core_url must be provided")
        self.prism_core_url = prism_core_url
        self.agent_name = agent_name

        # 에이전트 등록
        self.register_agent()

    def register_agent(self):
        """PRISM-Core에 에이전트를 등록합니다."""
        try:
            agent_data = {
                "name": self.agent_name,
                "description": DESCRIPTION,
                "role_prompt": ROLE_PROMPT,
                "tools": []  # 필요시 나중에 추가
            }

            response = requests.post(
                f"{self.prism_core_url}/core/api/agents",
                json=agent_data,
                timeout=5
            )

            if response.status_code == 200:
                print(f"monitoring_agent registered successfully to PRISM-Core at {self.prism_core_url}")
                return True
            else:
                print(f"monitoring_agent registration failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Agent registration failed: {e}")
            return False