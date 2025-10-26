import requests

DESCRIPTION = "제조 및 산업 도메인 전문 AI 어시스턴트입니다. 자동차, 배터리, 반도체 등 다양한 산업 현장의 데이터에 대한 사용자의 자연어 요청을 이해합니다. 주된 임무는 사용자의 요청에 맞는 대시보드를 반환하고, 공정의 이상 상태를 탐지 및 분석하는 것입니다. 이를 위해 사용자의 의도를 파악하고 가장 관련성 높은 산업 공정 데이터를 식별합니다."
ROLE_PROMPT = """당신은 제조 및 산업 데이터 분석을 전문으로 하는 AI 어시스턴트입니다. 당신의 핵심 임무는 사용자의 요청에 따라 관련 공정의 대시보드를 반환하고, 데이터에 기반한 이상 상태 탐지 및 심층 분석을 수행하는 것입니다. 이를 위해 사용자의 자연어 쿼리를 분석하여 관련 산업 공정을 식별합니다."""

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
    def __init__(self, url: str, agent_name: str="MonitoringAgent"):
        self.url = url
        self.agent_name = agent_name
        self.tools = TOOLS

        self.init()

    def register_agent(self):
        payload = {
            "name": self.agent_name,
            "description": DESCRIPTION,
            "role_prompt": ROLE_PROMPT,
            "tools": list(self.tools.keys())
        }
        response = requests.post(f"{self.url}/core/api/agents/", json=payload)
        print('register_agent', response.status_code, response.content)
        return 
    
    def delete_agent(self):
        response = requests.delete(f"{self.url}/core/api/agents/{self.agent_name}/")
        print('delete_agent', response.status_code, response.content)
        return 
    
    
    def delete_tools(self):
        for tool in self.tools:
            response = requests.delete(f"{self.url}/core/api/tools/{tool}/")
            print('delete_tools', response.status_code, response.content)
        return 
    
    def register_tools(self):
        for tool in self.tools.values():
            response = requests.post(f"{self.url}/core/api/tools/", json=tool)
            print('register_tools', response.status_code, response.content)
        return

    def init(self):
        try:
            self.delete_agent()
        except Exception as e:
            print(f"Agent deletion failed: {e}")
        try:
            self.delete_tools()
        except Exception as e:
            print(f"Tool deletion failed: {e}")
        self.register_tools()
        self.register_agent()