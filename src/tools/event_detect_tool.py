"""
Compliance Tool

안전 규정 및 법규 준수 여부를 검증하는 Tool입니다.
LLM을 통한 지능형 규정 준수 분석을 제공합니다.
PRISM Core의 공통 설정을 사용합니다.
"""

import requests
import json

from typing import Dict, Any, List, Optional

from prism_core.core.tools.base import BaseTool
from prism_core.core.tools.schemas import ToolRequest, ToolResponse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI 라이브러리가 설치되지 않았습니다. 기본 키워드 분석만 사용 가능합니다.")


class AnomalyDetectTool(BaseTool):
    """
    안전 규정 및 법규 준수 여부를 검증하는 Tool
    
    기능:
    - 제안된 조치의 안전 규정 준수 여부 검증
    - 관련 법규 및 사내 규정 매칭
    - LLM을 통한 지능형 준수 여부 판단
    - 준수 여부에 따른 권장사항 제공
    """
    
    def __init__(self, 
                 weaviate_url: Optional[str] = None,
                 openai_base_url: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 client_id: str = "monitoring",
                 class_prefix: str = "Monitoring"):
        super().__init__(
            name="anomaly_detect",
            description="주어진 시간 구간에 이상 이벤트가 있었는지 탐지"
            parameters_schema={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "시작 시간 (ISO 8601 형식)"},
                    "end": {"type": "string", "description": "종료 시간 (ISO 8601 형식)"}
                },
                "required": ["action"]
            }
        )
        # 에이전트별 설정 또는 기본값 사용
        self._weaviate_url = weaviate_url or settings.WEAVIATE_URL
        self._openai_base_url = openai_base_url or settings.VLLM_OPENAI_BASE_URL
        self._openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self._model_name = model_name or settings.DEFAULT_MODEL
        self._client_id = client_id
        
        # 에이전트별 클래스명 설정
        self._class_compliance = f"{class_prefix}Compliance"
        
        # OpenAI 클라이언트 초기화
        self._openai_client = None
        if OPENAI_AVAILABLE:
            self._initialize_openai()

    def _initialize_openai(self) -> None:
        """OpenAI 클라이언트 초기화"""
        try:
            self._openai_client = OpenAI(
                base_url=self._openai_base_url,
                api_key=self._openai_api_key
            )
            print("✅ OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            print(f"⚠️  OpenAI 클라이언트 초기화 실패: {str(e)}")
            self._openai_client = None

    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Tool 실행"""
        try:
            params = request.parameters
            action = params["action"]
            context = params.get("context", "")
            user_id = params.get("user_id", "")
            
            # 1. 관련 규정 검색
            compliance_docs = await self._search_compliance_rules(action, context)
            
            # 2. LLM을 통한 준수 여부 분석
            compliance_analysis = await self._analyze_compliance_with_llm(action, context, compliance_docs)
            
            # 3. 결과 반환
            return ToolResponse(
                success=True,
                data={
                    "action": action,
                    "compliance_checked": True,
                    "compliance_status": compliance_analysis["status"],
                    "related_rules": compliance_analysis["related_rules"],
                    "recommendations": compliance_analysis["recommendations"],
                    "risk_level": compliance_analysis["risk_level"],
                    "reasoning": compliance_analysis["reasoning"],
                    "domain": "compliance"
                }
            )
                
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"규정 준수 검증 실패: {str(e)}"
            )

    async def _search_compliance_rules(self, action: str, context: str) -> List[Dict[str, Any]]:
        """관련 규정 검색"""
        try:
            # 직접 Weaviate API 호출
            search_query = f"안전 규정 준수: {action} {context}"
            
            response = requests.post(
                f"{self._weaviate_url}/v1/objects/{self._class_compliance}/search",
                json={
                    "query": search_query,
                    "limit": 5
                },
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  규정 검색 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"⚠️  규정 검색 중 오류: {str(e)}")
            return []

    async def _analyze_compliance_with_llm(self, action: str, context: str, compliance_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """LLM을 통한 준수 여부 분석"""
        if not self._openai_client:
            # OpenAI가 없는 경우 기본 키워드 분석
            return self._basic_compliance_analysis(action, context, compliance_docs)
        
        try:
            # 관련 규정 정보 구성
            rules_text = ""
            for doc in compliance_docs:
                rules_text += f"- {doc.get('content', '')}\n"
            
            # LLM 프롬프트 구성
            prompt = f"""
다음 조치가 안전 규정을 준수하는지 분석해주세요:

**검토할 조치:**
{action}

**조치 맥락:**
{context}

**관련 안전 규정:**
{rules_text}

다음 JSON 형식으로 응답해주세요:
{{
    "status": "compliant|non_compliant|requires_review",
    "risk_level": "low|medium|high",
    "related_rules": ["규정1", "규정2"],
    "recommendations": ["권장사항1", "권장사항2"],
    "reasoning": "분석 근거"
}}
"""
            
            # LLM 호출
            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "당신은 안전 규정 준수 전문가입니다. 정확하고 신중한 분석을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 응답 파싱
            result_text = response.choices[0].message.content
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 분석 반환
                return self._basic_compliance_analysis(action, context, compliance_docs)
                
        except Exception as e:
            print(f"⚠️  LLM 분석 실패: {str(e)}")
            return self._basic_compliance_analysis(action, context, compliance_docs)

    def _basic_compliance_analysis(self, action: str, context: str, compliance_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """기본 키워드 기반 준수 여부 분석"""
        # 위험 키워드 체크
        danger_keywords = ["위험", "폭발", "화재", "독성", "고압", "고온", "전기"]
        safety_keywords = ["안전", "보호", "점검", "허가", "절차", "규정"]
        
        action_lower = action.lower()
        context_lower = context.lower()
        
        # 위험도 평가
        danger_count = sum(1 for keyword in danger_keywords if keyword in action_lower or keyword in context_lower)
        safety_count = sum(1 for keyword in safety_keywords if keyword in action_lower or keyword in context_lower)
        
        # 상태 결정
        if danger_count > safety_count:
            status = "non_compliant"
            risk_level = "high"
        elif safety_count > danger_count:
            status = "compliant"
            risk_level = "low"
        else:
            status = "requires_review"
            risk_level = "medium"
        
        return {
            "status": status,
            "risk_level": risk_level,
            "related_rules": [doc.get("title", "규정") for doc in compliance_docs[:3]],
            "recommendations": [
                "안전 규정을 다시 한번 확인하세요",
                "필요시 안전 담당자와 상의하세요",
                "작업 허가서를 발급받으세요"
            ],
            "reasoning": f"기본 키워드 분석 결과: 위험 키워드 {danger_count}개, 안전 키워드 {safety_count}개"
        } 