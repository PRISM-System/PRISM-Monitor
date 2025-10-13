# instruction_rf_client.py
import json
import uuid
from typing import Dict, List, Any
from datetime import datetime
import requests


class InstructionRefinementClient:
    """
    vLLM(OpenAI 호환) 서버에 직접 붙는 클라이언트.
    - server_url 은 반드시 '/v1' 까지의 베이스 URL 이어야 합니다. (예: http://127.0.0.1:8000/v1)
    - test_api_connection() 은 GET /v1/models 로 헬스체크합니다.
    - _call_llm_api() 는 POST /v1/completions 로 호출합니다. (prompt 기반)
    """

    def __init__(
        self,
        server_url: str,
        timeout: int = 60,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",  # ← 기본 모델 + 외부 주입 허용
    ):
        self.base = server_url.rstrip('/')        # 예: http://127.0.0.1:8000/v1
        self.server_url = self.base               # 하위 호환(기존 코드가 참조하는 경우 대비)
        self.timeout = timeout
        self.model = model                        # ← 전달받은 모델 사용
        # (선택) prompt_manager 연동이 필요하면 외부에서 주입하거나 이 위치에서 초기화하세요.
        self.prompt_manager = getattr(self, 'prompt_manager', None)

    def _url(self, path: str) -> str:
        """베이스 URL 과 상대 경로를 안전하게 결합"""
        return f"{self.base}/{path.lstrip('/')}"

    # ─────────────────────────────────────────────────────────────
    # 헬스 체크
    # ─────────────────────────────────────────────────────────────
    def test_api_connection(self) -> Dict[str, Any]:
        """
        서버 연결/응답 테스트 (가벼운 GET /v1/models)
        """
        try:
            r = requests.get(self._url("models"), timeout=self.timeout)
            r.raise_for_status()
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ─────────────────────────────────────────────────────────────
    # LLM 호출 (vLLM /v1/completions, prompt 방식)
    # ─────────────────────────────────────────────────────────────
    def _call_llm_api(self, prompt: str) -> str:
        """
        vLLM(OpenAI 호환) Completions 엔드포인트 호출
        - POST /v1/completions
        """
        payload = {
            "model": self.model,          # ← 하드코딩 제거, 생성자에서 받은 모델 사용
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.1
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self._url("completions"),
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # vLLM completions 응답 파싱
            if "choices" in result and result["choices"]:
                # 일반적으로 choices[0]["text"] 에 결과가 옴
                return result["choices"][0].get("text", str(result))
            # 혹시 다른 키로 오는 경우 대비
            return result.get("text", str(result))

        except requests.exceptions.Timeout:
            raise Exception(f"API 호출 타임아웃 ({self.timeout}초)")
        except requests.exceptions.ConnectionError:
            raise Exception(f"서버 연결 실패: {self._url('completions')}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP 에러: {e.response.status_code} - {self._url('completions')}")
        except Exception as e:
            raise Exception(f"API 호출 중 오류 발생: {str(e)}")

    # ─────────────────────────────────────────────────────────────
    # 사용 시나리오: 자연어 쿼리 → 구조화 JSON
    # ─────────────────────────────────────────────────────────────
    def refine_instruction(self, user_query: str) -> Dict[str, Any]:
        """
        자연어 쿼리를 구조화된 JSON 명령어로 변환
        (LLM이 JSON 을 내도록 강제하는 프롬프트)
        """
        prompt = f"""Convert this manufacturing query to JSON format:
Query: {user_query}

Required JSON format:
{{
  "instruction_id": "inst_[timestamp]",
  "original_query": "{user_query}",
  "intent_type": "ANOMALY_CHECK or PREDICTION or CONTROL or INFORMATION or OPTIMIZATION",
  "priority": "HIGH or MEDIUM or LOW",
  "target": {{
    "equipment_id": "equipment identifier",
    "parameter": "parameter name",
    "process": "process name"
  }},
  "tasks": [
    {{
      "task_id": "task_001",
      "agent": "MONITORING or PREDICTION or CONTROL or ORCHESTRATION",
      "action": "specific action",
      "parameters": {{}},
      "dependencies": [],
      "expected_output": "output description"
    }}
  ]
}}

Respond with valid JSON only:"""

        try:
            response_content = self._call_llm_api(prompt)

            # LLM 응답에서 첫 번째 완전한 JSON 객체 추출
            json_start = response_content.find('{')
            if json_start == -1:
                return self._create_fallback_instruction(user_query, "JSON 파싱 실패: 여는 중괄호를 찾지 못함")

            brace_count = 0
            json_end = json_start
            for i, ch in enumerate(response_content[json_start:], json_start):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end <= json_start:
                return self._create_fallback_instruction(user_query, "완전한 JSON 구조를 찾을 수 없음")

            json_content = response_content[json_start:json_end]
            try:
                parsed = json.loads(json_content)
            except json.JSONDecodeError:
                return self._create_fallback_instruction(user_query, f"JSON 파싱 실패: {json_content[:120]}...")

            # 필수 필드 보강
            if "instruction_id" not in parsed:
                parsed["instruction_id"] = f"inst_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            if "original_query" not in parsed:
                parsed["original_query"] = user_query

            # (선택) 스키마 검증
            if self.prompt_manager:
                try:
                    errs = self.prompt_manager.validate_output_schema(parsed)
                    if errs:
                        # 검증 경고만 출력 (치명적 오류로 간주하지 않음)
                        print(f"⚠️  스키마 검증 경고: {', '.join(errs)}")
                except Exception:
                    pass

            return parsed

        except json.JSONDecodeError as e:
            return self._create_fallback_instruction(user_query, f"JSON 디코딩 오류: {str(e)}")
        except Exception as e:
            return self._create_fallback_instruction(user_query, f"명령어 변환 중 오류: {str(e)}")

    # ─────────────────────────────────────────────────────────────
    # fallback / batch / validate
    # ─────────────────────────────────────────────────────────────
    def _create_fallback_instruction(self, user_query: str, error_msg: str) -> Dict[str, Any]:
        """오류 발생 시 기본 명령어 구조 생성"""
        return {
            "instruction_id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "original_query": user_query,
            "intent_type": "INFORMATION",
            "priority": "MEDIUM",
            "target": {},
            "tasks": [
                {
                    "task_id": "fallback_task",
                    "agent": "ORCHESTRATION",
                    "action": "manual_review_required",
                    "parameters": {"error": error_msg, "query": user_query},
                    "dependencies": [],
                    "expected_output": "manual_instruction"
                }
            ],
            "context_requirements": {
                "historical_data": False,
                "real_time_data": False,
                "external_knowledge": False,
                "simulation_needed": False
            },
            "constraints": {
                "safety_requirements": ["manual_review_required"],
                "regulatory_compliance": []
            },
            "error": error_msg
        }

    def batch_refine(self, queries: List[str]) -> List[Dict[str, Any]]:
        """여러 쿼리를 배치로 처리"""
        results = []
        for q in queries:
            try:
                results.append(self.refine_instruction(q))
            except Exception as e:
                results.append({
                    "error": str(e),
                    "original_query": q,
                    "instruction_id": str(uuid.uuid4())
                })
        return results

    def validate_instruction(self, instruction: Dict[str, Any]) -> bool:
        """생성된 명령어의 간단한 유효성 검증"""
        required_fields = [
            "instruction_id", "original_query", "intent_type",
            "priority", "tasks", "context_requirements"
        ]
        for f in required_fields:
            if f not in instruction:
                return False

        valid_intents = ["ANOMALY_CHECK", "PREDICTION", "CONTROL", "INFORMATION", "OPTIMIZATION"]
        if instruction.get("intent_type") not in valid_intents:
            return False

        valid_priorities = ["HIGH", "MEDIUM", "LOW"]
        if instruction.get("priority") not in valid_priorities:
            return False

        tasks = instruction.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return False

        task_req = ["task_id", "agent", "action", "parameters", "dependencies", "expected_output"]
        for t in tasks:
            for f in task_req:
                if f not in t:
                    return False

        return True


# 불필요한 실행 예제 호출 제거 (NameError 방지)
if __name__ == "__main__":
    # 간단한 자체 헬스체크만 수행 (선택)
    client = InstructionRefinementClient(server_url="http://127.0.0.1:8000/v1")
    print(client.test_api_connection())