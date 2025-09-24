"""
프롬프트 템플릿 관리자
YAML 기반의 구조화된 프롬프트 관리를 제공합니다.
"""

import yaml
import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    system_prompt: str
    instruction_schema: Dict[str, Any]
    refinement_rules: Dict[str, Any]
    examples: Dict[str, Any]
    processing_guidelines: Dict[str, Any]
    output_requirements: List[str]


class PromptManager:
    """프롬프트 템플릿 관리 클래스"""
    
    def __init__(self, prompts_file: str = None):
        """
        프롬프트 매니저 초기화
        
        Args:
            prompts_file: 프롬프트 YAML 파일 경로
        """
        if prompts_file is None:
            prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
        
        self.prompts_file = prompts_file
        self.template = self._load_prompts()
    
    def _load_prompts(self) -> PromptTemplate:
        """YAML 파일에서 프롬프트 템플릿 로드"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return PromptTemplate(
                system_prompt=data['system_prompt'],
                instruction_schema=data['instruction_schema'],
                refinement_rules=data['refinement_rules'],
                examples=data['examples'],
                processing_guidelines=data['processing_guidelines'],
                output_requirements=data['output_requirements']
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {self.prompts_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파싱 오류: {e}")
    
    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.template.system_prompt
    
    def get_full_prompt(self, user_query: str, include_examples: bool = True) -> str:
        """
        완전한 프롬프트 생성
        
        Args:
            user_query: 사용자 쿼리
            include_examples: 예제 포함 여부
            
        Returns:
            완성된 프롬프트
        """
        prompt_parts = [
            "System:",
            self.template.system_prompt,
            "",
            f"User: 다음 쿼리를 JSON 명령어로 변환해주세요:",
            user_query,
            "",
            "다음 스키마에 맞춰 응답하세요:",
            yaml.dump(self.template.instruction_schema, default_flow_style=False, allow_unicode=True),
        ]
        
        if include_examples:
            prompt_parts.extend([
                "참고 예제:",
                yaml.dump(self.template.examples, default_flow_style=False, allow_unicode=True),
            ])
        
        prompt_parts.extend([
            "출력 요구사항:",
            *[f"- {req}" for req in self.template.output_requirements],
            "",
            "반드시 유효한 JSON 형식으로만 응답하세요."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_agent_role_prompt(self) -> str:
        """에이전트 등록용 역할 프롬프트 반환"""
        return """당신은 제조업 AI 에이전트 오케스트레이션을 위한 명령어 변환 전문가입니다. 
사용자의 자연어 쿼리를 구조화된 JSON 명령어로 변환하여 제조 시스템의 다양한 AI 에이전트들이 실행할 수 있도록 합니다.

주요 역할:
1. 의도 분류: ANOMALY_CHECK, PREDICTION, CONTROL, INFORMATION, OPTIMIZATION
2. 작업 분해: 복잡한 요청을 순차적 하위 작업으로 분해  
3. 파라미터 추출: 장비ID, 공정 파라미터, 시간 범위 등 추출
4. 안전성 고려: 제조업 안전 요구사항 및 규정 준수

반드시 유효한 JSON 형식으로만 응답하며, 제조업 도메인 지식을 바탕으로 정확한 변환을 수행합니다."""
    
    def get_intent_types(self) -> Dict[str, str]:
        """의도 분류 타입 목록 반환"""
        return self.template.refinement_rules['intent_classification']
    
    def get_example_by_type(self, intent_type: str) -> Dict[str, Any]:
        """
        특정 의도 타입의 예제 반환
        
        Args:
            intent_type: 의도 타입 (예: 'anomaly_detection')
            
        Returns:
            해당 타입의 예제
        """
        return self.template.examples.get(intent_type, {})
    
    def validate_output_schema(self, output: Dict[str, Any]) -> List[str]:
        """
        출력 스키마 유효성 검증
        
        Args:
            output: 검증할 출력
            
        Returns:
            검증 오류 목록 (빈 리스트면 유효)
        """
        errors = []
        required_fields = ['instruction_id', 'original_query', 'intent_type', 'priority', 'tasks']
        
        for field in required_fields:
            if field not in output:
                errors.append(f"필수 필드 누락: {field}")
        
        # 의도 타입 검증
        if 'intent_type' in output:
            valid_intents = list(self.template.refinement_rules['intent_classification'].keys())
            if output['intent_type'] not in valid_intents:
                errors.append(f"잘못된 의도 타입: {output['intent_type']}. 유효한 타입: {valid_intents}")
        
        # 우선순위 검증
        if 'priority' in output:
            valid_priorities = ['HIGH', 'MEDIUM', 'LOW']
            if output['priority'] not in valid_priorities:
                errors.append(f"잘못된 우선순위: {output['priority']}. 유효한 값: {valid_priorities}")
        
        return errors
    
    def reload_prompts(self):
        """프롬프트 템플릿 다시 로드"""
        self.template = self._load_prompts()
    
    def export_schema_json(self, output_path: str = None) -> str:
        """
        JSON 스키마를 파일로 내보내기
        
        Args:
            output_path: 출력 파일 경로
            
        Returns:
            JSON 문자열
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), 'instruction_schema.json')
        
        schema_json = json.dumps(self.template.instruction_schema, indent=2, ensure_ascii=False)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(schema_json)
        
        return schema_json


if __name__ == "__main__":
    # 사용 예제
    manager = PromptManager()
    
    print("=== 프롬프트 관리자 테스트 ===")
    print(f"시스템 프롬프트 길이: {len(manager.get_system_prompt())} 글자")
    
    # 의도 타입 출력
    print("\n의도 분류 타입:")
    for intent, desc in manager.get_intent_types().items():
        print(f"  {intent}: {desc}")
    
    # 완전한 프롬프트 생성 예제
    test_query = "3번 엣칭 장비 상태 확인"
    full_prompt = manager.get_full_prompt(test_query)
    print(f"\n완전한 프롬프트 길이: {len(full_prompt)} 글자")
    
    # JSON 스키마 내보내기
    schema_json = manager.export_schema_json()
    print(f"\nJSON 스키마 내보내기 완료: instruction_schema.json")