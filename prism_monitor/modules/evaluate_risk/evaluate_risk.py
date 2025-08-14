import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

class RiskAssessmentModule:
    def __init__(self, vllm_endpoint="http://localhost:8001/v1/completions"):
        """
        위험 평가 모듈 초기화
        Args:
            vllm_endpoint: vLLM 서버 엔드포인트
        """
        self.vllm_endpoint = vllm_endpoint
        self.safety_regulations = [
            "산업안전보건법 준수",
            "위험물안전관리법 준수", 
            "환경안전 기준 준수",
            "작업자 보호 조치 필수"
        ]
    
    def generate_action_candidates(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        위험 이벤트에 대한 행동 후보군 생성
        Args:
            event_data: 위험 이벤트 데이터
        Returns:
            행동 후보군 리스트
        """
        prompt = f"""
        다음 위험 이벤트에 대한 대응 행동 후보를 생성해주세요.

        이벤트 정보:
        - 유형: {event_data.get('event_type', '')}
        - 심각도: {event_data.get('severity', '')}
        - 위치: {event_data.get('location', '')}
        - 설명: {event_data.get('description', '')}
        - 발생시간: {event_data.get('timestamp', '')}

        다음 형식으로 3-5개의 행동 후보를 제시해주세요:
        1. 즉시 조치 행동
        2. 단기 대응 행동  
        3. 중장기 예방 행동

        각 행동에 대해 구체적인 실행 방법을 포함해주세요.
        """
        
        response = self._call_llm(prompt)
        return self._parse_action_candidates(response)
    
    def evaluate_action_candidates(self, event_data: Dict[str, Any], 
                                 action_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        행동 후보군에 대한 위험 평가 수행
        Args:
            event_data: 위험 이벤트 데이터
            action_candidates: 행동 후보군
        Returns:
            평가 결과 리스트
        """
        evaluation_results = []
        
        for candidate in action_candidates:
            evaluation = self._evaluate_single_action(event_data, candidate)
            evaluation_results.append(evaluation)
        
        return evaluation_results
    
    def _evaluate_single_action(self, event_data: Dict[str, Any], 
                               action: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 행동 후보에 대한 상세 평가
        """
        prompt = f"""
        다음 위험 이벤트와 대응 행동을 평가해주세요.

        위험 이벤트:
        - 유형: {event_data.get('event_type', '')}
        - 심각도: {event_data.get('severity', '')}
        - 설명: {event_data.get('description', '')}

        대응 행동:
        - 행동명: {action.get('action_name', '')}
        - 내용: {action.get('description', '')}
        - 실행 방법: {action.get('execution_method', '')}

        다음 기준으로 평가해주세요 (각각 1-10점):
        1. 안전성 (Safety): 작업자와 설비 안전 확보 정도
        2. 효과성 (Effectiveness): 위험 해결 효과
        3. 실행가능성 (Feasibility): 현실적 실행 가능 정도
        4. 규제준수성 (Compliance): 산업안전보건법 등 규제 준수
        5. 비용효율성 (Cost-efficiency): 비용 대비 효과

        응답 형식:
        안전성: [점수]/10 - [평가 근거]
        효과성: [점수]/10 - [평가 근거]
        실행가능성: [점수]/10 - [평가 근거]
        규제준수성: [점수]/10 - [평가 근거]
        비용효율성: [점수]/10 - [평가 근거]
        종합점수: [총점]/50
        최종판정: [PASS/FAIL]
        권고사항: [구체적 개선 사항]
        """
        
        response = self._call_llm(prompt)
        return self._parse_evaluation_result(action, response)
    
    def _call_llm(self, prompt: str) -> str:
        """
        vLLM API 호출
        """
        payload = {
            "model": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": ["<|endoftext|>"]
        }
        
        try:
            response = requests.post(self.vllm_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return ""
    
    def _parse_action_candidates(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        LLM 응답에서 행동 후보군 파싱
        """
        candidates = []
        lines = llm_response.split('\n')
        
        current_action = {}
        action_count = 0
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                if current_action:
                    candidates.append(current_action)
                
                action_count += 1
                current_action = {
                    'action_id': f"action_{action_count}",
                    'action_name': line[2:].strip(),
                    'description': '',
                    'execution_method': '',
                    'category': self._categorize_action(line)
                }
            elif line and current_action:
                if 'description' not in current_action or not current_action['description']:
                    current_action['description'] = line
                else:
                    current_action['execution_method'] += line + ' '
        
        if current_action:
            candidates.append(current_action)
        
        return candidates
    
    def _parse_evaluation_result(self, action: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """
        LLM 평가 응답 파싱
        """
        result = {
            'action_id': action.get('action_id', ''),
            'action_name': action.get('action_name', ''),
            'scores': {},
            'total_score': 0,
            'pass_fail': 'FAIL',
            'evaluation_details': llm_response,
            'timestamp': datetime.now().isoformat()
        }
        
        lines = llm_response.split('\n')
        criteria = ['안전성', '효과성', '실행가능성', '규제준수성', '비용효율성']
        
        for line in lines:
            line = line.strip()
            for criterion in criteria:
                if line.startswith(criterion):
                    try:
                        score_part = line.split(':')[1].strip()
                        score = int(score_part.split('/')[0].strip())
                        result['scores'][criterion] = score
                    except:
                        result['scores'][criterion] = 0
            
            if line.startswith('종합점수'):
                try:
                    total_score = int(line.split(':')[1].split('/')[0].strip())
                    result['total_score'] = total_score
                except:
                    result['total_score'] = sum(result['scores'].values())
            
            if line.startswith('최종판정'):
                if 'PASS' in line.upper():
                    result['pass_fail'] = 'PASS'
        
        return result
    
    def _categorize_action(self, action_text: str) -> str:
        """
        행동 유형 분류
        """
        action_lower = action_text.lower()
        if any(word in action_lower for word in ['즉시', '긴급', '중단']):
            return '즉시조치'
        elif any(word in action_lower for word in ['점검', '수리', '교체']):
            return '단기대응'
        elif any(word in action_lower for word in ['예방', '교육', '시스템']):
            return '예방조치'
        else:
            return '일반조치'
    
    def generate_risk_report(self, event_data: Dict[str, Any], 
                           evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        위험 평가 종합 보고서 생성
        """
        passed_actions = [r for r in evaluation_results if r['pass_fail'] == 'PASS']
        failed_actions = [r for r in evaluation_results if r['pass_fail'] == 'FAIL']
        
        report = {
            'event_summary': event_data,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_candidates': len(evaluation_results),
            'passed_candidates': len(passed_actions),
            'failed_candidates': len(failed_actions),
            'recommended_actions': sorted(passed_actions, key=lambda x: x['total_score'], reverse=True),
            'rejected_actions': failed_actions,
            'risk_level': self._calculate_risk_level(evaluation_results),
            'compliance_status': self._check_compliance_status(evaluation_results)
        }
        
        return report

    def _calculate_risk_level(self, evaluation_results: List[Dict[str, Any]]) -> str:
        """
        전체 위험 수준 계산
        """
        if not evaluation_results:
            return 'HIGH'
        
        avg_score = sum(r['total_score'] for r in evaluation_results) / len(evaluation_results)
        
        if avg_score >= 40:
            return 'LOW'
        elif avg_score >= 30:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _check_compliance_status(self, evaluation_results: List[Dict[str, Any]]) -> bool:
        """
        규제 준수 상태 확인
        """
        for result in evaluation_results:
            if result['pass_fail'] == 'PASS' and result['scores'].get('규제준수성', 0) >= 8:
                return True
        return False


# 사용 예시
def evaluate_risk():
    # 위험 평가 모듈 초기화
    risk_module = RiskAssessmentModule()
    
    # 예시 위험 이벤트 데이터
    event_data = {
        'event_id': 'EVT_001',
        'event_type': '설비 이상',
        'severity': 'HIGH',
        'location': '생산라인 A',
        'description': '컨베이어 벨트 과열로 인한 화재 위험 감지',
        'timestamp': datetime.now().isoformat(),
        'affected_equipment': ['컨베이어_001', '센서_A12'],
        'current_status': '운영중단'
    }
    
    print("=== 위험 이벤트 위험 평가 시스템 ===")
    print(f"이벤트 ID: {event_data['event_id']}")
    print(f"이벤트 유형: {event_data['event_type']}")
    print(f"심각도: {event_data['severity']}")
    
    # 1단계: 행동 후보군 생성
    print("\n1. 행동 후보군 생성 중...")
    action_candidates = risk_module.generate_action_candidates(event_data)
    
    print(f"생성된 행동 후보 수: {len(action_candidates)}")
    for i, candidate in enumerate(action_candidates, 1):
        print(f"  {i}. {candidate['action_name']}")
    
    # 2단계: 행동 후보군 평가
    print("\n2. 행동 후보군 평가 중...")
    evaluation_results = risk_module.evaluate_action_candidates(event_data, action_candidates)
    
    # 3단계: 평가 결과 출력
    print("\n3. 평가 결과:")
    for result in evaluation_results:
        print(f"\n행동명: {result['action_name']}")
        print(f"종합점수: {result['total_score']}/50")
        print(f"최종판정: {result['pass_fail']}")
        
        print("세부 점수:")
        for criterion, score in result['scores'].items():
            print(f"  - {criterion}: {score}/10")
    
    # 4단계: 종합 보고서 생성
    print("\n4. 종합 보고서 생성 중...")
    final_report = risk_module.generate_risk_report(event_data, evaluation_results)
    
    print(f"\n=== 위험 평가 종합 보고서 ===")
    print(f"총 후보 수: {final_report['total_candidates']}")
    print(f"통과 후보 수: {final_report['passed_candidates']}")
    print(f"실패 후보 수: {final_report['failed_candidates']}")
    print(f"전체 위험 수준: {final_report['risk_level']}")
    print(f"규제 준수 상태: {final_report['compliance_status']}")
    
    if final_report['recommended_actions']:
        print(f"\n권장 조치 (최고 점수):")
        top_action = final_report['recommended_actions'][0]
        print(f"- {top_action['action_name']}")
        print(f"- 점수: {top_action['total_score']}/50")
    return final_report