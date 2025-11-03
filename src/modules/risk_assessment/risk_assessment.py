
from src.modules.llm.llm import LLMCallManager


def evaluate_event_risk(event_detect_analysis, event_detect_analysis_history=''):
    """이벤트 위험 평가를 위한 LLM 프롬프트 생성"""
    prompt = f"""
    반도체 제조 공정에서 발생한 이상 이벤트와 제안된 대응 행동을 평가해주세요.
    
    ## 현재 이벤트 정보
    {event_detect_analysis}
    
    ## 과거 유사 사례
    {event_detect_analysis_history}
    
    ## 제안된 대응 행동
    
    다음 항목을 평가해주세요:
    1. 근본 원인 분석의 정확성 (0-100점)
    2. 제안된 행동의 적절성 (0-100점)
    3. 규제 및 안전 기준 준수 여부 (통과/실패)
    4. 실행 가능성 (0-100점)
    5. 예상 효과성 (0-100점)
    
    유효한 JSON 형식으로만 응답해주세요. 다른 텍스트가 포함되면 에러가 발생합니다:
    {{
        "root_cause_accuracy": 점수,
        "action_appropriateness": 점수,
        "compliance_status": "PASS" 또는 "FAIL",
        "feasibility": 점수,
        "expected_effectiveness": 점수,
        "overall_score": 전체 평균 점수,
        "risk_level": "LOW", "MEDIUM", "HIGH" 중 하나,
        "recommendation": "승인", "조건부 승인", "거부" 중 하나,
        "reasoning": "평가 근거 설명",
        "improvement_suggestions": ["개선 제안 1", "개선 제안 2"]
    }}
    """
    response = LLMCallManager.invoke(prompt=prompt, max_tokens=len(prompt)+2048)
    print(response)
    return response


def prediction_risk(task_instructions, task_instructions_history=''):
    """예측 AI 결과물 위험 평가를 위한 LLM 프롬프트 생성"""
    prompt = f"""
    반도체 제조 장비의 예측 유지보수 계획을 평가해주세요.

    ## 과거 유지보수 이력
    {task_instructions_history}
     
    ## 제안된 유지보수 계획
    {task_instructions}
    
    다음 항목을 평가해주세요:
    1. 예측 모델의 신뢰성 (0-100점)
    2. 유지보수 시기의 적절성 (0-100점)
    3. 비용 효율성 (0-100점)
    4. 생산 영향 최소화 (0-100점)
    5. 규제 준수 여부 (통과/실패)
    
    유효한 JSON 형식으로만 응답해주세요. 다른 텍스트가 포함되면 에러가 발생합니다:
    {{
        "prediction_reliability": 점수,
        "timing_appropriateness": 점수,
        "cost_efficiency": 점수,
        "production_impact": 점수,
        "compliance_status": "PASS" 또는 "FAIL",
        "overall_score": 전체 평균 점수,
        "confidence_level": "HIGH", "MEDIUM", "LOW" 중 하나,
        "recommendation": "즉시 실행", "일정 조정 후 실행", "재검토 필요" 중 하나,
        "reasoning": "평가 근거 설명",
        "risk_factors": ["위험 요소 1", "위험 요소 2"],
        "optimization_suggestions": ["최적화 제안 1", "최적화 제안 2"]
    }}
    """
    response = LLMCallManager.invoke(prompt=prompt, max_tokens=len(prompt)+2048)
    return response