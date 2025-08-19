import pandas as pd

def event_output(status="complete", anomaly_detected=True, description="라인2-5 온도 이상 감지"):
    res = {
        "isSuccess": True,
        "code": 201,
        "message": "결과 전달 완료"
    }
    return res


def event_detect(start: str, end: str):
    from prism_monitor.modules.detect.detect import detect
    return detect()
     
def event_explain(anomaly_period: dict):
    # 실제 설명 분석 로직 대신 더미 응답 제공
    from prism_monitor.modules.explanation.explanation import explain
    data = '{"PNO":"PS024","EQUIPMENT_ID":"PHO_003","LOT_NO":"LOT24009A","WAFER_ID":"W001","TIMESTAMP":"2024-01-23 08:15:20","EXPOSURE_DOSE":41.2,"FOCUS_POSITION":-25.3,"STAGE_TEMP":23.02,"BAROMETRIC_PRESSURE":1014.1,"HUMIDITY":54.3,"ALIGNMENT_ERROR_X":2.6,"ALIGNMENT_ERROR_Y":2.8,"LENS_ABERRATION":4.5,"ILLUMINATION_UNIFORMITY":97.8,"RETICLE_TEMP":23.06}'
    return {
        'explain':explain(data)
    }

def event_cause_candidates(anomaly_period: dict):
    from prism_monitor.modules.explanation.explanation import cause_candidates
    data = '{"PNO":"PS024","EQUIPMENT_ID":"PHO_003","LOT_NO":"LOT24009A","WAFER_ID":"W001","TIMESTAMP":"2024-01-23 08:15:20","EXPOSURE_DOSE":41.2,"FOCUS_POSITION":-25.3,"STAGE_TEMP":23.02,"BAROMETRIC_PRESSURE":1014.1,"HUMIDITY":54.3,"ALIGNMENT_ERROR_X":2.6,"ALIGNMENT_ERROR_Y":2.8,"LENS_ABERRATION":4.5,"ILLUMINATION_UNIFORMITY":97.8,"RETICLE_TEMP":23.06}'
    return {
        "causeCandidates": cause_candidates(data)
    }

def event_precursor(line_id: int, sensors: list[str]):
    # 실제 예측 분석 로직 대신 더미 응답 제공
    return {
        "percursor": "10분 후 215도 이상이 되어서 기준이 초과할 예상이 된다."
    }

def event_evaluate_risk(current_temp):
    # 실제 위험 평가 로직 대신 더미 응답 제공
    from prism_monitor.modules.evaluate_risk.evaluate_risk import evaluate_risk
    evaluated = evaluate_risk()
    result = {
        'totalCandidates': evaluated['total_candidates'],
        'passedCandidates': evaluated['passed_candidates'],
        'failedCandidates': evaluated['failed_candidates'],
        'riskLevel': evaluated['risk_level'],
        'complianceStatus':evaluated['compliance_status']
    }
    recommended_actions = []
    for recommended_action in evaluated.get('recommended_actions',[]):
        recommended_actions.append({
            'actionName': recommended_action['action_name'],
            'totalScore': recommended_action['total_score']/50
        })
    result['recommendedActions'] = recommended_actions
    return result

def dashboard_update(field: str = "line_id", type: str = "LINE", status: str = "비정상", anomaly_detected: bool = True, anomaly_type: str = "temperature_spike", updated_at: str = "2025-07-17T12:01:03Z"):
    return {
        "isSuccess": True,
        "code": 200,
        "message": "대시보드 업데이트 완료"
    }