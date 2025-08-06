

def event_output(status="complete", anomaly_detected=True, description="라인2-5 온도 이상 감지"):
    res = {
        "isSuccess": True,
        "code": 201,
        "message": "결과 전달 완료"
    }
    return res


def event_detect(start: str, end: str):
    anomaly_periods = [
        {
            "start": start,
            "end": end
        }
    ]
    res = {
        "anomalyPeriods": anomaly_periods,
        "Value": "225C"
    }
    return res
     
def event_explain(anomaly_period: dict):
    # 실제 설명 분석 로직 대신 더미 응답 제공
    explanation = "센서 #2, #5가 다른 센서 대비 급격히 상승 추세를 보였습니다."

    return {
        "explanation": explanation
    }

def event_cause_candidates(anomaly_period: dict):
    candidates = [
        "센서 #2, #5의 상승",
        "라인 #2의 평균 온도 상승"
    ]
    return {
        "candidates": candidates
    }

def event_precursor(line_id: int, sensors: list[str]):
    # 실제 예측 분석 로직 대신 더미 응답 제공
    return {
        "percursor": "10분 후 215도 이상이 되어서 기준이 초과할 예상이 된다."
    }

def event_evaluate_risk(current_temp):
    # 실제 위험 평가 로직 대신 더미 응답 제공
    return {
        "riskLevel": "위험",
        "message": "법적 한계 기준 초과"
    }

def dashboard_update(field: str = "line_id", type: str = "LINE", status: str = "비정상", anomaly_detected: bool = True, anomaly_type: str = "temperature_spike", updated_at: str = "2025-07-17T12:01:03Z"):
    return {
        "isSuccess": True,
        "code": 200,
        "message": "대시보드 업데이트 완료"
    }