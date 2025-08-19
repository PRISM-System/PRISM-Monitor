import pandas as pd
import json


from prism_monitor.data.database import PrismCoreDataBase
from prism_monitor.modules.event.event_detect import detect_anomalies_in_timerange
from prism_monitor.modules.event_precursor.precursor import precursor
from prism_monitor.modules.explanation.explanation import event_explain
from prism_monitor.modules.explanation.explanation import event_cause_candidates


def monitoring_event_output(status="complete", anomaly_detected=True, description="라인2-5 온도 이상 감지"):
    res = {
        "isSuccess": True,
        "code": 201,
        "message": "결과 전달 완료"
    }
    return res


def monitoring_event_detect(prism_core_db: PrismCoreDataBase, start: str, end: str):
    datasets = {}
    for table_name in prism_core_db.get_tables():
        datasets[table_name] = prism_core_db.get_table_data(table_name)
    anomalies, analysis = detect_anomalies_in_timerange(datasets)
    return {
        'result':{
            'status':'complete',
            'anomalies': True if len(anomalies) else False,
            'description': json.dumps(analysis)
        }
    }

def monitoring_event_explain(url, event_detect_desc:str):
    # 실제 설명 분석 로직 대신 더미 응답 제공
    res = event_explain(
        url=url,
        event_detect_desc=event_detect_desc
    )
    print(res)
    return {
        'explain':res
    }

def monitoring_event_cause_candidates(url, event_detect_desc:str):
    res = event_cause_candidates(
        url=url,
        event_detect_desc=event_detect_desc
    )
    return {
        "causeCandidates": res
    }

def monitoring_event_precursor(prism_core_db: PrismCoreDataBase):
    datasets = {}
    for table_name in prism_core_db.get_tables():
        datasets[table_name] = prism_core_db.get_table_data(table_name)
    return precursor(datasets)


def monitoring_event_evaluate_risk(current_temp):
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

def monitoring_dashboard_update(field: str = "line_id", type: str = "LINE", status: str = "비정상", anomaly_detected: bool = True, anomaly_type: str = "temperature_spike", updated_at: str = "2025-07-17T12:01:03Z"):
    return {
        "isSuccess": True,
        "code": 200,
        "message": "대시보드 업데이트 완료"
    }