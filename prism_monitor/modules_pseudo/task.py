

from tinydb import TinyDB


from prism_monitor.llm.api import llm_generate
from prism_monitor.modules.monitoring import (
    monitoring_event_output, 
    monitoring_event_detect, 
    monitoring_event_explain, 
    monitoring_event_cause_candidates,
    monitoring_event_precursor,
    monitoring_event_evaluate_risk,
    monitoring_dashboard_update,
)
from prism_monitor.modules.event.llm import llm_parse_query



def monitoring_dashboard(task_id: int, type: str = "LINE", field: str = "line_id"):
    _res = {
        "status": "정상",
        "anomaly_detected": False,
        "anomaly_type": None,
        "updatedAt": "2025-07-17T12:01:03Z"
    }
    res = {
        'isSuccess': True,
        'code': 200,
        'message': "대시보드 데이터 조회 성공",
        'result': _res
    }
    return res

def monitoring_status_pending(task_id: int, status: str = "pending"):
    tasks = [
        {
            "taskId": task_id,
            "instruction": "이상 탐지 실행 중",
            "status": status
        }
    ]
    res = {
        "isSuccess": True,
        "code": 200,
        "message": "이상 탐지 실행 중",
        "tasks": tasks
    }
    return res

def monitoring_output(task_id, status: str = "complete", anomaly_detected: bool = True, description: str = "라인2-5 온도 이상 감지"):
    return {
        "isSuccess": True,
        "code": 201,
        "message": "결과 전달 완료"
    }

def workflow_start(llm_url: str, monitor_db: TinyDB, prism_core_db, task_id: str, query: str):
    from src.agent.monitoring_agent import MonitoringAgent
    agent = MonitoringAgent()
    
    # 워크플로우 시작 로직 구현
    # event detect > explain > cause-candidate > precursor > evaluate-risk 다실행
    parse_query_result = llm_parse_query(
        llm_url=llm_url,
        query=query
    )
    print(parse_query_result)
    start = parse_query_result.get('start', '2024-01-01T12:00:00Z')
    end = parse_query_result.get('end', '2024-01-01T13:00:00Z')
    detect_res = monitoring_event_detect(
        monitor_db=monitor_db, 
        prism_core_db=prism_core_db,
        start=start,
        end=end,
        task_id=task_id
    )
    explain_res = monitoring_event_explain(
        llm_url=llm_url,
        monitor_db=monitor_db,
        task_id=task_id
    )
    cause_candidates_res = monitoring_event_cause_candidates(
        llm_url=llm_url,
        monitor_db=monitor_db,
        task_id=task_id
    )
    precursor_res = monitoring_event_precursor(
        monitor_db=monitor_db,
        prism_core_db=prism_core_db,
        start=start,
        end=end,
        task_id=task_id
    )
    evaluate_risk_res = monitoring_event_evaluate_risk(
        llm_url=llm_url,
        monitor_db=monitor_db,
        task_id=task_id
    )
    result = {
        'detectResult': detect_res,
        'explainResult': explain_res,
        'causeCandidatesResult': cause_candidates_res,
        'precursorResult': precursor_res,
        'evaluateRiskResult': evaluate_risk_res
    }
    prompt = f'유저쿼리={query}에 대한 분석로그/결과={result}를 정리해줘'
    llm_result = llm_generate(llm_url, prompt)['text']
    return {
        'result': llm_result
    }
