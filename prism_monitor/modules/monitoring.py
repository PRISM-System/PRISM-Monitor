import os
import time
import pandas as pd
import json

from glob import glob
from tinydb import TinyDB, Query

from prism_monitor.data.database import PrismCoreDataBase
from prism_monitor.modules.event.event_detect import detect_anomalies_realtime
from prism_monitor.modules.event_precursor.precursor import precursor
from prism_monitor.modules.explanation.explanation import event_explain, event_cause_candidates
from prism_monitor.modules.risk_assessment.assessment import risk_assessment


def monitoring_event_output(status="complete", anomaly_detected=True, description="라인2-5 온도 이상 감지"):
    res = {
        "isSuccess": True,
        "code": 201,
        "message": "결과 전달 완료"
    }
    return res


# def monitoring_event_detect(monitor_db: TinyDB, prism_core_db: PrismCoreDataBase, start: str, end: str, task_id: str):
#     anomalies, svg, analysis, drift_results = detect_anomalies_realtime(prism_core_db, start=start, end=end)

#     event_record = {
#         "task_id": task_id,
#         "records": analysis,
#         "validation": {
#             "anomalies": anomalies,
#             "drift_results": drift_results
#         }
#     }
#     print(analysis)
    
#     Event = Query()
#     monitor_db.table('EventDetectHistory').upsert(event_record, Event.task_id == task_id)

#     return {
#         'result': {
#             'status': 'complete',
#             'anomalies': True if len(anomalies) else False,
#             'svg': svg
#         }
#     }

def monitoring_event_detect(monitor_db: TinyDB, prism_core_db, start: str, end: str, task_id: str):
    """모니터링 이벤트 감지 함수"""
    # detect_anomalies_realtime가 이제 5개 값을 반환 (drift_svg 추가)
    anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(prism_core_db, start=start, end=end)

    event_record = {
        "task_id": task_id,
        "records": analysis,
        "validation": {
            "anomalies": anomalies,
            "drift_results": drift_results,
        }
    }
    print(analysis)
    
    Event = Query()
    monitor_db.table('EventDetectHistory').upsert({'task_id':task_id, 'records':analysis}, Event.task_id == task_id)
    print(event_record)
    return {
        'result': {
            'status': 'complete',
            'anomalies': True if len(anomalies) else False,
            'drift_detected': True if len(drift_results) else False,  # 드리프트 감지 여부 추가
        }
    }


def monitoring_event_explain(llm_url, monitor_db: TinyDB, task_id: str):
    # task_id에 해당하는 이벤트 분석 데이터 조회
    Event = Query()
    event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)

    if not event_record:
        return {'error': f'No record found for task_id: {task_id}'}

    # 'records' 키에 담긴 분석 데이터 전달
    event_detect_analysis = event_record.get('records', [])

    explain = event_explain(
        llm_url=llm_url,
        event_detect_analysis=event_detect_analysis
    )
    res = {
        'explain': explain
    }
    Event = Query()
    monitor_db.table('EventExplainHistory').upsert(res, Event.task_id == task_id)

    return res


def monitoring_event_cause_candidates(llm_url, monitor_db: TinyDB, task_id: str):
    # task_id에 해당하는 이벤트 분석 데이터 조회
    Event = Query()
    event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)

    if not event_record:
        return {'error': f'No record found for task_id: {task_id}'}

    # 'records' 키에 담긴 분석 데이터 전달
    event_detect_analysis = event_record.get('records', [])

    cause_candidates = event_cause_candidates(
        llm_url=llm_url,
        event_detect_analysis=event_detect_analysis
    )
    res = {
        "causeCandidates": cause_candidates
    }
    Event = Query()
    monitor_db.table('EventCauseCandidatesHistory').upsert(res, Event.task_id == task_id)

    return res

def monitoring_event_precursor(monitor_db: TinyDB, prism_core_db: PrismCoreDataBase, start: str, end: str, task_id: str):
    start_time = pd.to_datetime(start, utc=True)
    end_time = pd.to_datetime(end, utc=True)

    datasets = {}
    try:
        raise ValueError('use local data')
        for table_name in prism_core_db.get_tables():
            df = prism_core_db.get_table_data(table_name)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            datasets[table_name] = df
    except Exception as e:
        print(f"dataset error raised {e}, use local data")
        data_paths = glob('prism_monitor/data/Industrial_DB_sample/*.csv')
        for data_path in data_paths:
            df = pd.read_csv(data_path)
            table_name = os.path.basename(data_path).split('.csv')[0].lower()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            datasets[table_name] = df

    res = precursor(datasets)
    print(res)
    Event = Query()
    monitor_db.table('EventPrecursorHistory').upsert(res, Event.task_id == task_id)

    return res


def monitoring_event_evaluate_risk(llm_url, monitor_db: TinyDB, task_id, topk=5):
    Event = Query()

    # 1. 현재 task_id에 해당하는 분석 결과
    event_detect_analysis = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)

    # 2. 과거 task_id 중 현재 task_id가 아닌 것만 상위 topk개
    event_detect_all = monitor_db.table('EventDetectHistory').all()
    event_detect_analysis_history = [
        r for r in event_detect_all if r.get('task_id') != task_id
    ][:topk]  # 정렬 기준 필요 시 추가

    # 3. 현재 task_id의 원인 후보 데이터
    task_instructions = monitor_db.table('EventCauseCandidatesHistory').get(Event.task_id == task_id)

    # 4. 과거 task_id의 원인 후보 데이터
    cause_all = monitor_db.table('EventCauseCandidatesHistory').all()
    task_instructions_history = [
        r for r in cause_all if r.get('task_id') != task_id
    ][:topk]

    # 위험 평가 수행
    event_evaluation, prediction_evaluation = risk_assessment(
        llm_url=llm_url,
        event_detect_analysis=event_detect_analysis,
        event_detect_analysis_history=event_detect_analysis_history,
        task_instructions=task_instructions,
        task_instructions_history=task_instructions_history
    )
    return {
        'eventEvaluation':event_evaluation,
        'predictionEvaluation':prediction_evaluation,
    }



def monitoring_dashboard_update(field: str = "line_id", type: str = "LINE", status: str = "비정상", anomaly_detected: bool = True, anomaly_type: str = "temperature_spike", updated_at: str = "2025-07-17T12:01:03Z"):
    return {
        "isSuccess": True,
        "code": 200,
        "message": "대시보드 업데이트 완료"
    }

# 아래 코드가 원래 코드 이 부분 주석 지우면 됨
# def monitoring_real_time(prism_core_db):
#     """모니터링 이벤트 감지 함수"""
#     # detect_anomalies_realtime가 이제 5개 값을 반환 (drift_svg 추가)
#     end = time.now()
#     start = time.now() - pd.Timedelta(minutes=10)
#     anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(prism_core_db, start=start, end=end)
#     result = vis_json

#     return {
#         'result': result
#     }


# 민주: 임의로 코드 수정해봄
def monitoring_real_time(prism_core_db):
    """모니터링 이벤트 감지 함수"""
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(minutes=10)
    
    anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(
        prism_core_db,
        start=start.isoformat(),  # 문자열로 변환
        end=end.isoformat()       # 문자열로 변환
    )
    
    return vis_json  # vis_json 직접 반환