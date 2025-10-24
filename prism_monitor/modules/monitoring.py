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

# 🆕 NEW VERSION: CSV 파일별 모델 지원 (query_decompose 통합 준비)
def monitoring_event_detect(monitor_db: TinyDB, prism_core_db, start: str, end: str, task_id: str,
                           target_file: str = None, target_process: str = None, user_query: str = None):
    """
    모니터링 이벤트 감지 함수

    Args:
        monitor_db: 모니터 DB
        prism_core_db: PRISM Core DB
        start: 시작 시간 (ISO format)
        end: 종료 시간 (ISO format)
        task_id: 태스크 ID
        target_process: 타겟 공정 (예: 'semi_cmp_sensors') - 직접 지정 시 사용
        user_query: 사용자 쿼리 - query_decompose로 공정 자동 판별 시 사용

    Returns:
        이벤트 감지 결과
    """
    classified_process = None

    # 1. user_query가 있다면 query_decompose로 공정 자동 판별 (향후 기능)
    if user_query and not target_process:
        try:
            # ⚠️ query_decompose가 classified_class를 반환하도록 수정된 후 활성화
            # from prism_monitor.modules.query_decompose.query_decompose import query_decompose
            # timestamp_min, timestamp_max, result_df, classified_class = query_decompose(user_query)
            # classified_process = classified_class
            # print(f"query_decompose identified process: {classified_process}")
            print("query_decompose integration pending (another developer's task)")
        except Exception as e:
            print(f"query_decompose failed: {e}, using target_process or legacy mode")

    # 2. target_file 우선순위: 직접 지정 > query_decompose 결과 > target_process(하위 호환)
    final_target_file = target_file or classified_process or target_process

    # 3. 이상 감지 수행 (파일별 모델 사용)
    anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(
        prism_core_db,
        start=start,
        end=end,
        target_file=final_target_file  # 🆕 파일 지정
    )

    event_record = {
        "task_id": task_id,
        "records": analysis,
        "target_file": final_target_file,  # 🆕 파일 정보 저장
        "validation": {
            "anomalies": anomalies,
            "drift_results": drift_results,
        }
    }
    print('analysis=', analysis)

    Event = Query()
    monitor_db.table('EventDetectHistory').upsert(
        {
            'task_id': task_id,
            'records': analysis,
            'target_file': final_target_file,  # 🆕 파일 정보 DB 저장
            'target_process': final_target_file  # 🆕 공정 정보 (explanation에서 사용)
        },
        Event.task_id == task_id
    )
    print(event_record)

    return {
        'result': {
            'status': 'complete',
            'anomalies': True if len(anomalies) else False,
            'drift_detected': True if len(drift_results) else False,
            'target_file': final_target_file  # 🆕 결과에 파일 정보 포함
        }
    }

# 📝 OLD VERSION (주석 처리 - 참고용)
# def monitoring_event_detect(monitor_db: TinyDB, prism_core_db, start: str, end: str, task_id: str):
#     """모니터링 이벤트 감지 함수"""
#     # detect_anomalies_realtime가 이제 5개 값을 반환 (drift_svg 추가)
#     anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(prism_core_db, start=start, end=end)
#
#     event_record = {
#         "task_id": task_id,
#         "records": analysis,
#         "validation": {
#             "anomalies": anomalies,
#             "drift_results": drift_results,
#         }
#     }
#     print('analysis=', analysis)
#
#     Event = Query()
#     monitor_db.table('EventDetectHistory').upsert({'task_id':task_id, 'records':analysis}, Event.task_id == task_id)
#     print(event_record)
#     return {
#         'result': {
#             'status': 'complete',
#             'anomalies': True if len(anomalies) else False,
#             'drift_detected': True if len(drift_results) else False,  # 드리프트 감지 여부 추가
#         }
#     }


# 🆕 NEW VERSION: 공정별 프롬프트 지원
def monitoring_event_explain(llm_url, monitor_db: TinyDB, task_id: str):
    # task_id에 해당하는 이벤트 분석 데이터 조회
    Event = Query()
    event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)

    if not event_record:
        return {'error': f'No record found for task_id: {task_id}'}

    # 'records' 키에 담긴 분석 데이터 전달
    event_detect_analysis = event_record.get('records', [])
    target_process = event_record.get('target_process')  # 🆕 공정 정보 조회

    explain = event_explain(
        llm_url=llm_url,
        event_detect_analysis=event_detect_analysis,
        process_type=target_process  # 🆕 공정별 프롬프트 사용
    )
    res = {
        'explain': explain,
        'target_process': target_process  # 🆕 공정 정보 포함
    }
    Event = Query()
    monitor_db.table('EventExplainHistory').upsert(res, Event.task_id == task_id)

    return res

# 📝 OLD VERSION (주석 처리 - 참고용)
# def monitoring_event_explain(llm_url, monitor_db: TinyDB, task_id: str):
#     # task_id에 해당하는 이벤트 분석 데이터 조회
#     Event = Query()
#     event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)
#
#     if not event_record:
#         return {'error': f'No record found for task_id: {task_id}'}
#
#     # 'records' 키에 담긴 분석 데이터 전달
#     event_detect_analysis = event_record.get('records', [])
#
#     explain = event_explain(
#         llm_url=llm_url,
#         event_detect_analysis=event_detect_analysis
#     )
#     res = {
#         'explain': explain
#     }
#     Event = Query()
#     monitor_db.table('EventExplainHistory').upsert(res, Event.task_id == task_id)
#
#     return res


# 🆕 NEW VERSION: 공정별 프롬프트 지원
def monitoring_event_cause_candidates(llm_url, monitor_db: TinyDB, task_id: str):
    # task_id에 해당하는 이벤트 분석 데이터 조회
    Event = Query()
    event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)

    if not event_record:
        return {'error': f'No record found for task_id: {task_id}'}

    # 'records' 키에 담긴 분석 데이터 전달
    event_detect_analysis = event_record.get('records', [])
    target_process = event_record.get('target_process')  # 🆕 공정 정보 조회

    cause_candidates = event_cause_candidates(
        llm_url=llm_url,
        event_detect_analysis=event_detect_analysis,
        process_type=target_process  # 🆕 공정별 프롬프트 사용
    )
    res = {
        "causeCandidates": cause_candidates,
        "target_process": target_process  # 🆕 공정 정보 포함
    }
    Event = Query()
    monitor_db.table('EventCauseCandidatesHistory').upsert(res, Event.task_id == task_id)

    return res

# 📝 OLD VERSION (주석 처리 - 참고용)
# def monitoring_event_cause_candidates(llm_url, monitor_db: TinyDB, task_id: str):
#     # task_id에 해당하는 이벤트 분석 데이터 조회
#     Event = Query()
#     event_record = monitor_db.table('EventDetectHistory').get(Event.task_id == task_id)
#
#     if not event_record:
#         return {'error': f'No record found for task_id: {task_id}'}
#
#     # 'records' 키에 담긴 분석 데이터 전달
#     event_detect_analysis = event_record.get('records', [])
#
#     cause_candidates = event_cause_candidates(
#         llm_url=llm_url,
#         event_detect_analysis=event_detect_analysis
#     )
#     res = {
#         "causeCandidates": cause_candidates
#     }
#     Event = Query()
#     monitor_db.table('EventCauseCandidatesHistory').upsert(res, Event.task_id == task_id)
#
#     return res

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

def monitoring_real_time(prism_core_db):
    """모니터링 이벤트 감지 함수"""
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(minutes=10)
    
    anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(
        prism_core_db,
        start=start.isoformat(),  # 문자열로 변환
        end=end.isoformat()       # 문자열로 변환
    )
    print(vis_json)
    
    return {'visJson': vis_json}  # vis_json 직접 반환