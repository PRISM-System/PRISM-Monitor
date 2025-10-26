from tinydb import TinyDB, Query

from src.modules.event.event_detect import detect_anomalies_realtime

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