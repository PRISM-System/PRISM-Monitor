

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
