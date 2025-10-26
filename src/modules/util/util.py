import pandas as pd

def convert_to_json_serializable(obj):
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    
    # numpy 배열이나 pandas Series/DataFrame 처리
    if isinstance(obj, (np.ndarray, pd.Series)):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    
    # scalar 값들에 대한 NA 체크 (배열이 아닌 경우만)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # 배열이거나 NA 체크가 불가능한 객체는 그냥 넘어감
        pass
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def dataframe_to_json_serializable(df):
    """DataFrame을 JSON 직렬화 가능한 dict로 변환"""
    if df.empty:
        return []
    
    # DataFrame을 dict로 변환
    records = df.to_dict(orient="records")
    
    # 각 레코드를 JSON 직렬화 가능하게 변환
    json_records = []
    for record in records:
        json_record = convert_to_json_serializable(record)
        json_records.append(json_record)
    
    return json_records