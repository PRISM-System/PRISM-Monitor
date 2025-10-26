import os

from src.test_scenarios.modeling import TestScenarioModel
from src.modules.util.util import dataframe_to_json_serializable

def detect_anomalies(model: TestScenarioModel, target_process: str, start: str, end: str, serialize: bool = False):
    # 이상치 탐지 모델 초기화
    # 이상치 탐지 수행
    anomalies = model.ad_detect_anomalies(target_process, start, end)
    summary = model.ad_summary(target_process, start, end)
    vis_json = model.ad_get_visual_data(target_process, start, end)

    results = {
        "anomalies": anomalies,
        "summary": summary,
        "visualization": vis_json
    }
    if serialize:
        return {
            "result":{
            "anomalies": dataframe_to_json_serializable(anomalies),
            "summary": summary,
            "visualization": vis_json
        }}
    return results