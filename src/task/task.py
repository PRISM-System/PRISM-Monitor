from src.modules.query2sql.query2sql import query2sql
from src.modules.event.event_detect import detect_anomalies
from src.modules.explanation.explanation import event_explain, event_cause_candidates
from src.modules.precursor.precursor import event_precursor
from src.modules.risk_assessment.risk_assessment import evaluate_event_risk, prediction_risk
from src.modules.llm.llm import llm_generate_bimatrix

from src.test_scenarios.modeling import TestScenarioModel

TEST_SCENARIO_MODEL = TestScenarioModel()
TEST_SCENARIO_MODEL.set_models()


def workflow_start(task_id: str, query: str, bimatrix_llm_url: str = ""):
    # 워크플로우 시작 로직 구현
    # query2sql > event detect > explain > cause-candidate > precursor > evaluate-risk 다실행
    timestamp_start, timestamp_end, target_process, query2sql_df, query2sql_res = query2sql(query, bimatrix_llm_url)
    print(f"query2sql result: start={timestamp_start}, end={timestamp_end}, process={target_process}")
    detect_res = detect_anomalies(model=TEST_SCENARIO_MODEL, target_process=target_process, start=timestamp_start, end=timestamp_end)
    print(detect_res)

    explain_res = event_explain(bimatrix_llm_url, detect_res['anomalies'], target_process)
    print(f"event explain result: {explain_res}")

    cause_candidates_res = event_cause_candidates(bimatrix_llm_url, detect_res['anomalies'], target_process)
    print(f"event cause candidates result: {cause_candidates_res}")

    precursor_res = event_precursor(model=TEST_SCENARIO_MODEL, target_process=target_process, start=timestamp_start, end=timestamp_end)
    print(f"event precursor result: {precursor_res}")

    try:
        evaluate_risk_res = evaluate_event_risk(bimatrix_llm_url, str(detect_res))
        print(f"event evaluate risk result: {evaluate_risk_res}")
    except Exception as e:
        print(f"Error during evaluate_event_risk: {e}")
        evaluate_risk_res = {}

    try:
        prediction_risk_res = prediction_risk(bimatrix_llm_url, str(cause_candidates_res))
        print(f"event prediction risk result: {prediction_risk_res}")
    except Exception as e:
        print(f"Error during prediction_risk: {e}")
        prediction_risk_res = {}

    sub_result = {
        'query2sqlResult': query2sql_res,
        'detectResult': detect_res,
        'explainResult': explain_res,
        'causeCandidatesResult': cause_candidates_res,
        'precursorResult': precursor_res,
        'evaluateRiskResult': evaluate_risk_res,
        'predictionRiskResult': prediction_risk_res
    }
    prompt = f'유저쿼리={query}에 대한 분석로그/결과={sub_result}를 정리해줘'
    try:
        result = llm_generate_bimatrix(bimatrix_llm_url, prompt, max_tokens=len(prompt)+2048, is_json=False)
    except Exception as e:
        print(f"Error during llm_generate_bimatrix: {e}")
        result = str(sub_result)
    return {'result': result}

def get_dashboard():
    res = TEST_SCENARIO_MODEL.get_dashboard()
    return {
        'dashboard': res
    }

    