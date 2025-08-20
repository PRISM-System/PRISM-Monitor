import json

# from _data_load import (
#     load_and_explore_data,
#     create_unified_dataset,
#     prepare_features
# )

from prism_monitor.modules.risk_assessment._data_load import (
    load_and_explore_data,
    create_unified_dataset,
    prepare_features
)

# from _assessment import (
#     evaluate_event_risk,
#     evaluate_prediction_risk
# )

from prism_monitor.modules.risk_assessment._assessment import (
    evaluate_event_risk,
    evaluate_prediction_risk
)

def evaluate_event_and_prediction_risk(datasets):
    unified_df = create_unified_dataset(datasets)
    processed_df, feature_cols, scaler = prepare_features(unified_df)

    

def risk_assessment(event_detect_analysis, event_detect_analysis_history, task_instructions, task_instructions_history):
    event_evaluation = evaluate_event_risk(
        event_detect_analysis,
        event_detect_analysis_history
    )
    
    # event_evaluation = evaluate_event_risk(datasets, sample_event)
    print("\n이벤트 위험 평가 결과:")
    print(json.dumps(event_evaluation, indent=2, ensure_ascii=False))
    
    prediction_evaluation = evaluate_prediction_risk(task_instructions, task_instructions_history)
    print("\n예측 AI 결과물 평가 결과:")
    print(json.dumps(prediction_evaluation, indent=2, ensure_ascii=False))
    
    # ===== 평가 결과 요약 =====
    print("\n========== 평가 요약 ==========")
    print(f"이벤트 위험 평가 최종 결정: {event_evaluation.get('final_decision', 'N/A')}")
    print(f"예측 AI 평가 최종 결정: {prediction_evaluation.get('final_decision', 'N/A')}")
    
    return event_evaluation, prediction_evaluation
