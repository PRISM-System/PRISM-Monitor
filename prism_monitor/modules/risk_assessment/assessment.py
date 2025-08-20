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

    

def risk_assessment(llm_url, event_detect_analysis, event_detect_analysis_history, task_instructions, task_instructions_history):
    event_evaluation = evaluate_event_risk(
        llm_url,
        event_detect_analysis,
        event_detect_analysis_history
    )
    
    prediction_evaluation = evaluate_prediction_risk(llm_url, task_instructions, task_instructions_history)
    
    return event_evaluation, prediction_evaluation
