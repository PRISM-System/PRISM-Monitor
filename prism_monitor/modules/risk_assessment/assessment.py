import json

# from _data_load import (
#     load_and_explore_data,
#     create_unified_dataset_2,
#     prepare_features
# )

from prism_monitor.modules.risk_assessment._data_load import (
    load_and_explore_data,
    create_unified_dataset_2,
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


def main():
    # 데이터 로드
    data_base_path = '../../data/Industrial_DB_sample/'
    # data_base_path = '/home/jonghak/agi/PRISM-Monitor/prism_monitor/data/Industrial_DB_sample/'
    datasets = load_and_explore_data(data_base_path)
    
    # 통합 데이터셋 생성
    unified_df = create_unified_dataset_2(datasets)
    
    # 특징 준비
    processed_df, feature_cols, scaler = prepare_features(unified_df)
    
    # ===== 예시 1: 이벤트 위험 평가 =====
    sample_event = {
        'lot_no': 'LOT001',
        'product_name': 'DRAM_256GB',
        'current_step': 'ETCH',
        'anomaly_type': 'YIELD_DROP',
        'severity': 'HIGH',
        'proposed_actions': [
            {
                'action': 'EQUIPMENT_CALIBRATION',
                'target': 'ETCH_001',
                'priority': 'IMMEDIATE',
                'estimated_time': '2 hours'
            },
            {
                'action': 'RECIPE_ADJUSTMENT',
                'parameter': 'RF_POWER',
                'change': '-5%',
                'validation_required': True
            }
        ]
    }
    
    event_evaluation = evaluate_event_risk(
        datasets,
        sample_event,
        processed_data = processed_df,
        feature_columns = feature_cols
    )
    
    # event_evaluation = evaluate_event_risk(datasets, sample_event)
    print("\n이벤트 위험 평가 결과:")
    print(json.dumps(event_evaluation, indent=2, ensure_ascii=False))
    
    # ===== 예시 2: 예측 AI 결과물 평가 =====
    sample_prediction = {
        'equipment_id': 'ETCH_001',
        'prediction_type': 'PREVENTIVE_MAINTENANCE',
        'confidence': 0.85,
        'predicted_failure_time': '2024-12-30',
        'maintenance_plan': {
            'scheduled_date': '2024-12-25',
            'maintenance_type': 'FULL_PM',
            'estimated_duration': '8 hours',
            'parts_replacement': ['RF_Generator', 'Chamber_Liner'],
            'estimated_cost': 50000
        }
    }
    
    prediction_evaluation = evaluate_prediction_risk(datasets, sample_prediction)
    print("\n예측 AI 결과물 평가 결과:")
    print(json.dumps(prediction_evaluation, indent=2, ensure_ascii=False))
    
    # ===== 평가 결과 요약 =====
    print("\n========== 평가 요약 ==========")
    print(f"이벤트 위험 평가 최종 결정: {event_evaluation.get('final_decision', 'N/A')}")
    print(f"예측 AI 평가 최종 결정: {prediction_evaluation.get('final_decision', 'N/A')}")
    
    return event_evaluation, prediction_evaluation

if __name__ == "__main__":   
    event_result, prediction_result = main()