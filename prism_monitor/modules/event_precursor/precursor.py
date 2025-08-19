# from prism_monitor.modules.event_precursor._precursor import (
#     load_and_explore_data,
#     create_unified_dataset,
#     prepare_features
# )

# from prism_monitor.modules.event_precursor._precursor import (
#     run_single_output_scenario,
#     run_real_time_monitoring_scenario,
#     run_multi_output_scenario
# )

from _precursor import (
    load_and_explore_data,
    create_unified_dataset,
    prepare_features
)

from _precursor import (
    run_single_output_scenario,
    run_real_time_monitoring_scenario,
    run_multi_output_scenario
)

from sklearn.model_selection import train_test_split

# import argparse

def main():
    DATA_BASE_PATH = '../../data/Industrial_DB_sample/'
    print("데이터 준비 및 전처리")
    all_datasets = load_and_explore_data(DATA_BASE_PATH)
    if not all_datasets:
        print("데이터 로딩 실패.")
        return
        
    unified_df = create_unified_dataset(all_datasets)
    if unified_df.empty:
        print("통합 데이터셋 생성 실패.")
        return

    processed_df, feature_cols, scaler = prepare_features(unified_df)
    print("=" * 40, "\n")

    train_val_df, test_df = train_test_split(processed_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)
    
    print(f"학습 데이터: {train_df.shape}")
    print(f"검증 데이터: {val_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    print("=" * 40, "\n")
    
    # 1. 단일 시점 이상 징후 예측 모델
    trained_model, model_scaler = run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler)
    
    # 2. 다중 시점 센서값 및 이상 징후 동시 예측 모델
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)
    
    # 3. 실시간 모니터링 시뮬레이션
    if trained_model is not None:
        anomaly_status = run_real_time_monitoring_scenario(trained_model, model_scaler, feature_cols, test_df)
        
    return {
        'summary': {
            'predicted_value': pred_value,
            'is_anomaly': anomaly_status
        }
    }

if __name__ == "__main__":
    main()