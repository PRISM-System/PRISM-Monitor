from prism_monitor.modules.event_precursor._precursor import (
    load_and_explore_data,
    create_unified_dataset,
    prepare_features
)

from prism_monitor.modules.event_precursor._precursor import (
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

    # Fixed: 데이터 분할 후 prepare_features 호출하여 미래 정보 누출 방지
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # Train 데이터로 통계 계산
    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)
    # Val/Test 데이터는 Train 통계 사용
    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)
    print("=" * 40, "\n")
    
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

def precursor(datasets):
    unified_df = create_unified_dataset(datasets)

    # Fixed: 데이터 분할 후 prepare_features 호출
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # Train 데이터로 통계 계산
    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)
    # Val/Test 데이터는 Train 통계 사용
    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

    trained_model, model_scaler = run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler)
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)
    anomaly_status = run_real_time_monitoring_scenario(trained_model, model_scaler, feature_cols, test_df)
    return {
        'summary': {
            'predicted_value': float(pred_value[0]),
            'is_anomaly': anomaly_status
        }
    }
if __name__ == "__main__":
    main()