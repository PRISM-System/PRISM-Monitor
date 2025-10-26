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


def main():
    # DATA_BASE_PATH = '../../data/Industrial_DB_sample/'
    DATA_BASE_PATH = '../../test-scenarios/test_data/'

    print("=" * 60)
    print("이상 징후 예측 모듈 시작")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/6] 데이터 로딩...")
    all_datasets = load_and_explore_data(DATA_BASE_PATH)
    if not all_datasets:
        print("데이터 로딩 실패.")
        return None

    # 2. 데이터 통합
    print("\n[2/6] 데이터 통합...")
    unified_df = create_unified_dataset(all_datasets)
    if unified_df.empty:
        print("통합 데이터셋 생성 실패.")
        return None

    print(f"통합 데이터셋 생성 완료: {unified_df.shape}")

    # 3. 데이터 분할 (시계열 순서 유지)
    print("\n[3/6] 데이터 분할 (Train/Val/Test)...")
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    # 4. 특성 전처리 (미래 정보 누출 방지)
    print("\n[4/6] 특성 전처리 및 이상 레이블링...")
    print("  - Train 데이터로 통계 계산")
    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)

    print("  - Val/Test 데이터는 Train 통계 사용 (데이터 누출 방지)")
    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

    print(f"\n데이터 분할 완료:")
    print(f"  - 학습 데이터: {train_df.shape}")
    print(f"  - 검증 데이터: {val_df.shape}")
    print(f"  - 테스트 데이터: {test_df.shape}")
    print(f"  - 특성 개수: {len(feature_cols)}")
    print("=" * 60)
    
    print("\n[5/6] 모델 학습 및 예측...")

    print("\n>> 시나리오 1: 단일 출력 이상 징후 예측")
    trained_model, model_scaler = run_single_output_scenario(
        train_df, val_df, test_df, feature_cols, scaler
    )

    print("\n>> 시나리오 2: 다중 출력 센서값 및 이상 징후 동시 예측")
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)

    print("\n[6/6] 실시간 모니터링...")
    anomaly_status = '0'  # 기본값

    if trained_model is not None:
        print("\n>> 시나리오 3: 실시간 모니터링 시뮬레이션")
        anomaly_status = run_real_time_monitoring_scenario(
            trained_model, model_scaler, feature_cols, test_df
        )
    else:
        print("학습된 모델이 없어 실시간 모니터링을 건너뜁니다.")

    print("\n" + "=" * 60)
    print("이상 징후 예측 완료")
    print("=" * 60)

    return {
        'summary': {
            'predicted_value': pred_value,
            'is_anomaly': anomaly_status
        }
    }

def precursor(datasets):

    print("=" * 60)
    print("Precursor 모듈 실행")
    print("=" * 60)

    # 1. 데이터 통합
    print("\n[1/5] 데이터 통합...")
    unified_df = create_unified_dataset(datasets)

    if unified_df.empty:
        print("통합 데이터셋이 비어있습니다.")
        return {
            'summary': {
                'predicted_value': 0.0,
                'is_anomaly': '0'
            },
            'error': '데이터셋 통합 실패'
        }

    print(f"통합 데이터셋: {unified_df.shape}")

    print("\n[2/5] 데이터 분할...")
    train_val_df, test_df = train_test_split(unified_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)

    print("\n[3/5] 특성 전처리...")

    train_df, feature_cols, scaler, train_stats = prepare_features(train_df, train_stats=None)

    val_df, _, _, _ = prepare_features(val_df, train_stats=train_stats)
    test_df, _, _, _ = prepare_features(test_df, train_stats=train_stats)

    print(f"특성 개수: {len(feature_cols)}")

    print("\n[4/5] 모델 학습 및 예측...")

    print("  - 단일 출력 모델 학습 중...")
    trained_model, model_scaler = run_single_output_scenario(
        train_df, val_df, test_df, feature_cols, scaler
    )
    print("  - 다중 출력 모델 학습 중...")
    pred_value = run_multi_output_scenario(train_df, val_df, test_df, feature_cols)

    print("\n[5/5] 실시간 모니터링...")
    anomaly_status = '0'  # 기본값

    if trained_model is not None:
        anomaly_status = run_real_time_monitoring_scenario(
            trained_model, model_scaler, feature_cols, test_df
        )
    else:
        print("모델 학습 실패 - 기본값 반환")

    print("\n" + "=" * 60)
    print(f"Precursor 완료 - 이상 상태: {anomaly_status}")
    print("=" * 60)

    return {
        'summary': {
            'predicted_value': float(pred_value[0]) if pred_value is not None else 0.0,
            'is_anomaly': anomaly_status
        }
    }

if __name__ == "__main__":
    main()