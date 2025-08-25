# from prism_monitor.modules.event_precursor._precursor import (
#     load_trained_models,
#     train_and_save_models
# )

# from prism_monitor.modules.event_precursor._precursor import (
#     run_single_output_simulation,
#     run_multi_output_simulation,
#     run_real_time_monitoring_simulation
# )

from _precursor_2 import (
    load_trained_models,
    train_and_save_models
)

from _precursor_2 import (
    run_single_output_simulation,
    run_multi_output_simulation,
    run_real_time_monitoring_simulation
)


def main():
    try:
        single_model, multi_model, scaler, feature_cols, test_df, metadata = load_trained_models()
        
        print(f"모델 훈련 일시: {metadata['model_info']['trained_at']}")
        print(f"사용 가능한 특성 수: {len(feature_cols)}")
        print(f"테스트 데이터 크기: {test_df.shape}")
        print("=" * 40, "\n")
        
        # 2. 단일 출력 모델 시뮬레이션
        pred_probs = None
        if single_model is not None:
            pred_probs = run_single_output_simulation(single_model, scaler, feature_cols, test_df)
        else:
            print("단일 출력 모델이 없어 해당 시뮬레이션을 건너뜁니다.")
        
        # 3. 다중 출력 모델 시뮬레이션
        pred_value = None
        if multi_model is not None:
            pred_value = run_multi_output_simulation(multi_model, feature_cols, test_df)
        else:
            print("다중 출력 모델이 없어 해당 시뮬레이션을 건너뜁니다.")
        
        # 4. 실시간 모니터링 시뮬레이션
        anomaly_status = None
        if single_model is not None:
            anomaly_status = run_real_time_monitoring_simulation(single_model, scaler, feature_cols, test_df)
        else:
            print("모델 저장 먼저 하세요.")
        
        print("=" * 60)
        print("시뮬레이션 완료")
        print("=" * 60)
        
        return {
            'summary': {
                'predicted_value': pred_value,
                'is_anomaly': anomaly_status,
                'prediction_probs': pred_probs
            }
        }
        
    except FileNotFoundError:
        print("먼저 train_and_save_models() 함수를 실행하여 모델을 학습하고 저장하세요.")
        return None
    
    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    import sys
    # 훈련만 실행. 데이터 불러오기, 학습, 모델 저장까지
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        data_path = '../../data/Industrial_DB_sample'
        # data_path = '/home/jonghak/agi/PRISM-Monitor/prism_monitor/data/Industrial_DB_sample'
        train_and_save_models(data_path)
    else:
    # 모델 불러와서 결과 출력
        result = main()
        if result:
            print("\n최종 결과:")
            print(f"- 예측값: {result['summary']['predicted_value']}")
            print(f"- 이상 상태: {result['summary']['is_anomaly']}")