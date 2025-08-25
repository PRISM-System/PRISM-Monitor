from prism_monitor.modules.event_precursor._precursor import LSTMRegressor, load_and_preprocess_data, create_sequences, \
                                    prepare_training_data, train_model, evaluate_model, predict_anomaly


def precurse():
    CSV_FILE = '/home/jonghak/agi/PRISM-Monitor/prism_monitor/data/local/SEMI_PHOTO_SENSORS.csv'
    TARGET_COLUMN = 'ALIGNMENT_ERROR_Y'
    TIME_STEPS = 5
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    OUTPUT_SIZE = 1
    EPOCHS = 30
    LEARNING_RATE = 0.001
    ANOMALY_THRESHOLD = 1.5
    
    # 1. 데이터 로드 및 전처리
    print("=== 데이터 로드 및 전처리 ===")
    scaled_features, feature_names, scaler = load_and_preprocess_data(CSV_FILE)
    print(f"데이터 형태: {scaled_features.shape}")
    print(f"특성 개수: {len(feature_names)}")
    
    # 2. 시퀀스 데이터 생성
    print("\n=== 시퀀스 데이터 생성 ===")
    X, y = create_sequences(scaled_features, feature_names, TARGET_COLUMN, time_steps=TIME_STEPS)
    print(f"시퀀스 데이터 형태 - X: {X.shape}, y: {y.shape}")
    
    # 3. 훈련/테스트 데이터 준비
    print("\n=== 훈련/테스트 데이터 준비 ===")
    X_train, X_test, y_train, y_test = prepare_training_data(X, y)
    print(f"Train Data: {X_train.shape}")
    print(f"Test Data: {X_test.shape}")
    
    # 4. 모델 생성
    print("\n=== 모델 생성 ===")
    INPUT_SIZE = X_train.shape[2]
    model = LSTMRegressor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    print("모델 구조:")
    print(model)
    
    # 5. 모델 훈련
    print(f"\n=== 모델 훈련 (에포크: {EPOCHS}) ===")
    model, criterion, optimizer = train_model(model, X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    # 6. 모델 평가
    print("=== 모델 평가 ===")
    test_loss = evaluate_model(model, criterion, X_test, y_test)
    
    # 7. 미래 이상 징후 예측
    print("\n=== 미래 이상 징후 예측 ===")
    predicted_value, is_anomaly = predict_anomaly(
        model, scaled_features, feature_names, TARGET_COLUMN, 
        scaler, time_steps=TIME_STEPS, anomaly_threshold=ANOMALY_THRESHOLD
    )
    
    # 8. 최종 결과 요약
    print(f"\n{'='*50}")
    print("최종 결과 요약")
    print(f"{'='*50}")
    print(f"테스트 손실 (MSE): {test_loss:.4f}")
    print(f"예측값: {predicted_value:.4f}")
    print(f"이상 징후 여부: {'예상됨' if is_anomaly else '정상'}")
    print(f"{'='*50}")
    
    return {
        'summary': {
            'test_loss': test_loss,
            'predicted_value': predicted_value,
            'is_anomaly': is_anomaly
        }
    }

if __name__ == "__main__":
    predict_anomaly()