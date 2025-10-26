import os

from src.modules.util.util import load_test_scenarios_data

def detect_anomalies_realtime(start: str, end: str,
                               target_file: str = None,  # 🆕 수정: CSV 파일 식별자 지정
                               target_process: str = None,  # 📝 DEPRECATED: 하위 호환용
                               model_dir: str = "models",
                               ):  # 🆕 CSV 파일 직접 읽기 옵션
    """
    실시간 데이터베이스 연동 이상탐지 수행

    Args:
        prism_core_db: 데이터베이스 연결 (use_csv=True면 None 가능)
        start: 시작 시간 (ISO format)
        end: 종료 시간 (ISO format)
        target_file: 🆕 탐지할 CSV 파일 (예: 'semiconductor_cmp_001', 'automotive_welding_001')
                    None이면 레거시 모드
        target_process: 📝 DEPRECATED - 하위 호환용, target_file 사용 권장
        model_dir: 모델 저장 디렉토리 (기본값: "models")
        use_csv: 로컬 CSV 파일 직접 읽기 (API 대신)

    Returns:
        (anomalies, drift_results, analysis_summary, vis_json)
    """
    print(f"🆕 실시간 이상탐지 시작 (File-Based Model Mode): {start} ~ {end}")

    # 하위 호환: target_process가 있으면 target_file로 변환
    if not target_file and target_process:
        print(f"   ⚠️  target_process는 deprecated됩니다. target_file을 사용하세요.")
        target_file = target_process

    if target_file:
        print(f"   대상 파일: {target_file}")

    # 🆕 공정별 모델 지원
    if target_file or target_process:
        return _detect_with_process_specific_model(
            prism_core_db, start, end, target_file or target_process, model_dir, use_csv=use_csv
        )
    else:
        # target이 지정되지 않으면 에러
        raise ValueError("target_file 또는 target_process를 지정해야 합니다.")
    
def _detect_with_process_specific_model(start: str, end: str,
                                        target_process: str, model_dir: str):
    """
    공정별 모델을 사용한 이상 탐지 (API 또는 CSV 기반 데이터 로딩)

    Args:
        target_process: 공정 식별자 (예: 'semi_cmp_sensors', 'semiconductor_cmp_001')
        use_csv: True이면 로컬 CSV 파일에서 데이터 로드
    """
    from prism_monitor.utils.process_model_manager import ProcessModelManager
    import pandas as pd

    # 하위 호환성을 위한 별칭
    target_file = target_process

    print(f"🔍 공정별 모델로 이상 탐지 수행: {target_process}")
    if use_csv:
        print(f"   📁 데이터 소스: 로컬 CSV 파일 (강제)")
    else:
        print(f"   🌐 데이터 소스: API 우선, 실패 시 CSV 폴백")

    try:
        # 1. ProcessModelManager 초기화
        process_model_manager = ProcessModelManager(base_model_dir=model_dir)

        # 2. 모델 로드
        try:
            model, scaler, metadata = process_model_manager.get_model_for_process(target_process)
            feature_cols = metadata['feature_columns']
            threshold = metadata['threshold']
            print(f"   ✓ {target_process} 모델 로드 완료 (features: {len(feature_cols)})")
        except Exception as e:
            print(f"   ✗ 모델 로드 실패 ({target_process}): {e}")
            return [], [], {'error': f'Model not found: {e}'}, {"error": str(e)}

        # 3. 데이터 로딩 (API 먼저 시도, 실패 시 CSV로 폴백)
        data = load_test_scenarios_data(target_process, start, end)


        if len(data) == 0:
            print(f"   ⚠️  시간 범위 내 데이터 없음")
            return [], [], {}, {"anomalies": [], "drift_results": [], "raw_data": {}}

        # 4. Feature 추출 및 전처리
        # 문자열 컬럼 자동 필터링 및 누락된 feature 처리
        numeric_feature_cols = []
        for col in feature_cols:
            if col not in data.columns:
                # 컬럼이 없으면 0으로 채움
                data[col] = 0
                numeric_feature_cols.append(col)
            else:
                # 숫자형 데이터만 포함
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_feature_cols.append(col)
                else:
                    print(f"   ⚠️  비숫자 컬럼 제외: {col} (타입: {data[col].dtype})")

        # 숫자형 feature만 사용
        if len(numeric_feature_cols) != len(feature_cols):
            print(f"   📝 Feature 조정: {len(feature_cols)} → {len(numeric_feature_cols)} (숫자형만)")
            feature_cols = numeric_feature_cols

        X_test = data[feature_cols].values
        X_test = np.nan_to_num(X_test, nan=0.0)
        X_test_scaled = scaler.transform(X_test)

        # 5. 이상탐지
        reconstructed = model.predict(X_test_scaled, verbose=0)
        mse_scores = np.mean(np.square(X_test_scaled - reconstructed), axis=1)

        # 6. 이상치 판정
        anomaly_mask = mse_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0]

        print(f"   📊 {len(anomaly_indices)}개 이상치 탐지 (전체 {len(data)}개 중)")

        # 7. 이상치 레코드 생성
        anomalies = []
        for idx in anomaly_indices:
            anomaly_record = {
                'table_name': target_file,
                'file_identifier': target_file,
                'timestamp': data.iloc[idx].get('timestamp', datetime.now()).isoformat() if hasattr(data.iloc[idx].get('timestamp'), 'isoformat') else str(data.iloc[idx].get('timestamp')),
                'equipment_id': data.iloc[idx].get('sensor_id') or data.iloc[idx].get('equipment_id', 'unknown'),
                'anomaly_type': 'autoencoder_reconstruction_error',
                'anomaly_score': float(mse_scores[idx]),
                'threshold': float(threshold),
                'severity': 'HIGH' if mse_scores[idx] > threshold * 2 else 'MEDIUM',
                'model_used': metadata['model_version'],
                'detection_method': 'file_specific_autoencoder'
            }
            anomalies.append(anomaly_record)

        # 8. 분석 요약
        analysis_summary = {
            'total_records': len(data),
            'anomalies_detected': len(anomalies),
            'target_file': target_file,
            'processing_mode': 'process_specific_model',
            'data_source': data_source,  # API 또는 CSV
            'processing_time': datetime.now().isoformat(),
            'model_version': metadata['model_version'],
            'threshold': float(threshold)
        }

        # 9. vis_json 생성
        vis_json = {
            "anomalies": convert_to_json_serializable(anomalies),
            "drift_results": [],
            "raw_data": {target_file: dataframe_to_json_serializable(data)},
            "analysis_summary": analysis_summary
        }

        print(f"✅ 파일별 이상탐지 완료: {len(anomalies)}개 이상 탐지")
        return anomalies, [], analysis_summary, vis_json

    except Exception as e:
        print(f"❌ 파일별 이상탐지 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return [], [], {'error': str(e)}, {"error": str(e)}
    

def _load_data_from_csv(target_process: str, start: str, end: str):
    """
    로컬 CSV 파일에서 데이터 로드

    Args:
        target_process: 공정 식별자
        start: 시작 시간
        end: 종료 시간

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    # 공정 이름 -> CSV 파일 경로 매핑
    process_to_csv = {
        # Semiconductor
        'semiconductor_cmp_001': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv',
        'semiconductor_etch_002': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_etch_002.csv',
        'semiconductor_deposition_003': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_deposition_003.csv',
        'semiconductor_full_004': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_full_004.csv',
        # Chemical
        'chemical_reactor_001': 'prism_monitor/test-scenarios/test_data/chemical/chemical_reactor_001.csv',
        'chemical_distillation_002': 'prism_monitor/test-scenarios/test_data/chemical/chemical_distillation_002.csv',
        'chemical_refining_003': 'prism_monitor/test-scenarios/test_data/chemical/chemical_refining_003.csv',
        'chemical_full_004': 'prism_monitor/test-scenarios/test_data/chemical/chemical_full_004.csv',
        # Automotive
        'automotive_welding_001': 'prism_monitor/test-scenarios/test_data/automotive/automotive_welding_001.csv',
        'automotive_painting_002': 'prism_monitor/test-scenarios/test_data/automotive/automotive_painting_002.csv',
        'automotive_press_003': 'prism_monitor/test-scenarios/test_data/automotive/automotive_press_003.csv',
        'automotive_assembly_004': 'prism_monitor/test-scenarios/test_data/automotive/automotive_assembly_004.csv',
        # Battery
        'battery_formation_001': 'prism_monitor/test-scenarios/test_data/battery/battery_formation_001.csv',
        'battery_coating_002': 'prism_monitor/test-scenarios/test_data/battery/battery_coating_002.csv',
        'battery_aging_003': 'prism_monitor/test-scenarios/test_data/battery/battery_aging_003.csv',
        'battery_production_004': 'prism_monitor/test-scenarios/test_data/battery/battery_production_004.csv',
        # Steel
        'steel_rolling_001': 'prism_monitor/test-scenarios/test_data/steel/steel_rolling_001.csv',
        'steel_converter_002': 'prism_monitor/test-scenarios/test_data/steel/steel_converter_002.csv',
        'steel_casting_003': 'prism_monitor/test-scenarios/test_data/steel/steel_casting_003.csv',
        'steel_production_004': 'prism_monitor/test-scenarios/test_data/steel/steel_production_004.csv',
    }

    # CSV 파일 경로 찾기
    csv_path = process_to_csv.get(target_process)
    if not csv_path:
        # 대문자로 시도
        csv_path = process_to_csv.get(target_process.upper())

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {target_process}")

    print(f"   📂 CSV 파일 로드: {csv_path}")

    # CSV 파일 읽기
    data = pd.read_csv(csv_path)
    print(f"   ✓ 로드 완료: {len(data)} 행")

    # 컬럼명 소문자 변환
    data.columns = data.columns.str.lower()

    # Timestamp 필터링
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        start_time = pd.to_datetime(start, utc=True)
        end_time = pd.to_datetime(end, utc=True)
        data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
        print(f"   ✓ 시간 필터링 완료: {len(data)} 행 (시간 범위: {start} ~ {end})")

    return data