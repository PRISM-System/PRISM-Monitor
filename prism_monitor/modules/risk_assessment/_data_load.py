import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_and_explore_data(data_base_path):
    print("Data Loading...")
    data_files = {
        'lot_manage': 'SEMI_LOT_MANAGE.csv',
        'process_history': 'SEMI_PROCESS_HISTORY.csv', 
        'param_measure': 'SEMI_PARAM_MEASURE.csv',
        'equipment_sensor': 'SEMI_EQUIPMENT_SENSOR.csv',
        'alert_config': 'SEMI_SENSOR_ALERT_CONFIG.csv',
        'photo_sensors': 'SEMI_PHOTO_SENSORS.csv',
        'etch_sensors': 'SEMI_ETCH_SENSORS.csv',
        'cvd_sensors': 'SEMI_CVD_SENSORS.csv',
        'implant_sensors': 'SEMI_IMPLANT_SENSORS.csv',
        'cmp_sensors': 'SEMI_CMP_SENSORS.csv'
    }
    
    datasets = {}
    for key, filename in data_files.items():
        file_path = os.path.join(data_base_path, filename)
        if os.path.exists(file_path):
            print(f"Loading: {filename}")
            try:
                df = pd.read_csv(file_path)
                datasets[key] = df
                print(f"  - Shape: {df.shape}")
            except Exception as e:
                print(f"  - Error: {e}")
        else:
            print(f"파일 없음: {file_path}")
    
    return datasets

def integrate_sensor_data(datasets):
    print("Integrating sensor data...")
    
    sensor_tables = ['photo_sensors', 'etch_sensors', 'cvd_sensors', 
                    'implant_sensors', 'cmp_sensors']
    
    integrated_sensors = []
    
    for table_name in sensor_tables:
        if table_name in datasets:
            df = datasets[table_name].copy()
            common_cols = ['PNO', 'EQUIPMENT_ID', 'LOT_NO', 'TIMESTAMP']
            available_common = [col for col in common_cols if col in df.columns]
            
            if available_common:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                sensor_cols = [col for col in numeric_cols if col != 'PNO']
                
                df['SENSOR_TABLE'] = table_name.replace('_sensors', '').upper()
                
                if sensor_cols:
                    df_long = df.melt(
                        id_vars=available_common + ['SENSOR_TABLE'],
                        value_vars=sensor_cols,
                        var_name='SENSOR_TYPE',
                        value_name='SENSOR_VALUE'
                    )
                    integrated_sensors.append(df_long)
                    print(f"  - {table_name}: Sensor count: {len(sensor_cols)}, Record count: {len(df)}")
    
    if integrated_sensors:
        result = pd.concat(integrated_sensors, ignore_index=True)
        print(f"Integration finish: Total records: {len(result)} sensors")
        return result
    else:
        return pd.DataFrame()

def create_unified_dataset(datasets):
    print("Creating unified dataset...")
    
    integrated_sensors = integrate_sensor_data(datasets)
    
    if 'lot_manage' in datasets:
        main_df = datasets['lot_manage'].copy()
        print(f"LOT data count: {len(main_df)} LOT")
    else:
        return pd.DataFrame()
    
    if not integrated_sensors.empty and 'LOT_NO' in integrated_sensors.columns:
        sensor_stats = integrated_sensors.groupby(['LOT_NO', 'SENSOR_TYPE'])['SENSOR_VALUE'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        sensor_features = sensor_stats.pivot_table(
            index='LOT_NO',
            columns='SENSOR_TYPE',
            values=['mean', 'std', 'min', 'max'],
            fill_value=0
        )
        
        sensor_features.columns = [f"{stat}_{sensor}" for stat, sensor in sensor_features.columns]
        sensor_features = sensor_features.reset_index()
        
        main_df = main_df.merge(sensor_features, on='LOT_NO', how='left')
        print(f"센서 특성 추가 완료: {sensor_features.shape[1]-1}개 특성")
    
    if 'process_history' in datasets:
        process_df = datasets['process_history']
        if 'LOT_NO' in process_df.columns:
            process_stats = process_df.groupby('LOT_NO').agg({
                'IN_QTY': ['mean', 'sum'],
                'OUT_QTY': ['mean', 'sum'],
            }).reset_index()
            
            process_stats.columns = [f"process_{col[0]}_{col[1]}" if col[1] else col[0] 
                                        for col in process_stats.columns]
            process_stats.columns = [col.replace('process_LOT_NO_', 'LOT_NO') for col in process_stats.columns]
            
            main_df = main_df.merge(process_stats, on='LOT_NO', how='left')
    
    if 'param_measure' in datasets:
        param_df = datasets['param_measure']
        if 'LOT_NO' in param_df.columns:
            param_stats = param_df.groupby('LOT_NO')['MEASURED_VAL'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            param_stats.columns = [f"param_{col}" if col != 'LOT_NO' else col 
                                        for col in param_stats.columns]
            
            main_df = main_df.merge(param_stats, on='LOT_NO', how='left')
    
    print(f"최종 통합 데이터셋: {main_df.shape}")
    return main_df


from functools import reduce

def create_unified_dataset_2(datasets):
    """
    여러 데이터셋을 'LOT_NO'를 기준으로 병합하여 통합 데이터셋을 생성합니다.
    OUTER JOIN을 사용하여 데이터 손실을 방지합니다.
    """
    print("Creating a unified dataset...")
    
    # 병합할 데이터프레임 리스트 (필요에 따라 추가)
    # 예시로 lot_manage, process_history, cmp_sensors를 병합합니다.
    df_list = [
        datasets['lot_manage'], 
        datasets['process_history'], 
        datasets['cmp_sensors']
        # 필요한 다른 데이터셋들을 여기에 추가하세요.
    ]
    
    # reduce 함수를 사용하여 리스트의 모든 데이터프레임을 순차적으로 병합
    # how='outer'로 설정하여 공통 LOT_NO가 없는 경우에도 데이터를 보존
    unified_df = reduce(lambda left, right: pd.merge(left, right, on='LOT_NO', how='outer'), df_list)
    
    # PNO, TIMESTAMP 등 중복될 수 있는 컬럼 이름 정리
    # 예시: 'PNO_x', 'PNO_y' -> 'PNO_lot', 'PNO_process'
    unified_df = unified_df.rename(columns={
        'PNO_x': 'PNO_lot', 
        'PNO_y': 'PNO_process',
        'TIMESTAMP_x': 'TIMESTAMP_process',
        'TIMESTAMP_y': 'TIMESTAMP_cmp'
    })

    print(f"Unified dataset created with shape: {unified_df.shape}")
    
    return unified_df


def prepare_features(df):
    print("Preparing features and preprocessing...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['PNO']
    if 'FINAL_YIELD' in numeric_cols:
        exclude_cols.append('FINAL_YIELD')
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    df_processed = df.copy()
    df_processed[feature_cols] = df_processed[feature_cols].fillna(0)
    
    if 'FINAL_YIELD' in df.columns:
        yield_threshold = df['FINAL_YIELD'].quantile(0.1)
        df_processed['is_anomaly'] = df_processed['FINAL_YIELD'] < yield_threshold
    else:
        feature_data = df_processed[feature_cols]
        z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std()).mean(axis=1)
        threshold = np.percentile(z_scores, 90)
        df_processed['is_anomaly'] = z_scores > threshold
    
    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    
    print(f"전처리 완료: {len(feature_cols)}개 특성")
    print(f"이상 LOT 비율: {df_processed['is_anomaly'].mean():.2%}")
    
    return df_processed, feature_cols, scaler