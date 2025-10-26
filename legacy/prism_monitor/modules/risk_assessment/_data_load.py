import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_and_explore_data(data_base_path):
    print("Data Loading...")
    data_files = {
        'semi_lot_manage': 'SEMI_LOT_MANAGE.csv',
        'semi_process_history': 'SEMI_PROCESS_HISTORY.csv', 
        'semi_param_measure': 'SEMI_PARAM_MEASURE.csv',
        'semi_equipment_sensor': 'SEMI_EQUIPMENT_SENSOR.csv',
        'semi_alert_config': 'SEMI_SENSOR_ALERT_CONFIG.csv',
        'semi_photo_sensors': 'SEMI_PHOTO_SENSORS.csv',
        'semi_etch_sensors': 'SEMI_ETCH_SENSORS.csv',
        'semi_cvd_sensors': 'SEMI_CVD_SENSORS.csv',
        'semi_implant_sensors': 'SEMI_IMPLANT_SENSORS.csv',
        'semi_cmp_sensors': 'SEMI_CMP_SENSORS.csv'
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
    
    sensor_tables = ['semi_photo_sensors', 'semi_etch_sensors', 'semi_cvd_sensors', 
                    'semi_implant_sensors', 'semi_cmp_sensors']
    
    integrated_sensors = []
    
    for table_name in sensor_tables:
        if table_name in datasets:
            df = datasets[table_name].copy()
            common_cols = ['pno', 'equipment_id', 'lot_no', 'timestamp']
            available_common = [col for col in common_cols if col in df.columns]
            
            if available_common:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                sensor_cols = [col for col in numeric_cols if col != 'PNO']
                
                df['sensor_table'] = table_name.replace('_sensors', '')
                if sensor_cols:
                    df_long = df.melt(
                        id_vars=available_common + ['sensor_table'],
                        value_vars=sensor_cols,
                        var_name='sensor_type',
                        value_name='sensor_value'
                    )
                    integrated_sensors.append(df_long)
                    print(f"  - {table_name}: Sensor count: {len(sensor_cols)}, Record count: {len(df)}")
    
    if integrated_sensors:
        result = pd.concat(integrated_sensors, ignore_index=True)
        print(f"Integration finish: Total records: {len(result)} sensors")
        return result
    else:
        return pd.DataFrame()

# def create_unified_dataset(datasets):
#     print("Creating unified dataset...")
    
#     integrated_sensors = integrate_sensor_data(datasets)
    
#     if 'lot_manage' in datasets:
#         main_df = datasets['lot_manage'].copy()
#         print(f"LOT data count: {len(main_df)} LOT")
#     else:
#         return pd.DataFrame()
    
#     if not integrated_sensors.empty and 'LOT_NO' in integrated_sensors.columns:
#         sensor_stats = integrated_sensors.groupby(['LOT_NO', 'SENSOR_TYPE'])['SENSOR_VALUE'].agg([
#             'mean', 'std', 'min', 'max', 'count'
#         ]).reset_index()
        
#         sensor_features = sensor_stats.pivot_table(
#             index='LOT_NO',
#             columns='SENSOR_TYPE',
#             values=['mean', 'std', 'min', 'max'],
#             fill_value=0
#         )
        
#         sensor_features.columns = [f"{stat}_{sensor}" for stat, sensor in sensor_features.columns]
#         sensor_features = sensor_features.reset_index()
        
#         main_df = main_df.merge(sensor_features, on='LOT_NO', how='left')
#         print(f"센서 특성 추가 완료: {sensor_features.shape[1]-1}개 특성")
    
#     if 'process_history' in datasets:
#         process_df = datasets['process_history']
#         if 'LOT_NO' in process_df.columns:
#             process_stats = process_df.groupby('LOT_NO').agg({
#                 'IN_QTY': ['mean', 'sum'],
#                 'OUT_QTY': ['mean', 'sum'],
#             }).reset_index()
            
#             process_stats.columns = [f"process_{col[0]}_{col[1]}" if col[1] else col[0] 
#                                         for col in process_stats.columns]
#             process_stats.columns = [col.replace('process_LOT_NO_', 'LOT_NO') for col in process_stats.columns]
            
#             main_df = main_df.merge(process_stats, on='LOT_NO', how='left')
    
#     if 'param_measure' in datasets:
#         param_df = datasets['param_measure']
#         if 'LOT_NO' in param_df.columns:
#             param_stats = param_df.groupby('LOT_NO')['MEASURED_VAL'].agg([
#                 'mean', 'std', 'min', 'max'
#             ]).reset_index()
            
#             param_stats.columns = [f"param_{col}" if col != 'LOT_NO' else col 
#                                         for col in param_stats.columns]
            
#             main_df = main_df.merge(param_stats, on='LOT_NO', how='left')
    
#     print(f"최종 통합 데이터셋: {main_df.shape}")
#     return main_df


from functools import reduce

def create_unified_dataset(datasets):
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
    unified_df = reduce(lambda left, right: pd.merge(left, right, on='lot_no', how='outer'), df_list)
    
    # PNO, TIMESTAMP 등 중복될 수 있는 컬럼 이름 정리
    # 예시: 'PNO_x', 'PNO_y' -> 'PNO_lot', 'PNO_process'
    unified_df = unified_df.rename(columns={
        'pno_x': 'pno_lot', 
        'pno_y': 'pno_process',
        'timestamp_x': 'timestamp_process',
        'timestamp_y': 'timestamp_cmp'
    })

    print(f"Unified dataset created with shape: {unified_df.shape}")
    
    return unified_df


def prepare_features(df):
    print("Preparing features and preprocessing...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['pno']
    if 'final_yield' in numeric_cols:
        exclude_cols.append('final_yield')
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    df_processed = df.copy()
    df_processed[feature_cols] = df_processed[feature_cols].fillna(0)
    
    if 'final_yield' in df.columns:
        yield_threshold = df['final_yield'].quantile(0.1)
        df_processed['is_anomaly'] = df_processed['final_yield'] < yield_threshold
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