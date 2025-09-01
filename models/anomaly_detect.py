import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import os
import json
import io
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tinydb import TinyDB, Query
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용

warnings.filterwarnings('ignore')

class ModelManager:
    """
    모델 저장, 로딩, 관리 클래스
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_metadata_file = os.path.join(model_dir, "model_metadata.json")
        self.scaler_file = os.path.join(model_dir, "scaler.pkl")
        
    def save_model(self, model, scaler, feature_cols: List[str], threshold: float, 
                   training_data_info: Dict, performance_metrics: Dict):
        """
        모델과 관련 정보 저장
        """
        try:
            # 모델 저장
            model_file = os.path.join(self.model_dir, "autoencoder_model.h5")
            model.save(model_file)
            
            # 스케일러 저장
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 메타데이터 저장
            metadata = {
                'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'training_timestamp': datetime.now().isoformat(),
                'feature_columns': feature_cols,
                'threshold': threshold,
                'training_data_info': training_data_info,
                'performance_metrics': performance_metrics,
                'model_file': model_file,
                'scaler_file': self.scaler_file
            }
            
            with open(self.model_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"모델 저장 완료: {model_file}")
            return True
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self) -> Tuple[Optional[keras.Model], Optional[StandardScaler], Optional[Dict]]:
        """
        저장된 모델과 관련 정보 로드
        """
        try:
            if not os.path.exists(self.model_metadata_file):
                print("저장된 모델이 없습니다.")
                return None, None, None
            
            # 메타데이터 로드
            with open(self.model_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # 모델 파일 존재 확인
            model_file = metadata.get('model_file')
            if not os.path.exists(model_file):
                print(f"모델 파일이 존재하지 않음: {model_file}")
                return None, None, None
            
            # 모델 로드
            from tensorflow.keras.metrics import MeanSquaredError
            model = keras.models.load_model(model_file, custom_objects={"mse": MeanSquaredError()})
            
            # 스케일러 로드
            scaler = None
            if os.path.exists(self.scaler_file):
                with open(self.scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
            
            print(f"모델 로드 완료: {metadata['model_version']}")
            return model, scaler, metadata
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None, None, None
    
    def is_model_available(self) -> bool:
        """
        사용 가능한 모델이 있는지 확인
        """
        model, scaler, metadata = self.load_model()
        return model is not None and scaler is not None

class NormalStateManager:
    """
    정상 상태 데이터 관리 모듈
    """
    
    def __init__(self, storage_path: str = "normal_state_profiles.json"):
        self.storage_path = storage_path
        self.normal_profiles = self.load_profiles()
    
    def load_profiles(self) -> Dict:
        """저장된 정상 상태 프로파일 로드"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"정상 상태 프로파일 로드 실패: {e}")
        return {}
    
    def save_profiles(self):
        """정상 상태 프로파일 저장"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.normal_profiles, f, indent=2, default=str)
        except Exception as e:
            print(f"정상 상태 프로파일 저장 실패: {e}")
    
    def update_normal_profile(self, equipment_id: str, process_step: str, data: pd.DataFrame):
        """
        장비별/공정별 정상 상태 프로파일 업데이트
        """
        profile_key = f"{equipment_id}_{process_step}"
        
        # 수치형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        profile = {
            'equipment_id': equipment_id,
            'process_step': process_step,
            'last_updated': datetime.now().isoformat(),
            'sample_count': len(data),
            'statistics': {}
        }
        
        for col in numeric_cols:
            if col in data.columns and not data[col].empty:
                profile['statistics'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'median': float(data[col].median()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'q25': float(data[col].quantile(0.25)),
                    'q75': float(data[col].quantile(0.75))
                }
        
        self.normal_profiles[profile_key] = profile
        self.save_profiles()
        
        print(f"정상 상태 프로파일 업데이트: {profile_key} (샘플 {len(data)}개)")
        return profile
    
    def get_normal_profile(self, equipment_id: str, process_step: str) -> Optional[Dict]:
        """정상 상태 프로파일 조회"""
        profile_key = f"{equipment_id}_{process_step}"
        return self.normal_profiles.get(profile_key)
    
    def detect_profile_drift(self, equipment_id: str, process_step: str, current_data: pd.DataFrame) -> Dict:
        """
        정상 상태 프로파일 변화 감지
        """
        profile = self.get_normal_profile(equipment_id, process_step)
        if not profile:
            return {'status': 'no_profile', 'message': '정상 상태 프로파일이 없습니다.'}
        
        drift_results = {
            'equipment_id': equipment_id,
            'process_step': process_step,
            'drift_detected': False,
            'drift_parameters': [],
            'drift_score': 0,
            'check_timestamp': datetime.now().isoformat()
        }
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        drift_count = 0
        total_params = 0
        
        for col in numeric_cols:
            if col in profile['statistics'] and not current_data[col].empty:
                total_params += 1
                current_mean = current_data[col].mean()
                normal_mean = profile['statistics'][col]['mean']
                normal_std = profile['statistics'][col]['std']
                
                # Z-score 계산 (3-sigma 룰)
                if normal_std > 0:
                    z_score = abs((current_mean - normal_mean) / normal_std)
                    if z_score > 3:  # 3-sigma를 벗어나면 drift 감지
                        drift_count += 1
                        drift_results['drift_parameters'].append({
                            'parameter': col,
                            'current_value': float(current_mean),
                            'normal_value': float(normal_mean),
                            'z_score': float(z_score),
                            'severity': 'HIGH' if z_score > 5 else 'MEDIUM'
                        })
        
        if total_params > 0:
            drift_results['drift_score'] = (drift_count / total_params) * 100
            drift_results['drift_detected'] = drift_results['drift_score'] > 10  # 10% 이상 파라미터에 drift
        
        return drift_results
    
    def get_all_profiles_summary(self) -> Dict:
        """모든 정상 상태 프로파일 요약"""
        return {
            'total_profiles': len(self.normal_profiles),
            'profiles': [
                {
                    'key': key,
                    'equipment_id': profile['equipment_id'],
                    'process_step': profile['process_step'],
                    'last_updated': profile['last_updated'],
                    'sample_count': profile['sample_count']
                }
                for key, profile in self.normal_profiles.items()
            ]
        }

class DataValidityChecker:
    """
    실시간 데이터 정합성 검증 모듈
    """
    
    def __init__(self):
        # 각 공정별 정상 범위 정의 (스키마 기반)
        self.normal_ranges = {
            'SEMI_PHOTO_SENSORS': {
                'EXPOSURE_DOSE': (20, 40),
                'FOCUS_POSITION': (-50, 50),
                'STAGE_TEMP': (22.9, 23.1),
                'HUMIDITY': (40, 50),
                'ALIGNMENT_ERROR_X': (0, 3),
                'ALIGNMENT_ERROR_Y': (0, 3),
                'LENS_ABERRATION': (0, 5),
                'ILLUMINATION_UNIFORMITY': (98, 100),
                'RETICLE_TEMP': (22.95, 23.05)
            },
            'SEMI_ETCH_SENSORS': {
                'RF_POWER_SOURCE': (500, 2000),
                'RF_POWER_BIAS': (50, 500),
                'CHAMBER_PRESSURE': (5, 200),
                'GAS_FLOW_CF4': (0, 200),
                'GAS_FLOW_O2': (0, 100),
                'GAS_FLOW_AR': (0, 500),
                'GAS_FLOW_CL2': (0, 200),
                'ELECTRODE_TEMP': (40, 80),
                'CHAMBER_WALL_TEMP': (60, 80),
                'HELIUM_PRESSURE': (5, 20),
                'PLASMA_DENSITY': (1e10, 1e12)
            },
            'SEMI_CVD_SENSORS': {
                'SUSCEPTOR_TEMP': (300, 700),
                'CHAMBER_PRESSURE': (0.1, 760),
                'PRECURSOR_FLOW_TEOS': (0, 500),
                'PRECURSOR_FLOW_SILANE': (0, 1000),
                'PRECURSOR_FLOW_WF6': (0, 100),
                'CARRIER_GAS_N2': (0, 20),
                'CARRIER_GAS_H2': (0, 10),
                'SHOWERHEAD_TEMP': (150, 250),
                'LINER_TEMP': (100, 200)
            },
            'SEMI_IMPLANT_SENSORS': {
                'BEAM_CURRENT': (0.1, 5000),
                'BEAM_ENERGY': (0.2, 3000),
                'TOTAL_DOSE': (1e11, 1e16),
                'IMPLANT_ANGLE': (0, 45),
                'WAFER_ROTATION': (0, 1200),
                'SOURCE_PRESSURE': (1e-6, 1e-4),
                'ANALYZER_PRESSURE': (1e-7, 1e-5),
                'END_STATION_PRESSURE': (1e-7, 1e-6),
                'BEAM_UNIFORMITY': (98, 100)
            },
            'SEMI_CMP_SENSORS': {
                'HEAD_PRESSURE': (2, 8),
                'RETAINER_PRESSURE': (2, 6),
                'PLATEN_ROTATION': (20, 150),
                'HEAD_ROTATION': (20, 150),
                'SLURRY_FLOW_RATE': (100, 300),
                'SLURRY_TEMP': (20, 25),
                'PAD_TEMP': (30, 50),
                'CONDITIONER_PRESSURE': (5, 9)
            }
        }
    
    def validate_data_integrity(self, df: pd.DataFrame, table_name: str) -> Dict:
        """
        데이터 정합성 검증
        """
        validation_results = {
            'table_name': table_name,
            'total_records': len(df),
            'missing_values': {},
            'out_of_range_values': {},
            'data_quality_score': 0,
            'anomalies': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # 1. 결측치 검사
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = {
            col: int(count) for col, count in missing_counts.items() if count > 0
        }
        
        # 2. 범위 검사 (정상 범위가 정의된 경우)
        if table_name in self.normal_ranges:
            ranges = self.normal_ranges[table_name]
            for col, (min_val, max_val) in ranges.items():
                if col in df.columns:
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                    if len(out_of_range) > 0:
                        validation_results['out_of_range_values'][col] = {
                            'count': len(out_of_range),
                            'percentage': len(out_of_range) / len(df) * 100,
                            'expected_range': f"{min_val} - {max_val}",
                            'actual_range': f"{df[col].min():.3f} - {df[col].max():.3f}"
                        }
        
        # 3. 데이터 품질 점수 계산
        total_issues = sum(len(issues) for issues in validation_results['out_of_range_values'].values())
        total_missing = sum(validation_results['missing_values'].values())
        total_problems = total_issues + total_missing
        
        if len(df) > 0:
            validation_results['data_quality_score'] = max(0, 100 - (total_problems / len(df) * 100))
        
        # 4. 심각한 이상 검출
        critical_threshold = 5  # 5% 이상 문제가 있으면 critical
        for col, info in validation_results['out_of_range_values'].items():
            if info['percentage'] > critical_threshold:
                validation_results['anomalies'].append({
                    'type': 'critical_out_of_range',
                    'column': col,
                    'severity': 'HIGH',
                    'description': f"{col}에서 {info['percentage']:.1f}%의 데이터가 정상 범위를 벗어남"
                })
        
        return validation_results
    
    def preprocess_and_clean(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        데이터 정제 및 전처리
        """
        df_clean = df.copy()
        
        # 1. 수치형 컬럼 식별
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # 2. 결측치 처리
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                # 센서 데이터는 이전 값으로 채우기 (forward fill)
                df_clean[col] = df_clean[col].fillna(method='ffill')
                # 여전히 NaN이 있으면 median으로 채우기
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 3. 이상치 처리 (정상 범위 기반)
        if table_name in self.normal_ranges:
            ranges = self.normal_ranges[table_name]
            for col, (min_val, max_val) in ranges.items():
                if col in df_clean.columns:
                    # 범위를 벗어나는 값들을 경계값으로 클리핑
                    df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
        
        return df_clean

class DataSplitter:
    """
    데이터를 train/test로 분할하는 클래스
    """
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_datasets(self, datasets: Dict[str, pd.DataFrame], 
                      split_by: str = 'time') -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        데이터셋을 train/test로 분할
        
        Args:
            datasets: 전체 데이터셋
            split_by: 분할 방식 ('time' 또는 'random')
            
        Returns:
            Tuple[train_datasets, test_datasets]
        """
        train_datasets = {}
        test_datasets = {}
        
        for table_name, df in datasets.items():
            if len(df) == 0:
                continue
                
            if split_by == 'time' and any(col in df.columns for col in ['TIMESTAMP', 'CREDATE', 'START_TIME', 'MEASURE_TIME']):
                # 시간 기준 분할
                time_col = None
                for col in ['TIMESTAMP', 'CREDATE', 'START_TIME', 'MEASURE_TIME']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if time_col:
                    df_sorted = df.sort_values(time_col)
                    split_idx = int(len(df_sorted) * (1 - self.test_size))
                    train_df = df_sorted.iloc[:split_idx].copy()
                    test_df = df_sorted.iloc[split_idx:].copy()
                else:
                    # 시간 컬럼이 없으면 랜덤 분할
                    train_df, test_df = train_test_split(
                        df, test_size=self.test_size, random_state=self.random_state
                    )
            else:
                # 랜덤 분할
                train_df, test_df = train_test_split(
                    df, test_size=self.test_size, random_state=self.random_state
                )
            
            train_datasets[table_name] = train_df
            test_datasets[table_name] = test_df
            
            print(f"{table_name}: Train {len(train_df)}, Test {len(test_df)}")
        
        return train_datasets, test_datasets

class SemiconductorRealDataDetector:
    """
    실제 반도체 공정 데이터를 활용한 이상탐지 딥러닝 모듈
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
        self.threshold = None
        self.data_files = {
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
        
    def _default_config(self):
        """기본 설정"""
        return {
            'sequence_length': 60,
            'contamination': 0.05,
            'threshold_percentile': 95,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
        }

    def load_local_data_and_explore(self, data_base_path):
        """
        실제 데이터 로딩 및 탐색
        """
        print("실제 반도체 데이터 로딩 및 탐색 중...")
        
        datasets = {}
        for key, filename in self.data_files.items():
            file_path = os.path.join(data_base_path, filename)
            if os.path.exists(file_path):
                print(f"로딩 중: {filename}")
                try:
                    df = pd.read_csv(file_path)
                    datasets[key] = df
                    print(f"  - 크기: {df.shape}")
                    print(f"  - 컬럼: {list(df.columns)}")
                    print()
                except Exception as e:
                    print(f"  - 오류: {e}")
            else:
                print(f"파일 없음: {file_path}")
        
        self.raw_datasets = datasets
        return datasets

    def integrate_sensor_data(self, datasets):
        """
        여러 센서 테이블을 통합하여 하나의 센서 데이터셋 생성
        """
        print("센서 데이터 통합 중...")
        
        sensor_tables = ['semi_photo_sensors', 'semi_etch_sensors', 'semi_cvd_sensors', 
                        'semi_implant_sensors', 'semi_cmp_sensors']
        
        integrated_sensors = []
        
        for table_name in sensor_tables:
            if table_name in datasets:
                df = datasets[table_name].copy()
                
                # 공통 컬럼들만 선택
                common_cols = ['pno', 'equipment_id', 'lot_no', 'timestamp']
                available_common = [col for col in common_cols if col in df.columns]
                
                if available_common:
                    # 수치형 센서 컬럼들 찾기
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    sensor_cols = [col for col in numeric_cols if col != 'pno']
                    
                    # 테이블 정보 추가
                    df['sensor_table'] = table_name.replace('_sensors', '')
                    
                    # 센서값들을 하나의 컬럼으로 변환
                    if sensor_cols:
                        df_long = df.melt(
                            id_vars=available_common + ['sensor_table'],
                            value_vars=sensor_cols,
                            var_name='sensor_type',
                            value_name='sensor_value'
                        )
                        integrated_sensors.append(df_long)
                        print(f"  - {table_name}: {len(sensor_cols)}개 센서, {len(df)}개 레코드")
        
        if integrated_sensors:
            result = pd.concat(integrated_sensors, ignore_index=True)
            print(f"통합 완료: 총 {len(result)}개 센서 레코드")
            return result
        else:
            print("통합할 센서 데이터가 없습니다.")
            return pd.DataFrame()

    def create_unified_dataset(self, datasets):
        """
        모든 테이블을 통합하여 분석용 데이터셋 생성
        """
        print("통합 데이터셋 생성 중...")
        
        # 1. 센서 데이터 통합
        integrated_sensors = self.integrate_sensor_data(datasets)
        
        # 2. LOT 관리 데이터 기준으로 통합
        if 'semi_lot_manage' in datasets:
            main_df = datasets['semi_lot_manage'].copy()
            print(f"기본 LOT 데이터: {len(main_df)}개 LOT")
        else:
            print("LOT 관리 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 3. 각 LOT별 센서 통계 생성
        if not integrated_sensors.empty and 'lot_no' in integrated_sensors.columns:
            sensor_stats = integrated_sensors.groupby(['lot_no', 'sensor_type'])['sensor_value'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            sensor_features = sensor_stats.pivot_table(
                index='lot_no',
                columns='sensor_type',
                values=['mean', 'std', 'min', 'max'],
                fill_value=0
            )
            
            sensor_features.columns = [f"{stat}_{sensor}" for stat, sensor in sensor_features.columns]
            sensor_features = sensor_features.reset_index()
            
            main_df = main_df.merge(sensor_features, on='lot_no', how='left')
            print(f"센서 특성 추가 완료: {sensor_features.shape[1]-1}개 특성")
        
        # 4. 공정 이력 데이터 통합
        if 'semi_process_history' in datasets:
            process_df = datasets['semi_process_history']
            if 'lot_no' in process_df.columns:
                process_stats = process_df.groupby('lot_no').agg({
                    'in_qty': ['mean', 'sum'],
                    'out_qty': ['mean', 'sum'],
                }).reset_index()
                
                process_stats.columns = [f"process_{col[0]}_{col[1]}" if col[1] else col[0] 
                                       for col in process_stats.columns]
                process_stats.columns = [col.replace('process_lot_no_', 'lot_no') for col in process_stats.columns]
                
                main_df = main_df.merge(process_stats, on='lot_no', how='left')
                print(f"공정 이력 특성 추가 완료")
        
        # 5. 파라미터 측정 데이터 통합
        if 'semi_param_measure' in datasets:
            param_df = datasets['semi_param_measure']
            if 'lot_no' in param_df.columns:
                param_stats = param_df.groupby('lot_no')['measured_val'].agg([
                    'mean', 'std', 'min', 'max'
                ]).reset_index()
                
                param_stats.columns = [f"param_{col}" if col != 'lot_no' else col 
                                     for col in param_stats.columns]
                
                main_df = main_df.merge(param_stats, on='lot_no', how='left')
                print(f"파라미터 측정 특성 추가 완료")
        
        print(f"최종 통합 데이터셋: {main_df.shape}")
        return main_df

    def prepare_features(self, df):
        """
        특성 준비 및 전처리
        """
        print("특성 준비 및 전처리 중...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_cols = ['pno']
        if 'final_yield' in numeric_cols:
            exclude_cols.append('final_yield')
            
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df_processed = df.copy()
        df_processed[feature_cols] = df_processed[feature_cols].fillna(0)
        
        # 이상치 라벨 생성
        if 'final_yield' in df.columns:
            yield_threshold = df['final_yield'].quantile(0.1)
            df_processed['is_anomaly'] = df_processed['final_yield'] < yield_threshold
        else:
            feature_data = df_processed[feature_cols]
            z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std()).mean(axis=1)
            threshold = np.percentile(z_scores, 90)
            df_processed['is_anomaly'] = z_scores > threshold
        
        print(f"전처리 완료: {len(feature_cols)}개 특성")
        print(f"이상 LOT 비율: {df_processed['is_anomaly'].mean():.2%}")
        
        return df_processed, feature_cols

    def build_autoencoder(self, input_dim):
        """
        Autoencoder 모델 구축
        """
        input_layer = layers.Input(shape=(input_dim,))
        
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(32, activation='relu')(encoded)
        
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder

    def train_model(self, train_datasets):
        """
        Train 데이터로 모델 훈련
        """
        print("Train 데이터로 모델 훈련 시작...")
        
        # 통합 데이터셋 생성
        unified_df = self.create_unified_dataset(train_datasets)
        if unified_df.empty:
            print("훈련용 데이터가 없습니다.")
            return None, []
        
        # 특성 준비
        processed_df, feature_cols = self.prepare_features(unified_df)
        
        # 정상 데이터만 사용하여 학습
        normal_data = processed_df[~processed_df['is_anomaly']]
        X_normal = normal_data[feature_cols].values
        
        if len(X_normal) < 50:
            print("훈련용 정상 데이터가 부족합니다.")
            return None, feature_cols
        
        print(f"학습 데이터: {len(X_normal)}개 정상 샘플")
        
        # 스케일링
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        
        # Autoencoder 학습
        model = self.build_autoencoder(len(feature_cols))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        history = model.fit(
            X_normal_scaled, X_normal_scaled,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 임계값 계산
        reconstructed = model.predict(X_normal_scaled, verbose=0)
        mse_normal = np.mean(np.square(X_normal_scaled - reconstructed), axis=1)
        self.threshold = np.percentile(mse_normal, self.config['threshold_percentile'])
        
        self.models['autoencoder'] = model
        self.history['autoencoder'] = history
        
        print(f"훈련 완료. 임계값: {self.threshold:.4f}")
        
        return model, feature_cols

    def predict_with_trained_model(self, test_datasets, feature_cols):
        """
        훈련된 모델로 Test 데이터 예측
        """
        print("훈련된 모델로 Test 데이터 예측 시작...")
        
        if 'autoencoder' not in self.models:
            print("훈련된 모델이 없습니다.")
            return None
        
        # 통합 데이터셋 생성
        unified_df = self.create_unified_dataset(test_datasets)
        if unified_df.empty:
            print("테스트 데이터가 없습니다.")
            return None
        
        # 동일한 전처리 과정
        processed_df, _ = self.prepare_features(unified_df)
        
        # 동일한 feature columns 사용
        X_test = processed_df[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # 이상탐지 수행
        reconstructed = self.models['autoencoder'].predict(X_test_scaled, verbose=0)
        mse_scores = np.mean(np.square(X_test_scaled - reconstructed), axis=1)
        
        # 결과 저장
        result_df = processed_df.copy()
        result_df['anomaly_score'] = mse_scores
        result_df['predicted_anomaly'] = mse_scores > self.threshold
        result_df['confidence'] = (mse_scores - self.threshold) / self.threshold
        
        print(f"예측 완료: {len(result_df)}개 LOT 중 {result_df['predicted_anomaly'].sum()}개 이상 탐지")
        
        return result_df

    def visualize_results(self, df_result):
        """
        결과 시각화 (SVG)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 이상 점수 분포
        if 'predicted_anomaly' in df_result.columns:
            normal_scores = df_result[~df_result['predicted_anomaly']]['anomaly_score']
            anomaly_scores = df_result[df_result['predicted_anomaly']]['anomaly_score']
            
            axes[0, 0].hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
            axes[0, 0].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
            axes[0, 0].axvline(self.threshold, color='red', linestyle='--', label='Threshold')
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Anomaly Score Distribution')
            axes[0, 0].legend()
        
        # 2. 수율 vs 이상 점수
        if 'final_yield' in df_result.columns and 'predicted_anomaly' in df_result.columns:
            colors = ['red' if x else 'blue' for x in df_result['predicted_anomaly']]
            axes[0, 1].scatter(df_result['final_yield'], df_result['anomaly_score'], 
                             c=colors, alpha=0.6, s=20)
            axes[0, 1].set_xlabel('Final Yield (%)')
            axes[0, 1].set_ylabel('Anomaly Score')
            axes[0, 1].set_title('Yield vs Anomaly Score')
        
        # 3. 혼동 행렬
        if 'is_anomaly' in df_result.columns and 'predicted_anomaly' in df_result.columns:
            cm = confusion_matrix(df_result['is_anomaly'], df_result['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
            axes[1, 0].set_title('Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # 4. 학습 손실
        if 'autoencoder' in self.history:
            history = self.history['autoencoder']
            axes[1, 1].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[1, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # SVG로 저장
        svg_buffer = io.StringIO()
        plt.savefig(svg_buffer, format='svg')
        svg_content = svg_buffer.getvalue()
        svg_buffer.close()
        
        plt.close(fig)
        
        return svg_content

    def analyze_results(self, df_result):
        """
        결과 분석 및 요약
        """
        print("결과 분석 중...")
        
        total_lots = len(df_result)
        detected_anomalies = df_result['predicted_anomaly'].sum() if 'predicted_anomaly' in df_result.columns else 0
        actual_anomalies = df_result['is_anomaly'].sum() if 'is_anomaly' in df_result.columns else 0
        
        print(f"\n=== 이상탐지 결과 요약 ===")
        print(f"전체 LOT 수: {total_lots}")
        if 'is_anomaly' in df_result.columns:
            print(f"실제 이상 LOT: {actual_anomalies} ({actual_anomalies/total_lots:.1%})")
        print(f"탐지된 이상 LOT: {detected_anomalies} ({detected_anomalies/total_lots:.1%})")
        
        # 성능 평가
        if 'is_anomaly' in df_result.columns and 'predicted_anomaly' in df_result.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            recall = recall_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            f1 = f1_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            
            print(f"\n=== 성능 지표 ===")
            print(f"정밀도 (Precision): {precision:.3f}")
            print(f"재현율 (Recall): {recall:.3f}")
            print(f"F1 점수: {f1:.3f}")
        
        # 이상 LOT 상세 분석
        if 'predicted_anomaly' in df_result.columns:
            anomaly_lots = df_result[df_result['predicted_anomaly']]
            if len(anomaly_lots) > 0:
                print(f"\n=== 이상 LOT 분석 ===")
                print("이상 점수가 높은 상위 5개 LOT:")
                top_anomalies = anomaly_lots.nlargest(5, 'anomaly_score')
                for _, row in top_anomalies.iterrows():
                    print(f"  LOT {row['lot_no']}: 점수 {row['anomaly_score']:.4f}")
                    if 'final_yield' in row:
                        print(f"    수율: {row['final_yield']:.1f}%")
        
        return df_result[df_result['predicted_anomaly']].to_dict("records") if 'predicted_anomaly' in df_result.columns else []

def load_data_from_database(prism_core_db, start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    데이터베이스에서 실시간 데이터 로딩
    
    Args:
        prism_core_db: PrismCoreDataBase 인스턴스
        start: 시작 시간 (ISO format)
        end: 종료 시간 (ISO format)
        
    Returns:
        Dict[str, pd.DataFrame]: 테이블명별 데이터프레임
    """
    print(f"데이터베이스에서 실시간 데이터 로딩: {start} ~ {end}")
    
    start_time = pd.to_datetime(start, utc=True)
    end_time = pd.to_datetime(end, utc=True)
    
    datasets = {}
    data_validator = DataValidityChecker()
    
    try:
        # 데이터베이스에서 모든 테이블 조회
        available_tables = prism_core_db.get_tables()
        print(f"사용 가능한 테이블: {len(available_tables)}개")
        
        for table_name in available_tables:
            print(f"로딩 중: {table_name}")
            
            try:
                # 테이블 데이터 조회
                df = prism_core_db.get_table_data(table_name)
                
                if df is not None and len(df) > 0:
                    # 시간 컬럼이 있는 경우 필터링
                    time_columns = ['timestamp', 'TIMESTAMP', 'credate', 'CREDATE', 
                                  'start_time', 'START_TIME', 'end_time', 'END_TIME',
                                  'measure_time', 'MEASURE_TIME']
                    
                    time_col = None
                    for col in time_columns:
                        if col in df.columns:
                            time_col = col
                            break
                    
                    if time_col:
                        # 시간 컬럼 변환 및 필터링
                        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                        
                        # 시간 범위 필터링
                        before_filter = len(df)
                        df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
                        after_filter = len(df)
                        
                        print(f"  - 시간 필터링: {before_filter} → {after_filter}개 레코드")
                    
                    if len(df) > 0:
                        # 데이터 정합성 검증 및 전처리
                        df_clean = preprocess_table(df, table_name)
                        datasets[table_name] = df_clean
                        
                        print(f"  - 최종 데이터: {len(df_clean)}개 레코드")
                    else:
                        print(f"  - 해당 기간 데이터 없음")
                        
                else:
                    print(f"  - 테이블이 비어있음")
                    
            except Exception as table_error:
                print(f"  - 테이블 {table_name} 로딩 실패: {table_error}")
                continue
                
    except Exception as e:
        print(f"데이터베이스 접근 오류: {e}")
        
    print(f"총 {len(datasets)}개 테이블 로딩 완료")
    return datasets

def preprocess_table(df: pd.DataFrame, table_name: str = 'GENERIC') -> pd.DataFrame:
    """
    테이블별 정합성 검증 및 정상화
    
    Args:
        df: 원본 데이터프레임
        table_name: 테이블명 (검증 규칙 적용용)
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    validator = DataValidityChecker()
    
    # 데이터 정합성 검증
    validation_result = validator.validate_data_integrity(df, table_name)
    
    # 심각한 품질 문제가 있는 경우 경고
    if validation_result['data_quality_score'] < 70:
        print(f"경고: {table_name} 데이터 품질이 낮습니다 ({validation_result['data_quality_score']:.1f}점)")
    
    # 데이터 정제 및 전처리
    df_clean = validator.preprocess_and_clean(df, table_name)
    
    return df_clean

class EnhancedSemiconductorRealTimeMonitor:
    """
    시나리오 B: 사전 훈련된 모델을 사용하는 빠른 이상탐지 시스템 (실시간 데이터 수집 포함)
    """
    
    def __init__(self, config=None, model_dir="models"):
        self.config = config or self._default_config()
        self.detector = SemiconductorRealDataDetector(config)
        self.model_manager = ModelManager(model_dir)
        self.data_validator = DataValidityChecker()
        self.data_splitter = DataSplitter()
        self.normal_state_manager = NormalStateManager()
        self.validation_results = []
        
        # 테이블 매핑 (데이터베이스 테이블명 → 내부 키)
        self.table_mapping = {
            'SEMI_LOT_MANAGE': 'semi_lot_manage',
            'SEMI_PROCESS_HISTORY': 'semi_process_history', 
            'SEMI_PARAM_MEASURE': 'semi_param_measure',
            'SEMI_EQUIPMENT_SENSOR': 'semi_equipment_sensor',
            'SEMI_PHOTO_SENSORS': 'semi_photo_sensors',
            'SEMI_ETCH_SENSORS': 'semi_etch_sensors',
            'SEMI_CVD_SENSORS': 'semi_cvd_sensors',
            'SEMI_IMPLANT_SENSORS': 'semi_implant_sensors',
            'SEMI_CMP_SENSORS': 'semi_cmp_sensors',
            'SEMI_SENSOR_ALERT_CONFIG': 'semi_alert_config'
        }
        
    def _default_config(self):
        return {
            'test_size': 0.3,
            'random_state': 42,
            'split_by': 'time'  # 'time' or 'random'
        }
    
    def setup_and_train_model(self, data_path: str):
        """
        데이터 로딩, 분할 및 모델 훈련 (최초 1회 실행)
        """
        print("=== 모델 설정 및 훈련 단계 ===")
        
        # 1. 전체 데이터 로딩
        print("1. 전체 데이터 로딩 중...")
        all_datasets = self.detector.load_local_data_and_explore(data_path)
        
        if not all_datasets:
            print("로딩된 데이터가 없습니다.")
            return False
        
        # 2. 데이터 검증
        print("2. 데이터 정합성 검증 중...")
        for table_name, df in all_datasets.items():
            validation = self.data_validator.validate_data_integrity(df, table_name)
            self.validation_results.append(validation)
            df_clean = self.data_validator.preprocess_and_clean(df, table_name)
            all_datasets[table_name] = df_clean
        
        # 3. 데이터 분할
        print("3. 데이터 Train/Test 분할 중...")
        train_datasets, test_datasets = self.data_splitter.split_datasets(
            all_datasets, split_by=self.config['split_by']
        )
        
        # 4. 모델 훈련
        print("4. 모델 훈련 중...")
        model, feature_cols = self.detector.train_model(train_datasets)
        
        if model is None:
            print("모델 훈련 실패")
            return False
        
        # 5. 모델 저장
        print("5. 모델 저장 중...")
        training_info = {
            'total_tables': len(train_datasets),
            'total_train_records': sum(len(df) for df in train_datasets.values()),
            'total_test_records': sum(len(df) for df in test_datasets.values()),
            'split_method': self.config['split_by']
        }
        
        performance_metrics = {
            'threshold': self.detector.threshold,
            'feature_count': len(feature_cols)
        }
        
        success = self.model_manager.save_model(
            model, self.detector.scaler, feature_cols, 
            self.detector.threshold, training_info, performance_metrics
        )
        
        if success:
            print("모델 설정 및 훈련 완료!")
            
            # 6. Test 데이터로 성능 검증
            print("6. Test 데이터로 성능 검증 중...")
            result_df = self.detector.predict_with_trained_model(test_datasets, feature_cols)
            if result_df is not None:
                self.detector.analyze_results(result_df)
            
            return True
        else:
            print("모델 저장 실패")
            return False
    
    def fast_anomaly_detection(self, test_data_path: str = None, test_datasets: Dict = None, unified_df: pd.DataFrame = None):
        """
        시나리오 B: 저장된 모델로 빠른 이상탐지 수행
        """
        print("=== 시나리오 B: 빠른 이상탐지 수행 ===")
        
        # 1. 저장된 모델 로드
        print("1. 저장된 모델 로딩 중...")
        model, scaler, metadata = self.model_manager.load_model()
        
        if model is None:
            print("저장된 모델이 없습니다. setup_and_train_model()을 먼저 실행하세요.")
            return None, "", []
        
        self.detector.models['autoencoder'] = model
        self.detector.scaler = scaler
        self.detector.threshold = metadata['threshold']
        feature_cols = metadata['feature_columns']
        
        print(f"모델 로드 완료: {metadata['model_version']}")
        print(f"특성 수: {len(feature_cols)}")
        print(f"임계값: {metadata['threshold']}")
        
        # 2. 테스트 데이터 준비
        if unified_df is not None:
            # 동일한 전처리 과정
            processed_df, _ = self.detector.prepare_features(unified_df)
            
            # 동일한 feature columns 사용
            X_test = processed_df[feature_cols].values
            X_test_scaled = self.scaler.transform(X_test)
            
            # 이상탐지 수행
            reconstructed = model.predict(X_test_scaled, verbose=0)
            mse_scores = np.mean(np.square(X_test_scaled - reconstructed), axis=1)
            
            # 결과 저장
            result_df = processed_df.copy()
            result_df['anomaly_score'] = mse_scores
            result_df['predicted_anomaly'] = mse_scores > self.threshold
            result_df['confidence'] = (mse_scores - self.threshold) / self.threshold      
            print(f"예측 완료: {len(result_df)}개 LOT 중 {result_df['predicted_anomaly'].sum()}개 이상 탐지")
        else:
            if test_datasets is None:
                if test_data_path:
                    print("2. 테스트 데이터 로딩 중...")
                    test_datasets = self.detector.load_local_data_and_explore(test_data_path)
                else:
                    print("테스트 데이터가 제공되지 않았습니다.")
                    return None, "", []
        
            result_df = self.detector.predict_with_trained_model(test_datasets, feature_cols)
        
        if result_df is None:
            return [], "", []
        
        # 4. 결과 분석 및 시각화
        print("4. 결과 분석 및 시각화 중...")
        analysis_results = self.detector.analyze_results(result_df)
        svg_content = self.detector.visualize_results(result_df)
        
        # 5. 이상 LOT 추출
        anomalies = result_df[result_df['predicted_anomaly']] if 'predicted_anomaly' in result_df.columns else pd.DataFrame()
        anomaly_records = anomalies.to_dict('records') if len(anomalies) > 0 else []
        
        print(f"빠른 이상탐지 완료!")
        print(f"- 총 {len(result_df)}개 LOT 분석")
        print(f"- {len(anomaly_records)}개 이상 LOT 탐지")
        
        return anomaly_records, svg_content, analysis_results
    
    def get_system_status(self):
        """
        실시간 시스템 상태 조회 (정상 상태 프로파일 포함)
        """
        model_status = self.get_model_status()
        normal_state_summary = self.normal_state_manager.get_all_profiles_summary()
        
        return {
            'model_status': model_status,
            'normal_state_profiles': normal_state_summary,
            'data_quality_summary': {
                'total_validations': len(self.validation_results),
                'average_quality_score': np.mean([r['data_quality_score'] for r in self.validation_results]) if self.validation_results else 0,
                'critical_issues': sum(len(r['anomalies']) for r in self.validation_results),
                'last_validation': self.validation_results[-1]['validation_timestamp'] if self.validation_results else None
            },
            'system_timestamp': datetime.now().isoformat()
        }
    
    def get_model_status(self):
        """
        저장된 모델 상태 조회
        """
        _, _, metadata = self.model_manager.load_model()
        
        if metadata:
            return {
                'model_available': True,
                'model_version': metadata['model_version'],
                'training_timestamp': metadata['training_timestamp'],
                'feature_count': len(metadata['feature_columns']),
                'threshold': metadata['threshold'],
                'training_info': metadata['training_data_info']
            }
        else:
            return {
                'model_available': False,
                'message': '저장된 모델이 없습니다.'
            }
    
    def get_model_status(self):
        """
        저장된 모델 상태 조회
        """
        _, _, metadata = self.model_manager.load_model()
        
        if metadata:
            return {
                'model_available': True,
                'model_version': metadata['model_version'],
                'training_timestamp': metadata['training_timestamp'],
                'feature_count': len(metadata['feature_columns']),
                'threshold': metadata['threshold'],
                'training_info': metadata['training_data_info']
            }
        else:
            return {
                'model_available': False,
                'message': '저장된 모델이 없습니다.'
            }
        
    def load_real_time_data_from_database(self, prism_core_db, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        데이터베이스에서 실시간 데이터 로딩 및 정상 상태 프로파일 업데이트
        
        Args:
            prism_core_db: PrismCoreDataBase 인스턴스
            start: 시작 시간
            end: 종료 시간
            
        Returns:
            Dict[str, pd.DataFrame]: 정제된 데이터셋
        """
        print("실시간 데이터베이스에서 데이터 수집 중...")
        
        # 데이터베이스에서 실시간 데이터 로딩
        raw_datasets = load_data_from_database(prism_core_db, start, end)
        
        if not raw_datasets:
            print("로딩된 데이터가 없습니다.")
            return {}
        
        # 테이블명 매핑 (데이터베이스명 → 내부키)
        datasets = {}
        for db_table_name, df in raw_datasets.items():
            internal_key = self.table_mapping.get(db_table_name, db_table_name.lower())
            datasets[internal_key] = df
            
            # 데이터 품질 검증 결과 저장
            validation_result = self.data_validator.validate_data_integrity(df, db_table_name)
            self.validation_results.append(validation_result)
        
        # 정상 상태 프로파일 업데이트 (장비 센서 데이터가 있는 경우)
        self._update_normal_state_profiles(datasets)
        
        print(f"실시간 데이터 수집 완료: {len(datasets)}개 테이블")
        return datasets
    
    def _update_normal_state_profiles(self, datasets: Dict[str, pd.DataFrame]):
        """
        실시간 데이터로 정상 상태 프로파일 업데이트
        """
        print("정상 상태 프로파일 업데이트 중...")
        
        sensor_tables = ['semi_photo_sensors', 'semi_etch_sensors', 'semi_cvd_sensors', 
                        'semi_implant_sensors', 'semi_cmp_sensors']
        
        updated_profiles = 0
        
        for table_key in sensor_tables:
            if table_key not in datasets:
                continue
                
            df = datasets[table_key]
            
            # 장비 ID와 공정 단계 컬럼 확인
            equipment_col = None
            process_col = None
            
            for col in ['equipment_id', 'EQUIPMENT_ID']:
                if col in df.columns:
                    equipment_col = col
                    break
                    
            for col in ['process_step', 'PROCESS_STEP', 'current_step', 'CURRENT_STEP']:
                if col in df.columns:
                    process_col = col
                    break
            
            if equipment_col and len(df) > 50:  # 최소 50개 샘플 필요
                # 장비별로 정상 상태 프로파일 업데이트
                equipment_groups = df.groupby(equipment_col)
                
                for equipment_id, equipment_data in equipment_groups:
                    if len(equipment_data) > 10:  # 최소 10개 샘플
                        process_step = equipment_data[process_col].iloc[0] if process_col else 'UNKNOWN'
                        
                        # 정상 상태 프로파일 업데이트
                        self.normal_state_manager.update_normal_profile(
                            str(equipment_id), str(process_step), equipment_data
                        )
                        updated_profiles += 1
        
        print(f"정상 상태 프로파일 업데이트 완료: {updated_profiles}개 프로파일")
    
    def detect_profile_drifts(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        정상 상태 프로파일 드리프트 감지
        """
        print("정상 상태 프로파일 드리프트 감지 중...")
        
        drift_results = []
        sensor_tables = ['semi_photo_sensors', 'semi_etch_sensors', 'semi_cvd_sensors', 
                        'semi_implant_sensors', 'semi_cmp_sensors']
        
        for table_key in sensor_tables:
            if table_key not in datasets:
                continue
                
            df = datasets[table_key]
            
            # 장비별 드리프트 검사
            equipment_col = None
            process_col = None
            
            for col in ['equipment_id', 'EQUIPMENT_ID']:
                if col in df.columns:
                    equipment_col = col
                    break
                    
            for col in ['process_step', 'PROCESS_STEP', 'current_step', 'CURRENT_STEP']:
                if col in df.columns:
                    process_col = col
                    break
            
            if equipment_col:
                equipment_groups = df.groupby(equipment_col)
                
                for equipment_id, equipment_data in equipment_groups:
                    if len(equipment_data) > 5:  # 최소 5개 샘플
                        process_step = equipment_data[process_col].iloc[0] if process_col else 'UNKNOWN'
                        
                        # 드리프트 감지
                        drift_result = self.normal_state_manager.detect_profile_drift(
                            str(equipment_id), str(process_step), equipment_data
                        )
                        
                        if drift_result.get('drift_detected'):
                            drift_results.append(drift_result)
        
        print(f"프로파일 드리프트 감지 완료: {len(drift_results)}개 드리프트 발견")
        return drift_results

    def fast_anomaly_detection_with_realtime_data(self, prism_core_db=None, start: str = None, 
                                                 end: str = None, test_data_path: str = None, 
                                                 test_datasets: Dict = None):
        """
        실시간 데이터베이스 연동 빠른 이상탐지 수행
        
        Args:
            prism_core_db: 실시간 데이터베이스 연결
            start: 시작 시간 (데이터베이스 모드)
            end: 종료 시간 (데이터베이스 모드)
            test_data_path: 로컬 파일 경로 (파일 모드)
            test_datasets: 직접 제공된 데이터 (직접 모드)
            
        Returns:
            Tuple: (이상 LOT 리스트, SVG 시각화, 분석 결과, 드리프트 결과)
        """
        print("=== 실시간 데이터베이스 연동 이상탐지 수행 ===")
        
        # 1. 저장된 모델 로드
        print("1. 저장된 모델 로딩 중...")
        model, scaler, metadata = self.model_manager.load_model()
        
        if model is None:
            print("저장된 모델이 없습니다. setup_and_train_model()을 먼저 실행하세요.")
            return None, "", [], []
        
        self.detector.models['autoencoder'] = model
        self.detector.scaler = scaler
        self.detector.threshold = metadata['threshold']
        feature_cols = metadata['feature_columns']
        
        print(f"모델 로드 완료: {metadata['model_version']}")
        print(f"특성 수: {len(feature_cols)}")
        print(f"임계값: {metadata['threshold']}")
        
        # 2. 데이터 준비 (우선순위: 직접 제공 > 데이터베이스 > 파일)
        if test_datasets is not None:
            print("2. 직접 제공된 테스트 데이터 사용")
            datasets = test_datasets
            drift_results = []
            
        elif prism_core_db and start and end:
            print("2. 실시간 데이터베이스에서 데이터 로딩")
            datasets = self.load_real_time_data_from_database(prism_core_db, start, end)
            
            if not datasets:
                return [], "", [], []
                
            # 정상 상태 프로파일 드리프트 감지
            drift_results = self.detect_profile_drifts(datasets)
            
        elif test_data_path:
            print("2. 로컬 파일에서 테스트 데이터 로딩")
            datasets = self.detector.load_local_data_and_explore(test_data_path)
            drift_results = []
            
        else:
            print("테스트 데이터가 제공되지 않았습니다.")
            return None, "", [], []
        
        # 3. 빠른 이상탐지 수행
        print("3. 이상탐지 수행 중...")
        result_df = self.detector.predict_with_trained_model(datasets, feature_cols)
        
        if result_df is None:
            return [], "", [], drift_results
        
        # 4. 결과 분석 및 시각화
        print("4. 결과 분석 및 시각화 중...")
        analysis_results = self.detector.analyze_results(result_df)
        svg_content = self.detector.visualize_results(result_df)
        
        # 5. 이상 LOT 추출
        anomalies = result_df[result_df['predicted_anomaly']] if 'predicted_anomaly' in result_df.columns else pd.DataFrame()
        anomaly_records = anomalies.to_dict('records') if len(anomalies) > 0 else []
        
        # 6. 결과 요약
        print(f"실시간 이상탐지 완료!")
        print(f"- 총 {len(result_df)}개 LOT 분석")
        print(f"- {len(anomaly_records)}개 이상 LOT 탐지")
        print(f"- {len(drift_results)}개 프로파일 드리프트 감지")
        
        # 데이터 품질 요약
        if self.validation_results:
            avg_quality = np.mean([r['data_quality_score'] for r in self.validation_results])
            print(f"- 평균 데이터 품질: {avg_quality:.1f}점")
        
        return anomaly_records, svg_content, analysis_results, drift_results
    
    # def fast_anomaly_detection(self, test_data_path: str = None, test_datasets: Dict = None):
    #     """
    #     시나리오 B: 저장된 모델로 빠른 이상탐지 수행 (기존 메서드 - 호환성 유지)
    #     """
    #     print("=== 시나리오 B: 빠른 이상탐지 수행 ===")
        
    #     # 1. 저장된 모델 로드
    #     print("1. 저장된 모델 로딩 중...")
    #     model, scaler, metadata = self.model_manager.load_model()
        
    #     if model is None:
    #         print("저장된 모델이 없습니다. setup_and_train_model()을 먼저 실행하세요.")
    #         return None, "", []
        
    #     self.detector.models['autoencoder'] = model
    #     self.detector.scaler = scaler
    #     self.detector.threshold = metadata['threshold']
    #     feature_cols = metadata['feature_columns']
        
    #     print(f"모델 로드 완료: {metadata['model_version']}")
    #     print(f"특성 수: {len(feature_cols)}")
    #     print(f"임계값: {metadata['threshold']}")
        
    #     # 2. 테스트 데이터 준비
    #     if test_datasets is None:
    #         if test_data_path:
    #             print("2. 테스트 데이터 로딩 중...")
    #             test_datasets = self.detector.load_local_data_and_explore(test_data_path)
    #         else:
    #             print("테스트 데이터가 제공되지 않았습니다.")
    #             return None, "", []
        
    #     # 3. 빠른 이상탐지 수행
    #     print("3. 이상탐지 수행 중...")
    #     result_df = self.detector.predict_with_trained_model(test_datasets, feature_cols)
        
    #     if result_df is None:
    #         return [], "", []
        
    #     # 4. 결과 분석 및 시각화
    #     print("4. 결과 분석 및 시각화 중...")
    #     analysis_results = self.detector.analyze_results(result_df)
    #     svg_content = self.detector.visualize_results(result_df)
        
    #     # 5. 이상 LOT 추출
    #     anomalies = result_df[result_df['predicted_anomaly']] if 'predicted_anomaly' in result_df.columns else pd.DataFrame()
    #     anomaly_records = anomalies.to_dict('records') if len(anomalies) > 0 else []
        
    #     print(f"빠른 이상탐지 완료!")
    #     print(f"- 총 {len(result_df)}개 LOT 분석")
    #     print(f"- {len(anomaly_records)}개 이상 LOT 탐지")
        
    #     return anomaly_records, svg_content, analysis_results
    #     model_status = self.get_model_status()
    #     normal_state_summary = self.normal_state_manager.get_all_profiles_summary()
        
    #     return {
    #         'model_status': model_status,
    #         'normal_state_profiles': normal_state_summary,
    #         'data_quality_summary': {
    #             'total_validations': len(self.validation_results),
    #             'average_quality_score': np.mean([r['data_quality_score'] for r in self.validation_results]) if self.validation_results else 0,
    #             'critical_issues': sum(len(r['anomalies']) for r in self.validation_results),
    #             'last_validation': self.validation_results[-1]['validation_timestamp'] if self.validation_results else None
    #         },
    #         'system_timestamp': datetime.now().isoformat()
    #     }

# 편의 함수들
def setup_model_training(data_path: str, model_dir: str = "models"):
    """
    모델 설정 및 훈련 (최초 1회 실행)
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.setup_and_train_model(data_path)

def detect_anomalies_fast(test_data_path: str = None, test_datasets: Dict = None, model_dir: str = "models"):
    """
    시나리오 B: 빠른 이상탐지 수행 (로컬 파일 모드)
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.fast_anomaly_detection(test_data_path, test_datasets)

def detect_anomalies_realtime(prism_core_db, start: str, end: str, model_dir: str = "models"):
    """
    실시간 데이터베이스 연동 이상탐지 수행
    
    Args:
        prism_core_db: PrismCoreDataBase 인스턴스
        start: 시작 시간 (ISO format)
        end: 종료 시간 (ISO format)
        model_dir: 모델 저장 디렉토리
        
    Returns:
        Tuple: (이상 LOT 리스트, SVG 시각화, 분석 결과, 드리프트 결과)
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.fast_anomaly_detection_with_realtime_data(
        prism_core_db=prism_core_db, 
        start=start, 
        end=end
    )

def get_model_info(model_dir: str = "models"):
    """
    저장된 모델 정보 조회
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.get_model_status()

def get_system_status(model_dir: str = "models"):
    """
    전체 시스템 상태 조회 (모델 + 정상 상태 프로파일 + 데이터 품질)
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.get_system_status()

def get_normal_state_summary(model_dir: str = "models"):
    """
    정상 상태 프로파일 요약 조회
    """
    manager = NormalStateManager()
    return manager.get_all_profiles_summary()

def detect_equipment_drift(prism_core_db, equipment_id: str, process_step: str, 
                          start: str, end: str, model_dir: str = "models"):
    """
    특정 장비의 프로파일 드리프트 감지
    
    Args:
        prism_core_db: 데이터베이스 연결
        equipment_id: 장비 ID  
        process_step: 공정 단계
        start: 시작 시간
        end: 종료 시간
        
    Returns:
        Dict: 드리프트 감지 결과
    """
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    
    # 해당 기간 데이터 로딩
    datasets = monitor.load_real_time_data_from_database(prism_core_db, start, end)
    
    # 해당 장비 데이터만 필터링
    equipment_data = None
    for table_key, df in datasets.items():
        if 'equipment_id' in df.columns or 'EQUIPMENT_ID' in df.columns:
            equipment_col = 'equipment_id' if 'equipment_id' in df.columns else 'EQUIPMENT_ID'
            equipment_df = df[df[equipment_col] == equipment_id]
            
            if len(equipment_df) > 0:
                equipment_data = equipment_df
                break
    
    if equipment_data is None or len(equipment_data) == 0:
        return {
            'status': 'no_data',
            'message': f'장비 {equipment_id}의 데이터가 없습니다.',
            'equipment_id': equipment_id,
            'process_step': process_step
        }
    
    # 드리프트 감지
    return monitor.normal_state_manager.detect_profile_drift(
        equipment_id, process_step, equipment_data
    )