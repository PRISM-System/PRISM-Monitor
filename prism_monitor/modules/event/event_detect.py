import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import IsolationForest
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
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

class ModelManager:
    """
    모델 저장, 로딩, 관리 클래스 (개선된 버전)
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_metadata_file = os.path.join(model_dir, "model_metadata.json")
        self.scaler_file = os.path.join(model_dir, "scaler.pkl")
        
    def save_model(self, model, scaler, feature_cols: List[str], threshold: float, 
                   training_data_info: Dict, performance_metrics: Dict):
        """
        모델과 관련 정보 저장 (다중 임계값 지원으로 개선)
        """
        try:
            model_file = os.path.join(self.model_dir, "autoencoder_model.h5")
            model.save(model_file)
            
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 임계값이 dict인 경우 처리
            if isinstance(threshold, dict):
                thresholds = threshold
            else:
                thresholds = {'default': threshold, 'legacy': threshold}
            
            metadata = {
                'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'training_timestamp': datetime.now().isoformat(),
                'feature_columns': feature_cols,
                'threshold': threshold,  # 기존 호환성
                'thresholds': thresholds,  # 새로운 다중 임계값
                'training_data_info': training_data_info,
                'performance_metrics': performance_metrics,
                'model_file': model_file,
                'scaler_file': self.scaler_file
            }
            
            with open(self.model_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"개선된 모델 저장 완료: {model_file}")
            if isinstance(threshold, dict):
                print(f"다중 임계값: {threshold}")
            else:
                print(f"임계값: {threshold}")
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
            
            with open(self.model_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model_file = metadata.get('model_file')
            if not os.path.exists(model_file):
                print(f"모델 파일이 존재하지 않음: {model_file}")
                return None, None, None
            
            from tensorflow.keras.metrics import MeanSquaredError
            model = keras.models.load_model(model_file, custom_objects={"mse": MeanSquaredError()})
            
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
    정상 상태 데이터 관리 모듈 (drift 시각화 개선)
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
        
    def visualize_drift_results(self, drift_results: List[Dict]) -> str:
        """
        프로파일 드리프트 결과 시각화 (SVG) - 개선된 버전
        """
        print(f"드리프트 시각화 시작: {len(drift_results)}개 드리프트 결과")
        
        if not drift_results:
            # 빈 드리프트 결과에 대한 기본 SVG
            svg_content = '''
            <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
                <rect width="800" height="400" fill="white" stroke="black" stroke-width="1"/>
                <text x="400" y="200" text-anchor="middle" font-size="18" fill="green">
                    드리프트가 감지되지 않았습니다.
                </text>
                <text x="400" y="230" text-anchor="middle" font-size="14" fill="gray">
                    모든 장비가 정상 상태를 유지하고 있습니다.
                </text>
            </svg>
            '''
            return svg_content.strip()
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 장비별 드리프트 점수
            equipment_scores = {}
            for drift in drift_results:
                equipment_id = drift.get('equipment_id', 'Unknown')
                if equipment_id not in equipment_scores:
                    equipment_scores[equipment_id] = []
                equipment_scores[equipment_id].append(drift.get('drift_score', 0))
            
            if equipment_scores:
                equipment_names = list(equipment_scores.keys())
                avg_scores = [np.mean(scores) for scores in equipment_scores.values()]
                
                colors = ['red' if score > 50 else 'orange' if score > 20 else 'yellow' for score in avg_scores]
                axes[0, 0].bar(equipment_names, avg_scores, color=colors, alpha=0.7)
                axes[0, 0].set_title('Equipment Drift Scores', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Average Drift Score (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                
                # 임계값 라인 추가
                axes[0, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning (10%)')
                axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critical (50%)')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'No equipment data', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Equipment Drift Scores')
            
            # 2. 심각도별 분포
            severity_counts = {'HIGH': 0, 'MEDIUM': 0}
            for drift in drift_results:
                for param in drift.get('drift_parameters', []):
                    severity = param.get('severity', 'MEDIUM')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                    else:
                        severity_counts['MEDIUM'] += 1
            
            total_severity = sum(severity_counts.values())
            if total_severity > 0:
                colors_pie = ['red', 'orange']
                wedges, texts, autotexts = axes[0, 1].pie(
                    severity_counts.values(), 
                    labels=severity_counts.keys(), 
                    autopct='%1.1f%%', 
                    colors=colors_pie,
                    startangle=90
                )
                axes[0, 1].set_title('Drift Severity Distribution', fontsize=12, fontweight='bold')
                
                # 텍스트 스타일 개선
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                axes[0, 1].text(0.5, 0.5, 'No severity data', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Drift Severity Distribution')
            
            # 3. 시간별 드리프트 발생 추이
            try:
                drift_times = []
                for drift in drift_results:
                    timestamp = drift.get('check_timestamp')
                    if timestamp:
                        try:
                            drift_times.append(pd.to_datetime(timestamp))
                        except:
                            drift_times.append(datetime.now())
                
                if drift_times:
                    time_df = pd.DataFrame({'timestamp': drift_times})
                    time_df['hour'] = time_df['timestamp'].dt.floor('H')
                    time_counts = time_df.groupby('hour').size()
                    
                    if len(time_counts) > 0:
                        axes[1, 0].plot(time_counts.index, time_counts.values, 
                                       marker='o', color='red', linewidth=2, markersize=6)
                        axes[1, 0].fill_between(time_counts.index, time_counts.values, 
                                              alpha=0.3, color='red')
                        axes[1, 0].set_title('Drift Detection Over Time', fontsize=12, fontweight='bold')
                        axes[1, 0].set_xlabel('Time')
                        axes[1, 0].set_ylabel('Number of Drifts')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                        axes[1, 0].grid(True, alpha=0.3)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No time data', ha='center', va='center', 
                                       transform=axes[1, 0].transAxes)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No time data', ha='center', va='center', 
                                   transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Drift Detection Over Time')
            except Exception as e:
                print(f"시간별 차트 생성 오류: {e}")
                axes[1, 0].text(0.5, 0.5, 'Time chart error', ha='center', va='center', 
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Drift Detection Over Time')
            
            # 4. 파라미터별 Z-score 분포 (개선됨)
            z_scores = []
            param_names = []
            colors_scatter = []
            
            for drift in drift_results:
                for param in drift.get('drift_parameters', []):
                    z_score = param.get('z_score', 0)
                    param_name = param.get('parameter', 'Unknown')[:15]
                    severity = param.get('severity', 'MEDIUM')
                    
                    z_scores.append(z_score)
                    param_names.append(param_name)
                    colors_scatter.append('red' if severity == 'HIGH' else 'orange')
            
            if z_scores:
                scatter = axes[1, 1].scatter(range(len(z_scores)), z_scores, 
                                           c=colors_scatter, alpha=0.7, s=60, edgecolors='black')
                
                # 임계값 라인들
                axes[1, 1].axhline(y=3, color='orange', linestyle='--', alpha=0.7, 
                                  linewidth=2, label='3-sigma threshold')
                axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.7, 
                                  linewidth=2, label='5-sigma threshold')
                
                axes[1, 1].set_title('Parameter Z-scores', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Parameter Index')
                axes[1, 1].set_ylabel('Z-score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Y축 범위 조정
                if max(z_scores) > 0:
                    axes[1, 1].set_ylim(0, max(z_scores) * 1.1)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Z-score data', ha='center', va='center', 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Parameter Z-scores')
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # SVG 생성
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg', bbox_inches='tight', 
                       facecolor='white', edgecolor='none', dpi=100)
            svg_content = svg_buffer.getvalue()
            svg_buffer.close()
            plt.close(fig)
            
            print(f"드리프트 SVG 시각화 생성 완료: {len(svg_content)} 문자")
            return svg_content
            
        except Exception as e:
            print(f"드리프트 시각화 생성 오류: {e}")
            # 오류 발생 시 기본 SVG 반환
            error_svg = f'''
            <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
                <rect width="800" height="400" fill="white" stroke="black" stroke-width="1"/>
                <text x="400" y="180" text-anchor="middle" font-size="16" fill="red">
                    드리프트 시각화 생성 중 오류가 발생했습니다.
                </text>
                <text x="400" y="210" text-anchor="middle" font-size="12" fill="gray">
                    오류: {str(e)[:50]}...
                </text>
                <text x="400" y="240" text-anchor="middle" font-size="12" fill="blue">
                    감지된 드리프트: {len(drift_results)}개
                </text>
            </svg>
            '''
            return error_svg.strip()

class DataValidityChecker:
    """
    실시간 데이터 정합성 검증 모듈 (개선된 버전)
    """
    
    def __init__(self):
        # 각 공정별 정상 범위 정의 (스키마 기반)
        self.normal_ranges = {
            'semi_photo_sensors': {
                'exposure_dose': (20, 40),
                'focus_position': (-50, 50),
                'stage_temp': (22.9, 23.1),
                'humidity': (40, 50),
                'alignment_error_x': (0, 3),
                'alignment_error_y': (0, 3),
                'lens_aberration': (0, 5),
                'illumination_uniformity': (98, 100),
                'reticle_temp': (22.95, 23.05)
            },
            'semi_etch_sensors': {
                'rf_power_source': (500, 2000),
                'rf_power_bias': (50, 500),
                'chamber_pressure': (5, 200),
                'gas_flow_cf4': (0, 200),
                'gas_flow_o2': (0, 100),
                'gas_flow_ar': (0, 500),
                'gas_flow_cl2': (0, 200),
                'electrode_temp': (40, 80),
                'chamber_wall_temp': (60, 80),
                'helium_pressure': (5, 20),
                'plasma_density': (1e10, 1e12)
            },
            'semi_cvd_sensors': {
                'susceptor_temp': (300, 700),
                'chamber_pressure': (0.1, 760),
                'precursor_flow_teos': (0, 500),
                'precursor_flow_silane': (0, 1000),
                'precursor_flow_wf6': (0, 100),
                'carrier_gas_n2': (0, 20),
                'carrier_gas_h2': (0, 10),
                'showerhead_temp': (150, 250),
                'liner_temp': (100, 200)
            },
            'semi_implant_sensors': {
                'beam_current': (0.1, 5000),
                'beam_energy': (0.2, 3000),
                'total_dose': (1e11, 1e16),
                'implant_angle': (0, 45),
                'wafer_rotation': (0, 1200),
                'source_pressure': (1e-6, 1e-4),
                'analyzer_pressure': (1e-7, 1e-5),
                'end_station_pressure': (1e-7, 1e-6),
                'beam_uniformity': (98, 100)
            },
            'semi_cmp_sensors': {
                'head_pressure': (2, 8),
                'retainer_pressure': (2, 6),
                'platen_rotation': (20, 150),
                'head_rotation': (20, 150),
                'slurry_flow_rate': (100, 300),
                'slurry_temp': (20, 25),
                'pad_temp': (30, 50),
                'conditioner_pressure': (5, 9)
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
        데이터 정제 및 전처리 (개선된 버전)
        """
        df_clean = df.copy()
        
        # 1. 수치형 컬럼 식별
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # 2. 개선된 결측치 처리
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                # Median 사용 (평균보다 이상치에 강건)
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # 3. IQR 기반 이상치 처리 (개선된 방법)
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                # 3×IQR 범위로 클리핑 (더 강건한 방법)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # 4. 정상 범위 기반 클리핑
        if table_name in self.normal_ranges:
            ranges = self.normal_ranges[table_name]
            for col, (min_val, max_val) in ranges.items():
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
        
        return df_clean

class EnhancedSemiconductorRealTimeMonitor:
    """
    향상된 반도체 실시간 모니터링 시스템
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_manager = ModelManager(model_dir)
        self.normal_state_manager = NormalStateManager()
        self.data_validator = DataValidityChecker()
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # 모델 로드 시도
        self.load_model()
    
    def load_model(self):
        """저장된 모델 로드"""
        try:
            self.model, self.scaler, self.metadata = self.model_manager.load_model()
            if self.model is None:
                print("Warning: 저장된 모델이 없습니다. 기본 알고리즘을 사용합니다.")
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
    
    def fast_anomaly_detection_with_realtime_data(self, prism_core_db, start: str, end: str) -> Tuple[List[Dict], str, Dict, List[Dict], str]:
        """
        실시간 데이터베이스 연동 이상탐지 수행 - 5개 반환값
        """
        print(f"실시간 이상탐지 시작: {start} ~ {end}")
        
        try:
            # 1. 데이터베이스에서 데이터 수집
            all_data = self._fetch_data_from_database(prism_core_db, start, end)
            
            if not all_data:
                return [], self._create_empty_svg(), {}, [], self._create_empty_drift_svg()
            
            # 2. 이상탐지 수행
            anomalies = []
            drift_results = []
            analysis_summary = {
                'total_records': 0,
                'tables_processed': 0,
                'anomalies_detected': 0,
                'drift_detected': 0,
                'processing_time': datetime.now().isoformat()
            }
            
            for table_name, data in all_data.items():
                print(f"처리 중인 테이블: {table_name}, 데이터 수: {len(data)}")
                
                if data.empty:
                    continue
                
                analysis_summary['total_records'] += len(data)
                analysis_summary['tables_processed'] += 1
                
                # 데이터 검증 및 전처리
                validated_data = self.data_validator.preprocess_and_clean(data, table_name)
                
                # 이상탐지 수행
                table_anomalies = self._detect_anomalies_in_data(validated_data, table_name)
                anomalies.extend(table_anomalies)
                
                # 드리프트 감지 (장비별로 수행)
                if 'equipment_id' in data.columns:
                    for equipment_id in data['equipment_id'].unique():
                        equipment_data = data[data['equipment_id'] == equipment_id]
                        drift_result = self.normal_state_manager.detect_profile_drift(
                            equipment_id, table_name, equipment_data
                        )
                        if drift_result.get('drift_detected'):
                            drift_results.append(drift_result)
            
            analysis_summary['anomalies_detected'] = len(anomalies)
            analysis_summary['drift_detected'] = len(drift_results)
            
            # 3. 시각화 생성
            svg_visualization = self._create_anomaly_visualization(anomalies, all_data)
            drift_svg = self.normal_state_manager.visualize_drift_results(drift_results)
            
            print(f"이상탐지 완료: 이상 {len(anomalies)}개, 드리프트 {len(drift_results)}개")
            
            return anomalies, svg_visualization, analysis_summary, drift_results, drift_svg
            
        except Exception as e:
            print(f"실시간 이상탐지 중 오류: {e}")
            error_analysis = {
                'error': str(e),
                'processing_time': datetime.now().isoformat(),
                'status': 'error'
            }
            return [], self._create_error_svg(str(e)), error_analysis, [], self._create_empty_drift_svg()
    
    def _fetch_data_from_database(self, prism_core_db, start: str, end: str) -> Dict[str, pd.DataFrame]:
        start_time = pd.to_datetime(start, utc=True)
        end_time = pd.to_datetime(end, utc=True)
        datasets = {}
        try:
            raise ValueError('use local data')
            for table_name in prism_core_db.get_tables():
                df = prism_core_db.get_table_data(table_name)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                datasets[table_name] = df
        except Exception as e:
            print(f"dataset error raised {e}, use local data")
            data_paths = glob('prism_monitor/data/Industrial_DB_sample/*.csv')
            for data_path in data_paths:
                df = pd.read_csv(data_path)
                table_name = os.path.basename(data_path).split('.csv')[0].lower()
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                datasets[table_name] = df
        return datasets
    
    def _detect_anomalies_in_data(self, data: pd.DataFrame, table_name: str) -> List[Dict]:
        """데이터에서 이상 감지"""
        anomalies = []
        
        try:
            # 수치형 컬럼만 선택
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return anomalies
            
            # Isolation Forest를 사용한 이상탐지
            if len(data) >= 10:  # 최소 데이터 수 확인
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data[numeric_cols].fillna(0))
                
                # 이상치 인덱스 찾기
                anomaly_indices = np.where(outlier_labels == -1)[0]
                
                for idx in anomaly_indices:
                    anomaly_record = {
                        'table_name': table_name,
                        'timestamp': data.iloc[idx].get('timestamp', datetime.now().isoformat()),
                        'equipment_id': data.iloc[idx].get('equipment_id', 'unknown'),
                        'anomaly_type': 'statistical_outlier',
                        'severity': 'MEDIUM',
                        'anomaly_score': abs(iso_forest.score_samples(data[numeric_cols].iloc[[idx]].fillna(0))[0]),
                        'affected_parameters': [],
                        'detection_method': 'isolation_forest'
                    }
                    
                    # 이상 파라미터 식별
                    for col in numeric_cols:
                        value = data.iloc[idx][col]
                        if pd.notna(value):
                            col_mean = data[col].mean()
                            col_std = data[col].std()
                            if col_std > 0:
                                z_score = abs((value - col_mean) / col_std)
                                if z_score > 2:  # 2-sigma 이상
                                    anomaly_record['affected_parameters'].append({
                                        'parameter': col,
                                        'value': float(value),
                                        'z_score': float(z_score),
                                        'mean': float(col_mean),
                                        'std': float(col_std)
                                    })
                    
                    if anomaly_record['affected_parameters']:
                        anomalies.append(anomaly_record)
                        
        except Exception as e:
            print(f"{table_name} 이상탐지 중 오류: {e}")
            
        return anomalies
    
    def _create_anomaly_visualization(self, anomalies: List[Dict], all_data: Dict[str, pd.DataFrame]) -> str:
        """이상 현상 시각화 생성"""
        if not anomalies and not all_data:
            return self._create_empty_svg()
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 테이블별 이상 개수
            if anomalies:
                table_counts = {}
                for anomaly in anomalies:
                    table_name = anomaly.get('table_name', 'unknown')
                    table_counts[table_name] = table_counts.get(table_name, 0) + 1
                
                if table_counts:
                    axes[0, 0].bar(table_counts.keys(), table_counts.values(), color='red', alpha=0.7)
                    axes[0, 0].set_title('Anomalies by Table')
                    axes[0, 0].set_ylabel('Anomaly Count')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=axes[0, 0].transAxes)
            else:
                axes[0, 0].text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Anomalies by Table')
            
            # 2. 심각도별 분포
            if anomalies:
                severity_counts = {}
                for anomaly in anomalies:
                    severity = anomaly.get('severity', 'MEDIUM')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                if severity_counts:
                    colors = ['red' if s == 'HIGH' else 'orange' if s == 'MEDIUM' else 'yellow' for s in severity_counts.keys()]
                    axes[0, 1].pie(severity_counts.values(), labels=severity_counts.keys(), 
                                  autopct='%1.1f%%', colors=colors, startangle=90)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No severity data', ha='center', va='center', transform=axes[0, 1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, 'No severity data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Severity Distribution')
            
            # 3. 데이터 품질 점수 (테이블별)
            if all_data:
                quality_scores = {}
                for table_name, data in all_data.items():
                    validation_result = self.data_validator.validate_data_integrity(data, table_name)
                    quality_scores[table_name] = validation_result.get('data_quality_score', 0)
                
                if quality_scores:
                    colors = ['green' if score > 80 else 'yellow' if score > 60 else 'red' for score in quality_scores.values()]
                    axes[1, 0].bar(quality_scores.keys(), quality_scores.values(), color=colors, alpha=0.7)
                    axes[1, 0].set_title('Data Quality Scores')
                    axes[1, 0].set_ylabel('Quality Score (%)')
                    axes[1, 0].set_ylim(0, 100)
                    axes[1, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No quality data', ha='center', va='center', transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, 'No quality data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Data Quality Scores')
            
            # 4. 시간별 이상 발생 추이
            if anomalies:
                try:
                    anomaly_times = []
                    for anomaly in anomalies:
                        timestamp = anomaly.get('timestamp')
                        if timestamp:
                            try:
                                anomaly_times.append(pd.to_datetime(timestamp))
                            except:
                                anomaly_times.append(datetime.now())
                    
                    if anomaly_times:
                        time_df = pd.DataFrame({'timestamp': anomaly_times})
                        time_df['hour'] = time_df['timestamp'].dt.floor('H')
                        time_counts = time_df.groupby('hour').size()
                        
                        if len(time_counts) > 0:
                            axes[1, 1].plot(time_counts.index, time_counts.values, 
                                           marker='o', color='red', linewidth=2)
                            axes[1, 1].fill_between(time_counts.index, time_counts.values, alpha=0.3, color='red')
                            axes[1, 1].set_xlabel('Time')
                            axes[1, 1].set_ylabel('Anomaly Count')
                            axes[1, 1].tick_params(axis='x', rotation=45)
                        else:
                            axes[1, 1].text(0.5, 0.5, 'No time data', ha='center', va='center', transform=axes[1, 1].transAxes)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'No time data', ha='center', va='center', transform=axes[1, 1].transAxes)
                except Exception as e:
                    axes[1, 1].text(0.5, 0.5, f'Time chart error: {str(e)[:30]}', ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, 'No anomaly timeline', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Anomaly Timeline')
            
            plt.tight_layout()
            
            # SVG 생성
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg', bbox_inches='tight', 
                       facecolor='white', edgecolor='none', dpi=100)
            svg_content = svg_buffer.getvalue()
            svg_buffer.close()
            plt.close(fig)
            
            return svg_content
            
        except Exception as e:
            print(f"시각화 생성 중 오류: {e}")
            return self._create_error_svg(str(e))
    
    def _create_empty_svg(self) -> str:
        """빈 결과용 SVG 생성"""
        return '''
        <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
            <rect width="800" height="400" fill="white" stroke="black" stroke-width="1"/>
            <text x="400" y="200" text-anchor="middle" font-size="18" fill="green">
                이상이 감지되지 않았습니다.
            </text>
            <text x="400" y="230" text-anchor="middle" font-size="14" fill="gray">
                모든 시스템이 정상 상태입니다.
            </text>
        </svg>
        '''.strip()
    
    def _create_empty_drift_svg(self) -> str:
        """빈 드리프트 결과용 SVG 생성"""
        return '''
        <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
            <rect width="800" height="400" fill="white" stroke="black" stroke-width="1"/>
            <text x="400" y="200" text-anchor="middle" font-size="18" fill="green">
                드리프트가 감지되지 않았습니다.
            </text>
            <text x="400" y="230" text-anchor="middle" font-size="14" fill="gray">
                모든 장비가 정상 상태를 유지하고 있습니다.
            </text>
        </svg>
        '''.strip()
    
    def _create_error_svg(self, error_message: str) -> str:
        """오류용 SVG 생성"""
        return f'''
        <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
            <rect width="800" height="400" fill="white" stroke="black" stroke-width="1"/>
            <text x="400" y="180" text-anchor="middle" font-size="16" fill="red">
                처리 중 오류가 발생했습니다.
            </text>
            <text x="400" y="210" text-anchor="middle" font-size="12" fill="gray">
                오류: {error_message[:50]}...
            </text>
            <text x="400" y="240" text-anchor="middle" font-size="12" fill="blue">
                관리자에게 문의하시기 바랍니다.
            </text>
        </svg>
        '''.strip()

# detect_anomalies_realtime 함수 수정 - 5개 반환값으로 통일
def detect_anomalies_realtime(prism_core_db, start: str, end: str, model_dir: str = "models"):
    """실시간 데이터베이스 연동 이상탐지 수행 - drift_svg 포함 5개 반환값"""
    monitor = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
    return monitor.fast_anomaly_detection_with_realtime_data(
        prism_core_db=prism_core_db, 
        start=start, 
        end=end
    )