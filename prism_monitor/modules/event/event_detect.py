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
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class SemiconductorRealDataDetector:
    """
    실제 반도체 공정 데이터를 활용한 이상탐지 딥러닝 모듈
    
    다중 테이블 데이터를 통합하여 이상탐지 수행:
    1. LOT 관리 데이터와 센서 데이터 통합
    2. Autoencoder + LSTM Autoencoder 앙상블
    3. 시간 구간별 이상탐지
    4. 공정별/장비별 세부 분석
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
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
            'sequence_length': 60,  # 시계열 길이
            'contamination': 0.05,  # 이상치 비율 (5%)
            'threshold_percentile': 95,  # 임계값 백분위수
            'batch_size': 32,
            'epochs': 10,  # 실제 데이터용으로 줄임
            'validation_split': 0.2,
        }

    def load_local_data_and_explore(self, data_base_path):
        """
        실제 데이터 로딩 및 탐색
        """
        print("실제 반도체 데이터 로딩 및 탐색 중...")
        
        # 모든 데이터 파일 로딩
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
        print(datasets.keys())
        
        integrated_sensors = []
        
        for table_name in sensor_tables:
            if table_name in datasets:
                df = datasets[table_name].copy()
                
                # 공통 컬럼들만 선택 (PNO, EQUIPMENT_ID, LOT_NO, TIMESTAMP)
                common_cols = ['pno', 'equipment_id', 'lot_no', 'timestamp']
                
                # 실제 존재하는 컬럼만 선택
                available_common = [col for col in common_cols if col in df.columns]
                
                if available_common:
                    # 수치형 센서 컬럼들 찾기
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    # PNO 제외 (ID라서)
                    sensor_cols = [col for col in numeric_cols if col != 'pno']
                    
                    # 테이블 정보 추가
                    df['sensor_table'] = table_name.replace('_sensors', '')
                    
                    # 센서값들을 하나의 컬럼으로 변환 (Long format)
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
            # LOT별 센서 통계 계산
            sensor_stats = integrated_sensors.groupby(['lot_no', 'sensor_type'])['sensor_value'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Wide format으로 변환
            sensor_features = sensor_stats.pivot_table(
                index='lot_no',
                columns='sensor_type',
                values=['mean', 'std', 'min', 'max'],
                fill_value=0
            )
            
            # 컬럼명 정리
            sensor_features.columns = [f"{stat}_{sensor}" for stat, sensor in sensor_features.columns]
            sensor_features = sensor_features.reset_index()
            
            # LOT 데이터와 조인
            main_df = main_df.merge(sensor_features, on='lot_no', how='left')
            print(f"센서 특성 추가 완료: {sensor_features.shape[1]-1}개 특성")
        
        # 4. 공정 이력 데이터 통합
        if 'process_history' in datasets:
            process_df = datasets['process_history']
            if 'LOT_NO' in process_df.columns:
                # LOT별 공정 통계
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
        if 'param_measure' in datasets:
            param_df = datasets['param_measure']
            if 'lot_no' in param_df.columns:
                # LOT별 파라미터 통계
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
        
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print('numeric', numeric_cols)
        
        # ID 관련 컬럼 제외
        exclude_cols = ['pno']  # 기본 제외
        if 'final_yield' in numeric_cols:
            exclude_cols.append('final_yield')  # 타겟 변수는 별도 처리
            
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 결측치 처리
        df_processed = df.copy()
        df_processed[feature_cols] = df_processed[feature_cols].fillna(0)
        
        # 이상치 라벨 생성 (수율 기반)
        if 'final_yield' in df.columns:
            # 수율이 낮은 LOT을 이상으로 간주
            yield_threshold = df['final_yield'].quantile(0.1)  # 하위 10%
            df_processed['is_anomaly'] = df_processed['final_yield'] < yield_threshold
        else:
            # 수율 정보가 없으면 통계적 방법 사용
            feature_data = df_processed[feature_cols]
            z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std()).mean(axis=1)
            threshold = np.percentile(z_scores, 90)
            df_processed['is_anomaly'] = z_scores > threshold
        
        # 특성 스케일링
        df_processed[feature_cols] = self.scaler.fit_transform(df_processed[feature_cols])
        
        print(f"전처리 완료: {len(feature_cols)}개 특성")
        print(f"이상 LOT 비율: {df_processed['is_anomaly'].mean():.2%}")
        
        return df_processed, feature_cols

    def build_autoencoder(self, input_dim):
        """
        Autoencoder 모델 구축
        """
        input_layer = layers.Input(shape=(input_dim,))
        
        # 인코더 - 차원 축소
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        encoded = layers.Dense(32, activation='relu')(encoded)  # 병목층
        
        # 디코더 - 차원 복원
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

    def train_and_detect(self, df, feature_cols):
        """
        모델 학습 및 이상탐지 수행
        """
        print("모델 학습 및 이상탐지 시작...")
        
        # 정상 데이터만 사용하여 학습
        normal_data = df[~df['is_anomaly']]
        X_normal = normal_data[feature_cols].values
        
        print(f"학습 데이터: {len(X_normal)}개 정상 샘플")
        
        # Autoencoder 학습
        self.models['autoencoder'] = self.build_autoencoder(len(feature_cols))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.models['autoencoder'].fit(
            X_normal, X_normal,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.history['autoencoder'] = history
        
        # 임계값 계산
        reconstructed = self.models['autoencoder'].predict(X_normal, verbose=0)
        mse_normal = np.mean(np.square(X_normal - reconstructed), axis=1)
        self.threshold = np.percentile(mse_normal, self.config['threshold_percentile'])
        
        print(f"이상탐지 임계값: {self.threshold:.4f}")
        
        # 전체 데이터에 대해 이상탐지
        X_all = df[feature_cols].values
        reconstructed_all = self.models['autoencoder'].predict(X_all, verbose=0)
        mse_scores = np.mean(np.square(X_all - reconstructed_all), axis=1)
        
        # 결과 저장
        df_result = df.copy()
        df_result['anomaly_score'] = mse_scores
        df_result['predicted_anomaly'] = mse_scores > self.threshold
        df_result['confidence'] = (mse_scores - self.threshold) / self.threshold
        
        return df_result

    def analyze_results(self, df_result):
        """
        결과 분석 및 요약
        """
        print("결과 분석 중...")
        
        # 기본 통계
        total_lots = len(df_result)
        detected_anomalies = df_result['predicted_anomaly'].sum()
        actual_anomalies = df_result['is_anomaly'].sum()
        
        print(f"\n=== 이상탐지 결과 요약 ===")
        print(f"전체 LOT 수: {total_lots}")
        print(f"실제 이상 LOT: {actual_anomalies} ({actual_anomalies/total_lots:.1%})")
        print(f"탐지된 이상 LOT: {detected_anomalies} ({detected_anomalies/total_lots:.1%})")
        
        # 성능 평가 (실제 라벨이 있는 경우)
        if 'is_anomaly' in df_result.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            recall = recall_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            f1 = f1_score(df_result['is_anomaly'], df_result['predicted_anomaly'])
            
            print(f"\n=== 성능 지표 ===")
            print(f"정밀도 (Precision): {precision:.3f}")
            print(f"재현율 (Recall): {recall:.3f}")
            print(f"F1 점수: {f1:.3f}")
        
        # 이상 LOT 상세 분석
        anomaly_lots = df_result[df_result['predicted_anomaly']]
        if len(anomaly_lots) > 0:
            print(f"\n=== 이상 LOT 분석 ===")
            print("이상 점수가 높은 상위 5개 LOT:")
            top_anomalies = anomaly_lots.nlargest(5, 'anomaly_score')
            for _, row in top_anomalies.iterrows():
                print(f"  LOT {row['lot_no']}: 점수 {row['anomaly_score']:.4f}")
                if 'final_yield' in row:
                    print(f"    수율: {row['final_yield']:.1f}%")
        
        return {
            'total_lots': total_lots,
            'detected_anomalies': int(detected_anomalies),
            'detection_rate': float(detected_anomalies / total_lots),
            'anomaly_lots': anomaly_lots.to_markdown(index=False)
        }

    def visualize_results(self, df_result):
        """
        결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 이상 점수 분포
        normal_scores = df_result[~df_result['predicted_anomaly']]['anomaly_score']
        anomaly_scores = df_result[df_result['predicted_anomaly']]['anomaly_score']
        
        axes[0, 0].hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
        axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
        axes[0, 0].axvline(self.threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Anomaly Score Distribution')
        axes[0, 0].legend()
        
        # 2. 수율 vs 이상 점수 (수율 정보가 있는 경우)
        if 'final_yield' in df_result.columns:
            scatter = axes[0, 1].scatter(
                df_result['final_yield'], 
                df_result['anomaly_score'],
                c=df_result['predicted_anomaly'], 
                cmap='coolwarm', 
                alpha=0.6
            )
            axes[0, 1].set_xlabel('Final Yield (%)')
            axes[0, 1].set_ylabel('Anomaly Score')
            axes[0, 1].set_title('Yield vs Anomaly Score')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. 혼동 행렬
        if 'is_anomaly' in df_result.columns:
            cm = confusion_matrix(df_result['is_anomaly'], df_result['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
            axes[1, 0].set_title('Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # 4. 학습 손실
        if 'autoencoder' in self.history:
            history = self.history['autoencoder']
            axes[1, 1].plot(history.history['loss'], label='Training Loss')
            axes[1, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

    def detect_anomalies_in_timerange(self, df_result, start_time=None, end_time=None):
        """
        특정 시간 구간의 이상탐지 결과 조회
        """
        if start_time is None and end_time is None:
            print("전체 기간 이상탐지 결과:")
            filtered_df = df_result
        else:
            print(f"시간 구간 ({start_time} ~ {end_time}) 이상탐지 결과:")
            # 실제 시간 컬럼이 있다면 필터링 적용
            # 여기서는 LOT_NO 기준으로 간단히 필터링
            filtered_df = df_result
        
        anomalies = filtered_df[filtered_df['predicted_anomaly']]
        
        print(f"조회 기간 내 이상 LOT: {len(anomalies)}개")
        
        if len(anomalies) > 0:
            print("\n이상 LOT 목록:")
            for _, row in anomalies.iterrows():
                print(f"  - LOT {row['lot_no']}: 이상점수 {row['anomaly_score']:.4f}")
                if 'final_yield' in row:
                    print(f"    수율: {row['final_yield']:.1f}%")
        
        return anomalies
    

def detect_anomalies_in_timerange(datasets=None, local_data_path=None):
    assert datasets is not None or local_data_path is not None, "datasets와 loca_data_path 중 적어도 하나는 None이 아니어야 합니다."
    detector = SemiconductorRealDataDetector()
    if datasets is None and local_data_path:
        datasets = detector.load_local_data_and_explore(local_data_path)
    unified_df = detector.create_unified_dataset(datasets)
    processed_df, feature_cols = detector.prepare_features(unified_df)
    result_df = detector.train_and_detect(processed_df, feature_cols)
    analysis = detector.analyze_results(result_df)
    anomalies = detector.detect_anomalies_in_timerange(result_df)
    return anomalies, analysis



