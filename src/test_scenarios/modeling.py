import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from pyod.models.auto_encoder import AutoEncoder
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model
from skforecast.utils import save_forecaster, load_forecaster

from src.modules.util.util import dataframe_to_json_serializable

DIR_PATH = Path(__file__).parent.resolve()

TEST_SCENARIOS_DATA_MAPPING = {
    # Semiconductor
    'semiconductor_cmp': DIR_PATH / 'test_data/semiconductor/semiconductor_cmp_001.csv',
    'semiconductor_etch': DIR_PATH / 'test_data/semiconductor/semiconductor_etch_002.csv',
    'semiconductor_deposition': DIR_PATH / 'test_data/semiconductor/semiconductor_deposition_003.csv',
    'semiconductor_full': DIR_PATH / 'test_data/semiconductor/semiconductor_full_004.csv',
    # Chemical
    'chemical_reactor': DIR_PATH / 'test_data/chemical/chemical_reactor_001.csv',
    'chemical_distillation': DIR_PATH / 'test_data/chemical/chemical_distillation_002.csv',
    'chemical_refining': DIR_PATH / 'test_data/chemical/chemical_refining_003.csv',
    'chemical_full': DIR_PATH / 'test_data/chemical/chemical_full_004.csv',
    # Automotive
    'automotive_welding': DIR_PATH / 'test_data/automotive/automotive_welding_001.csv',
    'automotive_painting': DIR_PATH / 'test_data/automotive/automotive_painting_002.csv',
    'automotive_press': DIR_PATH / 'test_data/automotive/automotive_press_003.csv',
    'automotive_assembly': DIR_PATH / 'test_data/automotive/automotive_assembly_004.csv',
    # Battery
    'battery_formation': DIR_PATH / 'test_data/battery/battery_formation_001.csv',
    'battery_coating': DIR_PATH / 'test_data/battery/battery_coating_002.csv',
    'battery_aging': DIR_PATH / 'test_data/battery/battery_aging_003.csv',
    'battery_production': DIR_PATH / 'test_data/battery/battery_production_004.csv',
    # Steel
    'steel_rolling': DIR_PATH / 'test_data/steel/steel_rolling_001.csv',
    'steel_converter': DIR_PATH / 'test_data/steel/steel_converter_002.csv',
    'steel_casting': DIR_PATH / 'test_data/steel/steel_casting_003.csv',
    'steel_production': DIR_PATH / 'test_data/steel/steel_production_004.csv',
}


class TestScenarioModel:
    """
    테스트 시나리오용 이상치 탐지 모델 클래스
    """
    def __init__(self, ad_contamination=0.01, forecasting_lag=30, forecasting_step=5):
        """
        Args:
            ad_contamination: 이상치 탐지를 위한 오염 비율
            forecasting_lag: 시계열 예측을 위한 지연 시간
            forecasting_step: 시계열 예측을 위한 예측 단계
        """
        self.ad_contamination = ad_contamination
        self.forecasting_lag = forecasting_lag
        self.forecasting_step = forecasting_step
        self.device = 'cpu'
        self.ad_models = {}
        self.forecasting_models = {}

    def train_and_save_models(self):
        """
        주어진 공정에 대해 AutoEncoder 모델을 학습하고 저장합니다.

        Args:
            target_process: 공정 식별자
        """
        for target_process, csv_path in TEST_SCENARIOS_DATA_MAPPING.items():
            print(f"Training model for {target_process} using data from {csv_path}")

            ad_model = self.train_anomaly_detection_model(target_process)

            model_path = DIR_PATH / 'models' / f'{target_process}_anomaly_detection.joblib'
            model_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(ad_model, model_path)
            print(f"   📂 Model saved to {model_path}\n")

            # 데이터 로드
            df = pd.read_csv(csv_path)
            print(f"   ✓ Loaded data: {len(df)} rows")
            df['ANOMALY_SCORE'] = ad_model.predict_proba(df.iloc[:, 2:])[:, 1]

            forecasting_model = self.train_forecasting_model(df)

            model_path = DIR_PATH / 'models' / f'{target_process}_forecasting.joblib'
            model_path.parent.mkdir(parents=True, exist_ok=True)

            save_forecaster(forecasting_model, model_path)
            print(f"   📂 Forecasting model saved to {model_path}\n")

    def train_anomaly_detection_model(self, target_process: str):
        """
        지정된 공정에 대해 AutoEncoder 모델을 학습하고 저장합니다.

        Args:
            target_process: 공정 식별자
        """
        csv_path = TEST_SCENARIOS_DATA_MAPPING[target_process]
        print(f"Training model for {target_process} using data from {csv_path}")

        # 데이터 로드
        data = pd.read_csv(csv_path)
        print(f"   ✓ Loaded data: {len(data)} rows")

        # TIMESTAMP 열 제거
        X = data.iloc[:, 2:]  # Assuming first two columns are TIMESTAMP and another identifier

        # AutoEncoder 모델 초기화 및 학습
        clf = AutoEncoder(contamination=self.ad_contamination, device=self.device)
        clf.fit(X)
        return clf

    def train_forecasting_model(self, df: pd.DataFrame):
        """
        시계열 예측 모델을 학습합니다.

        Args:
            X: 입력 데이터 (pandas DataFrame)
        """
        df.set_index('TIMESTAMP', inplace=True)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%dT%H:%M:%SZ")
        df = df.asfreq('10S')
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categoric_cols = df.select_dtypes(exclude='number').columns.tolist()
        df = df[numeric_cols].copy()

        levels = numeric_cols # Multiple target series to predict

        model = create_and_compile_model(
            series                  = df,    # DataFrame with all series (predictors)
            levels                  = levels, 
            lags                    = self.forecasting_lag, 
            steps                   = self.forecasting_step, 
            recurrent_layer         = "LSTM",
        )

        # Forecaster Creation
        # ==============================================================================
        forecaster = ForecasterRnn(
            regressor=model,
            levels=levels,
            lags=self.forecasting_lag,
            transformer_series=MinMaxScaler(),
        )

        # Fit forecaster
        # ==============================================================================
        forecaster.fit(df)
        return forecaster


    def set_models(self):
        """
        모든 공정에 대해 저장된 모델을 로드합니다.
        """
        try:
            for target_process in TEST_SCENARIOS_DATA_MAPPING.keys():
                self.ad_models[target_process] = self.load_ad_model(target_process)
                self.forecasting_models[target_process] = self.load_forecasting_model(target_process)
        except Exception as e:
            print(f"Error loading models: {e}")
            self.train_and_save_models()

    def load_ad_model(self, target_process: str):
        """
        지정된 공정에 대해 저장된 모델을 로드합니다.

        Args:
            target_process: 공정 식별자

        Returns:
            로드된 모델 객체
        """
        model_path = DIR_PATH / 'models' / f'{target_process}_anomaly_detection.joblib'
        clf = joblib.load(model_path)
        print(f"   📂 Model loaded from {model_path}")
        return clf
    
    def load_forecasting_model(self, target_process: str):
        """
        지정된 공정에 대해 저장된 시계열 예측 모델을 로드합니다.

        Args:
            target_process: 공정 식별자

        Returns:
            로드된 시계열 예측 모델 객체
        """
        model_path = DIR_PATH / 'models' / f'{target_process}_forecasting.joblib'
        forecaster = load_forecaster(model_path)
        print(f"   📂 Forecasting model loaded from {model_path}")
        return forecaster
    
    def ad_predict(self, target_process: str, start: str, end: str):
        X = load_test_scenarios_data(target_process, start, end)
        clf = self.ad_models.get(target_process)
        if clf is None:
            clf = self.load_model(target_process)
        return clf.predict(X.iloc[:, 2:])  # TIMESTAMP 제외
    
    def ad_predict_proba(self, target_process: str, start: str, end: str):
        X = load_test_scenarios_data(target_process, start, end)
        clf = self.ad_models.get(target_process)
        if clf is None:
            clf = self.load_model(target_process)
        return clf.predict_proba(X.iloc[:, 2:])[:, 1]

    def ad_detect_anomalies(self, target_process: str, start: str, end: str):
        """
        지정된 공정에 대해 이상치 탐지를 수행합니다.

        Args:
            target_process: 공정 식별자
            X: 입력 데이터 (pandas DataFrame) 첫 두 열(ID, TIMESTAMP)

        Returns:
            이상치 인덱스 리스트
        """
        X = load_test_scenarios_data(target_process, start, end)
        preds = self.ad_predict(target_process, start, end)
        probs = self.ad_predict_proba(target_process, start, end)
        X['ANOMALY'] = preds
        X['ANOMALY_SCORE'] = probs
        return X

    def ad_get_anomaly_records(self, target_process: str, start: str = None, end: str = None):
        """
        지정된 공정에 대해 이상치 탐지를 수행합니다.

        Args:
            target_process: 공정 식별자
            X: 입력 데이터 (pandas DataFrame) 첫 두 열(ID, TIMESTAMP)

        Returns:
            이상치 인덱스 리스트
        """
        X = load_test_scenarios_data(target_process, start, end)
        preds = self.ad_predict(target_process, start, end)
        probs = self.ad_predict_proba(target_process, start, end)
        X['ANOMALY'] = preds
        X['ANOMALY_SCORE'] = probs
        return X[X['ANOMALY'] == 1]

    def ad_summary(self, target_process: str, start: str, end: str):
        """
        지정된 공정에 대해 이상치 탐지 요약 정보를 반환합니다.

        Args:
            target_process: 공정 식별자
            X: 입력 데이터 (pandas DataFrame) 첫 두 열(ID, TIMESTAMP)

        Returns:
            이상치 탐지 요약 정보 (딕셔너리)
        """
        X = load_test_scenarios_data(target_process, start, end)
        anomalies = self.ad_get_anomaly_records(target_process, start, end)
        summary = {
            'total_records': len(X),
            'anomaly_count': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(X) * 100 if len(X) > 0 else 0,
        }
        return summary

    def ad_get_visual_data(self, target_process: str, start: str, end: str):
        """
        지정된 공정에 대해 이상치 탐지 결과를 시각화할 수 있는 데이터를 반환합니다.

        Args:
            target_process: 공정 식별자
            X: 입력 데이터 (pandas DataFrame) 첫 두 열(ID, TIMESTAMP)

        Returns:
            시각화용 데이터 (pandas DataFrame)
        """
        X = self.ad_detect_anomalies(target_process, start, end)
        return dataframe_to_json_serializable(X)

    def forecasting_predict(self, target_process: str, start: str = None, end: str = None):
        """
        지정된 공정에 대해 시계열 예측을 수행합니다.

        Args:
            target_process: 공정 식별자
            start: 시작 시간 (ISO format)
            end: 종료 시간 (ISO format)

        Returns:
            예측 결과 (pandas DataFrame)
        """
        forecaster = self.forecasting_models.get(target_process)
        if forecaster is None:
            forecaster = self.load_forecasting_model(target_process)
        
        X = load_test_scenarios_data(target_process, start, end)
        X.set_index('TIMESTAMP', inplace=True)
        X.index = pd.to_datetime(X.index, format="%Y-%m-%dT%H:%M:%SZ")
        X = X.asfreq('10S')

        probs = self.ad_predict_proba(target_process, start, end)
        X['ANOMALY_SCORE'] = probs
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        X = X[numeric_cols].copy()

        predictions = forecaster.predict(last_window=X)
        predictions = predictions.reset_index(names='TIMESTAMP').pivot(index='TIMESTAMP', columns='level', values='pred')

        return {
            'predictions': predictions,
            'current_data': X
        }
    
    def get_dashboard(self):
        res = {}
        for target_process in TEST_SCENARIOS_DATA_MAPPING.keys():
            forecasting_predict_res = self.forecasting_predict(target_process)
            predictions_df = forecasting_predict_res['predictions']
            current_data_df = forecasting_predict_res['current_data']
            predictions = dataframe_to_json_serializable(predictions_df)
            current_data = dataframe_to_json_serializable(current_data_df)
            res[target_process] = {
                'predictions': predictions,
                'current_data': current_data
            }
        return res

def load_test_scenarios_data(target_process: str, start: str = None, end: str = None, sample_len: int = 30):
    """
    테스트 시나리오 데이터를 로드합니다.

    Args:
        target_process: 공정 식별자
        start: 시작 시간 (ISO format)
        end: 종료 시간 (ISO format)

    Returns:
        로드된 데이터 (pandas DataFrame)
    """
    csv_path = TEST_SCENARIOS_DATA_MAPPING[target_process]

    data = pd.read_csv(csv_path)

    # TIMESTAMP 필터링
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    if start is None or end is None:
        end_idx = data.iloc[-sample_len:].sample(1).index[0]
        start_idx = end_idx - sample_len
        return data.iloc[start_idx:end_idx]
    
    mask = (data['TIMESTAMP'] >= pd.to_datetime(start, utc=True)) & (data['TIMESTAMP'] <= pd.to_datetime(end, utc=True))
    filtered_data = data.loc[mask]
    if filtered_data.shape[0] < sample_len:
        end_idx = data.iloc[-sample_len:].sample(1).index[0]
        start_idx = end_idx - sample_len
        return data.iloc[start_idx:end_idx]
    return filtered_data