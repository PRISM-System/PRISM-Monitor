"""
==============================================================================
Event Precursor Module - Core Implementation (주석 버전)
==============================================================================

목적: 제조 설비의 센서 데이터를 분석하여 미래의 이상 징후를 사전에 예측
기술: LSTM 기반 시계열 딥러닝 모델
지원 산업: Automotive, Battery, Chemical, Semiconductor, Steel 등

주요 기능:
1. 다중 CSV 파일 로딩 및 통합
2. Z-score 기반 자동 이상 레이블링 (비지도 학습)
3. Sliding Window 방식 시계열 데이터 생성
4. LSTM 모델 학습 및 예측
5. 실시간 모니터링 시뮬레이션
6. RUL (잔여 유효 수명) 예측
7. 위험도별 경고 생성

==============================================================================
"""

# ===========================
# 라이브러리 임포트
# ===========================
import numpy as np              # 수치 연산
import pandas as pd             # 데이터 처리
import torch                    # PyTorch 딥러닝 프레임워크
import torch.nn as nn           # 신경망 레이어
import torch.optim as optim     # 최적화 알고리즘
from torch.utils.data import DataLoader, TensorDataset  # 데이터 로더
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 정규화
from sklearn.model_selection import train_test_split    # 데이터 분할
import warnings                 # 경고 메시지
import os                       # 파일 시스템
from datetime import datetime, timedelta  # 시간 처리
import matplotlib.pyplot as plt # 시각화
import time                     # 시간 측정


# ==============================================================================
# 1. 데이터 로딩 함수
# ==============================================================================

def load_single_csv(csv_path):
    """
    단일 CSV 파일을 로드하는 함수

    Args:
        csv_path (str): CSV 파일 경로

    Returns:
        DataFrame: 로드된 데이터프레임 (TIMESTAMP는 datetime 형식으로 변환)
    """
    df = pd.read_csv(csv_path)

    # TIMESTAMP 컬럼이 있으면 datetime 타입으로 변환
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    return df


def load_and_explore_data(data_base_path):
    """
    지정된 디렉토리에서 모든 CSV 파일을 로드하는 함수

    디렉토리 구조:
    data_base_path/
    ├── file1.csv              # 최상위 CSV 파일
    ├── category1/             # 카테고리 폴더
    │   ├── file2.csv
    │   └── file3.csv
    └── category2/
        └── file4.csv

    Args:
        data_base_path (str): 데이터 폴더의 기본 경로

    Returns:
        dict: {filename: DataFrame} 형태의 딕셔너리
              filename은 .csv 확장자 제거된 이름

    동작 방식:
        1. 최상위 디렉토리의 CSV 파일 로드
        2. 하위 디렉토리 탐색하여 CSV 파일 로드
        3. 에러 발생 시 메시지 출력하고 계속 진행
    """
    print("Data Loading...")
    datasets = {}  # 로드된 데이터를 저장할 딕셔너리

    # 디렉토리가 존재하는지 확인
    if os.path.isdir(data_base_path):
        # 디렉토리 내 모든 항목 순회
        for industry_dir in os.listdir(data_base_path):
            industry_path = os.path.join(data_base_path, industry_dir)

            # 케이스 1: 최상위 레벨에 CSV 파일이 있는 경우
            if os.path.isfile(industry_path) and industry_path.endswith('.csv'):
                filename = os.path.basename(industry_path)
                key = filename.replace('.csv', '')  # 확장자 제거
                print(f"Loading: {filename}")
                try:
                    df = pd.read_csv(industry_path)
                    datasets[key] = df
                    print(f"  - Shape: {df.shape}")
                except Exception as e:
                    print(f"  - Error: {e}")

            # 케이스 2: 하위 디렉토리인 경우 (예: automotive/, battery/ 등)
            elif os.path.isdir(industry_path):
                # 하위 디렉토리 내 파일들 순회
                for filename in os.listdir(industry_path):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(industry_path, filename)

                        key = filename.replace('.csv', '')
                        print(f"Loading: {industry_dir}/{filename}")
                        try:
                            df = pd.read_csv(file_path)
                            datasets[key] = df
                            print(f"  - Shape: {df.shape}")
                        except Exception as e:
                            print(f"  - Error: {e}")

    return datasets


# ==============================================================================
# 2. 데이터 통합 함수
# ==============================================================================

def integrate_sensor_data(datasets):
    """
    여러 데이터셋의 센서 데이터를 Long Format으로 통합

    Wide Format (원본):
        TIMESTAMP | SENSOR_ID | TEMP | PRESSURE | VOLTAGE
        2025-01   | S001      | 25.0 | 1.2      | 220

    Long Format (변환 후):
        TIMESTAMP | SENSOR_ID | sensor_table | sensor_type | sensor_value
        2025-01   | S001      | file1        | TEMP        | 25.0
        2025-01   | S001      | file1        | PRESSURE    | 1.2
        2025-01   | S001      | file1        | VOLTAGE     | 220

    Args:
        datasets (dict): {filename: DataFrame} 형태

    Returns:
        DataFrame: Long format 통합 데이터
                  Columns: ['TIMESTAMP', equipment_col, 'sensor_table',
                           'sensor_type', 'sensor_value']

    처리 과정:
        1. 각 데이터셋에서 장비 ID 컬럼 찾기
        2. 수치형 컬럼(센서 데이터) 추출
        3. Wide → Long format 변환 (melt)
        4. 모든 데이터셋 통합 (concat)
    """
    print("센서 데이터 통합...")

    integrated_sensors = []  # 통합될 데이터 리스트

    # 각 테이블(CSV 파일)별로 처리
    for table_name, df in datasets.items():
        # 빈 데이터프레임은 건너뛰기
        if df.empty:
            continue

        df_copy = df.copy()

        # TIMESTAMP 컬럼을 datetime 타입으로 변환
        if 'TIMESTAMP' in df_copy.columns:
            df_copy['TIMESTAMP'] = pd.to_datetime(df_copy['TIMESTAMP'])

        # ----------------------------------------------------------------------
        # 장비 ID 컬럼 찾기
        # ----------------------------------------------------------------------
        equipment_col = None

        # 우선순위가 높은 ID 컬럼 리스트
        possible_id_cols = [
            'SENSOR_ID',       # 센서 ID
            'CHAMBER_ID',      # 챔버 ID (반도체)
            'EQUIPMENT_ID',    # 장비 ID
            'CELL_ID',         # 셀 ID (배터리)
            'LINE_ID',         # 라인 ID
            'PRODUCTION_LINE'  # 생산 라인 (비표준)
        ]

        # 우선순위 순서대로 컬럼 찾기
        for col in possible_id_cols:
            if col in df_copy.columns:
                equipment_col = col
                break

        # 위 리스트에 없으면 '_ID'로 끝나는 컬럼 찾기 (폴백)
        if equipment_col is None:
            id_cols = [col for col in df_copy.columns if col.endswith('_ID')]
            if id_cols:
                equipment_col = id_cols[0]  # 첫 번째 발견된 컬럼 사용

        # ----------------------------------------------------------------------
        # 수치형 컬럼 추출 (센서 데이터)
        # ----------------------------------------------------------------------
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        # 원본 테이블 이름 추가 (어느 파일에서 왔는지 추적)
        df_copy['sensor_table'] = table_name

        # ----------------------------------------------------------------------
        # Wide → Long Format 변환 (melt)
        # ----------------------------------------------------------------------
        if numeric_cols:
            # melt 함수에서 유지할 컬럼 (id_vars)
            id_vars = ['TIMESTAMP']
            if equipment_col:
                id_vars.append(equipment_col)
            id_vars.append('sensor_table')

            # Wide → Long 변환
            # value_vars: 녹일(melt) 컬럼들 (센서 데이터)
            # var_name: 컬럼 이름이 들어갈 새 컬럼
            # value_name: 값이 들어갈 새 컬럼
            df_long = df_copy.melt(
                id_vars=id_vars,
                value_vars=numeric_cols,
                var_name='sensor_type',    # 센서 종류 (TEMP, PRESSURE 등)
                value_name='sensor_value'  # 센서 값
            )

            integrated_sensors.append(df_long)
            print(f"  - {table_name}: Sensor count: {len(numeric_cols)}, Record count: {len(df_copy)}")

    # 모든 Long format 데이터를 하나로 합치기
    if integrated_sensors:
        result = pd.concat(integrated_sensors, ignore_index=True)
        print(f"Integration finish: Total records: {len(result)} sensors")
        return result
    else:
        return pd.DataFrame()


def create_unified_dataset(datasets):
    """
    Long Format 데이터를 Pivot하여 Wide Format 통합 데이터셋 생성

    Long Format:
        TIMESTAMP | sensor_type | sensor_value
        2025-01   | TEMP        | 25.0
        2025-01   | PRESSURE    | 1.2

    Wide Format (pivot 후):
        TIMESTAMP | sensor_TEMP | sensor_PRESSURE
        2025-01   | 25.0        | 1.2

    Args:
        datasets (dict): {filename: DataFrame}

    Returns:
        DataFrame: Wide format 통합 데이터셋
                  Index: TIMESTAMP
                  Columns: sensor_* (각 센서별 컬럼) + equipment_id (optional)

    처리 과정:
        1. integrate_sensor_data() 호출하여 Long format 생성
        2. pivot_table로 Wide format 변환
        3. 결측치 처리 (forward fill → backward fill → 0)
        4. equipment_id 정보 병합 (있는 경우)
    """
    print("Creating unified dataset...")

    # Step 1: Long format 데이터 생성
    integrated_sensors = integrate_sensor_data(datasets)

    # 데이터가 없으면 빈 DataFrame 반환
    if integrated_sensors.empty:
        print("센서 데이터가 없습니다.")
        return pd.DataFrame()

    # TIMESTAMP 컬럼 확인
    if 'TIMESTAMP' not in integrated_sensors.columns:
        print("TIMESTAMP 컬럼이 없습니다.")
        return pd.DataFrame()

    # Step 2: Long → Wide format 변환 (pivot_table)
    # index: 행 인덱스 (TIMESTAMP)
    # columns: 컬럼으로 펼칠 값 (sensor_type)
    # values: 셀에 들어갈 값 (sensor_value)
    # aggfunc: 중복값 처리 방법 (평균)
    sensor_pivot = integrated_sensors.pivot_table(
        index='TIMESTAMP',
        columns='sensor_type',
        values='sensor_value',
        aggfunc='mean'  # 같은 TIMESTAMP의 중복 값은 평균
    )

    # 컬럼 이름에 'sensor_' 접두사 추가
    # 예: 'TEMP' → 'sensor_TEMP'
    sensor_pivot.columns = [f"sensor_{col}" for col in sensor_pivot.columns]

    # 인덱스를 컬럼으로 변환
    sensor_pivot = sensor_pivot.reset_index()

    # Step 3: 결측치 처리
    # ffill: forward fill (앞 값으로 채우기)
    # bfill: backward fill (뒤 값으로 채우기)
    # fillna(0): 그래도 남은 결측치는 0으로
    sensor_pivot = sensor_pivot.ffill().bfill().fillna(0)

    # Step 4: equipment_id 정보 병합 (선택적)
    # 원본 데이터셋에서 equipment ID 정보 추출
    equipment_info = None
    for df in datasets.values():
        if 'TIMESTAMP' in df.columns:
            df_temp = df.copy()
            df_temp['TIMESTAMP'] = pd.to_datetime(df_temp['TIMESTAMP'])

            # equipment ID 컬럼 찾기 (integrate_sensor_data와 동일 로직)
            equipment_col = None
            possible_id_cols = ['SENSOR_ID', 'CHAMBER_ID', 'EQUIPMENT_ID', 'CELL_ID', 'LINE_ID']
            for col in possible_id_cols:
                if col in df_temp.columns:
                    equipment_col = col
                    break

            # 폴백: '_ID'로 끝나는 컬럼
            if equipment_col is None:
                id_cols = [col for col in df_temp.columns if col.endswith('_ID')]
                if id_cols:
                    equipment_col = id_cols[0]

            # equipment 정보 추출 (TIMESTAMP별 중복 제거)
            if equipment_col:
                equipment_info = df_temp[['TIMESTAMP', equipment_col]].drop_duplicates('TIMESTAMP')
                equipment_info = equipment_info.rename(columns={equipment_col: 'equipment_id'})
                break  # 첫 번째 발견된 것 사용

    # equipment_id 정보가 있으면 병합
    if equipment_info is not None:
        sensor_pivot = sensor_pivot.merge(equipment_info, on='TIMESTAMP', how='left')

    print(f"최종 통합 데이터셋: {sensor_pivot.shape}")
    return sensor_pivot


# ==============================================================================
# 3. 특성 전처리 및 이상 레이블링
# ==============================================================================

def prepare_features(df, train_stats=None):
    """
    특성 전처리 및 Z-score 기반 자동 이상 레이블링

    Args:
        df (DataFrame): 입력 데이터
        train_stats (dict, optional): Train 데이터의 통계량
            - None: 새로 계산 (Train 데이터용)
            - dict{'mean': ..., 'std': ...}: 기존 통계 사용 (Val/Test용)

    Returns:
        tuple:
            - df_processed: 전처리된 DataFrame (is_anomaly 컬럼 추가)
            - feature_cols: 특성 컬럼 이름 리스트
            - scaler: 학습된 StandardScaler 객체
            - train_stats: 통계량 dict

    처리 과정:
        1. 수치형 컬럼 추출
        2. 결측치를 0으로 채우기
        3. Z-score 계산하여 이상 레이블 생성
        4. StandardScaler로 정규화

    데이터 누출 방지:
        - Train 데이터: train_stats=None으로 호출 → 새로 통계 계산
        - Val/Test 데이터: Train의 train_stats 전달 → 동일 통계 사용
    """
    print("Preparing features and preprocessing...")

    # ----------------------------------------------------------------------
    # Step 1: 수치형 컬럼 추출
    # ----------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = []

    # TIMESTAMP가 숫자형으로 인식되면 제외
    if 'TIMESTAMP' in df.columns and 'TIMESTAMP' in numeric_cols:
        exclude_cols.append('TIMESTAMP')

    # 실제 특성 컬럼 (센서 데이터)
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # ----------------------------------------------------------------------
    # Step 2: 결측치 처리
    # ----------------------------------------------------------------------
    df_processed = df.copy()
    df_processed[feature_cols] = df_processed[feature_cols].fillna(0)

    feature_data = df_processed[feature_cols]

    # ----------------------------------------------------------------------
    # Step 3: Z-score 기반 이상 레이블링
    # ----------------------------------------------------------------------
    # Train 데이터의 통계로 이상 탐지 (데이터 누출 방지)
    if train_stats is None:
        # Train 데이터: 새로 통계 계산
        mean_vals = feature_data.mean()
        std_vals = feature_data.std() + 1e-8  # 0으로 나누기 방지
        train_stats = {'mean': mean_vals, 'std': std_vals}
    else:
        # Val/Test 데이터: Train 통계 사용
        mean_vals = train_stats['mean']
        std_vals = train_stats['std']

    # Z-score 계산: |값 - 평균| / 표준편차
    # 각 행(시간)별 평균 Z-score 계산
    z_scores = np.abs((feature_data - mean_vals) / std_vals).mean(axis=1)

    # 상위 10% (90 percentile)를 이상으로 분류
    threshold = np.percentile(z_scores, 90)
    df_processed['is_anomaly'] = z_scores > threshold

    # ----------------------------------------------------------------------
    # Step 4: 정규화 (StandardScaler)
    # ----------------------------------------------------------------------
    # 평균 0, 표준편차 1로 스케일링
    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])

    print(f"전처리 완료: {len(feature_cols)}개 특성")
    print(f"이상 데이터 비율: {df_processed['is_anomaly'].mean():.2%}")

    return df_processed, feature_cols, scaler, train_stats


# ==============================================================================
# 4. 시계열 데이터 생성
# ==============================================================================

def create_time_series_data(data, feature_cols, sequence_length=10, prediction_horizon=5):
    """
    Sliding Window 방식으로 시계열 학습 데이터 생성

    개념:
        [과거 10 스텝] → [미래 5 스텝 내 이상 발생?]
         ↑                    ↑
      sequence_length    prediction_horizon

    Args:
        data (DataFrame): 전처리된 데이터 (is_anomaly 컬럼 포함)
        feature_cols (list): 특성 컬럼 리스트
        sequence_length (int): 입력 시퀀스 길이 (과거 몇 스텝)
        prediction_horizon (int): 예측 구간 (미래 몇 스텝)

    Returns:
        tuple:
            - X: 입력 시퀀스 (shape: [샘플수, sequence_length, 특성수])
            - y: 타겟 레이블 (shape: [샘플수])
                 1: 미래 구간에 이상 발생
                 0: 정상

    예시:
        sequence_length=10, prediction_horizon=5

        시간:  t-9  t-8  ...  t-1  t  |  t+1  t+2  t+3  t+4  t+5
               [--------입력---------]    [-----예측 구간-----]

        X[0] = data[0:10]의 센서값
        y[0] = data[11:16] 중 이상 발생 여부 (any)
    """
    print(f"Creating time series data... (sequence_length: {sequence_length}, prediction_horizon: {prediction_horizon})")

    X, y = [], []
    feature_data = data[feature_cols].values  # numpy 배열로 변환

    # ----------------------------------------------------------------------
    # Step 1: 미래 이상 여부 사전 계산
    # ----------------------------------------------------------------------
    future_anomalies = []
    for i in range(len(data) - prediction_horizon):
        # i+1부터 i+1+prediction_horizon까지의 미래 구간
        future_window = data['is_anomaly'].iloc[i+1:i+1+prediction_horizon]
        # 미래 구간 중 하나라도 이상이면 1, 모두 정상이면 0
        future_anomalies.append(1 if future_window.any() else 0)

    # ----------------------------------------------------------------------
    # Step 2: Sliding Window로 시퀀스 생성
    # ----------------------------------------------------------------------
    for i in range(sequence_length, len(feature_data) - prediction_horizon):
        # 입력: [i-sequence_length : i]의 과거 데이터
        X.append(feature_data[i-sequence_length:i])

        # 출력: i 시점 이후의 미래 이상 여부
        y.append(future_anomalies[i])

    # numpy 배열로 변환 (PyTorch 입력용)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"시계열 데이터 생성 완료: X shape={X.shape}, y shape={y.shape}")
    return X, y


def create_multi_output_data(data, feature_cols, sequence_length=10, prediction_steps=5):
    """
    다중 출력 모델용 시계열 데이터 생성

    단일 출력과의 차이:
        - 단일: 미래 이상 여부만 예측 (binary)
        - 다중: 미래 센서값 + 이상 여부 동시 예측

    Args:
        data (DataFrame): 전처리된 데이터
        feature_cols (list): 특성 컬럼
        sequence_length (int): 입력 시퀀스 길이
        prediction_steps (int): 예측할 미래 스텝 수

    Returns:
        tuple:
            - X: 입력 시퀀스 (shape: [N, sequence_length, 특성수])
            - y_values: 미래 센서값 (shape: [N, prediction_steps, 특성수])
            - y_anomalies: 미래 이상 여부 (shape: [N, prediction_steps])
    """
    X, y_values, y_anomalies = [], [], []
    feature_data = data[feature_cols].values

    for i in range(sequence_length, len(feature_data) - prediction_steps):
        # 입력: 과거 시퀀스
        X.append(feature_data[i-sequence_length:i])

        # 출력 1: 미래 센서값
        y_values.append(feature_data[i:i+prediction_steps])

        # 출력 2: 미래 이상 여부
        future_anomalies = data['is_anomaly'].iloc[i:i+prediction_steps].values
        y_anomalies.append(future_anomalies)

    return (np.array(X, dtype=np.float32),
            np.array(y_values, dtype=np.float32),
            np.array(y_anomalies, dtype=np.float32))


# ==============================================================================
# 5. LSTM 모델 정의
# ==============================================================================

def create_lstm_model(input_size, hidden_size=64, num_layers=2, dropout=0.2):
    """
    단일 출력 LSTM 모델 생성 (이상 확률 예측)

    모델 구조:
        Input (batch, seq_len, features)
            ↓
        LSTM Layers (hidden_size=64, num_layers=2)
            ↓
        Last Time Step Output
            ↓
        FC Layer 1 (64 → 32)
            ↓
        ReLU + Dropout
            ↓
        FC Layer 2 (32 → 1)
            ↓
        Sigmoid
            ↓
        Anomaly Probability [0-1]

    Args:
        input_size (int): 입력 특성 개수 (센서 개수)
        hidden_size (int): LSTM hidden state 크기
        num_layers (int): LSTM 레이어 수
        dropout (float): Dropout 비율

    Returns:
        LSTMPredictor: PyTorch 모델
    """
    # Sequential로는 LSTM의 출력 처리가 복잡하므로 커스텀 클래스 사용

    class LSTMPredictor(nn.Module):
        """LSTM 기반 이상 확률 예측 모델"""

        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()

            # LSTM 레이어
            # batch_first=True: 입력 형태 (batch, seq, feature)
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout  # 레이어 간 dropout
            )

            # Fully Connected 레이어들
            self.fc1 = nn.Linear(hidden_size, 32)  # 64 → 32
            self.relu = nn.ReLU()                  # 활성화 함수
            self.dropout = nn.Dropout(dropout)     # Dropout 정규화
            self.fc2 = nn.Linear(32, 1)            # 32 → 1 (이상 확률)
            self.sigmoid = nn.Sigmoid()            # 0~1 사이로 변환

        def forward(self, x):
            """
            순전파(Forward Pass)

            Args:
                x: 입력 텐서 (batch, seq_len, features)

            Returns:
                anomaly_prob: 이상 확률 (batch, 1)
            """
            # LSTM 통과
            # lstm_out: (batch, seq_len, hidden_size)
            # _: (hidden_state, cell_state) - 사용 안 함
            lstm_out, _ = self.lstm(x)

            # 마지막 시간 스텝의 출력만 사용
            last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

            # FC 레이어들 통과
            x = self.fc1(last_output)    # (batch, 32)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)              # (batch, 1)

            return self.sigmoid(x)       # (batch, 1), 값: [0, 1]

    return LSTMPredictor(input_size, hidden_size, num_layers, dropout)


def create_multi_output_lstm(input_size, hidden_size=128, num_layers=3,
                            prediction_steps=5, num_features=None):
    """
    다중 출력 LSTM 모델 생성 (센서값 + 이상 확률 동시 예측)

    모델 구조:
        Input (batch, seq_len, features)
            ↓
        Shared LSTM (hidden_size=128, num_layers=3)
            ↓
        Last Time Step Output
            ↓
         ┌────────┴────────┐
         ↓                 ↓
    Value Predictor   Anomaly Predictor
    (센서값 예측)      (이상 확률 예측)
         ↓                 ↓
    FC: 128→64→output  FC: 128→32→output
         ↓                 ↓
    (batch, 5, features)  (batch, 5)

    Args:
        input_size (int): 입력 특성 개수
        hidden_size (int): LSTM hidden state 크기 (더 큼)
        num_layers (int): LSTM 레이어 수 (더 깊음)
        prediction_steps (int): 예측할 미래 스텝 수
        num_features (int): 출력 특성 개수

    Returns:
        MultiOutputLSTM: PyTorch 모델
    """

    class MultiOutputLSTM(nn.Module):
        """다중 출력 LSTM 모델 (Multi-task Learning)"""

        def __init__(self):
            super().__init__()

            # 공유 LSTM 백본
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.2
            )

            # 헤드 1: 센서값 예측 (Regression)
            self.value_predictor = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, prediction_steps * num_features)
                # 출력: (batch, prediction_steps * num_features)
                # 나중에 reshape: (batch, prediction_steps, num_features)
            )

            # 헤드 2: 이상 확률 예측 (Binary Classification)
            self.anomaly_predictor = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, prediction_steps),  # (batch, prediction_steps)
                nn.Sigmoid()  # 각 스텝별 이상 확률 [0, 1]
            )

        def forward(self, x):
            """
            순전파

            Args:
                x: 입력 텐서 (batch, seq_len, features)

            Returns:
                tuple:
                    - values: 예측 센서값 (batch, prediction_steps, features)
                    - anomalies: 예측 이상 확률 (batch, prediction_steps)
            """
            # 공유 LSTM 통과
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

            # 헤드 1: 센서값 예측
            values = self.value_predictor(last_output)
            # reshape: (batch, pred_steps * features) → (batch, pred_steps, features)
            values = values.view(-1, prediction_steps, num_features)

            # 헤드 2: 이상 확률 예측
            anomalies = self.anomaly_predictor(last_output)

            return values, anomalies

    return MultiOutputLSTM()


# ==============================================================================
# 6. 모델 학습
# ==============================================================================

def train_lstm_model(model, X_train, y_train, X_val, y_val,
                    epochs=50, batch_size=32, learning_rate=0.001):
    """
    단일 출력 LSTM 모델 학습

    Args:
        model: PyTorch 모델
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        learning_rate (float): 학습률

    Returns:
        tuple:
            - trained_model: 학습된 모델
            - train_losses: 에포크별 학습 손실 리스트
            - val_losses: 에포크별 검증 손실 리스트

    주요 기법:
        - Class Imbalance 처리: Weighted BCE Loss
        - Learning Rate Scheduling: ReduceLROnPlateau
        - Early Stopping 가능 (현재 미구현)
    """
    # ----------------------------------------------------------------------
    # 디바이스 설정 (GPU/CPU)
    # ----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ----------------------------------------------------------------------
    # 데이터 로더 생성
    # ----------------------------------------------------------------------
    # TensorDataset: numpy → PyTorch Tensor
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------------------------------------------------
    # Class Imbalance 처리
    # ----------------------------------------------------------------------
    # 이상 데이터는 전체의 10%만 있음 (불균형)
    # → 이상 샘플의 loss에 더 큰 가중치 부여
    num_pos = (y_train == 1).sum()  # 이상 샘플 수
    num_neg = (y_train == 0).sum()  # 정상 샘플 수

    if num_pos > 0:
        # pos_weight = 정상 수 / 이상 수
        # 예: 4500 / 500 = 9.0 (이상 샘플에 9배 가중치)
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        print(f"클래스 불균형 처리: pos_weight={pos_weight.item():.2f} (양성: {num_pos}, 음성: {num_neg})")
    else:
        pos_weight = torch.tensor([1.0]).to(device)
        print("경고: 양성 샘플이 없습니다. pos_weight=1.0 사용")

    # ----------------------------------------------------------------------
    # Loss Function 및 Optimizer 정의
    # ----------------------------------------------------------------------
    criterion = nn.BCELoss(weight=None)  # 검증용 (가중치 없음)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler
    # 검증 손실이 5 에포크 동안 개선되지 않으면 LR을 0.5배로 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,   # 개선 없이 기다릴 에포크 수
        factor=0.5    # LR 감소 비율
    )

    train_losses, val_losses = [], []

    # ----------------------------------------------------------------------
    # 학습 루프
    # ----------------------------------------------------------------------
    print("모델 학습 시작...")
    for epoch in range(epochs):
        # ========== Training Phase ==========
        model.train()  # 학습 모드 (Dropout, BatchNorm 활성화)
        train_loss = 0

        for batch_X, batch_y in train_loader:
            # 데이터를 디바이스로 이동
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Gradient 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(batch_X).squeeze(-1)  # (batch, 1) → (batch,)

            # Weighted Loss 계산
            # batch_y == 1인 샘플에는 pos_weight, 아니면 1.0
            batch_weights = torch.where(
                batch_y == 1,
                pos_weight.squeeze(),
                torch.tensor(1.0).to(device)
            )
            loss = nn.functional.binary_cross_entropy(
                outputs, batch_y, weight=batch_weights
            )

            # 역전파 및 파라미터 업데이트
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ========== Validation Phase ==========
        model.eval()  # 평가 모드 (Dropout, BatchNorm 비활성화)
        val_loss = 0

        with torch.no_grad():  # Gradient 계산 안 함 (메모리 절약)
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # 평균 손실 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning Rate 조정
        scheduler.step(avg_val_loss)

        # 10 에포크마다 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses


def train_multi_output_model(model, X_train, y_values_train, y_anomalies_train,
                            X_val, y_values_val, y_anomalies_val,
                            epochs=50, batch_size=32):
    """
    다중 출력 LSTM 모델 학습

    Args:
        model: MultiOutputLSTM 모델
        X_train: 입력 시퀀스
        y_values_train: 타겟 센서값
        y_anomalies_train: 타겟 이상 여부
        X_val, y_values_val, y_anomalies_val: 검증 데이터
        epochs (int): 에포크 수
        batch_size (int): 배치 크기

    Returns:
        trained_model: 학습된 모델

    Loss Function:
        Total Loss = MSE(센서값) + BCE(이상 확률)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 데이터 로더 생성 (3개의 타겟)
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_values_train),
        torch.tensor(y_anomalies_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss Functions
    mse_loss = nn.MSELoss()     # 센서값 예측용 (Regression)
    bce_loss = nn.BCELoss()     # 이상 확률 예측용 (Classification)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_values, batch_anomalies in train_loader:
            # 데이터를 디바이스로 이동
            batch_X = batch_X.to(device)
            batch_values = batch_values.to(device)
            batch_anomalies = batch_anomalies.to(device)

            optimizer.zero_grad()

            # 순전파 (2개의 출력)
            pred_values, pred_anomalies = model(batch_X)

            # 각 헤드별 Loss 계산
            value_loss = mse_loss(pred_values, batch_values)      # 센서값 MSE
            anomaly_loss = bce_loss(pred_anomalies, batch_anomalies)  # 이상 BCE

            # Total Loss = 두 Loss의 합
            loss = value_loss + anomaly_loss

            # 역전파 및 업데이트
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 10 에포크마다 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    return model


# ==============================================================================
# 7. 예측 및 경고 생성
# ==============================================================================

def predict_future_anomalies(model, X_test, threshold=0.5):
    """
    테스트 데이터에 대한 이상 확률 예측

    Args:
        model: 학습된 PyTorch 모델
        X_test (np.ndarray): 테스트 입력 시퀀스
        threshold (float): 이상 판정 임계값 (기본: 0.5)

    Returns:
        tuple:
            - anomaly_probs: 이상 확률 배열 [0-1]
            - anomaly_labels: 이진 레이블 {0, 1}

    예시:
        probs = [0.23, 0.78, 0.45, 0.92]
        labels = [0, 1, 0, 1]  (threshold=0.5)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 평가 모드

    with torch.no_grad():  # Gradient 계산 안 함
        # numpy → Tensor → 디바이스 이동
        X_test_tensor = torch.tensor(X_test).to(device)

        # 예측
        predictions = model(X_test_tensor).cpu().numpy()

    # reshape: (N, 1) → (N,)
    anomaly_probs = predictions.reshape(-1)

    # 임계값 기준으로 이진 레이블 생성
    anomaly_labels = (anomaly_probs > threshold).astype(int)

    return anomaly_probs, anomaly_labels


def generate_alerts(anomaly_probs, lot_numbers=None,
                alert_threshold=0.7, warning_threshold=0.5):
    """
    이상 확률을 기반으로 경고 메시지 생성

    Args:
        anomaly_probs (array): 이상 확률 [0-1]
        lot_numbers (list, optional): 샘플 식별자 (TIMESTAMP 또는 ID)
        alert_threshold (float): 위험 경고 임계값 (기본: 0.7)
        warning_threshold (float): 주의 경고 임계값 (기본: 0.5)

    Returns:
        list of dict: 경고 목록
            각 경고 dict:
                - sample_id: 샘플 식별자
                - alert_level: 'CRITICAL' or 'WARNING'
                - probability: 이상 확률
                - message: 경고 메시지
                - action: 권장 조치
                - timestamp: 경고 생성 시각

    경고 레벨:
        - CRITICAL: probability ≥ 0.7 (즉시 점검 필요)
        - WARNING: 0.5 ≤ probability < 0.7 (예방 점검 권장)
        - (무시): probability < 0.5
    """
    alerts = []

    for i, prob in enumerate(anomaly_probs):
        # 샘플 식별자 (lot_numbers가 없으면 인덱스 사용)
        sample_id = lot_numbers[i] if lot_numbers else f"sample_{i:04d}"

        # 경고 레벨 판정
        if prob >= alert_threshold:
            # 위험 경고
            alert_level = "CRITICAL"
            message = f"위험! : Sample {sample_id} - 이상 발생 확률 {prob:.1%}"
            action = "즉시 점검 필요"
        elif prob >= warning_threshold:
            # 주의 경고
            alert_level = "WARNING"
            message = f"경고: Sample {sample_id} - 이상 징후 감지 (확률 {prob:.1%})"
            action = "예방 점검 권장"
        else:
            # 정상 - 경고 생성 안 함
            continue

        # 경고 정보 저장
        alerts.append({
            'sample_id': sample_id,
            'alert_level': alert_level,
            'probability': prob,
            'message': message,
            'action': action,
            'timestamp': datetime.now()
        })

    return alerts


def calculate_remaining_useful_life(model, current_data_seq, max_horizon=100,
                                failure_threshold=0.8):
    """
    잔여 유효 수명(RUL) 예측

    개념:
        현재 설비 상태에서 시작하여, 이상 확률이 failure_threshold에
        도달할 때까지 몇 스텝이 걸리는지 예측

    Args:
        model: 학습된 모델
        current_data_seq (np.ndarray): 현재 시퀀스 (sequence_length, features)
        max_horizon (int): 최대 예측 범위 (기본: 100 스텝)
        failure_threshold (float): 고장 판정 확률 (기본: 0.8)

    Returns:
        int: 예상 잔여 수명 (스텝 수)

    동작 방식:
        1. 현재 상태로 이상 확률 예측
        2. 시간에 따른 열화를 시뮬레이션 (degradation_factor)
        3. 열화 적용 확률이 threshold 초과 시 해당 horizon 반환

    주의:
        - 현재는 단순 시뮬레이션 (degradation_factor = 1 + horizon * 0.015)
        - 실제 적용 시 물리 모델 기반 열화 계수 사용 권장
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # numpy → Tensor (batch 차원 추가)
    sample_tensor = torch.tensor(current_data_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # 미래 시점별로 확률 계산
        for horizon in range(1, max_horizon + 1):
            # 현재 상태로 이상 확률 예측
            prob = model(sample_tensor).cpu().item()

            # 시간에 따른 가상 열화 적용
            # horizon이 클수록 설비 상태가 나빠짐
            degradation_factor = 1 + (horizon * 0.015)
            adjusted_prob = min(prob * degradation_factor, 1.0)

            # 고장 임계값 초과 시 해당 horizon 반환
            if adjusted_prob >= failure_threshold:
                return horizon

    # max_horizon까지도 고장 안 남 → max_horizon 반환
    return max_horizon


# ==============================================================================
# 8. 실시간 모니터링
# ==============================================================================

def real_time_monitoring(model, scaler, feature_cols, new_data_stream,
                        sequence_length=10, update_interval=1):
    """
    실시간 데이터 스트림 모니터링 및 이상 예측

    개념:
        - 순환 버퍼(Circular Buffer) 방식
        - 새 데이터가 들어올 때마다 최근 sequence_length개로 예측

    Args:
        model: 학습된 모델
        scaler: StandardScaler 객체
        feature_cols (list): 특성 컬럼 리스트
        new_data_stream: Iterator, yields (timestamp, data_row)
        sequence_length (int): 시퀀스 길이
        update_interval (int): 업데이트 주기 (초) - 현재 미사용

    Returns:
        str: 전체 모니터링 기간 중 최대 위험 레벨
            - '0': 정상 (모든 확률 < 0.3)
            - '1': 주의 발생 (0.3 ≤ 일부 확률 < 0.7)
            - '2': 위험 발생 (일부 확률 ≥ 0.7)

    처리 흐름:
        1. 새 데이터 도착
        2. 전처리 (scaler)
        3. 버퍼에 추가
        4. 버퍼 길이 ≥ sequence_length이면 예측
        5. 확률에 따라 경고 출력
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 순환 버퍼 초기화
    data_buffer = []
    max_anomaly_prob = 0.0
    max_status = '0'  # 상태 코드

    print("실시간 모니터링 시작...")

    # 데이터 스트림 처리
    for timestamp, new_data in new_data_stream:
        # ----------------------------------------------------------------------
        # Step 1: 새 데이터 전처리
        # ----------------------------------------------------------------------
        # 특성 컬럼만 추출 → reshape → scaler 적용
        processed_data = scaler.transform(new_data[feature_cols].values.reshape(1, -1))
        data_buffer.append(processed_data[0])

        # ----------------------------------------------------------------------
        # Step 2: 예측 (버퍼가 충분히 쌓이면)
        # ----------------------------------------------------------------------
        if len(data_buffer) >= sequence_length:
            # 최근 sequence_length개 데이터로 시퀀스 구성
            input_sequence = np.array(data_buffer[-sequence_length:])

            # numpy → Tensor (batch 차원 추가)
            input_tensor = torch.tensor(input_sequence).unsqueeze(0).float().to(device)

            # 이상 확률 예측
            with torch.no_grad():
                anomaly_prob = model(input_tensor).cpu().numpy()[0, 0]

            # 버퍼가 너무 커지면 오래된 데이터 제거
            if len(data_buffer) > sequence_length * 2:
                data_buffer = data_buffer[-sequence_length:]

            # ----------------------------------------------------------------------
            # Step 3: 확률에 따른 경고 출력
            # ----------------------------------------------------------------------
            if anomaly_prob >= 0.7:
                # 위험 경고
                print(f"[{timestamp}] 위험 경고: 이상 확률 {anomaly_prob:.1%}")
                max_status = '2'
                max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)
            elif anomaly_prob >= 0.3:
                # 주의 경고
                print(f"[{timestamp}] 주의: 이상 징후 감지 (확률 {anomaly_prob:.1%})")
                if max_status < '1':
                    max_status = '1'
                max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)
            else:
                # 정상
                print(f"[{timestamp}] 안전: 이상 징후 발생 가능성 낮음 (확률 {anomaly_prob:.1%})")
                max_anomaly_prob = max(max_anomaly_prob, anomaly_prob)

    print(f"\n모니터링 완료: 최대 이상 확률 {max_anomaly_prob:.1%}, 상태: {max_status}")
    return max_status


def create_mock_real_time_stream(test_df, feature_cols, num_samples=10):
    """
    실시간 데이터 스트림 시뮬레이션 (Generator)

    테스트용으로 데이터프레임의 행을 순차적으로 yield

    Args:
        test_df (DataFrame): 테스트 데이터
        feature_cols (list): 특성 컬럼 (미사용, 호환성용)
        num_samples (int): 생성할 샘플 수

    Yields:
        tuple: (timestamp, data_row)

    동작:
        - test_df에서 num_samples개 행을 순차적으로 yield
        - 실시간 시뮬레이션을 위해 0.5초 간격으로 sleep
    """
    print(f"실시간 데이터 스트림 생성 중... (총 {num_samples}개 샘플)")

    # 처음 num_samples개 행 추출
    sample_data = test_df.head(num_samples)

    start_time = datetime.now()

    for i, (_, row) in enumerate(sample_data.iterrows()):
        # 가상 타임스탬프 생성 (1초 간격)
        current_timestamp = start_time + timedelta(seconds=i)

        yield current_timestamp, row

        # 실시간 시뮬레이션: 0.5초 대기
        # 실제 환경에서는 제거 가능
        time.sleep(0.5)


# ==============================================================================
# 9. 시나리오 실행 함수
# ==============================================================================

def run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler):
    """
    시나리오 1: 단일 출력 이상 징후 예측 모델

    전체 파이프라인:
        1. 시계열 데이터 생성
        2. LSTM 모델 생성
        3. 모델 학습
        4. 테스트 데이터 예측
        5. 경고 생성
        6. RUL 예측 (첫 번째 샘플)

    Args:
        train_df, val_df, test_df: 전처리된 데이터
        feature_cols: 특성 컬럼 리스트
        scaler: StandardScaler 객체

    Returns:
        tuple:
            - trained_model: 학습된 모델
            - scaler: scaler 객체
    """
    print("1. 단일 출력 이상 징후 예측 모델")

    # 파라미터 설정
    SEQ_LENGTH = 2          # 짧은 시퀀스 (데모용)
    PREDICTION_HORIZON = 1  # 1 스텝 후 예측

    # ----------------------------------------------------------------------
    # Step 1: 시계열 데이터 생성
    # ----------------------------------------------------------------------
    X_train, y_train = create_time_series_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_time_series_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_time_series_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)

    # 데이터가 부족하면 건너뛰기
    if X_train.shape[0] == 0:
        print("학습 데이터가 부족하여 시나리오 1을 건너뜁니다.")
        return None, None

    # ----------------------------------------------------------------------
    # Step 2: 모델 생성
    # ----------------------------------------------------------------------
    input_size = X_train.shape[2]  # 특성 개수
    model = create_lstm_model(input_size=input_size)

    # ----------------------------------------------------------------------
    # Step 3: 모델 학습
    # ----------------------------------------------------------------------
    trained_model, _, _ = train_lstm_model(
        model, X_train, y_train, X_val, y_val,
        epochs=10  # 빠른 실행을 위해 10 에포크만
    )

    # ----------------------------------------------------------------------
    # Step 4: 테스트 데이터 예측
    # ----------------------------------------------------------------------
    print("\n 테스트 데이터 예측 및 경고 생성")
    probs, labels = predict_future_anomalies(trained_model, X_test)

    # 샘플 식별자 생성 (TIMESTAMP 또는 equipment_id)
    test_indices = test_df.index[SEQ_LENGTH : len(probs) + SEQ_LENGTH]

    if 'TIMESTAMP' in test_df.columns:
        timestamps = test_df.loc[test_indices, 'TIMESTAMP'].tolist()
        lot_numbers = [str(ts) for ts in timestamps]
    elif 'equipment_id' in test_df.columns:
        lot_numbers = test_df.loc[test_indices, 'equipment_id'].tolist()
    else:
        lot_numbers = None

    # ----------------------------------------------------------------------
    # Step 5: 경고 생성
    # ----------------------------------------------------------------------
    alerts = generate_alerts(probs, lot_numbers=lot_numbers)

    if alerts:
        print(f"총 {len(alerts)}개의 경고가 생성되었습니다.")
        # 처음 5개만 출력
        for alert in alerts[:5]:
            print(f"  - {alert['message']}")
    else:
        print("생성된 경고가 없습니다.")

    # ----------------------------------------------------------------------
    # Step 6: RUL 예측 예시
    # ----------------------------------------------------------------------
    print("\nRUL 예측 예시 (테스트 데이터 첫 번째 샘플)")
    first_test_sequence = X_test[0]
    predicted_rul = calculate_remaining_useful_life(trained_model, first_test_sequence)
    print(f"첫 번째 테스트 샘플의 예측 RUL: {predicted_rul} 스텝")
    print("=" * 50)

    return trained_model, scaler


def run_multi_output_scenario(train_df, val_df, test_df, feature_cols):
    """
    시나리오 2: 다중 출력 동시 예측 모델

    센서값 + 이상 확률을 동시에 예측

    Args:
        train_df, val_df, test_df: 전처리된 데이터
        feature_cols: 특성 컬럼 리스트

    Returns:
        np.ndarray: 첫 번째 테스트 샘플의 예측 이상 확률
    """
    print("\n2. 다중 출력 동시 예측 모델")

    SEQ_LENGTH = 2
    PREDICTION_STEPS = 1

    # 시계열 데이터 생성 (다중 출력용)
    X_train, y_vals_train, y_anom_train = create_multi_output_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
    X_val, y_vals_val, y_anom_val = create_multi_output_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
    X_test, y_vals_test, y_anom_test = create_multi_output_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)

    if X_train.shape[0] == 0:
        print("학습 데이터가 부족하여 시나리오 2를 건너뜁니다.")
        return

    # 모델 생성
    input_size = X_train.shape[2]
    num_features = y_vals_train.shape[2]
    model = create_multi_output_lstm(
        input_size=input_size,
        prediction_steps=PREDICTION_STEPS,
        num_features=num_features
    )

    # 모델 학습
    trained_model = train_multi_output_model(
        model, X_train, y_vals_train, y_anom_train,
        X_val, y_vals_val, y_anom_val,
        epochs=20
    )

    # 예측 예시 (첫 번째 테스트 샘플)
    print("\n다중 출력 모델 예측 예시 (테스트 데이터 첫 번째 샘플)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()

    sample_x = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0).to(device)
    pred_vals, pred_anoms = trained_model(sample_x)

    print(f"입력 데이터 형태: {sample_x.shape}")
    print(f"예측 센서값 형태: {pred_vals.shape}")
    print(f"예측 이상확률 형태: {pred_anoms.shape}")
    print(f"\n예측된 미래 이상 발생 확률 ({PREDICTION_STEPS} 스텝):")
    pred_value = pred_anoms.detach().cpu().numpy().flatten()
    print(pred_value)
    print("=" * 50)

    return pred_value


def run_real_time_monitoring_scenario(trained_model, scaler, feature_cols, test_df):
    """
    시나리오 3: 실시간 모니터링 시뮬레이션

    테스트 데이터를 실시간 스트림처럼 처리

    Args:
        trained_model: 학습된 모델
        scaler: StandardScaler
        feature_cols: 특성 컬럼
        test_df: 테스트 데이터

    Returns:
        str: 최종 이상 상태 ('0', '1', '2')
    """
    print("\n3. 실시간 모니터링 시뮬레이션")
    print("=" * 50)

    # Mock 데이터 스트림 생성
    mock_data_stream = create_mock_real_time_stream(test_df, feature_cols)

    print("실시간 모니터링을 시작합니다...")

    # 실시간 모니터링 실행
    anomaly_status = real_time_monitoring(
        model=trained_model,
        scaler=scaler,
        feature_cols=feature_cols,
        new_data_stream=mock_data_stream,
        sequence_length=2,
        update_interval=1
    )

    print("\n실시간 모니터링 시뮬레이션이 완료되었습니다.")
    print("=" * 50)
    return anomaly_status


# ==============================================================================
# 10. 시각화 함수
# ==============================================================================

def visualize_predictions(anomaly_probs, actual_labels=None, save_path=None):
    """
    예측 결과 시각화

    2개의 서브플롯:
        1. 이상 확률 추이 (시간에 따른 확률 변화)
        2. 실제 vs 예측 비교 (actual_labels가 있는 경우)

    Args:
        anomaly_probs (array): 예측 이상 확률
        actual_labels (array, optional): 실제 레이블
        save_path (str, optional): 저장 경로
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ----------------------------------------------------------------------
    # 서브플롯 1: 이상 확률 추이
    # ----------------------------------------------------------------------
    axes[0].plot(anomaly_probs, label='Anomaly Probability', color='red', alpha=0.7)
    axes[0].axhline(y=0.5, color='orange', linestyle='--', label='Warning Threshold')
    axes[0].axhline(y=0.7, color='red', linestyle='--', label='Alert Threshold')
    axes[0].fill_between(range(len(anomaly_probs)), anomaly_probs, alpha=0.3, color='red')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Anomaly Probability')
    axes[0].set_title('Future Anomaly Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ----------------------------------------------------------------------
    # 서브플롯 2: 실제 vs 예측
    # ----------------------------------------------------------------------
    if actual_labels is not None:
        axes[1].plot(actual_labels, label='Actual', color='blue', alpha=0.7)
        predicted = (anomaly_probs > 0.5).astype(int)
        axes[1].plot(predicted, label='Predicted', color='red', alpha=0.7, linestyle='--')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Anomaly Label')
        axes[1].set_title('Actual vs Predicted Anomalies')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_rul_distribution(rul_predictions, save_path=None):
    """
    RUL 분포 시각화

    여러 샘플의 RUL을 히스토그램으로 표시

    Args:
        rul_predictions (array): RUL 예측값 배열
        save_path (str, optional): 저장 경로
    """
    plt.figure(figsize=(10, 6))

    # 히스토그램
    plt.hist(rul_predictions, bins=30, edgecolor='black', alpha=0.7)

    # 평균 RUL 표시
    plt.axvline(
        x=np.mean(rul_predictions),
        color='red',
        linestyle='--',
        label=f'Mean RUL: {np.mean(rul_predictions):.1f}'
    )

    plt.xlabel('Remaining Useful Life (time steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Remaining Useful Life')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()


# ==============================================================================
# 파일 끝
# ==============================================================================
