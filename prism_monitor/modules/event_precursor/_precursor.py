import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time


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


def create_time_series_data(data, feature_cols, sequence_length=10, prediction_horizon=5):
    print(f"Creating time series data... (sequence_length: {sequence_length}, prediction_horizon: {prediction_horizon})")
    
    X, y = [], []
    feature_data = data[feature_cols].values

    future_anomalies = []
    for i in range(len(data) - prediction_horizon):
        future_window = data['is_anomaly'].iloc[i+1:i+1+prediction_horizon]
        future_anomalies.append(1 if future_window.any() else 0)
    
    # 시계열 시퀀스 생성
    for i in range(sequence_length, len(feature_data) - prediction_horizon):
        X.append(feature_data[i-sequence_length:i])
        y.append(future_anomalies[i-sequence_length])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"시계열 데이터 생성 완료: X shape={X.shape}, y shape={y.shape}")
    return X, y

def create_multi_output_data(data, feature_cols, sequence_length=10, prediction_steps=5):
    X, y_values, y_anomalies = [], [], []
    feature_data = data[feature_cols].values
    
    for i in range(sequence_length, len(feature_data) - prediction_steps):
        X.append(feature_data[i-sequence_length:i])

        y_values.append(feature_data[i:i+prediction_steps])

        future_anomalies = data['is_anomaly'].iloc[i:i+prediction_steps].values
        y_anomalies.append(future_anomalies)
    
    return (np.array(X, dtype=np.float32), 
            np.array(y_values, dtype=np.float32),
            np.array(y_anomalies, dtype=np.float32))


def create_lstm_model(input_size, hidden_size=64, num_layers=2, dropout=0.2):
    model = nn.Sequential(
        nn.LSTM(input_size, hidden_size, num_layers, 
                batch_first=True, dropout=dropout),
        nn.Linear(hidden_size, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    # LSTM의 경우 Sequential이 제대로 작동하지 않으므로 간단한 wrapper 필요
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            x = self.fc1(last_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)
    
    return LSTMPredictor(input_size, hidden_size, num_layers, dropout)

def create_multi_output_lstm(input_size, hidden_size=128, num_layers=3, 
                            prediction_steps=5, num_features=None):
    
    class MultiOutputLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
            
            # 센서값 예측 헤드
            self.value_predictor = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, prediction_steps * num_features)
            )
            
            # 이상탐지 예측 헤드
            self.anomaly_predictor = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, prediction_steps),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            
            # 센서값 예측
            values = self.value_predictor(last_output)
            values = values.view(-1, prediction_steps, num_features)
            
            # 이상 확률 예측
            anomalies = self.anomaly_predictor(last_output)
            
            return values, anomalies
    
    return MultiOutputLSTM()


def train_lstm_model(model, X_train, y_train, X_val, y_val, 
                    epochs=50, batch_size=32, learning_rate=0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses, val_losses = [], []
    
    print("모델 학습 시작...")
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1) # 원래 squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze(-1) # 데이터 좀 커지면 원래대로 squeeze() 사용 가능
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        # print(len(train_loader))
        # print(len(val_loader))
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses

def train_multi_output_model(model, X_train, y_values_train, y_anomalies_train,
                            X_val, y_values_val, y_anomalies_val,
                            epochs=50, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_values_train),
        torch.tensor(y_anomalies_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_values, batch_anomalies in train_loader:
            batch_X = batch_X.to(device)
            batch_values = batch_values.to(device)
            batch_anomalies = batch_anomalies.to(device)
            
            optimizer.zero_grad()
            
            pred_values, pred_anomalies = model(batch_X)

            value_loss = mse_loss(pred_values, batch_values)
            anomaly_loss = bce_loss(pred_anomalies, batch_anomalies)
            loss = value_loss + anomaly_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")
    
    return model


def predict_future_anomalies(model, X_test, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    anomaly_probs = predictions.reshape(-1) # squeeze()
    anomaly_labels = (anomaly_probs > threshold).astype(int)
    
    return anomaly_probs, anomaly_labels

def generate_alerts(anomaly_probs, lot_numbers=None, # feature_names
                alert_threshold=0.7, warning_threshold=0.5):
    alerts = []
    
    for i, prob in enumerate(anomaly_probs):
        lot_no = lot_numbers[i] if lot_numbers else f"LOT_{i:04d}"
        
        if prob >= alert_threshold:
            alert_level = "CRITICAL"
            message = f"🚨 위험: LOT {lot_no} - 24시간 내 이상 발생 확률 {prob:.1%}"
            action = "즉시 점검 필요"
        elif prob >= warning_threshold:
            alert_level = "WARNING"
            message = f"⚠️ 경고: LOT {lot_no} - 이상 징후 감지 (확률 {prob:.1%})"
            action = "예방 점검 권장"
        else:
            continue
        
        alerts.append({
            'lot_no': lot_no,
            'alert_level': alert_level,
            'probability': prob,
            'message': message,
            'action': action,
            'timestamp': datetime.now()
        })
    
    return alerts

# def calculate_remaining_useful_life(model, current_data, max_horizon=100, 
#                                 failure_threshold=0.8):
#     """잔여 유효 수명(RUL) 예측"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     rul_predictions = []
    
#     with torch.no_grad():
#         for sample in current_data:
#             sample_tensor = torch.tensor(sample).unsqueeze(0).to(device)
            
#             # 시간에 따른 이상 확률 예측
#             future_probs = []
#             for horizon in range(1, max_horizon + 1):
#                 # 여기서는 간단히 현재 데이터로 예측 (실제로는 rolling 예측 필요)
#                 prob = model(sample_tensor).cpu().numpy()[0, 0]
#                 # 시간에 따른 열화 시뮬레이션 (간단한 예시)
#                 degradation_factor = 1 + (horizon * 0.01)
#                 adjusted_prob = min(prob * degradation_factor, 1.0)
#                 future_probs.append(adjusted_prob)
                
#                 if adjusted_prob >= failure_threshold:
#                     rul_predictions.append(horizon)
#                     break
#             else:
#                 rul_predictions.append(max_horizon)
    
#     return np.array(rul_predictions)

def calculate_remaining_useful_life(model, current_data_seq, max_horizon=100, 
                                failure_threshold=0.8):
    """잔여 유효 수명(RUL) 예측 (시뮬레이션 기반 예시)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    sample_tensor = torch.tensor(current_data_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for horizon in range(1, max_horizon + 1):
            prob = model(sample_tensor).cpu().item()
            
            # 시간에 따른 가상 열화 계수
            degradation_factor = 1 + (horizon * 0.015) 
            adjusted_prob = min(prob * degradation_factor, 1.0)
            
            if adjusted_prob >= failure_threshold:
                return horizon
    
    return max_horizon


def run_single_output_scenario(train_df, val_df, test_df, feature_cols, scaler):
    print("1. 단일 출력 이상 징후 예측 모델")

    SEQ_LENGTH = 2
    PREDICTION_HORIZON = 1
    
    X_train, y_train = create_time_series_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_time_series_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_time_series_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)

    if X_train.shape[0] == 0:
        print("학습 데이터가 부족하여 시나리오 1을 건너뜁니다.")
        return None, None

    input_size = X_train.shape[2]
    model = create_lstm_model(input_size=input_size)
    
    # 테스트를 위해 낮은 epoch 설정
    trained_model, _, _ = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=20)

    print("\n 테스트 데이터 예측 및 경고 생성")
    probs, labels = predict_future_anomalies(trained_model, X_test)

    test_indices = test_df.index[SEQ_LENGTH : len(probs) + SEQ_LENGTH]
    lot_numbers = test_df.loc[test_indices, 'LOT_NO'].tolist() if 'LOT_NO' in test_df else None

    alerts = generate_alerts(probs, lot_numbers=lot_numbers)
    
    if alerts:
        print(f"총 {len(alerts)}개의 경고가 생성되었습니다.")
        for alert in alerts[:5]:
            print(f"  - {alert['message']}")
    else:
        print("생성된 경고가 없습니다.")
        
    # 잔여 유효 수명(RUL) 예측 예시
    print("\nRUL 예측 예시 (테스트 데이터 첫 번째 샘플)")
    first_test_sequence = X_test[0]
    predicted_rul = calculate_remaining_useful_life(trained_model, first_test_sequence)
    print(f"첫 번째 테스트 샘플의 예측 RUL: {predicted_rul} 스텝")
    print("=" * 50)
    
    # 실시간 모니터링에서 사용할 수 있도록 모델과 스케일러를 반환
    return trained_model, scaler


def run_multi_output_scenario(train_df, val_df, test_df, feature_cols):
    print("\n2. 다중 출력 동시 예측 모델")

    SEQ_LENGTH = 2
    PREDICTION_STEPS = 1

    X_train, y_vals_train, y_anom_train = create_multi_output_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
    X_val, y_vals_val, y_anom_val = create_multi_output_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
    X_test, y_vals_test, y_anom_test = create_multi_output_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)

    if X_train.shape[0] == 0:
        print("학습 데이터가 부족하여 시나리오 2를 건너뜁니다.")
        return

    input_size = X_train.shape[2]
    num_features = y_vals_train.shape[2]
    model = create_multi_output_lstm(input_size=input_size, prediction_steps=PREDICTION_STEPS, num_features=num_features)
    
    trained_model = train_multi_output_model(model, X_train, y_vals_train, y_anom_train,
                                                X_val, y_vals_val, y_anom_val, epochs=20)

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


def real_time_monitoring(model, scaler, feature_cols, new_data_stream, 
                        sequence_length=10, update_interval=1):
    """실시간 데이터 스트림 모니터링 및 예측"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 버퍼 초기화
    data_buffer = []
    
    print("실시간 모니터링 시작...")
    
    for timestamp, new_data in new_data_stream:
        # 데이터 전처리
        processed_data = scaler.transform(new_data[feature_cols].values.reshape(1, -1))
        data_buffer.append(processed_data[0])
        
        # 버퍼가 충분히 채워졌을 때만 예측
        if len(data_buffer) >= sequence_length:
            # 최근 sequence_length 만큼의 데이터 사용
            input_sequence = np.array(data_buffer[-sequence_length:])
            input_tensor = torch.tensor(input_sequence).unsqueeze(0).float().to(device)
            
            # 예측 수행
            with torch.no_grad():
                anomaly_prob = model(input_tensor).cpu().numpy()[0, 0]

            # 버퍼 크기 제한
            if len(data_buffer) > sequence_length * 2:
                data_buffer = data_buffer[-sequence_length:]

            # 경고 판단
            if anomaly_prob >= 0.7:
                print(f"[{timestamp}] 위험 경고: 이상 확률 {anomaly_prob:.1%}")
                return '2'
            elif anomaly_prob >= 0.3:
                print(f"[{timestamp}] 주의: 이상 징후 감지 (확률 {anomaly_prob:.1%})")
                return '1'
            else:
                print(f"[{timestamp}] 안전: 이상 징후 발생 가능성 낮음 (확률 {anomaly_prob:.1%})")
                return '0'



def run_real_time_monitoring_scenario(trained_model, scaler, feature_cols, test_df):
    print("\n3. 실시간 모니터링 시뮬레이션")
    print("=" * 50)

    mock_data_stream = create_mock_real_time_stream(test_df, feature_cols)
    
    print("실시간 모니터링을 시작합니다...")    
    # 실시간 모니터링 함수 호출
    # 이 함수는 새로운 데이터가 들어올 때마다 이상 징후를 예측하고 경고를 발생시킵니다
    anomaly_status = real_time_monitoring(
        model=trained_model,
        scaler=scaler,
        feature_cols=feature_cols,
        new_data_stream=mock_data_stream,
        sequence_length=2,  # 시계열 예측에 사용할 과거 데이터 길이
        update_interval=1   # 예측 업데이트 간격
    )
    
    print("\n실시간 모니터링 시뮬레이션이 완료되었습니다.")
    print("=" * 50)
    return anomaly_status


def create_mock_real_time_stream(test_df, feature_cols, num_samples=10):

    print(f"실시간 데이터 스트림 생성 중... (총 {num_samples}개 샘플)")

    sample_data = test_df.head(num_samples)

    start_time = datetime.now()
    
    for i, (_, row) in enumerate(sample_data.iterrows()):
        current_timestamp = start_time + timedelta(seconds=i)
        yield current_timestamp, row

        time.sleep(0.5)  # 시뮬레이션을 위해 0.5초 간격으로 데이터 생성. 실제로 할 때는 지워도 됨.


def visualize_predictions(anomaly_probs, actual_labels=None, save_path=None):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 이상 확률 추이
    axes[0].plot(anomaly_probs, label='Anomaly Probability', color='red', alpha=0.7)
    axes[0].axhline(y=0.5, color='orange', linestyle='--', label='Warning Threshold')
    axes[0].axhline(y=0.7, color='red', linestyle='--', label='Alert Threshold')
    axes[0].fill_between(range(len(anomaly_probs)), anomaly_probs, alpha=0.3, color='red')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Anomaly Probability')
    axes[0].set_title('Future Anomaly Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 실제 vs 예측 (실제 라벨이 있는 경우)
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
    """RUL 분포 시각화"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(rul_predictions, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(rul_predictions), color='red', linestyle='--', 
                label=f'Mean RUL: {np.mean(rul_predictions):.1f}')
    plt.xlabel('Remaining Useful Life (time steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Remaining Useful Life')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()