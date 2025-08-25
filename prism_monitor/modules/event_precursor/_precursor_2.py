import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle
import joblib

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
                sensor_cols = [col for col in numeric_cols if col != 'pno']
                
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

def create_unified_dataset(datasets):
    print("Creating unified dataset...")
    
    integrated_sensors = integrate_sensor_data(datasets)
    
    if 'semi_lot_manage' in datasets:
        main_df = datasets['semi_lot_manage'].copy()
        print(f"LOT data count: {len(main_df)} LOT")
    else:
        return pd.DataFrame()
    
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
    
    if 'semi_param_measure' in datasets:
        param_df = datasets['semi_param_measure']
        if 'lot_no' in param_df.columns:
            param_stats = param_df.groupby('lot_no')['measured_val'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            param_stats.columns = [f"param_{col}" if col != 'lot_no' else col 
                                        for col in param_stats.columns]
            
            main_df = main_df.merge(param_stats, on='lot_no', how='left')
    
    print(f"최종 통합 데이터셋: {main_df.shape}")
    return main_df

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

def create_lstm_model(input_size, hidden_size=64, num_layers=2, dropout=0.2):
    return LSTMPredictor(input_size, hidden_size, num_layers, dropout)


class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
                prediction_steps=5, num_features=None):
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
        
        self.prediction_steps = prediction_steps
        self.num_features = num_features
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # 센서값 예측
        values = self.value_predictor(last_output)
        values = values.view(-1, self.prediction_steps, self.num_features)
        
        # 이상 확률 예측
        anomalies = self.anomaly_predictor(last_output)
        
        return values, anomalies


def create_multi_output_lstm(input_size, hidden_size=128, num_layers=3, 
                            prediction_steps=5, num_features=None):
    return MultiOutputLSTM(input_size, hidden_size, num_layers, prediction_steps, num_features)



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
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
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


def save_trained_models(single_model, multi_model, scaler, feature_cols, test_df, model_dir="saved_models"):
    """학습된 모델과 전처리 정보를 저장"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 단일 출력 LSTM 모델 저장
    if single_model is not None:
        torch.save(single_model.state_dict(), os.path.join(model_dir, 'single_lstm_model.pth'))
        print(f"단일 출력 LSTM 모델 저장: {model_dir}/single_lstm_model.pth")
    
    # 다중 출력 LSTM 모델 저장
    if multi_model is not None:
        torch.save(multi_model.state_dict(), os.path.join(model_dir, 'multi_lstm_model.pth'))
        torch.save(multi_model, os.path.join(model_dir, 'multi_lstm_full_model.pth'))  # 전체 모델도 저장
        print(f"다중 출력 LSTM 모델 저장: {model_dir}/multi_lstm_model.pth")
    
    # 스케일러와 특성 정보 저장
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # 특성 컬럼과 기타 메타데이터 저장
    metadata = {
        'feature_cols': feature_cols,
        'input_size': len(feature_cols),
        'model_info': {
            'single_model_available': single_model is not None,
            'multi_model_available': multi_model is not None,
            'trained_at': datetime.now().isoformat()
        }
    }
    
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # 테스트 데이터도 시뮬레이션용으로 저장
    test_df.to_csv(os.path.join(model_dir, 'test_data.csv'), index=False)
    
    print(f"전처리 정보 및 메타데이터 저장 완료: {model_dir}/")

def load_trained_models(model_dir="saved_models"):
    """저장된 모델과 전처리 정보를 로드"""
    print(f"저장된 모델 로딩 중: {model_dir}")
    
    # 메타데이터 로드
    with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    feature_cols = metadata['feature_cols']
    input_size = metadata['input_size']
    
    # 스케일러 로드
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # 단일 출력 모델 로드
    single_model = None
    if metadata['model_info']['single_model_available']:
        single_model = create_lstm_model(input_size=input_size)
        single_model.load_state_dict(torch.load(os.path.join(model_dir, 'single_lstm_model.pth')))
        single_model.eval()
        print("단일 출력 LSTM 모델 로드 완료")
    
    # 다중 출력 모델 로드
    multi_model = None
    if metadata['model_info']['multi_model_available']:
        try:
            multi_model = torch.load(os.path.join(model_dir, 'multi_lstm_full_model.pth'))
            multi_model.eval()
            print("다중 출력 LSTM 모델 로드 완료")
        except:
            print("다중 출력 모델 로드 실패 - 모델 구조가 변경되었을 수 있습니다.")
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(os.path.join(model_dir, 'test_data.csv'))
    
    print(f"모델 로딩 완료 - 특성 수: {len(feature_cols)}")
    
    return single_model, multi_model, scaler, feature_cols, test_df, metadata

###################
# 모델 학습 및 저장 #
###################

def train_and_save_models(data_path):
    print("="*60)
    print("모델 훈련 모드: 데이터 전처리 및 LSTM 모델 학습")
    print("="*60)
    
    # 1. 데이터 준비 및 전처리
    print("1단계: 데이터 준비 및 전처리")
    all_datasets = load_and_explore_data(data_path)
    if not all_datasets:
        print("데이터 로딩 실패.")
        return
        
    unified_df = create_unified_dataset(all_datasets)
    if unified_df.empty:
        print("통합 데이터셋 생성 실패.")
        return

    processed_df, feature_cols, scaler = prepare_features(unified_df)
    print("=" * 40, "\n")

    # 2. 데이터 분할
    print("2단계: 데이터 분할")
    train_val_df, test_df = train_test_split(processed_df, test_size=0.1, shuffle=False)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, shuffle=False)
    
    print(f"학습 데이터: {train_df.shape}")
    print(f"검증 데이터: {val_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    print("=" * 40, "\n")
    
    # 3. 단일 출력 LSTM 모델 학습
    print("3단계: 단일 출력 LSTM 모델 학습")
    trained_single_model = None
    try:
        SEQ_LENGTH = 2
        PREDICTION_HORIZON = 1
        
        X_train, y_train = create_time_series_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
        X_val, y_val = create_time_series_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)
        
        if X_train.shape[0] > 0:
            input_size = X_train.shape[2]
            single_model = create_lstm_model(input_size=input_size)
            trained_single_model, _, _ = train_lstm_model(single_model, X_train, y_train, X_val, y_val, epochs=50)
            print("단일 출력 모델 학습 완료")
        else:
            print("학습 데이터가 부족하여 단일 출력 모델 학습을 건너뜁니다.")
    except Exception as e:
        print(f"단일 출력 모델 학습 중 오류: {e}")
    
    # 4. 다중 출력 LSTM 모델 학습
    print("\n4단계: 다중 출력 LSTM 모델 학습")
    trained_multi_model = None
    try:
        SEQ_LENGTH = 2
        PREDICTION_STEPS = 1

        X_train, y_vals_train, y_anom_train = create_multi_output_data(train_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
        X_val, y_vals_val, y_anom_val = create_multi_output_data(val_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)
        
        if X_train.shape[0] > 0:
            input_size = X_train.shape[2]
            num_features = y_vals_train.shape[2]
            multi_model = create_multi_output_lstm(input_size=input_size, prediction_steps=PREDICTION_STEPS, num_features=num_features)
            trained_multi_model = train_multi_output_model(multi_model, X_train, y_vals_train, y_anom_train,
                                                        X_val, y_vals_val, y_anom_val, epochs=30)
            print("다중 출력 모델 학습 완료")
        else:
            print("학습 데이터가 부족하여 다중 출력 모델 학습을 건너뜁니다.")
    except Exception as e:
        print(f"다중 출력 모델 학습 중 오류: {e}")
    
    # 5. 모델 및 전처리 정보 저장
    print("\n5단계: 모델 및 전처리 정보 저장")
    save_trained_models(trained_single_model, trained_multi_model, scaler, feature_cols, test_df)
    
    print("모델 훈련 및 저장이 완료")


def predict_future_anomalies(model, X_test, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    anomaly_probs = predictions.reshape(-1)
    anomaly_labels = (anomaly_probs > threshold).astype(int)
    
    return anomaly_probs, anomaly_labels

def generate_alerts(anomaly_probs, lot_numbers=None,
                alert_threshold=0.7, warning_threshold=0.5):
    alerts = []
    
    for i, prob in enumerate(anomaly_probs):
        lot_no = lot_numbers[i] if lot_numbers else f"lot_{i:04d}"
        
        if prob >= alert_threshold:
            alert_level = "CRITICAL"
            message = f"위험: LOT {lot_no} - 24시간 내 이상 발생 확률 {prob:.1%}"
            action = "즉시 점검 필요"
        elif prob >= warning_threshold:
            alert_level = "WARNING"
            message = f"경고: LOT {lot_no} - 이상 징후 감지 (확률 {prob:.1%})"
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

def calculate_remaining_useful_life(model, current_data_seq, max_horizon=100, 
                                failure_threshold=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    sample_tensor = torch.tensor(current_data_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for horizon in range(1, max_horizon + 1):
            prob = model(sample_tensor).cpu().item()

            degradation_factor = 1 + (horizon * 0.015) 
            adjusted_prob = min(prob * degradation_factor, 1.0)
            
            if adjusted_prob >= failure_threshold:
                return horizon
    
    return max_horizon

def create_mock_real_time_stream(test_df, feature_cols, stream_length=10):
    print(f"가상 실시간 데이터 스트림 생성 (길이: {stream_length})")
    
    stream_data = []
    for i in range(min(stream_length, len(test_df))):
        timestamp = f"2025-08-24 {10+i//60:02d}:{i%60:02d}:00"
        data_point = test_df.iloc[i:i+1]  # 한 행씩
        stream_data.append((timestamp, data_point))
    
    return stream_data

def real_time_monitoring(model, scaler, feature_cols, new_data_stream, 
                        sequence_length=10, update_interval=1):
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

            # 경고 판단 및 반환
            if anomaly_prob >= 0.7:
                print(f"[{timestamp}] 위험 경고: 이상 확률 {anomaly_prob:.1%}")
                return '2'
            elif anomaly_prob >= 0.3:
                print(f"[{timestamp}] 주의: 이상 징후 감지 (확률 {anomaly_prob:.1%})")
                return '1'
            else:
                print(f"[{timestamp}] 안전: 이상 징후 발생 가능성 낮음 (확률 {anomaly_prob:.1%})")
                return '0'
    
    return '0'  # 기본값

def run_single_output_simulation(model, scaler, feature_cols, test_df):
    """단일 출력 모델 시뮬레이션"""
    print("1. 단일 출력 이상 징후 예측 시뮬레이션")
    print("-" * 50)

    SEQ_LENGTH = 2
    PREDICTION_HORIZON = 1
    
    X_test, y_test = create_time_series_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_HORIZON)

    if X_test.shape[0] == 0:
        print("테스트 데이터가 부족하여 시뮬레이션을 건너뜁니다.")
        return None

    print("테스트 데이터 예측 및 경고 생성")
    probs, labels = predict_future_anomalies(model, X_test)

    test_indices = test_df.index[SEQ_LENGTH : len(probs) + SEQ_LENGTH]
    lot_numbers = test_df.loc[test_indices, 'lot_no'].tolist() if 'lot_no' in test_df else None

    alerts = generate_alerts(probs, lot_numbers=lot_numbers)
    
    if alerts:
        print(f"총 {len(alerts)}개의 경고가 생성되었습니다.")
        for alert in alerts[:5]:
            print(f"  - {alert['message']}")
    else:
        print("생성된 경고가 없습니다.")

    print("\nRUL 예측 예시 (테스트 데이터 첫 번째 샘플)")
    first_test_sequence = X_test[0]
    predicted_rul = calculate_remaining_useful_life(model, first_test_sequence)
    print(f"첫 번째 테스트 샘플의 예측 RUL: {predicted_rul} 스텝")
    
    return probs

def run_multi_output_simulation(model, feature_cols, test_df):

    print("\n2. 다중 출력 동시 예측 시뮬레이션")
    print("-" * 50)

    SEQ_LENGTH = 2
    PREDICTION_STEPS = 1

    X_test, y_vals_test, y_anom_test = create_multi_output_data(test_df, feature_cols, SEQ_LENGTH, PREDICTION_STEPS)

    if X_test.shape[0] == 0:
        print("테스트 데이터가 부족하여 시뮬레이션을 건너뜁니다.")
        return None

    print("다중 출력 모델 예측 예시 (테스트 데이터 첫 번째 샘플)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model.eval()
    sample_x = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0).to(device)
    pred_vals, pred_anoms = model(sample_x)
    
    print(f"입력 데이터 형태: {sample_x.shape}")
    print(f"예측 센서값 형태: {pred_vals.shape}")
    print(f"예측 이상확률 형태: {pred_anoms.shape}")
    print(f"\n예측된 미래 이상 발생 확률 ({PREDICTION_STEPS} 스텝):")
    pred_value = pred_anoms.detach().cpu().numpy().flatten()
    print(pred_value)
    
    return pred_value

def run_real_time_monitoring_simulation(model, scaler, feature_cols, test_df):
    print("\n3. 실시간 모니터링 시뮬레이션")
    print("-" * 50)

    mock_data_stream = create_mock_real_time_stream(test_df, feature_cols, stream_length=20)
    
    print("실시간 모니터링을 시작합니다...")    
    anomaly_status = real_time_monitoring(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        new_data_stream=mock_data_stream,
        sequence_length=2,
        update_interval=1
    )
    
    print("\n실시간 모니터링 시뮬레이션이 완료")
    return anomaly_status