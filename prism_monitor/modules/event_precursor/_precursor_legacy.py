# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_and_preprocess_data(csv_file):
    # 데이터 로드
    photo_sensors_data = pd.read_csv(csv_file)
    
    # 타임스탬프를 datetime 형식으로 변환하고 인덱스로 설정
    photo_sensors_data['TIMESTAMP'] = pd.to_datetime(photo_sensors_data['TIMESTAMP'])
    photo_sensors_data = photo_sensors_data.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    
    # PNO, EQUIPMENT_ID 등 비수치 데이터 제외
    features = photo_sensors_data.select_dtypes(include=np.number)
    feature_names = features.columns.tolist()
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, feature_names, scaler

def create_sequences(data, feature_names, target_column, time_steps=5):
    X, y = [], []
    target_idx = feature_names.index(target_column)
    
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, target_idx])
    
    return np.array(X), np.array(y)

def prepare_training_data(X, y, test_size=0.2):
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # PyTorch Tensor로 변환
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().view(-1, 1)
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, epochs=30, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("--- 모델 훈련 시작 ---")
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    print("--- 모델 훈련 종료 ---\n")
    return model, criterion, optimizer

def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'테스트 데이터셋 손실 (MSE): {test_loss.item():.4f}')
    
    return test_loss.item()

def predict_anomaly(model, scaled_features, feature_names, target_column, scaler, time_steps=5, anomaly_threshold=1.5):
    print("--- 미래 이상 징후 예측 시작 ---")
    
    # 현재 상태 (마지막 time_steps개 데이터)
    current_state_scaled = scaled_features[-time_steps:]
    current_state_tensor = torch.from_numpy(current_state_scaled).float().unsqueeze(0)
    
    # 예측 수행
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(current_state_tensor)
    
    # 예측값을 원래 스케일로 변환
    dummy_array = np.zeros((1, len(feature_names)))
    target_idx = feature_names.index(target_column)
    dummy_array[0, target_idx] = predicted_scaled.item()
    predicted_original = scaler.inverse_transform(dummy_array)[0, target_idx]
    
    # 결과 출력
    print(f"\n[입력] 현재 상태 (마지막 {time_steps}개 데이터)를 기반으로 예측")
    print(f"[예측] 다음 시점의 '{target_column}' 값: {predicted_original:.4f}")
    
    is_anomaly = predicted_original > anomaly_threshold
    if is_anomaly:
        print(f"[판단] 🚨 경고: 예측값이 임계치({anomaly_threshold})를 초과하여 '이상 징후'가 예상됩니다.")
    else:
        print(f"[판단] ✅ 정상: 예측값이 임계치({anomaly_threshold}) 이하로 안정적인 상태가 예상됩니다.")
    
    return predicted_original, is_anomaly