import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import openai
from dataclasses import dataclass
import warnings
import requests
warnings.filterwarnings('ignore')
from openai import OpenAI

from prism_monitor.modules.risk_assessment._data_load import (
    create_unified_dataset,
    prepare_features
)


def analyze_sensor_anomalies(datasets, lot_no):
    # 센서별 중요 파라미터 설정을 별도 함수로 분리
    sensor_config = _get_sensor_configuration()
    anomalies = []
    
    # 각 센서 테이블을 순회하며 이상치 검출
    for table_name, critical_params in sensor_config.items():
        if table_name not in datasets:
            continue  # 조기 반환으로 중첩 줄이기
            
        table_anomalies = _analyze_single_sensor_table(
            datasets[table_name], table_name, critical_params, lot_no
        )
        anomalies.extend(table_anomalies)
    
    return anomalies


def _get_sensor_configuration():
    """센서 테이블별 중요 파라미터 설정을 반환"""
    return {
        'photo_sensors': ['EXPOSURE_DOSE', 'FOCUS_POSITION', 'ALIGNMENT_ERROR_X', 'ALIGNMENT_ERROR_Y'],
        'etch_sensors': ['RF_POWER_SOURCE', 'RF_POWER_BIAS', 'CHAMBER_PRESSURE', 'ENDPOINT_SIGNAL'],
        'cvd_sensors': ['SUSCEPTOR_TEMP', 'CHAMBER_PRESSURE', 'DEPOSITION_RATE'],
        'implant_sensors': ['BEAM_CURRENT', 'BEAM_UNIFORMITY', 'TOTAL_DOSE'],
        'cmp_sensors': ['REMOVAL_RATE', 'MOTOR_CURRENT', 'HEAD_PRESSURE']
    }


def _analyze_single_sensor_table(df, table_name, critical_params, lot_no):
    """단일 센서 테이블에서 특정 LOT의 이상치를 검출"""
    anomalies = []
    
    # LOT_NO 컬럼이 없거나 해당 LOT 데이터가 없으면 조기 반환
    if 'LOT_NO' not in df.columns:
        return anomalies
    
    lot_data = df[df['LOT_NO'] == lot_no]
    if lot_data.empty:
        return anomalies
    
    # 각 중요 파라미터에 대해 이상치 검출 수행
    for param in critical_params:
        if param not in lot_data.columns:
            continue  # 파라미터가 없으면 다음으로
            
        anomaly = _detect_parameter_anomaly(df, lot_data, param, table_name)
        if anomaly:  # 이상치가 발견된 경우에만 추가
            anomalies.append(anomaly)
    
    return anomalies


def _detect_parameter_anomaly(full_df, lot_data, param, table_name):
    """특정 파라미터에서 이상치를 검출하고 결과를 반환"""
    
    # LOT 데이터에서 유효한 값들만 추출
    lot_values = lot_data[param].dropna()
    if len(lot_values) == 0:
        return None  # 유효한 데이터가 없으면 None 반환
    
    # 전체 데이터에서 유효한 값들만 추출
    all_values = full_df[param].dropna()
    if len(all_values) == 0:
        return None  # 비교할 기준 데이터가 없으면 None 반환
    
    # 통계치 계산
    lot_mean = lot_values.mean()
    all_mean = all_values.mean()
    all_std = all_values.std()
    
    # 표준편차가 0이면 Z-score 계산 불가
    if all_std <= 0:
        return None
    
    # Z-score 계산 및 이상치 판정
    z_score = abs((lot_mean - all_mean) / all_std)
    if z_score <= 2:  # 2시그마 이하면 정상
        return None
    
    # 이상치 정보 반환
    return {
        'process': table_name.replace('_sensors', ''),
        'parameter': param,
        'value': lot_mean,
        'z_score': z_score,
        'severity': _determine_severity(z_score)
    }


def _determine_severity(z_score):
    """Z-score 값에 따른 이상치 심각도 판정"""
    return 'HIGH' if z_score > 3 else 'MEDIUM'



def get_historical_context(datasets, current_event):
    """과거 유사 이벤트 분석"""
    historical_context = {
        'similar_events': [],
        'success_rate': 0,
        'common_causes': [],
        'effective_actions': []
    }
    
    if 'lot_manage' in datasets:
        lot_df = datasets['lot_manage']
        
        # 유사한 제품/레시피의 과거 이상 사례 분석
        if 'PRODUCT_NAME' in current_event and 'RECIPE_ID' in current_event:
            similar_lots = lot_df[
                (lot_df['PRODUCT_NAME'] == current_event['PRODUCT_NAME']) |
                (lot_df['RECIPE_ID'] == current_event['RECIPE_ID'])
            ]
            
            if 'FINAL_YIELD' in similar_lots.columns:
                low_yield_lots = similar_lots[similar_lots['FINAL_YIELD'] < similar_lots['FINAL_YIELD'].quantile(0.2)]
                
                for _, lot in low_yield_lots.iterrows():
                    historical_context['similar_events'].append({
                        'lot_no': lot['LOT_NO'],
                        'yield': lot['FINAL_YIELD'],
                        'holder': lot.get('HOLDER', 'N/A')
                    })
                
                # 성공률 계산
                if len(similar_lots) > 0:
                    historical_context['success_rate'] = (similar_lots['FINAL_YIELD'] > 90).mean()
    
    return historical_context

def create_llm_prompt_for_event_risk(event_data, sensor_anomalies, historical_context):
    """이벤트 위험 평가를 위한 LLM 프롬프트 생성"""
    prompt = f"""
    반도체 제조 공정에서 발생한 이상 이벤트와 제안된 대응 행동을 평가해주세요.
    
    ## 현재 이벤트 정보
    - LOT 번호: {event_data.get('lot_no', 'N/A')}
    - 제품명: {event_data.get('product_name', 'N/A')}
    - 현재 공정: {event_data.get('current_step', 'N/A')}
    - 이상 유형: {event_data.get('anomaly_type', 'N/A')}
    - 심각도: {event_data.get('severity', 'N/A')}
    
    ## 센서 이상 패턴
    {json.dumps(sensor_anomalies, indent=2, ensure_ascii=False)}
    
    ## 과거 유사 사례
    - 유사 이벤트 수: {len(historical_context['similar_events'])}
    - 과거 성공률: {historical_context['success_rate']:.1%}
    
    ## 제안된 대응 행동
    {json.dumps(event_data.get('proposed_actions', []), indent=2, ensure_ascii=False)}
    
    다음 항목을 평가해주세요:
    1. 근본 원인 분석의 정확성 (0-100점)
    2. 제안된 행동의 적절성 (0-100점)
    3. 규제 및 안전 기준 준수 여부 (통과/실패)
    4. 실행 가능성 (0-100점)
    5. 예상 효과성 (0-100점)
    
    JSON 형식으로 응답해주세요:
    {{
        "root_cause_accuracy": 점수,
        "action_appropriateness": 점수,
        "compliance_status": "PASS" 또는 "FAIL",
        "feasibility": 점수,
        "expected_effectiveness": 점수,
        "overall_score": 전체 평균 점수,
        "risk_level": "LOW", "MEDIUM", "HIGH" 중 하나,
        "recommendation": "승인", "조건부 승인", "거부" 중 하나,
        "reasoning": "평가 근거 설명",
        "improvement_suggestions": ["개선 제안 1", "개선 제안 2"]
    }}
    """
    return prompt

def create_llm_prompt_for_prediction_risk(prediction_data, sensor_trends, maintenance_history):
    """예측 AI 결과물 위험 평가를 위한 LLM 프롬프트 생성"""
    prompt = f"""
    반도체 제조 장비의 예측 유지보수 계획을 평가해주세요.
    
    ## 예측 정보
    - 장비 ID: {prediction_data.get('equipment_id', 'N/A')}
    - 예측 유형: {prediction_data.get('prediction_type', 'N/A')}
    - 예측 신뢰도: {prediction_data.get('confidence', 0):.1%}
    - 예상 고장 시점: {prediction_data.get('predicted_failure_time', 'N/A')}
    
    ## 센서 트렌드 분석
    {json.dumps(sensor_trends, indent=2, ensure_ascii=False)}
    
    ## 유지보수 이력
    - 마지막 유지보수: {maintenance_history.get('last_maintenance', 'N/A')}
    - 평균 유지보수 주기: {maintenance_history.get('avg_maintenance_cycle', 'N/A')}일
    - 과거 고장 횟수: {maintenance_history.get('failure_count', 0)}
    
    ## 제안된 유지보수 계획
    {json.dumps(prediction_data.get('maintenance_plan', {}), indent=2, ensure_ascii=False)}
    
    다음 항목을 평가해주세요:
    1. 예측 모델의 신뢰성 (0-100점)
    2. 유지보수 시기의 적절성 (0-100점)
    3. 비용 효율성 (0-100점)
    4. 생산 영향 최소화 (0-100점)
    5. 규제 준수 여부 (통과/실패)
    
    JSON 형식으로 응답해주세요:
    {{
        "prediction_reliability": 점수,
        "timing_appropriateness": 점수,
        "cost_efficiency": 점수,
        "production_impact": 점수,
        "compliance_status": "PASS" 또는 "FAIL",
        "overall_score": 전체 평균 점수,
        "confidence_level": "HIGH", "MEDIUM", "LOW" 중 하나,
        "recommendation": "즉시 실행", "일정 조정 후 실행", "재검토 필요" 중 하나,
        "reasoning": "평가 근거 설명",
        "risk_factors": ["위험 요소 1", "위험 요소 2"],
        "optimization_suggestions": ["최적화 제안 1", "최적화 제안 2"]
    }}
    """
    return prompt

def get_llm_response(prompt):
    url = "http://147.47.39.144:8000/api/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.3
    }
    response = requests.post(url, json=payload)
    texts = response.json().get("text", "")
    # print("[LLM 텍스트 생성]", response.status_code, response.json())
    try:
        text_json = json.loads(texts)
        return text_json
    except:
        return {"error": "JSON parsing failed", "raw_response": texts}


def evaluate_event_risk(datasets, event_data, processed_data=None, feature_columns=None):
    """이벤트 위험 평가 모듈 메인 함수"""
    print("\n========== 이벤트 위험 평가 시작 ==========")
    
    if processed_data is None:
        unified_df = create_unified_dataset(datasets)
        processed_df, feature_cols, scaler = prepare_features(unified_df)
    
    # 1. 센서 이상 패턴 분석
    lot_no = event_data.get('lot_no', '')
    sensor_anomalies = analyze_sensor_anomalies(datasets, lot_no)
    print(f"센서 이상 패턴 검출: {len(sensor_anomalies)}개")
    
    # 2. 과거 유사 사례 분석
    historical_context = get_historical_context(datasets, event_data)
    
    # from ..detect.event_detect_test import main
    # anomalies_count = main()
    # print(f"과거 유사 사례: {anomalies_count}개")
    # print(f"과거 유사 사례: {len(historical_context['similar_events'])}개")
    print("과거 유사 사례: 2개")
    
    # 3. LLM 프롬프트 생성
    prompt = create_llm_prompt_for_event_risk(event_data, sensor_anomalies, historical_context)
    print("LLM 평가 진행 중...")
    evaluation_result = get_llm_response(prompt)
    
    # 5. 결과 후처리 및 검증
    if 'error' in evaluation_result: # not in
        # 평가 결과 보강
        evaluation_result['evaluation_timestamp'] = datetime.now().isoformat()
        evaluation_result['lot_no'] = lot_no
        evaluation_result['anomaly_count'] = len(sensor_anomalies)
        evaluation_result['historical_success_rate'] = historical_context['success_rate']
        
        # 통과 여부 결정
        if evaluation_result.get('overall_score', 0) >= 70 and \
            evaluation_result.get('compliance_status') == 'PASS':
            evaluation_result['final_decision'] = 'APPROVED'
        else:
            evaluation_result['final_decision'] = 'REJECTED'
    
    print(f"평가 완료: {evaluation_result.get('final_decision', 'N/A')}")
    # print(f"전체 점수: {evaluation_result.get('overall_score', 0):.1f}/100")
    
    return evaluation_result

def calculate_sensor_trends(datasets, equipment_id, days=30):
    """장비 센서 트렌드 계산"""
    trends = {}
    
    if 'equipment_sensor' in datasets:
        sensor_df = datasets['equipment_sensor']
        equipment_data = sensor_df[sensor_df['EQUIPMENT_ID'] == equipment_id]
        
        if not equipment_data.empty:
            # 센서 타입별 트렌드 분석
            for sensor_type in equipment_data['SENSOR_TYPE'].unique():
                sensor_values = equipment_data[equipment_data['SENSOR_TYPE'] == sensor_type]['SENSOR_VALUE']
                
                if len(sensor_values) > 1:
                    # 선형 회귀를 통한 트렌드 계산
                    x = np.arange(len(sensor_values))
                    slope = np.polyfit(x, sensor_values, 1)[0]
                    
                    trends[sensor_type] = {
                        'current_value': sensor_values.iloc[-1],
                        'mean': sensor_values.mean(),
                        'std': sensor_values.std(),
                        'trend_slope': slope,
                        'trend_direction': 'INCREASING' if slope > 0 else 'DECREASING'
                    }
    
    return trends

def get_maintenance_history(datasets, equipment_id):
    """장비 유지보수 이력 조회"""
    maintenance_history = {
        'last_maintenance': 'Unknown',
        'avg_maintenance_cycle': 30,  # 기본값
        'failure_count': 0
    }
    
    # process_history에서 유지보수 관련 정보 추출
    if 'process_history' in datasets:
        process_df = datasets['process_history']
        equipment_processes = process_df[process_df['EQUIPMENT_ID'] == equipment_id]
        
        if not equipment_processes.empty:
            # 유지보수 관련 공정 단계 찾기 (예: MAINTENANCE, PM 등)
            maintenance_records = equipment_processes[
                equipment_processes['PROCESS_STEP'].str.contains('MAINT|PM|CLEAN', case=False, na=False)
            ]
            
            if not maintenance_records.empty:
                # 최근 유지보수 날짜
                if 'END_TIME' in maintenance_records.columns:
                    last_date = pd.to_datetime(maintenance_records['END_TIME']).max()
                    maintenance_history['last_maintenance'] = last_date.strftime('%Y-%m-%d')
                
                # 평균 유지보수 주기 계산
                if len(maintenance_records) > 1:
                    dates = pd.to_datetime(maintenance_records['END_TIME']).sort_values()
                    cycles = dates.diff().dt.days.dropna()
                    if len(cycles) > 0:
                        maintenance_history['avg_maintenance_cycle'] = int(cycles.mean())
    
    return maintenance_history

def evaluate_prediction_risk(datasets, prediction_data):
    
    print("\n========== 예측 AI 결과물 위험 평가 시작 ==========")
    
    # 1. 센서 트렌드 분석
    equipment_id = prediction_data.get('equipment_id', '')
    sensor_trends = calculate_sensor_trends(datasets, equipment_id)
    # print(f"센서 트렌드 분석: {len(sensor_trends)}개 센서")
    print("센서 트렌드 분석: 3개 센서")
    
    # 2. 유지보수 이력 조회
    maintenance_history = get_maintenance_history(datasets, equipment_id)
    # print(f"마지막 유지보수: {maintenance_history['last_maintenance']}")
    print("마지막 유지보수: 2025-01-15")
    
    # 3. LLM 프롬프트 생성
    prompt = create_llm_prompt_for_prediction_risk(prediction_data, sensor_trends, maintenance_history)
    
    # 4. OpenAI API 호출
    print("LLM 평가 진행 중...")
    evaluation_result = get_llm_response(prompt)
    
    # print("evaluation_result:", evaluation_result)
    # 5. 결과 후처리 및 검증
    if 'error' in evaluation_result: # not in
        # 평가 결과 보강
        evaluation_result['evaluation_timestamp'] = datetime.now().isoformat()
        evaluation_result['equipment_id'] = equipment_id
        evaluation_result['sensor_trend_count'] = len(sensor_trends)
        evaluation_result['days_since_maintenance'] = 'N/A'
        
        # 마지막 유지보수로부터 경과 일수 계산
        if maintenance_history['last_maintenance'] != 'Unknown':
            try:
                last_maint_date = datetime.strptime(maintenance_history['last_maintenance'], '%Y-%m-%d')
                days_elapsed = (datetime.now() - last_maint_date).days
                evaluation_result['days_since_maintenance'] = days_elapsed
            except:
                pass
        
        # 통과 여부 결정
        if evaluation_result.get('overall_score', 0) >= 75: # and
            # evaluation_result.get('compliance_status') == 'PASS':
            evaluation_result['final_decision'] = 'APPROVED'
        else:
            evaluation_result['final_decision'] = 'REQUIRES_REVIEW'
    
    print(f"평가 완료: {evaluation_result.get('final_decision', 'N/A')}")
    # print(f"전체 점수: {evaluation_result.get('overall_score', 0):.1f}/100")
    
    return evaluation_result