
# 정상 범위 기반 이상치 탐지 함수
# SEMI_PHOTO_SENSORS 데이터 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 🆕 API 연동을 위한 추가 import
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any

# ===============================================
# 🆕 API 모델 정의 (클래스 분리)
# ===============================================

class AnomalyDetail(BaseModel):
    row_index: int = Field(description="이상치가 발견된 행 인덱스")
    pno: str = Field(description="측정 고유번호")
    equipment_id: str = Field(description="장비 ID")
    lot_no: str = Field(description="LOT 번호")
    wafer_id: str = Field(description="웨이퍼 ID")
    timestamp: str = Field(description="측정 시간")
    anomaly_count: int = Field(description="해당 행의 이상 파라미터 개수")
    anomalous_parameters: List[str] = Field(description="이상이 발견된 파라미터 목록")
    
    # 모든 센서 측정값들
    exposure_dose: float = Field(description="노광 에너지")
    focus_position: float = Field(description="포커스 위치")
    stage_temp: float = Field(description="스테이지 온도")
    barometric_pressure: float = Field(description="대기압")
    humidity: float = Field(description="습도")
    alignment_error_x: float = Field(description="X축 정렬 오차")
    alignment_error_y: float = Field(description="Y축 정렬 오차")
    lens_aberration: float = Field(description="렌즈 수차")
    illumination_uniformity: float = Field(description="조명 균일도")
    reticle_temp: float = Field(description="레티클 온도")

class AnomalySummary(BaseModel):
    total_measurements: int = Field(description="총 측정 건수")
    total_anomalies: int = Field(description="총 이상치 건수")
    anomaly_rate: float = Field(description="이상치 비율 (%)")
    affected_equipment: List[str] = Field(description="영향받은 장비 목록")
    affected_parameters: List[str] = Field(description="이상이 발견된 파라미터 목록")
    analysis_period: Dict[str, str] = Field(description="분석 기간")

class EventOutputResult(BaseModel):
    status: Literal["complete", "failed"] = "complete"
    anomalyDetected: bool = True
    description: str = "SEMI_PHOTO_SENSORS 이상치 탐지 완료"
    summary: AnomalySummary
    anomaly_details: List[AnomalyDetail]

class EventOutputRequest(BaseModel):
    result: EventOutputResult

# ===============================================
# 🆕 API 변환 함수
# ===============================================

def convert_to_enhanced_api_format(anomaly_details, summary, total_measurements, analysis_start=None, analysis_end=None):
    """
    이상치 탐지 결과를 향상된 API 형태로 변환
    
    Parameters:
    - anomaly_details: 이상치 상세 정보 리스트
    - summary: 요약 정보
    - total_measurements: 총 측정 건수
    - analysis_start: 분석 시작 시간
    - analysis_end: 분석 종료 시간
    
    Returns:
    - EventOutputRequest: 상세 정보가 포함된 API 형태
    """
    
    # 이상치가 있는지 확인
    has_anomalies = len(anomaly_details) > 0
    
    # 영향받은 장비와 파라미터 수집
    affected_equipment = set()
    affected_parameters = set()
    
    # API 형태의 이상치 상세 정보 생성
    api_anomaly_details = []
    
    for anomaly in anomaly_details:
        affected_equipment.add(anomaly['equipment_id'])
        affected_parameters.update(anomaly['anomalous_parameters'])
        
        # 각 이상치를 API AnomalyDetail 형태로 변환
        full_data = anomaly['full_row_data']
        
        api_detail = AnomalyDetail(
            row_index=anomaly['row_index'],
            pno=anomaly['pno'],
            equipment_id=anomaly['equipment_id'],
            lot_no=anomaly['lot_no'],
            wafer_id=anomaly['wafer_id'],
            timestamp=str(anomaly['timestamp']),
            anomaly_count=anomaly['anomaly_count'],
            anomalous_parameters=anomaly['anomalous_parameters'],
            
            # 모든 센서 값들
            exposure_dose=full_data.get('EXPOSURE_DOSE', 0),
            focus_position=full_data.get('FOCUS_POSITION', 0),
            stage_temp=full_data.get('STAGE_TEMP', 0),
            barometric_pressure=full_data.get('BAROMETRIC_PRESSURE', 0),
            humidity=full_data.get('HUMIDITY', 0),
            alignment_error_x=full_data.get('ALIGNMENT_ERROR_X', 0),
            alignment_error_y=full_data.get('ALIGNMENT_ERROR_Y', 0),
            lens_aberration=full_data.get('LENS_ABERRATION', 0),
            illumination_uniformity=full_data.get('ILLUMINATION_UNIFORMITY', 0),
            reticle_temp=full_data.get('RETICLE_TEMP', 0)
        )
        api_anomaly_details.append(api_detail)
    
    # 요약 정보 생성
    anomaly_rate = (len(anomaly_details) / total_measurements * 100) if total_measurements > 0 else 0
    
    api_summary = AnomalySummary(
        total_measurements=total_measurements,
        total_anomalies=len(anomaly_details),
        anomaly_rate=round(anomaly_rate, 2),
        affected_equipment=list(affected_equipment),
        affected_parameters=list(affected_parameters),
        analysis_period={
            "start": analysis_start or "전체 기간",
            "end": analysis_end or "전체 기간"
        }
    )
    
    # 전체 설명 생성
    if has_anomalies:
        description = f"총 {len(anomaly_details)}건의 이상치가 탐지되었습니다. "
        description += f"이상률: {anomaly_rate:.2f}%. "
        description += f"영향받은 장비: {', '.join(list(affected_equipment))}. "
        description += f"주요 이상 파라미터: {', '.join(list(affected_parameters)[:3])}"
        
        if len(affected_parameters) > 3:
            description += f" 외 {len(affected_parameters)-3}개"
        
        status = "complete"
        anomaly_detected = True
    else:
        description = "분석 기간 내 모든 측정값이 정상 범위 내에 있습니다."
        status = "complete"
        anomaly_detected = False
    
    # 최종 API 요청 생성
    api_result = EventOutputResult(
        status=status,
        anomalyDetected=anomaly_detected,
        description=description,
        summary=api_summary,
        anomaly_details=api_anomaly_details
    )
    
    api_request = EventOutputRequest(result=api_result)
    
    return api_request

# 🆕 API 연동을 위한 추가 import
from pydantic import BaseModel, Field
from typing import Literal

# 🆕 API 모델 정의
class EventOutputRequest(BaseModel):
    class Result(BaseModel):
        status: Literal["complete", "failed"] = "complete"
        anomalyDetected: bool = True
        description: str = "SEMI_PHOTO_SENSORS 이상치 탐지"
    result: Result = Result()

class RangeBasedAnomalyDetector:
    """정상 범위 기반 이상치 탐지 클래스"""
    
    def __init__(self, normal_ranges=None):
        """
        초기화
        
        Parameters:
        - normal_ranges: 정상 범위 딕셔너리 (선택사항)
        """
        self.normal_ranges = normal_ranges or {
            'EXPOSURE_DOSE': (20, 40),
            'FOCUS_POSITION': (-50, 50),
            'STAGE_TEMP': (22.9, 23.1),
            'HUMIDITY': (40, 50),
            'ALIGNMENT_ERROR_X': (0, 3),
            'ALIGNMENT_ERROR_Y': (0, 3),
            'LENS_ABERRATION': (0, 5),
            'ILLUMINATION_UNIFORMITY': (98, 100),
            'RETICLE_TEMP': (22.95, 23.05)
        }
    
    def detect_anomalies(self, df):
        """
        정상 범위를 벗어나는 이상치를 탐지
        
        Parameters:
        - df: 데이터프레임
        
        Returns:
        - anomaly_details: 이상치가 발견된 모든 행의 상세 정보
        - summary: 요약 정보
        """
        anomaly_details = []
        summary = {}
        
        # 각 행을 검사
        for idx, row in df.iterrows():
            row_anomalies = []
            
            # 각 파라미터에 대해 정상 범위 체크
            for param, (min_val, max_val) in self.normal_ranges.items():
                if param in row:
                    value = row[param]
                    if pd.notna(value) and (value < min_val or value > max_val):
                        row_anomalies.append({
                            'parameter': param,
                            'value': value,
                            'normal_min': min_val,
                            'normal_max': max_val,
                            'deviation': min(abs(value - min_val), abs(value - max_val))
                        })
            
            # 이상치가 발견된 행이면 상세 정보 저장
            if row_anomalies:
                anomaly_info = {
                    'row_index': idx,
                    'pno': row.get('PNO', 'N/A'),
                    'equipment_id': row.get('EQUIPMENT_ID', 'N/A'),
                    'lot_no': row.get('LOT_NO', 'N/A'),
                    'wafer_id': row.get('WAFER_ID', 'N/A'),
                    'timestamp': row.get('TIMESTAMP', 'N/A'),
                    'anomalous_parameters': [item['parameter'] for item in row_anomalies],
                    'anomaly_count': len(row_anomalies),
                    'anomaly_details': row_anomalies,
                    'full_row_data': row.to_dict()
                }
                anomaly_details.append(anomaly_info)
        
        # 요약 정보 생성
        for param in self.normal_ranges.keys():
            param_anomalies = [detail for detail in anomaly_details 
                              if param in detail['anomalous_parameters']]
            summary[param] = {
                'anomaly_count': len(param_anomalies),
                'percentage': (len(param_anomalies) / len(df)) * 100 if len(df) > 0 else 0
            }
        
        return anomaly_details, summary
    
    def analyze_by_equipment(self, df, anomaly_details):
        """장비별 이상치 분석"""
        equipment_analysis = {}
        
        for anomaly in anomaly_details:
            equipment = anomaly['equipment_id']
            if equipment not in equipment_analysis:
                equipment_analysis[equipment] = {
                    'total_anomalies': 0,
                    'anomalous_measurements': 0,
                    'parameters': {}
                }
            
            equipment_analysis[equipment]['anomalous_measurements'] += 1
            equipment_analysis[equipment]['total_anomalies'] += anomaly['anomaly_count']
            
            for param in anomaly['anomalous_parameters']:
                if param not in equipment_analysis[equipment]['parameters']:
                    equipment_analysis[equipment]['parameters'][param] = 0
                equipment_analysis[equipment]['parameters'][param] += 1
        
        # 총 측정 수 추가
        for equipment in equipment_analysis.keys():
            if 'EQUIPMENT_ID' in df.columns:
                total_measurements = len(df[df['EQUIPMENT_ID'] == equipment])
                equipment_analysis[equipment]['total_measurements'] = total_measurements
                equipment_analysis[equipment]['anomaly_rate'] = (
                    equipment_analysis[equipment]['anomalous_measurements'] / total_measurements * 100
                )
        
        return equipment_analysis
    
    def save_results(self, anomaly_details, filename='anomaly_results.csv'):
        """이상치 탐지 결과를 CSV 파일로 저장"""
        if not anomaly_details:
            print("저장할 이상치 데이터가 없습니다.")
            return False
        
        rows = []
        for anomaly in anomaly_details:
            base_info = {
                'row_index': anomaly['row_index'],
                'pno': anomaly['pno'],
                'equipment_id': anomaly['equipment_id'],
                'lot_no': anomaly['lot_no'],
                'wafer_id': anomaly['wafer_id'],
                'timestamp': anomaly['timestamp'],
                'anomaly_count': anomaly['anomaly_count'],
                'anomalous_parameters': ', '.join(anomaly['anomalous_parameters'])
            }
            base_info.update(anomaly['full_row_data'])
            rows.append(base_info)
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"이상치 결과 저장됨: {filename}")
        return True
    
    def visualize_anomalies(self, summary, equipment_analysis=None, save_plot=True):
        """이상치 시각화"""
        param_counts = {param: info['anomaly_count'] for param, info in summary.items() 
                       if info['anomaly_count'] > 0}
        
        if not param_counts:
            print("시각화할 이상치가 없습니다.")
            return
        
        if equipment_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 파라미터별 이상치 개수
        ax1.bar(param_counts.keys(), param_counts.values())
        ax1.set_title('Anomalies by Parameter')
        ax1.set_ylabel('Number of Anomalies')
        ax1.tick_params(axis='x', rotation=45)
        
        # 장비별 이상치 개수
        if equipment_analysis:
            equipment_counts = {eq: analysis['anomalous_measurements'] 
                              for eq, analysis in equipment_analysis.items()}
            if equipment_counts:
                ax2.bar(equipment_counts.keys(), equipment_counts.values())
                ax2.set_title('Anomalous Measurements by Equipment')
                ax2.set_ylabel('Number of Anomalous Measurements')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('range_based_anomalies.png', dpi=300, bbox_inches='tight')
            print("시각화 저장됨: range_based_anomalies.png")
        
        plt.show()


# ===============================================
# 🆕 API 변환 함수 (새로 추가)
# ===============================================

def convert_to_api_format(anomaly_details, summary):
    """
    기존 이상치 탐지 결과를 API 형태로 변환
    
    Parameters:
    - anomaly_details: detect_anomalies 함수의 결과
    - summary: detect_anomalies 함수의 요약 정보
    
    Returns:
    - EventOutputRequest: API에서 사용할 수 있는 형태
    """
    
    # 이상치가 있는지 확인
    has_anomalies = len(anomaly_details) > 0
    
    if has_anomalies:
        # 이상치가 있을 때
        total_anomalies = len(anomaly_details)
        
        # 어떤 파라미터에서 이상치가 발생했는지 찾기
        affected_params = set()
        affected_equipment = set()
        
        for anomaly in anomaly_details:
            affected_params.update(anomaly['anomalous_parameters'])
            affected_equipment.add(anomaly['equipment_id'])
        
        # 설명 문구 만들기
        description = f"총 {total_anomalies}건의 이상치 탐지. "
        description += f"영향받은 파라미터: {', '.join(list(affected_params)[:3])}"
        
        if len(affected_params) > 3:
            description += f" 외 {len(affected_params)-3}개"
        
        description += f". 영향받은 장비: {', '.join(list(affected_equipment))}"
        
        status = "complete"
        anomaly_detected = True
        
    else:
        # 이상치가 없을 때
        description = "모든 측정값이 정상 범위 내에 있습니다"
        status = "complete"
        anomaly_detected = False
    
    # API 형태로 변환
    api_request = EventOutputRequest(
        result=EventOutputRequest.Result(
            status=status,
            anomalyDetected=anomaly_detected,
            description=description
        )
    )
    
    return api_request


def detect_range_based_anomalies(file_path, 
                                 normal_ranges=None, 
                                 start_time=None, 
                                 end_time=None,
                                 verbose=True,
                                 save_results=False,
                                 visualize=False,
                                 output_filename=None):
    """
    정상 범위 기반 이상치 탐지 메인 함수
    
    Parameters:
    - file_path: CSV 파일 경로
    - normal_ranges: 정상 범위 딕셔너리 (선택사항)
    - start_time: 시작 시간 (선택사항)
    - end_time: 종료 시간 (선택사항)  
    - verbose: 상세 출력 여부
    - save_results: 결과 저장 여부
    - visualize: 시각화 여부
    - output_filename: 출력 파일명
    
    Returns:
    - result: 분석 결과 딕셔너리
    """
    
    try:
        # 1. 데이터 로딩
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if verbose:
            print("=" * 80)
            print("정상 범위 기반 이상치 탐지 시스템")
            print("=" * 80)
            print(f"파일 로딩 성공: {file_path}")
            print(f"데이터 형태: {df.shape}")
        
        # TIMESTAMP 컬럼 처리
        if 'TIMESTAMP' in df.columns:
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            
            if verbose:
                print(f"데이터 기간: {df['TIMESTAMP'].min()} ~ {df['TIMESTAMP'].max()}")
        
        # 시간 필터링
        if start_time or end_time:
            if 'TIMESTAMP' not in df.columns:
                print("경고: TIMESTAMP 컬럼이 없어 시간 필터링을 건너뜁니다.")
            else:
                if start_time:
                    start_time = pd.to_datetime(start_time)
                    df = df[df['TIMESTAMP'] >= start_time]
                if end_time:
                    end_time = pd.to_datetime(end_time)
                    df = df[df['TIMESTAMP'] <= end_time]
                
                if len(df) == 0:
                    return {"message": "지정된 시간 구간에 데이터가 없습니다.", "anomalies": []}
                
                if verbose:
                    print(f"시간 필터링 후 데이터: {df.shape[0]}행")
        
        # 2. 이상치 탐지 실행
        detector = RangeBasedAnomalyDetector(normal_ranges)
        anomaly_details, summary = detector.detect_anomalies(df)
        
        # 3. 장비별 분석
        equipment_analysis = None
        if 'EQUIPMENT_ID' in df.columns:
            equipment_analysis = detector.analyze_by_equipment(df, anomaly_details)
        
        # 4. 결과 출력
        if verbose:
            print(f"\n총 {len(anomaly_details)}개 행에서 이상치 발견")
            print(f"전체 데이터의 {(len(anomaly_details)/len(df)*100):.2f}%")
            
            # 파라미터별 요약
            print("\n파라미터별 이상치 요약:")
            has_anomalies = False
            for param, info in summary.items():
                if info['anomaly_count'] > 0:
                    print(f"  {param}: {info['anomaly_count']}건 ({info['percentage']:.1f}%)")
                    has_anomalies = True
            
            if not has_anomalies:
                print("  모든 파라미터가 정상 범위 내에 있습니다.")
            
            # 상세 이상치 정보 (처음 5개만)
            if anomaly_details:
                print(f"\n상세 이상치 정보 (처음 5개):")
                for i, anomaly in enumerate(anomaly_details[:5], 1):
                    print(f"\n[이상치 {i}]")
                    print(f"  행 인덱스: {anomaly['row_index']}")
                    print(f"  PNO: {anomaly['pno']}")
                    print(f"  장비 ID: {anomaly['equipment_id']}")
                    print(f"  측정 시간: {anomaly['timestamp']}")
                    print(f"  이상 파라미터: {', '.join(anomaly['anomalous_parameters'])}")
                    
                    for detail in anomaly['anomaly_details']:
                        print(f"    - {detail['parameter']}: {detail['value']:.3f} "
                              f"(정상범위: {detail['normal_min']} ~ {detail['normal_max']})")
                
                if len(anomaly_details) > 5:
                    print(f"\n... 외 {len(anomaly_details) - 5}개 더")
            
            # 장비별 분석
            if equipment_analysis:
                print(f"\n장비별 이상치 분석:")
                for equipment, analysis in equipment_analysis.items():
                    print(f"  {equipment}:")
                    print(f"    총 측정: {analysis['total_measurements']}회")
                    print(f"    이상 측정: {analysis['anomalous_measurements']}회 "
                          f"({analysis['anomaly_rate']:.1f}%)")
                    
                    if analysis['parameters']:
                        top_param = max(analysis['parameters'], key=analysis['parameters'].get)
                        print(f"    주요 이상 파라미터: {top_param} ({analysis['parameters'][top_param]}회)")
        
        # 5. 결과 저장
        if save_results:
            filename = output_filename or 'range_based_anomaly_results.csv'
            detector.save_results(anomaly_details, filename)
        
        # 6. 시각화
        if visualize and anomaly_details:
            detector.visualize_anomalies(summary, equipment_analysis)
        
        # 7. 결과 반환
        result = {
            'total_rows': len(df),
            'anomaly_count': len(anomaly_details),
            'anomaly_rate': (len(anomaly_details) / len(df)) * 100 if len(df) > 0 else 0,
            'anomalies': anomaly_details,
            'summary': summary,
            'equipment_analysis': equipment_analysis,
            'normal_ranges': detector.normal_ranges
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print("분석 완료!")
            print(f"{'='*80}")
        
        return result
        
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        if verbose:
            print(error_msg)
        return {"error": error_msg, "anomalies": []}


def detect_anomalies_from_dataframe(df, 
                                   normal_ranges=None,
                                   start_time=None,
                                   end_time=None,
                                   verbose=False):
    """
    DataFrame에서 직접 이상치 탐지
    
    Parameters:
    - df: 데이터프레임
    - normal_ranges: 정상 범위 딕셔너리
    - start_time: 시작 시간
    - end_time: 종료 시간
    - verbose: 상세 출력 여부
    
    Returns:
    - anomaly_details: 이상치 목록
    - summary: 요약 정보
    """
    
    try:
        # 시간 필터링
        filtered_df = df.copy()
        if (start_time or end_time) and 'TIMESTAMP' in df.columns:
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            if start_time:
                start_time = pd.to_datetime(start_time)
                filtered_df = filtered_df[filtered_df['TIMESTAMP'] >= start_time]
            if end_time:
                end_time = pd.to_datetime(end_time)
                filtered_df = filtered_df[filtered_df['TIMESTAMP'] <= end_time]
        
        # 이상치 탐지
        detector = RangeBasedAnomalyDetector(normal_ranges)
        anomaly_details, summary = detector.detect_anomalies(filtered_df)
        
        if verbose:
            print(f"총 {len(anomaly_details)}개 이상치 탐지됨")
            for param, info in summary.items():
                if info['anomaly_count'] > 0:
                    print(f"  {param}: {info['anomaly_count']}건")
        
        return anomaly_details, summary
        
    except Exception as e:
        if verbose:
            print(f"오류 발생: {str(e)}")
        return [], {}


# ===================================================================
# 사용 예시 및 메인 실행부
# ===================================================================

def detect():
    # 기본 설정
    CSV_FILE_PATH = 'prism_monitor/data/Industrial_DB_sample/SEMI_PHOTO_SENSORS.csv'
    
    # 사용자 정의 정상 범위 (선택사항)
    custom_ranges = {
        'EXPOSURE_DOSE': (20, 40),
        'FOCUS_POSITION': (-50, 50),
        'STAGE_TEMP': (22.9, 23.1),
        'HUMIDITY': (40, 50),
        'ALIGNMENT_ERROR_X': (0, 3),
        'ALIGNMENT_ERROR_Y': (0, 3),
        'LENS_ABERRATION': (0, 5),
        'ILLUMINATION_UNIFORMITY': (98, 100),
        'RETICLE_TEMP': (22.95, 23.05)
    }
    
    # 메인 함수 실행
    print("정상 범위 기반 이상치 탐지를 실행합니다...")
    
    result = detect_range_based_anomalies(
        file_path=CSV_FILE_PATH,
        normal_ranges=custom_ranges,
        start_time=None,  # '2024-01-15 08:00:00'
        end_time=None,    # '2024-01-16 09:00:00'
        verbose=True,
        save_results=True,
        visualize=True,
        output_filename='detected_anomalies.csv'
    )
    api_result = convert_to_api_format(result['anomalies'], result['summary'])
    return api_result
