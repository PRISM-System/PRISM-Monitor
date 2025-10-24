"""
데이터 정규화 유틸리티

새로운 데이터 형식을 기존 코드가 기대하는 형식으로 변환합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


# 🆕 NEW VERSION: CSV 파일별 매핑 (총 20개)
# 각 CSV 파일이 독립적인 모델을 가짐

FILE_TO_TABLE_MAPPING = {
    # Semiconductor (4개)
    'semiconductor_cmp_001.csv': 'semiconductor_cmp_001',
    'semiconductor_etch_002.csv': 'semiconductor_etch_002',
    'semiconductor_deposition_003.csv': 'semiconductor_deposition_003',
    'semiconductor_full_004.csv': 'semiconductor_full_004',

    # Automotive (4개)
    'automotive_welding_001.csv': 'automotive_welding_001',
    'automotive_painting_002.csv': 'automotive_painting_002',
    'automotive_press_003.csv': 'automotive_press_003',
    'automotive_assembly_004.csv': 'automotive_assembly_004',

    # Battery (4개)
    'battery_formation_001.csv': 'battery_formation_001',
    'battery_coating_002.csv': 'battery_coating_002',
    'battery_aging_003.csv': 'battery_aging_003',
    'battery_production_004.csv': 'battery_production_004',

    # Chemical (4개)
    'chemical_reactor_001.csv': 'chemical_reactor_001',
    'chemical_distillation_002.csv': 'chemical_distillation_002',
    'chemical_refining_003.csv': 'chemical_refining_003',
    'chemical_full_004.csv': 'chemical_full_004',

    # Steel (4개)
    'steel_rolling_001.csv': 'steel_rolling_001',
    'steel_converter_002.csv': 'steel_converter_002',
    'steel_casting_003.csv': 'steel_casting_003',
    'steel_production_004.csv': 'steel_production_004',
}

# 테이블명 → 파일명 매핑 (역방향)
TABLE_TO_FILE_MAPPING = {v: k for k, v in FILE_TO_TABLE_MAPPING.items()}

# 파일명 → 전체 경로 매핑
FILE_TO_PATH_MAPPING = {
    # Semiconductor
    'semiconductor_cmp_001.csv': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv',
    'semiconductor_etch_002.csv': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_etch_002.csv',
    'semiconductor_deposition_003.csv': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_deposition_003.csv',
    'semiconductor_full_004.csv': 'prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_full_004.csv',

    # Automotive
    'automotive_welding_001.csv': 'prism_monitor/test-scenarios/test_data/automotive/automotive_welding_001.csv',
    'automotive_painting_002.csv': 'prism_monitor/test-scenarios/test_data/automotive/automotive_painting_002.csv',
    'automotive_press_003.csv': 'prism_monitor/test-scenarios/test_data/automotive/automotive_press_003.csv',
    'automotive_assembly_004.csv': 'prism_monitor/test-scenarios/test_data/automotive/automotive_assembly_004.csv',

    # Battery
    'battery_formation_001.csv': 'prism_monitor/test-scenarios/test_data/battery/battery_formation_001.csv',
    'battery_coating_002.csv': 'prism_monitor/test-scenarios/test_data/battery/battery_coating_002.csv',
    'battery_aging_003.csv': 'prism_monitor/test-scenarios/test_data/battery/battery_aging_003.csv',
    'battery_production_004.csv': 'prism_monitor/test-scenarios/test_data/battery/battery_production_004.csv',

    # Chemical
    'chemical_reactor_001.csv': 'prism_monitor/test-scenarios/test_data/chemical/chemical_reactor_001.csv',
    'chemical_distillation_002.csv': 'prism_monitor/test-scenarios/test_data/chemical/chemical_distillation_002.csv',
    'chemical_refining_003.csv': 'prism_monitor/test-scenarios/test_data/chemical/chemical_refining_003.csv',
    'chemical_full_004.csv': 'prism_monitor/test-scenarios/test_data/chemical/chemical_full_004.csv',

    # Steel
    'steel_rolling_001.csv': 'prism_monitor/test-scenarios/test_data/steel/steel_rolling_001.csv',
    'steel_converter_002.csv': 'prism_monitor/test-scenarios/test_data/steel/steel_converter_002.csv',
    'steel_casting_003.csv': 'prism_monitor/test-scenarios/test_data/steel/steel_casting_003.csv',
    'steel_production_004.csv': 'prism_monitor/test-scenarios/test_data/steel/steel_production_004.csv',
}


def map_file_to_table_name(filename: str) -> Optional[str]:
    """
    파일명을 테이블명으로 매핑

    Args:
        filename: 예) 'semiconductor_cmp_001.csv'

    Returns:
        테이블명: 예) 'semi_cmp_sensors'
    """
    return FILE_TO_TABLE_MAPPING.get(filename)


def get_table_to_file_mapping() -> Dict[str, str]:
    """테이블명 → 파일명 매핑 반환"""
    return TABLE_TO_FILE_MAPPING.copy()


def normalize_semiconductor_data(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    새 데이터 형식을 기존 형식으로 정규화

    변환 작업:
    1. 컬럼명 대문자 → 소문자 변환
    2. SENSOR_ID/CHAMBER_ID/EQUIPMENT_ID → equipment_id 통일
    3. lot_no 생성 (SENSOR_ID 기반)
    4. pno 생성 (인덱스 기반)
    5. timestamp 컬럼명 통일

    Args:
        df: 원본 DataFrame
        source_file: 소스 파일명 (예: 'semiconductor_cmp_001.csv')

    Returns:
        정규화된 DataFrame
    """
    df_normalized = df.copy()

    # 1. 컬럼명 소문자 변환
    df_normalized.columns = df_normalized.columns.str.lower()

    # 2. equipment_id 생성 (우선순위: sensor_id > chamber_id > equipment_id)
    if 'equipment_id' not in df_normalized.columns:
        if 'sensor_id' in df_normalized.columns:
            df_normalized['equipment_id'] = df_normalized['sensor_id']
        elif 'chamber_id' in df_normalized.columns:
            df_normalized['equipment_id'] = df_normalized['chamber_id']
        else:
            # equipment_id가 없으면 더미값 생성
            df_normalized['equipment_id'] = 'UNKNOWN_EQUIP'

    # 3. lot_no 생성 (SENSOR_ID 기반 더미값)
    if 'lot_no' not in df_normalized.columns:
        table_name = map_file_to_table_name(source_file)
        scenario_prefix = table_name.upper() if table_name else 'SCENARIO'

        if 'equipment_id' in df_normalized.columns:
            df_normalized['lot_no'] = scenario_prefix + '_LOT_' + df_normalized['equipment_id'].astype(str)
        else:
            df_normalized['lot_no'] = scenario_prefix + '_LOT_UNKNOWN'

    # 4. pno 생성 (인덱스 기반)
    if 'pno' not in df_normalized.columns:
        df_normalized['pno'] = 'TS' + df_normalized.index.astype(str).str.zfill(8)

    # 5. timestamp 처리는 호출하는 쪽에서 수행 (여기서는 컬럼명만 통일)
    # timestamp 컬럼이 이미 소문자로 변환되어 있음

    return df_normalized


def validate_normalized_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    정규화된 데이터가 필수 컬럼을 가지고 있는지 검증

    Args:
        df: 정규화된 DataFrame

    Returns:
        검증 결과 딕셔너리
    """
    required_columns = ['timestamp', 'equipment_id', 'lot_no', 'pno']

    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'has_data': len(df) > 0,
    }

    for col in required_columns:
        if col not in df.columns:
            validation_result['is_valid'] = False
            validation_result['missing_columns'].append(col)

    return validation_result


def denormalize_for_display(df: pd.DataFrame, target_format: str = 'new') -> pd.DataFrame:
    """
    정규화된 데이터를 표시용 형식으로 역변환 (선택적)

    Args:
        df: 정규화된 DataFrame
        target_format: 'new' (신규 형식) 또는 'old' (기존 형식)

    Returns:
        변환된 DataFrame
    """
    if target_format == 'new':
        df_display = df.copy()

        # 컬럼명 대문자 변환
        df_display.columns = df_display.columns.str.upper()

        # TIMESTAMP → TIMESTAMP (이미 대문자)
        # equipment_id → SENSOR_ID로 복원
        if 'EQUIPMENT_ID' in df_display.columns:
            df_display['SENSOR_ID'] = df_display['EQUIPMENT_ID']

        return df_display

    return df  # 'old' 형식은 그대로 반환


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd

    # 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'TIMESTAMP': ['2025-05-01T00:00:00Z', '2025-05-01T00:00:10Z'],
        'SENSOR_ID': ['CMP_001', 'CMP_002'],
        'MOTOR_CURRENT': [16.5, 17.2],
        'SLURRY_FLOW_RATE': [250, 260],
        'PRESSURE': [3.0, 3.1]
    })

    print("=== 원본 데이터 ===")
    print(sample_data.head())
    print(f"컬럼: {list(sample_data.columns)}")

    # 정규화 수행
    normalized = normalize_semiconductor_data(sample_data, 'semiconductor_cmp_001.csv')

    print("\n=== 정규화된 데이터 ===")
    print(normalized.head())
    print(f"컬럼: {list(normalized.columns)}")

    # 검증
    validation = validate_normalized_data(normalized)
    print(f"\n=== 검증 결과 ===")
    print(f"유효성: {validation['is_valid']}")
    print(f"누락 컬럼: {validation['missing_columns']}")
    print(f"데이터 존재: {validation['has_data']}")
