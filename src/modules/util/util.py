import pandas as pd

TEST_SCENARIOS_DATA_MAPPING = {
    # Semiconductor
    'semiconductor_cmp': 'src/test-scenarios/semiconductor/semiconductor_cmp_001.csv',
    'semiconductor_etch': 'src/test-scenarios/semiconductor/semiconductor_etch_002.csv',
    'semiconductor_deposition': 'src/test-scenarios/semiconductor/semiconductor_deposition_003.csv',
    'semiconductor_full': 'src/test-scenarios/semiconductor/semiconductor_full_004.csv',
    # Chemical
    'chemical_reactor': 'src/test-scenarios/chemical/chemical_reactor_001.csv',
    'chemical_distillation': 'src/test-scenarios/chemical/chemical_distillation_002.csv',
    'chemical_refining': 'src/test-scenarios/chemical/chemical_refining_003.csv',
    'chemical_full': 'src/test-scenarios/chemical/chemical_full_004.csv',
    # Automotive
    'automotive_welding': 'src/test-scenarios/automotive/automotive_welding_001.csv',
    'automotive_painting': 'src/test-scenarios/automotive/automotive_painting_002.csv',
    'automotive_press': 'src/test-scenarios/automotive/automotive_press_003.csv',
    'automotive_assembly': 'src/test-scenarios/automotive/automotive_assembly_004.csv',
    # Battery
    'battery_formation': 'src/test-scenarios/battery/battery_formation_001.csv',
    'battery_coating': 'src/test-scenarios/battery/battery_coating_002.csv',
    'battery_aging': 'src/test-scenarios/battery/battery_aging_003.csv',
    'battery_production': 'src/test-scenarios/battery/battery_production_004.csv',
    # Steel
    'steel_rolling': 'src/test-scenarios/steel/steel_rolling_001.csv',
    'steel_converter': 'src/test-scenarios/steel/steel_converter_002.csv',
    'steel_casting': 'src/test-scenarios/steel/steel_casting_003.csv',
    'steel_production': 'src/test-scenarios/steel/steel_production_004.csv',
}


def load_test_scenarios_data(target_process: str, start: str, end: str):
    """
    로컬 CSV 파일에서 데이터 로드

    Args:
        target_process: 공정 식별자
        start: 시작 시간
        end: 종료 시간

    Returns:
        pandas DataFrame
    """
    

    # CSV 파일 경로 찾기
    csv_path = TEST_SCENARIOS_DATA_MAPPING.get(target_process)

    # CSV 파일 읽기
    data = pd.read_csv(csv_path)
    print(f"   ✓ 로드 완료: {len(data)} 행")

    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], utc=True)
    start_time = pd.to_datetime(start, utc=True)
    end_time = pd.to_datetime(end, utc=True)
    data = data[(data['TIMESTAMP'] >= start_time) & (data['TIMESTAMP'] <= end_time)]

    return data