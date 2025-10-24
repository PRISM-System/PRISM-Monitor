"""
설정 파일 로더

sample_data_templates.json에서 정상 범위 및 설정을 로드합니다.
"""

import json
import os
from typing import Dict, Optional, List


def load_normal_ranges_from_template(
    template_path: str = "prism_monitor/test-scenarios/test_data/sample_data_templates.json"
) -> Dict[str, Dict[str, tuple]]:
    """
    sample_data_templates.json에서 정상 범위를 로드하여
    DataValidityChecker가 사용하는 형식으로 변환

    Args:
        template_path: JSON 템플릿 파일 경로

    Returns:
        {
            'semi_cmp_sensors': {
                'motor_current': (15.0, 18.0),
                'slurry_flow_rate': (200, 300),
                ...
            },
            'semi_etch_sensors': {...},
            ...
        }
    """
    normal_ranges = {}

    try:
        if not os.path.exists(template_path):
            print(f"Warning: Template file not found: {template_path}")
            return {}

        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # semiconductor 섹션 확인
        if 'templates' not in template_data or 'semiconductor' not in template_data['templates']:
            print("Warning: 'templates.semiconductor' section not found in template")
            return {}

        semiconductor_templates = template_data['templates']['semiconductor']

        # 각 공정별 정상 범위 추출
        process_mapping = {
            'cmp_sensors': 'semi_cmp_sensors',
            'etch_sensors': 'semi_etch_sensors',
            'deposition_sensors': 'semi_cvd_sensors',  # CVD
            # ion_sensors, photo_sensors 등 추가 가능
        }

        for template_key, table_name in process_mapping.items():
            if template_key not in semiconductor_templates:
                continue

            process_config = semiconductor_templates[template_key]

            if 'normal_ranges' not in process_config:
                continue

            # 정상 범위를 (min, max) 튜플로 변환
            ranges = {}
            for param_name, range_values in process_config['normal_ranges'].items():
                if isinstance(range_values, list) and len(range_values) == 2:
                    # 컬럼명 소문자 변환 (데이터 정규화와 일치)
                    param_name_lower = param_name.lower()
                    ranges[param_name_lower] = (range_values[0], range_values[1])

            if ranges:
                normal_ranges[table_name] = ranges

        print(f"정상 범위 로드 완료: {len(normal_ranges)}개 공정")
        for table_name, ranges in normal_ranges.items():
            print(f"  - {table_name}: {len(ranges)}개 파라미터")

    except Exception as e:
        print(f"Error loading normal ranges from template: {e}")
        import traceback
        traceback.print_exc()

    return normal_ranges


def load_process_config(
    process_name: str,
    template_path: str = "prism_monitor/test-scenarios/test_data/sample_data_templates.json"
) -> Optional[Dict]:
    """
    특정 공정의 전체 설정을 로드

    Args:
        process_name: 공정명 (예: 'semi_cmp_sensors')
        template_path: JSON 템플릿 파일 경로

    Returns:
        공정 설정 딕셔너리 또는 None
    """
    try:
        if not os.path.exists(template_path):
            return None

        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        semiconductor_templates = template_data.get('templates', {}).get('semiconductor', {})

        # 공정명 매핑 (역방향)
        table_to_template_mapping = {
            'semi_cmp_sensors': 'cmp_sensors',
            'semi_etch_sensors': 'etch_sensors',
            'semi_cvd_sensors': 'deposition_sensors',
        }

        template_key = table_to_template_mapping.get(process_name)

        if template_key and template_key in semiconductor_templates:
            return semiconductor_templates[template_key]

    except Exception as e:
        print(f"Error loading process config for {process_name}: {e}")

    return None


def get_available_processes(
    template_path: str = "prism_monitor/test-scenarios/test_data/sample_data_templates.json"
) -> List[str]:
    """
    사용 가능한 공정 목록 조회

    Returns:
        공정명 리스트 (예: ['semi_cmp_sensors', 'semi_etch_sensors', ...])
    """
    processes = []

    try:
        if not os.path.exists(template_path):
            return []

        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        semiconductor_templates = template_data.get('templates', {}).get('semiconductor', {})

        table_to_template_mapping = {
            'semi_cmp_sensors': 'cmp_sensors',
            'semi_etch_sensors': 'etch_sensors',
            'semi_cvd_sensors': 'deposition_sensors',
        }

        for table_name, template_key in table_to_template_mapping.items():
            if template_key in semiconductor_templates:
                processes.append(table_name)

    except Exception as e:
        print(f"Error getting available processes: {e}")

    return processes


def get_process_columns(
    process_name: str,
    template_path: str = "prism_monitor/test-scenarios/test_data/sample_data_templates.json"
) -> List[str]:
    """
    특정 공정의 컬럼 목록 조회

    Args:
        process_name: 공정명 (예: 'semi_cmp_sensors')

    Returns:
        컬럼명 리스트 (소문자 변환됨)
    """
    config = load_process_config(process_name, template_path)

    if config and 'columns' in config:
        # 컬럼명 소문자 변환
        return [col.lower() for col in config['columns']]

    return []


if __name__ == "__main__":
    # 테스트 코드
    print("=== 정상 범위 로드 테스트 ===")
    normal_ranges = load_normal_ranges_from_template()

    for table_name, ranges in normal_ranges.items():
        print(f"\n{table_name}:")
        for param, (min_val, max_val) in list(ranges.items())[:3]:  # 처음 3개만 출력
            print(f"  {param}: {min_val} ~ {max_val}")

    print("\n=== 사용 가능한 공정 ===")
    processes = get_available_processes()
    print(processes)

    print("\n=== CMP 공정 설정 ===")
    cmp_config = load_process_config('semi_cmp_sensors')
    if cmp_config:
        print(f"Dataset: {cmp_config.get('dataset_file')}")
        print(f"Scenario: {cmp_config.get('scenario_id')}")
        print(f"Columns: {cmp_config.get('columns')[:5]}...")  # 처음 5개만

    print("\n=== CMP 공정 컬럼 목록 ===")
    cmp_columns = get_process_columns('semi_cmp_sensors')
    print(cmp_columns)
