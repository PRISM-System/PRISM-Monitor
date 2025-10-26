#!/usr/bin/env python3
"""
데이터 로딩 테스트 스크립트
_precursor.py가 test_data 폴더의 데이터를 제대로 불러오는지 확인
"""

import sys
sys.path.insert(0, '/home/jonghak/PRISM-Monitor')

from prism_monitor.modules.event_precursor._precursor import load_and_explore_data

def test_data_loading():
    DATA_PATH = '/home/jonghak/PRISM-Monitor/prism_monitor/test-scenarios/test_data/'

    print("=" * 70)
    print("데이터 로딩 테스트 시작")
    print("=" * 70)
    print(f"데이터 경로: {DATA_PATH}\n")

    # 데이터 로드
    datasets = load_and_explore_data(DATA_PATH)

    print("\n" + "=" * 70)
    print("로딩 결과 요약")
    print("=" * 70)
    print(f"총 로드된 파일 수: {len(datasets)}")
    print()

    # 각 데이터셋 상세 정보
    for name, df in datasets.items():
        print(f"\n📁 {name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # ID 컬럼 확인
        id_cols = [col for col in df.columns if col.endswith('_ID')]
        if id_cols:
            print(f"   ID 컬럼: {id_cols}")

        # 숫자형 컬럼 수
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"   숫자형 센서 수: {len(numeric_cols)}")

    print("\n" + "=" * 70)

    # 카테고리별 통계
    categories = {}
    for name in datasets.keys():
        # 파일명에서 카테고리 추출 (예: automotive_welding_001 -> automotive)
        parts = name.split('_')
        if len(parts) > 0:
            category = parts[0]
            categories[category] = categories.get(category, 0) + 1

    print("\n📊 카테고리별 파일 수:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}개 파일")

    print("\n" + "=" * 70)

    return datasets

if __name__ == "__main__":
    datasets = test_data_loading()

    if datasets:
        print("\n✅ 데이터 로딩 성공!")
    else:
        print("\n❌ 데이터 로딩 실패!")
