#!/usr/bin/env python3
"""
데이터 로딩 간단 테스트 - torch 없이
"""

import pandas as pd
import os

def simple_load_test(data_base_path):
    print("=" * 70)
    print("데이터 로딩 테스트 (간단 버전)")
    print("=" * 70)
    print(f"데이터 경로: {data_base_path}\n")

    datasets = {}

    # 디렉토리 탐색
    if os.path.isdir(data_base_path):
        for industry_dir in os.listdir(data_base_path):
            industry_path = os.path.join(data_base_path, industry_dir)

            # 파일인 경우 (최상위 CSV)
            if os.path.isfile(industry_path) and industry_path.endswith('.csv'):
                filename = os.path.basename(industry_path)
                key = filename.replace('.csv', '')
                print(f"📄 Loading: {filename}")
                try:
                    df = pd.read_csv(industry_path)
                    datasets[key] = df
                    print(f"   Shape: {df.shape}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")

            # 디렉토리인 경우 (카테고리 폴더)
            elif os.path.isdir(industry_path):
                print(f"\n📁 Category: {industry_dir}")
                for filename in os.listdir(industry_path):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(industry_path, filename)
                        key = filename.replace('.csv', '')
                        print(f"   Loading: {filename}")
                        try:
                            df = pd.read_csv(file_path)
                            datasets[key] = df
                            print(f"      Shape: {df.shape}")

                            # ID 컬럼 확인
                            id_cols = [col for col in df.columns if col.endswith('_ID')]
                            if id_cols:
                                print(f"      ID 컬럼: {id_cols[0]}")

                            # 숫자형 컬럼
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            print(f"      숫자형 센서: {len(numeric_cols)}개")

                        except Exception as e:
                            print(f"      ❌ Error: {e}")
                print()

    print("\n" + "=" * 70)
    print("로딩 결과 요약")
    print("=" * 70)
    print(f"총 로드된 파일 수: {len(datasets)}\n")

    # 카테고리별 통계
    categories = {}
    for name in datasets.keys():
        parts = name.split('_')
        if len(parts) > 0:
            category = parts[0]
            categories[category] = categories.get(category, 0) + 1

    print("📊 카테고리별 파일 수:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}개 파일")

    print("\n" + "=" * 70)

    # ID 컬럼 타입 확인
    print("\n🔍 ID 컬럼 분석:")
    id_types = {}
    for name, df in datasets.items():
        id_cols = [col for col in df.columns if col.endswith('_ID')]
        if id_cols:
            id_type = id_cols[0]
            if id_type not in id_types:
                id_types[id_type] = []
            id_types[id_type].append(name)

    for id_type, files in sorted(id_types.items()):
        print(f"\n   {id_type}: {len(files)}개 파일")
        for f in files[:3]:  # 처음 3개만 표시
            print(f"      - {f}")
        if len(files) > 3:
            print(f"      ... 외 {len(files) - 3}개")

    return datasets

if __name__ == "__main__":
    DATA_PATH = '/home/jonghak/PRISM-Monitor/prism_monitor/test-scenarios/test_data/'
    datasets = simple_load_test(DATA_PATH)

    if datasets:
        print("\n✅ 데이터 로딩 성공!")
    else:
        print("\n❌ 데이터 로딩 실패!")
