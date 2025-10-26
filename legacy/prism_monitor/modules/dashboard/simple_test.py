"""
Dashboard.py 간단 테스트 예제
원하는 기능만 골라서 테스트할 수 있습니다
"""
from dashboard import (
    _iter_csv_datasets,
    _tf_init_devices,
    build_anomaly_registry_from_root,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_MODELS_ROOT,
)

# ========================================
# 예제 1: CSV 데이터 로딩만 테스트
# ========================================
print("📂 CSV 데이터 로딩 테스트\n")

datasets = _iter_csv_datasets(DEFAULT_TEST_DATA_DIR)
print(f"✅ {len(datasets)}개 데이터셋 로드됨\n")

# 첫 번째 데이터셋 확인
if datasets:
    ds = datasets[0]
    print(f"첫 번째 데이터셋: {ds['csv_name']}")
    print(f"  - 산업: {ds['industry']}")
    print(f"  - 공정: {ds['process']}")
    print(f"  - 데이터 행 수: {len(ds['data'])}")
    print(f"  - 메트릭: {ds['metric_cols'][:3]}...")

    # 첫 5개 행 미리보기
    print(f"\n데이터 미리보기:")
    print(ds['data'].head())

# ========================================
# 예제 2: 특정 산업 데이터만 필터링
# ========================================
print("\n" + "="*60)
print("🔍 특정 산업 데이터만 필터링\n")

battery_datasets = [d for d in datasets if d['industry'].lower() == 'battery']
print(f"Battery 산업 데이터셋: {len(battery_datasets)}개")
for ds in battery_datasets:
    print(f"  - {ds['csv_name']}")

# ========================================
# 예제 3: 모델 로딩 및 이상치 탐지
# ========================================
print("\n" + "="*60)
print("🤖 모델 로딩 및 이상치 탐지\n")

# TensorFlow 초기화
_tf_init_devices("auto")

# 모델 로드
anomaly_models = build_anomaly_registry_from_root(DEFAULT_MODELS_ROOT, datasets)
print(f"✅ {len(anomaly_models)}개 모델 로드됨\n")

# 첫 번째 데이터셋으로 이상치 탐지 테스트
if datasets and datasets[0]['csv_name'] in anomaly_models:
    ds = datasets[0]
    anomaly_fn = anomaly_models[ds['csv_name']]

    # 처음 3개 행에 대해 이상치 탐지
    print(f"📊 {ds['csv_name']} 이상치 탐지 결과:\n")

    for i in range(min(3, len(ds['data']))):
        row = ds['data'].iloc[i]
        result = anomaly_fn(row)

        print(f"  행 {i+1}:")
        print(f"    - 점수: {result['score']:.4f}")
        print(f"    - 임계값: {result['threshold']:.4f}")
        print(f"    - 이상치: {'⚠️  YES' if result['is_anomaly'] else '✅ NO'}")

# ========================================
# 예제 4: 특정 CSV 파일만 테스트
# ========================================
print("\n" + "="*60)
print("🎯 특정 CSV 파일만 테스트\n")

# 원하는 CSV 이름 지정
target_csv = "battery_formation_001.csv"

target_ds = next((d for d in datasets if d['csv_name'] == target_csv), None)

if target_ds:
    print(f"✅ {target_csv} 찾음!")
    print(f"  - 라인 컬럼: {target_ds['line']}")
    print(f"  - 메트릭: {target_ds['metric_cols']}")

    # 랜덤 샘플 10개 추출
    import random
    sample_indices = random.sample(range(len(target_ds['data'])), min(10, len(target_ds['data'])))

    if target_csv in anomaly_models:
        anomaly_fn = anomaly_models[target_csv]
        anomaly_count = 0

        for idx in sample_indices:
            row = target_ds['data'].iloc[idx]
            result = anomaly_fn(row)
            if result['is_anomaly']:
                anomaly_count += 1

        print(f"\n  랜덤 샘플 10개 중 {anomaly_count}개 이상치 감지")
else:
    print(f"❌ {target_csv}를 찾을 수 없습니다")

print("\n" + "="*60)
print("✅ 테스트 완료!")
