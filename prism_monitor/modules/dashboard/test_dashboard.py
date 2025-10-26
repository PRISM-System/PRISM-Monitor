"""
Dashboard.py 기능 테스트 스크립트
주요 함수들이 정상 동작하는지 확인
"""
import sys
import logging
from pathlib import Path

# 현재 스크립트 위치 기준으로 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# dashboard 모듈 import
from dashboard import (
    _iter_csv_datasets,
    _tf_init_devices,
    build_anomaly_registry_from_root,
    default_state_fn,
    _resolve_state,
    _to_python_scalar,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_MODELS_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_tf_device_init():
    """TensorFlow 디바이스 초기화 테스트"""
    print_section("TEST 1: TensorFlow 디바이스 초기화")

    try:
        _tf_init_devices("auto")
        print("✅ TensorFlow 디바이스 초기화 성공")
    except Exception as e:
        print(f"❌ TensorFlow 디바이스 초기화 실패: {e}")
        return False

    return True


def test_csv_loading():
    """CSV 데이터셋 로딩 테스트"""
    print_section("TEST 2: CSV 데이터셋 로딩")

    print(f"📂 테스트 데이터 경로: {DEFAULT_TEST_DATA_DIR}")

    try:
        datasets = _iter_csv_datasets(DEFAULT_TEST_DATA_DIR)

        if not datasets:
            print("❌ 로드된 데이터셋이 없습니다")
            return False, None

        print(f"✅ {len(datasets)}개 데이터셋 로드 성공\n")

        # 각 데이터셋 정보 출력
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds['csv_name']}")
            print(f"   - 산업: {ds['industry']}")
            print(f"   - 공정: {ds['process']}")
            print(f"   - 라인 컬럼: {ds['line']}")
            print(f"   - 메트릭 개수: {len(ds['metric_cols'])}")
            print(f"   - 데이터 행 수: {len(ds['data'])}")

        return True, datasets

    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_loading(datasets):
    """모델 로딩 테스트"""
    print_section("TEST 3: 이상치 탐지 모델 로딩")

    if not datasets:
        print("⚠️  데이터셋이 없어 모델 로딩 스킵")
        return False, None

    print(f"📂 모델 경로: {DEFAULT_MODELS_ROOT}")

    try:
        anomaly_models = build_anomaly_registry_from_root(DEFAULT_MODELS_ROOT, datasets)

        if not anomaly_models:
            print("⚠️  로드된 모델이 없습니다")
            return False, None

        print(f"✅ {len(anomaly_models)}개 모델 로드 성공\n")

        # 모델 정보 출력
        for csv_name in anomaly_models.keys():
            print(f"  - {csv_name}: 모델 함수 등록됨")

        return True, anomaly_models

    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_anomaly_detection(datasets, anomaly_models):
    """이상치 탐지 함수 실행 테스트"""
    print_section("TEST 4: 이상치 탐지 실행")

    if not datasets or not anomaly_models:
        print("⚠️  데이터셋 또는 모델이 없어 이상치 탐지 스킵")
        return False

    success_count = 0
    fail_count = 0

    # 각 데이터셋의 첫 번째 행으로 테스트
    for ds in datasets:
        csv_name = ds["csv_name"]

        if csv_name not in anomaly_models:
            continue

        anomaly_fn = anomaly_models[csv_name]
        df = ds["data"]

        if len(df) == 0:
            continue

        # 첫 번째 행 테스트
        row = df.iloc[0]

        try:
            result = anomaly_fn(row)

            print(f"\n📊 {csv_name}:")
            print(f"   - 모델: {result.get('model')}")
            print(f"   - 점수: {result.get('score'):.6f}")
            print(f"   - 임계값: {result.get('threshold'):.6f}")
            print(f"   - 이상치 여부: {result.get('is_anomaly')}")

            success_count += 1

        except Exception as e:
            print(f"\n❌ {csv_name} 이상치 탐지 실패: {e}")
            fail_count += 1

    print(f"\n✅ 성공: {success_count}개")
    if fail_count > 0:
        print(f"❌ 실패: {fail_count}개")

    return success_count > 0


def test_state_resolution(datasets):
    """상태 resolver 테스트"""
    print_section("TEST 5: 공정 상태 추론")

    if not datasets:
        print("⚠️  데이터셋이 없어 상태 추론 스킵")
        return False

    # 첫 번째 데이터셋으로 테스트
    ds = datasets[0]
    df = ds["data"]

    if len(df) == 0:
        print("⚠️  데이터가 없습니다")
        return False

    row = df.iloc[0]

    try:
        state = _resolve_state(row, default_state_fn, list(df.columns))
        print(f"✅ 상태 추론 성공: {state}")
        print(f"   - 데이터셋: {ds['csv_name']}")
        print(f"   - 첫 행 데이터: {dict(row[:5])}")

        return True

    except Exception as e:
        print(f"❌ 상태 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_conversion(datasets):
    """데이터 변환 유틸리티 테스트"""
    print_section("TEST 6: 데이터 변환 유틸리티")

    if not datasets:
        print("⚠️  데이터셋이 없어 변환 테스트 스킵")
        return False

    ds = datasets[0]
    df = ds["data"]

    if len(df) == 0:
        print("⚠️  데이터가 없습니다")
        return False

    row = df.iloc[0]

    try:
        # 각 컬럼 값을 Python 스칼라로 변환
        for col in list(row.index)[:5]:  # 처음 5개 컬럼만 테스트
            original = row[col]
            converted = _to_python_scalar(original)
            print(f"  {col}: {type(original).__name__} → {type(converted).__name__} ({converted})")

        print("✅ 데이터 변환 성공")
        return True

    except Exception as e:
        print(f"❌ 데이터 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "🔬" * 30)
    print("  DASHBOARD.PY 기능 테스트 시작")
    print("🔬" * 30)

    results = {}

    # TEST 1: TensorFlow 디바이스 초기화
    results["tf_device"] = test_tf_device_init()

    # TEST 2: CSV 로딩
    csv_success, datasets = test_csv_loading()
    results["csv_loading"] = csv_success

    # TEST 3: 모델 로딩
    model_success, anomaly_models = test_model_loading(datasets)
    results["model_loading"] = model_success

    # TEST 4: 이상치 탐지
    results["anomaly_detection"] = test_anomaly_detection(datasets, anomaly_models)

    # TEST 5: 상태 추론
    results["state_resolution"] = test_state_resolution(datasets)

    # TEST 6: 데이터 변환
    results["data_conversion"] = test_data_conversion(datasets)

    # 결과 요약
    print_section("테스트 결과 요약")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n총 {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n🎉 모든 테스트 통과!")
        return 0
    else:
        print(f"\n⚠️  {total - passed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
