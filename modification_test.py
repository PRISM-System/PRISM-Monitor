import sys
import os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from prism_monitor.modules.event_precursor._precursor import load_and_explore_data
from prism_monitor.modules.event_precursor.precursor import precursor


def test_precursor_with_test_data():
    print("=" * 80)
    print("이상 징후 예측 모듈 테스트 시작")
    print("=" * 80)

    test_data_path = project_root / "prism_monitor" / "test-scenarios" / "test_data"

    if not test_data_path.exists():
        print(f"테스트 데이터 경로가 존재하지 않습니다: {test_data_path}")
        return False

    print(f"테스트 데이터 경로: {test_data_path}")

    # 2. 데이터 로드
    print("\n" + "=" * 80)
    print("Step 1: 데이터 로딩")
    print("=" * 80)

    try:
        datasets = load_and_explore_data(str(test_data_path))

        if not datasets:
            print("데이터 로딩 실패!")
            return False

        print(f"\n총 {len(datasets)}개의 데이터셋 로드 완료")
        print("\n로드된 데이터셋 목록:")
        for name, df in datasets.items():
            print(f"  - {name}: {df.shape}")

    except Exception as e:
        print(f"\n데이터 로딩 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Precursor 모듈 실행
    print("\n" + "=" * 80)
    print("Step 2: Precursor 모듈 실행")
    print("=" * 80)

    try:
        result = precursor(datasets)

        print("\n" + "=" * 80)
        print("Step 3: 결과 확인")
        print("=" * 80)

        # 결과 출력
        print("\n예측 결과:")
        print("-" * 80)

        if 'summary' in result:
            summary = result['summary']
            predicted_value = summary.get('predicted_value', 'N/A')
            is_anomaly = summary.get('is_anomaly', 'N/A')

            print(f"  예측값 (Predicted Value): {predicted_value}")
            print(f"  이상 상태 (Anomaly Status): {is_anomaly}")

            # 이상 상태 해석
            status_map = {
                '0': '정상 (Normal)',
                '1': '경고 (Warning)',
                '2': '위험 (Critical)'
            }
            status_text = status_map.get(is_anomaly, '알 수 없음')
            print(f"  상태 해석: {status_text}")

        if 'error' in result:
            print(f"\n경고: {result['error']}")

        print("-" * 80)

        # 전체 결과 출력
        print("\n 전체 결과 (Raw):")
        print(result)

        print("\n" + "=" * 80)
        print("테스트 완료!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nPrecursor 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_scenarios():
    print("\n" + "=" * 80)
    print("개별 산업별 테스트")
    print("=" * 80)

    test_data_path = project_root / "prism_monitor" / "test-scenarios" / "test_data"
    industries = ['semiconductor', 'battery', 'automotive', 'chemical', 'steel']

    for industry in industries:
        industry_path = test_data_path / industry
        if industry_path.exists():
            print(f"\n{'='*60}")
            print(f"테스트: {industry.upper()}")
            print(f"{'='*60}")

            try:
                datasets = load_and_explore_data(str(industry_path))
                if datasets:
                    print(f"✅ {industry}: {len(datasets)}개 파일 로드")
                    for name, df in datasets.items():
                        print(f"  - {name}: {df.shape}")
                else:
                    print(f"⚠️  {industry}: 데이터 없음")
            except Exception as e:
                print(f"❌ {industry}: 오류 - {e}")


if __name__ == "__main__":

    success = test_precursor_with_test_data()

    print("\n\n")
    user_input = input("개별 산업별 세부 테스트를 실행하시겠습니까? (y/n): ")

    if user_input.lower() in ['y', 'yes']:
        test_individual_scenarios()

    print("\n\n" + "=" * 80)
    if success:
        print("모든 테스트 통과!")
    else:
        print("테스트 실패 - 위 오류 메시지를 확인하세요.")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)
