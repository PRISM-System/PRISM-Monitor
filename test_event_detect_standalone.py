"""
이상치 탐지 모듈 단독 테스트 스크립트

사용법:
    # 기본 사용 (API 시도 → 실패 시 CSV 자동 폴백)
    python test_event_detect_standalone.py --process semi_cmp_sensors

    # 시간 범위 지정
    python test_event_detect_standalone.py --process semi_cmp_sensors \
        --start "2025-05-01T00:00:00Z" --end "2025-05-01T01:00:00Z"

    # 데이터베이스 URL 지정 (API 우선 사용)
    python test_event_detect_standalone.py --process semi_cmp_sensors \
        --url https://147.47.39.144:8000

    # CSV 강제 사용 (API 시도 안 함)
    python test_event_detect_standalone.py --process semi_cmp_sensors --csv

동작 방식:
    1. URL 제공 없음: 로컬 CSV 파일 사용
    2. URL 제공: API 시도 → 실패 시 자동으로 CSV 폴백
    3. --csv 플래그: API 시도 없이 CSV 직접 사용
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# TensorFlow GPU 설정 (모듈 import 전에 설정)
# CuDNN 버전 불일치 문제 우회를 위해 GPU 비활성화
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 완전히 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 최소화

# PRISM Monitor 모듈 import
from prism_monitor.data.database import PrismCoreDataBase
from prism_monitor.modules.event.event_detect import detect_anomalies_realtime


def test_process_mode(base_url: str, process_name: str, start: str = None, end: str = None, use_csv: bool = False):
    """
    공정별 모델 모드 테스트
    """
    print("="*70)
    print(f"공정별 모델 테스트: {process_name}")
    if use_csv:
        print("모드: 로컬 CSV 파일")
    else:
        print("모드: API (PrismCoreDataBase)")
    print("="*70)

    # DB 연결
    prism_core_db = None
    if base_url and not use_csv:
        # URL이 제공되고 CSV 강제 모드가 아닌 경우 DB 연결 시도
        try:
            print(f"\n[1/6] DB 연결 시도: {base_url}")
            prism_core_db = PrismCoreDataBase(base_url=base_url)
            print(f"   ✓ DB 연결 성공")
        except Exception as e:
            print(f"   ⚠️  DB 연결 실패: {e}")
            print(f"   → 로컬 CSV 파일로 폴백됩니다")
    else:
        if use_csv:
            print(f"\n[1/6] CSV 모드 (DB 연결 생략)")
        else:
            print(f"\n[1/6] URL 미제공 (로컬 CSV 사용)")

    # 시간 범위 설정
    if not start or not end:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        start = start_time.isoformat() + 'Z'
        end = end_time.isoformat() + 'Z'

    print(f"[2/6] 시간 범위: {start} ~ {end}")

    # 모델 확인
    print(f"[3/6] 모델 확인 중: {process_name}")
    from prism_monitor.utils.process_model_manager import ProcessModelManager

    manager = ProcessModelManager(base_model_dir='models')
    available = manager.list_available_processes()

    if process_name not in available:
        print(f"\n⚠️  경고: {process_name} 모델이 아직 훈련되지 않았습니다.")
        print(f"   사용 가능한 공정: {available}")
        print(f"\n   해결 방법:")
        print(f"   1. MULTI_PROCESS_MODEL_TRAINING_GUIDE.md를 참고하여 모델 훈련")
        return False

    model_info = manager.get_model_info(process_name)
    print(f"   ✓ 모델 버전: {model_info.get('model_version', 'unknown')}")
    print(f"   ✓ Feature 개수: {model_info.get('feature_count', 0)}")
    print(f"   ✓ Threshold: {model_info.get('threshold', 0):.6f}")

    # 이상 탐지 수행
    print(f"[4/6] 이상 탐지 수행 중 ({process_name})...")
    try:
        anomalies, drift_results, analysis, vis_json = detect_anomalies_realtime(
            prism_core_db,
            start=start,
            end=end,
            target_process=process_name,  # 공정 지정
            use_csv=use_csv  # CSV 모드 옵션
        )

        print("[5/6] 결과 분석 중...")
        print(f"\n✓ 탐지 완료!")
        print(f"  - 탐지된 이상치: {len(anomalies)}건")
        print(f"  - Drift 탐지: {len(drift_results)}건")
        print(f"  - 분석 레코드: {len(analysis) if analysis else 0}개")

        # 이상치 상세 정보
        if anomalies:
            print(f"\n[6/6] 이상치 상세 정보:")
            for i, anomaly in enumerate(anomalies[:5], 1):
                print(f"\n  이상치 {i}:")
                if isinstance(anomaly, dict):
                    for key, value in anomaly.items():
                        if key in ['timestamp', 'table_name', 'anomaly_score', 'pno', 'lot_no', 'equipment_id']:
                            print(f"    - {key}: {value}")
                else:
                    print(f"    {anomaly}")

            if len(anomalies) > 5:
                print(f"\n  ... 외 {len(anomalies) - 5}건")
        else:
            print("\n[6/6] ✓ 이상치가 탐지되지 않았습니다.")

        # 분석 데이터 저장
        if analysis:
            output_file = f'test_results_{process_name}.json'
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 분석 결과 저장: {output_file}")

        return True

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_availability():
    """
    사용 가능한 데이터 확인
    """
    print("\n" + "="*70)
    print("데이터 가용성 확인")
    print("="*70)

    import os
    from glob import glob

    # 신규 데이터 확인
    new_data_path = 'prism_monitor/test-scenarios/test_data/semiconductor'
    if os.path.exists(new_data_path):
        new_files = glob(os.path.join(new_data_path, '*.csv'))
        print(f"\n✓ 신규 데이터 ({len(new_files)}개):")
        for file in new_files:
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  - {os.path.basename(file)} ({size_mb:.2f} MB)")
    else:
        print(f"\n✗ 신규 데이터 없음: {new_data_path}")

    # 레거시 데이터 확인
    old_data_path = 'prism_monitor/data/Industrial_DB_sample'
    if os.path.exists(old_data_path):
        old_files = glob(os.path.join(old_data_path, '*.csv'))
        print(f"\n✓ 레거시 데이터 ({len(old_files)}개):")
        for file in old_files[:5]:  # 최대 5개만
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  - {os.path.basename(file)} ({size_mb:.2f} MB)")
        if len(old_files) > 5:
            print(f"  ... 외 {len(old_files) - 5}개")
    else:
        print(f"\n✗ 레거시 데이터 없음: {old_data_path}")

    # 모델 확인
    print(f"\n✓ 모델 디렉토리:")
    from prism_monitor.utils.process_model_manager import ProcessModelManager

    manager = ProcessModelManager()
    available = manager.list_available_processes()

    if available:
        print(f"  공정별 모델 ({len(available)}개):")
        for proc in available:
            info = manager.get_model_info(proc)
            print(f"  - {proc}: v{info.get('model_version', 'unknown')}")
    else:
        print(f"  ⚠️  공정별 모델 없음 (훈련 필요)")


def main():
    parser = argparse.ArgumentParser(
        description='이상치 탐지 모듈 단독 테스트 (공정별 모델)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 데이터 및 모델 확인
  python test_event_detect_standalone.py --check

  # 기본 사용 (API 시도 → 실패 시 CSV 자동 폴백)
  python test_event_detect_standalone.py --process semi_cmp_sensors

  # 시간 범위 지정
  python test_event_detect_standalone.py --process semi_cmp_sensors \\
      --start "2025-05-01T00:00:00Z" --end "2025-05-01T01:00:00Z"

  # API 우선 사용
  python test_event_detect_standalone.py --process semi_cmp_sensors \\
      --url https://147.47.39.144:8000

  # CSV 강제 사용
  python test_event_detect_standalone.py --process semi_cmp_sensors --csv
        """
    )

    parser.add_argument('--process', type=str,
                       help='공정 이름 (예: semi_cmp_sensors, semi_etch_sensors)')
    parser.add_argument('--start', type=str,
                       help='시작 시간 (ISO format, 예: 2025-05-01T00:00:00Z)')
    parser.add_argument('--end', type=str,
                       help='종료 시간 (ISO format, 예: 2025-05-01T01:00:00Z)')
    parser.add_argument('--url', type=str,
                       help='PRISM Core Database URL (선택사항, 미제공시 CSV 사용)')
    parser.add_argument('--csv', action='store_true',
                       help='로컬 CSV 파일 강제 사용 (API 시도 안 함)')
    parser.add_argument('--check', action='store_true',
                       help='데이터 및 모델 가용성만 확인')

    args = parser.parse_args()

    # 데이터 확인 모드
    if args.check:
        check_data_availability()
        return

    # 공정 이름 필수
    if not args.process:
        print("오류: --process 또는 --check를 지정해야 합니다.")
        print("사용 가능한 공정: semi_cmp_sensors, semi_etch_sensors, semi_cvd_sensors 등")
        parser.print_help()
        sys.exit(1)

    # 데이터베이스 URL 가져오기 (선택사항)
    base_url = args.url
    if not base_url and not args.csv:
        # --url도 없고 --csv도 아니면 환경변수 확인
        base_url = os.environ.get('PRISM_CORE_DATABASE_URL')
        if base_url:
            print(f"✓ 환경변수에서 URL 로드: {base_url}")
        else:
            print(f"⚠️  URL 미제공 - 로컬 CSV 파일만 사용합니다")

    # 데이터 확인
    check_data_availability()

    # 테스트 실행
    success = test_process_mode(base_url, args.process, args.start, args.end, use_csv=args.csv)

    # 결과 출력
    print("\n" + "="*70)
    if success:
        print("✓ 테스트 완료!")
    else:
        print("✗ 테스트 실패")
    print("="*70)


if __name__ == '__main__':
    main()
