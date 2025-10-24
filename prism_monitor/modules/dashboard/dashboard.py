from __future__ import annotations

import os
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Callable, Sequence

import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# 상태 컬럼 후보
STATE_COL_CANDIDATES = ("STATE", "STATUS", "PROCESS_STATE")
LINE_COL_PRIORITY = (
    "EQUIPMENT_ID", "PROCESS_ID", "PRODUCTION_LINE",
    "STATION_ID", "CHAMBER_ID", "PRESS_ID", "SENSOR_ID",
    "LINE_ID", "CELL_ID", "BOOTH_ID", "CASTER_ID", "CONVERTER_ID",
)

# 기본 경로 자동화 (파일 위치 기준)
MODULE_DIR = Path(__file__).resolve().parent
MODULES_ROOT = MODULE_DIR.parent
PRISM_ROOT = MODULES_ROOT.parent
REPO_ROOT = PRISM_ROOT.parent
DEFAULT_TEST_DATA_DIR = str(PRISM_ROOT / "test-scenarios" / "test_data")
DEFAULT_MODELS_ROOT = str(REPO_ROOT / "models")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "dashboard_output")

# 타입 힌트
AnomalyFn = Callable[[pd.Series], Dict[str, object]]
StateFn = Callable[[pd.Series], str]


def _infer_line_col(columns: Sequence[str]) -> Optional[str]:
    # 우선순위 컬럼 먼저
    for c in LINE_COL_PRIORITY:
        if c in columns:
            return c
    # 그 외 `_ID` 로 끝나는 첫 컬럼
    for c in columns:
        if str(c).upper().endswith("_ID"):
            return c
    return None


def _to_python_scalar(x):
    """NumPy 스칼라를 파이썬 기본형으로, NaN은 None으로 변환."""
    if pd.isna(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def _resolve_state(row: pd.Series, resolver: Optional[StateFn], df_columns: List[str]) -> Optional[str]:
    """공정 상태 문자열을 결정한다. 콜백 우선, 없으면 컬럼에서 추출."""
    if resolver is not None:
        try:
            state = resolver(row)
            if isinstance(state, str) and state:
                return state
        except Exception as e:
            logging.warning("state_resolver 실행 중 오류: %s", e)
    for col in STATE_COL_CANDIDATES:
        if col in df_columns:
            val = row[col]
            if pd.isna(val):
                continue
            return str(val)
    return None


def _iter_csv_datasets(test_data_dir: str) -> List[Dict]:
    """CSV 자동 스캔 후 메타/데이터 로드."""
    datasets: List[Dict] = []

    if not os.path.isdir(test_data_dir):
        logging.error("test_data_dir가 존재하지 않습니다: %s", test_data_dir)
        return datasets

    for root, _dirs, files in os.walk(test_data_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".csv"):
                continue
            csv_path = os.path.join(root, fn)
            csv_name = fn
            csv_stem = os.path.splitext(fn)[0]

            # 산업명: 상위 폴더명
            industry = os.path.basename(os.path.dirname(csv_path))

            # 공정명 추정: 파일명 토큰화
            toks = csv_stem.split("_")
            process_guess = toks[1].title() if len(toks) >= 2 else "Unknown"

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logging.error("CSV 로딩 실패: %s (%s)", csv_path, e)
                continue

            # 라인 컬럼 자동 추론
            line_col = _infer_line_col(df.columns)
            if line_col is None:
                logging.warning("라인 식별 컬럼을 찾지 못해 스킵: %s", csv_name)
                continue

            # 메트릭 컬럼 = 전체 - {TIMESTAMP, line_col, 상태 후보}
            exclude_cols = {"TIMESTAMP", line_col, *STATE_COL_CANDIDATES}
            metric_cols = [c for c in df.columns if c not in exclude_cols]

            datasets.append({
                "key": csv_name,
                "industry": industry.title(),
                "process": process_guess,
                "line": line_col,
                "metric_cols": metric_cols,
                "data": df.reset_index(drop=True),
                "csv_path": csv_path,
                "csv_name": csv_name,
                "csv_stem": csv_stem,
            })

    if not datasets:
        logging.warning("로딩된 CSV 데이터셋이 없습니다: %s", test_data_dir)

    return datasets


def send_dashboard_data(
    test_data_dir: str,
    interval: float = 10.0,
    shorter_strategy: Literal["skip", "repeat", "cycle"] = "skip",
    use_csv_timestamp: bool = True,
    anomaly_providers: Optional[Dict[str, AnomalyFn]] = None,
    state_resolvers: Optional[Dict[str, StateFn]] = None,
) -> Iterator[str]:
    """대시보드용 JSON 프레임을 주기적으로 생성하는 제너레이터."""
    datasets = _iter_csv_datasets(test_data_dir)
    if not datasets:
        return

    anomaly_providers = anomaly_providers or {}
    state_resolvers = state_resolvers or {}

    lengths = [len(d["data"]) for d in datasets]
    if not lengths:
        return
    max_len = max(lengths)

    for i in range(max_len):
        frame: List[Dict] = []
        for d in datasets:
            df: pd.DataFrame = d["data"]
            n = len(df)

            # 인덱스 결정
            if i < n:
                idx = i
            else:
                if shorter_strategy == "skip":
                    continue
                elif shorter_strategy == "repeat":
                    idx = n - 1
                elif shorter_strategy == "cycle":
                    idx = i % n
                else:
                    continue

            row = df.iloc[idx]

            # 라인 식별 값
            line_value = _to_python_scalar(row[d["line"]])

            # 메트릭 딕셔너리 (NaN→None, NumPy→파이썬 스칼라)
            metrics = {c: _to_python_scalar(row[c]) for c in d["metric_cols"]}

            # 공정 상태
            state_fn = state_resolvers.get(d["key"])  # 없으면 None
            state = _resolve_state(row, state_fn, list(df.columns))

            # 이상치 평가
            anomaly_fn = anomaly_providers.get(d["key"])  # 없으면 None
            anomaly_payload = None
            if anomaly_fn is not None:
                try:
                    result = anomaly_fn(row)
                    anomaly_payload = {
                        "model": str(result.get("model", "unknown")),
                        "score": float(result.get("score", 0.0)),
                        "threshold": float(result.get("threshold", 0.0)),
                        "is_anomaly": bool(result.get("is_anomaly", False)),
                        "details": result.get("details", {}),
                    }
                except Exception as e:
                    logging.error("anomaly_fn 오류(key=%s): %s", d["key"], e)
                    anomaly_payload = {
                        "model": "error",
                        "score": 0.0,
                        "threshold": 0.0,
                        "is_anomaly": False,
                        "details": {"error": str(e)},
                    }

            # 타임스탬프 결정
            if use_csv_timestamp and "TIMESTAMP" in df.columns:
                ts_raw = row["TIMESTAMP"]
                if pd.isna(ts_raw):
                    ts_out = datetime.now(timezone.utc).isoformat()
                else:
                    try:
                        ts_out = pd.to_datetime(ts_raw, utc=True).isoformat()
                    except Exception:
                        ts_out = datetime.now(timezone.utc).isoformat()
            else:
                ts_out = datetime.now(timezone.utc).isoformat()

            payload = {
                "timestamp": ts_out,
                "industry": d["industry"],
                "process": d["process"],
                "line": line_value,
                "state": state,
                "anomaly": anomaly_payload,
                "metrics": metrics,
            }
            frame.append(payload)

        if not frame:
            continue

        yield json.dumps(frame, ensure_ascii=False, indent=2)
        time.sleep(interval)


# -----------------------------
# 예시 콜백
# -----------------------------
def example_anomaly_fn_factory(model_name: str, threshold: float = 0.8) -> AnomalyFn:
    def _fn(row: pd.Series) -> Dict[str, object]:
        temp = float(row.get("TEMP", 0) or 0)
        press = float(row.get("PRESSURE", 0) or 0)
        score = min(1.0, (abs(temp - 400) / 400) * 0.6 + (abs(press - 1.0)) * 0.4)
        return {
            "model": model_name,
            "score": score,
            "threshold": threshold,
            "is_anomaly": bool(score >= threshold),
            "details": {"used_cols": ["TEMP", "PRESSURE"]},
        }
    return _fn


def example_state_fn(row: pd.Series) -> str:
    for col in STATE_COL_CANDIDATES:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    rpm = float(row.get("RPM", 0) or 0)
    return "RUNNING" if rpm > 0 else "IDLE"


def stream_to_files(
    test_data_dir: str,
    output_dir: str,
    interval: float = 10.0,
    shorter_strategy: Literal["skip", "repeat", "cycle"] = "skip",
    use_csv_timestamp: bool = True,
    anomaly_providers: Optional[Dict[str, AnomalyFn]] = None,
    state_resolvers: Optional[Dict[str, StateFn]] = None,
    start_index: int = 0,
    max_frames: Optional[int] = None,
    print_progress: bool = True,
) -> None:
    """send_dashboard_data를 파일 저장 모드로 실행한다."""
    os.makedirs(output_dir, exist_ok=True)
    gen = send_dashboard_data(
        test_data_dir=test_data_dir,
        interval=interval,
        shorter_strategy=shorter_strategy,
        use_csv_timestamp=use_csv_timestamp,
        anomaly_providers=anomaly_providers,
        state_resolvers=state_resolvers,
    )

    idx = start_index
    saved = 0
    for json_data in gen:
        filename = os.path.join(output_dir, f"frame_{idx:04d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_data)
        if print_progress:
            print(f"✅ saved {filename}")
        idx += 1
        saved += 1
        if max_frames is not None and saved >= max_frames:
            break


# -----------------------------
# 실제 모델 로딩/연결: Keras Autoencoder (+ Scaler + Metadata)
# -----------------------------
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# === Device selection / fallback (GPU→CPU) ===
_TF_DEVICE_PREF = os.environ.get("PRISM_DEVICE", "auto").lower()  # auto|cpu|gpu
_TF_HAS_GPU = False
_TF_GPU_USABLE = False  # GPU가 실제로 사용 가능한지 (CuDNN 체크 포함)
_TF_FALLBACK_ONCE = False  # GPU 실패 시 최초 1회 경고만

def _tf_init_devices(device_pref: str = "auto"):
    """TensorFlow 디바이스 선호 설정 및 가용 GPU 탐지 + 실사용 가능 여부 테스트."""
    global _TF_HAS_GPU, _TF_GPU_USABLE, _TF_DEVICE_PREF
    _TF_DEVICE_PREF = device_pref
    if tf is None:
        return
    gpus = tf.config.list_physical_devices("GPU")
    _TF_HAS_GPU = bool(gpus)

    if device_pref == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("TensorFlow: GPU 비활성화(강제 CPU)")
        except Exception:
            pass
        _TF_GPU_USABLE = False
    elif device_pref == "gpu":
        if not _TF_HAS_GPU:
            logging.warning("GPU 강제 모드지만 사용 가능한 GPU가 없습니다. CPU로 진행합니다.")
            _TF_GPU_USABLE = False
        else:
            _TF_GPU_USABLE = _test_gpu_usability()
    else:
        # auto: GPU 있으면 실제 사용 가능 여부 테스트
        if _TF_HAS_GPU:
            _TF_GPU_USABLE = _test_gpu_usability()
            if not _TF_GPU_USABLE:
                logging.warning("GPU가 감지되었지만 사용할 수 없습니다(CuDNN 문제 등). CPU 모드로 실행합니다.")
        else:
            _TF_GPU_USABLE = False

def _test_gpu_usability():
    """간단한 Keras 모델로 GPU가 실제 사용 가능한지 테스트 (CuDNN 포함)."""
    try:
        with tf.device("/GPU:0"):
            # Dense layer를 포함한 간단한 모델 생성 (CuDNN 초기화 필요)
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
                tf.keras.layers.Dense(1)
            ])
            # 더미 데이터로 예측 수행 (실제 CuDNN 사용)
            dummy_input = tf.constant([[1.0, 2.0]])
            _ = model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        msg = str(e)
        if "DNN" in msg or "cuDNN" in msg or "CUDA" in msg:
            logging.info("GPU 사용 불가 감지: %s", msg[:200])
        return False

def _predict_with_fallback(model, Xs):
    """디바이스 설정에 따라 예측 수행. 모델이 로드된 디바이스와 동일한 디바이스 사용."""
    if tf is None:
        return model.predict(Xs, verbose=0)

    # 실제 사용 가능한 디바이스에서 예측
    predict_device = "/GPU:0" if _TF_GPU_USABLE else "/CPU:0"

    with tf.device(predict_device):
        return model.predict(Xs, verbose=0)


# feature_cols_override 지원 + compile=False + GPU→CPU 폴백 사용
def make_keras_autoencoder_anomaly_fn(
    model_dir: str,
    feature_cols_override: Optional[List[str]] = None,
) -> AnomalyFn:
    """Keras .h5 오토인코더 + 전처리 스케일러 + 메타데이터 기반 이상치 콜백 생성."""
    if tf is None:
        raise RuntimeError("TensorFlow가 설치되어 있지 않습니다. (keras 모델 로딩에 필요)")
    if joblib is None:
        raise RuntimeError("joblib이 설치되어 있지 않습니다. (scaler 로딩에 필요)")

    model_path = os.path.join(model_dir, "autoencoder_model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일이 없습니다: {scaler_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"메타데이터 파일이 없습니다: {meta_path}")

    # 실제 사용 가능한 디바이스에 모델 로드
    load_device = "/GPU:0" if _TF_GPU_USABLE else "/CPU:0"

    # 명시적으로 디바이스 컨텍스트에서 모델 로드
    with tf.device(load_device):
        model = tf.keras.models.load_model(model_path, compile=False)

    scaler = joblib.load(scaler_path)
    meta = _load_json(meta_path)

    # feature_cols 우선순위: override → meta → scaler.feature_names_in_
    feature_cols: List[str] = []
    if feature_cols_override:
        feature_cols = list(feature_cols_override)
        logging.warning("[폴백] feature_cols를 CSV metric_cols로 사용합니다: %s", feature_cols)
    else:
        feature_cols = list(meta.get("feature_cols", []))  # type: ignore
        if not feature_cols and hasattr(scaler, "feature_names_in_"):
            try:
                feature_cols = [str(c) for c in scaler.feature_names_in_]  # type: ignore
                logging.info("[폴백] 메타 없음 → scaler.feature_names_in_ 사용: %s", feature_cols)
            except Exception:
                feature_cols = []

    # 엄격 모드(선택): 폴백 금지 가능
    strict = (os.environ.get("PRISM_STRICT_FEATURE_COLS", "0") == "1")
    if not feature_cols:
        msg = ("model_metadata.json 에 'feature_cols'가 필요합니다. "
               "또는 scaler.feature_names_in_ / CSV metric_cols 폴백이 있어야 합니다.")
        if strict:
            raise ValueError(msg)
        raise ValueError(msg)

    threshold: float = float(meta.get("threshold", 0.0))  # type: ignore
    error_metric: str = str(meta.get("error_metric", "mse")).lower()  # "mse" or "mae"
    model_name: str = str(meta.get("model_name", os.path.basename(model_dir)))

    def _fn(row: pd.Series) -> Dict[str, object]:
        # 1) 특징 추출 (결측치는 0으로 대체)
        x = [float(row.get(c, 0) or 0) for c in feature_cols]
        X = np.array(x, dtype=float).reshape(1, -1)

        # 2) 스케일링
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = scaler.transform(np.asarray(X).reshape(1, -1))

        # 3) 재구성 및 에러 계산 (GPU 우선, 실패 시 CPU 폴백)
        Xh = _predict_with_fallback(model, Xs)
        diff = Xs - Xh
        if error_metric == "mae":
            err = float(np.mean(np.abs(diff)))
        else:  # default mse
            err = float(np.mean(diff ** 2))

        is_anom = bool(err >= threshold)
        return {
            "model": model_name,
            "score": err,            # reconstruction error
            "threshold": threshold,
            "is_anomaly": is_anom,
            "details": {
                "error_metric": error_metric,
                "features": feature_cols,
            },
        }

    return _fn


def _match_model_to_dataset_key(meta: Dict[str, object], dirname: str) -> Optional[str]:
    """메타데이터로 CSV 매칭 키 결정."""
    csv_name = meta.get("csv_name")
    if isinstance(csv_name, str) and csv_name.endswith(".csv"):
        return csv_name

    csv_glob = meta.get("csv_glob")
    if isinstance(csv_glob, str):
        return csv_glob  # 실제 적용은 나중에 glob 매칭으로 처리

    return f"{dirname}.csv"


# CSV metric_cols를 최후 폴백으로 주입
def build_anomaly_registry_from_root(models_root: str, datasets: List[Dict]) -> Dict[str, AnomalyFn]:
    """모델 루트 디렉터리 하위의 각 서브폴더를 탐색하여 anomaly_providers 생성."""
    registry: Dict[str, AnomalyFn] = {}
    if not os.path.isdir(models_root):
        logging.error("모델 루트 디렉터리를 찾을 수 없습니다: %s", models_root)
        return registry

    # 데이터셋 파일명 목록 및 인덱스
    csv_names = [d["csv_name"] for d in datasets]
    ds_by_csv = {d["csv_name"]: d for d in datasets}

    for dirname in sorted(os.listdir(models_root)):
        dirpath = os.path.join(models_root, dirname)
        if not os.path.isdir(dirpath):
            continue

        meta_path = os.path.join(dirpath, "model_metadata.json")
        if not os.path.exists(meta_path):
            logging.warning("model_metadata.json 없음: %s", dirpath)
            continue
        try:
            meta: Dict[str, object] = _load_json(meta_path)
        except Exception as e:
            logging.error("메타데이터 로딩 실패: %s (%s)", meta_path, e)
            continue

        key_or_glob = _match_model_to_dataset_key(meta, dirname)

        # glob 매칭
        matched_keys: List[str] = []
        if key_or_glob and "*" in key_or_glob:
            import fnmatch
            matched_keys = [name for name in csv_names if fnmatch.fnmatch(name, key_or_glob)]
        elif key_or_glob in csv_names:
            matched_keys = [key_or_glob]
        else:
            fallback = f"{dirname}.csv"
            if fallback in csv_names:
                matched_keys = [fallback]

        if not matched_keys:
            logging.warning("매칭되는 CSV가 없어 모델 스킵: %s", dirpath)
            continue

        # 폴백 판단: 메타 feature_cols 없음 + scaler에 feature_names_in_ 없음 → CSV metric_cols override
        feature_override: Optional[List[str]] = None
        has_feature_cols = bool(meta.get("feature_cols"))

        if not has_feature_cols:
            has_names_in = False
            if joblib is not None:
                try:
                    _scaler = joblib.load(os.path.join(dirpath, "scaler.pkl"))
                    has_names_in = hasattr(_scaler, "feature_names_in_")
                except Exception:
                    has_names_in = False
            if not has_names_in:
                # 첫 번째 매칭 CSV 기준으로 폴백
                ds0 = ds_by_csv.get(matched_keys[0])
                if ds0:
                    feature_override = list(ds0["metric_cols"])
                    logging.warning(
                        "[폴백] %s: feature_cols 누락 + scaler 이름정보 없음 → CSV(%s)의 metric_cols 사용: %s",
                        dirname, matched_keys[0], feature_override
                    )

        # 모델 콜백 생성 (필요시 override 전달)
        try:
            fn = make_keras_autoencoder_anomaly_fn(dirpath, feature_cols_override=feature_override)
        except Exception as e:
            logging.error("모델 로딩 실패: %s (%s)", dirpath, e)
            continue

        for k in matched_keys:
            if k in registry:
                logging.warning("중복 키 감지, 마지막으로 등록: %s", k)
            registry[k] = fn
            logging.info("등록 완료: %s → %s", k, dirpath)

    return registry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM dashboard stream (low-hardcode)")
    parser.add_argument("--test_data_dir", default=os.environ.get("PRISM_TEST_DATA_DIR") or DEFAULT_TEST_DATA_DIR)
    parser.add_argument("--models_root", default=os.environ.get("PRISM_MODELS_ROOT") or DEFAULT_MODELS_ROOT)
    parser.add_argument("--output_dir", default=os.environ.get("PRISM_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval", type=float, default=float(os.environ.get("PRISM_INTERVAL", 10.0)))
    parser.add_argument("--shorter_strategy", choices=["skip", "repeat", "cycle"],
                        default=os.environ.get("PRISM_SHORTER_STRATEGY", "skip"))
    # 기본값을 ENV로 제어(기본 ON)
    parser.add_argument("--use_csv_timestamp", action="store_true",
                        default=os.environ.get("PRISM_USE_CSV_TS", "1") != "0")
    parser.add_argument("--max_frames", type=int, default=None)
    # 디버그용 드라이런
    parser.add_argument("--dry_run", action="store_true")
    # 디바이스 선택 옵션 추가 (auto|cpu|gpu)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"],
                        default=os.environ.get("PRISM_DEVICE", "auto").lower(),
                        help="TensorFlow 실행 디바이스 선택 (auto|cpu|gpu). 기본 auto")
    args = parser.parse_args()

    test_data_dir = args.test_data_dir or DEFAULT_TEST_DATA_DIR
    models_root = args.models_root or ""
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    if not os.path.isdir(test_data_dir):
        logging.error("테스트 데이터 경로를 찾을 수 없습니다: %s", test_data_dir)
        raise SystemExit(1)

    if models_root and not os.path.isdir(models_root):
        logging.warning("모델 경로를 찾을 수 없어 이상치 모델 없이 실행합니다: %s", models_root)
        models_root = ""

    logging.info("Paths:\n  test_data_dir=%s\n  models_root=%s\n  output_dir=%s",
                 test_data_dir, models_root or "(disabled)", output_dir)

    # 디바이스 초기화 (모델 로딩/예측 전에)
    _tf_init_devices(args.device)

    datasets_probe = _iter_csv_datasets(test_data_dir)

    # 모델 레지스트리 자동 생성 (없으면 빈 dict)
    if models_root and (tf is None or joblib is None):
        logging.warning("TensorFlow 또는 joblib이 없어 모델 기반 이상치 평가를 건너뜁니다.")
        anomaly_models: Dict[str, AnomalyFn] = {}
    else:
        anomaly_models = build_anomaly_registry_from_root(models_root, datasets_probe) if models_root else {}

    # 상태 기본 규칙을 모든 CSV에 자동 등록 (컬럼 있으면 컬럼 우선)
    state_resolvers: Dict[str, StateFn] = {d["csv_name"]: example_state_fn for d in datasets_probe}

    # 드라이런: 첫 프레임만 출력 후 종료 (빠른 검증)
    if args.dry_run:
        gen = send_dashboard_data(
            test_data_dir=test_data_dir,
            interval=0.0,
            shorter_strategy=args.shorter_strategy,
            use_csv_timestamp=args.use_csv_timestamp,
            anomaly_providers=anomaly_models,
            state_resolvers=state_resolvers,
        )
        try:
            print(next(gen))
        except StopIteration:
            logging.warning("생성된 프레임이 없습니다. CSV를 확인하세요.")
        raise SystemExit(0)

    # 파일 저장 실행
    stream_to_files(
        test_data_dir=test_data_dir,
        output_dir=output_dir,
        interval=args.interval,
        shorter_strategy=args.shorter_strategy,
        use_csv_timestamp=args.use_csv_timestamp,
        anomaly_providers=anomaly_models,
        state_resolvers=state_resolvers,
        start_index=0,
        max_frames=args.max_frames,
        print_progress=True,
    )
