"""
PRISM Dashboard - Core Utilities
대시보드 API를 위한 핵심 유틸리티 함수 모음
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Sequence

import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# 상태 및 라인 컬럼 우선순위
STATE_COL_CANDIDATES = ("STATE", "STATUS", "PROCESS_STATE")
LINE_COL_PRIORITY = (
    "EQUIPMENT_ID", "PROCESS_ID", "PRODUCTION_LINE",
    "STATION_ID", "CHAMBER_ID", "PRESS_ID", "SENSOR_ID",
    "LINE_ID", "CELL_ID", "BOOTH_ID", "CASTER_ID", "CONVERTER_ID",
)

# 기본 경로
MODULE_DIR = Path(__file__).resolve().parent
MODULES_ROOT = MODULE_DIR.parent
PRISM_ROOT = MODULES_ROOT.parent
REPO_ROOT = PRISM_ROOT.parent
DEFAULT_TEST_DATA_DIR = str(PRISM_ROOT / "test-scenarios" / "test_data")
DEFAULT_MODELS_ROOT = str(REPO_ROOT / "models")

# 타입 힌트
AnomalyFn = Callable[[pd.Series], Dict[str, object]]
StateFn = Callable[[pd.Series], str]


# ===========================
# CSV 데이터 로딩
# ===========================

def _infer_line_col(columns: Sequence[str]) -> Optional[str]:
    """라인 식별 컬럼을 우선순위에 따라 추론"""
    for c in LINE_COL_PRIORITY:
        if c in columns:
            return c
    for c in columns:
        if str(c).upper().endswith("_ID"):
            return c
    return None


def _iter_csv_datasets(test_data_dir: str) -> List[Dict]:
    """CSV 파일들을 자동 스캔하여 메타데이터와 함께 로드"""
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

            # 공정명: 파일명에서 추정
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


# ===========================
# 데이터 변환 유틸리티
# ===========================

def _to_python_scalar(x):
    """NumPy 스칼라를 파이썬 기본형으로, NaN은 None으로 변환"""
    if pd.isna(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def _resolve_state(row: pd.Series, resolver: Optional[StateFn], df_columns: List[str]) -> Optional[str]:
    """공정 상태 문자열을 결정 (콜백 우선, 없으면 컬럼에서 추출)"""
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


def default_state_fn(row: pd.Series) -> str:
    """기본 상태 결정 함수 (STATE 컬럼 우선, 없으면 RPM 기반)"""
    for col in STATE_COL_CANDIDATES:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    rpm = float(row.get("RPM", 0) or 0)
    return "RUNNING" if rpm > 0 else "IDLE"


# ===========================
# TensorFlow 모델 로딩
# ===========================

try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


def _load_json(path: str) -> Dict[str, object]:
    """JSON 파일 로딩"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# GPU/CPU 디바이스 관리
_TF_HAS_GPU = False
_TF_GPU_USABLE = False


def _tf_init_devices(device_pref: str = "auto"):
    """TensorFlow 디바이스 초기화 및 GPU 사용 가능 여부 테스트"""
    global _TF_HAS_GPU, _TF_GPU_USABLE

    if tf is None:
        return

    gpus = tf.config.list_physical_devices("GPU")
    _TF_HAS_GPU = bool(gpus)

    if device_pref == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("TensorFlow: GPU 비활성화 (강제 CPU)")
        except Exception:
            pass
        _TF_GPU_USABLE = False
    elif device_pref == "gpu":
        if not _TF_HAS_GPU:
            logging.warning("GPU 강제 모드지만 사용 가능한 GPU가 없습니다. CPU로 진행합니다.")
            _TF_GPU_USABLE = False
        else:
            _TF_GPU_USABLE = _test_gpu_usability()
    else:  # auto
        if _TF_HAS_GPU:
            _TF_GPU_USABLE = _test_gpu_usability()
            if not _TF_GPU_USABLE:
                logging.warning("GPU가 감지되었지만 사용할 수 없습니다. CPU 모드로 실행합니다.")
        else:
            _TF_GPU_USABLE = False


def _test_gpu_usability():
    """GPU가 실제 사용 가능한지 테스트 (CuDNN 포함)"""
    try:
        with tf.device("/GPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
                tf.keras.layers.Dense(1)
            ])
            dummy_input = tf.constant([[1.0, 2.0]])
            _ = model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        msg = str(e)
        if "DNN" in msg or "cuDNN" in msg or "CUDA" in msg:
            logging.info("GPU 사용 불가 감지: %s", msg[:200])
        return False


def _predict_with_fallback(model, Xs):
    """디바이스 설정에 따라 예측 수행"""
    if tf is None:
        return model.predict(Xs, verbose=0)

    predict_device = "/GPU:0" if _TF_GPU_USABLE else "/CPU:0"
    with tf.device(predict_device):
        return model.predict(Xs, verbose=0)


# ===========================
# Keras Autoencoder 이상치 탐지
# ===========================

def make_keras_autoencoder_anomaly_fn(
    model_dir: str,
    feature_cols_override: Optional[List[str]] = None,
) -> AnomalyFn:
    """Keras Autoencoder 기반 이상치 탐지 함수 생성"""
    if tf is None:
        raise RuntimeError("TensorFlow가 설치되어 있지 않습니다")
    if joblib is None:
        raise RuntimeError("joblib이 설치되어 있지 않습니다")

    model_path = os.path.join(model_dir, "autoencoder_model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일이 없습니다: {scaler_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"메타데이터 파일이 없습니다: {meta_path}")

    # 모델 로드
    load_device = "/GPU:0" if _TF_GPU_USABLE else "/CPU:0"
    with tf.device(load_device):
        model = tf.keras.models.load_model(model_path, compile=False)

    scaler = joblib.load(scaler_path)
    meta = _load_json(meta_path)

    # feature_cols 결정 (override → meta → scaler)
    feature_cols: List[str] = []
    if feature_cols_override:
        feature_cols = list(feature_cols_override)
    else:
        feature_cols = list(meta.get("feature_cols", []))  # type: ignore
        if not feature_cols and hasattr(scaler, "feature_names_in_"):
            try:
                feature_cols = [str(c) for c in scaler.feature_names_in_]  # type: ignore
            except Exception:
                feature_cols = []

    if not feature_cols:
        raise ValueError("feature_cols를 찾을 수 없습니다 (meta/scaler/override 모두 없음)")

    threshold: float = float(meta.get("threshold", 0.0))  # type: ignore
    error_metric: str = str(meta.get("error_metric", "mse")).lower()
    model_name: str = str(meta.get("model_name", os.path.basename(model_dir)))

    def _fn(row: pd.Series) -> Dict[str, object]:
        # 특징 추출
        x = [float(row.get(c, 0) or 0) for c in feature_cols]
        X = np.array(x, dtype=float).reshape(1, -1)

        # 스케일링
        Xs = scaler.transform(X)

        # 재구성 및 에러 계산
        Xh = _predict_with_fallback(model, Xs)
        diff = Xs - Xh

        if error_metric == "mae":
            err = float(np.mean(np.abs(diff)))
        else:  # mse
            err = float(np.mean(diff ** 2))

        is_anom = bool(err >= threshold)
        return {
            "model": model_name,
            "score": err,
            "threshold": threshold,
            "is_anomaly": is_anom,
            "details": {
                "error_metric": error_metric,
                "features": feature_cols,
            },
        }

    return _fn


def _match_model_to_dataset_key(meta: Dict[str, object], dirname: str) -> Optional[str]:
    """메타데이터로 CSV 매칭 키 결정"""
    csv_name = meta.get("csv_name")
    if isinstance(csv_name, str) and csv_name.endswith(".csv"):
        return csv_name

    csv_glob = meta.get("csv_glob")
    if isinstance(csv_glob, str):
        return csv_glob

    return f"{dirname}.csv"


def build_anomaly_registry_from_root(models_root: str, datasets: List[Dict]) -> Dict[str, AnomalyFn]:
    """모델 루트 디렉터리에서 모든 모델을 로드하여 레지스트리 생성"""
    registry: Dict[str, AnomalyFn] = {}

    if not os.path.isdir(models_root):
        logging.error("모델 루트 디렉터리를 찾을 수 없습니다: %s", models_root)
        return registry

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

        # feature_cols 폴백 처리
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
                ds0 = ds_by_csv.get(matched_keys[0])
                if ds0:
                    feature_override = list(ds0["metric_cols"])
                    logging.warning(
                        "[폴백] %s: CSV(%s)의 metric_cols 사용",
                        dirname, matched_keys[0]
                    )

        # 모델 콜백 생성
        try:
            fn = make_keras_autoencoder_anomaly_fn(dirpath, feature_cols_override=feature_override)
        except Exception as e:
            logging.error("모델 로딩 실패: %s (%s)", dirpath, e)
            continue

        for k in matched_keys:
            if k in registry:
                logging.warning("중복 키 감지: %s", k)
            registry[k] = fn
            logging.info("등록 완료: %s → %s", k, dirpath)

    return registry
