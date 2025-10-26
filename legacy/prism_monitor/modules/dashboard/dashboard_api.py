"""
PRISM Dashboard API
외부 요청에 응답하는 FastAPI 기반 대시보드 서버
"""
from __future__ import annotations

import os
import random
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd

# dashboard.py의 유틸리티 함수들 import
from .dashboard import (
    _iter_csv_datasets,
    _to_python_scalar,
    _resolve_state,
    build_anomaly_registry_from_root,
    default_state_fn,
    _tf_init_devices,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_MODELS_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# ===========================
# 환경변수 기반 설정
# ===========================
TEST_DATA_DIR = os.environ.get("PRISM_TEST_DATA_DIR", DEFAULT_TEST_DATA_DIR)
MODELS_ROOT = os.environ.get("PRISM_MODELS_ROOT", DEFAULT_MODELS_ROOT)
DEVICE = os.environ.get("PRISM_DEVICE", "auto").lower()

# ===========================
# FastAPI 앱
# ===========================
app = FastAPI(
    title="PRISM Dashboard API",
    description="제조업 공정 모니터링 및 이상치 탐지 API",
    version="1.0.0",
)

# 전역 상태 (서버 시작 시 한 번만 로드)
datasets: List[Dict] = []
anomaly_models: Dict = {}
state_resolvers: Dict = {}


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 데이터 및 모델 초기화"""
    global datasets, anomaly_models, state_resolvers

    logging.info("=" * 60)
    logging.info("PRISM Dashboard API 시작")
    logging.info("=" * 60)
    logging.info("설정:")
    logging.info("  - 테스트 데이터: %s", TEST_DATA_DIR)
    logging.info("  - 모델 경로: %s", MODELS_ROOT)
    logging.info("  - 디바이스: %s", DEVICE)

    # TensorFlow 디바이스 초기화
    _tf_init_devices(DEVICE)

    # CSV 데이터셋 로드
    datasets = _iter_csv_datasets(TEST_DATA_DIR)
    logging.info("✅ %d개 데이터셋 로드 완료", len(datasets))

    if not datasets:
        logging.error("❌ 로드된 데이터셋이 없습니다!")
        return

    # 이상치 탐지 모델 로드
    if MODELS_ROOT and os.path.isdir(MODELS_ROOT):
        try:
            anomaly_models = build_anomaly_registry_from_root(MODELS_ROOT, datasets)
            logging.info("✅ %d개 이상치 모델 로드 완료", len(anomaly_models))
        except Exception as e:
            logging.error("❌ 모델 로드 실패: %s", e)
            anomaly_models = {}
    else:
        logging.warning("⚠️  모델 경로가 없어 이상치 탐지 비활성화")
        anomaly_models = {}

    # 상태 resolver 설정 (모든 CSV에 기본 함수 적용)
    state_resolvers = {d["csv_name"]: default_state_fn for d in datasets}

    logging.info("=" * 60)
    logging.info("✅ 서버 초기화 완료!")
    logging.info("=" * 60)


def build_payload(dataset: Dict, row: pd.Series) -> Dict:
    """단일 데이터셋의 한 행에 대해 JSON payload 생성"""

    # 라인 식별 값
    line_value = _to_python_scalar(row[dataset["line"]])

    # 메트릭 딕셔너리
    metrics = {c: _to_python_scalar(row[c]) for c in dataset["metric_cols"]}

    # 공정 상태
    state_fn = state_resolvers.get(dataset["csv_name"])
    df_columns = list(dataset["data"].columns)
    state = _resolve_state(row, state_fn, df_columns)

    # 이상치 평가
    anomaly_payload = None
    anomaly_fn = anomaly_models.get(dataset["csv_name"])

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
            logging.error("이상치 탐지 오류 (key=%s): %s", dataset["key"], e)
            anomaly_payload = {
                "model": "error",
                "score": 0.0,
                "threshold": 0.0,
                "is_anomaly": False,
                "details": {"error": str(e)},
            }

    # 타임스탬프 (현재 시각)
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "industry": dataset["industry"],
        "process": dataset["process"],
        "line": line_value,
        "state": state,
        "anomaly": anomaly_payload,
        "metrics": metrics,
    }


# ===========================
# API 엔드포인트
# ===========================

@app.get("/")
async def root():
    """헬스체크"""
    return {
        "status": "healthy",
        "service": "PRISM Dashboard API",
        "datasets_loaded": len(datasets),
        "models_loaded": len(anomaly_models),
    }


@app.get("/api/dashboard")
async def get_dashboard_data():
    """
    대시보드 데이터 조회 (10초마다 외부에서 호출)

    Returns:
        각 데이터셋의 랜덤 샘플 + 이상치 탐지 결과
    """
    if not datasets:
        raise HTTPException(status_code=503, detail="데이터셋이 로드되지 않았습니다")

    frame = []

    for d in datasets:
        df = d["data"]

        # 랜덤 행 선택
        idx = random.randint(0, len(df) - 1)
        row = df.iloc[idx]

        # payload 생성
        payload = build_payload(d, row)
        frame.append(payload)

    return frame


@app.get("/api/dashboard/{industry}")
async def get_dashboard_by_industry(industry: str):
    """
    특정 산업군의 대시보드 데이터만 조회

    Args:
        industry: 산업명 (예: semiconductor, steel)
    """
    filtered = [d for d in datasets if d["industry"].lower() == industry.lower()]

    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"'{industry}' 산업 데이터를 찾을 수 없습니다"
        )

    frame = []
    for d in filtered:
        df = d["data"]
        idx = random.randint(0, len(df) - 1)
        row = df.iloc[idx]
        payload = build_payload(d, row)
        frame.append(payload)

    return frame


@app.get("/api/info")
async def get_info():
    """API 정보 및 로드된 데이터셋 목록"""
    return {
        "datasets": [
            {
                "key": d["csv_name"],
                "industry": d["industry"],
                "process": d["process"],
                "line_column": d["line"],
                "metrics": d["metric_cols"],
                "rows": len(d["data"]),
                "has_model": d["csv_name"] in anomaly_models,
            }
            for d in datasets
        ],
        "total_datasets": len(datasets),
        "total_models": len(anomaly_models),
    }


# ===========================
# 서버 실행
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Dashboard API Server")
    parser.add_argument("--host", default="0.0.0.0", help="호스트 주소")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (자동 재시작)")
    args = parser.parse_args()

    uvicorn.run(
        "prism_monitor.modules.dashboard.dashboard_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
