"""
ys_abnormality.py — 모듈형 유틸
- generate_abnormality_report(data, ...) : 이상치 설명 생성
- generate_action_checklist(data, ...)   : 간단한 확인 과업 목록 생성
"""

from typing import Dict, Any, Optional, List
from openai import OpenAI

# 기본 설정(필요시 호출 시 덮어쓰기 가능)
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_API_KEY = "EMPTY"      # vLLM/OpenAI 호환 서버 사용 시 "EMPTY"로 둬도 됨
DEFAULT_API_BASE: Optional[str] = None  # 예: "http://localhost:8001/v1"

def _build_system_prompt() -> str:
    return r"""
당신은 반도체 제조 공정 내 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 당신이 접근할 수 있는 제조공정 DB 종류와 컬럼 설명, 그리고 정상범위(규격)입니다.
아래의 정상범위와 사용자 입력 내의 data를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.
정상범위 내의 컬럼의 경우 설명을 생성하지 않습니다. 
아래의 예시를 참고하여 답변을 생성하고, 답변은 반드시 한국어로 대답하세요.

────────────────────────────────────────
[테이블 및 정상범위]

1. SEMI_LOT_MANAGE (LOT 단위 생산 이력)
- PNO (PK): LOT 관리 고유번호
- LOT_NO: LOT 번호
- PRODUCT_NAME: 제품명
- RECIPE_ID: 사용 레시피 ID
- START_QTY: 시작 웨이퍼 수
- CURRENT_STEP: 현재 공정 단계
- PRIORITY: 우선순위(HOT/NORMAL)
- CREDATE: LOT 생성일
- HOLDER: 보류 사유(이상 발생 시)
- FINAL_YIELD: 최종 수율(%)
- GOOD_DIE / TOTAL_DIE: 양품·전체 다이 수
2. SEMI_PROCESS_HISTORY (공정별 이력)
- PNO (PK)
- LOT_NO
- PROCESS_STEP
- EQUIPMENT_ID
- RECIPE_ID
- OPERATOR
- START_TIME / END_TIME
- IN_QTY / OUT_QTY
3. SEMI_PARAM_MEASURE (공정 파라미터 측정)
- PNO (PK)
- LOT_NO, WAFER_ID
- PROCESS_STEP, EQUIPMENT_ID
- CATEGORY (예: CD, THICKNESS)
- PARAM_NAME
- UNIT
- MEASURED_VAL
- TARGET_VAL
- USL / LSL: 상·하한 규격
- MEASURE_TIME
4. SEMI_EQUIPMENT_SENSOR (장비 센서 데이터)
- PNO (PK)
- EQUIPMENT_ID, LOT_NO
- SENSOR_TYPE: TEMP/PRESSURE/FLOW 등
- SENSOR_VALUE
- TIMESTAMP
- CHAMBER_ID
- RECIPE_STEP
5. SEMI_PHOTO_SENSORS (포토리소그래피)
- EXPOSURE_DOSE (mJ/cm²) 정상범위: 20–40
- FOCUS_POSITION (nm) 정상범위: ±50
- STAGE_TEMP (°C) 정상범위: 23±0.1
- BAROMETRIC_PRESSURE (hPa) 정상범위: 없음(모니터링용)
- HUMIDITY (%) 정상범위: 45±5
- ALIGNMENT_ERROR_X (nm) 정상범위: <3
- ALIGNMENT_ERROR_Y (nm) 정상범위: <3
- LENS_ABERRATION (mλ) 정상범위: <5
- ILLUMINATION_UNIFORMITY (%) 정상범위: >98
- RETICLE_TEMP (°C) 정상범위: 23±0.05
6. SEMI_ETCH_SENSORS (에칭)
- RF_POWER_SOURCE (W) 정상범위: 500–2000
- RF_POWER_BIAS (W) 정상범위: 50–500
- CHAMBER_PRESSURE (mTorr) 정상범위: 5–200
- GAS_FLOW_CF4 (sccm) 정상범위: 0–200
- GAS_FLOW_O2 (sccm) 정상범위: 0–100
- GAS_FLOW_AR (sccm) 정상범위: 0–500
- GAS_FLOW_CL2 (sccm) 정상범위: 0–200
- ELECTRODE_TEMP (°C) 정상범위: 40–80
- CHAMBER_WALL_TEMP (°C) 정상범위: 60–80
- HELIUM_PRESSURE (Torr) 정상범위: 5–20
- ENDPOINT_SIGNAL (a.u.) 정상범위: 없음
- PLASMA_DENSITY (ions/cm³) 정상범위: 1e10–1e12
7. SEMI_CVD_SENSORS (CVD)
- SUSCEPTOR_TEMP (°C) 정상범위: 300–700
- CHAMBER_PRESSURE (Torr) 정상범위: 0.1–760
- PRECURSOR_FLOW_TEOS (sccm) 정상범위: 0–500
- PRECURSOR_FLOW_SILANE (sccm) 정상범위: 0–1000
- PRECURSOR_FLOW_WF6 (sccm) 정상범위: 0–100
- CARRIER_GAS_N2 (slm) 정상범위: 0–20
- CARRIER_GAS_H2 (slm) 정상범위: 0–10
- SHOWERHEAD_TEMP (°C) 정상범위: 150–250
- LINER_TEMP (°C) 정상범위: 100–200
- DEPOSITION_RATE (Å/min) 정상범위: 없음
- FILM_STRESS (MPa) 정상범위: 없음
8. SEMI_IMPLANT_SENSORS (이온주입)
- BEAM_CURRENT (μA) 정상범위: 0.1–5000
- BEAM_ENERGY (keV) 정상범위: 0.2–3000
- DOSE_RATE (ions/cm²/s) 정상범위: 없음
- TOTAL_DOSE (ions/cm²) 정상범위: 1e11–1e16
- IMPLANT_ANGLE (°) 정상범위: 0–45
- WAFER_ROTATION (rpm) 정상범위: 0–1200
- SOURCE_PRESSURE (Torr) 정상범위: 1e-6–1e-4
- ANALYZER_PRESSURE (Torr) 정상범위: 1e-7–1e-5
- END_STATION_PRESSURE (Torr) 정상범위: 1e-7–1e-6
- BEAM_UNIFORMITY (%) 정상범위: >98
- FARADAY_CUP_CURRENT (μA) 정상범위: 없음
9. SEMI_CMP_SENSORS (CMP)
- HEAD_PRESSURE (psi) 정상범위: 2–8
- RETAINER_PRESSURE (psi) 정상범위: 2–6
- PLATEN_ROTATION (rpm) 정상범위: 20–150
- HEAD_ROTATION (rpm) 정상범위: 20–150
- SLURRY_FLOW_RATE (ml/min) 정상범위: 100–300
- SLURRY_TEMP (°C) 정상범위: 20–25
- PAD_TEMP (°C) 정상범위: 30–50
- REMOVAL_RATE (Å/min) 정상범위: 없음
- MOTOR_CURRENT (A) 정상범위: 없음
- CONDITIONER_PRESSURE (lbs) 정상범위: 5–9
- ENDPOINT_SIGNAL (a.u.) 정상범위: 없음
10. SEMI_SENSOR_ALERT_CONFIG
- PARAM_NAME
- WARNING_UPPER / WARNING_LOWER
- ALARM_UPPER / ALARM_LOWER
- INTERLOCK_UPPER / INTERLOCK_LOWER
- MOVING_AVG_WINDOW
- ALERT_TYPE (INSTANT/AVERAGE/TREND)
- ENABLED
""".strip()

def _client(api_key: str, api_base: Optional[str]) -> OpenAI:
    # base_url이 None이면 OpenAI 기본 엔드포인트 사용
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)

def generate_abnormality_report(
    input_data: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    api_base: Optional[str] = DEFAULT_API_BASE,
    temperature: float = 0.2,
) -> str:
    """
    제조공정 데이터(dict)를 받아 정상범위에서 벗어난 항목에 대해서만
    한국어 설명을 생성합니다.
    """
    client = _client(api_key, api_base)
    sys_prompt = _build_system_prompt()

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": str(input_data)}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    # OpenAI SDK v1 스타일
    try:
        return resp.choices[0].message.content
    except Exception:
        # 호환 서버의 경우 dict 접근 방식이 다를 수 있어 보강
        return getattr(resp.choices[0].message, "content", str(resp))

def generate_action_checklist(
    input_data: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    api_base: Optional[str] = DEFAULT_API_BASE,
    temperature: float = 0.1,
) -> str:
    """
    간단한 '확인 과업' 목록을 생성합니다.
    형식 예)
    - PHO_001 포토 장비의 정렬 상태(ALIGNMENT_ERROR_X/Y) 재점검
    - 노광량(EXPOSURE_DOSE) 캘리브레이션 수행 여부 확인
    - 레시피 RECIPE_ID 파라미터 변경 이력 확인
    
    규칙:
    - 과도한 장황함 없이 5개 내외의 불릿으로
    - '무슨 무슨 장비에 대한 확인이 필요합니다' 톤의 자연어
    - 입력 데이터의 키를 활용(가능하면 장비/레시피/스텝/센서명)
    """
    checklist_prompt = (
        "아래 제조공정 데이터를 참고하여, 문제가 생겼을 가능성이 있는 설비/파라미터에 대해 "
        "간단한 확인 과업 목록을 한국어 불릿 리스트로 3~6개 생성하세요. "
        "항목마다 한 줄, 지시형 문장으로 쓰되, 불필요한 설명은 생략하세요.\n\n"
        f"[입력 데이터]\n{input_data}"
    ).format(input_data=json.dumps(input_data, ensure_ascii=False, indent=2))

    client = _client(api_key, api_base)
    messages = [
        {"role": "system", "content": "당신은 반도체 공정 모니터링/운영 보조 에이전트입니다. 사용자의 요청에 맞춰 간결한 점검 과업 목록을 작성합니다."},
        {"role": "user", "content": checklist_prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return getattr(resp.choices[0].message, "content", str(resp))

__all__ = [
    "generate_abnormality_report",
    "generate_action_checklist",
]
