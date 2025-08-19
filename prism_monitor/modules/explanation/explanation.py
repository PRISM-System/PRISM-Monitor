from openai import OpenAI
import json

# 이상치 설명 생성용 프롬프트
EXPLANATION_SYSTEM_PROMPT = r"""
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
"""

# 분석 과업 생성용 프롬프트
TASK_SYSTEM_PROMPT = r"""
당신은 반도체 제조 공정 내 탐지된 이상치를 분석하기 위한 과업을 생성하는 에이전트입니다.
아래는 당신이 접근할 수 있는 제조공정 DB 종류와 컬럼 설명입니다.
데이터 내 이상치가 있는 컬럼을 확인하여 이상치에 대한 분석을 생성하세요.
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
"""

# Few-shot 예시 데이터
# LOT/Param + Sensor 등 다중 데이터 활용의 예시
FEWSHOT_USER_CASE_MULTI_1 = r"""
data:
{"LOT_NO":"LOT30012A","PRODUCT_NAME":"DRAM_512","START_QTY":25,"CURRENT_STEP":"PHOTO","FINAL_YIELD":75.0,
 "EXPOSURE_DOSE":45.0,"FOCUS_POSITION":80.0,"STAGE_TEMP":23.4,"HUMIDITY":60.0}
answer:
"""
FEWSHOT_EXPLANATION_MULTI_1 = (
    "LOT30012A의 최종 수율이 75.0%로 정상 기준(>90%)보다 낮습니다.\n"
    "동일 시간대 PHOTO 공정에서 노광량 45.0mJ/cm²(정상 20–40), 초점 위치 80.0nm(정상 ±50), "
    "스테이지 온도 23.4°C(정상 22.9–23.1°C), 습도 60.0%(정상 40–50)가 정상 범위를 벗어났습니다.\n"
    "따라서, 이는 공정 조건 불량이 수율 저하의 주요 원인일 가능성이 높습니다."
)

FEWSHOT_TASK_MULTI_1 = (
    "- LOT30012A, PHOTO 장비의 노광 조건(노광량, 초점 위치)에 대한 예측 분석이 필요합니다.\n"
    "- PHOTO 장비의 스테이지 온도 및 습도 제어가 수율 저하와 어떤 상관관계가 있는지 분석이 필요합니다."
)

FEWSHOT_USER_CASE_MULTI_2 = r"""
data:
{"LOT_NO":"LOT30015B","WAFER_ID":"W012","CATEGORY":"THICKNESS","PARAM_NAME":"OXIDE_THK",
 "MEASURED_VAL":58.0,"TARGET_VAL":50.0,"USL":55.0,"LSL":45.0,
 "SUSCEPTOR_TEMP":750.0,"CHAMBER_PRESSURE":780.0,"PRECURSOR_FLOW_SILANE":1200.0}
answer:
"""
FEWSHOT_EXPLANATION_MULTI_2 = (
    "LOT30015B, WAFER W012의 산화막 두께가 58.0nm로 상한(55.0nm)을 초과했습니다.\n"
    "동일한 CVD 공정에서 서셉터 온도 750°C(정상 300–700), 챔버 압력 780Torr(정상 0.1–760), "
    "실란 유량 1200sccm(정상 0–1000)도 기준을 벗어났습니다.\n"
    "따라서, 이는 과도한 증착 조건이 두께 초과의 직접적 원인일 가능성이 높습니다."
)

FEWSHOT_TASK_MULTI_2 = (
    "- LOT30015B, CVD 장비의 서셉터 온도와 챔버 압력 이상이 두께 편차에 미치는 영향 분석이 필요합니다.\n"
    "- 실란 유량 과다 공급이 증착 조건 불안정에 어떤 영향을 주는지 예측 분석이 필요합니다."
)

# Sensor Only 등 단일 센서 데이터 활용의 예시
FEWSHOT_USER_CASE_SINGLE_1 = r"""
data:
{"RF_POWER_SOURCE":2200.0,"CHAMBER_PRESSURE":250.0,"CHAMBER_TEMP":85.0}
answer:
"""
FEWSHOT_EXPLANATION_SINGLE_1 = (
    "에칭 장비에서 RF Power Source 2200W(정상 500–2000), "
    "챔버 압력 250mTorr(정상 5–200), 챔버 온도 85°C(정상 40–80)가 기준을 벗어났습니다.\n"
    "따라서, 이는 플라즈마 과도 형성과 비정상 식각 속도의 원인일 가능성이 있습니다."
)

FEWSHOT_TASK_SINGLE_1 = (
    "- 에칭 장비의 RF Power Source 이상치가 플라즈마 안정성에 미치는 영향 분석이 필요합니다.\n"
    "- 챔버 압력 및 온도 편차가 식각 균일도에 어떤 영향을 주는지 예측이 필요합니다."
)

FEWSHOT_USER_CASE_SINGLE_2 = r"""
data:
{"HEAD_PRESSURE":9.5,"SLURRY_FLOW_RATE":350.0,"PAD_TEMP":55.0}
answer:
"""
FEWSHOT_EXPLANATION_SINGLE_2 = (
    "CMP 장비에서 헤드 압력 9.5psi(정상 2–8), 슬러리 유량 350ml/min(정상 100–300), "
    "패드 온도 55°C(정상 30–50)가 모두 정상 범위를 벗어났습니다.\n"
    "따라서, 이는 연마 불균일과 과도한 마모를 일으킬 가능성이 있습니다."
)

FEWSHOT_TASK_SINGLE_2 = (
    "- CMP 장비의 헤드 압력, 슬러리 유량, 패드 온도 이상치가 연마 균일도에 미치는 영향 분석이 필요합니다.\n"
    "- CMP 공정의 마모 패턴 변화에 대한 예측 분석이 요구됩니다."
)

def _call_api(system_prompt, fewshots, data):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    # few-shot 예시 추가
    for user_ex, assistant_ex in fewshots:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    # 실제 입력
    messages.append({"role": "user", "content": f"data:\n{data}\n\nanswer:\n"})
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_tokens=8192,
        temperature=0.8,
        presence_penalty=0.0,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


def event_explain(data):
    """이상치 값에 대한 설명을 생성"""
    try:
        parsed = json.loads(data)
    except Exception:
        parsed = {}
    
    # 데이터 행 개수 판단
    if isinstance(parsed, dict):
        row_count = 1
    elif isinstance(parsed, list):
        row_count = len(parsed)
    else:
        row_count = 1

    if row_count == 1:
        fewshots = [
            (FEWSHOT_USER_CASE_SINGLE_1, FEWSHOT_EXPLANATION_SINGLE_1),
            (FEWSHOT_USER_CASE_SINGLE_2, FEWSHOT_EXPLANATION_SINGLE_2),
        ]
    else:
        fewshots = [
            (FEWSHOT_USER_CASE_MULTI_1, FEWSHOT_EXPLANATION_MULTI_1),
            (FEWSHOT_USER_CASE_MULTI_2, FEWSHOT_EXPLANATION_MULTI_2),
        ]
    
    return _call_api(EXPLANATION_SYSTEM_PROMPT, fewshots, data)

def event_cause_candidates(data):
    """이상치 분석을 위한 과업을 생성"""
    try:
        parsed = json.loads(data)
    except Exception:
        parsed = {}
    row_count = 1 if isinstance(parsed, dict) else len(parsed)
    if row_count == 1:
        fewshots = [
            (FEWSHOT_USER_CASE_SINGLE_1, FEWSHOT_TASK_SINGLE_1),
            (FEWSHOT_USER_CASE_SINGLE_2, FEWSHOT_TASK_SINGLE_2),
        ]
    else:
        fewshots = [
            (FEWSHOT_USER_CASE_MULTI_1, FEWSHOT_TASK_MULTI_1),
            (FEWSHOT_USER_CASE_MULTI_2, FEWSHOT_TASK_MULTI_2),
        ]
    return _call_api(TASK_SYSTEM_PROMPT, fewshots, data)

if __name__ == "__main__":
    test_data_single = '{"RF_POWER_SOURCE":2200.0,"CHAMBER_PRESSURE":250.0,"CHAMBER_TEMP":85.0}'
    test_data_multi = '{"LOT_NO":"LOT30012A","PRODUCT_NAME":"DRAM_512","START_QTY":25,"CURRENT_STEP":"PHOTO","FINAL_YIELD":75.0,"EXPOSURE_DOSE":45.0,"FOCUS_POSITION":80.0,"STAGE_TEMP":23.4,"HUMIDITY":60.0}'
    
    print("=== 단일 행 입력 (설명) ===")
    print(event_explain(test_data_single))
    print("\n=== 단일 행 입력 (과업) ===")
    print(event_cause_candidates(test_data_single))
    
    print("\n=== 다중 속성 입력 (설명) ===")
    print(event_explain(test_data_multi))
    print("\n=== 다중 속성 입력 (과업) ===")
    print(event_cause_candidates(test_data_multi))