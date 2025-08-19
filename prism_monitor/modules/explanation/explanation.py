from openai import OpenAI

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
FEWSHOT_USER = r"""
data:
{"PNO":"PS001","EQUIPMENT_ID":"PHO_001","LOT_NO":"LOT24001A","WAFER_ID":"W001","TIMESTAMP":"2024-01-15 08:30:15","EXPOSURE_DOSE":45.0.5,"FOCUS_POSITION":60.0,"STAGE_TEMP":23.3,"BAROMETRIC_PRESSURE":800.0,"HUMIDITY":60.0,"ALIGNMENT_ERROR_X":5.0,"ALIGNMENT_ERROR_Y":3.5,"LENS_ABERRATION":7.0,"ILLUMINATION_UNIFORMITY":95.0,"RETICLE_TEMP":23.2}

answer:
"""

FEWSHOT_EXPLANATION = (
    """
    2024-01-15 08:30:15에 PHO_001 장비에서 노광량 45.0mJ/cm²(정상 20~40), 
    초점 위치 60.0nm(정상 ±50), 스테이지 온도 23.3°C(정상 22.9~23.1°C),
    습도 60.0%(정상 40~50%), 얼라인먼트 오차 X 5.0nm(정상 <3),
    얼라인먼트 오차 Y 3.5nm(정상 <3), 렌즈 수차 7.0mλ(정상 <5),
    레티클 온도 23.2°C(정상 22.95~23.05°C)가 각각 정상 범위를 초과했습니다.
    조명 균일도 95.0%(정상 >98)는 정상 범위에 미달했습니다.
    """
)

FEWSHOT_TASK = (
    """
    - PHO_001 장비의 노광량 설정을 확인하세요. 
    - PHO_001 장비의 포커스 위치를 점검하세요.
    - PHO_001 장비의 스테이지 온도를 점검하세요. 
    - PHO_001 장비의 기압 센서를 확인하세요.
    - PHO_001 장비의 습도 조절 시스템을 점검하세요.
    - PHO_001 장비의 정렬 오차(X/Y)를 확인하세요.
    - PHO_001 장비의 렌즈 수차를 점검하세요.
    - PHO_001 장비의 조명 균일도를 확인하세요.
    - PHO_001 장비의 레티클 온도를 점검하세요.
    """
)

def _call_api(system_prompt, fewshot_assistant, data):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://0.0.0.0:8001/v1"
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": fewshot_assistant},
            {"role": "user", "content": f"data:\n{data}\n\nanswer:\n"},
        ],
        max_tokens=8192,
        temperature=0.8,
        presence_penalty=0.0,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content

def explain(data):
    """이상치 값에 대한 설명을 생성"""
    # return _call_api(EXPLANATION_SYSTEM_PROMPT, FEWSHOT_EXPLANATION, data)
    import time
    time.sleep(5)
    return (
            "최근 7일간 설비 A의 온도 센서에서 정상 범위(60~80℃)를 벗어나는 값이 5회 이상 "
            "감지되었습니다. 해당 시점은 모두 야간 교대 시간대에 발생했으며, "
            "진동 수치 또한 평균 대비 20% 이상 높게 나타났습니다. "
            "이러한 패턴은 베어링 마모 또는 냉각 장치 효율 저하 가능성을 시사합니다."
        )

def cause_candidates(data):
    """이상치 분석을 위한 과업을 생성"""
    # return _call_api(TASK_SYSTEM_PROMPT, FEWSHOT_TASK, data)
    import time
    time.sleep(5)
    return (
            "- 야간 근무 시 냉각 장치 가동 불안정\n"
            "- 베어링 마모로 인한 과열 현상\n"
            "- 센서 자체 오작동(교정 필요)\n"
            "- 외부 환경 온도 상승(환기 불충분)"
        )

def explanation():
    test_data = '{"PNO":"PS024","EQUIPMENT_ID":"PHO_003","LOT_NO":"LOT24009A","WAFER_ID":"W001","TIMESTAMP":"2024-01-23 08:15:20","EXPOSURE_DOSE":41.2,"FOCUS_POSITION":-25.3,"STAGE_TEMP":23.02,"BAROMETRIC_PRESSURE":1014.1,"HUMIDITY":54.3,"ALIGNMENT_ERROR_X":2.6,"ALIGNMENT_ERROR_Y":2.8,"LENS_ABERRATION":4.5,"ILLUMINATION_UNIFORMITY":97.8,"RETICLE_TEMP":23.06}'
    
    # print("=== 이상치 설명 ===")
    # explain = event_explain(test_data)
    
    # print("\n=== 분석 과업 ===")
    # cause_candidates = event_cause_candidates(test_data)
    # # return {
    # #     'explain': explain,
    # #     'causeCandidates': cause_candidates
    # # }

    return {
        'explain': (
            "최근 7일간 설비 A의 온도 센서에서 정상 범위(60~80℃)를 벗어나는 값이 5회 이상 "
            "감지되었습니다. 해당 시점은 모두 야간 교대 시간대에 발생했으며, "
            "진동 수치 또한 평균 대비 20% 이상 높게 나타났습니다. "
            "이러한 패턴은 베어링 마모 또는 냉각 장치 효율 저하 가능성을 시사합니다."
        ),
        'causeCandidates': (
            "- 야간 근무 시 냉각 장치 가동 불안정\n"
            "- 베어링 마모로 인한 과열 현상\n"
            "- 센서 자체 오작동(교정 필요)\n"
            "- 외부 환경 온도 상승(환기 불충분)"
        )
    }
