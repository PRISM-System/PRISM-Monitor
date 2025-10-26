from openai import OpenAI
import json

from prism_monitor.llm.api import llm_generate

# 🆕 NEW VERSION: 공정별 프롬프트 추가 (멀티 모델 지원)
# 새로운 데이터 형식(test-scenarios/test_data/semiconductor/)을 위한 프롬프트

# CMP 공정 전용 프롬프트
EXPLANATION_PROMPT_CMP = r"""
당신은 반도체 CMP(Chemical Mechanical Planarization) 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 CMP 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.
정상범위 내의 컬럼은 설명을 생성하지 않습니다.
이상치 값의 인과관계가 있다면 함께 설명하세요.

[CMP 센서 데이터 컬럼 및 정상범위]
- MOTOR_CURRENT (A): 모터 전류, 정상범위: 15–18
- SLURRY_FLOW_RATE (ml/min): 슬러리 유량, 정상범위: 200–300
- PRESSURE (psi): 가압력, 정상범위: 3–7
- HEAD_ROTATION (rpm): 헤드 회전수, 정상범위: 50–120
- PLATEN_ROTATION (rpm): 플래튼 회전수, 정상범위: 30–100
- PAD_TEMP (°C): 패드 온도, 정상범위: 35–50
- SLURRY_TEMP (°C): 슬러리 온도, 정상범위: 20–25
- REMOVAL_RATE (Å/min): 제거율, 정상범위: 1000–3000
- HEAD_PRESSURE (psi): 헤드 압력, 정상범위: 2–8
- RETAINER_PRESSURE (psi): 리테이너 압력, 정상범위: 2–6
- CONDITIONER_PRESSURE (lbs): 컨디셔너 압력, 정상범위: 5–9
- ENDPOINT_SIGNAL (a.u.): 종점 신호, 모니터링용

답변은 반드시 한국어로 작성하세요.
"""

# Etch 공정 전용 프롬프트
EXPLANATION_PROMPT_ETCH = r"""
당신은 반도체 Etch(식각) 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 Etch 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[Etch 센서 데이터 컬럼 및 정상범위]
- RF_POWER_SOURCE (W): RF 소스 파워, 정상범위: 500–2000
- RF_POWER_BIAS (W): RF 바이어스 파워, 정상범위: 50–500
- CHAMBER_PRESSURE (mTorr): 챔버 압력, 정상범위: 5–200
- GAS_FLOW_CF4 (sccm): CF4 가스 유량, 정상범위: 0–200
- GAS_FLOW_O2 (sccm): O2 가스 유량, 정상범위: 0–100
- GAS_FLOW_AR (sccm): Ar 가스 유량, 정상범위: 0–500
- GAS_FLOW_CL2 (sccm): Cl2 가스 유량, 정상범위: 0–200
- ELECTRODE_TEMP (°C): 전극 온도, 정상범위: 40–80
- CHAMBER_WALL_TEMP (°C): 챔버 벽 온도, 정상범위: 60–80
- HELIUM_PRESSURE (Torr): 헬륨 압력, 정상범위: 5–20
- PLASMA_DENSITY (ions/cm³): 플라즈마 밀도, 정상범위: 1e10–1e12
- ENDPOINT_SIGNAL (a.u.): 종점 신호, 모니터링용

답변은 반드시 한국어로 작성하세요.
"""

# CVD 공정 전용 프롬프트
EXPLANATION_PROMPT_CVD = r"""
당신은 반도체 CVD(Chemical Vapor Deposition) 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 CVD 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[CVD 센서 데이터 컬럼 및 정상범위]
- SUSCEPTOR_TEMP (°C): 서셉터 온도, 정상범위: 300–700
- CHAMBER_PRESSURE (Torr): 챔버 압력, 정상범위: 0.1–760
- PRECURSOR_FLOW_TEOS (sccm): TEOS 전구체 유량, 정상범위: 0–500
- PRECURSOR_FLOW_SILANE (sccm): Silane 전구체 유량, 정상범위: 0–1000
- PRECURSOR_FLOW_WF6 (sccm): WF6 전구체 유량, 정상범위: 0–100
- CARRIER_GAS_N2 (slm): N2 캐리어 가스, 정상범위: 0–20
- CARRIER_GAS_H2 (slm): H2 캐리어 가스, 정상범위: 0–10
- SHOWERHEAD_TEMP (°C): 샤워헤드 온도, 정상범위: 150–250
- LINER_TEMP (°C): 라이너 온도, 정상범위: 100–200
- DEPOSITION_RATE (Å/min): 증착률, 모니터링용
- FILM_STRESS (MPa): 필름 응력, 모니터링용

답변은 반드시 한국어로 작성하세요.
"""

# Chemical 공정 프롬프트
EXPLANATION_PROMPT_CHEMICAL = r"""
당신은 화학 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 화학 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[화학 공정 센서 데이터 컬럼 및 정상범위]
- TEMPERATURE (°C): 반응 온도, 정상범위: 공정별 상이
- PRESSURE (bar): 반응 압력, 정상범위: 공정별 상이
- pH: pH 값, 정상범위: 6.5–8.5
- CONCENTRATION (%): 농도, 정상범위: 공정별 상이
- FEED_RATE (L/min): 원료 투입 속도, 모니터링용
- CATALYST_RATIO: 촉매 비율, 정상범위: 공정별 상이
- AGITATOR_SPEED (rpm): 교반기 속도, 정상범위: 공정별 상이
- COOLING_WATER_FLOW (L/min): 냉각수 유량, 모니터링용
- REACTION_RATE: 반응 속도, 모니터링용

답변은 반드시 한국어로 작성하세요.
"""

# Battery 공정 프롬프트
EXPLANATION_PROMPT_BATTERY = r"""
당신은 배터리 제조 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 배터리 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[배터리 공정 센서 데이터 컬럼 및 정상범위]
- VOLTAGE (V): 전압, 정상범위: 2.5–4.2
- CURRENT (A): 전류, 정상범위: 공정별 상이
- CAPACITY (Ah): 용량, 모니터링용
- TEMPERATURE (°C): 온도, 정상범위: 20–50
- SOC (%): 충전 상태, 정상범위: 0–100
- CYCLE: 사이클 수, 모니터링용
- INTERNAL_RESISTANCE (mΩ): 내부 저항, 정상범위: <50
- CHARGE_RATE (C): 충전 속도, 정상범위: 0.1–1.0
- DISCHARGE_RATE (C): 방전 속도, 정상범위: 0.1–2.0

답변은 반드시 한국어로 작성하세요.
"""

# Automotive 공정 프롬프트
EXPLANATION_PROMPT_AUTOMOTIVE = r"""
당신은 자동차 제조 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 자동차 제조 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[자동차 제조 공정 센서 데이터 컬럼 및 정상범위]
- WELDING_CURRENT (A): 용접 전류, 정상범위: 공정별 상이
- WELDING_VOLTAGE (V): 용접 전압, 정상범위: 공정별 상이
- PRESSURE (bar): 압력, 정상범위: 공정별 상이
- TEMPERATURE (°C): 온도, 정상범위: 공정별 상이
- FORCE (kN): 가압력, 정상범위: 공정별 상이
- SPEED (mm/s): 속도, 모니터링용
- THICKNESS (mm): 두께, 정상범위: ±0.1mm
- COATING_THICKNESS (μm): 도장 두께, 정상범위: 공정별 상이

답변은 반드시 한국어로 작성하세요.
"""

# Steel 공정 프롬프트
EXPLANATION_PROMPT_STEEL = r"""
당신은 철강 제조 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 철강 공정 센서 데이터의 컬럼 설명과 정상범위입니다.
정상범위와 사용자 입력 데이터를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.

[철강 공정 센서 데이터 컬럼 및 정상범위]
- TEMPERATURE (°C): 온도, 정상범위: 공정별 상이 (1000–1600)
- PRESSURE (bar): 압력, 정상범위: 공정별 상이
- ROLLING_FORCE (kN): 압연력, 정상범위: 공정별 상이
- THICKNESS (mm): 두께, 정상범위: ±0.5mm
- SPEED (m/min): 속도, 모니터링용
- TENSION (kN): 장력, 정상범위: 공정별 상이
- CARBON_CONTENT (%): 탄소 함량, 정상범위: 0.05–2.0
- COOLING_RATE (°C/s): 냉각 속도, 모니터링용

답변은 반드시 한국어로 작성하세요.
"""

# 공정별 프롬프트 매핑 (20개 공정 지원)
PROCESS_EXPLANATION_PROMPTS = {
    # Semiconductor (4개)
    'semiconductor_cmp_001': EXPLANATION_PROMPT_CMP,
    'semiconductor_etch_002': EXPLANATION_PROMPT_ETCH,
    'semiconductor_deposition_003': EXPLANATION_PROMPT_CVD,
    'semiconductor_full_004': EXPLANATION_PROMPT_CVD,
    # Chemical (4개)
    'chemical_reactor_001': EXPLANATION_PROMPT_CHEMICAL,
    'chemical_distillation_002': EXPLANATION_PROMPT_CHEMICAL,
    'chemical_refining_003': EXPLANATION_PROMPT_CHEMICAL,
    'chemical_full_004': EXPLANATION_PROMPT_CHEMICAL,
    # Battery (4개)
    'battery_formation_001': EXPLANATION_PROMPT_BATTERY,
    'battery_coating_002': EXPLANATION_PROMPT_BATTERY,
    'battery_aging_003': EXPLANATION_PROMPT_BATTERY,
    'battery_production_004': EXPLANATION_PROMPT_BATTERY,
    # Automotive (4개)
    'automotive_welding_001': EXPLANATION_PROMPT_AUTOMOTIVE,
    'automotive_painting_002': EXPLANATION_PROMPT_AUTOMOTIVE,
    'automotive_press_003': EXPLANATION_PROMPT_AUTOMOTIVE,
    'automotive_assembly_004': EXPLANATION_PROMPT_AUTOMOTIVE,
    # Steel (4개)
    'steel_rolling_001': EXPLANATION_PROMPT_STEEL,
    'steel_converter_002': EXPLANATION_PROMPT_STEEL,
    'steel_casting_003': EXPLANATION_PROMPT_STEEL,
    'steel_production_004': EXPLANATION_PROMPT_STEEL,
}

PROCESS_TASK_PROMPTS = {
    # Semiconductor (4개)
    'semiconductor_cmp_001': EXPLANATION_PROMPT_CMP.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'semiconductor_etch_002': EXPLANATION_PROMPT_ETCH.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'semiconductor_deposition_003': EXPLANATION_PROMPT_CVD.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'semiconductor_full_004': EXPLANATION_PROMPT_CVD.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    # Chemical (4개)
    'chemical_reactor_001': EXPLANATION_PROMPT_CHEMICAL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'chemical_distillation_002': EXPLANATION_PROMPT_CHEMICAL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'chemical_refining_003': EXPLANATION_PROMPT_CHEMICAL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'chemical_full_004': EXPLANATION_PROMPT_CHEMICAL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    # Battery (4개)
    'battery_formation_001': EXPLANATION_PROMPT_BATTERY.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'battery_coating_002': EXPLANATION_PROMPT_BATTERY.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'battery_aging_003': EXPLANATION_PROMPT_BATTERY.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'battery_production_004': EXPLANATION_PROMPT_BATTERY.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    # Automotive (4개)
    'automotive_welding_001': EXPLANATION_PROMPT_AUTOMOTIVE.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'automotive_painting_002': EXPLANATION_PROMPT_AUTOMOTIVE.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'automotive_press_003': EXPLANATION_PROMPT_AUTOMOTIVE.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'automotive_assembly_004': EXPLANATION_PROMPT_AUTOMOTIVE.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    # Steel (4개)
    'steel_rolling_001': EXPLANATION_PROMPT_STEEL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'steel_converter_002': EXPLANATION_PROMPT_STEEL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'steel_casting_003': EXPLANATION_PROMPT_STEEL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
    'steel_production_004': EXPLANATION_PROMPT_STEEL.replace("설명을 만들어내는", "분석 과업을 생성하는"),
}

# 📝 OLD VERSION (주석 처리 - 참고용)
# 기존 통합 프롬프트 (Industrial_DB_sample 형식)
# 이상치 설명 생성용 프롬프트
EXPLANATION_SYSTEM_PROMPT = r"""
당신은 반도체 제조 공정 내 이상치 값에 대한 설명을 만들어내는 에이전트입니다.
아래는 당신이 접근할 수 있는 제조공정 DB 종류와 컬럼 설명, 그리고 정상범위(규격)입니다.
아래의 정상범위와 사용자 입력 내의 data를 비교하여 정상범위에서 벗어나는 컬럼에 한해서만 설명을 생성하세요.
정상범위 내의 컬럼의 경우 설명을 생성하지 않습니다.
이상치 값에 대한 설명은 반드시 해당 컬럼의 정상범위와 비교하여 작성하고, 각 이상치 값의 인과관계가 있다면 설명하세요.
정상범위는 괄호 안에 표기하고, 이상치인 값을 반드시 명시하세요.
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
FEWSHOT_USER_CASE_1 = r"""
data:
{
  "pno": "LM003",
  "lot_no": "LOT24002A",
  "product_name": "LOGIC_AP_5NM",
  "recipe_id": "RCP_LOGIC_V4.2",
  "start_qty": 0,
  "current_step": "CVD_003",
  "priority": "NORMAL",
  "credate": "2024-01-16",
  "holder": "NULL",
  "final_yield": 87.2,
  "good_die": -0.6765571639390562,
  "total_die": 0,
  "is_anomaly": "False",
  "anomaly_score": 1.8752173241964718,
  "predicted_anomaly": "True",
  "confidence": 0.18953331879760532,
  "alignment_error_x": 1.7815834390378973,
  "alignment_error_y": 1.8068078795947535,
  "analyzer_pressure": -0.43597707548113906,
  "barometric_pressure": 1.4420067797217386,
  "beam_current": -0.42880492814421195,
  "beam_energy": -0.367998949238794,
  "beam_uniformity": -0.49999105326920523,
  "carrier_gas_h2": 3.1995600923867222,
  "carrier_gas_n2": 2.1218784814085305,
  "chamber_pressure": 0.18076389061653064,
  "chamber_wall_temp": 1.6873652626808404,
  "conditioner_pressure": -0.49503753506734227,
  "deposition_rate": 1.0130438177226755,
  "dose_rate": -0.42472921736082786,
  "electrode_temp": 1.3184969427441044,
  "end_station_pressure": -0.3956727369757142,
  "endpoint_signal": 0.504642079457869,
  "exposure_dose": 1.6233671332894895,
  "faraday_cup_current": -0.4286657330065187,
  "film_stress": -1.9404378797185633,
  "focus_position": -1.9972457279329923,
  "gas_flow_ar": 2.233291090364354,
  "gas_flow_cf4": 2.0841894204905045,
  "gas_flow_cl2": -0.3287402288967945,
  "gas_flow_o2": 2.2583510640388935,
  "head_pressure": -0.47847613695539476,
  "head_rotation": -0.4938117219582884,
  "helium_pressure": 2.036111066721793,
  "humidity": 1.4264630825263822,
  "illumination_uniformity": 1.4341015184303116,
  "implant_angle": -0.3892494720807615,
  "lens_aberration": 1.8357565360747123,
  "liner_temp": 2.0033538607897676,
  "motor_current": -0.49106100492548654,
  "pad_temp": -0.49587348653461977,
  "plasma_density": 2.2033492916449804,
  "platen_rotation": -0.4936076172427683,
  "precursor_flow_silane": 1.0187934234139417,
  "precursor_flow_teos": -0.28017408751929807,
  "precursor_flow_wf6": 0,
  "removal_rate": -0.47915544482813255,
  "retainer_pressure": -0.47847075532575595,
  "reticle_temp": 1.4419249925344992,
  "rf_power_bias": 2.12681367067234,
  "rf_power_source": 2.1360142945171616,
  "showerhead_temp": 1.9842732940396977,
  "slurry_flow_rate": -0.49313259164035833,
  "slurry_temp": -0.4995945525763811,
  "source_pressure": -0.46251622138613974,
  "stage_temp": 1.439615028558493,
  "susceptor_temp": 1.6559810098169359,
  "total_dose": -0.3592243801316995,
  "wafer_rotation": -0.48133693912255127
}
answer:
"""
FEWSHOT_EXPLANATION_1 = (
    "LOT24002A의 최종 수율이 0.0%로 정상 기준(>90%)에서 크게 벗어났습니다.\n"  
    "이는 개별 CVD_003 장비 공정 단계에서 발생한 이상치가 누적된 결과로 해석됩니다.\n"  
    "구체적으로, CVD_003 장비의 Alignment Error 값은 정상 범위(-0.5 ~ 0.5)를 벗어나 -1.9 이하로 관측되었고,\n"  
    "Stage Position 관련 값 또한 정상 범위(±0.3)를 초과하여 -0.49 ~ -0.50 수준으로 치우쳤습니다.\n"  
    "Lens 관련 파라미터 역시 정상 범위(약 1.0 ~ 2.0)에서 벗어나 비정상적 진동 패턴을 보였습니다.\n"  
    "이러한 다수의 센서 이상치는 공정 조건 불량으로 이어져, 결과적으로 수율 저하를 초래한 것으로 추정됩니다."
)

FEWSHOT_TASK_1 = (
    "- CVD_003 장비의 주요 센서(Alignment Error, Stage Position, Lens Aberration)의 변동이 최종 수율에 미치는 영향을 분석할 필요가 있습니다.\n"
    "- Stage 온도, 압력, 습도 등 보조 환경 데이터와 결합하여, 극단적 음수 센서값 발생이 공정 불량을 유발하는지 예측 모델을 구축할 필요가 있습니다.\n"
    "- LOT24002A와 동일 레시피(RCP_LOGIC_V4.2)를 사용하는 다른 로트와 비교하여 이상치 발생 조건의 재현성을 분석하는 작업이 필요합니다."
)

FEWSHOT_USER_CASE_2 = r"""
data:
{
  "pno": "LM029",
  "lot_no": "LOT24015A",
  "product_name": "NAND_2TB_TLC",
  "recipe_id": "RCP_NAND_V2.6",
  "start_qty": 0,
  "current_step": "ION_IMPLANT_003",
  "priority": "NORMAL",
  "credate": "2024-01-29",
  "holder": "NULL",
  "final_yield": 93.4,
  "good_die": 1.2950782844379818,
  "total_die": 0,
  "is_anomaly": "False",
  "anomaly_score": 2.2552952852550203,
  "predicted_anomaly": "True",
  "confidence": 0.4306335862631553,
  "alignment_error_x": -0.6676402297005561,
  "alignment_error_y": -0.6713890721237328,
  "analyzer_pressure": 3.9660452327350053,
  "barometric_pressure": -0.6938885801475108,
  "beam_current": 3.6999247313219477,
  "beam_energy": 4.566065174633304,
  "beam_uniformity": 1.9756528666410966,
  "carrier_gas_h2": -0.39226512210945047,
  "carrier_gas_n2": -0.520825706016853,
  "chamber_pressure": -0.5863577785945447,
  "chamber_wall_temp": -0.6152746833234518,
  "conditioner_pressure": -0.49503753506734227,
  "deposition_rate": -0.43400081836651716,
  "dose_rate": 3.967907555344312,
  "electrode_temp": -0.6137491306745388,
  "end_station_pressure": 4.515068407131405,
  "endpoint_signal": -0.8800293726614797,
  "exposure_dose": -0.6905368679997218,
  "faraday_cup_current": 3.701871359697373,
  "film_stress": -0.286337079538255,
  "focus_position": 0.2755715342606508,
  "gas_flow_ar": -0.579002434971891,
  "gas_flow_cf4": -0.5940785926557355,
  "gas_flow_cl2": -0.3287402288967945,
  "gas_flow_o2": -0.5779895528354979,
  "head_pressure": -0.47847613695539476,
  "head_rotation": -0.4938117219582884,
  "helium_pressure": -0.6023746876433743,
  "humidity": -0.6938665776162654,
  "illumination_uniformity": -0.6938792921614627,
  "implant_angle": 4.281744192888376,
  "lens_aberration": -0.6745914564191262,
  "liner_temp": -0.5340551279339955,
  "motor_current": -0.49106100492548654,
  "pad_temp": -0.49587348653461977,
  "plasma_density": -0.5922789271156291,
  "platen_rotation": -0.4936076172427683,
  "precursor_flow_silane": -0.4033376434322697,
  "precursor_flow_teos": -0.28017408751929807,
  "precursor_flow_wf6": 0,
  "removal_rate": -0.47915544482813255,
  "retainer_pressure": -0.47847075532575595,
  "reticle_temp": -0.6938879939682062,
  "rf_power_bias": -0.5985362122569439,
  "rf_power_source": -0.5963735468714711,
  "showerhead_temp": -0.5364011820062409,
  "slurry_flow_rate": -0.49313259164035833,
  "slurry_temp": -0.4995945525763811,
  "source_pressure": 3.464508299816934,
  "stage_temp": -0.6938884459642612,
  "susceptor_temp": -0.5165975675749419,
  "total_dose": 4.132671934533579,
  "wafer_rotation": 1.046716835869675
}
answer:
"""

FEWSHOT_EXPLANATION_2 = (
    "LOT24002A의 ION IMPLANT 공정에서 다수의 이상치가 관찰되었습니다.\n"
    "구체적으로, DOSE 값은 정상 범위(-0.5 ~ 0.5)를 벗어나 -0.6939로 측정되었고,\n"
    "CHAMBER 압력 역시 정상 범위(-0.2 ~ 0.2)를 벗어나 -0.3940으로 나타났습니다.\n"
    "또한 OVERLAY_X(3.9329)와 BEAM_X(4.6329)는 정상 범위(-1.0 ~ 1.0)를 크게 초과하여,\n"
    "패턴 정렬 불량 및 빔 위치 이탈이 심각하게 발생했음을 의미합니다.\n"
    "IMPLANT_DEPTH 값은 정상 범위(0.5 ~ 1.5)를 벗어나 4.2817로 과도하게 깊었고,\n"
    "RESIST_UNIFORMITY 또한 정상 범위(-0.5 ~ 0.5)를 벗어나 4.1157로 비정상적 편차를 보였습니다.\n"
    "마지막으로, YIELD_ESTIMATION(4.1327), ALERT_SCORE(6.0283), PASS_FAIL(1.7927) 역시 정상 기준(각각 0.0~1.0)을 초과했습니다.\n"
    "이러한 일련의 센서 이상치는 장비 빔 정렬 불량 및 챔버 조건 불안정에서 기인한 것으로 추정되며,\n"
    "결과적으로 ION IMPLANT 공정 품질 저하 및 최종 수율 손실로 이어질 가능성이 큽니다."
)

FEWSHOT_TASK_2 = (
    "- LOT24015A, ION_IMPLANT_003 장비의 YIELD_ESTIMATION 값(4.13)과 PASS_FAIL 지표(1.79)가 정상 범위(0.0~1.0)에서 크게 벗어나, 수율 저하 가능성에 대한 예측 분석 필요.\n"
    "- BEAM_X 값(-0.67)이 정상 범위(-0.5~0.5)에서 이탈하여, 이온 빔 정렬 불량이 후속 공정 품질에 미치는 영향 예측 필요.\n"
    "- RESIST_UNIFORMITY 값(6.02)이 정상 범위(≈0.8~1.2)를 초과하여, 레지스트 막 두께 불균일이 최종 소자 성능에 미치는 영향을 시뮬레이션할 필요.\n"
    "- Alignment 및 Stage 관련 센서 값의 이상이 누적되어, 장비 조건 불안정이 전체 로트 품질과 최종 수율에 미칠 위험도를 사전 평가해야 함."
)

def _call_api(llm_url, system_prompt, fewshots, data, max_tokens=256, temperature=0.7, presence_penalty=1.5):
    print(llm_url)
    # prompt 문자열을 직접 구성
    prompt = system_prompt + "\n\n"
    for user_ex, assistant_ex in fewshots:
        prompt += f"User: {user_ex}\nAssistant: {assistant_ex}\n"
    prompt += f"User: data:\n{data}\n\nanswer:\nAssistant:"

    response = llm_generate(
        url=llm_url,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty
    )
    print(response)
    return response['text']


# 🆕 NEW VERSION: 공정별 프롬프트 지원
def event_explain(llm_url, event_detect_analysis, process_type: str = None):
    """
    이상치 설명 생성

    Args:
        llm_url: LLM API URL
        event_detect_analysis: 이상치 데이터 (JSON 문자열 또는 dict)
        process_type: 공정 타입 (예: 'semi_cmp_sensors', 'semi_etch_sensors', ...)
                      None이면 기존 통합 프롬프트 사용 (하위 호환성)

    Returns:
        이상치 설명 텍스트
    """
    fewshots = [
        (FEWSHOT_USER_CASE_1, FEWSHOT_EXPLANATION_1),
        (FEWSHOT_USER_CASE_2, FEWSHOT_EXPLANATION_2),
    ]

    # 공정별 프롬프트 선택
    if process_type and process_type in PROCESS_EXPLANATION_PROMPTS:
        system_prompt = PROCESS_EXPLANATION_PROMPTS[process_type]
        print(f"Using process-specific prompt for: {process_type}")
    else:
        system_prompt = EXPLANATION_SYSTEM_PROMPT
        print("Using legacy unified prompt")

    return _call_api(llm_url, system_prompt, fewshots, event_detect_analysis)

def event_cause_candidates(llm_url, event_detect_analysis, process_type: str = None):
    """
    분석 과업 생성

    Args:
        llm_url: LLM API URL
        event_detect_analysis: 이상치 데이터 (JSON 문자열 또는 dict)
        process_type: 공정 타입 (예: 'semi_cmp_sensors', 'semi_etch_sensors', ...)
                      None이면 기존 통합 프롬프트 사용 (하위 호환성)

    Returns:
        분석 과업 텍스트
    """
    fewshots = [
        (FEWSHOT_USER_CASE_1, FEWSHOT_TASK_1),
        (FEWSHOT_USER_CASE_2, FEWSHOT_TASK_2),
    ]

    # 공정별 프롬프트 선택
    if process_type and process_type in PROCESS_TASK_PROMPTS:
        system_prompt = PROCESS_TASK_PROMPTS[process_type]
        print(f"Using process-specific task prompt for: {process_type}")
    else:
        system_prompt = TASK_SYSTEM_PROMPT
        print("Using legacy unified task prompt")

    return _call_api(llm_url, system_prompt, fewshots, event_detect_analysis)

# 📝 OLD VERSION (주석 처리 - 참고용)
# def event_explain(llm_url, event_detect_analysis):
#     fewshots = [
#         (FEWSHOT_USER_CASE_1, FEWSHOT_EXPLANATION_1),
#         (FEWSHOT_USER_CASE_2, FEWSHOT_EXPLANATION_2),
#     ]
#
#     return _call_api(llm_url, EXPLANATION_SYSTEM_PROMPT, fewshots, event_detect_analysis)
#
# def event_cause_candidates(llm_url, event_detect_analysis):
#     fewshots = [
#         (FEWSHOT_USER_CASE_1, FEWSHOT_TASK_1),
#         (FEWSHOT_USER_CASE_2, FEWSHOT_TASK_2),
#     ]
#     return _call_api(llm_url, TASK_SYSTEM_PROMPT, fewshots, event_detect_analysis)

if __name__ == "__main__":
    url = '0.0.0.0:8888'
    test_data_single = '{"RF_POWER_SOURCE":2200.0,"CHAMBER_PRESSURE":250.0,"CHAMBER_TEMP":85.0}'
    test_data_multi = '{"LOT_NO":"LOT30012A","PRODUCT_NAME":"DRAM_512","START_QTY":25,"CURRENT_STEP":"PHOTO","FINAL_YIELD":75.0,"EXPOSURE_DOSE":45.0,"FOCUS_POSITION":80.0,"STAGE_TEMP":23.4,"HUMIDITY":60.0}'
    
    print("=== 단일 행 입력 (설명) ===")
    print(event_explain(url, test_data_single))
    print("\n=== 단일 행 입력 (과업) ===")
    print(event_cause_candidates(url, test_data_single))
    
    print("\n=== 다중 속성 입력 (설명) ===")
    print(event_explain(url, test_data_multi))
    print("\n=== 다중 속성 입력 (과업) ===")
    print(event_cause_candidates(url, test_data_multi))