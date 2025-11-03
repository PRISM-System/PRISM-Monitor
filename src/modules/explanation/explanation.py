# explanation.py
# 모든 20개 공정에 대해 동일 수준의 상세 프롬프트(컬럼/단위/정상범위)를 자동 생성
# - 레거시 키(semi_*_sensors) 제거
# - process_type은 반드시 20개 중 하나여야 함
# - 설명(event_explain) / 과업(event_cause_candidates) 둘 다 동일한 상세도 유지

from typing import Dict, List, Tuple, Any
from src.modules.llm.llm import LLMCallManager

# ---------------------------------------------------------------------
# 1) 공정 스펙 사전 (사용자 제공 템플릿을 코드로 내장)
#    - columns / units / normal_ranges / anomaly_scenario
# ---------------------------------------------------------------------

PROCESS_SPECS: Dict[str, Dict[str, Any]] = {
    # ========================= Semiconductor =========================
    "semiconductor_cmp": {
        "domain": "semiconductor",
        "title": "CMP Sensors",
        "dataset_file": "semiconductor_cmp_001.csv",
        "scenario_id": "SCENARIO_01",
        "columns": [
            "TIMESTAMP","SENSOR_ID","MOTOR_CURRENT","SLURRY_FLOW_RATE","HEAD_ROTATION",
            "PRESSURE","TEMPERATURE","POLISH_TIME","PAD_THICKNESS","SLURRY_TEMP","VIBRATION"
        ],
        "units": {
            "MOTOR_CURRENT": "A",
            "SLURRY_FLOW_RATE": "mL/min",
            "HEAD_ROTATION": "RPM",
            "PRESSURE": "bar",
            "TEMPERATURE": "°C",
            "POLISH_TIME": "seconds",
            "PAD_THICKNESS": "mm",
            "SLURRY_TEMP": "°C",
            "VIBRATION": "mm/s"
        },
        "normal_ranges": {
            "MOTOR_CURRENT": [15.0, 18.0],
            "SLURRY_FLOW_RATE": [200, 300],
            "HEAD_ROTATION": [100, 140],
            "PRESSURE": [2.8, 3.5],
            "TEMPERATURE": [23.0, 27.0]
        },
        "anomaly_scenario": "Slurry flow rate instability around timestamp 13:50:00",
    },
    "semiconductor_etch": {
        "domain": "semiconductor",
        "title": "Etch Sensors",
        "dataset_file": "semiconductor_etch_002.csv",
        "scenario_id": "SCENARIO_02",
        "columns": [
            "TIMESTAMP","CHAMBER_ID","PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER",
            "TEMPERATURE","ETCH_RATE","BIAS_VOLTAGE","CHAMBER_HUMIDITY","GAS_COMPOSITION"
        ],
        "units": {
            "PRESSURE": "mTorr",
            "VACUUM_PUMP": "%",
            "GAS_FLOW_RATE": "sccm",
            "RF_POWER": "W",
            "TEMPERATURE": "°C",
            "ETCH_RATE": "nm/min",
            "BIAS_VOLTAGE": "V",
            "CHAMBER_HUMIDITY": "%",
            "GAS_COMPOSITION": "ratio"
        },
        "normal_ranges": {
            "PRESSURE": [50, 100],
            "VACUUM_PUMP": [85, 95],
            "GAS_FLOW_RATE": [100, 150],
            "RF_POWER": [800, 1200],
            "TEMPERATURE": [20, 30]
        },
        "anomaly_scenario": "Pressure continuously increasing toward critical threshold",
    },
    "semiconductor_deposition": {
        "domain": "semiconductor",
        "title": "Deposition Sensors",
        "dataset_file": "semiconductor_deposition_003.csv",
        "scenario_id": "SCENARIO_03",
        "columns": [
            "TIMESTAMP","CHAMBER_ID","TEMPERATURE","PRESSURE","GAS_FLOW_RATE","RF_POWER",
            "DEPOSITION_RATE","FILM_THICKNESS","SUBSTRATE_TEMP","PRECURSOR_FLOW","UNIFORMITY"
        ],
        "units": {
            "TEMPERATURE": "°C",
            "PRESSURE": "mTorr",
            "GAS_FLOW_RATE": "sccm",
            "RF_POWER": "W",
            "DEPOSITION_RATE": "nm/min",
            "FILM_THICKNESS": "nm",
            "SUBSTRATE_TEMP": "°C",
            "PRECURSOR_FLOW": "sccm",
            "UNIFORMITY": "%"
        },
        "normal_ranges": {
            "TEMPERATURE": [350, 450],
            "PRESSURE": [200, 300],
            "GAS_FLOW_RATE": [150, 250],
            "RF_POWER": [500, 700],
            "DEPOSITION_RATE": [80, 120]
        },
        "anomaly_scenario": "Temperature instability requiring auto control",
    },
    "semiconductor_full": {
        "domain": "semiconductor",
        "title": "Full Process Sensors",
        "dataset_file": "semiconductor_full_004.csv",
        "scenario_id": "SCENARIO_04",
        "columns": [
            "TIMESTAMP","EQUIPMENT_ID","RF_POWER","PRESSURE","TEMPERATURE","GAS_FLOW_RATE",
            "VACUUM_PUMP","PLASMA_DENSITY","ELECTRON_TEMP","PROCESS_YIELD","DEFECT_COUNT","COMPLIANCE_STATUS"
        ],
        "units": {
            "RF_POWER": "W",
            "PRESSURE": "mTorr",
            "TEMPERATURE": "°C",
            "GAS_FLOW_RATE": "sccm",
            "VACUUM_PUMP": "%",
            "PLASMA_DENSITY": "cm^-3",
            "ELECTRON_TEMP": "eV",
            "PROCESS_YIELD": "%",
            "DEFECT_COUNT": "count",
            "COMPLIANCE_STATUS": "binary"
        },
        "normal_ranges": {
            "RF_POWER": [900, 1100],
            "PRESSURE": [70, 90],
            "TEMPERATURE": [25, 35],
            "PROCESS_YIELD": [95, 100]
        },
        "anomaly_scenario": "RF_POWER increasing trend requiring prediction, control, and compliance check",
    },

    # =========================== Chemical ============================
    "chemical_reactor": {
        "domain": "chemical",
        "title": "Reactor Sensors",
        "dataset_file": "chemical_reactor_001.csv",
        "scenario_id": "SCENARIO_05",
        "columns": [
            "TIMESTAMP","REACTOR_ID","TEMPERATURE","PRESSURE","pH","CONCENTRATION",
            "FEED_RATE","CATALYST_RATIO","AGITATOR_SPEED","COOLING_WATER_FLOW","REACTION_RATE"
        ],
        "units": {
            "TEMPERATURE": "°C",
            "PRESSURE": "bar",
            "pH": "pH",
            "CONCENTRATION": "%",
            "FEED_RATE": "L/min",
            "CATALYST_RATIO": "ratio",
            "AGITATOR_SPEED": "RPM",
            "COOLING_WATER_FLOW": "L/min",
            "REACTION_RATE": "mol/min"
        },
        "normal_ranges": {
            "TEMPERATURE": [180, 190],
            "PRESSURE": [2.0, 2.5],
            "pH": [7.0, 7.5],
            "CONCENTRATION": [75, 85],
            "FEED_RATE": [40, 50]
        },
        "anomaly_scenario": "Rapid temperature increase in reactor REACT_A3",
    },
    "chemical_distillation": {
        "domain": "chemical",
        "title": "Distillation Sensors",
        "dataset_file": "chemical_distillation_002.csv",
        "scenario_id": "SCENARIO_06",
        "columns": [
            "TIMESTAMP","TOWER_ID","PRESSURE","TEMPERATURE","REFLUX_RATIO","REBOILER_HEAT",
            "FEED_FLOW_RATE","TOP_TEMP","BOTTOM_TEMP","DISTILLATE_PURITY","CONDENSATE_FLOW"
        ],
        "units": {
            "PRESSURE": "bar",
            "TEMPERATURE": "°C",
            "REFLUX_RATIO": "ratio",
            "REBOILER_HEAT": "kW",
            "FEED_FLOW_RATE": "L/min",
            "TOP_TEMP": "°C",
            "BOTTOM_TEMP": "°C",
            "DISTILLATE_PURITY": "%",
            "CONDENSATE_FLOW": "L/min"
        },
        "normal_ranges": {
            "PRESSURE": [1.8, 2.2],
            "TEMPERATURE": [95, 105],
            "REFLUX_RATIO": [2.0, 3.0],
            "REBOILER_HEAT": [180, 220],
            "FEED_FLOW_RATE": [55, 65]
        },
        "anomaly_scenario": "Pressure continuously increasing toward safety limit",
    },
    "chemical_refining": {
        "domain": "chemical",
        "title": "Refining Sensors",
        "dataset_file": "chemical_refining_003.csv",
        "scenario_id": "SCENARIO_07",
        "columns": [
            "TIMESTAMP","REFINE_ID","PURITY","TEMPERATURE","FILTER_PRESSURE","FLOW_RATE",
            "FILTRATION_RATE","CONTAMINANT_LEVEL","MEMBRANE_CONDUCTIVITY","PERMEATE_QUALITY","BACKWASH_CYCLE"
        ],
        "units": {
            "PURITY": "%", "TEMPERATURE": "°C", "FILTER_PRESSURE": "bar", "FLOW_RATE": "L/min",
            "FILTRATION_RATE": "L/m²/h", "CONTAMINANT_LEVEL": "ppm", "MEMBRANE_CONDUCTIVITY": "μS/cm",
            "PERMEATE_QUALITY": "%", "BACKWASH_CYCLE": "count"
        },
        "normal_ranges": {
            "PURITY": [92, 98],
            "TEMPERATURE": [35, 45],
            "FILTER_PRESSURE": [2.5, 3.5],
            "FLOW_RATE": [45, 55]
        },
        "anomaly_scenario": "Purity degradation below target requiring auto control",
    },
    "chemical_full": {
        "domain": "chemical",
        "title": "Full Process Sensors",
        "dataset_file": "chemical_full_004.csv",
        "scenario_id": "SCENARIO_08",
        "columns": [
            "TIMESTAMP","PROCESS_ID","CATALYST_RATIO","TEMPERATURE","PRESSURE","pH",
            "CONCENTRATION","YIELD","SELECTIVITY","EMISSION_LEVEL","WASTE_TREATMENT","COMPLIANCE_STATUS"
        ],
        "units": {
            "CATALYST_RATIO": "ratio", "TEMPERATURE": "°C", "PRESSURE": "bar", "pH": "pH",
            "CONCENTRATION": "%", "YIELD": "%", "SELECTIVITY": "%", "EMISSION_LEVEL": "ppm",
            "WASTE_TREATMENT": "pH", "COMPLIANCE_STATUS": "binary"
        },
        "normal_ranges": {
            "CATALYST_RATIO": [0.05, 0.12],
            "TEMPERATURE": [175, 195],
            "PRESSURE": [2.1, 2.4],
            "pH": [6.8, 7.4],
            "YIELD": [85, 95]
        },
        "anomaly_scenario": "Catalyst ratio increasing requiring environmental compliance verification",
    },

    # =========================== Automotive ==========================
    "automotive_welding": {
        "domain": "automotive",
        "title": "Welding Sensors",
        "dataset_file": "automotive_welding_001.csv",
        "scenario_id": "SCENARIO_09",
        "columns": [
            "TIMESTAMP","LINE_ID","WELD_CURRENT","WELD_VOLTAGE","WIRE_SPEED","TRAVEL_SPEED",
            "SHIELDING_GAS","ARC_LENGTH","HEAT_INPUT","PENETRATION_DEPTH","WELD_QUALITY_INDEX"
        ],
        "units": {
            "WELD_CURRENT": "A","WELD_VOLTAGE": "V","WIRE_SPEED": "m/min","TRAVEL_SPEED": "cm/min",
            "SHIELDING_GAS": "L/min","ARC_LENGTH": "mm","HEAT_INPUT": "kJ/mm",
            "PENETRATION_DEPTH": "mm","WELD_QUALITY_INDEX": "score"
        },
        "normal_ranges": {
            "WELD_CURRENT": [180, 220],
            "WELD_VOLTAGE": [24, 28],
            "WIRE_SPEED": [8, 12],
            "TRAVEL_SPEED": [40, 60],
            "SHIELDING_GAS": [15, 20]
        },
        "anomaly_scenario": "Welding current fluctuation affecting quality",
    },
    "automotive_painting": {
        "domain": "automotive",
        "title": "Painting Sensors",
        "dataset_file": "automotive_painting_002.csv",
        "scenario_id": "SCENARIO_10",
        "columns": [
            "TIMESTAMP","BOOTH_ID","TEMPERATURE","HUMIDITY","SPRAY_PRESSURE","PAINT_FLOW_RATE",
            "BOOTH_AIRFLOW","ATOMIZATION_QUALITY","FILM_THICKNESS","OVERSPRAY_RATE","CURING_TEMP"
        ],
        "units": {
            "TEMPERATURE": "°C","HUMIDITY": "%","SPRAY_PRESSURE": "bar","PAINT_FLOW_RATE": "mL/min",
            "BOOTH_AIRFLOW": "m³/min","ATOMIZATION_QUALITY": "μm","FILM_THICKNESS": "μm",
            "OVERSPRAY_RATE": "%","CURING_TEMP": "°C"
        },
        "normal_ranges": {
            "TEMPERATURE": [22, 26],
            "HUMIDITY": [50, 65],
            "SPRAY_PRESSURE": [2.5, 3.5],
            "PAINT_FLOW_RATE": [180, 220],
            "BOOTH_AIRFLOW": [1800, 2200]
        },
        "anomaly_scenario": "Humidity continuously rising affecting coating quality",
    },
    "automotive_press": {
        "domain": "automotive",
        "title": "Press Sensors",
        "dataset_file": "automotive_press_003.csv",
        "scenario_id": "SCENARIO_11",
        "columns": [
            "TIMESTAMP","PRESS_ID","PRESS_FORCE","DIE_TEMPERATURE","STROKE_SPEED","BLANK_THICKNESS",
            "LUBRICATION","FORMING_DEPTH","SPRING_BACK","SURFACE_ROUGHNESS","DEFECT_RATE"
        ],
        "units": {
            "PRESS_FORCE": "kN","DIE_TEMPERATURE": "°C","STROKE_SPEED": "mm/s","BLANK_THICKNESS": "mm",
            "LUBRICATION": "ml/cycle","FORMING_DEPTH": "mm","SPRING_BACK": "mm",
            "SURFACE_ROUGHNESS": "μm","DEFECT_RATE": "%"
        },
        "normal_ranges": {
            "PRESS_FORCE": [800, 1200],
            "DIE_TEMPERATURE": [180, 220],
            "STROKE_SPEED": [80, 120],
            "BLANK_THICKNESS": [0.8, 1.2],
            "DEFECT_RATE": [0, 2]
        },
        "anomaly_scenario": "Press force imbalance requiring auto adjustment",
    },
    "automotive_assembly": {
        "domain": "automotive",
        "title": "Assembly Sensors",
        "dataset_file": "automotive_assembly_004.csv",
        "scenario_id": "SCENARIO_12",
        "columns": [
            "TIMESTAMP","STATION_ID","TORQUE","PRESSURE","TEMPERATURE","CYCLE_TIME","DEFECT_RATE",
            "TIGHTENING_ANGLE","BOLT_COUNT","POSITION_ACCURACY","QUALITY_SCORE","SAFETY_STATUS"
        ],
        "units": {
            "TORQUE": "Nm","PRESSURE": "bar","TEMPERATURE": "°C","CYCLE_TIME": "seconds",
            "DEFECT_RATE": "%","TIGHTENING_ANGLE": "degrees","BOLT_COUNT": "count",
            "POSITION_ACCURACY": "mm","QUALITY_SCORE": "score","SAFETY_STATUS": "binary"
        },
        "normal_ranges": {
            "TORQUE": [80, 120],
            "PRESSURE": [5.5, 6.5],
            "TEMPERATURE": [20, 28],
            "CYCLE_TIME": [45, 55],
            "DEFECT_RATE": [0, 1.5]
        },
        "anomaly_scenario": "Torque increasing requiring safety compliance verification",
    },

    # ============================= Battery ===========================
    "battery_formation": {
        "domain": "battery",
        "title": "Formation Sensors",
        "dataset_file": "battery_formation_001.csv",
        "scenario_id": "SCENARIO_13",
        "columns": [
            "TIMESTAMP","CELL_ID","VOLTAGE","CURRENT","CAPACITY","TEMPERATURE",
            "SOC","CYCLE","INTERNAL_RESISTANCE","CHARGE_RATE","DISCHARGE_RATE"
        ],
        "units": {
            "VOLTAGE": "V","CURRENT": "A","CAPACITY": "Ah","TEMPERATURE": "°C",
            "SOC": "%","CYCLE": "count","INTERNAL_RESISTANCE": "mΩ","CHARGE_RATE": "C","DISCHARGE_RATE": "C"
        },
        "normal_ranges": {
            "VOLTAGE": [4.05, 4.2],
            "CURRENT": [1.0, 1.5],
            "CAPACITY": [3.3, 3.7],
            "TEMPERATURE": [40, 50],
            "SOC": [80, 90]
        },
        "anomaly_scenario": "Cell voltage abnormal fluctuation",
    },
    "battery_coating": {
        "domain": "battery",
        "title": "Coating Sensors",
        "dataset_file": "battery_coating_002.csv",
        "scenario_id": "SCENARIO_14",
        "columns": [
            "TIMESTAMP","LINE_ID","COATING_THICKNESS","COATING_SPEED","SLURRY_VISCOSITY",
            "TEMPERATURE","HUMIDITY","DRYING_TEMP","ADHESION_STRENGTH","UNIFORMITY","DEFECT_DENSITY"
        ],
        "units": {
            "COATING_THICKNESS": "μm","COATING_SPEED": "m/min","SLURRY_VISCOSITY": "cP",
            "TEMPERATURE": "°C","HUMIDITY": "%","DRYING_TEMP": "°C","ADHESION_STRENGTH": "N/m",
            "UNIFORMITY": "%","DEFECT_DENSITY": "count/m²"
        },
        "normal_ranges": {
            "COATING_THICKNESS": [95, 105],
            "COATING_SPEED": [8, 12],
            "SLURRY_VISCOSITY": [1800, 2200],
            "TEMPERATURE": [23, 27],
            "HUMIDITY": [35, 45]
        },
        "anomaly_scenario": "Coating thickness deviating from target",
    },
    "battery_aging": {
        "domain": "battery",
        "title": "Aging Sensors",
        "dataset_file": "battery_aging_003.csv",
        "scenario_id": "SCENARIO_15",
        "columns": [
            "TIMESTAMP","CELL_ID","SOH","SOC","CAPACITY","TEMPERATURE","HUMIDITY",
            "AGING_DAYS","CAPACITY_RETENTION","IMPEDANCE_GROWTH","SELF_DISCHARGE_RATE"
        ],
        "units": {
            "SOH": "%","SOC": "%","CAPACITY": "Ah","TEMPERATURE": "°C","HUMIDITY": "%",
            "AGING_DAYS": "days","CAPACITY_RETENTION": "%","IMPEDANCE_GROWTH": "%","SELF_DISCHARGE_RATE": "%/month"
        },
        "normal_ranges": {
            "SOH": [95, 100],
            "SOC": [45, 55],
            "CAPACITY": [3.4, 3.6],
            "TEMPERATURE": [23, 27],
            "HUMIDITY": [45, 55]
        },
        "anomaly_scenario": "SOH degrading faster than expected",
    },
    "battery_production": {
        "domain": "battery",
        "title": "Production Sensors",
        "dataset_file": "battery_production_004.csv",
        "scenario_id": "SCENARIO_16",
        "columns": [
            "TIMESTAMP","PRODUCTION_LINE","TEMPERATURE","VOLTAGE","CURRENT","CAPACITY","SOC",
            "CELL_WEIGHT","ELECTROLYTE_FILL","SEAL_QUALITY","YIELD_RATE","SAFETY_STATUS"
        ],
        "units": {
            "TEMPERATURE": "°C","VOLTAGE": "V","CURRENT": "A","CAPACITY": "Ah","SOC": "%",
            "CELL_WEIGHT": "g","ELECTROLYTE_FILL": "mL","SEAL_QUALITY": "score","YIELD_RATE": "%","SAFETY_STATUS": "binary"
        },
        "normal_ranges": {
            "TEMPERATURE": [24, 28],
            "VOLTAGE": [4.1, 4.18],
            "CURRENT": [1.1, 1.4],
            "CAPACITY": [3.4, 3.6],
            "YIELD_RATE": [96, 100]
        },
        "anomaly_scenario": "Temperature increasing requiring safety compliance",
    },

    # ============================== Steel ============================
    "steel_rolling": {
        "domain": "steel",
        "title": "Rolling Sensors",
        "dataset_file": "steel_rolling_001.csv",
        "scenario_id": "SCENARIO_17",
        "columns": [
            "TIMESTAMP","LINE_ID","THICKNESS","ROLL_GAP","ROLLING_SPEED","TENSION",
            "TEMPERATURE","ROLL_FORCE","WIDTH","FLATNESS","SURFACE_QUALITY"
        ],
        "units": {
            "THICKNESS": "mm","ROLL_GAP": "mm","ROLLING_SPEED": "m/min","TENSION": "MPa",
            "TEMPERATURE": "°C","ROLL_FORCE": "kN","WIDTH": "mm","FLATNESS": "I-unit","SURFACE_QUALITY": "score"
        },
        "normal_ranges": {
            "THICKNESS": [3.6, 4.0],
            "ROLL_GAP": [3.8, 4.2],
            "ROLLING_SPEED": [160, 200],
            "TENSION": [270, 300],
            "TEMPERATURE": [820, 880]
        },
        "anomaly_scenario": "Thickness exceeding target deviation",
    },
    "steel_converter": {
        "domain": "steel",
        "title": "Converter Sensors",
        "dataset_file": "steel_converter_002.csv",
        "scenario_id": "SCENARIO_18",
        "columns": [
            "TIMESTAMP","CONVERTER_ID","TEMPERATURE","OXYGEN_FLOW","LANCE_POSITION","PRESSURE",
            "SLAG_COMPOSITION","CARBON_CONTENT","PHOSPHORUS_LEVEL","SULFUR_LEVEL","BATH_LEVEL"
        ],
        "units": {
            "TEMPERATURE": "°C","OXYGEN_FLOW": "Nm³/min","LANCE_POSITION": "mm","PRESSURE": "bar",
            "SLAG_COMPOSITION": "CaO/SiO2","CARBON_CONTENT": "%","PHOSPHORUS_LEVEL": "ppm",
            "SULFUR_LEVEL": "ppm","BATH_LEVEL": "mm"
        },
        "normal_ranges": {
            "TEMPERATURE": [1600, 1700],
            "OXYGEN_FLOW": [450, 550],
            "LANCE_POSITION": [1400, 1600],
            "PRESSURE": [0.9, 1.1],
            "SLAG_COMPOSITION": [2.8, 3.5]
        },
        "anomaly_scenario": "Temperature and oxygen flow fluctuation affecting slag status",
    },
    "steel_casting": {
        "domain": "steel",
        "title": "Casting Sensors",
        "dataset_file": "steel_casting_003.csv",
        "scenario_id": "SCENARIO_19",
        "columns": [
            "TIMESTAMP","CASTER_ID","SLAB_THICKNESS","CASTING_SPEED","MOLD_WIDTH","TEMPERATURE",
            "COOLING_WATER_FLOW","MOLD_LEVEL","STRAND_TEMPERATURE","WITHDRAWAL_FORCE","SURFACE_DEFECTS"
        ],
        "units": {
            "SLAB_THICKNESS": "mm","CASTING_SPEED": "m/min","MOLD_WIDTH": "mm","TEMPERATURE": "°C",
            "COOLING_WATER_FLOW": "m³/h","MOLD_LEVEL": "mm","STRAND_TEMPERATURE": "°C",
            "WITHDRAWAL_FORCE": "kN","SURFACE_DEFECTS": "count"
        },
        "normal_ranges": {
            "SLAB_THICKNESS": [220, 240],
            "CASTING_SPEED": [0.8, 1.2],
            "MOLD_WIDTH": [1500, 1600],
            "TEMPERATURE": [1540, 1580],
            "SURFACE_DEFECTS": [0, 2]
        },
        "anomaly_scenario": "Slab thickness imbalance requiring casting speed adjustment",
    },
    "steel_production": {
        "domain": "steel",
        "title": "Production Sensors",
        "dataset_file": "steel_production_004.csv",
        "scenario_id": "SCENARIO_20",
        "columns": [
            "TIMESTAMP","PRODUCTION_LINE","TEMPERATURE","PRESSURE","OXYGEN_FLOW","SLAG_COMPOSITION",
            "THICKNESS","PRODUCTION_RATE","QUALITY_INDEX","EMISSION_LEVEL","ENERGY_CONSUMPTION","COMPLIANCE_STATUS"
        ],
        "units": {
            "TEMPERATURE": "°C","PRESSURE": "bar","OXYGEN_FLOW": "Nm³/min","SLAG_COMPOSITION": "CaO/SiO2",
            "THICKNESS": "mm","PRODUCTION_RATE": "tons/hour","QUALITY_INDEX": "score",
            "EMISSION_LEVEL": "mg/Nm³","ENERGY_CONSUMPTION": "kWh/ton","COMPLIANCE_STATUS": "binary"
        },
        "normal_ranges": {
            "TEMPERATURE": [1620, 1680],
            "PRESSURE": [0.95, 1.05],
            "OXYGEN_FLOW": [480, 520],
            "THICKNESS": [3.7, 3.9],
            "PRODUCTION_RATE": [180, 220]
        },
        "anomaly_scenario": "Temperature increasing requiring environmental compliance verification",
    },
}

SUPPORTED_PROCESSES = set(PROCESS_SPECS.keys())

# ---------------------------------------------------------------------
# 2) 공통 프롬프트 빌더
#    - 모든 도메인에서 동일 형식으로 컬럼/단위/정상범위 표기
#    - 정상범위가 정의되지 않은 컬럼은 "모니터링용"으로 명시
# ---------------------------------------------------------------------

def _build_prompt(spec: Dict[str, Any], mode: str) -> str:
    """
    mode: 'explain' -> 이상치 설명 생성
          'task'    -> 분석 과업 생성
    """
    assert mode in ("explain", "task")
    head = (
        "당신은 제조 공정의 이상치 값에 대한 설명을 만들어내는 에이전트입니다."
        if mode == "explain"
        else "당신은 제조 공정에서 탐지된 이상치를 분석하기 위한 구체적인 과업을 생성하는 에이전트입니다."
    )

    guide = (
        "아래는 본 공정 센서 데이터의 컬럼과 단위, 정상범위입니다.\n"
        "정상범위와 사용자 입력 데이터를 비교하여 **정상범위를 벗어난 컬럼만** 다루세요.\n"
        "정상범위 내의 컬럼은 설명/과업을 생성하지 마세요.\n"
        "여러 항목이 동시에 이탈하면 인과관계를 함께 설명(또는 과업의 선후·의존 관계로 제시)하세요.\n"
        "최종 답변은 반드시 한국어로 작성하세요."
    )

    lines: List[str] = []
    lines.append(f"[데이터셋] {spec['title']}  |  파일: {spec['dataset_file']}  |  시나리오ID: {spec['scenario_id']}")
    lines.append("[센서 데이터 컬럼 · 단위 · 정상범위]")
    units = spec.get("units", {})
    ranges = spec.get("normal_ranges", {})
    for col in spec["columns"]:
        if col.upper() in ("TIMESTAMP","SENSOR_ID","EQUIPMENT_ID","CHAMBER_ID","REACTOR_ID","TOWER_ID",
                           "REFINE_ID","PROCESS_ID","LINE_ID","PRESS_ID","STATION_ID","CASTER_ID",
                           "PRODUCTION_LINE","CELL_ID","PRODUCTION_LINE"):
            # 식별자/시간 컬럼
            lines.append(f"- {col}: 식별/시간용 컬럼")
            continue
        u = units.get(col, None)
        if col in ranges:
            lo, hi = ranges[col]
            if u:
                lines.append(f"- {col} ({u}): 정상범위 {lo}–{hi}")
            else:
                lines.append(f"- {col}: 정상범위 {lo}–{hi}")
        else:
            # 정상범위 미지정 -> 모니터링/참고
            if u:
                lines.append(f"- {col} ({u}): 모니터링용")
            else:
                lines.append(f"- {col}: 모니터링용")

    anomaly_note = spec.get("anomaly_scenario")
    if anomaly_note:
        lines.append(f"\n[시나리오 포인트] {anomaly_note}")

    if mode == "explain":
        tail = (
            "\n[지시]\n"
            "1) 정상범위를 벗어난 컬럼만 나열하고, 각 항목에 대해 실제 값과 정상범위를 함께 명시하세요.\n"
            "2) 이탈 원인 가설(장비/환경/레시피/소모품)을 1~3개 제시하세요.\n"
            "3) 다중 항목 이탈 시, 파라미터 간 인과관계를 1~2문장으로 설명하세요.\n"
            "4) 필요한 경우 관련 규제/안전/품질 준수(Compliance) 리스크를 한 줄로 요약하세요."
        )
    else:
        tail = (
            "\n[지시]\n"
            "1) 우선순위가 높은 과업 3~6개를 번호 목록으로 제시하세요.\n"
            "2) 각 과업은 '목표/데이터/방법/판정기준'을 1줄로 요약하세요.\n"
            "3) 제어행동(세트포인트 조정/인터록/점검 등)이 필요한 경우 과업에 포함하세요.\n"
            "4) 필요한 경우 준수(Compliance) 확인 과업을 포함하세요."
        )

    prompt = f"""{head}

{guide}

{chr(10).join(lines)}
{tail}
"""
    return prompt


def _require_supported_process(process_type: str, func_name: str) -> None:
    if not process_type:
        raise ValueError(
            f"[{func_name}] process_type는 필수입니다. 지원 목록: {sorted(SUPPORTED_PROCESSES)}"
        )
    if process_type not in SUPPORTED_PROCESSES:
        raise ValueError(
            f"[{func_name}] 지원하지 않는 process_type='{process_type}'. 지원 목록: {sorted(SUPPORTED_PROCESSES)}"
        )


# ---------------------------------------------------------------------
# 3) Few-shot (간결 버전 — 포맷 예시용)
#    * 데이터는 실제 호출 시 사용자가 넘기는 JSON/딕트가 들어옵니다.
# ---------------------------------------------------------------------

FEWSHOT_USER_CASE_1 = """data:
{"TEMPERATURE": 28.3, "PRESSURE": 3.6, "SLURRY_FLOW_RATE": 320, "MOTOR_CURRENT": 18.9}
answer:
"""
FEWSHOT_EXPLANATION_1 = (
    "- SLURRY_FLOW_RATE=320 (정상 200–300): 과도 유량 → 패드/웨이퍼 간 유막 두꺼워져 제거율 변동 가능.\n"
    "- MOTOR_CURRENT=18.9A (정상 15–18): 구동부 부하 증가 → 패드 마모/정합 불량 의심.\n"
    "인과: 유량 증가가 마찰 조건을 바꿔 모터 부하 상승을 동반했을 수 있습니다."
)

FEWSHOT_TASK_1 = (
    "1) 유량-제거율 상관 분석(데이터: SLURRY_FLOW_RATE/REMOVAL_RATE; 방법: 회귀; 기준: R²>0.6)\n"
    "2) 모터 전류 이상 원인 분리(데이터: MOTOR_CURRENT/VIBRATION; 방법: FFT; 기준: 특정 주파수 피크)\n"
    "3) 세트포인트 조정 실험(데이터: 최근 2h; 방법: 5% 단계응답; 기준: 30분 내 안정화)"
)

FEWSHOT_USER_CASE_2 = """data:
{"PRESSURE": 108, "TEMPERATURE": 31.5, "RF_POWER": 1250, "GAS_FLOW_RATE": 160}
answer:
"""
FEWSHOT_EXPLANATION_2 = (
    "- PRESSURE=108 mTorr (정상 50–100): 압력 상한 초과 → 식각 균일도 저하 가능성.\n"
    "- RF_POWER=1250W (정상 800–1200): 전력 상한 초과 → 챔버 온도 상승과 플라즈마 불안정 유발.\n"
    "인과: 전력 상향이 가스 밀도·온도를 끌어올려 압력 상승을 동반했을 수 있습니다."
)

FEWSHOT_TASK_2 = (
    "1) 압력 상승 원인 추정(데이터: RF_POWER/CHAMBER_PRESSURE; 방법: 그랜저 인과성; 기준: p<0.05)\n"
    "2) 가스 유량 최적화(데이터: GAS_FLOW_RATE/균일도; 방법: DOE 2×2; 기준: 변동률<±3%)\n"
    "3) 파워 세트포인트 재설정(방법: 50W 단계감소; 기준: 압력 95 mTorr 재진입)"
)


# ---------------------------------------------------------------------
# 4) 공개 API
# ---------------------------------------------------------------------

def event_explain(event_detect_analysis: Any, process_type: str) -> str:
    """
    이상치 설명 생성 (모든 도메인 동일 상세도)
    - process_type: 20개 키 중 하나 (예: 'semiconductor_etch_002')
    """
    _require_supported_process(process_type, "event_explain")
    spec = PROCESS_SPECS[process_type]
    system_prompt = _build_prompt(spec, mode="explain")
    fewshots = [(FEWSHOT_USER_CASE_1, FEWSHOT_EXPLANATION_1),
                (FEWSHOT_USER_CASE_2, FEWSHOT_EXPLANATION_2)]
    print(f"[event_explain] Using spec: {process_type} -> {spec['title']}")

    prompt = system_prompt + "\n\n"
    for user_ex, assistant_ex in fewshots:
        prompt += f"User:\n{user_ex}\nAssistant:\n{assistant_ex}\n"
    prompt += f"User: data:\n{event_detect_analysis}\n\nanswer:\nAssistant:"
    response = LLMCallManager.invoke(prompt=prompt, max_tokens=len(prompt)+1024, is_json=False)
    return response


def event_cause_candidates(event_detect_analysis: Any, process_type: str) -> str:
    """
    분석 과업 생성 (모든 도메인 동일 상세도)
    - process_type: 20개 키 중 하나 (예: 'steel_production_004')
    """
    _require_supported_process(process_type, "event_cause_candidates")
    spec = PROCESS_SPECS[process_type]
    system_prompt = _build_prompt(spec, mode="task")
    fewshots = [(FEWSHOT_USER_CASE_1, FEWSHOT_TASK_1),
                (FEWSHOT_USER_CASE_2, FEWSHOT_TASK_2)]
    print(f"[event_cause_candidates] Using spec: {process_type} -> {spec['title']}")

    prompt = system_prompt + "\n\n"
    for user_ex, assistant_ex in fewshots:
        prompt += f"User:\n{user_ex}\nAssistant:\n{assistant_ex}\n"
    prompt += f"User: data:\n{event_detect_analysis}\n\nanswer:\nAssistant:"
    response = LLMCallManager.invoke(prompt=prompt, max_tokens=len(prompt)+1024, is_json=False)

    return response


# ---------------------------------------------------------------------
# 5) 로컬 테스트
# ---------------------------------------------------------------------
if __name__ == "__main__":
    url = "0.0.0.0:8888"

    # 예시 1: 반도체 Etch
    data1 = '{"PRESSURE":108,"TEMPERATURE":31.5,"RF_POWER":1250,"GAS_FLOW_RATE":160}'
    print("=== Semiconductor Etch - 설명 ===")
    print(event_explain(url, data1, "semiconductor_etch_002"))
    print("\n=== Semiconductor Etch - 과업 ===")
    print(event_cause_candidates(url, data1, "semiconductor_etch_002"))

    # 예시 2: 철강 Rolling
    data2 = '{"THICKNESS":4.3,"ROLL_GAP":4.4,"ROLLING_SPEED":210,"TENSION":305,"TEMPERATURE":890}'
    print("\n=== Steel Rolling - 설명 ===")
    print(event_explain(url, data2, "steel_rolling_001"))
    print("\n=== Steel Rolling - 과업 ===")
    print(event_cause_candidates(url, data2, "steel_rolling_001"))