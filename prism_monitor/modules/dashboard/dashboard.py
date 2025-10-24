"""
dashboard.py
------------------
- 각 산업(industry) / 공정(process)별 CSV 파일을 10초 간격으로 읽어
  JSON 형식으로 출력 (대시보드용 실시간 스트림 시뮬레이터)
"""

import os
import time
import json
import pandas as pd
from datetime import datetime, timezone
import numpy as np


def send_dashboard_data(test_data_dir, interval= 10):
    DATA_INFO = {
        # '산업' / '공정' / '라인 식별자' 컬럼 정보
        "automotive_welding_001.csv": ("Automotive", "Welding", "LINE_ID"),
        "automotive_painting_002.csv": ("Automotive", "Painting", "BOOTH_ID"),
        "automotive_press_003.csv": ("Automotive", "Press", "PRESS_ID"),
        "automotive_assembly_004.csv": ("Automotive", "Assembly", "STATION_ID"),
        "battery_formation_001.csv": ("Battery", "Formation", "CELL_ID"),
        "battery_coating_002.csv": ("Battery", "Coating", "LINE_ID"),
        "battery_aging_003.csv": ("Battery", "Aging", "CELL_ID"),
        "battery_production_004.csv": ("Battery", "Production", "PRODUCTION_LINE"),
        "chemical_reactor_001.csv": ("Chemical", "Reactor", "REACTOR_ID"),
        "chemical_distillation_002.csv": ("Chemical", "Distillation", "TOWER_ID"),
        "chemical_refining_003.csv": ("Chemical", "Refining", "REFINE_ID"),
        "chemical_full_004.csv": ("Chemical", "Full", "PROCESS_ID"),
        "semiconductor_cmp_001.csv": ("Semiconductor", "CMP", "SENSOR_ID"),
        "semiconductor_etch_002.csv": ("Semiconductor", "Etch", "CHAMBER_ID"),
        "semiconductor_deposition_003.csv": ("Semiconductor", "Deposition", "CHAMBER_ID"),
        "semiconductor_full_004.csv": ("Semiconductor", "Full", "EQUIPMENT_ID"),
        "steel_rolling_001.csv": ("Steel", "Rolling", "LINE_ID"),
        "steel_converter_002.csv": ("Steel", "Converter", "CONVERTER_ID"),
        "steel_casting_003.csv": ("Steel", "Casting", "CASTER_ID"),
        "steel_production_004.csv": ("Steel", "Production", "PRODUCTION_LINE"),
    }

    datasets = []

    for industry_dir in os.listdir(test_data_dir):
        industry_path = os.path.join(test_data_dir, industry_dir)
        if industry_dir.lower().endswith(".json"):
            continue
        if not os.path.isdir(industry_path):
            continue

        for csv_file in os.listdir(industry_path):
            industry, process, line_col = DATA_INFO[csv_file]
            file_path = os.path.join(industry_path, csv_file)
            df = pd.read_csv(file_path)

            exclude_cols = {"TIMESTAMP", line_col}
            metric_cols = [c for c in df.columns if c not in exclude_cols]

            datasets.append({
                "industry": industry,
                "process": process,
                "line": line_col,
                "metric_cols": metric_cols,
                "data": df
            })

    max_len = max(len(d["data"]) for d in datasets)

    for i in range(max_len):
        frame = []
        for d in datasets:
            df = d["data"]
            row = df.iloc[i]
            line_value = row[d["line"]]

            metrics = {
                col: (row[col].item() if isinstance(row[col], (np.generic,)) else row[col])
                for col in d["metric_cols"]
            }

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "industry": d["industry"],
                "process": d["process"],
                "line": line_value,
                "metrics": metrics
            }

            frame.append(payload)

        yield json.dumps(frame, ensure_ascii=False, indent=2)
        time.sleep(interval)

if __name__ == "__main__":
    for json_data in send_dashboard_data(
        test_data_dir="/home/oesnuy/project/agi/PRISM-Monitor-1/prism_monitor/test-scenarios/test_data",
        interval=10
    ):
        print(json_data)