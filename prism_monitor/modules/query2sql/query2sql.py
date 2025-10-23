
import requests
import json
import pandas as pd
import pandasql as psql

from InstructionRF.prompt_manager import PromptManager
from prism_monitor.llm.api import temp_llm_call
from prism_monitor.modules.query_decompose.query_decompose import _query_decompose

"Assembly(Automotive) | Painting(Automotive) | Press(Automotive) | Welding(Automotive) | Aging(Battery) | Coating(Battery) | Formation(Battery) | Production(Battery) | Distillation(Chemical) | Full(Chemical) | Reactor(Chemical) | Refining(Chemical) | Cmp(Semiconductor) | Deposition(Semiconductor) | Etch(Semiconductor) | Full(Semiconductor) | Casting(Steel) | Converter(Steel) | Production(Steel) | Rolling(Steel)"
SUBCLASS_CSV_MAP = {
    "Assembly(Automotive)": "prism_monitor/test-scenarios/test_data/automotive/automotive_assembly_004.csv",
    "Painting(Automotive)": "prism_monitor/test-scenarios/test_data/automotive/automotive_painting_002.csv",
    "Press(Automotive)": "prism_monitor/test-scenarios/test_data/automotive/automotive_press_004.csv",
    "Welding(Automotive)": "prism_monitor/test-scenarios/test_data/automotive/automotive_welding_001.csv",
    "Aging(Battery)": "prism_monitor/test-scenarios/test_data/battery/battery_aging_003.csv",
    "Coating(Battery)": "prism_monitor/test-scenarios/test_data/battery/battery_coating_002.csv",
    "Formation(Battery)": "prism_monitor/test-scenarios/test_data/battery/battery_formation_001.csv",
    "Production(Battery)": "prism_monitor/test-scenarios/test_data/battery/battery_production_004.csv",
    "Distillation(Chemical)": "prism_monitor/test-scenarios/test_data/chemical/chemical_distillation_002.csv",
    "Full(Chemical)": "prism_monitor/test-scenarios/test_data/chemical/chemical_full_004.csv",
    "Reactor(Chemical)": "prism_monitor/test-scenarios/test_data/chemical/chemical_reactor_001.csv",
    "Refining(Chemical)": "prism_monitor/test-scenarios/test_data/chemical/chemical_refining_003.csv",
    "Cmp(Semiconductor)": "prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_cmp_001.csv",
    "Deposition(Semiconductor)": "prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_deposition_003.csv",
    "Etch(Semiconductor)": "prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_etch_002.csv",
    "Full(Semiconductor)": "prism_monitor/test-scenarios/test_data/semiconductor/semiconductor_full_004.csv",
    "Casting(Steel)": "prism_monitor/test-scenarios/test_data/steel/steel_casting_003.csv",
    "Converter(Steel)": "prism_monitor/test-scenarios/test_data/steel/steel_converter_002.csv",
    "Production(Steel)": "prism_monitor/test-scenarios/test_data/steel/steel_production_004.csv",
    "Rolling(Steel)": "prism_monitor/test-scenarios/test_data/steel/steel_rolling_001.csv",
}

def _query2sql(user_query):
    timestamp_start, timestamp_end = None, None
    manager = PromptManager('prism_monitor/modules/query2sql/scenarios_prompt.yaml')
    prompt = manager.get_full_prompt(user_query)
    res = temp_llm_call(prompt)
    print(res)
    sub_class = res['classification_subclass']
    df = pd.read_csv(SUBCLASS_CSV_MAP[sub_class])
    query = res['sql']
    result_df = psql.sqldf(query, locals())
    if result_df.shape[0] > 0:
        timestamp = pd.to_datetime(result_df['TIMESTAMP'])
        timestamp_start, timestamp_end = timestamp.min(), timestamp.max()
    return timestamp_start, timestamp_end, result_df, res


def query2sql(user_query):
    try:
        timestamp_min, timestamp_max, result_df, res = _query2sql(user_query)
    except Exception as e:
        print(e)
        timestamp_min, timestamp_max, result_df, res = None, None, None, None
    return timestamp_min, timestamp_max, result_df, res