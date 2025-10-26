
import requests
import json
import pandas as pd
import pandasql as psql

from pathlib import Path
from InstructionRF.prompt_manager import PromptManager
from src.modules.llm.llm import llm_generate_bimatrix
from src.test_scenarios.modeling import TEST_SCENARIOS_DATA_MAPPING

def _query2sql(user_query, bimatrix_llm_url):
    timestamp_start, timestamp_end, sub_class, result_df, res = None, None, None, None, None
    manager = PromptManager(Path(__file__).parent.resolve() / 'scenarios_prompt.yaml')
    prompt = manager.get_full_prompt(user_query)
    res = llm_generate_bimatrix(bimatrix_llm_url, prompt)
    print(res)
    sub_class = res['classification_subclass']
    df = pd.read_csv(TEST_SCENARIOS_DATA_MAPPING[sub_class])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    query = res['sql']
    try:
        result_df = psql.sqldf(query, locals())
        if result_df.shape[0] > 0:
            timestamp = pd.to_datetime(result_df['TIMESTAMP'])
            timestamp_start, timestamp_end = timestamp.min(), timestamp.max()
    except Exception as e:
        print(f"SQL 실행 중 오류 발생: {e}")
    return timestamp_start, timestamp_end, sub_class, result_df, res


def query2sql(user_query, bimatrix_llm_url: str = "", serialize: bool = False):
    timestamp_min, timestamp_max, sub_class, result_df, res = _query2sql(user_query, bimatrix_llm_url)
    if serialize:
        return res
    return timestamp_min, timestamp_max, sub_class, result_df, res
