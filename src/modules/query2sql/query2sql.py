
import requests
import json
import difflib
import pandas as pd
import pandasql as psql

from pathlib import Path
from src.modules.llm.prompt_manager import PromptManager
from src.modules.llm.llm import LLMCallManager
from src.test_scenarios.modeling import TEST_SCENARIOS_DATA_MAPPING

def _query2sql(user_query):
    timestamp_start, timestamp_end, sub_class, result_df, description, res = None, None, None, None, '', {}
    try:
        manager = PromptManager(Path(__file__).parent.resolve() / 'scenarios_prompt_semionly.yaml')
        prompt = manager.get_full_prompt(user_query)
        # Use generate endpoint with messages format and thinking disabled
        res = LLMCallManager.llm_agent_invoke_bimatrix(prompt=prompt, is_json=True)
        print(res)
        cls = res['class']
        sub_class = res['subclass']
        description = res['description']
        sql_query = res['sql'] 
        if TEST_SCENARIOS_DATA_MAPPING.get(sub_class) is None:
            print(f"알 수 없는 소분류입니다: {sub_class}. 가장 유사한 소분류를 찾습니다.")
            possible_subclasses = list(TEST_SCENARIOS_DATA_MAPPING.keys())
            sub_class = difflib.get_close_matches(sub_class, possible_subclasses, n=1, cutoff=.0)[0]
            print(f"가장 유사한 소분류로 대체합니다: {sub_class}")
            sql_query = 'SELECT * FROM df ORDER BY TIMESTAMP DESC LIMIT 100;'
            print(f"기본 SQL 쿼리로 대체합니다: {sql_query}")

        df = pd.read_csv(TEST_SCENARIOS_DATA_MAPPING[sub_class])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.strftime('%Y-%m-%d %H:%M:%S')
        result_df = psql.sqldf(sql_query, locals())
        print(result_df)
        if result_df.shape[0] > 0:
            timestamp = pd.to_datetime(result_df['TIMESTAMP'])
            timestamp_start, timestamp_end = timestamp.min(), timestamp.max()
        else:
            print("SQL 쿼리 결과가 없습니다.")
            result_df = df
            print("기본 데이터프레임으로 대체합니다.")
            timestamp_start, timestamp_end = df['TIMESTAMP'].min(), df['TIMESTAMP'].max()
    except Exception as e:
        print(f"SQL 실행 중 오류 발생: {e}")
    return timestamp_start, timestamp_end, sub_class, result_df, description, res


def query2sql(user_query: str = "", serialize: bool = False):
    timestamp_min, timestamp_max, sub_class, result_df, description, res = _query2sql(user_query)
    if serialize:
        return res
    return timestamp_min, timestamp_max, sub_class, result_df, description, res
