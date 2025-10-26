
import requests
import json
import pandas as pd
import pandasql as psql

from InstructionRF.prompt_manager import PromptManager
from prism_monitor.llm.api import temp_llm_call



def _query_decompose(user_query):
    manager = PromptManager('prism_monitor/modules/query_decompose/db_decompose.yaml')
    prompt = manager.get_full_prompt(user_query)
    res = temp_llm_call(prompt)
    print(res)
    table = res['classified_class']
    if table == 'semi_cmp_sensors':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_semi_cmp_sensors.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_CMP_SENSORS.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['timestamp'])
    elif table == 'semi_etch_sensors':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_semi_etch_sensors.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_ETCH_SENSORS.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['timestamp'])
    elif table == 'semi_cvd_sensors':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_semi_cvd_sensors.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_CVD_SENSORS.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['timestamp'])
    elif table == 'semi_ion_sensors':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_semi_ion_sensors.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_ION_SENSORS.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['timestamp'])
    elif table == 'semi_photo_sensors':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_semi_photo_sensors.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_PHOTO_SENSORS.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['timestamp'])
    elif table == 'semi_lot_manage':
        manager = PromptManager('prism_monitor/modules/query_decompose/query_lot_manage.yaml')
        prompt = manager.get_full_prompt(user_query)
        res = temp_llm_call(prompt)
        query = res['sql']
        df = pd.read_csv('prism_monitor/data/Industrial_DB_sample/SEMI_LOT_MANAGE.csv')
        result_df = psql.sqldf(query, locals())
        timestamp = pd.to_datetime(result_df['credate'])
    
    if len(timestamp):
        timestamp_min, timestamp_max = timestamp.min(), timestamp.max()
    else:
        timestamp_min, timestamp_max = None, None
    return timestamp_min, timestamp_max, result_df

def query_decompose(user_query):
    try:
        timestamp_min, timestamp_max, result_df = _query_decompose(user_query)
    except Exception as e:
        print(e)
        timestamp_min, timestamp_max, result_df = None, None, None
    return timestamp_min, timestamp_max, result_df