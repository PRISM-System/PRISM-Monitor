from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
import streamlit as st
import os
import logging
import time

from tinydb import TinyDB, Query
from typing import Union, Literal
from urllib.parse import urljoin

from prism_monitor.data.database import PrismCoreDataBase

# Constants
DATABASE_PATH = "monitor_db.json"
LOCAL_FILE_DIR = 'prism_monitor/data/local'
LLM_URL = os.environ['LLM_URL']
MONITOR_DB = TinyDB(DATABASE_PATH)
PRISM_CORE_DB = PrismCoreDataBase(os.environ['PRISM_CORE_DATABASE_URL'])
BACKEND_URL = 'http://localhost:8000'

# Streamlit Config
st.set_page_config(layout="wide", page_title="PRISM Monitoring Workflow")
st.title("🔍 PRISM: Proactive Monitoring Workflow")
st.markdown("This dashboard provides a step-by-step interface for the PRISM industrial monitoring and analysis system.")

# User input
query = st.text_input("질문을 입력하세요", "2024년 1월부터 2월까지의 이상치를 탐색해줘")

# Button to trigger workflow
if st.button("입력"):
    with st.spinner("📡 이상치 탐색 중..."):
        # Step 1: Detect Anomalies
        detect_data = {
            "taskId": "TASK_0001",
            "start": "2024-01-01T12:00:00Z",
            "end": "2024-02-01T12:30:00Z"
        }
        detect_res = requests.post(urljoin(BACKEND_URL, '/api/v1/monitoring/event/detect'), json=detect_data).json()['result']

        st.success("1️⃣ 이상치 탐지 완료")
        with st.expander("🔎 탐지 결과 보기"):
            st.write('**Anomalies:**', str(detect_res['anomalies']))
            st.write(detect_res['svg'], unsafe_allow_html=True)

        Event = Query()
        event_record = MONITOR_DB.table('EventDetectHistory').get(Event.task_id == detect_data['taskId'])['records']
        record_df = pd.DataFrame(event_record)
        st.dataframe(record_df, use_container_width=True)

    with st.spinner("🧠 이상치 설명 생성 중..."):
        # Step 2: Explain Anomalies
        explain_data = {"taskId": "TASK_0001"}
        explain_res = requests.post(urljoin(BACKEND_URL, '/api/v1/monitoring/event/explain'), json=explain_data).json()

        st.success("2️⃣ 이상치 설명 생성 완료")
        with st.expander("💬 이상치 설명"):
            st.text_area("설명 결과", explain_res['explain'], height=200)

    with st.spinner("🔍 문제 원인 후보군 분석 중..."):
        # Step 3: Cause Candidates
        cause_data = {"taskId": "TASK_0001"}
        cause_res = requests.post(urljoin(BACKEND_URL, '/api/v1/monitoring/event/cause-candidates'), json=cause_data).json()

        st.success("3️⃣ 문제 원인 후보군 생성 완료")
        with st.expander("🧩 원인 후보군"):
            st.text_area("후보군", str(cause_res['causeCandidates']), height=150)

    with st.spinner("🔮 이상징후 예측 중..."):
        # Step 4: Precursor Prediction
        precursor_data = {
            "taskId": "TASK_0001",
            "start": "2024-01-01T12:00:00Z",
            "end": "2024-02-01T12:30:00Z"
        }
        precursor_res = requests.post(urljoin(BACKEND_URL, '/api/v1/monitoring/event/precursor'), json=precursor_data).json()

        st.success("4️⃣ 이상징후 예측 완료")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="예측값", value=precursor_res['summary']['predicted_value'])
        with col2:
            st.metric(label="이상 여부", value=str(precursor_res['summary']['is_anomaly']))

    with st.spinner("🛡 위험도 평가 중..."):
        # Step 5: Risk Evaluation
        risk_data = {
            "taskId": "TASK_0001",
            "topk": 5
        }
        risk_res = requests.post(urljoin(BACKEND_URL, '/api/v1/monitoring/event/evaluate-risk'), json=risk_data).json()

        st.success("5️⃣ 위험도 평가 완료")
        st.subheader("⚠ 현재 상태 위험도 평가")
        st.markdown("**Event Evaluation:**")
        st.write(risk_res['summary']['eventEvaluation'])
        st.markdown("**Prediction Evaluation:**")
        st.write(risk_res['summary']['predictionEvaluation'])
