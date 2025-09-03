from dotenv import load_dotenv
load_dotenv()



import sys
sys.path.append('/home/minjoo/Github/prism_monitor/PRISM-Monitor')


import os
import json
import random
from datetime import datetime, timedelta
from prism_monitor.llm.api import llm_generate

LLM_URL = os.environ['LLM_URL']

def func(query: str):
    
    # LLM 프롬프트
    prompt = f'''너는 자연어에서 시간 범위를 추출하여 JSON으로 변환하는 파서다.

출력 규칙:
- JSON 객체만 출력, 다른 텍스트 절대 금지
- 시간 정보 없으면 start, end를 null로 설정
- 시간 형식: YYYY-MM-DDTHH:MM:SSZ
- 데이터 범위: 2024-01-01T00:00:00Z ~ 2024-01-02T13:29:00Z

예시:
입력: "2024년 1월 1일 데이터"
출력: {{"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T23:59:59Z"}}

입력: "분석 방법 알려줘"  
출력: {{"start": null, "end": null}}

요청: "{query}"
출력:'''

    # LLM 호출
    res = llm_generate(url=LLM_URL, prompt=prompt)
    
    # JSON 파싱
    try:
        res = json.loads(res.strip())
    except:
        res = {"start": None, "end": None}
    
    # null이면 랜덤 생성
    if res.get('end') is None or res.get('start') is None:
        # 데이터 존재 범위에서 랜덤 생성
        min_time = datetime(2024, 1, 1, 0, 0, 0)
        max_time = datetime(2024, 1, 2, 13, 29, 0)
        
        # 랜덤 시작 시간
        total_seconds = int((max_time - min_time).total_seconds())
        start_seconds = random.randint(0, total_seconds - 3600)
        start_time = min_time + timedelta(seconds=start_seconds)
        
        # 랜덤 기간 (30분~2시간)
        duration = random.randint(30, 120)
        end_time = start_time + timedelta(minutes=duration)
        
        if end_time > max_time:
            end_time = max_time
            
        res = {
            "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    
    return res

# 테스트
if __name__ == "__main__":
    queries = [
        "2024년 1월 1일 오전 데이터 보여줘",
        "1월 2일 13시 데이터",
        "이상치 분석 방법 알려줘",
        "30분간 데이터"
    ]
    
    for q in queries:
        result = func(q)
        print(f"'{q}' -> {result}")