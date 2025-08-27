import requests
import json
import numpy as np
import pandas as pd

from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

from models.anomaly_detect import SemiconductorRealDataDetector
from prism_core.core.tools.base import BaseTool
from prism_core.core.tools.schemas import ToolRequest, ToolResponse


class AnomalyDataBaseTool(BaseTool):
    """
    이상치 탐지 데이터베이스 도구
    """
    def __init__(self, 
                 database_url: Optional[str] = None,
                 client_id: str = "monitoring",
    ):
        super().__init__(
            name="anomaly_database_tool",
            description="주어진 SQL 쿼리를 실행하여 이상치 탐지에 필요한 데이터를 조회",
            parameters_schema={
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "실행할 SQL 쿼리"}
                },
                "required": ["sql_query"]
            }
        )
        # 에이전트별 설정 또는 기본값 사용
        self._database_url = database_url
        self._client_id = client_id
        self._detector = SemiconductorRealDataDetector()

        # 공용 세션 (재시도 전략 적용)
        retry_strategy = Retry(
            total=2,  # 재시도 횟수 (즉, 최초 1회 + 재시도 2회 = 최대 3회 시도)
            backoff_factor=1,  # 지수 백오프 (1s, 2s, 4s)
            status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 상태 코드
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def execute(self, request: ToolRequest) -> ToolResponse:
        """Tool 실행"""
        try:
            params = request.parameters
            sql_query = params["sql_query"]
            
            # 1. 관련 규정 검색
            datasets = self.get_datasets()
            unified_df = self._detector.create_unified_dataset(datasets)
            lot_no_list = self.get_lot_no_list(sql_query)
            selected_df = unified_df[unified_df['lot_no'].isin(lot_no_list)]
            
            return ToolResponse(
                success=True,
                data={
                    "result": selected_df
                }
            )
                
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"Tool 실행 중 오류 발생: {str(e)}"
            )

    def get_table_list(self) -> List[str]:
        """데이터베이스의 테이블 목록 조회"""
        # 1. API 호출
        # 2. 응답 파싱
        # 3. 테이블 목록 반환
        url = urljoin(self._database_url, 'api/db/tables')
        response = self.session.get(url, timeout=5, verify=False)
        response.raise_for_status()
        res = response.json()['tables']
        return res
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """특정 테이블의 스키마 조회"""
        url = urljoin(self._database_url, f'api/db/tables/{table_name}/schema')
        response = self.session.get(url, timeout=5, verify=False)
        response.raise_for_status()
        return response.json()

    def get_table_data(self, table_name: str, page_size: int = 50) -> Any:
        """특정 테이블의 데이터 조회 (페이징 처리)"""
        all_rows = []
        offset = 0

        while True:
            url = urljoin(self._database_url, f"api/db/tables/{table_name}/data")
            params = {"limit": page_size, "offset": offset}

            resp = self.session.get(url, params=params, verify=False, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            rows = payload.get("data", [])

            if not rows:
                print("[INFO] 더 이상 행이 없습니다. 루프 종료.")
                break

            all_rows.extend(rows)

            if len(rows) < page_size:
                print("[INFO] 마지막 페이지 도달.")
                break

            offset += page_size

        df = pd.DataFrame(all_rows)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")  # 숫자로 변환 가능한 건 자동 변환
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
    
    def get_datasets(self) -> Dict[str, pd.DataFrame]:
        datasets = {}
        for table_name in self.get_table_list():
            df = self.get_table_data(table_name)
            datasets[table_name] = df

        return datasets
    
    def get_lot_no_list(self, sql_query: str) -> List[str]:
        """LOT 번호 목록 조회"""
        url = urljoin(self._database_url, 'api/db/query')
        req_data = {
            "query": sql_query,
            "params": []
        }
        response = self.session.post(url, json=req_data, timeout=60, verify=False)
        response.raise_for_status()
        res = response.json()['data']
        lot_no_list = [row['lot_no'] for row in res if 'lot_no' in row]
        return lot_no_list