import requests
import json
import numpy as np
import pandas as pd
import pandasql as psql

from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

from prism_core.core.tools.base import BaseTool
from prism_core.core.tools.schemas import ToolRequest, ToolResponse


class PrismDataBaseTool(BaseTool):
    """
    데이터베이스 도구
    """
    def __init__(self, 
                 database_url: Optional[str] = None,
                 client_id: str = "monitoring",
    ):
        super().__init__(
            name="query_raw_data",
            description="주어진 SQL 쿼리를 실행하여 데이터를 조회",
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
            url = urljoin(self.base_url, 'api/db/query')
            req_data = {
                "query": sql_query,
                "params": []
            }
            response = self.session.post(url, json=req_data, timeout=10, verify=False)
            response.raise_for_status()
            res = response.json()['data']
            
            return ToolResponse(
                success=True,
                data={
                    "res": res
                }
            )
                
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"Tool 실행 중 오류 발생: {str(e)}"
            )
