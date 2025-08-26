import requests
import json
import numpy as np
import pandas as pd

from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

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
            name="query_data_for_anomaly_detection",
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
            unified_df = self.create_unified_dataset(datasets)
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
    
    def integrate_sensor_data(self, datasets):
        """
        여러 센서 테이블을 통합하여 하나의 센서 데이터셋 생성
        """
        print("센서 데이터 통합 중...")
        
        sensor_tables = ['semi_photo_sensors', 'semi_etch_sensors', 'semi_cvd_sensors', 
                        'semi_implant_sensors', 'semi_cmp_sensors']
        
        integrated_sensors = []
        
        for table_name in sensor_tables:
            df = datasets[table_name].copy()
            
            common_cols = ['pno', 'equipment_id', 'lot_no', 'timestamp']

            # 실제 존재하는 컬럼만 선택
            available_common = [col for col in common_cols if col in df.columns]
            
            # 수치형 센서 컬럼들 찾기
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # PNO 제외 (ID라서)
            sensor_cols = [col for col in numeric_cols if col != 'pno']
            
            # 테이블 정보 추가
            df['sensor_table'] = table_name.replace('_sensors', '')
            
            # 센서값들을 하나의 컬럼으로 변환 (Long format)
            df_long = df.melt(
                id_vars=available_common + ['sensor_table'],
                value_vars=sensor_cols,
                var_name='sensor_type',
                value_name='sensor_value'
            )
            integrated_sensors.append(df_long)
            print(f"  - {table_name}: {len(sensor_cols)}개 센서, {len(df)}개 레코드")
        
        result = pd.concat(integrated_sensors, ignore_index=True)
        print(f"통합 완료: 총 {len(result)}개 센서 레코드")
        return result

        
    def create_unified_dataset(self, datasets):
        """
        모든 테이블을 통합하여 분석용 데이터셋 생성
        """
        print("통합 데이터셋 생성 중...")
        
        # 1. 센서 데이터 통합
        integrated_sensors = self.integrate_sensor_data(datasets)
        
        # 2. LOT 관리 데이터 기준으로 통합

        main_df = datasets['lot_manage'].copy()
        print(f"기본 LOT 데이터: {len(main_df)}개 LOT")

        
        # 3. 각 LOT별 센서 통계 생성
        # LOT별 센서 통계 계산
        sensor_stats = integrated_sensors.groupby(['lot_no', 'sensor_type'])['sensor_value'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Wide format으로 변환
        sensor_features = sensor_stats.pivot_table(
            index='lot_no',
            columns='sensor_type',
            values=['mean', 'std', 'min', 'max'],
            fill_value=0
        )
        
        # 컬럼명 정리
        sensor_features.columns = [f"{stat}_{sensor}" for stat, sensor in sensor_features.columns]
        sensor_features = sensor_features.reset_index()
        
        # LOT 데이터와 조인
        main_df = main_df.merge(sensor_features, on='lot_no', how='left')
        print(f"센서 특성 추가 완료: {sensor_features.shape[1]-1}개 특성")
        
        # 4. 공정 이력 데이터 통합
        process_df = datasets['process_history']

        # LOT별 공정 통계
        process_stats = process_df.groupby('lot_no').agg({
            'in_qty': ['mean', 'sum'],
            'out_qty': ['mean', 'sum'],
        }).reset_index()
        
        process_stats.columns = [f"process_{col[0]}_{col[1]}" if col[1] else col[0] 
                                for col in process_stats.columns]
        process_stats.columns = [col.replace('process_lot_no_', 'lot_no') for col in process_stats.columns]
        
        main_df = main_df.merge(process_stats, on='lot_no', how='left')
        print(f"공정 이력 특성 추가 완료")
    
    # 5. 파라미터 측정 데이터 통합
        param_df = datasets['param_measure']
        # LOT별 파라미터 통계
        param_stats = param_df.groupby('lot_no')['measured_val'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        param_stats.columns = [f"param_{col}" if col != 'lot_no' else col 
                                for col in param_stats.columns]
        
        main_df = main_df.merge(param_stats, on='lot_no', how='left')
        print(f"파라미터 측정 특성 추가 완료")
        
        print(f"최종 통합 데이터셋: {main_df.shape}")
        return main_df
    
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