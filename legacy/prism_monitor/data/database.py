import requests
import pandas as pd

from urllib.parse import urljoin

class PrismCoreDataBase:
    def __init__(self, base_url):
        if not base_url.endswith("/"):
            base_url += "/"
        self.base_url = base_url
        self.session = requests.Session()

    def get_tables(self):
        url = urljoin(self.base_url, 'api/db/tables')
        return requests.get(url, verify=False).json()['tables']
    
    def get_table_schema(self, table_name):
        url = urljoin(self.base_url, f'api/db/tables/{table_name}/schema')
        return requests.get(url, verify=False).json()
    
    def get_table_data(self, table_name: str, page_size: int = 50):
        all_rows = []
        offset = 0

        while True:
            url = urljoin(self.base_url, f"api/db/tables/{table_name}/data")
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

    
    