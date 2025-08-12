import os
import sqlite3
import pandas as pd

from glob import glob


# class Database:
#     def __init__(self, db_path):
#         self.db_path = db_path
#         self.conn = None
#         self.cursor = None

#     def connect(self):
#         self.conn = sqlite3.connect(self.db_path)
#         self.cursor = self.conn.cursor()

#     def execute(self, query, params=None):
#         if params is None:
#             params = ()
#         self.cursor.execute(query, params)
#         self.conn.commit()
#         return self.cursor

#     def fetchall(self):
#         return self.cursor.fetchall()

#     def fetchone(self):
#         return self.cursor.fetchone()

#     def close(self):
#         if self.cursor:
#             self.cursor.close()
#         if self.conn:
#             self.conn.close()


def generate_temp_database(db_path='my_database.db', local_file_dir='prism_monitor/data/local'):
    conn = sqlite3.connect(db_path)
    local_files_path = glob(f'{local_file_dir}/*.csv')
    for local_file_path in local_files_path:
        table_name = os.path.basename(local_file_path).replace('.csv','')
        df = pd.read_csv(local_file_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()