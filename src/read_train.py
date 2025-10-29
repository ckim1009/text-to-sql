import json
import sqlparse
import sqlite3
from sqlparse.sql import Identifier, IdentifierList, Where
from sqlparse.tokens import Keyword, DML
from datasets import Dataset
import re
import sqlglot

def get_db_schemas_full(tables_data):
    # instruction-style 데이터셋 생성
    db_schemas_full = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        pk_ids = db["primary_keys"]
        fk_pairs = db["foreign_keys"]

        table_schemas = {}
        for table in table_names:
            # 컬럼 정의
            cols_def = []
            for idx, (t_idx, col_name) in enumerate(column_names):
                if t_idx == table_names.index(table):
                    col_type = column_types[idx]
                    cols_def.append(f"{col_name} {col_type}")

            # PK 정의
            table_pks = [column_names[pk_id][1] for pk_id in pk_ids if column_names[pk_id][0] == table_names.index(table)]
            if table_pks:
                pk_def = f"PRIMARY KEY ({', '.join(table_pks)})"
                cols_def.append(pk_def)

            # FK 정의
            table_fks = []
            for from_id, to_id in fk_pairs:
                from_t, from_c = column_names[from_id]
                to_t, to_c = column_names[to_id]
                if from_t == table_names.index(table):
                    fk_def = f"FOREIGN KEY ({from_c}) REFERENCES {table_names[to_t]}({to_c})"
                    cols_def.append(fk_def)

            # CREATE TABLE 문
            ddl = f"CREATE TABLE {table} (\n    " + ",\n    ".join(cols_def) + "\n);"
            table_schemas[table] = ddl

        db_schemas_full[db_id] = table_schemas
    return db_schemas_full


def load_ddl(path, train_data_path):
    """
    각 SQL에서 사용된 테이블만 포함하고,
    스키마를 SQL DDL(CREATE TABLE) 형식으로 표현
    """
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    with open(f'{path}/tables.json', "r") as f:
        tables_data = json.load(f)

   
    
    
    # instruction-style 데이터셋 생성
    db_schemas_full = get_db_schemas_full(tables_data)

    check = True

    # instruction-style 데이터셋 생성
    records = []
    for item in train_data:
        db_id = item["db_id"]
        question = item["question"]
        sql = item["query"]
        # ast = item["sql"]  # 이미 AST가 있음
        # schema = schema_map[db_id]

        # # SQL에서 사용된 테이블 추출
        # used_tables = set()
        # parsed = sqlparse.parse(sql)
        # for stmt in parsed:
        #     for token in stmt.tokens:
        #         token_str = str(token).lower()
        #         for table_name in db_schemas_full[db_id].keys():
        #             if table_name.lower() in token_str:
        #                 used_tables.add(table_name)
        used_tables = set()
        used_columns = set()
        try:
            tree = sqlglot.parse_one(sql)  # SQL 파싱
            for column in tree.find_all(sqlglot.exp.Column):
                column_name = column.name
                used_columns.add(column_name)
                
            for table in tree.find_all(sqlglot.exp.Table):
                table_name = table.name
                # 스키마에 있는 테이블만 필터링
                if table_name in db_schemas_full[db_id]:
                    used_tables.add(table_name)
        except Exception as e:
            print(f"SQL parsing error: {e}")

        # 사용된 테이블만 DDL로 선택
        selected_schema_parts = [db_schemas_full[db_id][t] for t in used_tables]
        selected_schema = "\n\n".join(selected_schema_parts)
        
        whole_schema = ""
        for t in db_schemas_full[db_id]:
            # print(db_schemas_full[db_id][t])
            whole_schema += db_schemas_full[db_id][t]
            whole_schema += "\n\n"

        # print(whole_schema)
        # if used_tables is None or None in used_tables:
        # print(item['query'])
        # print(used_columns)
        
        
        # selected_schema = "\n\n".join(ddls)
        
        if check:
            print(question)
            print(whole_schema)
            print(used_columns)
            print(sql)
            check=False
        
        records.append({
            "instruction": 
                '''You are an SQL translator. Your task is to translate the given natural language question into a valid SQL query.
- Follow the database schema strictly.
- Correctly infer which columns are needed from the schema to answer the question.
- Use JOINs, GROUP BY, ORDER BY, HAVING, subqueries, and other SQL constructs as needed to produce correct results.
- Output valid SQL only, no explanations or extra text.''',
            # "ast": f'{ast}',
            # "schema_map": f'{schema_map}',
            "whole_schema":f"{whole_schema}",
            "schema":f"{selected_schema}",
            "hint":f"The SQL query might involve the following columns: {used_columns}",
            "input": f"{question}",
            "output": sql
        })

    return Dataset.from_list(records)
