# src/utils/io.py
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DTYPE_POLICY = {
    'ticker':       'category',
    'country':      'category',
    'sector':       'category',
    'size_tier':    'category',
    'currency':     'category',
    'date':         'datetime64[ns]',
    'close':        'float32',
    'volume':       'float32',
    'market_cap':   'float32',
}

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in DTYPE_POLICY:
            try:
                df[col] = df[col].astype(DTYPE_POLICY[col])
            except Exception:
                pass
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        elif pd.api.types.is_object_dtype(df[col]):
            # 리스트와 같이 unhashable 타입이 섞여있을 때의 예외 처리 추가
            try:
                if df[col].nunique() / max(len(df), 1) < 0.5:
                    df[col] = df[col].astype('category')
            except TypeError:
                pass
    return df

def save_parquet(df: pd.DataFrame, path: str):
    df = optimize_dtypes(df)
    df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine='pyarrow')
    return optimize_dtypes(df)

def report_memory(df: pd.DataFrame, label: str = ""):
    bytes_size = df.memory_usage(deep=True).sum()
    if bytes_size >= 1e5:  # 100 KB 이상
        mem_mb = bytes_size / 1e6
        print(f"[{label}] shape={df.shape}, memory={mem_mb:.2f} MB")
    else:
        mem_kb = bytes_size / 1024
        print(f"[{label}] shape={df.shape}, memory={mem_kb:.2f} KB")
