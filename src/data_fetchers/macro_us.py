# src/data_fetchers/macro_us.py
from fredapi import Fred
import pandas as pd
import os
from .base import BaseFetcher
from src.utils.io import optimize_dtypes

FRED_SERIES = {
    'M_INT_001': 'DFF',           # Fed Funds Rate
    'M_INT_002': 'DGS2',          # 2Y Treasury
    'M_INT_003': 'DGS10',         # 10Y Treasury
    'M_LIQ_002': 'M2SL',          # M2 Money Stock
    'M_INF_001': 'CPIAUCSL',      # CPI
    'M_INF_002': 'CPILFESL',      # Core CPI
    'M_ECO_004': 'NAPM',          # ISM Manufacturing PMI
    'M_ECO_008': 'UNRATE',        # Unemployment
    'M_SNT_001': 'VIXCLS',        # VIX
}

class FredMacroFetcher(BaseFetcher):
    
    def __init__(self, cache_dir, api_key=None):
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not set in environment or arguments.")
        self.fred = Fred(api_key=self.api_key)
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        cached = self.load("macro_us_raw")
        if cached is not None:
            print("Loaded US macro from cache.")
            return cached
            
        print("Fetching FRED US macro indicators...")
        all_series = {}
        for indicator_id, fred_id in FRED_SERIES.items():
            try:
                # FRED API 호출
                series = self.fred.get_series(fred_id, start_date, end_date)
                all_series[indicator_id] = series
            except Exception as e:
                print(f"Error fetching FRED series {fred_id} ({indicator_id}): {e}")
                continue
                
        if not all_series:
            print("Warning: No FRED data fetched.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_series)
        # 분기말 리샘플링 (마지막 값)
        df = df.resample('QE').last()
        df.index.name = 'date'
        df = df.reset_index()
        
        df = optimize_dtypes(df)
        
        self.save(df, "macro_us_raw")
        return df
