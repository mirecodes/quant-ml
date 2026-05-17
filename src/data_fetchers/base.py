# src/data_fetchers/base.py
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from src.utils.io import save_parquet, load_parquet

class BaseFetcher(ABC):
    """모든 데이터 소스의 공통 인터페이스."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """원본 데이터 수집."""
        pass
    
    def save(self, df: pd.DataFrame, name: str):
        save_parquet(df, str(self.cache_dir / f"{name}.parquet"))
    
    def load(self, name: str) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"
        return load_parquet(str(path)) if path.exists() else None
