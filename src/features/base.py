# src/features/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class IndicatorMeta:
    id: str
    name: str
    module: str          # MACRO, FUNDAMENTAL, KOREAN_ASSET, COMPUTED
    category: str
    unit: str
    frequency: str       # quarterly
    source: str
    countries: list
    lag_days: int = 45   # PiT lag

class BaseIndicator(ABC):
    meta: IndicatorMeta
    
    @abstractmethod
    def compute(self, raw_data: pd.DataFrame) -> pd.Series:
        pass
    
    def to_quarterly(self, series: pd.Series) -> pd.Series:
        """분기말 값으로 표준화."""
        return series.resample('QE').last()
