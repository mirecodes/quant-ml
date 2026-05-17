# src/features/computed/quality_score.py
from ..base import BaseIndicator, IndicatorMeta
import pandas as pd
import numpy as np

class QualityScoreIndicator(BaseIndicator):
    meta = IndicatorMeta(
        id="C_QUALITY",
        name="Quality Score (Asness-style)",
        module="COMPUTED",
        category="quality",
        unit="z_score",
        frequency="quarterly",
        source="derived",
        countries=["KR", "US"],
        lag_days=45,
    )
    
    def compute(self, fin: pd.DataFrame) -> pd.Series:
        # 안전한 계산을 위해 컬럼 존재 여부 확인
        for col in ['gross_profit', 'total_assets', 'eps', 'debt', 'equity', 'dividends', 'net_income']:
            if col not in fin.columns:
                fin[col] = 1.0 if col in ['total_assets', 'equity'] else 0.0
                
        prof = self._zscore(fin['gross_profit'] / (fin['total_assets'] + 1e-8))
        growth = self._zscore(fin['eps'].pct_change(4))    # YoY
        safety = -self._zscore(fin['debt'] / (fin['equity'] + 1e-8))
        payout = self._zscore(fin['dividends'] / (fin['net_income'] + 1e-8))
        
        return prof + growth + safety + payout
    
    @staticmethod
    def _zscore(s):
        s = s.fillna(0.0)
        std = s.std()
        if pd.isna(std) or std < 1e-8:
            return s - s.mean()
        return (s - s.mean()) / std
