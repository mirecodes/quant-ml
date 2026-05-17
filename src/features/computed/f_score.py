# src/features/computed/f_score.py
from ..base import BaseIndicator, IndicatorMeta
import pandas as pd

class FScoreIndicator(BaseIndicator):
    meta = IndicatorMeta(
        id="C_FSCORE",
        name="Piotroski F-Score",
        module="COMPUTED",
        category="financial_health",
        unit="score",
        frequency="quarterly",
        source="derived",
        countries=["KR", "US"],
        lag_days=45,
    )
    
    def compute(self, fin: pd.DataFrame) -> pd.Series:
        """fin: 종목별 분기 재무제표 DataFrame."""
        scores = pd.DataFrame(index=fin.index)
        
        # 9개 항목 계산 (각 컬럼이 존재하는지 체크하여 안전하게 처리)
        for col in ['roa', 'cfo', 'net_income', 'leverage', 'current_ratio', 'shares', 'gross_margin', 'asset_turnover']:
            if col not in fin.columns:
                fin[col] = 0.0
                
        scores['c1'] = (fin['roa'] > 0).astype(int)
        scores['c2'] = (fin['cfo'] > 0).astype(int)
        scores['c3'] = (fin['roa'] > fin['roa'].shift(4)).astype(int)
        scores['c4'] = (fin['cfo'] > fin['net_income']).astype(int)
        scores['c5'] = (fin['leverage'] < fin['leverage'].shift(4)).astype(int)
        scores['c6'] = (fin['current_ratio'] > fin['current_ratio'].shift(4)).astype(int)
        scores['c7'] = (fin['shares'] <= fin['shares'].shift(4)).astype(int)
        scores['c8'] = (fin['gross_margin'] > fin['gross_margin'].shift(4)).astype(int)
        scores['c9'] = (fin['asset_turnover'] > fin['asset_turnover'].shift(4)).astype(int)
        
        return scores.sum(axis=1)
