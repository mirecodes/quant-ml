# src/models/baseline_accounting.py
import pandas as pd
import numpy as np

class AccountingBaseline:
    """학술 합성 지표 baseline."""
    
    def __init__(self, baseline_type: str):
        """baseline_type: 'fscore', 'quality', 'composite', 'equal_1n'"""
        self.type = baseline_type
    
    def score(self, features: pd.DataFrame) -> pd.Series:
        """단순 공식으로 매력도 점수 산출 — 학습 없음."""
        features = features.copy()
        
        # 모사 컬럼이 rename 되었으므로 이름 맵핑 처리
        fscore_col = 'C_FSCORE' if 'C_FSCORE' in features.columns else features.columns[0]
        quality_col = 'C_QUALITY' if 'C_QUALITY' in features.columns else features.columns[0]
        
        if self.type == 'fscore':
            return features[fscore_col]
        
        elif self.type == 'quality':
            return features[quality_col]
        
        elif self.type == 'gp_a':
            gp_a = features['F_FUND_GROSS_PROFIT'] / (features['F_FUND_TOTAL_ASSETS'] + 1e-8)
            return self._zscore(gp_a)
        
        elif self.type == 'composite':
            gp_a = features['F_FUND_GROSS_PROFIT'] / (features['F_FUND_TOTAL_ASSETS'] + 1e-8)
            # PER 대용으로 mock F_FUND_EPS 사용
            per = features['close'] / (features['F_FUND_EPS'] + 1e-8)
            return (
                self._zscore(features[quality_col]) +
                self._zscore(features[fscore_col]) +
                self._zscore(gp_a) +
                -self._zscore(per)   # PER (역방향)
            ) / 4
        
        elif self.type == 'equal_1n':
            cols = [c for c in features.columns if c.startswith(('C_', 'F_FUND_'))]
            if not cols:
                return pd.Series(0.0, index=features.index)
            # numeric 컬럼만 필터링하여 안전하게 평균 계산
            num_features = features[cols].select_dtypes(include=[np.number])
            return num_features.apply(self._zscore).mean(axis=1)
        
        else:
            raise ValueError(self.type)
    
    @staticmethod
    def _zscore(s):
        s = s.fillna(0.0)
        std = s.std()
        if pd.isna(std) or std < 1e-8:
            return s - s.mean()
        return (s - s.mean()) / std
