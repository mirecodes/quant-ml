# src/evaluation/metrics.py
import numpy as np
import pandas as pd
from scipy import stats

def spearman_ic(predicted, realized):
    """Spearman rank IC (NaN handling)."""
    predicted = np.asarray(predicted)
    realized = np.asarray(realized)
    mask = ~(np.isnan(predicted) | np.isnan(realized) | np.isinf(predicted) | np.isinf(realized))
    if mask.sum() < 5:  # 디버깅/테스트용 최소 표본 수 완화
        return np.nan
    return stats.spearmanr(predicted[mask], realized[mask])[0]

def icir(ic_series):
    """IC Information Ratio."""
    ic_series = pd.Series(ic_series).dropna()
    if len(ic_series) < 2:
        return np.nan
    std = ic_series.std()
    return ic_series.mean() / (std if std > 1e-8 else 1.0)

def decile_returns(predictions, realized, n_deciles=5): # 소규모 데이터셋을 위해 분위수를 5개로 완화
    """분위별 실현값 평균."""
    df = pd.DataFrame({'pred': predictions, 'real': realized}).dropna()
    if len(df) < n_deciles:
        return pd.Series(0.0, index=range(1, n_deciles + 1))
    df['decile'] = pd.qcut(df['pred'].rank(method='first'), n_deciles, 
                            labels=False, duplicates='drop') + 1
    return df.groupby('decile')['real'].mean()

def long_short_spread(predictions, realized, n_deciles=5):
    """D5 - D1 스프레드 (최대 분위 - 최소 분위)."""
    deciles = decile_returns(predictions, realized, n_deciles)
    if deciles.empty:
        return 0.0
    return deciles.iloc[-1] - deciles.iloc[0]

def diebold_mariano(errors1, errors2, h=1):
    """두 모델 예측 오차의 통계적 유의성 비교."""
    d = errors1**2 - errors2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var < 1e-8:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
