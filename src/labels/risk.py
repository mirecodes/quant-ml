# src/labels/risk.py
import pandas as pd
import numpy as np

def compute_risk(
    prices_quarterly: pd.DataFrame,
    max_horizon_years: int = 5,
    min_forward_quarters: int = 4,
) -> pd.DataFrame:
    """위험도 라벨 (단일 값)."""
    max_quarters = max_horizon_years * 4
    prices_quarterly = prices_quarterly.sort_values(['ticker', 'date'])
    results = []
    
    for ticker, group in prices_quarterly.groupby('ticker', observed=True, sort=False):
        closes = group['close'].to_numpy(dtype=np.float32)
        dates = group['date'].to_numpy()
        n = len(closes)
        
        # log return 계산
        if n < 2:
            continue
        log_rets = np.log(closes[1:] / (closes[:-1] + 1e-8))
        
        for i in range(n - 1):
            ret_start = i
            ret_end = min(i + max_quarters, n - 1)
            forward_rets = log_rets[ret_start : ret_end]
            
            # NaN 제외 유효 리턴 필터링
            valid = forward_rets[~np.isnan(forward_rets) & ~np.isinf(forward_rets)]
            
            if len(valid) < min_forward_quarters:
                continue
            
            # 위험도: 미래 분기 로그수익률의 연환산 표준편차 (분기 데이터이므로 sqrt(4))
            R = valid.std(ddof=1) * np.sqrt(4)
            
            results.append({
                'ticker': ticker,
                'date': dates[i],
                'R': np.float32(R),
                'R_quarters_used': np.int8(len(valid)),
            })
            
    return pd.DataFrame(results)
