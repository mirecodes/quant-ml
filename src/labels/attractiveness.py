# src/labels/attractiveness.py
import pandas as pd
import numpy as np

def compute_attractiveness(
    prices_quarterly: pd.DataFrame,
    max_horizon_years: int = 5,
    min_forward_quarters: int = 4,
) -> pd.DataFrame:
    """매력도 라벨 (단일 값)."""
    max_quarters = max_horizon_years * 4
    # v5.1 패치: log_5 고정
    log_base_value = np.log(max_horizon_years)
    
    prices_quarterly = prices_quarterly.sort_values(['ticker', 'date'])
    results = []
    
    for ticker, group in prices_quarterly.groupby('ticker', observed=True, sort=False):
        closes = group['close'].to_numpy(dtype=np.float32)
        dates = group['date'].to_numpy()
        n = len(closes)
        
        for i in range(n):
            p_t = closes[i]
            if p_t <= 0:
                continue
            
            end_idx = min(i + max_quarters, n - 1)
            forward = closes[i+1 : end_idx+1]
            
            if len(forward) < min_forward_quarters:
                continue
            
            max_price = forward.max()
            if max_price <= 0:
                continue
            
            # log_5(max_price / p_t)
            A = np.log(max_price / p_t) / log_base_value
            
            results.append({
                'ticker': ticker,
                'date': dates[i],
                'A': np.float32(A),
                'A_quarters_used': np.int8(len(forward)),
            })
    
    return pd.DataFrame(results)
