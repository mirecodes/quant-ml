# src/labels/attractiveness.py
import pandas as pd
import numpy as np

def compute_attractiveness(
    prices_quarterly: pd.DataFrame,
    max_horizon_years: int = 5,
    min_forward_quarters: int = 8,
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
            
            # 가용 기간(분기) 기반 N년 계산
            n_quarters = len(forward)
            n_years = n_quarters / 4.0
            
            # 최대 horizon_years(5년) 이내에서, N년 기간을 기준으로 log base 설정
            base_years = min(n_years, max_horizon_years)
            # log(1)=0 이므로 수학적 안정성을 위해 최소 2.0 적용 (하한값 상향)
            base_years = max(base_years, 2.0)
            
            log_base_value = np.log(base_years)
            
            # log_N(max_price / p_t)
            A = np.log(max_price / p_t) / log_base_value
            
            results.append({
                'ticker': ticker,
                'date': dates[i],
                'A': np.float32(A),
                'A_quarters_used': np.int8(n_quarters),
            })
    
    return pd.DataFrame(results)
