# tests/test_pipeline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from src.labels.attractiveness import compute_attractiveness
from src.labels.risk import compute_risk
from src.utils.io import optimize_dtypes

def test_attractiveness_label():
    """A_5Y 계산이 max-window 및 log_5 기반인지 검증."""
    # 5분기 동안의 sample price 데이터 생성
    sample_df = pd.DataFrame({
        'ticker': ['TEST'] * 6,
        'date': pd.date_range(start='2020-01-01', periods=6, freq='QE'),
        'close': [100.0, 110.0, 105.0, 120.0, 130.0, 125.0]
    })
    
    # 5년 max horizon (max_horizon_years=5, min_forward_quarters=1)
    a_df = compute_attractiveness(sample_df, max_horizon_years=5, min_forward_quarters=1)
    
    # t=0 (price=100) 일 때, 미래 5분기의 max price는 130
    # expected = log_5(130/100) = log_5(1.3)
    p_t = 100.0
    max_price = 130.0
    expected = np.log(max_price / p_t) / np.log(5)
    
    actual = a_df.loc[a_df['date'] == '2020-03-31', 'A'].values[0]
    assert abs(actual - expected) < 1e-6
    print(f"Attractiveness test passed: Expected {expected:.6f}, got {actual:.6f}")

def test_risk_label():
    """위험도(R) 계산이 미래 로그수익률의 연환산 표준편차인지 검증."""
    sample_df = pd.DataFrame({
        'ticker': ['TEST'] * 5,
        'date': pd.date_range(start='2020-01-01', periods=5, freq='QE'),
        'close': [100.0, 105.0, 102.0, 108.0, 104.0]
    })
    
    r_df = compute_risk(sample_df, max_horizon_years=5, min_forward_quarters=1)
    
    # log returns:
    # 105/100 -> ln(1.05)
    # 102/105 -> ln(102/105)
    # 108/102 -> ln(108/102)
    # 104/108 -> ln(104/108)
    closes = np.array([100.0, 105.0, 102.0, 108.0, 104.0])
    log_rets = np.log(closes[1:] / closes[:-1])
    
    # t=0일 때, 미래 4개의 log returns의 표본표준편차 * sqrt(4)
    expected_std = log_rets.std(ddof=1) * np.sqrt(4)
    
    actual = r_df.loc[r_df['date'] == '2020-03-31', 'R'].values[0]
    assert abs(actual - expected_std) < 1e-6
    print(f"Risk test passed: Expected {expected_std:.6f}, got {actual:.6f}")

def test_optimize_dtypes():
    """데이터타입 및 메모리 최적화 검증."""
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'close': [150.0, 300.0],
        'volume': [1000000.0, 2000000.0],
        'shares': [10, 20]
    })
    
    opt_df = optimize_dtypes(df)
    
    # DTYPE_POLICY 적용 확인
    assert opt_df['ticker'].dtype == 'category'
    assert opt_df['close'].dtype == 'float32'
    assert opt_df['shares'].dtype == 'int8'  # 10, 20은 int8 범위에 해당하므로 변환됨
    print("Optimize dtypes test passed")
