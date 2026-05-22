"""
src/labels/risk.py  (v5.6 - Robust Interpercentile Range Volatility)

위험도 = 분기 내 일별 High 90% 퍼센타일과 Low 10% 퍼센타일의 차이 비율
R(t) = (Percentile_90(High_d) - Percentile_10(Low_d)) / Close_t  for d in quarter t
"""
import numpy as np
import pandas as pd


def compute_risk(
    prices_daily: pd.DataFrame,
    min_close: float = 1e-6,
    min_trading_days: int = 10,
) -> pd.DataFrame:
    """
    위험도 라벨 생성 (Robust Interpercentile Range Volatility).

    Args:
        prices_daily      : [ticker, date, high, low, close] 컬럼 포함 일별 DataFrame
        min_close         : Close 최솟값 (0 나눗셈 방지)
        min_trading_days  : 분기당 최소 거래일 수 (미달 시 라벨 생성 안 함)

    Returns:
        [ticker, date, R, R_trading_days] DataFrame
          date            : 분기 마지막 거래일 (분기 식별자)
          R               : 90% High와 10% Low의 차이를 분기말 종가로 나눈 값, float32
          R_trading_days  : 해당 분기 거래일 수, int16
    """
    required = {'ticker', 'date', 'high', 'low', 'close'}
    missing  = required - set(prices_daily.columns)
    if missing:
        raise ValueError(f"prices_daily에 필요한 컬럼 없음: {missing}")

    df = prices_daily.copy()
    df['date']  = pd.to_datetime(df['date'])
    df['high']  = df['high'].astype('float32')
    df['low']   = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')

    # 유효성 필터
    valid = (
        (df['close'] > min_close) &
        (df['high']  >= df['low'])       # high < low 오류 제거
    )
    df = df[valid].copy()

    results = []

    for ticker, grp in df.groupby('ticker', observed=True, sort=False):
        grp = grp.sort_values('date')
        
        # 분기별 집계 (QE: 분기 마지막 거래일을 키로)
        quarterly = grp.resample('QE', on='date').agg(
            high_90=('high', lambda x: np.percentile(x, 90) if len(x) >= min_trading_days else np.nan),
            low_10=('low', lambda x: np.percentile(x, 10) if len(x) >= min_trading_days else np.nan),
            close_t=('close', 'last'),
            R_trading_days=('close', 'count'),
        ).dropna().reset_index()
        
        quarterly['ticker'] = ticker
        
        # 위험도 R = (High_90 - Low_10) / Close_t
        quarterly['R'] = (quarterly['high_90'] - quarterly['low_10']) / quarterly['close_t']

        # 최소 거래일 필터
        quarterly = quarterly[quarterly['R_trading_days'] >= min_trading_days]
        results.append(quarterly)

    if not results:
        return pd.DataFrame(columns=['ticker', 'date', 'R', 'R_trading_days'])

    out = pd.concat(results, ignore_index=True)
    out['R']               = out['R'].astype('float32')
    out['R_trading_days']  = out['R_trading_days'].astype('int16')

    return out[['ticker', 'date', 'R', 'R_trading_days']]
