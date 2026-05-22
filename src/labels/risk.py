"""
src/labels/risk.py  (v5.6)

위험도 = 분기 내 일별 High-Low Range의 평균
R(t) = mean( (High_d - Low_d) / Close_d  for d in quarter t )
"""
import numpy as np
import pandas as pd


def compute_risk(
    prices_daily: pd.DataFrame,
    min_close: float = 1e-6,
    min_trading_days: int = 10,
) -> pd.DataFrame:
    """
    위험도 라벨 생성.

    Args:
        prices_daily      : [ticker, date, high, low, close] 컬럼 포함 일별 DataFrame
        min_close         : Close 최솟값 (0 나눗셈 방지)
        min_trading_days  : 분기당 최소 거래일 수 (미달 시 라벨 생성 안 함)

    Returns:
        [ticker, date, R, R_trading_days] DataFrame
          date            : 분기 마지막 거래일 (분기 식별자)
          R               : 분기 내 일별 변동폭 평균, float32
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

    # 일별 변동폭 계산
    df['hlr'] = (df['high'] - df['low']) / df['close']  # float32

    # 분기별 집계 (QE: 분기 마지막 거래일을 키로)
    df = df.set_index('date')
    results = []

    for ticker, grp in df.groupby('ticker', observed=True, sort=False):
        quarterly = grp['hlr'].resample('QE').agg(
            R='mean',
            R_trading_days='count',
        ).reset_index()
        quarterly['ticker'] = ticker

        # 최소 거래일 필터
        quarterly = quarterly[quarterly['R_trading_days'] >= min_trading_days]
        results.append(quarterly)

    if not results:
        return pd.DataFrame(columns=['ticker', 'date', 'R', 'R_trading_days'])

    out = pd.concat(results, ignore_index=True)
    out['R']               = out['R'].astype('float32')
    out['R_trading_days']  = out['R_trading_days'].astype('int16')

    return out[['ticker', 'date', 'R', 'R_trading_days']]
