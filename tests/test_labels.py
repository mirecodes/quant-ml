# tests/test_labels.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.labels.risk import compute_risk

def test_risk_basic_average():
    """일별 변동폭 평균 계산 확인."""
    # 3일짜리 분기 1개
    prices = pd.DataFrame({
        'ticker': ['A'] * 3,
        'date':   pd.to_datetime(['2020-01-02', '2020-01-03', '2020-03-31']),
        'high':   [110.0, 115.0, 120.0],
        'low':    [ 90.0,  95.0, 100.0],
        'close':  [100.0, 100.0, 110.0],
    })
    # 일별 hlr: 20/100=0.20, 20/100=0.20, 20/110≈0.1818
    # 평균: (0.20 + 0.20 + 0.1818) / 3 ≈ 0.1939

    result = compute_risk(prices, min_trading_days=3)
    assert len(result) == 1
    expected = np.mean([20/100, 20/100, 20/110])
    assert abs(result.iloc[0]['R'] - expected) < 1e-4


def test_risk_non_negative():
    """R은 항상 0 이상."""
    prices = pd.DataFrame({
        'ticker': ['B'] * 5,
        'date':   pd.date_range('2020-01-01', periods=5, freq='B'),
        'high':   [105.0] * 5,
        'low':    [ 95.0] * 5,
        'close':  [100.0] * 5,
    })
    result = compute_risk(prices, min_trading_days=5)
    assert (result['R'] >= 0).all()


def test_risk_two_quarters():
    """두 분기가 올바르게 분리 집계되는지 확인."""
    # Q1 (1월~3월): hlr=0.10, Q2 (4월~6월): hlr=0.20
    dates_q1 = pd.date_range('2020-01-02', '2020-03-31', freq='B')
    dates_q2 = pd.date_range('2020-04-01', '2020-06-30', freq='B')

    def make_rows(dates, hlr_val):
        return pd.DataFrame({
            'ticker': ['C'] * len(dates),
            'date':   dates,
            'high':   [100 * (1 + hlr_val)] * len(dates),
            'low':    [100 * (1 - hlr_val)] * len(dates),
            'close':  [100.0] * len(dates),
        })

    prices = pd.concat([make_rows(dates_q1, 0.05),
                         make_rows(dates_q2, 0.10)], ignore_index=True)
    result = compute_risk(prices, min_trading_days=10)
    result = result.sort_values('date').reset_index(drop=True)

    assert len(result) == 2
    # Q1 평균 hlr = (100*(1+0.05) - 100*(1-0.05)) / 100 = 0.10
    assert abs(result.iloc[0]['R'] - 0.10) < 1e-3
    # Q2 평균 hlr = 0.20
    assert abs(result.iloc[1]['R'] - 0.20) < 1e-3


def test_risk_min_trading_days_filter():
    """최소 거래일 미달 분기는 결과에서 제외."""
    # 3일짜리 분기 (min_trading_days=10 설정 시 제외)
    prices = pd.DataFrame({
        'ticker': ['D'] * 3,
        'date':   pd.to_datetime(['2020-01-02', '2020-01-03', '2020-03-31']),
        'high':   [110.0, 115.0, 120.0],
        'low':    [ 90.0,  95.0, 100.0],
        'close':  [100.0, 100.0, 110.0],
    })
    result = compute_risk(prices, min_trading_days=10)
    assert len(result) == 0


def test_risk_invalid_high_low_excluded():
    """high < low 오류 행은 제외되어 평균에 영향 없음."""
    prices = pd.DataFrame({
        'ticker': ['E'] * 4,
        'date':   pd.date_range('2020-01-01', periods=4, freq='B'),
        'high':   [110.0,  90.0, 110.0, 110.0],  # 2번째는 high < low 오류
        'low':    [ 90.0, 100.0,  90.0,  90.0],
        'close':  [100.0, 100.0, 100.0, 100.0],
    })
    # 오류 행 제외 후 유효 3행: hlr = 0.20, 0.20, 0.20 → 평균 0.20
    result = compute_risk(prices, min_trading_days=3)
    assert len(result) == 1
    assert abs(result.iloc[0]['R'] - 0.20) < 1e-4
