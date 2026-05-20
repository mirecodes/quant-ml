# tests/test_pipeline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import pytest
import pandas as pd
import numpy as np
from src.labels.attractiveness import compute_attractiveness
from src.labels.risk import compute_risk
from src.utils.io import optimize_dtypes

def test_attractiveness_label():
    """A_5Y 계산이 max-window 및 log_N (N < 5 일 때) 기반인지 검증."""
    # 5분기 동안의 sample price 데이터 생성
    sample_df = pd.DataFrame({
        'ticker': ['TEST'] * 6,
        'date': pd.date_range(start='2020-01-01', periods=6, freq='QE'),
        'close': [100.0, 110.0, 105.0, 120.0, 130.0, 125.0]
    })
    
    # 5년 max horizon (max_horizon_years=5, min_forward_quarters=1)
    a_df = compute_attractiveness(sample_df, max_horizon_years=5, min_forward_quarters=1)
    
    # t=0 (price=100) 일 때, 미래 5분기의 max price는 130
    # forward quarters = 5 -> base = 1.25
    # expected = log_1.25(130/100) = log_1.25(1.3)
    p_t = 100.0
    max_price = 130.0
    base_years = 1.25
    expected = np.log(max_price / p_t) / np.log(base_years)
    
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


def test_lstm_encoder_variable_length():
    """가변 길이 시퀀스를 패딩 없이 처리하는지 확인."""
    from src.models.lstm_encoder import LSTMEncoder
    enc = LSTMEncoder(input_size=10, hidden_size=32, bidirectional=True)
    enc.eval()

    # 길이가 다른 3개 샘플
    seqs = [
        torch.randn(5, 10),
        torch.randn(12, 10),
        torch.randn(3, 10),
    ]
    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(seqs, batch_first=True)   # (3, 12, 10)
    lengths = torch.tensor([5, 12, 3])

    with torch.no_grad():
        ctx = enc(padded, lengths)

    assert ctx.shape == (3, 64)   # 32 * 2 = 64
    # 길이가 다른 샘플들의 결과가 서로 달라야 함
    assert not torch.allclose(ctx[0], ctx[1])


def test_ft_transformer_output_shape():
    """FT-Transformer 출력 차원 확인."""
    from src.models.ft_transformer import FTTransformer
    model = FTTransformer(
        context_dims=[64, 32, 16],
        n_num_features=5,
        cat_cardinalities=[10, 5, 3],
        d_token=64,
        n_heads=4,
        n_layers=2,
    )
    model.eval()
    B = 8
    contexts = [torch.randn(B, 64), torch.randn(B, 32), torch.randn(B, 16)]
    x_num = torch.randn(B, 5)
    x_cat = torch.randint(0, 3, (B, 3))

    with torch.no_grad():
        A, R = model(contexts, x_num, x_cat)

    assert A.shape == (B,)
    assert R.shape == (B,)
    assert (R >= 0).all(), "위험도는 항상 ≥ 0 이어야 함 (Softplus)"


def test_collate_padding():
    """collate_fn이 가변 길이 시퀀스를 올바르게 패딩하는지 확인."""
    from src.data.dataset import collate_fn
    batch = [
        {
            's_seq':    torch.randn(5, 10),
            'm_seq':    torch.randn(5, 8),
            'theme':    torch.randn(16),
            'snap_num': torch.randn(7),
            'snap_cat': torch.zeros(3, dtype=torch.long),
            'A': torch.tensor(0.5),
            'R': torch.tensor(0.2),
        },
        {
            's_seq':    torch.randn(12, 10),
            'm_seq':    torch.randn(12, 8),
            'theme':    torch.randn(16),
            'snap_num': torch.randn(7),
            'snap_cat': torch.zeros(3, dtype=torch.long),
            'A': torch.tensor(0.8),
            'R': torch.tensor(0.3),
        },
    ]
    out = collate_fn(batch)
    assert out['s_seq'].shape   == (2, 12, 10)
    assert out['s_lengths'][0]  == 5
    assert out['s_lengths'][1]  == 12
    assert out['theme'].shape   == (2, 16)
