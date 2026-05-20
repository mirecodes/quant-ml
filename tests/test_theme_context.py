# tests/test_theme_context.py (v5.5 추가 테스트)

def test_mktcap_weight_sums_to_one():
    """시총 비중의 합이 1인지 확인."""
    import pandas as pd
    import numpy as np
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':     ['A', 'B', 'C'],
        'market_cap': [300.0, 500.0, 200.0],
        'F_VAL_003':  [1.0, 2.0, 0.8],   # PBR
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    vB = compute_theme_vector('B', peers[peers['ticker']=='B'].iloc[0], peers)
    vC = compute_theme_vector('C', peers[peers['ticker']=='C'].iloc[0], peers)

    # 시총 비중 합산 = 1
    total_w = vA[0] + vB[0] + vC[0]
    assert abs(total_w - 1.0) < 1e-5, f"mktcap weight sum = {total_w}"

    # 개별 비중 = 기대값
    assert abs(vA[0] - 0.30) < 1e-4   # 300/1000
    assert abs(vB[0] - 0.50) < 1e-4   # 500/1000
    assert abs(vC[0] - 0.20) < 1e-4   # 200/1000


def test_relative_pbr():
    """상대 PBR: 테마 평균 PBR = 2.0일 때 PBR=1.0 종목의 상대값 = 0.5."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':    ['A', 'B', 'C'],
        'market_cap':[100.0, 100.0, 100.0],
        'F_VAL_003': [1.0, 2.0, 3.0],   # 평균 = 2.0
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    assert abs(vA[6] - 0.5) < 1e-4, f"rel_pbr = {vA[6]}, expected 0.5"


def test_hhi_monopoly():
    """한 종목이 시총 100% 점유 시 HHI = 1.0."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':    ['A', 'B'],
        'market_cap':[999.0, 1.0],
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    assert vA[16] > 0.99 - 1e-3, f"HHI expected ~1.0, got {vA[16]}"


def test_negative_ebitda_weight():
    """음수 EBITDA 기업의 비중이 음수로 정확히 표현되는지."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':          ['A', 'B', 'C'],
        'market_cap':      [200.0, 200.0, 200.0],
        'F_PRF_ebitda_abs':[100.0, 200.0, -50.0],  # C는 적자
    })
    vC = compute_theme_vector('C', peers[peers['ticker']=='C'].iloc[0], peers)
    # C의 EBITDA 비중 = -50 / (100+200) = -0.1667
    assert vC[2] < 0, f"Negative EBITDA should give negative weight, got {vC[2]}"
    assert abs(vC[2] - (-50.0/300.0)) < 1e-3
