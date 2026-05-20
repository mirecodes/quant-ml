"""
src/theme/context.py

각 (종목, 분기) 시점에 대해 테마 내 비중 벡터를 계산한다.

출력 벡터 (16차원, float32):
  [0]  theme_mktcap_weight        : 시총 비중 (0~1)
  [1]  theme_mktcap_rank_pct      : 시총 순위 백분위
  [2]  theme_pbr_rank_pct         : PBR 순위 백분위 (낮을수록 저평가)
  [3]  theme_per_rank_pct         : PER 순위 백분위
  [4]  theme_ev_ebitda_rank_pct   : EV/EBITDA 순위 백분위
  [5]  theme_roe_rank_pct         : ROE 순위 백분위 (높을수록 우수)
  [6]  theme_rev_growth_rank_pct  : 매출성장률 순위 백분위
  [7]  theme_gp_a_rank_pct        : GP/Assets 순위 백분위
  [8]  theme_ret_1q_rank_pct      : 1분기 수익률 순위 백분위
  [9]  theme_ret_4q_rank_pct      : 4분기 수익률 순위 백분위
  [10] theme_avg_ret_4q           : 테마 평균 4분기 수익률
  [11] theme_ret_dispersion       : 테마 내 수익률 표준편차
  [12] theme_avg_pbr              : 테마 평균 PBR
  [13] theme_pbr_dispersion       : 테마 내 PBR 표준편차
  [14] theme_n_stocks             : 테마 종목 수 (로그 스케일)
  [15] theme_hhi                  : Herfindahl 시총 집중도
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.theme.loader import load_themes

THEME_VEC_DIM = 16
NEUTRAL = 0.5   # NaN 대체값


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    """0~1 백분위 변환. NaN은 NEUTRAL로."""
    if series.isna().all():
        return pd.Series(NEUTRAL, index=series.index)
    ranked = series.rank(method='average', ascending=ascending, pct=True)
    return ranked.fillna(NEUTRAL).astype('float32')


def _herfindahl(weights: pd.Series) -> float:
    """HHI 집중도 (0~1). 높을수록 특정 종목에 집중."""
    w = weights.fillna(0).values
    if w.sum() < 1e-8:
        return NEUTRAL
    w_norm = w / w.sum()
    return float(np.sum(w_norm ** 2))


def compute_theme_context(
    df: pd.DataFrame,
    processed_path: str = 'data/themes/processed/themes.yaml',
    peer_tickers: set = None,    # None이면 전체 peer 사용
                                 # set이면 해당 종목만 peer로 사용
) -> pd.DataFrame:
    """
    전체 데이터프레임에 대해 테마 비중 벡터를 계산한다 (벡터화).
    """
    mapping = load_themes(processed_path)
    th2t    = mapping['theme_to_tickers']

    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    results = []

    for date, date_group in df.groupby('date', observed=True, sort=False):
        theme_vectors: Dict[str, List[np.ndarray]] = {}

        # 1. 모든 테마에 대한 피어 벡터 미리 계산 (벡터화)
        for theme_id, all_peers in th2t.items():
            # peer 필터 적용
            if peer_tickers is not None:
                peer_list = [t for t in all_peers if t in peer_tickers]
            else:
                peer_list = all_peers

            peers_df = date_group[date_group['ticker'].isin(peer_list)]
            if len(peers_df) < 2:
                continue

            n_peers = len(peers_df)
            vec_matrix = np.full((n_peers, THEME_VEC_DIM), NEUTRAL, dtype=np.float32)

            mktcap_col = 'market_cap' if 'market_cap' in peers_df.columns else None
            if mktcap_col:
                caps = peers_df[mktcap_col].fillna(0.0).to_numpy()
                total_cap = caps.sum()
                if total_cap > 0:
                    vec_matrix[:, 0] = caps / total_cap
                vec_matrix[:, 1] = _rank_pct(peers_df[mktcap_col], ascending=True).to_numpy()

            for i, (col, asc) in enumerate([
                ('F_VAL_pbr',        True),   # [2]
                ('F_VAL_per',        True),   # [3]
                ('F_VAL_ev_ebitda',  True),   # [4]
                ('F_PRF_roe',        False),  # [5]
                ('F_GRW_rev_cagr',   False),  # [6]
                ('C_GP_A',           False),  # [7]
                ('ret_1q',           False),  # [8]
                ('ret_4q',           False),  # [9]
            ], start=2):
                if col in peers_df.columns:
                    vec_matrix[:, i] = _rank_pct(peers_df[col], ascending=asc).to_numpy()

            if 'ret_4q' in peers_df.columns:
                ret4q = peers_df['ret_4q'].dropna().to_numpy()
                if len(ret4q) > 0:
                    vec_matrix[:, 10] = ret4q.mean()
                if len(ret4q) > 1:
                    vec_matrix[:, 11] = ret4q.std()
                else:
                    vec_matrix[:, 11] = 0.0

            if 'F_VAL_pbr' in peers_df.columns:
                pbr = peers_df['F_VAL_pbr'].dropna().to_numpy()
                if len(pbr) > 0:
                    vec_matrix[:, 12] = pbr.mean()
                if len(pbr) > 1:
                    vec_matrix[:, 13] = pbr.std()
                else:
                    vec_matrix[:, 13] = 0.0

            vec_matrix[:, 14] = np.log1p(n_peers)
            if mktcap_col:
                vec_matrix[:, 15] = _herfindahl(peers_df[mktcap_col])

            for ticker, vec in zip(peers_df['ticker'].to_list(), vec_matrix):
                theme_vectors.setdefault(ticker, []).append(vec)

        # 2. 각 종목에 대해 테마 벡터 평균 산출
        for _, row in date_group.iterrows():
            ticker = row['ticker']
            vecs = theme_vectors.get(ticker)
            if vecs:
                vec = np.mean(vecs, axis=0).astype(np.float32)
            else:
                vec = np.full(THEME_VEC_DIM, NEUTRAL, dtype=np.float32)

            entry = {'ticker': ticker, 'date': date}
            for i, v in enumerate(vec):
                entry[f'theme_ctx_{i}'] = v
            results.append(entry)

    out = pd.DataFrame(results)
    ctx_cols = [f'theme_ctx_{i}' for i in range(THEME_VEC_DIM)]
    out[ctx_cols] = out[ctx_cols].astype('float32')
    return out
