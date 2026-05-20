"""
src/theme/context.py  (v5.5)

테마 비중 벡터: 순위 백분위 → 실질 비중 및 상대 배수로 교체.
"""
import numpy as np
import pandas as pd
from typing import Optional, Set
from src.theme.loader import load_themes

THEME_VEC_DIM = 18
NEUTRAL_WEIGHT = 0.0        # 비중 기본값 (데이터 없을 때)
NEUTRAL_RELATIVE = 1.0      # 상대 배수 기본값 (평균과 같음)


def _safe_weight(numerator: float, denominator: float) -> float:
    """비중 계산. denominator가 0이거나 음수이면 0 반환."""
    if denominator <= 0 or np.isnan(denominator) or np.isnan(numerator):
        return NEUTRAL_WEIGHT
    val = numerator / denominator
    return float(np.clip(val, 0.0, 1.0))


def _safe_relative(value: float, mean: float) -> float:
    """
    상대 배수 = value / mean.
    mean이 0이거나 부호가 다른 경우 NEUTRAL 반환.
    극단값 클리핑: [0.1, 10.0] 범위로 제한.
    """
    if mean == 0 or np.isnan(mean) or np.isnan(value):
        return NEUTRAL_RELATIVE
    if mean < 0 and value < 0:
        # 둘 다 음수: 비율 의미 있음
        ratio = value / mean
    elif mean < 0 or value < 0:
        # 부호 다름: 의미 없음
        return NEUTRAL_RELATIVE
    else:
        ratio = value / mean
    return float(np.clip(ratio, 0.1, 10.0))


def _signed_weight(numerator: float, denominator: float) -> float:
    """
    음수 허용 비중 (EBITDA, FCF, 순이익).
    분모는 양수 합계만 사용 (음수 기업은 분모에서 제외).
    결과는 [-1, 1] 클리핑.
    """
    if np.isnan(numerator) or denominator <= 0:
        return NEUTRAL_WEIGHT
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def compute_theme_vector(
    ticker: str,
    ticker_row: pd.Series,
    peers: pd.DataFrame,
) -> np.ndarray:
    """
    단일 테마에 대한 18차원 벡터 계산.

    Args:
        ticker     : 대상 종목 코드
        ticker_row : 대상 종목의 현재 시점 피처 (pd.Series)
        peers      : 같은 테마·같은 시점의 종목 DataFrame (대상 포함)

    Returns:
        (18,) float32 벡터
    """
    vec = np.full(THEME_VEC_DIM, NEUTRAL_WEIGHT, dtype=np.float32)

    if len(peers) < 2:
        return vec

    def col_sum_pos(col):
        """양수 합계 (음수 기업은 분모에서 제외)."""
        return peers[col].clip(lower=0).sum() if col in peers.columns else 0.0

    def col_sum_all(col):
        return peers[col].sum() if col in peers.columns else 0.0

    def own(col):
        mask = peers['ticker'] == ticker
        if col in peers.columns and mask.any():
            v = peers.loc[mask, col].iloc[0]
            return float(v) if not pd.isna(v) else np.nan
        return np.nan

    def col_mean(col):
        if col not in peers.columns:
            return np.nan
        vals = peers[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else np.nan

    # ── [0] 시총 비중 ──────────────────────────────────────────────────
    total_mktcap = col_sum_pos('market_cap')
    vec[0] = _safe_weight(own('market_cap') or 0.0, total_mktcap)

    # ── [1] 매출 비중 ──────────────────────────────────────────────────
    total_rev = col_sum_pos('F_GRW_rev_base')   # 매출 절대값 컬럼
    vec[1] = _safe_weight(own('F_GRW_rev_base') or 0.0, total_rev)

    # ── [2] EBITDA 비중 (음수 허용) ───────────────────────────────────
    total_ebitda_pos = col_sum_pos('F_PRF_ebitda_abs')
    vec[2] = _signed_weight(own('F_PRF_ebitda_abs') or 0.0, total_ebitda_pos)

    # ── [3] FCF 비중 (음수 허용) ──────────────────────────────────────
    total_fcf_pos = col_sum_pos('F_CF_002')
    vec[3] = _signed_weight(own('F_CF_002') or 0.0, total_fcf_pos)

    # ── [4] 순이익 비중 (음수 허용) ───────────────────────────────────
    total_ni_pos = col_sum_pos('F_PRF_net_income_abs')
    vec[4] = _signed_weight(own('F_PRF_net_income_abs') or 0.0, total_ni_pos)

    # ── [5] 자산 비중 ──────────────────────────────────────────────────
    total_assets = col_sum_pos('F_FIN_total_assets')
    vec[5] = _safe_weight(own('F_FIN_total_assets') or 0.0, total_assets)

    # ── [6]~[11] 상대 배수 ────────────────────────────────────────────
    for i, col in enumerate([
        'F_VAL_003',      # PBR    [6]
        'F_VAL_001',      # PER    [7]
        'F_VAL_005',      # EV/EBITDA [8]
        'F_PRF_005',      # ROE   [9]
        'F_GRW_001',      # 매출 CAGR [10]
        'F_CF_003',       # FCF 마진 [11]
    ], start=6):
        mean_val = col_mean(col)
        own_val  = own(col)
        if own_val is not None and not np.isnan(own_val):
            vec[i] = _safe_relative(own_val, mean_val)

    # ── [12]~[17] 테마 전체 상태 ──────────────────────────────────────

    # [12] 테마 전체 시총 (로그 스케일, 억 단위 정규화)
    if total_mktcap > 0:
        vec[12] = float(np.log1p(total_mktcap / 1e8))   # 억 단위

    # [13] 테마 시총 4분기 성장률 (데이터 없으면 0)
    if 'theme_mktcap_prev4q' in peers.columns:
        prev = peers['theme_mktcap_prev4q'].mean()
        if prev > 0:
            vec[13] = float(np.clip(total_mktcap / prev - 1, -1.0, 3.0))

    # [14] 테마 평균 4분기 수익률
    if 'ret_4q' in peers.columns:
        ret4q = peers['ret_4q'].dropna()
        if len(ret4q) > 0:
            vec[14] = float(np.clip(ret4q.mean(), -1.0, 3.0))

    # [15] 테마 수익률 변동성
    if 'ret_4q' in peers.columns:
        ret4q = peers['ret_4q'].dropna()
        if len(ret4q) > 1:
            vec[15] = float(np.clip(ret4q.std(), 0.0, 2.0))

    # [16] HHI 집중도
    if total_mktcap > 0 and 'market_cap' in peers.columns:
        weights = peers['market_cap'].clip(lower=0) / total_mktcap
        vec[16] = float(np.clip((weights ** 2).sum(), 0.0, 1.0))

    # [17] 테마 종목 수 (로그 스케일)
    vec[17] = float(np.log1p(len(peers)))

    return vec


def compute_theme_context(
    df: pd.DataFrame,
    processed_path: str = 'data/themes/processed/themes.yaml',
    peer_tickers: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    전체 DataFrame에 대해 테마 비중 벡터를 계산한다.

    Args:
        df             : 종목-분기별 DataFrame. 필수: ['ticker', 'date']
        processed_path : themes.yaml 경로
        peer_tickers   : None=전체, set=Train 오염 방지용 지정 종목만

    Returns:
        ['ticker', 'date', 'theme_ctx_0', ..., 'theme_ctx_17'] DataFrame
    """
    mapping = load_themes(processed_path)
    t2th    = mapping['tickers']
    th2t    = mapping['theme_to_tickers']

    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    results = []

    for date, date_group in df.groupby('date', observed=True, sort=False):
        ticker_rows = {row['ticker']: row for _, row in date_group.iterrows()}

        for _, row in date_group.iterrows():
            ticker  = row['ticker']
            info    = t2th.get(ticker, {})
            themes  = info.get('themes', [])

            if not themes:
                vec = np.zeros(THEME_VEC_DIM, dtype=np.float32)
                vec[6:12] = NEUTRAL_RELATIVE    # 상대 배수는 1.0 (평균)
            else:
                vecs = []
                for theme_id in themes:
                    all_peers = th2t.get(theme_id, [])
                    if peer_tickers is not None:
                        peers_list = [t for t in all_peers
                                      if t in peer_tickers and t in ticker_rows]
                    else:
                        peers_list = [t for t in all_peers if t in ticker_rows]

                    if not peers_list:
                        continue
                    peers_df = pd.DataFrame([ticker_rows[t] for t in peers_list])
                    vecs.append(compute_theme_vector(ticker, row, peers_df))

                if vecs:
                    # 시총 비중 기준 가중 평균
                    # primary 테마(첫 번째)에 더 높은 가중치 (2:1)
                    if len(vecs) == 1:
                        vec = vecs[0]
                    else:
                        weights = np.array(
                            [2.0] + [1.0] * (len(vecs) - 1), dtype=np.float32
                        )
                        weights /= weights.sum()
                        vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)
                else:
                    vec = np.zeros(THEME_VEC_DIM, dtype=np.float32)
                    vec[6:12] = NEUTRAL_RELATIVE

            entry = {'ticker': ticker, 'date': date}
            for i, v in enumerate(vec):
                entry[f'theme_ctx_{i}'] = float(v)
            results.append(entry)

    out = pd.DataFrame(results)
    ctx_cols = [f'theme_ctx_{i}' for i in range(THEME_VEC_DIM)]
    out[ctx_cols] = out[ctx_cols].astype('float32')
    return out
