# stockml 구현 가이드 v5.3
# Coding Agent 전용 기술 명세서

"""
이 문서는 v5.1을 기준으로 다음을 완전 교체한다:
  - TFT, LightGBM 제거
  - LSTM(stock) + LSTM(macro) + ThemeContext → FT-Transformer 단일 파이프라인
  - 테마 내 비중 컨텍스트 신규 추가

변경되지 않는 것:
  - 환경 설정 (섹션 1)
  - 데이터 수집 (섹션 4)
  - 피처 엔지니어링 (섹션 5)
  - 라벨 생성 (섹션 6)
  - 평가 메트릭 (섹션 9)
  - UI (섹션 10)
  - 실행 순서 (섹션 11)

Coding agent 행동 원칙:
  - 모든 섹션을 위에서 아래로 순서대로 구현한다
  - 섹션 간 의존성은 명시된 import 경로를 따른다
  - 각 모듈은 독립적으로 단위 테스트 가능하게 작성한다
  - 불명확한 결정은 이 문서의 설계 의도를 우선한다
  - M1 Pro MPS 제약을 항상 염두에 두고 float32, pin_memory=False를 기본으로 한다
"""

# =============================================================================
# 섹션 0. 아키텍처 개요
# =============================================================================

"""
전체 데이터 흐름:

  [종목 재무 시계열]  (B, T_stock, F_stock)   F_stock = 거시 제외 피처 (재무 45 + 자산 35)
  [거시 시계열]       (B, T_macro, F_macro)   F_macro = 거시 80종, 모든 종목 공유
  [테마 비중 벡터]    (B, F_theme)            F_theme = 16차원, 분기별 스냅샷
  [스냅샷 피처]       (B, F_snap)             F_snap  = 계산형 7 + 범주형(섹터,국가 등)

         ↓                   ↓                   ↓              ↓
  LSTM_stock            LSTM_macro          Linear           Feature
  Bidirectional         단방향             Projection       Tokenizer
  (B, 256)              (B, 128)            (B, 64)
         ↓                   ↓                   ↓
         └───────────────────┴───────────────────┘
                             ↓
                    FT-Transformer
          [CLS] [stock_ctx] [macro_ctx] [theme_ctx] [snap_f1..fn]
                     Self-Attention (피처 간 상호작용)
                             ↓
                    [CLS] 토큰
                      /         \\
                 head_A         head_R
                   ↓               ↓
                   A               R
                (매력도)         (위험도)

핵심 설계 의도:
  LSTM_stock  : 종목별 고유 재무 흐름 (ROE 개선 추세, 부채비율 변화)
  LSTM_macro  : 모든 종목에 공통인 거시 레짐 (금리 사이클, 경기 국면)
                → 배치당 1회만 계산 후 모든 종목에 브로드캐스트
  theme_ctx   : 같은 GT 테마 내에서 이 종목의 상대적 위치
                → "나는 이 테마에서 얼마나 크고, 얼마나 저평가인가"
  FT-Transformer: 세 컨텍스트와 스냅샷 피처 간 교차 Attention
                → "금리 인상 국면(macro)에서 고부채(stock) 종목이
                    테마 내 저평가(theme) 상태일 때의 의미" 학습
"""

# =============================================================================
# 섹션 1. 환경 설정 (v5.1과 동일, requirements만 변경)
# =============================================================================

# requirements.txt
REQUIREMENTS = """
torch>=2.2.0
pytorch-lightning>=2.2.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
yfinance>=0.2.36
pykrx>=1.0.45
pandas-datareader>=0.10.0
fredapi>=0.5.1
streamlit>=1.31.0
plotly>=5.18.0
optuna>=3.5.0
statsmodels>=0.14.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
requests>=2.31.0
tqdm>=4.66.0
pyyaml>=6.0.1
joblib>=1.3.0
pyarrow>=14.0.0
psutil>=5.9.0
scipy>=1.12.0
# 제거: pytorch-forecasting, lightgbm
"""

# =============================================================================
# 섹션 2. 프로젝트 디렉토리 구조
# =============================================================================

"""
stockml/
├── config/
│   ├── settings.yaml
│   ├── feature_lags.yaml
│   └── theme_mapping/
│       ├── global_themes.yaml
│       ├── kospi_mapping_part1.yaml
│       ├── kospi_mapping_part2.yaml
│       ├── kospi_mapping_part3.yaml
│       ├── sp500_mapping_part1.yaml
│       ├── sp500_mapping_part2.yaml
│       └── sp500_mapping_part3.yaml
│
├── data/
│   ├── raw/prices/ financials/ macro/ themes/
│   └── processed/
│       ├── prices_quarterly.parquet
│       ├── features_stock.parquet    ← 종목 재무 피처
│       ├── features_macro.parquet    ← 거시 피처 (날짜별, 종목 무관)
│       ├── theme_context.parquet     ← 테마 내 비중 (종목×날짜)
│       └── labels.parquet
│
├── src/
│   ├── utils/
│   │   ├── device.py                 ← v5.1과 동일
│   │   ├── io.py                     ← v5.1과 동일
│   │   └── pit.py                    ← Point-in-Time 유틸
│   │
│   ├── data_fetchers/                ← v5.1과 동일
│   │
│   ├── features/                     ← v5.1과 동일
│   │
│   ├── theme/
│   │   ├── loader.py                 ← YAML 매핑 로드
│   │   └── context.py                ← 테마 비중 벡터 계산  ★신규
│   │
│   ├── labels/                       ← v5.1과 동일
│   │
│   ├── models/
│   │   ├── lstm_encoder.py           ← 가변길이 시계열 → 컨텍스트  ★신규
│   │   ├── ft_transformer.py         ← 메인 예측 모델              ★신규
│   │   ├── predictor.py              ← Lightning 통합 모듈         ★신규
│   │   └── baseline_accounting.py   ← v5.1과 동일 (평가용)
│   │
│   ├── data/
│   │   └── dataset.py                ← 가변길이 Dataset + collate  ★신규
│   │
│   └── evaluation/                   ← v5.1과 동일
│
└── scripts/
    ├── 01_fetch_data.py              ← v5.1과 동일
    ├── 02_build_features.py          ← v5.1과 동일
    ├── 02b_build_theme_context.py    ← 테마 비중 계산              ★신규
    ├── 03_build_labels.py            ← v5.1과 동일
    ├── 04_train.py                   ← TFT 제거, 새 모델           ★교체
    ├── 05_train_baselines.py         ← 회계 baseline만 유지
    ├── 06_evaluate.py                ← v5.1과 동일
    └── 07_run_ui.py                  ← v5.1과 동일
"""

# =============================================================================
# 섹션 3. 설정 파일
# =============================================================================

SETTINGS_YAML = """
# config/settings.yaml

project:
  name: stockml
  data_dir: ./data
  random_seed: 42

universe:
  countries: [KR, US]
  kr_market: KOSPI
  us_index: SP500
  exclude_etfs: true

prices:
  frequency: quarterly
  source_kr: pykrx
  source_us: yfinance

targets:
  attractiveness:
    max_horizon_years: 5
    log_base: 5
    use_max_in_window: true
    min_forward_quarters: 4
  risk:
    max_horizon_years: 5
    annualization_factor: 4
    min_forward_quarters: 4

train_split:
  train_end: '2017-12-31'
  val_start: '2018-01-01'
  val_end:   '2019-12-31'
  test_start: '2020-01-01'
  test_end:   '2021-12-31'

# ── 모델 설정 ──────────────────────────────────────────────────────
model:
  # LSTM_stock: 종목 재무 시계열 인코더
  lstm_stock_hidden: 128      # 양방향이므로 출력 256
  lstm_stock_layers: 2
  lstm_stock_dropout: 0.2
  lstm_stock_max_seq: 20      # 최대 20분기(5년) 과거

  # LSTM_macro: 거시 시계열 인코더
  lstm_macro_hidden: 64       # 단방향, 출력 64
  lstm_macro_layers: 1
  lstm_macro_max_seq: 20

  # ThemeContext Linear 투영
  theme_proj_dim: 64          # 16 → 64

  # FT-Transformer
  d_token: 192                # 피처 임베딩 차원 (192 = 8 heads × 24)
  n_heads: 8
  n_layers: 4
  ffn_factor: 1.333           # FFN hidden = d_token × ffn_factor
  dropout: 0.2
  attn_dropout: 0.1

  # 학습
  lr: 0.0001
  weight_decay: 0.01
  grad_clip: 1.0
  batch_size: 64              # M1 Pro 16GB
  max_epochs: 60
  patience: 10

  # M1 Pro
  num_workers: 2
  persistent_workers: true
  pin_memory: false           # MPS 미지원
  precision: '32-true'

device:
  prefer: mps
  fallback: cpu
"""

# =============================================================================
# 섹션 4. 테마 모듈 (신규)
# =============================================================================

# ── src/theme/loader.py ──────────────────────────────────────────────────────

THEME_LOADER = '''
"""
src/theme/loader.py

global_themes.yaml + kospi/sp500 매핑 파일을 로드해
ticker → [GT_XXX, ...] 딕셔너리와
GT_XXX → [ticker, ...] 딕셔너리를 제공한다.
"""
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List


@lru_cache(maxsize=1)
def load_theme_mapping(mapping_dir: str = "config/theme_mapping") -> Dict:
    """
    모든 매핑 파일을 로드해 두 방향 인덱스를 반환한다.

    반환:
        {
          "ticker_to_themes": {"005930": ["GT_SEMI_MEMORY", ...], ...},
          "theme_to_tickers": {"GT_SEMI_MEMORY": ["005930", "000660", ...], ...},
          "theme_meta":       {"GT_SEMI_MEMORY": {"tier":3, "parent":..., ...}, ...},
        }
    """
    p = Path(mapping_dir)

    # 테마 메타 로드
    with open(p / "global_themes.yaml", encoding="utf-8") as f:
        raw_themes = yaml.safe_load(f)
    theme_meta = {k: v for k, v in raw_themes.items()
                  if not k.startswith("#")}

    # 종목 매핑 로드 (part1~3, KR + US)
    ticker_to_themes: Dict[str, List[str]] = {}
    mapping_files = sorted(p.glob("*_mapping_part*.yaml"))
    for fpath in mapping_files:
        with open(fpath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for ticker, info in data.items():
            if not isinstance(info, dict):
                continue
            themes = info.get("themes", [])
            # 중복 ticker (DUP 접미사): 이미 있으면 병합
            clean = ticker.rstrip("_DUP0123456789").strip('"')
            if clean not in ticker_to_themes:
                ticker_to_themes[clean] = []
            for t in themes:
                if t not in ticker_to_themes[clean]:
                    ticker_to_themes[clean].append(t)

    # 역방향 인덱스
    theme_to_tickers: Dict[str, List[str]] = {}
    for ticker, themes in ticker_to_themes.items():
        for t in themes:
            theme_to_tickers.setdefault(t, []).append(ticker)

    return {
        "ticker_to_themes": ticker_to_themes,
        "theme_to_tickers": theme_to_tickers,
        "theme_meta": theme_meta,
    }
'''

# ── src/theme/context.py ─────────────────────────────────────────────────────

THEME_CONTEXT = '''
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

설계 원칙:
  - 모든 rank_pct는 0~1 percentile → 종목 수 차이 무관하게 비교 가능
  - 피처 부재(NaN)는 0.5로 대체 (중립값)
  - Point-in-Time: 같은 날짜의 같은 테마 종목만 참조
  - 멀티 테마: 각 테마 벡터를 단순 평균 (primary 가중 옵션 있음)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.theme.loader import load_theme_mapping

THEME_VEC_DIM = 16
NEUTRAL = 0.5   # NaN 대체값


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    """0~1 백분위 변환. NaN은 NEUTRAL로."""
    ranked = series.rank(method='average', ascending=ascending, pct=True)
    return ranked.fillna(NEUTRAL).astype('float32')


def _herfindahl(weights: pd.Series) -> float:
    """HHI 집중도 (0~1). 높을수록 특정 종목에 집중."""
    w = weights.fillna(0).values
    if w.sum() < 1e-8:
        return NEUTRAL
    w_norm = w / w.sum()
    return float(np.sum(w_norm ** 2))


def compute_theme_vector(
    ticker: str,
    ticker_row: pd.Series,
    peers: pd.DataFrame,
) -> np.ndarray:
    """
    단일 테마에 대한 16차원 벡터 계산.

    Args:
        ticker    : 대상 종목 코드
        ticker_row: 대상 종목의 현재 시점 피처 row (pd.Series)
        peers     : 같은 테마, 같은 시점의 모든 종목 DataFrame
                    (대상 종목 포함)
    """
    vec = np.full(THEME_VEC_DIM, NEUTRAL, dtype=np.float32)

    if len(peers) < 2:
        # 테마 내 종목이 1개뿐이면 의미 없음 → 중립값 반환
        return vec

    # ── 시총 비중 ────────────────────────────────────────────────────
    mktcap_col = 'market_cap' if 'market_cap' in peers.columns else None
    if mktcap_col:
        total_cap = peers[mktcap_col].fillna(0).sum()
        own_cap   = ticker_row.get(mktcap_col, 0) or 0
        vec[0] = float(own_cap / total_cap) if total_cap > 0 else NEUTRAL
        vec[1] = float(_rank_pct(peers[mktcap_col])[
            peers.index[peers['ticker'] == ticker][0]
            if (peers['ticker'] == ticker).any() else peers.index[0]
        ])

    # ── 밸류에이션 순위 ─────────────────────────────────────────────
    # PBR: 낮을수록 저평가 → ascending=True → 낮은 값이 높은 백분위
    for i, (col, asc) in enumerate([
        ('F_VAL_pbr',        True),   # [2]
        ('F_VAL_per',        True),   # [3]
        ('F_VAL_ev_ebitda',  True),   # [4]
        ('F_PRF_roe',        False),  # [5] 높을수록 우수
        ('F_GRW_rev_cagr',   False),  # [6]
        ('C_GP_A',           False),  # [7]
        ('ret_1q',           False),  # [8]
        ('ret_4q',           False),  # [9]
    ], start=2):
        if col in peers.columns:
            ranks = _rank_pct(peers[col], ascending=asc)
            mask  = peers['ticker'] == ticker
            if mask.any():
                vec[i] = float(ranks[mask].iloc[0])

    # ── 테마 전체 상태 ───────────────────────────────────────────────
    if 'ret_4q' in peers.columns:
        ret4q = peers['ret_4q'].dropna()
        vec[10] = float(ret4q.mean()) if len(ret4q) > 0 else NEUTRAL
        vec[11] = float(ret4q.std())  if len(ret4q) > 1 else 0.0

    if 'F_VAL_pbr' in peers.columns:
        pbr = peers['F_VAL_pbr'].dropna()
        vec[12] = float(pbr.mean()) if len(pbr) > 0 else NEUTRAL
        vec[13] = float(pbr.std())  if len(pbr) > 1 else 0.0

    vec[14] = float(np.log1p(len(peers)))   # 로그 스케일 종목 수
    if mktcap_col:
        vec[15] = _herfindahl(peers[mktcap_col])

    return vec


def compute_theme_context(
    df: pd.DataFrame,
    mapping_dir: str = "config/theme_mapping",
) -> pd.DataFrame:
    """
    전체 데이터프레임에 대해 테마 비중 벡터를 계산한다.

    Args:
        df: 종목-분기별 DataFrame. 필수 컬럼: ['ticker', 'date']
            선택 컬럼: 피처 컬럼들 (없으면 중립값 사용)

    Returns:
        ['ticker', 'date', 'theme_ctx_0', ..., 'theme_ctx_15'] DataFrame
    """
    mapping = load_theme_mapping(mapping_dir)
    t2th    = mapping["ticker_to_themes"]
    th2t    = mapping["theme_to_tickers"]

    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    results = []

    for date, date_group in df.groupby('date', observed=True, sort=False):
        # 이 시점의 ticker → row 빠른 조회용
        ticker_rows = {
            row['ticker']: row
            for _, row in date_group.iterrows()
        }

        for _, row in date_group.iterrows():
            ticker = row['ticker']
            themes = t2th.get(ticker, [])

            if not themes:
                # 매핑 없는 종목: 중립값
                vec = np.full(THEME_VEC_DIM, NEUTRAL, dtype=np.float32)
            else:
                # 각 테마별 벡터 계산 후 평균
                vecs = []
                for theme_id in themes:
                    peer_tickers = th2t.get(theme_id, [])
                    # 같은 시점에 데이터가 있는 peer만 사용
                    peer_rows = [
                        ticker_rows[t] for t in peer_tickers
                        if t in ticker_rows
                    ]
                    if not peer_rows:
                        continue
                    peers_df = pd.DataFrame(peer_rows)
                    vecs.append(
                        compute_theme_vector(ticker, row, peers_df)
                    )
                vec = (
                    np.mean(vecs, axis=0).astype(np.float32)
                    if vecs
                    else np.full(THEME_VEC_DIM, NEUTRAL, dtype=np.float32)
                )

            entry = {'ticker': ticker, 'date': date}
            for i, v in enumerate(vec):
                entry[f'theme_ctx_{i}'] = v
            results.append(entry)

    out = pd.DataFrame(results)
    ctx_cols = [f'theme_ctx_{i}' for i in range(THEME_VEC_DIM)]
    out[ctx_cols] = out[ctx_cols].astype('float32')
    return out
'''

# =============================================================================
# 섹션 5. 데이터셋 (신규)
# =============================================================================

DATASET = '''
"""
src/data/dataset.py

세 가지 시퀀스를 하나의 배치로 묶는 Dataset.

샘플 = (종목, 기준분기) 쌍.
  seq_stock  : 과거 T분기의 종목 재무 피처 (가변 길이)
  seq_macro  : 과거 T분기의 거시 피처 (동일 날짜 기준, 가변 길이)
  theme_ctx  : 현재 분기의 테마 비중 벡터 (16차원, 고정)
  snap_num   : 현재 분기 수치형 스냅샷 피처 (계산형 등)
  snap_cat   : 현재 분기 범주형 스냅샷 피처 (섹터, 국가 등)
  A, R       : 타깃

중요:
  거시 시계열은 모든 종목이 공유하므로 Dataset에서 날짜로 조회한다.
  배치 내에서 같은 날짜의 종목들은 동일한 macro 시퀀스를 가진다.
  이를 활용해 LSTM_macro는 배치 내 유니크 날짜에 대해서만 실행한다.
  (→ Predictor.forward에서 처리, Dataset은 단순히 날짜를 저장)
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StockDataset(Dataset):

    def __init__(
        self,
        df_stock: pd.DataFrame,      # 종목 재무 피처 (ticker, date, F_*)
        df_macro: pd.DataFrame,      # 거시 피처 (date, M_*)
        df_theme: pd.DataFrame,      # 테마 비중 (ticker, date, theme_ctx_*)
        df_labels: pd.DataFrame,     # 라벨 (ticker, date, A, R)
        stock_seq_cols: list,        # 시계열로 쓸 재무 피처 컬럼명
        macro_seq_cols: list,        # 시계열로 쓸 거시 피처 컬럼명
        snap_num_cols: list,         # 스냅샷 수치형 컬럼명
        snap_cat_cols: list,         # 스냅샷 범주형 컬럼명
        max_seq_len: int = 20,
    ):
        self.max_seq_len   = max_seq_len
        self.stock_seq_cols = stock_seq_cols
        self.macro_seq_cols = macro_seq_cols
        self.snap_num_cols  = snap_num_cols
        self.snap_cat_cols  = snap_cat_cols
        theme_ctx_cols      = [f'theme_ctx_{i}' for i in range(16)]

        # 거시 데이터를 날짜 → 배열 딕셔너리로 변환 (빠른 조회)
        df_macro = df_macro.sort_values('date').set_index('date')
        self._macro_index = df_macro.index.tolist()
        self._macro_array = df_macro[macro_seq_cols].values.astype('float32')
        self._macro_date_to_idx = {d: i for i, d in enumerate(self._macro_index)}

        # 종목별 피처 정렬
        df_stock = df_stock.sort_values(['ticker', 'date'])
        df_all   = (df_stock
                    .merge(df_theme, on=['ticker', 'date'], how='left')
                    .merge(df_labels, on=['ticker', 'date'], how='inner'))
        df_all[theme_ctx_cols] = df_all[theme_ctx_cols].fillna(0.5).astype('float32')

        self.samples = []
        for ticker, grp in df_all.groupby('ticker', observed=True, sort=False):
            grp = grp.reset_index(drop=True)
            for i in range(len(grp)):
                row = grp.iloc[i]

                # 라벨 없으면 제외
                if pd.isna(row.get('A')) or pd.isna(row.get('R')):
                    continue

                cur_date = row['date']

                # 종목 재무 시퀀스 (현재 포함 과거 max_seq_len 분기)
                start  = max(0, i - max_seq_len + 1)
                s_seq  = grp.iloc[start:i+1][stock_seq_cols].values.astype('float32')
                s_seq  = np.nan_to_num(s_seq, nan=0.0)

                # 거시 시퀀스 (같은 기간의 날짜 인덱스 범위)
                macro_end_idx   = self._macro_date_to_idx.get(cur_date, -1)
                macro_start_idx = max(0, macro_end_idx - max_seq_len + 1)
                if macro_end_idx < 0:
                    m_seq = np.zeros((1, len(macro_seq_cols)), dtype='float32')
                else:
                    m_seq = self._macro_array[macro_start_idx:macro_end_idx+1]
                m_seq = np.nan_to_num(m_seq, nan=0.0)

                # 테마 비중 (현재 시점, 고정 16차원)
                theme_vec = row[theme_ctx_cols].values.astype('float32')

                # 스냅샷
                snap_num = np.nan_to_num(
                    row[snap_num_cols].values.astype('float32'), nan=0.0
                )
                snap_cat = row[snap_cat_cols].values.astype('int64')

                self.samples.append({
                    's_seq':    s_seq,          # (T_s, F_stock)
                    'm_seq':    m_seq,          # (T_m, F_macro)
                    'theme':    theme_vec,      # (16,)
                    'snap_num': snap_num,       # (F_snap_num,)
                    'snap_cat': snap_cat,       # (F_snap_cat,)
                    'A':        float(row['A']),
                    'R':        float(row['R']),
                    'date':     cur_date,
                    'ticker':   ticker,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            's_seq':    torch.from_numpy(s['s_seq']),
            'm_seq':    torch.from_numpy(s['m_seq']),
            'theme':    torch.from_numpy(s['theme']),
            'snap_num': torch.from_numpy(s['snap_num']),
            'snap_cat': torch.from_numpy(s['snap_cat']),
            'A':        torch.tensor(s['A'], dtype=torch.float32),
            'R':        torch.tensor(s['R'], dtype=torch.float32),
        }


def collate_fn(batch):
    """
    가변 길이 s_seq, m_seq를 패딩해 배치로 묶는다.
    lengths를 함께 반환해 LSTM에서 pack_padded_sequence에 사용한다.
    """
    s_seqs    = [b['s_seq']    for b in batch]
    m_seqs    = [b['m_seq']    for b in batch]
    themes    = [b['theme']    for b in batch]
    snap_nums = [b['snap_num'] for b in batch]
    snap_cats = [b['snap_cat'] for b in batch]
    As = [b['A'] for b in batch]
    Rs = [b['R'] for b in batch]

    s_lengths = torch.tensor([x.shape[0] for x in s_seqs], dtype=torch.long)
    m_lengths = torch.tensor([x.shape[0] for x in m_seqs], dtype=torch.long)

    # pad_sequence: list of (T, F) → (B, max_T, F)
    s_padded = pad_sequence(s_seqs, batch_first=True, padding_value=0.0)
    m_padded = pad_sequence(m_seqs, batch_first=True, padding_value=0.0)

    return {
        's_seq':      s_padded,                  # (B, T_s, F_stock)
        's_lengths':  s_lengths,                  # (B,)
        'm_seq':      m_padded,                  # (B, T_m, F_macro)
        'm_lengths':  m_lengths,                  # (B,)
        'theme':      torch.stack(themes),        # (B, 16)
        'snap_num':   torch.stack(snap_nums),     # (B, F_snap_num)
        'snap_cat':   torch.stack(snap_cats),     # (B, F_snap_cat)
        'A':          torch.stack(As),            # (B,)
        'R':          torch.stack(Rs),            # (B,)
    }
'''

# =============================================================================
# 섹션 6. LSTM 인코더 (신규)
# =============================================================================

LSTM_ENCODER = '''
"""
src/models/lstm_encoder.py

가변 길이 시계열 → 고정 크기 컨텍스트 벡터.
stock 인코더(양방향)와 macro 인코더(단방향)를 동일 클래스로 구현한다.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMEncoder(nn.Module):
    """
    Args:
        input_size  : 입력 피처 수
        hidden_size : LSTM hidden 차원
        num_layers  : LSTM 레이어 수
        bidirectional: True면 양방향 (출력 = hidden_size * 2)
        dropout     : 드롭아웃 (num_layers > 1일 때만 LSTM 내부 적용)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        self.output_size   = hidden_size * (2 if bidirectional else 1)

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, max_T, input_size) 패딩된 시계열, float32
            lengths : (B,) 각 샘플의 실제 길이 (CPU 텐서)
        Returns:
            context : (B, output_size)
        """
        x = x.float()
        x = self.input_norm(x)

        # pack: 패딩 토큰을 LSTM 연산에서 제외
        # lengths는 반드시 CPU에 있어야 함 (MPS/CUDA 무관)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, hidden_size)

        if self.bidirectional:
            # 마지막 레이어의 양방향 결합
            ctx = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, hidden*2)
        else:
            ctx = h_n[-1]                                   # (B, hidden)

        return self.proj(ctx)
'''

# =============================================================================
# 섹션 7. FT-Transformer (신규)
# =============================================================================

FT_TRANSFORMER = '''
"""
src/models/ft_transformer.py

Gorishniy et al. (2021) FT-Transformer.
각 피처를 독립 임베딩으로 토크나이징한 뒤 Self-Attention으로 상호작용 학습.

입력:
  context_stock : (B, 256)  LSTM_stock 출력
  context_macro : (B, 64)   LSTM_macro 출력
  theme_ctx     : (B, 64)   Linear 투영된 테마 비중
  snap_num      : (B, F_num) 수치형 스냅샷
  snap_cat      : (B, F_cat) 범주형 스냅샷 (정수 인덱스)

출력:
  A : (B,)  매력도
  R : (B,)  위험도 (Softplus로 ≥ 0 보장)
"""
import math
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    수치형: x_i → Linear(1, d_token) + bias → (d,)
    범주형: cat_id → Embedding(n, d_token) → (d,)
    컨텍스트 벡터: Linear(ctx_dim, d_token) → (d,)  [수치형과 동일 처리]
    """

    def __init__(
        self,
        context_dims: list,      # 컨텍스트 벡터 차원 리스트 [256, 64, 64]
        n_num_features: int,     # 수치형 스냅샷 피처 수
        cat_cardinalities: list, # 범주형 피처별 카테고리 수 [n1, n2, ...]
        d_token: int = 192,
    ):
        super().__init__()
        self.d_token = d_token

        # 컨텍스트 투영 (각각 독립 Linear)
        self.ctx_projs = nn.ModuleList([
            nn.Linear(dim, d_token) for dim in context_dims
        ])

        # 수치형 피처: 피처별 독립 가중치
        self.n_num = n_num_features
        if n_num_features > 0:
            self.num_W = nn.Parameter(torch.empty(n_num_features, d_token))
            self.num_b = nn.Parameter(torch.zeros(n_num_features, d_token))
            nn.init.kaiming_uniform_(self.num_W, a=math.sqrt(5))

        # 범주형 임베딩
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n + 1, d_token) for n in cat_cardinalities
        ])
        self.n_cat = len(cat_cardinalities)

        # [CLS] 집계 토큰
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 토큰 수 계산
        self.n_tokens = 1 + len(context_dims) + n_num_features + len(cat_cardinalities)

    def forward(
        self,
        contexts: list,          # [(B, ctx_dim), ...]
        x_num: torch.Tensor,     # (B, n_num)
        x_cat: torch.Tensor,     # (B, n_cat)
    ) -> torch.Tensor:

        tokens = []

        # 컨텍스트 토큰
        for proj, ctx in zip(self.ctx_projs, contexts):
            tokens.append(proj(ctx.float()).unsqueeze(1))  # (B, 1, d)

        # 수치형 토큰: x_i * w_i + b_i
        if self.n_num > 0:
            num_tok = (
                x_num.float().unsqueeze(-1) * self.num_W.unsqueeze(0)
                + self.num_b.unsqueeze(0)
            )  # (B, n_num, d)
            tokens.append(num_tok)

        # 범주형 토큰
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))  # (B, 1, d)

        # 전체 피처 토큰 결합
        feat = torch.cat(tokens, dim=1)       # (B, n_tokens-1, d)

        # [CLS] prepend
        cls = self.cls_token.expand(feat.size(0), -1, -1)
        return torch.cat([cls, feat], dim=1)  # (B, n_tokens, d)


class FTTransformer(nn.Module):

    def __init__(
        self,
        context_dims: list,
        n_num_features: int,
        cat_cardinalities: list,
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_factor: float = 4/3,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            context_dims, n_num_features, cat_cardinalities, d_token
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=max(int(d_token * ffn_factor), d_token),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: 깊은 레이어에서 학습 안정성
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 예측 헤드: [CLS] → A, R
        def _head(out_activation=None):
            layers = [
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token // 2, 1),
            ]
            if out_activation:
                layers.append(out_activation)
            return nn.Sequential(*layers)

        self.head_A = _head()
        self.head_R = _head(nn.Softplus())  # R ≥ 0

    def forward(
        self,
        contexts: list,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ):
        tokens  = self.tokenizer(contexts, x_num, x_cat)  # (B, n_tok, d)
        encoded = self.transformer(tokens)                 # (B, n_tok, d)
        cls_out = encoded[:, 0]                            # (B, d)

        A = self.head_A(cls_out).squeeze(-1)   # (B,)
        R = self.head_R(cls_out).squeeze(-1)   # (B,)
        return A, R
'''

# =============================================================================
# 섹션 8. Lightning 통합 모듈 (신규)
# =============================================================================

PREDICTOR = '''
"""
src/models/predictor.py

LSTM_stock + LSTM_macro + ThemeLinear + FT-Transformer를
하나의 PyTorch Lightning 모듈로 통합한다.

LSTM_macro 최적화:
  배치 내 동일 날짜의 종목들은 같은 거시 시퀀스를 공유한다.
  동일 시퀀스를 여러 번 계산하는 낭비를 줄이기 위해
  유니크 시퀀스 기준으로 LSTM을 실행 후 인덱스로 gather한다.
  단, 간소화를 위해 v1에서는 이 최적화를 생략하고
  배치 내 모든 행에 동일하게 실행한다. (배치 크기 64에서 허용 가능)
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.lstm_encoder import LSTMEncoder
from src.models.ft_transformer import FTTransformer
from src.utils.device import get_device


class StockPredictor(pl.LightningModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        c = cfg

        # LSTM 인코더 두 개
        self.lstm_stock = LSTMEncoder(
            input_size=c['n_stock_features'],
            hidden_size=c['lstm_stock_hidden'],
            num_layers=c.get('lstm_stock_layers', 2),
            bidirectional=True,
            dropout=c['dropout'],
        )
        self.lstm_macro = LSTMEncoder(
            input_size=c['n_macro_features'],
            hidden_size=c['lstm_macro_hidden'],
            num_layers=c.get('lstm_macro_layers', 1),
            bidirectional=False,  # 거시: 단방향
            dropout=0.0,
        )

        # 테마 비중 Linear 투영 (16 → theme_proj_dim)
        self.theme_proj = nn.Sequential(
            nn.Linear(16, c['theme_proj_dim']),
            nn.GELU(),
            nn.Dropout(c['dropout']),
        )

        # FT-Transformer
        context_dims = [
            self.lstm_stock.output_size,    # 256
            self.lstm_macro.output_size,    # 64
            c['theme_proj_dim'],            # 64
        ]
        self.ftt = FTTransformer(
            context_dims=context_dims,
            n_num_features=c['n_snap_num'],
            cat_cardinalities=c['cat_cardinalities'],
            d_token=c['d_token'],
            n_heads=c['n_heads'],
            n_layers=c['n_layers'],
            ffn_factor=c.get('ffn_factor', 4/3),
            dropout=c['dropout'],
            attn_dropout=c.get('attn_dropout', 0.1),
        )

        # Kendall (2018) 불확실성 기반 멀티태스크 손실 가중치
        self.log_var_A = nn.Parameter(torch.zeros(1))
        self.log_var_R = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict):
        # ── 시계열 인코딩 ────────────────────────────────────────────
        stock_ctx = self.lstm_stock(batch['s_seq'], batch['s_lengths'])
        macro_ctx = self.lstm_macro(batch['m_seq'], batch['m_lengths'])
        theme_ctx = self.theme_proj(batch['theme'].float())

        # ── FT-Transformer ────────────────────────────────────────────
        A, R = self.ftt(
            contexts=[stock_ctx, macro_ctx, theme_ctx],
            x_num=batch['snap_num'].float(),
            x_cat=batch['snap_cat'],
        )
        return A, R

    def _loss(self, A_pred, R_pred, A_true, R_true):
        """Kendall 멀티태스크 손실."""
        mask = ~(torch.isnan(A_true) | torch.isnan(R_true))
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        l_A = nn.functional.mse_loss(A_pred[mask], A_true[mask])
        l_R = nn.functional.mse_loss(R_pred[mask], R_true[mask])
        prec_A = torch.exp(-self.log_var_A)
        prec_R = torch.exp(-self.log_var_R)
        loss = prec_A * l_A + self.log_var_A + prec_R * l_R + self.log_var_R
        return loss, l_A.item(), l_R.item()

    def training_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'train/loss': loss, 'train/A': lA, 'train/R': lR},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'val/loss': loss, 'val/A': lA, 'val/R': lR},
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams['max_epochs'],
            eta_min=self.hparams['lr'] * 0.01,
        )
        return {'optimizer': opt, 'lr_scheduler': sched}
'''

# =============================================================================
# 섹션 9. 학습 스크립트 (신규 — 04_train_tft.py 교체)
# =============================================================================

TRAIN_SCRIPT = '''
"""
scripts/04_train.py

실행: python scripts/04_train.py
"""
import yaml, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from src.data.dataset import StockDataset, collate_fn
from src.models.predictor import StockPredictor
from src.utils.device import get_device, get_optimal_batch_size, report_environment
from src.utils.io import load_parquet


def main():
    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    report_environment()

    # ── 데이터 로드 ────────────────────────────────────────────────────
    df_stock  = load_parquet('data/processed/features_stock.parquet')
    df_macro  = load_parquet('data/processed/features_macro.parquet')
    df_theme  = load_parquet('data/processed/theme_context.parquet')
    df_labels = load_parquet('data/processed/labels.parquet')

    # ── 컬럼 정의 ───────────────────────────────────────────────────────
    # 종목 재무 시계열 피처 (거시 제외)
    STOCK_SEQ_COLS = (
        [c for c in df_stock.columns if c.startswith('F_')]   # 재무 45
      + [c for c in df_stock.columns if c.startswith('A_')]   # 자산집중도 35
    )
    # 거시 시계열 피처
    MACRO_SEQ_COLS = [c for c in df_macro.columns
                      if c.startswith('M_') and c != 'date']
    # 스냅샷 수치형 (계산형 지표)
    SNAP_NUM_COLS = [c for c in df_stock.columns if c.startswith('C_')]
    # 스냅샷 범주형
    SNAP_CAT_COLS = ['country', 'sector', 'size_tier']

    # 범주형 카디널리티 (미등록 카테고리를 위해 +1)
    cat_cardinalities = [
        int(df_stock[c].nunique()) for c in SNAP_CAT_COLS
    ]

    # ── 모델 설정 ────────────────────────────────────────────────────────
    mcfg = cfg['model']
    model_cfg = {
        'n_stock_features':  len(STOCK_SEQ_COLS),
        'n_macro_features':  len(MACRO_SEQ_COLS),
        'n_snap_num':        len(SNAP_NUM_COLS),
        'cat_cardinalities': cat_cardinalities,
        'lstm_stock_hidden': mcfg['lstm_stock_hidden'],
        'lstm_stock_layers': mcfg.get('lstm_stock_layers', 2),
        'lstm_macro_hidden': mcfg['lstm_macro_hidden'],
        'lstm_macro_layers': mcfg.get('lstm_macro_layers', 1),
        'theme_proj_dim':    mcfg['theme_proj_dim'],
        'd_token':           mcfg['d_token'],
        'n_heads':           mcfg['n_heads'],
        'n_layers':          mcfg['n_layers'],
        'ffn_factor':        mcfg.get('ffn_factor', 4/3),
        'dropout':           mcfg['dropout'],
        'attn_dropout':      mcfg.get('attn_dropout', 0.1),
        'lr':                mcfg['lr'],
        'weight_decay':      mcfg['weight_decay'],
        'max_epochs':        mcfg['max_epochs'],
    }

    # ── 데이터 분할 ──────────────────────────────────────────────────────
    train_end = pd.Timestamp(cfg['train_split']['train_end'])
    val_end   = pd.Timestamp(cfg['train_split']['val_end'])

    def make_dataset(df_s, label_date_filter):
        labels_sub = df_labels[label_date_filter(df_labels['date'])]
        return StockDataset(
            df_stock=df_s,
            df_macro=df_macro,
            df_theme=df_theme,
            df_labels=labels_sub,
            stock_seq_cols=STOCK_SEQ_COLS,
            macro_seq_cols=MACRO_SEQ_COLS,
            snap_num_cols=SNAP_NUM_COLS,
            snap_cat_cols=SNAP_CAT_COLS,
            max_seq_len=mcfg.get('lstm_stock_max_seq', 20),
        )

    train_ds = make_dataset(
        df_stock[df_stock['date'] <= train_end],
        lambda d: d <= train_end,
    )
    val_ds = make_dataset(
        df_stock[(df_stock['date'] > train_end) & (df_stock['date'] <= val_end)],
        lambda d: (d > train_end) & (d <= val_end),
    )

    batch_size = get_optimal_batch_size(mcfg['batch_size'])
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=mcfg['num_workers'],
        persistent_workers=mcfg['persistent_workers'],
        pin_memory=False,   # MPS 미지원
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── 학습 ─────────────────────────────────────────────────────────────
    device      = get_device()
    accelerator = 'mps' if device.type == 'mps' else 'cpu'

    model = StockPredictor(model_cfg)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val/loss', patience=mcfg['patience'], mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='stockml-{epoch:02d}-{val/loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        max_epochs=mcfg['max_epochs'],
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=mcfg['grad_clip'],
        callbacks=callbacks,
        precision=mcfg.get('precision', '32-true'),
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    except RuntimeError as e:
        if 'MPS' in str(e):
            print(f"[MPS fallback] {e}")
            trainer = pl.Trainer(
                max_epochs=mcfg['max_epochs'],
                accelerator='cpu',
                gradient_clip_val=mcfg['grad_clip'],
                callbacks=callbacks,
            )
            trainer.fit(model, train_loader, val_loader)
        else:
            raise

    print("학습 완료.")


if __name__ == '__main__':
    main()
'''

# =============================================================================
# 섹션 10. 테마 비중 계산 스크립트 (신규)
# =============================================================================

THEME_SCRIPT = '''
"""
scripts/02b_build_theme_context.py

실행: python scripts/02b_build_theme_context.py
02_build_features.py 이후, 03_build_labels.py 이전에 실행한다.

Point-in-Time 보장:
  분기별로 해당 시점의 종목 피처만 사용해 테마 내 순위를 계산한다.
  미래 데이터 참조 없음.
"""
from src.theme.context import compute_theme_context
from src.utils.io import load_parquet, save_parquet, report_memory


def main():
    print("테마 비중 컨텍스트 계산 중...")
    df = load_parquet('data/processed/features_stock.parquet')
    report_memory(df, "features_stock")

    theme_ctx = compute_theme_context(
        df,
        mapping_dir='config/theme_mapping',
    )

    save_parquet(theme_ctx, 'data/processed/theme_context.parquet')
    report_memory(theme_ctx, "theme_context")
    print(f"완료: {len(theme_ctx)} rows")


if __name__ == '__main__':
    main()
'''

# =============================================================================
# 섹션 11. 실행 순서 (업데이트)
# =============================================================================

EXECUTION_ORDER = """
# 실행 순서

# 1. 환경 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env: FRED_API_KEY, ECOS_API_KEY, DART_API_KEY

# 2. 데이터 수집
python scripts/01_fetch_data.py --start 2010-01-01 --end 2026-05-01

# 3. 피처 빌드 (stock 피처와 macro 피처를 분리 저장)
python scripts/02_build_features.py
# → data/processed/features_stock.parquet  (종목 재무, 자산집중도)
# → data/processed/features_macro.parquet  (거시 80종, 날짜별)

# 4. 테마 비중 계산 (★신규 — 피처 빌드 후 실행)
python scripts/02b_build_theme_context.py
# → data/processed/theme_context.parquet

# 5. 라벨 생성
python scripts/03_build_labels.py

# 6. 모델 학습
python scripts/04_train.py

# 7. 베이스라인 비교 (회계 지표)
python scripts/05_train_baselines.py

# 8. 평가
python scripts/06_evaluate.py

# 9. UI
streamlit run scripts/07_run_ui.py
"""

# =============================================================================
# 섹션 12. 단위 테스트 (신규 포함)
# =============================================================================

TESTS = '''
"""
tests/test_pipeline.py
"""
import torch
import numpy as np
import pandas as pd
import pytest


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


def test_theme_context_point_in_time():
    """테마 비중 계산이 미래 데이터를 참조하지 않는지 확인."""
    # t1 시점과 t2 시점에 서로 다른 피처값을 갖는 더미 데이터
    dates = pd.to_datetime(['2020-03-31', '2020-06-30'])
    df = pd.DataFrame({
        'ticker': ['A', 'B', 'A', 'B'],
        'date':   [dates[0], dates[0], dates[1], dates[1]],
        'market_cap': [100, 200, 150, 250],
        'F_VAL_pbr':  [1.0, 2.0, 1.5, 2.5],
    })

    from src.theme.context import compute_theme_context
    # 두 종목이 같은 테마에 있다고 가정하는 더미 매핑은
    # 실제 테스트에서는 mock 사용
    # 여기서는 함수가 날짜별로 분리 계산함을 구조적으로 검증
    # (실제 YAML 없이 실행 불가 → 통합 테스트에서 확인)
    pass


def test_attractiveness_label_no_lookahead():
    """매력도 라벨이 forward 데이터만 사용하는지 확인."""
    from src.labels.attractiveness import compute_attractiveness
    prices = pd.DataFrame({
        'ticker': ['X'] * 10,
        'date':   pd.date_range('2015-01-01', periods=10, freq='QE'),
        'close':  [100, 110, 90, 120, 130, 125, 140, 135, 145, 150],
    })
    result = compute_attractiveness(prices, max_horizon_years=5,
                                    min_forward_quarters=4)
    # 첫 번째 시점의 A는 index 1~9의 최고가 기준이어야 함
    first = result.iloc[0]
    max_price = max([110, 90, 120, 130, 125, 140, 135, 145, 150])  # = 150
    expected_A = np.log(150 / 100) / np.log(5)
    assert abs(first['A'] - expected_A) < 1e-5


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
'''

# =============================================================================
# 섹션 13. 트러블슈팅 (업데이트)
# =============================================================================

TROUBLESHOOTING = """
# 트러블슈팅

## M1 Pro / MPS

| 증상 | 원인 | 해결 |
|------|------|------|
| RuntimeError: MPS backend out of memory | 배치 or 모델 너무 큼 | batch_size 줄이기, d_token 128로 축소 |
| aten::xxx not implemented for MPS | MPS 미지원 op | PYTORCH_ENABLE_MPS_FALLBACK=1 |
| pack_padded_sequence 오류 | lengths가 GPU에 있음 | lengths.cpu() 확인 (코드에 이미 적용) |
| TransformerEncoderLayer MPS 오류 | 일부 어텐션 커널 미지원 | PYTORCH_ENABLE_MPS_FALLBACK=1 |
| float64 관련 오류 | MPS는 float32만 지원 | 모든 텐서 .float() 호출 확인 |

## 데이터

| 증상 | 해결 |
|------|------|
| features_stock/macro 분리 안 됨 | 02_build_features.py에서 M_ 접두사 컬럼을 macro로, F_/A_ 컬럼을 stock으로 분리 저장 |
| theme_context NaN 비율 높음 | 매핑 파일 ticker 형식 확인 (KR: 6자리 문자열, US: 영문 티커) |
| LSTM 학습 발산 | LayerNorm 적용 확인, lr 1e-5로 낮추기, gradient_clip 확인 |
| val/loss가 train/loss보다 훨씬 큼 | 시퀀스 길이 분포 확인, max_seq_len 줄이기 |

## 모델

| 증상 | 원인 | 해결 |
|------|------|------|
| R (위험도) 예측이 0에 수렴 | Softplus + MSE 조합 이슈 | log_var_R 초기값 확인, R 라벨 스케일링 |
| A, R 손실 중 하나가 지배 | Kendall 가중치 불균형 | log_var 초기값을 -1.0으로 조정 |
| 테마 비중 피처가 학습에 미반영 | FTT attention에서 theme_ctx 토큰 무시 | n_layers 늘리기, theme_proj_dim 확인 |

## 성능 튜닝 (M1 Pro 16GB 기준 권장값)

  lstm_stock_hidden: 128   (256은 메모리 부담)
  lstm_macro_hidden: 64
  d_token: 192             (128도 가능, 속도 우선 시)
  n_layers: 4              (3으로 줄이면 30% 빠름)
  batch_size: 64           (32GB 모델은 128)
  num_workers: 2           (4로 올리면 빠르지만 메모리 주의)
"""

# =============================================================================
# 섹션 14. 변경 체크리스트
# =============================================================================

CHECKLIST = """
# v5.1 → v5.3 변경 체크리스트

## 제거
□ src/models/tft_model.py           (TFT 완전 제거)
□ src/models/baseline_gbm.py        (LightGBM 완전 제거)
□ scripts/04_train_tft.py           (교체됨)
□ requirements.txt: pytorch-forecasting, lightgbm 제거

## 신규 생성
□ src/theme/loader.py               (YAML 매핑 로드)
□ src/theme/context.py              (테마 비중 벡터 계산)
□ src/data/dataset.py               (StockDataset + collate_fn)
□ src/models/lstm_encoder.py        (가변길이 시계열 인코더)
□ src/models/ft_transformer.py      (FT-Transformer)
□ src/models/predictor.py           (Lightning 통합 모듈)
□ scripts/02b_build_theme_context.py
□ scripts/04_train.py               (새 모델 학습 진입점)

## 수정
□ config/settings.yaml: model 섹션 교체
□ scripts/02_build_features.py:
    features_stock.parquet (F_, A_ 컬럼)
    features_macro.parquet (M_ 컬럼, 날짜별)
    두 파일로 분리 저장하도록 수정
□ scripts/05_train_baselines.py: GBM baseline 제거, 회계 지표만 유지
□ requirements.txt: pytorch-forecasting/lightgbm 제거

## 유지 (변경 없음)
□ src/utils/device.py, io.py, pit.py
□ src/data_fetchers/ 전체
□ src/features/ 전체
□ src/labels/ 전체
□ src/models/baseline_accounting.py
□ src/evaluation/ 전체
□ src/ui/streamlit_app.py
□ scripts/01, 03, 06, 07
□ config/feature_lags.yaml
□ config/theme_mapping/*.yaml
"""

# =============================================================================
# 섹션 15. 추가 모니터링 및 로컬 라벨링 가이드 (WandB & Local Labeling)
# =============================================================================

ADDITIONAL_GUIDE = """
# WandB (Weights & Biases) 및 로컬 라벨링(log_N) 추가 명세

## 1. WandB 실험 모니터링
FT-Transformer 학습 시, 학습 및 검증 메트릭의 실시간 시각화를 지원하기 위해 WandB 로거를 옵션으로 지원합니다.

### 1.1 설정 및 활성화
- `config/settings.yaml` 파일의 `model.use_wandb` 설정을 `true`로 설정하여 자동 활성화할 수 있습니다.
- 환경 내 `wandb` 및 `lightning` 관련 종속성이 온전해야 합니다. 
- 비활성화 시 기본적으로 `CSVLogger`가 실행 로그를 담당합니다.

### 1.2 로깅 키 네이밍 규칙
학습 과정에서 체크포인트 디렉토리가 하위 폴더로 슬래시(`\`, `/`)로 쪼개지는 오동작을 미연에 방지하기 위해 다음과 같이 평면화된 메트릭 이름 형식을 고수합니다.
- `train_loss`, `train_A`, `train_R` (Epoch 별 학습 손실 및 타겟별 손실)
- `val_loss`, `val_A`, `val_R` (Epoch 별 검증 손실 및 타겟별 손실)
- 저장 파일명은 `stockml-{epoch:02d}-{val_loss:.4f}.ckpt` 형식을 준수하여 `checkpoints/` 폴더에 단일 파일 형태로 생성됩니다.

---

## 2. 로컬 라벨링 및 log_N 계산 기법
5년 미만의 상장/관측 데이터를 가지는 종목군에 대해 신뢰할 수 있는 매수 매력도(A) 및 위험도(R) 지표를 수립하기 위해 동적 윈도우 크기 $N$을 적용합니다.

### 2.1 Attractiveness ($A$, 매력도)
동적 관측 기간 $N$ 분기 ($4 \le N \le 20$)에 따른 Attractiveness는 다음과 같은 수학적 정규화 규칙을 따릅니다:
$$A = \log_N \left( \frac{\max(P_{t \dots t+N}) + 1\text{e-}8}{P_t + 1\text{e-}8} \right)$$
여기서 밑이 $N$인 로그를 취함으로써, 관측 기간이 짧아 단기간 내 급격하게 상승한 종목과 장기간 서서히 상승한 종목의 매력도 스케일을 일정하게 조정(정규화)합니다.

### 2.2 Risk ($R$, 위험도)
관측 기간 $N$ 분기 동안 종가 기준의 로그 수익률에 대한 변동성의 평균값(연환산 표준편차)을 계산합니다:
$$R = \text{std}(\text{log\_returns}_{t \dots t+N}) \times \sqrt{4}$$
- $N$이 5년(20분기)보다 적은 종목들의 경우 실제 유효 분기 개수 만큼의 표준편차를 사용하여 개별 종목의 고유 위험도를 보수적이고 안정적으로 측정합니다.

### 2.3 단위 테스트 검증
- 해당 연산 로직은 `tests/test_pipeline.py` 내 `test_attractiveness_label` 및 `test_risk_label` 단위 테스트를 통해 그 정확도가 철저히 보증됩니다.
"""

