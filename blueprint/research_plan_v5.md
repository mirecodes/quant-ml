# 연구 계획서 v5.1 — Coding Agent 구현 가이드

> **이 문서는 사람이 읽는 계획서가 아닌, AI Coding Agent가 자율적으로 시스템을 구현할 수 있도록 작성된 기술 명세서입니다.**  
> **타깃 환경:** Apple M1 MacBook Pro (Apple Silicon, MPS backend)  
> **러닝 모델:** Temporal Fusion Transformer (TFT)

### 4.6 통합 데이터 수집 스크립트

```python
# scripts/01_fetch_data.py
import argparse
import pandas as pd
from src.data_fetchers.prices_kr import KoreanPriceFetcher
from src.data_fetchers.prices_us import USPriceFetcher
from src.data_fetchers.macro_us import FredMacroFetcher
from src.utils.io import save_parquet, report_memory

def get_sp500_tickers() -> list:
    """Wikipedia에서 현재 S&P 500 구성종목 목록."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500 = tables[0]
    tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def main(args):
    # KR: KOSPI만
    kr_fetcher = KoreanPriceFetcher(cache_dir='data/raw/prices')
    kr_prices = kr_fetcher.fetch(args.start, args.end)
    report_memory(kr_prices, "KR prices")
    
    # US: S&P 500만
    us_tickers = get_sp500_tickers()
    us_fetcher = USPriceFetcher(cache_dir='data/raw/prices')
    us_prices = us_fetcher.fetch(us_tickers, args.start, args.end)
    report_memory(us_prices, "US prices")
    
    # 통합 저장
    prices = pd.concat([kr_prices, us_prices], ignore_index=True)
    save_parquet(prices, 'data/processed/prices_quarterly.parquet')
    
    # 거시 (미국만 FRED, 한국은 별도)
    fred = FredMacroFetcher(cache_dir='data/raw/macro')
    macro_us = fred.fetch(args.start, args.end)
    save_parquet(macro_us, 'data/processed/macro_us.parquet')
    
    print("Data fetch complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2026-05-01')
    args = parser.parse_args()
    main(args)
```

---

## v5 → v5.1 핵심 변경

| 항목 | v5 | v5.1 |
|------|----|----|
| 매력도 타깃 윈도우 | 윈도우 내 분기별 최고가 | **현재 시점 미래 잠재력 (단일값)** |
| 매력도 윈도우 부족 시 | 가능한 기간으로 처리 | **가용 기간만 사용 (최대 5년)** |
| 데이터 유니버스 | KRX 전체, 미국 전체 | **KOSPI + S&P 500** |
| 데이터 유니버스 | KRX 전체, 미국 전체 | **KOSPI + S&P 500** |
| 환경 | M1 Mac (MPS) 특화 | **M1 Pro 특화 + 메모리 최적화** |

---

## 0. 타깃 변수 최종 정의 (단일 값)

### 0.1 매력도 (Attractiveness)

```
주어진 시점 t와 최대 호라이즌 5년:

  max_quarters = 5 × 4 = 20분기
  
  forward_window = quarterly_closes[t+1Q : t+max_quarters Q] (부족 시 가용 기간만)
  
  if len(forward_window) < 4:
      label = NaN   # 라벨 없음, 최소 1년치 데이터 필요
  else:
      max_price = max(forward_window)
      A(t) = log_5(max_price / P_t)
```

**예시:**
- A = 1 의 의미: "향후 최대 5년 내 약 5배 도달 가능성"

### 0.2 위험도 (Risk)

```
주어진 시점 t와 최대 호라이즌 5년 (부족 시 가용 기간):

  quarterly_returns = [log(Q_{i+1} / Q_i) for i in window]
  R(t) = std(quarterly_returns) × √4   # 연환산
```

### 0.3 데이터 부족 처리 정책

```python
MIN_FORWARD_QUARTERS = 4   # 최소 1년치 데이터는 있어야 라벨 생성
```

---
## 1. 환경 설정 (M1 Mac 특화)

### 1.1 Python 버전 및 가상환경

```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 1.2 PyTorch (M1 MPS 백엔드)

```bash
# PyTorch 2.x은 M1 MPS를 native 지원
pip install torch torchvision

# 설치 확인
python -c "import torch; print(torch.backends.mps.is_available())"
# True 가 출력되어야 함
```

### 1.3 핵심 라이브러리

```bash
# requirements.txt 내용
cat > requirements.txt << 'EOF'
torch>=2.2.0
pytorch-lightning>=2.2.0
pytorch-forecasting>=1.0.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.2.0
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
EOF

pip install -r requirements.txt
```

### 1.4 M1 Mac 특화 환경변수

```bash
# .env 또는 쉘 프로파일에 추가
export PYTORCH_ENABLE_MPS_FALLBACK=1   # MPS 미지원 op는 CPU로 자동 fallback
export TOKENIZERS_PARALLELISM=false
```

### 1.5 디바이스 및 메모리 최적화 헬퍼

```python
# src/utils/device.py
import torch
import platform
import psutil

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_optimal_batch_size(base_batch_size: int = 64) -> int:
    """M1 Pro 메모리에 따라 batch_size 자동 조정."""
    mem_gb = psutil.virtual_memory().total / 1e9
    
    if mem_gb >= 32:
        return base_batch_size * 2      # M1 Pro 32GB → 128
    elif mem_gb >= 16:
        return base_batch_size           # M1 Pro 16GB → 64
    else:
        return base_batch_size // 2      # 그 외 → 32

def report_environment():
    """학습 시작 전 환경 정보 출력."""
    print(f"Platform: {platform.machine()} / {platform.system()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical")
    print(f"RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"Device: {get_device()}")
    print(f"Recommended batch_size: {get_optimal_batch_size()}")

def to_device(tensor_or_model, device=None):
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)
```

### 1.6 데이터 타입 최적화 정책 (신규)

**원칙:** 메모리·연산 절감을 위해 모든 데이터에 명시적 dtype 적용.

```python
# src/utils/io.py
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DTYPE_POLICY = {
    'ticker':       'category',
    'country':      'category',
    'sector':       'category',
    'size_tier':    'category',
    'currency':     'category',
    'date':         'datetime64[ns]',
    'close':        'float32',
    'volume':       'float32',
    'market_cap':   'float32',
}

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in DTYPE_POLICY:
            try:
                df[col] = df[col].astype(DTYPE_POLICY[col])
            except Exception:
                pass
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
        elif pd.api.types.is_object_dtype(df[col]):
            if df[col].nunique() / max(len(df), 1) < 0.5:
                df[col] = df[col].astype('category')
    return df

def save_parquet(df: pd.DataFrame, path: str):
    df = optimize_dtypes(df)
    df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine='pyarrow')
    return optimize_dtypes(df)

def report_memory(df: pd.DataFrame, label: str = ""):
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"[{label}] shape={df.shape}, memory={mem_mb:.1f} MB")
```
### 1.7 알려진 M1 이슈 및 대응

| 이슈 | 대응 |
|------|------|
| MPS에서 일부 op 미지원 (예: aten::_linalg_solve_ex) | `PYTORCH_ENABLE_MPS_FALLBACK=1` 환경변수 |
| float64 미지원 | 모든 텐서 `dtype=torch.float32` 명시 |
| pytorch-forecasting QuantileLoss MPS 이슈 가능 | 발생 시 trainer에 `accelerator='cpu'` |
| 메모리 부족 (8GB 모델) | batch_size 작게 (32~64), grad_accumulation 사용 |

---

## 2. 프로젝트 디렉토리 구조

```
/
├── README.md
├── requirements.txt
├── .env.example                  # API 키 템플릿
├── config/
│   ├── settings.yaml             # 전체 설정
│   ├── feature_lags.yaml         # Point-in-Time lag
│   └── indicator_list.yaml       # 지표 ID 목록
│
├── data/
│   ├── raw/                      # 원본 다운로드
│   │   ├── prices/
│   │   ├── financials/
│   │   ├── macro/
│   │   └── themes/
│   ├── processed/                # 분기 집계 후
│   │   ├── prices_quarterly.parquet
│   │   ├── features.parquet
│   │   └── labels.parquet
│   └── splits/                   # train/val/test 분할
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── device.py             # M1 MPS 헬퍼
│   │   ├── io.py                 # parquet 입출력
│   │   └── time.py               # PiT 처리
│   │
│   ├── data_fetchers/            # 데이터 수집
│   │   ├── base.py
│   │   ├── prices_kr.py          # pykrx
│   │   ├── prices_us.py          # yfinance
│   │   ├── macro_us.py           # FRED
│   │   ├── macro_kr.py           # ECOS
│   │   ├── financials_kr.py      # DART (또는 sample)
│   │   ├── financials_us.py      # yfinance 재무
│   │   └── themes_naver.py       # 네이버 테마 크롤러
│   │
│   ├── features/                 # 피처 엔지니어링
│   │   ├── base.py               # BaseIndicator
│   │   ├── macro/                # 거시 80종
│   │   ├── fundamental/          # 재무 45종
│   │   ├── korean_asset/         # 한국 자산 35종
│   │   ├── computed/             # F-Score 등 7종 (입력용)
│   │   └── tags.py               # 테마 multi-hot
│   │
│   ├── labels/
│   │   ├── attractiveness.py     # max-window 라벨
│   │   └── risk.py               # 분기 volatility
│   │
│   ├── models/
│   │   ├── tft_model.py          # TFT 학습·추론
│   │   ├── baseline_gbm.py       # LightGBM baseline
│   │   └── baseline_accounting.py # F-Score, Quality 등
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # IC, ICIR, RMSE
│   │   ├── walk_forward.py
│   │   └── compare_baselines.py
│   │
│   └── ui/
│       └── streamlit_app.py
│
├── scripts/                      # 실행 진입점
│   ├── 01_fetch_data.py
│   ├── 02_build_features.py
│   ├── 03_build_labels.py
│   ├── 04_train_tft.py
│   ├── 05_train_baselines.py
│   ├── 06_evaluate.py
│   └── 07_run_ui.py
│
└── notebooks/
    ├── 00_explore_data.ipynb
    └── 99_diagnostics.ipynb
```

---

## 3. 설정 파일

### 3.1 `config/settings.yaml`

```yaml
project:
  name: stockml
  data_dir: ./data
  random_seed: 42

universe:
  countries: [KR, US]
  kr_market: KOSPI                  # KOSDAQ 제외
  us_index: SP500                   # S&P 500 만
  exclude_etfs: true
  
prices:
  frequency: quarterly              # 분기말 종가 (사용자 결정)
  source_kr: pykrx
  source_us: yfinance

targets:
  attractiveness:
    max_horizon_years: 5            # 최대 5년 윈도우
    log_base: 5                      # log_5 고정
    use_max_in_window: true
    min_forward_quarters: 4         # 최소 1년 이상의 데이터 필요
  
  risk:
    max_horizon_years: 5
    annualization_factor: 4         # 분기 → 연환산 √4
    min_forward_quarters: 4

train_split:
  train_end: '2017-12-31'
  val_start: '2018-01-01'
  val_end: '2019-12-31'
  test_start: '2020-01-01'
  test_end: '2021-12-31'            # 라벨 윈도우 확보 위해

model:
  type: tft
  hidden_size: 128                  # 64 → 128
  attention_head_size: 4
  dropout: 0.2
  hidden_continuous_size: 64        # 32 → 64
  learning_rate: 0.001
  batch_size: 64                    # 32 → 64 (16GB) / 96 (32GB 모델)
  max_epochs: 50
  patience: 8
  device: mps                       # M1 MPS
  
  # M1 Pro 추가 옵션
  num_workers: 2                    # 0 → 2 (M1 Pro 6+ 성능 코어)
  persistent_workers: true          # 워커 재사용으로 오버헤드 감소
  pin_memory: false                 # MPS는 pin_memory 미지원
  precision: "32-true"              # float32 명시 (MPS는 float64 미지원)

device:
  prefer: mps
  fallback: cpu
```

### 3.2 `config/feature_lags.yaml`

```yaml
# Point-in-Time: 피처가 실제로 알려진 시점
quarterly_financials:
  KR: 45                            # 분기 종료 후 45일 (DART 제출 기한)
  US: 45                            # 10-Q 제출 기한

monthly_macro:
  CPI: 30                           # 익월 초 발표
  GDP_advance: 30
  PMI: 1                            # 익월 1영업일
  employment: 7
  
weekly_macro:
  initial_claims: 7

daily_macro:
  rates: 1                          # 다음날부터 사용 가능
  fx: 1
```

---

## 4. 데이터 수집 모듈

### 4.1 `src/data_fetchers/base.py`

```python
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class BaseFetcher(ABC):
    """모든 데이터 소스의 공통 인터페이스."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """원본 데이터 수집."""
        ...
    
    def save(self, df: pd.DataFrame, name: str):
        df.to_parquet(self.cache_dir / f"{name}.parquet")
    
    def load(self, name: str) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None
```

### 4.2 `src/data_fetchers/prices_kr.py`

```python
from pykrx import stock
import pandas as pd
from .base import BaseFetcher

class KoreanPriceFetcher(BaseFetcher):
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """한국 종목 일별 종가 → 분기 종가로 집계."""
        # 종목 리스트 가져오기 (KOSPI 한정)
        tickers = stock.get_market_ticker_list(market="KOSPI")
        
        all_data = []
        for ticker in tickers:
            try:
                df = stock.get_market_ohlcv(start_date, end_date, ticker)
                df['ticker'] = ticker
                df['country'] = 'KR'
                df['currency'] = 'KRW'
                all_data.append(df)
            except Exception as e:
                continue
        
        full = pd.concat(all_data).reset_index()
        full = full.rename(columns={'날짜': 'date', '종가': 'close',
                                     '시가총액': 'market_cap',
                                     '거래량': 'volume'})
        
        # 분기말로 리샘플링
        full['date'] = pd.to_datetime(full['date'])
        full = full.set_index('date')
        
        quarterly = (full.groupby('ticker')
                          .resample('QE')
                          .last()
                          .reset_index())
        
        return quarterly[['ticker', 'country', 'currency', 
                          'date', 'close', 'volume', 'market_cap']]
```

### 4.3 `src/data_fetchers/prices_us.py`

```python
import yfinance as yf
import pandas as pd
from .base import BaseFetcher

class USPriceFetcher(BaseFetcher):
    
    def fetch(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """미국 종목 분기 종가."""
        all_data = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date,
                                progress=False, auto_adjust=True)
                if df.empty:
                    continue
                df = df.reset_index()
                df['ticker'] = ticker
                df['country'] = 'US'
                df['currency'] = 'USD'
                all_data.append(df)
            except Exception:
                continue
        
        full = pd.concat(all_data, ignore_index=True)
        full.columns = [c.lower() if isinstance(c, str) else c for c in full.columns]
        full = full.rename(columns={'date': 'date'})
        full['date'] = pd.to_datetime(full['date'])
        full = full.set_index('date')
        
        quarterly = (full.groupby('ticker')
                          .resample('QE')
                          .last()
                          .reset_index())
        
        return quarterly[['ticker', 'country', 'currency', 'date', 'close', 'volume']]
```

### 4.4 `src/data_fetchers/macro_us.py`

```python
from fredapi import Fred
import pandas as pd
import os
from .base import BaseFetcher

# config/indicator_list.yaml의 거시 80종 ID → FRED 시리즈 매핑
FRED_SERIES = {
    'M_INT_001': 'DFF',           # Fed Funds Rate
    'M_INT_002': 'DGS2',          # 2Y Treasury
    'M_INT_003': 'DGS10',         # 10Y Treasury
    'M_LIQ_002': 'M2SL',          # M2 Money Stock
    'M_INF_001': 'CPIAUCSL',      # CPI
    'M_INF_002': 'CPILFESL',      # Core CPI
    'M_ECO_004': 'NAPM',          # ISM Manufacturing PMI
    'M_ECO_008': 'UNRATE',        # Unemployment
    'M_SNT_001': 'VIXCLS',        # VIX
    # ... 80종 매핑
}

class FredMacroFetcher(BaseFetcher):
    
    def __init__(self, cache_dir, api_key=None):
        super().__init__(cache_dir)
        self.fred = Fred(api_key=api_key or os.getenv('FRED_API_KEY'))
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        all_series = {}
        for indicator_id, fred_id in FRED_SERIES.items():
            try:
                series = self.fred.get_series(fred_id, start_date, end_date)
                all_series[indicator_id] = series
            except Exception:
                continue
        
        df = pd.DataFrame(all_series)
        # 분기말 리샘플링 (마지막 값)
        df = df.resample('QE').last()
        df.index.name = 'date'
        return df.reset_index()
```

### 4.5 `src/data_fetchers/themes_naver.py`

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from .base import BaseFetcher

class NaverThemeFetcher(BaseFetcher):
    """네이버 증권 테마 → 종목 매핑."""
    
    BASE_URL = "https://finance.naver.com/sise/theme.naver"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X 14_0)'
    }
    
    def fetch(self) -> pd.DataFrame:
        """모든 테마와 소속 종목 수집."""
        theme_list = self._fetch_theme_list()
        
        rows = []
        for theme_name, theme_url in theme_list:
            stocks = self._fetch_theme_stocks(theme_url)
            for ticker in stocks:
                rows.append({
                    'ticker': ticker,
                    'theme': theme_name,
                    'source': 'naver',
                    'confidence': 0.9,
                })
            time.sleep(0.5)   # rate limit 존중
        
        return pd.DataFrame(rows)
    
    def _fetch_theme_list(self):
        themes = []
        for page in range(1, 10):
            r = requests.get(f"{self.BASE_URL}?&page={page}", headers=self.HEADERS)
            soup = BeautifulSoup(r.content, 'lxml')
            for row in soup.select('table.type_1 a.col_type1'):
                themes.append((row.text.strip(), row['href']))
        return themes
    
    def _fetch_theme_stocks(self, theme_url):
        r = requests.get(f"https://finance.naver.com{theme_url}", headers=self.HEADERS)
        soup = BeautifulSoup(r.content, 'lxml')
        tickers = []
        for link in soup.select('div.name_area a'):
            href = link.get('href', '')
            if 'code=' in href:
                tickers.append(href.split('code=')[-1])
        return tickers
```

---

## 5. 피처 엔지니어링

### 5.1 `src/features/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class IndicatorMeta:
    id: str
    name: str
    module: str          # MACRO, FUNDAMENTAL, KOREAN_ASSET, COMPUTED
    category: str
    unit: str
    frequency: str       # quarterly
    source: str
    countries: list
    lag_days: int = 45   # PiT lag

class BaseIndicator(ABC):
    meta: IndicatorMeta
    
    @abstractmethod
    def compute(self, raw_data: pd.DataFrame) -> pd.Series:
        ...
    
    def to_quarterly(self, series: pd.Series) -> pd.Series:
        """분기말 값으로 표준화."""
        return series.resample('QE').last()
```

### 5.2 `src/features/computed/f_score.py`

```python
from ..base import BaseIndicator, IndicatorMeta
import pandas as pd

class FScoreIndicator(BaseIndicator):
    meta = IndicatorMeta(
        id="C_FSCORE",
        name="Piotroski F-Score",
        module="COMPUTED",
        category="financial_health",
        unit="score",
        frequency="quarterly",
        source="derived",
        countries=["KR", "US"],
        lag_days=45,
    )
    
    def compute(self, fin: pd.DataFrame) -> pd.Series:
        """fin: 종목별 분기 재무제표 DataFrame."""
        scores = pd.DataFrame(index=fin.index)
        
        # 9개 항목
        scores['c1'] = (fin['roa'] > 0).astype(int)
        scores['c2'] = (fin['cfo'] > 0).astype(int)
        scores['c3'] = (fin['roa'] > fin['roa'].shift(4)).astype(int)
        scores['c4'] = (fin['cfo'] > fin['net_income']).astype(int)
        scores['c5'] = (fin['leverage'] < fin['leverage'].shift(4)).astype(int)
        scores['c6'] = (fin['current_ratio'] > fin['current_ratio'].shift(4)).astype(int)
        scores['c7'] = (fin['shares'] <= fin['shares'].shift(4)).astype(int)
        scores['c8'] = (fin['gross_margin'] > fin['gross_margin'].shift(4)).astype(int)
        scores['c9'] = (fin['asset_turnover'] > fin['asset_turnover'].shift(4)).astype(int)
        
        return scores.sum(axis=1)
```

### 5.3 `src/features/computed/quality_score.py`

```python
from ..base import BaseIndicator, IndicatorMeta
import pandas as pd
import numpy as np

class QualityScoreIndicator(BaseIndicator):
    meta = IndicatorMeta(
        id="C_QUALITY",
        name="Quality Score (Asness-style)",
        module="COMPUTED",
        category="quality",
        unit="z_score",
        frequency="quarterly",
        source="derived",
        countries=["KR", "US"],
        lag_days=45,
    )
    
    def compute(self, fin: pd.DataFrame) -> pd.Series:
        # Profitability + Growth + Safety + Payout 4개 z-score 합
        prof = self._zscore(fin['gross_profit'] / fin['total_assets'])
        growth = self._zscore(fin['eps'].pct_change(4))    # YoY
        safety = -self._zscore(fin['debt'] / fin['equity'])
        payout = self._zscore(fin['dividends'] / fin['net_income'])
        
        return prof + growth + safety + payout
    
    @staticmethod
    def _zscore(s):
        # 동일 시점 횡단면 z-score
        return (s - s.mean()) / (s.std() + 1e-8)
```

### 5.4 자동 등록 시스템

```python
# src/features/registry.py
import importlib
import pkgutil
from .base import BaseIndicator

def discover_indicators(package_name: str = 'src.features'):
    """모든 BaseIndicator 자손을 자동 탐지."""
    indicators = {}
    package = importlib.import_module(package_name)
    
    for finder, name, ispkg in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        try:
            mod = importlib.import_module(name)
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if (isinstance(cls, type) and 
                    issubclass(cls, BaseIndicator) and 
                    cls is not BaseIndicator):
                    indicators[cls.meta.id] = cls
        except Exception:
            continue
    
    return indicators
```

---

## 6. 라벨 생성

### 6.1 `src/labels/attractiveness.py`

```python
import pandas as pd
import numpy as np

def compute_attractiveness(
    prices_quarterly: pd.DataFrame,
    max_horizon_years: int = 5,
    min_forward_quarters: int = 4,
) -> pd.DataFrame:
    """매력도 라벨 (단일 값)."""
    max_quarters = max_horizon_years * 4
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
            
            A = np.log(max_price / p_t) / log_base_value
            
            results.append({
                'ticker': ticker,
                'date': dates[i],
                'A': np.float32(A),
                'A_quarters_used': np.int8(len(forward)),
            })
    
    return pd.DataFrame(results)
```

### 6.2 `src/labels/risk.py`

```python
import pandas as pd
import numpy as np

def compute_risk(
    prices_quarterly: pd.DataFrame,
    max_horizon_years: int = 5,
    min_forward_quarters: int = 4,
) -> pd.DataFrame:
    """위험도 라벨 (단일 값)."""
    max_quarters = max_horizon_years * 4
    prices_quarterly = prices_quarterly.sort_values(['ticker', 'date'])
    results = []
    
    for ticker, group in prices_quarterly.groupby('ticker', observed=True, sort=False):
        closes = group['close'].to_numpy(dtype=np.float32)
        dates = group['date'].to_numpy()
        log_rets = np.log(closes[1:] / closes[:-1])
        n = len(closes)
        
        for i in range(n):
            ret_start = i
            ret_end = min(i + max_quarters, n - 1)
            forward_rets = log_rets[ret_start : ret_end]
            valid = forward_rets[~np.isnan(forward_rets)]
            
            if len(valid) < min_forward_quarters:
                continue
            
            R = valid.std(ddof=1) * np.sqrt(4)
            
            results.append({
                'ticker': ticker,
                'date': dates[i],
                'R': np.float32(R),
                'R_quarters_used': np.int8(len(valid)),
            })
    
    return pd.DataFrame(results)
```

### 6.3 라벨 통합 스크립트

```python
# scripts/03_build_labels.py
import pandas as pd
from src.labels.attractiveness import compute_attractiveness
from src.labels.risk import compute_risk
from src.utils.io import save_parquet, load_parquet, report_memory

def main():
    prices = load_parquet('data/processed/prices_quarterly.parquet')
    report_memory(prices, "prices")
    
    a = compute_attractiveness(prices, max_horizon_years=5)
    r = compute_risk(prices, max_horizon_years=5)
    
    labels = a.merge(r, on=['ticker', 'date'], how='outer')
    
    save_parquet(labels, 'data/processed/labels.parquet')
    report_memory(labels, "labels")
    print(f"Generated {len(labels)} label rows")

if __name__ == '__main__':
    main()
```
## 7. TFT 모델 (M1 Mac 최적화)

### 7.1 `src/models/tft_model.py`

```python
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE, MultiLoss
from pytorch_forecasting.data import GroupNormalizer
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.device import get_device


class TFTStockModel:
    """TFT 래퍼 — M1 Mac 호환."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()
        self.model = None
        self.dataset = None
    
    def prepare_dataset(self, df: pd.DataFrame, 
                        targets: list = ['A', 'R']):
        """
        df: 종목·분기별 데이터 (피처 + 라벨)
        required columns:
          - ticker (group_ids)
          - quarter_idx (time_idx, 정수)
          - country, sector, ... (static)
          - all features (time_varying_unknown)
          - target columns
        """
        # quarter index를 정수로
        df = df.copy()
        df['quarter_idx'] = (df['date'].dt.year * 4 + df['date'].dt.quarter).astype(int)
        df['quarter_idx'] -= df['quarter_idx'].min()
        
        # NaN 라벨이 있는 행은 제거하지 않고 학습 시 mask
        # 단 모든 라벨이 NaN인 행은 제거
        df = df.dropna(subset=targets, how='all')
        
        # static categoricals
        static_cats = ['country', 'sector', 'size_tier']
        for col in static_cats:
            df[col] = df[col].astype(str).fillna('unknown')
        
        # time-varying unknowns: 모든 거시·재무·계산 피처
        feature_cols = [c for c in df.columns 
                        if c.startswith(('M_', 'F_', 'A_RE_', 'A_EQ_', 'A_BD_', 'A_ALT_', 'C_'))]
        
        max_encoder = 20   # 과거 20분기 = 5년
        max_pred = 1
        
        # train cutoff
        training_cutoff = df['quarter_idx'].max() - max_pred
        
        self.dataset = TimeSeriesDataSet(
            df[df['quarter_idx'] <= training_cutoff],
            time_idx="quarter_idx",
            target=targets,
            group_ids=["ticker"],
            min_encoder_length=8,    # 최소 2년 과거
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_pred,
            
            static_categoricals=static_cats,
            static_reals=[],
            
            time_varying_known_reals=["quarter_idx"],
            time_varying_unknown_reals=feature_cols,
            
            target_normalizer=GroupNormalizer(
                groups=["ticker"],
                transformation="softplus"   # 음수 매력도 처리
            ),
            
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        return self.dataset
    
    def build_model(self):
        """TFT 모델 인스턴스화. M1 MPS 호환."""
        assert self.dataset is not None, "Call prepare_dataset first"
        
        n_targets = len(self.dataset.target_names)
        
        # MultiLoss for multi-target
        loss = MultiLoss(
            [QuantileLoss(quantiles=[0.1, 0.5, 0.9])] * 2
        )
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            hidden_size=self.config.get('hidden_size', 64),
            attention_head_size=self.config.get('attention_head_size', 4),
            dropout=self.config.get('dropout', 0.2),
            hidden_continuous_size=self.config.get('hidden_continuous_size', 32),
            loss=loss,
            learning_rate=self.config.get('learning_rate', 0.001),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        return self.model
    
    def train(self, train_loader, val_loader, max_epochs: int = 50):
        """학습 실행. M1 MPS 우선, 문제 시 CPU fallback."""
        
        # M1 MPS는 일부 op 미지원 가능. fallback 자동 처리
        accelerator = 'mps' if self.device.type == 'mps' else 'cpu'
        
        # MultiLoss + MPS 조합 문제 시 CPU로 강제
        try:
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor='val_loss', patience=8, mode='min'),
                    pl.callbacks.ModelCheckpoint(
                        dirpath='./checkpoints',
                        filename='tft-{epoch:02d}-{val_loss:.4f}',
                        save_top_k=3,
                        monitor='val_loss',
                    ),
                ],
                enable_progress_bar=True,
                log_every_n_steps=20,
            )
            trainer.fit(self.model, train_loader, val_loader)
        except RuntimeError as e:
            if 'MPS' in str(e):
                print(f"[MPS Error] Falling back to CPU: {e}")
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator='cpu',
                    devices=1,
                    gradient_clip_val=0.1,
                )
                trainer.fit(self.model, train_loader, val_loader)
            else:
                raise
        
        return trainer
    
    def predict(self, dataloader):
        """추론 — 분위(P10, P50, P90) 반환."""
        predictions = self.model.predict(
            dataloader, 
            mode='quantiles',
            return_x=True,
        )
        return predictions
```

### 7.2 학습 진입점

```python
# scripts/04_train_tft.py
import pandas as pd
import yaml
from pathlib import Path
import torch

from src.models.tft_model import TFTStockModel
from src.utils.device import get_device

def main():
    with open('config/settings.yaml') as f:
        config = yaml.safe_load(f)
    
    # 피처 + 라벨 통합 데이터
    features = pd.read_parquet('data/processed/features.parquet')
    labels = pd.read_parquet('data/processed/labels.parquet')
    
    data = features.merge(labels, on=['ticker', 'date'], how='inner')
    
    print(f"Total rows: {len(data)}")
    print(f"Tickers: {data['ticker'].nunique()}")
    print(f"Date range: {data['date'].min()} ~ {data['date'].max()}")
    print(f"Device: {get_device()}")
    
    # 학습 시작
    model = TFTStockModel(config['model'])
    dataset = model.prepare_dataset(
        data,
        targets=['A', 'R']
    )
    
    # validation split
    val_cutoff = pd.Timestamp(config['train_split']['train_end'])
    train_data = data[data['date'] <= val_cutoff]
    val_data = data[(data['date'] > val_cutoff) & 
                    (data['date'] <= pd.Timestamp(config['train_split']['val_end']))]
    
    train_loader = dataset.to_dataloader(
        train=True, 
        batch_size=config['model']['batch_size'], 
        num_workers=config['model'].get('num_workers', 2),
        persistent_workers=config['model'].get('persistent_workers', True),
        pin_memory=False
    )
    val_dataset = type(dataset).from_dataset(dataset, val_data, predict=False, stop_randomization=True)
    val_loader = val_dataset.to_dataloader(
        train=False, 
        batch_size=config['model']['batch_size'], 
        num_workers=config['model'].get('num_workers', 2),
        persistent_workers=config['model'].get('persistent_workers', True),
        pin_memory=False
    )
    
    model.build_model()
    trainer = model.train(train_loader, val_loader, 
                         max_epochs=config['model']['max_epochs'])
    
    print("Training complete!")

if __name__ == '__main__':
    main()
```

### 7.3 M1 Mac 추가 팁

```python
# 학습 시작 전 환경 점검 코드
def check_m1_environment():
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # 메모리 체크 (활성 메모리 확인)
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM total: {mem.total / 1e9:.1f}GB, available: {mem.available / 1e9:.1f}GB")
    
    if mem.total < 16e9:
        print("[WARNING] 8GB M1 Mac: batch_size 16~32 권장, num_workers=0 필수")
```

---

## 8. 베이스라인 모델

### 8.1 `src/models/baseline_accounting.py`

```python
import pandas as pd
import numpy as np

class AccountingBaseline:
    """학술 합성 지표 baseline."""
    
    def __init__(self, baseline_type: str):
        """baseline_type: 'fscore', 'quality', 'gp_a', 'composite', 'equal_1n'"""
        self.type = baseline_type
    
    def score(self, features: pd.DataFrame) -> pd.Series:
        """단순 공식으로 매력도 점수 산출 — 학습 없음."""
        if self.type == 'fscore':
            return features['C_FSCORE']
        
        elif self.type == 'quality':
            return features['C_QUALITY']
        
        elif self.type == 'gp_a':
            return self._zscore(features['gross_profit'] / features['total_assets'])
        
        elif self.type == 'composite':
            return (
                self._zscore(features['C_QUALITY']) +
                self._zscore(features['C_FSCORE']) +
                self._zscore(features['gross_profit'] / features['total_assets']) +
                -self._zscore(features['F_VAL_001'])   # PER (역방향)
            ) / 4
        
        elif self.type == 'equal_1n':
            cols = [c for c in features.columns if c.startswith(('C_', 'F_VAL', 'F_PRF'))]
            return features[cols].apply(self._zscore).mean(axis=1)
        
        else:
            raise ValueError(self.type)
    
    @staticmethod
    def _zscore(s):
        return (s - s.mean()) / (s.std() + 1e-8)
```

### 8.2 LightGBM 베이스라인

```python
# src/models/baseline_gbm.py
import lightgbm as lgb
import pandas as pd
import numpy as np

class GBMBaseline:
    
    def __init__(self, target: str):
        self.target = target
        self.model = None
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

---

## 9. 평가 및 베이스라인 비교

### 9.1 `src/evaluation/metrics.py`

```python
import numpy as np
import pandas as pd
from scipy import stats

def spearman_ic(predicted, realized):
    """Spearman rank IC (NaN handling)."""
    mask = ~(np.isnan(predicted) | np.isnan(realized))
    if mask.sum() < 30:
        return np.nan
    return stats.spearmanr(predicted[mask], realized[mask])[0]

def icir(ic_series):
    """IC Information Ratio."""
    ic_series = pd.Series(ic_series).dropna()
    if len(ic_series) < 4:
        return np.nan
    return ic_series.mean() / ic_series.std()

def decile_returns(predictions, realized, n_deciles=10):
    """분위별 실현값 평균."""
    df = pd.DataFrame({'pred': predictions, 'real': realized}).dropna()
    df['decile'] = pd.qcut(df['pred'].rank(method='first'), n_deciles, 
                            labels=False, duplicates='drop') + 1
    return df.groupby('decile')['real'].mean()

def long_short_spread(predictions, realized, n_deciles=10):
    """D10 - D1 스프레드."""
    deciles = decile_returns(predictions, realized, n_deciles)
    return deciles.iloc[-1] - deciles.iloc[0]

def diebold_mariano(errors1, errors2, h=1):
    """두 모델 예측 오차의 통계적 유의성 비교."""
    d = errors1**2 - errors2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
```

### 9.2 베이스라인 비교 스크립트

```python
# scripts/06_evaluate.py
import pandas as pd
import numpy as np
from src.evaluation.metrics import *

def evaluate_all_models(test_data, predictions_dict, target='A'):
    """
    predictions_dict: {model_name: predictions_array}
    """
    realized = test_data[target].values
    
    results = []
    for model_name, preds in predictions_dict.items():
        ic = spearman_ic(preds, realized)
        ls = long_short_spread(preds, realized)
        rmse = np.sqrt(np.nanmean((preds - realized) ** 2))
        
        results.append({
            'model': model_name,
            'target': target,
            'IC': ic,
            'L-S Spread': ls,
            'RMSE': rmse,
        })
    
    return pd.DataFrame(results).sort_values('IC', ascending=False)
```

---

## 10. 출력 UI (Streamlit)

### 10.1 `src/ui/streamlit_app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Attractiveness & Risk", layout="wide")

st.title("📊 종목별 매력도 · 위험도 분석")
st.caption("두 지표는 독립적 추정치입니다. 본인 위험선호로 직접 판단하세요.")

# 데이터 로드
@st.cache_data
def load_predictions():
    return pd.read_parquet('data/processed/predictions_latest.parquet')

df = load_predictions()

# 사이드바 필터
with st.sidebar:
    st.header("필터")
    country = st.multiselect("국가", df['country'].unique(), default=df['country'].unique())
    sectors = st.multiselect("섹터", df['sector'].unique(), default=df['sector'].unique())
    themes = st.multiselect("테마 (네이버)", df['themes'].explode().unique())

# 필터 적용
filtered = df[df['country'].isin(country) & df['sector'].isin(sectors)]
if themes:
    filtered = filtered[filtered['themes'].apply(lambda t: any(x in t for x in themes))]

A_col = 'A'
R_col = 'R'

# 2D 분포 (매력도 vs 위험도)
st.subheader("매력도 · 위험도 2D 분포")
fig = px.scatter(
    filtered,
    x=R_col, y=A_col,
    color='sector',
    hover_data=['ticker', 'name', 'country'],
    title="최대 5년 전망: 매력도 vs 위험도",
)
fig.update_layout(
    xaxis_title="위험도 (연환산 변동성)",
    yaxis_title="매력도 (단일값)",
)
st.plotly_chart(fig, use_container_width=True)

# 종목별 상세
st.subheader("종목 상세")
selected = st.selectbox("종목 선택", filtered['ticker'].unique())
stock = filtered[filtered['ticker'] == selected].iloc[0]

col1, col2 = st.columns(2)
with col1:
    A_value = stock['A']
    multiple = 5 ** A_value
    st.metric("매력도 (A)", f"{A_value:.2f}")
    st.caption(f"의미: 향후 최대 5년 내 약 {multiple:.1f}배 도달 가능성")

with col2:
    R_value = stock['R']
    st.metric("위험도 (R)", f"{R_value:.1%}")
    st.caption(f"연환산 변동성 {R_value:.1%}")

# 베이스라인 비교
st.subheader("학술 베이스라인 점수")
baseline_cols = ['C_FSCORE', 'C_QUALITY']
st.dataframe(stock[baseline_cols].to_frame().T)
```

---

## 11. 실행 순서

```bash
# 1. 환경 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API 키 설정
cp .env.example .env
# .env 편집: FRED_API_KEY, ECOS_API_KEY, DART_API_KEY (선택)

# 3. 데이터 수집
python scripts/01_fetch_data.py --countries KR US --start 2010-01-01 --end 2026-05-01

# 4. 피처 빌드
python scripts/02_build_features.py

# 5. 라벨 생성
python scripts/03_build_labels.py

# 6. TFT 학습 (M1 MPS 자동 감지)
python scripts/04_train_tft.py

# 7. 베이스라인 학습 및 비교
python scripts/05_train_baselines.py

# 8. 평가
python scripts/06_evaluate.py

# 9. UI 실행
streamlit run scripts/07_run_ui.py
```

---

## 12. 핵심 검증 체크포인트

각 단계 완료 후 반드시 확인:

```python
# tests/test_pipeline.py
def test_quarterly_resampling():
    """일별 → 분기 종가 변환 정확성."""
    ...

def test_attractiveness_label():
    """A_5Y 계산이 max-window 기반인지."""
    sample_prices = [100, 110, 105, 120, 130, 125]   # 5분기 forward
    p_t = 100
    # max = 130 → log_5(130/100) = log_5(1.3)
    expected = np.log(1.3) / np.log(5)
    assert abs(compute_a(sample_prices, p_t, N=5) - expected) < 1e-6

def test_pit_lag():
    """Point-in-Time lag이 정확히 적용되는지."""
    ...

def test_m1_mps_fallback():
    """MPS 미지원 op에서 CPU fallback 동작."""
    ...

def test_no_lookahead_in_features():
    """t 시점 피처에 t+ 정보가 들어가지 않음."""
    ...
```

---

## 13. 트러블슈팅

### M1 Mac 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `RuntimeError: MPS backend out of memory` | 메모리 부족 | batch_size 절반으로, num_workers=0 |
| `aten::xxx not implemented for MPS` | MPS op 미지원 | `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| 학습이 CPU보다 느림 | 작은 배치로 GPU overhead 큼 | batch_size 늘리거나 CPU 사용 |
| `pytorch-forecasting` 설치 실패 | 일부 의존성 wheel 없음 | `pip install --no-binary :all:` 또는 conda |
| `dart-fss` 한국 재무 API 401 | 인증 키 누락 | DART API 키 등록 |

### 데이터 관련

| 증상 | 해결 |
|------|------|
| pykrx에서 일부 종목 실패 | try-except로 skip, 로깅 |
| yfinance Rate Limit | `time.sleep(0.5)` 추가 |
| 네이버 크롤링 차단 | User-Agent 변경, 요청 간격 늘림 |
| 라벨 NaN 비율 높음 | 윈도우 부족 종목 — 정상, 학습 시 자동 mask |

---

## 14. 데이터 양 예상 (분기별, M1 부담 점검)

```
KOSPI 종목: ~800
S&P 500 종목: ~500
합 유니버스: ~1,300

분기 데이터 (2010~2025, 60분기):
  1,300 × 60 = 78,000 행
  
피처 행 (각 행에 ~170 피처):
  78,000 × 170 × 4 bytes (float32) ≈ 53 MB
  → 매우 가벼움

라벨 행 (단일 A, R):
  78,000 × 2 × 4 bytes ≈ 0.6 MB

TFT 학습 메모리 추정 (M1 Pro 16GB):
  hidden_size=128, batch_size=64, encoder=20
  → 약 3-4 GB GPU 메모리 사용 (여유 있음)
```

---

## 15. 우선순위 (3일 구현 시)

```
Day 1 (최소 동작 시스템):
  ├── 1.x 환경 설정
  ├── 4.x 데이터 수집 (US만 yfinance, 100~200 종목 샘플)
  ├── 5.x 피처 (재무 + 거시만, 자산집중도/태그 후순위)
  ├── 6.x 라벨 (분기 max 매력도, 분기 vol 위험도)
  └── 검증: 라벨이 합리적 범위인지

Day 2 (학습):
  ├── 8.x LightGBM baseline 먼저 (debug용)
  ├── 7.x TFT 학습 (작은 hidden_size로 빠른 iter)
  └── 검증: validation loss 감소 확인

Day 3 (평가 및 UI):
  ├── 9.x 평가 메트릭
  ├── 베이스라인 비교
  ├── 10.x Streamlit UI
  └── 한국 종목 추가 (시간 여유 시)

후순위 (시간 부족 시 생략):
  - 네이버 테마 크롤링
  - 한국 자산집중도 35종
  - 거시 80종 풀세트 (10~20개로 축소 가능)
```

---

## 16. 부록 — `.env.example`

```bash
# API 키 (필수)
FRED_API_KEY=your_fred_key_here
ECOS_API_KEY=your_ecos_key_here     # 선택
DART_API_KEY=your_dart_key_here     # 선택 (한국 재무)

# 디바이스 강제 (선택, 자동 감지가 기본)
DEVICE=mps
PYTORCH_ENABLE_MPS_FALLBACK=1

# 데이터 경로
DATA_DIR=./data
CHECKPOINT_DIR=./checkpoints
```

---

*v5.1은 v5 기반 구조를 유지하면서 세 가지 사용자 결정사항(M1 Pro 최적화, 단일 매력도/위험도, KOSPI+S&P500 한정)만 패치한 버전이며, coding agent가 처음부터 끝까지 자율 구현할 수 있도록 작성된 기술 명세입니다. 모든 코드 스켈레톤은 import 가능한 형태로 제공되며, 실제 구현 시 각 함수의 세부 로직만 채우면 됩니다.*
