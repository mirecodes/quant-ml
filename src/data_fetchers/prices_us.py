# src/data_fetchers/prices_us.py
import yfinance as yf
import pandas as pd
from .base import BaseFetcher
from src.utils.io import optimize_dtypes

class USPriceFetcher(BaseFetcher):
    
    def fetch(self, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        """미국 종목 분기 종가 수집 (yf.download 벌크 쿼리 + 스태킹 최적화)."""
        cached = self.load("prices_us_raw")
        if cached is not None:
            print("Loaded US prices from cache.")
            return cached
            
        print(f"Downloading US prices in bulk for {len(tickers)} S&P 500 tickers...")
        
        # yfinance 벌크 다운로드 (멀티스레드 지원)
        # yfinance는 한 번에 너무 많은 티커를 넣으면 누락이 생길 수 있으므로, 100개씩 청크로 나누어 다운로드
        chunk_size = 100
        chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        
        all_dfs = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"Downloading chunk {i+1}/{len(chunks)} ({len(chunk)} tickers)...")
                # auto_adjust=True로 수정주가 반영
                df = yf.download(chunk, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if df.empty:
                    continue
                
                # MultiIndex 컬럼 처리 및 Stacking으로 Long Format 전환
                # Columns: (Attribute, Ticker) -> Index: (Date, Ticker) with Columns: Attribute
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiIndex인 경우 스택 수행
                    df_stacked = df.stack(level=1, future_stack=True)
                    df_stacked.index.names = ['date', 'ticker']
                    df_stacked = df_stacked.reset_index()
                else:
                    # 단일 티커인 경우 컬럼이 MultiIndex가 아님
                    ticker = chunk[0]
                    df_stacked = df.reset_index()
                    df_stacked['ticker'] = ticker
                    df_stacked.columns = [c.lower() for c in df_stacked.columns]
                
                all_dfs.append(df_stacked)
            except Exception as e:
                print(f"Error downloading chunk {i+1}: {e}")
                continue
                
        if not all_dfs:
            print("Warning: No US price data fetched.")
            return pd.DataFrame()
            
        full = pd.concat(all_dfs, ignore_index=True)
        
        # 컬럼명 통일 및 정제
        full.columns = [c.lower() for c in full.columns]
        full = full.rename(columns={'close': 'close', 'volume': 'volume'})
        
        # 필요한 컬럼만 추출 및 국가/통화 추가
        full['country'] = 'US'
        full['currency'] = 'USD'
        full['date'] = pd.to_datetime(full['date'])
        
        # 시가총액은 yfinance 일별 데이터에 없으므로 (필요하다면 분기 데이터 병합 시 처리), 0 또는 임시값으로 채워둠
        if 'market_cap' not in full.columns:
            full['market_cap'] = 0.0
            
        full = full[['ticker', 'country', 'currency', 'date', 'close', 'volume', 'market_cap']]
        
        # 분기말로 리샘플링
        full = full.set_index('date')
        quarterly = (full.groupby('ticker', observed=True)
                     .resample('QE')
                     .last()
                     .drop(columns=['ticker'])
                     .reset_index())
                     
        quarterly = optimize_dtypes(quarterly)
        
        self.save(quarterly, "prices_us_raw")
        return quarterly
