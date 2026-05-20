from pykrx import stock
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from .base import BaseFetcher
from src.utils.io import optimize_dtypes

class KoreanPriceFetcher(BaseFetcher):
    
    def fetch(self, start_date: str, end_date: str, limit: int = None) -> pd.DataFrame:
        """한국 KOSPI 종목 분기 종가 수집 (yfinance 벌크 쿼리 + 스태킹 + 로컬 리샘플링 최적화)."""
        cached = self.load("prices_kr_raw")
        if cached is not None:
            print("Loaded KR prices from cache.")
            return cached
            
        print("Fetching KOSPI tickers...")
        
        # 1. KOSPI 종목 리스트 수집 (pykrx API 활용 - 로그인 필요 없음)
        try:
            tickers = stock.get_market_ticker_list(market="KOSPI")
            print(f"Found {len(tickers)} KOSPI tickers from pykrx.")
        except Exception as e:
            print(f"Warning: Failed to get KOSPI tickers from pykrx ({e}). Using fallback ticker list.")
            # 극도로 기본적인 fallback 티커 (가장 거래가 많은 대형주 위주)
            tickers = ["005930", "000660", "035420", "035720", "051910", "005380", "006400", "000270", "000050", "000030"]
            
        if limit:
            tickers = tickers[:limit]
        
        # 2. yfinance (.KS 포맷) 벌크 쿼리 실행
        print(f"Downloading Korean prices in bulk for {len(tickers)} KOSPI tickers via yfinance...")
        fallback_tickers = [f"{t}.KS" for t in tickers]
        
        chunk_size = 100
        chunks = [fallback_tickers[i:i + chunk_size] for i in range(0, len(fallback_tickers), chunk_size)]
        
        all_yf_dfs = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"Downloading KR chunk {i+1}/{len(chunks)} ({len(chunk)} tickers)...")
                df = yf.download(chunk, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if df.empty:
                    continue
                
                if isinstance(df.columns, pd.MultiIndex):
                    df_stacked = df.stack(level=1, future_stack=True)
                    df_stacked.index.names = ['date', 'ticker']
                    df_stacked = df_stacked.reset_index()
                else:
                    ticker = chunk[0]
                    df_stacked = df.reset_index()
                    df_stacked['ticker'] = ticker
                    df_stacked.columns = [c.lower() for c in df_stacked.columns]
                    
                all_yf_dfs.append(df_stacked)
            except Exception as e:
                print(f"Error in KR yfinance chunk {i+1}: {e}")
                continue
                
        if not all_yf_dfs:
            print("Warning: No KR price data fetched from yfinance.")
            return pd.DataFrame()
            
        full_yf = pd.concat(all_yf_dfs, ignore_index=True)
        full_yf.columns = [c.lower() for c in full_yf.columns]
        
        full_yf['ticker'] = full_yf['ticker'].str.replace('.KS', '', regex=False)
        full_yf['country'] = 'KR'
        full_yf['currency'] = 'KRW'
        full_yf['market_cap'] = 0.0
        full_yf['date'] = pd.to_datetime(full_yf['date'])
        
        # 0값 및 결측치 필터링
        full_yf = full_yf.dropna(subset=['close'])
        full_yf = full_yf[full_yf['close'] > 0]
        
        # 3. 로컬 리샘플링을 통한 분기 캔들 병합 및 메타데이터 보존
        full_yf = full_yf.set_index('date')
        quarterly_yf = (full_yf.groupby('ticker', observed=True)
                        .resample('QE')
                        .agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum',
                            'market_cap': 'last',
                            'country': 'last',
                            'currency': 'last'
                        })
                        .drop(columns=['ticker'], errors='ignore')
                        .reset_index())
        
        quarterly_yf = optimize_dtypes(quarterly_yf)
        self.save(quarterly_yf, "prices_kr_raw")
        return quarterly_yf

