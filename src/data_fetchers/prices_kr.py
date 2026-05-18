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
        """한국 KOSPI 종목 분기 종가 수집 (pykrx + yfinance 이중 장애복구 최적화)."""
        cached = self.load("prices_kr_raw")
        if cached is not None:
            print("Loaded KR prices from cache.")
            return cached
            
        print("Fetching KOSPI quarterly prices...")
        
        # 1. KRX 로그인 환경 변수 검증 및 사용자 알림
        import os
        krx_id = os.getenv("KRX_ID")
        krx_pw = os.getenv("KRX_PW")
        
        if not (krx_id and krx_pw):
            print("\n" + "="*80)
            print("  Error: KRX_ID or KRX_PW is missing or not configured in .env.")
            print("  To fetch Korean stocks successfully, please register at https://data.krx.co.kr and set KRX_ID & KRX_PW in .env.")
            print("="*80 + "\n")
            raise RuntimeError("KRX credentials missing in .env")
        else:
            print(f"KRX credentials found (ID: {krx_id}). Initiating pykrx full market fetch...")
            pykrx_failed = False
            
            # KOSPI 종목 리스트 수집 시도
            try:
                tickers = stock.get_market_ticker_list(market="KOSPI")
            except Exception as e:
                print(f"Error: Failed to get KOSPI tickers from pykrx: {e}")
                raise
        
        if limit:
            tickers = tickers[:limit]
        
        # 2. pykrx 벌크 쿼리 시도 (계정이 등록되어 있을 때만 실행)
        all_data = []
        if not pykrx_failed:
            date_range = pd.date_range(start=start_date, end=end_date, freq='QE')
            print(f"Attempting pykrx bulk date queries for {len(date_range)} quarter-ends...")
            
            for q_end in tqdm(date_range):
                df_q = pd.DataFrame()
                for offset in range(5):
                    target_date = (q_end - timedelta(days=offset)).strftime("%Y%m%d")
                    try:
                        df_q = stock.get_market_ohlcv_by_ticker(target_date, market="KOSPI")
                        if not df_q.empty:
                            df_q = df_q.reset_index()
                            df_q['date'] = pd.to_datetime(q_end)
                            break
                    except Exception as e:
                        if "Expecting value" in str(e):
                            pykrx_failed = True
                        continue
                
                if pykrx_failed:
                    print("\n[pykrx login/query issue detected] Switching to yfinance fallback...")
                    break
                    
                if not df_q.empty:
                    df_q['country'] = 'KR'
                    df_q['currency'] = 'KRW'
                    df_q = df_q.rename(columns={
                        '티커': 'ticker',
                        '종가': 'close',
                        '거래량': 'volume',
                        '시가총액': 'market_cap'
                    })
                    df_q = df_q[['ticker', 'country', 'currency', 'date', 'close', 'volume', 'market_cap']]
                    all_data.append(df_q)
                    
        # 3. pykrx 실패/비로그인 시 yfinance (.KS 포맷) 폴백 동작
        if pykrx_failed or not all_data:
            print(f"Running yfinance fallback for {len(tickers)} Korean tickers...")
            fallback_tickers = [f"{t}.KS" for t in tickers]
            
            chunk_size = 50
            chunks = [fallback_tickers[i:i + chunk_size] for i in range(0, len(fallback_tickers), chunk_size)]
            
            all_yf_dfs = []
            for i, chunk in enumerate(chunks):
                try:
                    print(f"Downloading KR chunk {i+1}/{len(chunks)} via yfinance...")
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
                    
            if all_yf_dfs:
                full_yf = pd.concat(all_yf_dfs, ignore_index=True)
                full_yf.columns = [c.lower() for c in full_yf.columns]
                
                full_yf['ticker'] = full_yf['ticker'].str.replace('.KS', '', regex=False)
                full_yf['country'] = 'KR'
                full_yf['currency'] = 'KRW'
                full_yf['market_cap'] = 0.0
                full_yf['date'] = pd.to_datetime(full_yf['date'])
                
                full_yf = full_yf.set_index('date')
                quarterly_yf = (full_yf.groupby('ticker', observed=True)
                                .resample('QE')
                                .last()
                                .drop(columns=['ticker'])
                                .reset_index())
                
                quarterly_yf = optimize_dtypes(quarterly_yf)
                self.save(quarterly_yf, "prices_kr_raw")
                return quarterly_yf
                
        if not all_data:
            print("Warning: No KR price data fetched from either pykrx or yfinance.")
            return pd.DataFrame()
            
        full = pd.concat(all_data, ignore_index=True)
        full = optimize_dtypes(full)
        
        self.save(full, "prices_kr_raw")
        return full
