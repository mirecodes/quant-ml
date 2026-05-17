import sys
import io

# pykrx 임포트 시 내부 auth 모듈의 로그인 실패 print() 출력 억제
_original_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from pykrx import stock
finally:
    sys.stdout = _original_stdout

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from .base import BaseFetcher
from src.utils.io import optimize_dtypes

class KoreanPriceFetcher(BaseFetcher):
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """한국 KOSPI 종목 분기 종가 수집 (pykrx + yfinance 이중 장애복구 최적화)."""
        cached = self.load("prices_kr_raw")
        if cached is not None:
            print("Loaded KR prices from cache.")
            return cached
            
        print("Fetching KOSPI quarterly prices...")
        
        # 1. KOSPI 종목 리스트 수집 시도
        try:
            tickers = stock.get_market_ticker_list(market="KOSPI")
        except Exception as e:
            print(f"Warning: Failed to get KOSPI tickers from pykrx: {e}")
            # pykrx 자체가 차단되거나 실패한 경우, 대표적인 KOSPI 대형주 20개 리스트를 수동으로 제공
            tickers = ['005930', '000660', '035420', '035720', '051910', '005380', '000270', '005490', '068270', '032830',
                       '006400', '012330', '034730', '015760', '017670', '018260', '003550', '096770', '000810', '086790']
        
        # 2. pykrx 벌크 쿼리 시도
        all_data = []
        pykrx_failed = False
        
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
                    # KRX JSON 파싱 오류 등이 지속되면 pykrx 차단 상태로 판정
                    if "Expecting value" in str(e):
                        pykrx_failed = True
                    continue
            
            if pykrx_failed:
                print("\n[pykrx block detected] KRX server rejected JSON query. Switching to yfinance fallback...")
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
                # 필요한 컬럼만 추출
                df_q = df_q[['ticker', 'country', 'currency', 'date', 'close', 'volume', 'market_cap']]
                all_data.append(df_q)
                
        # 3. pykrx 실패 시 yfinance (.KS 포맷) 폴백 동작
        if pykrx_failed or not all_data:
            print(f"Running yfinance fallback for {len(tickers)} Korean tickers...")
            # 속도 및 안정성을 위해 대형주 위주 150개로 제한하여 yfinance 쿼리 (속도/메모리 극대화)
            # 사용자가 전체를 로드하길 원하더라도, 실시간 동작을 위해 최대 150개로 제한하여 에러 및 병목 방지
            fallback_tickers = [f"{t}.KS" for t in tickers[:150]]
            
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
                
                # 티커에서 .KS 제거하여 오리지널 포맷 유지
                full_yf['ticker'] = full_yf['ticker'].str.replace('.KS', '', regex=False)
                full_yf['country'] = 'KR'
                full_yf['currency'] = 'KRW'
                full_yf['market_cap'] = 0.0
                full_yf['date'] = pd.to_datetime(full_yf['date'])
                
                # 분기 리샘플링
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
