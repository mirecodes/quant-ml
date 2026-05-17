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
            print("  [알림] KRX 로그인 계정 정보(KRX_ID, KRX_PW)가 .env 파일에 설정되지 않았습니다.")
            print("  ")
            print("  2026년 2월 한국거래소 포털의 보안/회원제 접근 정책 변경으로 인해,")
            print("  pykrx 라이브러리를 통해 한국 주식 데이터를 100% 온전히 수집하려면 계정이 필수적입니다.")
            print("  ")
            print("  [조치 방법]")
            print("  1. https://data.krx.co.kr 에 접속하여 무료 회원가입을 진행해 주세요.")
            print("  2. 프로젝트 루트의 `.env` 파일에 아래와 같이 계정을 입력해 주세요:")
            print("     KRX_ID=본인_아이디")
            print("     KRX_PW=본인_패스워드")
            print("  ")
            print("  * 현재는 계정이 설정되지 않아 yfinance 및 대형주 대표 110개 종목에 대한")
            print("    제한적 다운그레이드(장애복구) 모드로 동작합니다.")
            print("="*80 + "\n")
            
            # yfinance 폴백 즉시 실행을 위해 pykrx 실패 판정 유도
            pykrx_failed = True
            tickers = [
                '005930', '000660', '207940', '005380', '068270', '000270', '051910', '005490', '035420', '006400',
                '035720', '105560', '055550', '012330', '028260', '066570', '015760', '003550', '032830', '086790',
                '096770', '000810', '033780', '003490', '017670', '009150', '010140', '003670', '011200', '018260',
                '323410', '034730', '000720', '000100', '009540', '024110', '377300', '259960', '138040', '004020',
                '001040', '010950', '002790', '011170', '028050', '097950', '030200', '078930', '008770', '010130',
                '088350', '000080', '005830', '005940', '032640', '009830', '071050', '161390', '000120', '267250',
                '302440', '004990', '009900', '011780', '014680', '128940', '271560', '011070', '180640', '282330',
                '000150', '016360', '004370', '021240', '007070', '001450', '034220', '052690', '036570', '006360',
                '001800', '086280', '002380', '001230', '005180', '000880', '006800', '012750', '023530', '079550',
                '272210', '004800', '298020', '298050', '036460', '010060', '011790', '051900', '000240', '004170',
                '011210', '035250', '069960', '073240', '090430', '192820', '285130', '336260', '352820', '383220'
            ]
        else:
            print(f"KRX credentials found (ID: {krx_id}). Initiating pykrx full market fetch...")
            pykrx_failed = False
            
            # KOSPI 종목 리스트 수집 시도
            try:
                tickers = stock.get_market_ticker_list(market="KOSPI")
            except Exception as e:
                print(f"Warning: Failed to get KOSPI tickers from pykrx: {e}")
                # 대표주 110여 개 리스트 제공
                tickers = [
                    '005930', '000660', '207940', '005380', '068270', '000270', '051910', '005490', '035420', '006400',
                    '035720', '105560', '055550', '012330', '028260', '066570', '015760', '003550', '032830', '086790',
                    '096770', '000810', '033780', '003490', '017670', '009150', '010140', '003670', '011200', '018260',
                    '323410', '034730', '000720', '000100', '009540', '024110', '377300', '259960', '138040', '004020',
                    '001040', '010950', '002790', '011170', '028050', '097950', '030200', '078930', '008770', '010130',
                    '088350', '000080', '005830', '005940', '032640', '009830', '071050', '161390', '000120', '267250',
                    '302440', '004990', '009900', '011780', '014680', '128940', '271560', '011070', '180640', '282330',
                    '000150', '016360', '004370', '021240', '007070', '001450', '034220', '052690', '036570', '006360',
                    '001800', '086280', '002380', '001230', '005180', '000880', '006800', '012750', '023530', '079550',
                    '272210', '004800', '298020', '298050', '036460', '010060', '011790', '051900', '000240', '004170',
                    '011210', '035250', '069960', '073240', '090430', '192820', '285130', '336260', '352820', '383220'
                ]
        
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
