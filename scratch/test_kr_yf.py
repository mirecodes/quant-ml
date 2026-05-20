import yfinance as yf
import pandas as pd

def test_yf():
    tickers = ["000050.KS", "000030.KS"]
    start_date = "2010-01-01"
    end_date = "2026-05-01"
    
    print("Downloading via yfinance...")
    df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    print("Raw columns:", df.columns)
    
    if isinstance(df.columns, pd.MultiIndex):
        df_stacked = df.stack(level=1, future_stack=True)
        df_stacked.index.names = ['date', 'ticker']
        df_stacked = df_stacked.reset_index()
    else:
        # Single ticker fallback
        df_stacked = df.reset_index()
        df_stacked['ticker'] = tickers[0]
        df_stacked.columns = [c.lower() for c in df_stacked.columns]
        
    df_stacked.columns = [c.lower() for c in df_stacked.columns]
    print("\nStacked head:")
    print(df_stacked.head())
    
    df_stacked['ticker'] = df_stacked['ticker'].str.replace('.KS', '', regex=False)
    df_stacked['country'] = 'KR'
    df_stacked['currency'] = 'KRW'
    df_stacked['market_cap'] = 0.0
    df_stacked['date'] = pd.to_datetime(df_stacked['date'])
    
    df_stacked = df_stacked.dropna(subset=['close'])
    df_stacked = df_stacked[df_stacked['close'] > 0]
    
    # Resample to quarterly
    df_stacked = df_stacked.set_index('date')
    quarterly = (df_stacked.groupby('ticker', observed=True)
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
                 
    print("\nQuarterly resampled for 000050 (first 15 quarters):")
    print(quarterly[quarterly['ticker'] == '000050'].head(15))
    print(f"\nTotal quarters for 000050: {len(quarterly[quarterly['ticker'] == '000050'])}")
    
    # Check if there are any zeros or nulls in quarterly
    print("\nQuarterly null check:")
    print(quarterly.isnull().sum())
    print("\nQuarterly zero price check:")
    print((quarterly[['open', 'high', 'low', 'close']] == 0).sum())

if __name__ == '__main__':
    test_yf()
