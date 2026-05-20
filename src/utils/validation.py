import pandas as pd
import numpy as np

def check_intermediate_gaps(df: pd.DataFrame, context_name: str = "Dataset"):
    """
    데이터 중간에 누락된 분기(Gap)나 가격/거래량이 0 또는 NaN인 결측치가 존재할 경우 
    터미널에 경고(Warning)를 출력하여 즉시 확인할 수 있도록 합니다.
    """
    print(f"\n=================== [Validation] Checking {context_name} for intermediate gaps & invalid values ===================")
    
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    
    issues_found = False
    
    # 1. NaN(결측치) 체크
    key_cols = ['open', 'high', 'low', 'close', 'volume']
    present_cols = [c for c in key_cols if c in df.columns]
    
    null_counts = df[present_cols].isnull().sum()
    if null_counts.sum() > 0:
        print("\n\033[1;31m[⚠️ WARNING] NaN (Missing) values detected in key columns:\033[0m")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  - Column '{col}': {count} NaN values")
                # Show sample tickers with NaNs
                sample_nulls = df[df[col].isnull()].head(5)
                print(f"    Sample tickers: {sample_nulls['ticker'].unique()[:5].tolist()}")
                print(f"    Sample dates: {[d.strftime('%Y-%m-%d') for d in sample_nulls['date'].head(5)]}")
        issues_found = True
        
    # 2. 가격이 0인 값 체크 (open, high, low, close)
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
    if price_cols:
        zero_prices = (df[price_cols] == 0).sum()
        if zero_prices.sum() > 0:
            print("\n\033[1;31m[⚠️ WARNING] Zero values (0.0) detected in price columns:\033[0m")
            for col, count in zero_prices.items():
                if count > 0:
                    print(f"  - Column '{col}': {count} zero values")
                    sample_zeros = df[df[col] == 0].head(5)
                    print(f"    Sample tickers: {sample_zeros['ticker'].unique()[:5].tolist()}")
            issues_found = True

    # 3. 거래량이 0인 값 체크 (경고성 알림)
    if 'volume' in df.columns:
        zero_volume_count = (df['volume'] == 0).sum()
        if zero_volume_count > 0:
            print(f"\n\033[1;33m[⚠️ NOTICE] Zero volume (0) detected in {zero_volume_count} rows:\033[0m")
            sample_zeros = df[df['volume'] == 0].head(5)
            print(f"    Sample tickers: {sample_zeros['ticker'].unique()[:5].tolist()}")
            print("    (Note: Volume of 0 can occur during trading suspensions or in preferred shares.)")

    # 4. 중간 분기 누락(Gaps) 정밀 검사
    # 각 종목별로 첫 날짜와 마지막 날짜 사이의 모든 분기말('QE') 날짜 리스트 생성 후 비교
    gaps = []
    
    for ticker, group in df.groupby('ticker', observed=True):
        if len(group) < 2:
            continue
        
        group = group.sort_values('date')
        first_date = group['date'].min()
        last_date = group['date'].max()
        
        # Expected quarters between first and last date
        expected_range = pd.date_range(start=first_date, end=last_date, freq='QE')
        actual_dates = set(group['date'])
        
        missing_quarters = [q for q in expected_range if q not in actual_dates]
        if missing_quarters:
            gaps.append({
                'ticker': ticker,
                'first_date': first_date.strftime('%Y-%m-%d'),
                'last_date': last_date.strftime('%Y-%m-%d'),
                'missing_count': len(missing_quarters),
                'missing_samples': [q.strftime('%Y-%m-%d') for q in missing_quarters[:5]]
            })
            
    if gaps:
        print(f"\n\033[1;31m[⚠️ WARNING] Intermediate temporal gaps detected inside active lifetimes (found {len(gaps)} tickers with gaps):\033[0m")
        # Show top 10 tickers with gaps
        for gap_info in gaps[:10]:
            print(f"  - Ticker '{gap_info['ticker']}': Active from {gap_info['first_date']} to {gap_info['last_date']}")
            print(f"    Missing {gap_info['missing_count']} quarters. Samples: {gap_info['missing_samples']}")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more tickers have missing quarters.")
        issues_found = True
        
    if not issues_found:
        print("\033[1;32m✔ Verification Successful: No intermediate gaps, zero prices, or NaNs detected!\033[0m")
        print("  Data is perfectly continuous, aligned, and valid for model training.")
    else:
        print("\n\033[1;33m[💡 Debug Tip] Gaps or NaNs suggest issue with raw fetching. Try running the fetcher with '--no-cache' or check provider logs.\033[0m")
        
    print("====================================================================================================")
