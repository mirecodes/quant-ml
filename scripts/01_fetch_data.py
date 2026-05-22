# scripts/01_fetch_data.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import ssl
import urllib.request

# macOS SSL Certificate verify error 근본적 해결
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
load_dotenv()

from src.data_fetchers.prices_kr import KoreanPriceFetcher
from src.data_fetchers.prices_us import USPriceFetcher
from src.data_fetchers.macro_us import FredMacroFetcher
from src.utils.io import save_parquet, report_memory

def get_sp500_tickers() -> list:
    """Wikipedia에서 현재 S&P 500 구성종목 목록 (User-Agent 헤더 추가하여 403 방지)."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        with urllib.request.urlopen(req) as response:
            html = response.read()
        tables = pd.read_html(html)
        sp500 = tables[0]
        tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        raise

def main(args):
    # 캐시 무시 처리
    if args.no_cache:
        print("Bypassing cache. Deleting local cache files...")
        cache_paths = [
            Path('data/raw/prices/prices_kr_raw.parquet'),
            Path('data/raw/prices/prices_us_raw.parquet'),
            Path('data/raw/prices/prices_kr_daily_raw.parquet'),
            Path('data/raw/prices/prices_us_daily_raw.parquet'),
            Path('data/raw/prices/prices_quarterly.parquet'),
            Path('data/raw/prices/themes_naver_raw.parquet'),
            Path('data/processed/themes_naver.parquet')
        ]
        for p in cache_paths:
            if p.exists():
                p.unlink()

    # KR: KOSPI만 벌크 수집
    print("=== Step 1: Fetching Korean Prices ===")
    kr_fetcher = KoreanPriceFetcher(cache_dir='data/raw/prices')
    kr_prices = kr_fetcher.fetch(args.start, args.end, limit=args.limit)
    if not kr_prices.empty:
        report_memory(kr_prices, "KR prices")
        
    # KR 일별 OHLCV 수집 (위험도 라벨 계산용)
    print("=== Step 1b: Fetching Korean Daily Prices ===")
    kr_daily = kr_fetcher.fetch_daily(args.start, args.end, limit=args.limit)
    if not kr_daily.empty:
        save_parquet(kr_daily, 'data/processed/prices_daily_kr.parquet')
        report_memory(kr_daily, "KR daily prices")
    
    # US: S&P 500만 벌크 수집
    print("\n=== Step 2: Fetching US Prices ===")
    us_tickers = get_sp500_tickers()
    if args.limit:
        us_tickers = us_tickers[:args.limit]
    us_fetcher = USPriceFetcher(cache_dir='data/raw/prices')
    us_prices = us_fetcher.fetch(us_tickers, args.start, args.end)
    if not us_prices.empty:
        report_memory(us_prices, "US prices")
        
    # US 일별 OHLCV 수집 (위험도 라벨 계산용)
    print("=== Step 2b: Fetching US Daily Prices ===")
    us_daily = us_fetcher.fetch_daily(us_tickers, args.start, args.end)
    if not us_daily.empty:
        save_parquet(us_daily, 'data/processed/prices_daily_us.parquet')
        report_memory(us_daily, "US daily prices")
    
    # 통합 저장
    print("\n=== Step 3: Merging & Saving Price Data ===")
    merged_prices = pd.DataFrame()
    if not kr_prices.empty and not us_prices.empty:
        merged_prices = pd.concat([kr_prices, us_prices], ignore_index=True)
    elif not kr_prices.empty:
        merged_prices = kr_prices
    elif not us_prices.empty:
        merged_prices = us_prices
        
    if not merged_prices.empty:
        save_parquet(merged_prices, 'data/processed/prices_quarterly.parquet')
        report_memory(merged_prices, "prices_quarterly.parquet")
        
        # 중간 결측치 및 분기 누락 정밀 검증 경고 출력
        from src.utils.validation import check_intermediate_gaps
        check_intermediate_gaps(merged_prices, "Merged Raw Prices")
    else:
        print("Warning: No price data merged.")
    
    # 거시 (미국 FRED)
    print("\n=== Step 4: Fetching Macro Indicators ===")
    try:
        fred = FredMacroFetcher(cache_dir='data/raw/macro')
        macro_us = fred.fetch(args.start, args.end)
        if not macro_us.empty:
            save_parquet(macro_us, 'data/processed/macro_us.parquet')
            report_memory(macro_us, "macro_us.parquet")
    except Exception as e:
        print(f"Error fetching US Macro data: {e}")
        raise
        
    # KR Themes: 글로벌 테마 매핑으로 대체되어 네이버 테마 크롤러는 비활성화됩니다.
    print("\n=== Step 5: (Skipped) Global Theme Mapping Active ===")
    
    print("\nData fetch complete.")

if __name__ == '__main__':
    import yaml
    try:
        with open('config/settings.yaml') as f:
            config = yaml.safe_load(f)
        default_start = config.get('prices', {}).get('start_date', '2010-01-01')
        default_end = config.get('prices', {}).get('end_date', '2026-05-01')
    except Exception:
        default_start = '2010-01-01'
        default_end = '2026-05-01'

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=default_start)
    parser.add_argument('--end', default=default_end)
    parser.add_argument('--no-cache', action='store_true', help='Ignore cached data and fetch fresh data')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of tickers to fetch per market')
    args = parser.parse_args()
    main(args)
