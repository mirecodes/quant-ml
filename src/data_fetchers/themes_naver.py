# src/data_fetchers/themes_naver.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from .base import BaseFetcher
from src.utils.io import optimize_dtypes

class NaverThemeFetcher(BaseFetcher):
    """네이버 증권 테마 → 종목 매핑."""
    
    BASE_URL = "https://finance.naver.com/sise/theme.naver"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    def fetch(self) -> pd.DataFrame:
        """모든 테마와 소속 종목 수집."""
        cached = self.load("themes_naver_raw")
        if cached is not None:
            print("Loaded Naver themes from cache.")
            return cached
            
        print("Fetching Naver Stock Themes...")
        try:
            theme_list = self._fetch_theme_list()
        except Exception as e:
            print(f"Error fetching theme list: {e}")
            return pd.DataFrame()
            
        print(f"Found {len(theme_list)} themes. Scraping stock mappings...")
        rows = []
        # 시간 단축을 위해 상위 10개 테마만 가져오거나 슬립 조정
        # 실사용에서는 전체를 돌리되, yfinance/pykrx처럼 오래 걸리지 않도록 sleep 0.1로 조정
        for theme_name, theme_url in theme_list[:20]: # 속도를 위해 주요 20개 테마만 매핑 수집 (메모리/시간 최적화)
            try:
                stocks = self._fetch_theme_stocks(theme_url)
                for ticker in stocks:
                    rows.append({
                        'ticker': ticker,
                        'theme': theme_name,
                        'source': 'naver',
                        'confidence': 0.9,
                    })
                time.sleep(0.1)  # Rate limit 존중하되 속도 최적화
            except Exception as e:
                continue
                
        if not rows:
            print("Warning: No Naver theme data fetched.")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        df = optimize_dtypes(df)
        
        self.save(df, "themes_naver_raw")
        return df
    
    def _fetch_theme_list(self):
        themes = []
        # 보통 1~7 페이지까지 존재
        for page in range(1, 4):  # 주요 3페이지까지만 수집 (시간/네트워크 최적화)
            r = requests.get(f"{self.BASE_URL}?&page={page}", headers=self.HEADERS)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, 'lxml')
            for row in soup.select('table.type_1 a.col_type1'):
                themes.append((row.text.strip(), row['href']))
        return themes
    
    def _fetch_theme_stocks(self, theme_url):
        r = requests.get(f"https://finance.naver.com{theme_url}", headers=self.HEADERS)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.content, 'lxml')
        tickers = []
        for link in soup.select('div.name_area a'):
            href = link.get('href', '')
            if 'code=' in href:
                tickers.append(href.split('code=')[-1])
        return tickers
