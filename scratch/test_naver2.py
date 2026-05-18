import requests
from bs4 import BeautifulSoup
import lxml

URL = "https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=584"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

r = requests.get(URL, headers=HEADERS)
r.encoding = 'euc-kr'
soup = BeautifulSoup(r.text, 'lxml')

tickers = []
links = soup.select('div.name_area a')
print(f"Total links in div.name_area a: {len(links)}")
for link in links[:5]:
    href = link.get('href', '')
    if 'code=' in href:
        tickers.append(href.split('code=')[-1])
print(tickers)
