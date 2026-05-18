import requests
from bs4 import BeautifulSoup
import lxml

URL = "https://finance.naver.com/sise/theme.naver?&page=1"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

r = requests.get(URL, headers=HEADERS)
print(f"Status Code: {r.status_code}")
soup = BeautifulSoup(r.content, 'lxml')

themes = soup.select('table.type_1 a')
print(f"Total links in table.type_1: {len(themes)}")
for row in themes[:5]:
    print(row.text.strip(), row.get('href'))
    
# Let's also check for td.col_type1
themes_col = soup.select('td.col_type1 a')
print(f"Total links in td.col_type1: {len(themes_col)}")
for row in themes_col[:5]:
    print(row.text.strip(), row.get('href'))
