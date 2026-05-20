"""
src/theme/loader.py

merged_themes.yaml 파일로부터 종목 및 테마 매핑을 읽는다.
"""
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List


@lru_cache(maxsize=1)
def load_theme_mapping(merged_path: str = "themes/processed/merged_themes.yaml") -> Dict:
    """
    병합된 테마 파일을 로드하여 두 방향 인덱스와 메타데이터를 반환한다.

    반환:
        {
          "ticker_to_themes": {"005930": ["GT_SEMI_MEMORY", ...], ...},
          "theme_to_tickers": {"GT_SEMI_MEMORY": ["005930", "000660", ...], ...},
          "theme_meta":       {"GT_SEMI_MEMORY": {"tier":3, "parent":..., ...}, ...},
        }
    """
    p = Path(merged_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Merged themes file not found at '{merged_path}'. "
            f"Please run `python scripts/build_themes.py` first."
        )

    # Disable boolean parsing for ON, NO, etc. which are valid stock tickers
    from yaml.resolver import Resolver
    for char in list("yYnNoOtTfF"):
        if char in Resolver.yaml_implicit_resolvers:
            Resolver.yaml_implicit_resolvers[char] = [
                (tag, regexp) for tag, regexp in Resolver.yaml_implicit_resolvers[char]
                if tag != 'tag:yaml.org,2002:bool'
            ]

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    global_themes = data.get("global_themes", {})
    mappings = data.get("mappings", {})

    ticker_to_themes: Dict[str, List[str]] = {}
    theme_to_tickers: Dict[str, List[str]] = {}

    for ticker, info in mappings.items():
        # clean ticker key
        clean_ticker = str(ticker).strip('"')
        themes = info.get("themes", [])
        ticker_to_themes[clean_ticker] = themes

        for theme_id in themes:
            theme_to_tickers.setdefault(theme_id, []).append(clean_ticker)

    return {
        "ticker_to_themes": ticker_to_themes,
        "theme_to_tickers": theme_to_tickers,
        "theme_meta": global_themes,
    }
