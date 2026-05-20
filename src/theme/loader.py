"""
src/theme/loader.py

processed/themes.yaml 만 읽는다.
파일이 없으면 에러 메시지로 scripts/00_merge_themes.py 실행 안내.
"""
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List


@lru_cache(maxsize=1)
def load_themes(processed_path: str = 'data/themes/processed/themes.yaml') -> Dict:
    """
    병합된 themes.yaml 로드.

    반환:
        {
          "meta":             {...},
          "themes":           {"GT_XXX": {...}, ...},
          "tickers":          {"005930": {"market","themes","primary_tier1/2/3",...}, ...},
          "theme_to_tickers": {"GT_XXX": ["005930", ...], ...},
        }
    """
    p = Path(processed_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 파일이 없습니다.\n"
            "다음 명령을 먼저 실행하세요:\n"
            "  python scripts/00_merge_themes.py"
        )

    with open(p, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_tier_map(processed_path: str = 'data/themes/processed/themes.yaml') -> Dict[str, int]:
    """GT_XXX → tier 정수 딕셔너리."""
    data = load_themes(processed_path)
    return {k: v.get('tier', 0) for k, v in data['themes'].items()}
