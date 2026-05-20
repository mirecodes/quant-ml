"""
scripts/00_merge_themes.py

실행 조건:
  - 최초 1회 필수 실행
  - data/themes/raw/ 파일 변경 시 --force 옵션으로 재실행
  - 그 외에는 processed/themes.yaml이 존재하면 건너뜀

역할:
  raw/global_themes.yaml + raw/kospi/*.yaml + raw/sp500/*.yaml 를
  하나의 processed/themes.yaml로 병합한다.
"""
import argparse
import yaml
import re
from pathlib import Path
from datetime import datetime


def get_tier1(theme_id: str, tier_map: dict, parent_map: dict) -> str:
    cur = theme_id
    for _ in range(10):
        p = parent_map.get(cur, 'null')
        if p == 'null' or p not in tier_map:
            return cur
        cur = p
    return cur


def get_primary_tiers(themes: list, tier_map: dict, parent_map: dict) -> dict:
    """
    themes 리스트에서 primary tier1/tier2/tier3 결정.
    tier3인 첫 번째 테마를 primary_tier3로 사용.
    없으면 tier2, 그래도 없으면 tier1.
    """
    primary = themes[0] if themes else None
    tier3, tier2, tier1 = None, None, None

    for t in themes:
        tier = tier_map.get(t, 0)
        if tier == 3 and tier3 is None:
            tier3 = t
            tier2 = parent_map.get(t)
            tier1 = get_tier1(t, tier_map, parent_map)
            break
        elif tier == 2 and tier2 is None:
            tier2 = t
            tier1 = get_tier1(t, tier_map, parent_map)
        elif tier == 1 and tier1 is None:
            tier1 = t

    if tier3 is None and tier2 is not None:
        tier3 = tier2   # fallback: tier2를 tier3 자리에
    if tier2 is None and tier1 is not None:
        tier2 = tier1
    if tier1 is None:
        tier1 = 'UNMAPPED'

    return {
        'primary_theme': primary,
        'primary_tier3': tier3 or 'UNMAPPED',
        'primary_tier2': tier2 or 'UNMAPPED',
        'primary_tier1': tier1 or 'UNMAPPED',
    }


def parse_mapping_file(path: Path, market: str, tier_map: dict, parent_map: dict) -> dict:
    """
    단일 매핑 YAML 파일 파싱.
    DUP 접미사 ticker는 이미 등록된 ticker와 병합.
    """
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    result = {}
    for key, info in raw.items():
        if not isinstance(info, dict):
            continue
        # DUP 정리: "005930_DUP", "005930_DUP2" → "005930"
        clean_key = re.sub(r'_DUP\d*$', '', str(key)).strip('"').strip("'")

        themes = info.get('themes', [])
        if clean_key in result:
            # 이미 등록된 ticker: 테마 병합 (중복 제거, 순서 유지)
            existing = result[clean_key]['themes']
            for t in themes:
                if t not in existing:
                    existing.append(t)
        else:
            tier_info = get_primary_tiers(themes, tier_map, parent_map)
            result[clean_key] = {
                'name': info.get('name', clean_key),
                'market': market,
                'themes': themes,
                **tier_info,
            }

    return result


def merge(raw_dir: str, output_path: str, force: bool = False):
    raw = Path(raw_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"[skip] {out} 이미 존재. 재생성하려면 --force 사용.")
        return

    # ── global_themes 로드 ───────────────────────────────────────────
    with open(raw / 'global_themes.yaml', encoding='utf-8') as f:
        themes_raw = yaml.safe_load(f)

    # tier/parent 인덱스 구축
    tier_map, parent_map = {}, {}
    for k, v in themes_raw.items():
        if not k.startswith('GT_') or not isinstance(v, dict):
            continue
        tier_map[k]   = v.get('tier', 0)
        parent_map[k] = v.get('parent', 'null')

    # ── 종목 매핑 로드 ───────────────────────────────────────────────
    all_tickers = {}

    kr_files = sorted((raw / 'kospi').glob('*.yaml'))
    for fpath in kr_files:
        parsed = parse_mapping_file(fpath, 'KR', tier_map, parent_map)
        for ticker, info in parsed.items():
            if ticker not in all_tickers:
                all_tickers[ticker] = info
            else:
                for t in info['themes']:
                    if t not in all_tickers[ticker]['themes']:
                        all_tickers[ticker]['themes'].append(t)

    us_files = sorted((raw / 'sp500').glob('*.yaml'))
    for fpath in us_files:
        parsed = parse_mapping_file(fpath, 'US', tier_map, parent_map)
        for ticker, info in parsed.items():
            if ticker not in all_tickers:
                all_tickers[ticker] = info
            else:
                for t in info['themes']:
                    if t not in all_tickers[ticker]['themes']:
                        all_tickers[ticker]['themes'].append(t)

    # ── 역방향 인덱스 ────────────────────────────────────────────────
    theme_to_tickers = {}
    for ticker, info in all_tickers.items():
        for t in info['themes']:
            theme_to_tickers.setdefault(t, []).append(ticker)

    # ── 통계 ─────────────────────────────────────────────────────────
    n_kr = sum(1 for v in all_tickers.values() if v['market'] == 'KR')
    n_us = sum(1 for v in all_tickers.values() if v['market'] == 'US')

    # ── 저장 ─────────────────────────────────────────────────────────
    output = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'n_themes': len(tier_map),
            'n_tickers_kr': n_kr,
            'n_tickers_us': n_us,
            'n_tickers_total': len(all_tickers),
            'source_files': (
                [str(f) for f in kr_files] +
                [str(f) for f in us_files]
            ),
        },
        'themes': {k: v for k, v in themes_raw.items()
                   if k.startswith('GT_')},
        'tickers': all_tickers,
        'theme_to_tickers': theme_to_tickers,
    }

    with open(out, 'w', encoding='utf-8') as f:
        yaml.dump(output, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    print(f"[done] {out}")
    print(f"  테마: {len(tier_map)}개 | KR: {n_kr}종목 | US: {n_us}종목")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir',  default='data/themes/raw')
    parser.add_argument('--output',   default='data/themes/processed/themes.yaml')
    parser.add_argument('--force',    action='store_true',
                        help='processed 파일이 있어도 강제 재생성')
    args = parser.parse_args()
    merge(args.raw_dir, args.output, args.force)
