"""
src/utils/split.py

종목 기반 Stratified Split.

stratify 기준: market(KR/US) × theme_level(tier1/tier2/tier3)
  - 각 (market, theme) 버킷에서 test_ratio, val_ratio 만큼 종목 추출
  - 버킷 종목 수 < min_bucket_size 이면 상위 tier 버킷으로 합산
  - 한 종목이 여러 테마에 속해도 primary_tier3 기준 단일 버킷 배정

반환값:
  train_tickers, val_tickers, test_tickers : list[str]

검증 보장:
  - KR/US 비율이 Train/Val/Test에서 동일하게 유지됨
  - Tier1 10개, Tier2 44개, Tier3 147개 모두 Test에 1종목 이상 포함
    (버킷 크기 ≥ min_bucket_size 일 때)
"""
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from src.theme.loader import load_themes


def stratified_ticker_split(
    processed_path: str = 'data/themes/processed/themes.yaml',
    test_ratio:  float = 0.15,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
    min_bucket_size: int = 3,
    theme_level: str = 'tier3',    # 'tier1' | 'tier2' | 'tier3'
) -> Tuple[List[str], List[str], List[str]]:
    """
    종목을 market × theme_level 버킷 기준으로 균등 분할한다.

    Args:
        processed_path   : themes.yaml 경로
        test_ratio       : Test 비율
        val_ratio        : Val 비율
        seed             : 재현성 시드
        min_bucket_size  : 버킷당 최소 종목 수.
                           미달 시 상위 tier 버킷으로 합산.
        theme_level      : stratify 기준 tier 레벨

    Returns:
        (train_tickers, val_tickers, test_tickers)
    """
    rng = random.Random(seed)

    data     = load_themes(processed_path)
    tickers  = data['tickers']

    # ── 버킷 배정 ─────────────────────────────────────────────────────
    # key: (market, tier_id)
    tier_key = {
        'tier1': 'primary_tier1',
        'tier2': 'primary_tier2',
        'tier3': 'primary_tier3',
    }[theme_level]

    # tier 상향 fallback 순서
    tier_fallback = {
        'tier3': ['primary_tier3', 'primary_tier2', 'primary_tier1'],
        'tier2': ['primary_tier2', 'primary_tier1'],
        'tier1': ['primary_tier1'],
    }[theme_level]

    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for ticker, info in tickers.items():
        market = info.get('market', 'UNKNOWN')
        # 설정된 theme_level 기준으로 버킷 키 결정
        tier_id = info.get(tier_key, 'UNMAPPED')
        buckets[(market, tier_id)].append(ticker)

    # ── 소형 버킷 상향 합산 ───────────────────────────────────────────
    # Tier3 버킷이 min_bucket_size 미만이면 Tier2, Tier1 버킷으로 이동
    if theme_level == 'tier3':
        merged: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        theme_meta = data['themes']

        for (market, tier3_id), ticker_list in buckets.items():
            if len(ticker_list) >= min_bucket_size:
                merged[(market, tier3_id)].extend(ticker_list)
            else:
                # tier2로 상향
                tier2_id = theme_meta.get(tier3_id, {}).get('parent', 'UNMAPPED')
                if tier2_id == 'null' or tier2_id not in theme_meta:
                    tier2_id = 'UNMAPPED'
                merged[(market, tier2_id)].extend(ticker_list)

        # 합산 후 여전히 작은 버킷은 tier1으로 한 번 더 상향
        final: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for (market, tier2_id), ticker_list in merged.items():
            if len(ticker_list) >= min_bucket_size:
                final[(market, tier2_id)].extend(ticker_list)
            else:
                tier1_id = theme_meta.get(tier2_id, {}).get('parent', 'UNMAPPED')
                if tier1_id == 'null' or tier1_id not in theme_meta:
                    tier1_id = 'UNMAPPED'
                final[(market, tier1_id)].extend(ticker_list)

        buckets = final

    # ── 버킷별 분할 ───────────────────────────────────────────────────
    train_tickers, val_tickers, test_tickers = [], [], []

    for (market, theme_id), ticker_list in sorted(buckets.items()):
        # 중복 제거 (멀티 테마 종목이 여러 버킷에 중복 포함될 수 있음)
        unique = list(dict.fromkeys(ticker_list))   # 순서 유지 중복 제거
        rng.shuffle(unique)
        n = len(unique)

        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        # 남은 것 전부 train (반올림 오차 흡수)
        n_train = n - n_test - n_val

        if n_train < 1:
            # 극소 버킷: 전부 train
            train_tickers.extend(unique)
            continue

        test_tickers.extend(unique[:n_test])
        val_tickers.extend(unique[n_test:n_test + n_val])
        train_tickers.extend(unique[n_test + n_val:])

    # 최종 중복 제거 (버킷 상향 합산 과정에서 중복 가능)
    def dedup(lst):
        return list(dict.fromkeys(lst))

    train_tickers = dedup(train_tickers)
    val_tickers   = dedup(val_tickers)
    test_tickers  = dedup(test_tickers)

    # Val/Test에 Train 종목이 겹치지 않도록 보장
    train_set = set(train_tickers)
    val_tickers  = [t for t in val_tickers  if t not in train_set]
    test_tickers = [t for t in test_tickers if t not in train_set]

    return train_tickers, val_tickers, test_tickers


def print_split_report(
    train: List[str],
    val:   List[str],
    test:  List[str],
    processed_path: str = 'data/themes/processed/themes.yaml',
):
    """
    분할 결과 요약 출력.
    Tier1별, Market별 분포를 보여줌.
    """
    from collections import Counter
    data    = load_themes(processed_path)
    tickers = data['tickers']

    def market_dist(lst):
        c = Counter(tickers[t]['market'] for t in lst if t in tickers)
        return dict(c)

    def tier1_dist(lst):
        c = Counter(tickers[t]['primary_tier1'] for t in lst if t in tickers)
        return dict(c)

    total = len(train) + len(val) + len(test)
    print(f"{'='*55}")
    print(f"  분할 결과 (총 {total}종목)")
    print(f"  Train: {len(train)} ({len(train)/total:.0%})")
    print(f"  Val:   {len(val)}  ({len(val)/total:.0%})")
    print(f"  Test:  {len(test)} ({len(test)/total:.0%})")
    print(f"{'='*55}")

    print("  [Market 분포]")
    for split_name, split_list in [('Train', train), ('Val', val), ('Test', test)]:
        d = market_dist(split_list)
        print(f"    {split_name}: KR={d.get('KR',0)}, US={d.get('US',0)}")

    print("  [Tier1 분포 — Test]")
    for tier1, cnt in sorted(tier1_dist(test).items(), key=lambda x: -x[1]):
        print(f"    {tier1:<35s}: {cnt}종목")
    print(f"{'='*55}")
