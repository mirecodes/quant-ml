# tests/test_split.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_stratified_split_market_ratio():
    """KR/US 비율이 Train/Val/Test에서 유사하게 유지되는지."""
    from src.utils.split import stratified_ticker_split
    from src.theme.loader import load_themes

    train, val, test = stratified_ticker_split()
    data    = load_themes()
    tickers = data['tickers']

    def kr_ratio(lst):
        kr = sum(1 for t in lst if tickers.get(t, {}).get('market') == 'KR')
        return kr / len(lst) if lst else 0

    r_train = kr_ratio(train)
    r_val   = kr_ratio(val)
    r_test  = kr_ratio(test)

    # KR 비율 차이가 5% 이내
    assert abs(r_train - r_test) < 0.05, f"KR ratio mismatch: train={r_train:.2f}, test={r_test:.2f}"
    assert abs(r_train - r_val)  < 0.05


def test_stratified_split_no_overlap():
    """Train/Val/Test 간 종목 중복 없음."""
    from src.utils.split import stratified_ticker_split

    train, val, test = stratified_ticker_split()
    train_set = set(train)
    val_set   = set(val)
    test_set  = set(test)

    assert len(train_set & val_set)  == 0, "Train-Val overlap"
    assert len(train_set & test_set) == 0, "Train-Test overlap"
    assert len(val_set   & test_set) == 0, "Val-Test overlap"


def test_stratified_split_tier1_coverage():
    """Test 셋에 모든 Tier1 카테고리가 최소 1종목 포함."""
    from src.utils.split import stratified_ticker_split
    from src.theme.loader import load_themes

    _, _, test = stratified_ticker_split()
    data    = load_themes()
    tickers = data['tickers']

    tier1_in_test = set(
        tickers[t]['primary_tier1'] for t in test if t in tickers
    )
    all_tier1 = set(
        k for k, v in data['themes'].items() if v.get('tier') == 1
    )
    missing = all_tier1 - tier1_in_test
    assert not missing, f"Test에 없는 Tier1: {missing}"


def test_merge_themes_idempotent(tmp_path):
    """00_merge_themes.py 두 번 실행해도 결과 동일."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("merge_themes", "scripts/00_merge_themes.py")
    merge_themes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge_themes)
    merge = merge_themes.merge

    out = tmp_path / 'themes.yaml'
    merge('data/themes/raw', str(out), force=True)
    import yaml
    with open(out) as f:
        content1 = yaml.safe_load(f)
    merge('data/themes/raw', str(out), force=True)
    with open(out) as f:
        content2 = yaml.safe_load(f)
    # 종목 수, 테마 수 동일
    assert content1['meta']['n_tickers_total'] == content2['meta']['n_tickers_total']
    assert content1['meta']['n_themes'] == content2['meta']['n_themes']
