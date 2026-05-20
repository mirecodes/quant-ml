"""
scripts/02b_build_theme_context.py

실행: python scripts/02b_build_theme_context.py
02_build_features.py 이후, 03_build_labels.py 이전에 실행한다.

Point-in-Time 보장:
  분기별로 해당 시점의 종목 피처만 사용해 테마 내 순위를 계산한다.
  미래 데이터 참조 없음.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.theme.context import compute_theme_context
from src.utils.io import load_parquet, save_parquet, report_memory


def main():
    print("테마 비중 컨텍스트 계산 중...")
    df = load_parquet('data/processed/features_stock.parquet')
    report_memory(df, "features_stock")

    import yaml
    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    # 전체 peer 기준 (학습 전 전처리 단계이므로 peer 제한 없음)
    # Train/Val 분리는 04_train.py에서 처리
    theme_ctx = compute_theme_context(
        df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=None,
    )

    save_parquet(theme_ctx, 'data/processed/theme_context.parquet')
    report_memory(theme_ctx, "theme_context")
    print(f"완료: {len(theme_ctx)} rows")


if __name__ == '__main__':
    main()
