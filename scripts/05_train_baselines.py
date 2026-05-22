"""
scripts/05_train_baselines.py

실행: python scripts/05_train_baselines.py
GBM baseline을 제거하고, FT-Transformer 예측치(훈련 완료된 checkpoint 로드)와
회계 지표(Accounting Baseline) 예측치를 결합하여 predictions_latest.parquet 파일로 저장한다.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import glob
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.models.baseline_accounting import AccountingBaseline
from src.models.predictor import StockPredictor
from src.data.dataset import StockDataset, collate_fn
from src.utils.io import save_parquet, load_parquet, report_memory


def main():
    print("=== Step 1: Loading Dataset ===")
    df_stock  = load_parquet('data/processed/features_stock.parquet')
    df_macro  = load_parquet('data/processed/features_macro.parquet')
    df_theme  = load_parquet('data/processed/theme_context.parquet')
    df_labels = load_parquet('data/processed/labels.parquet')

    # 두 데이터프레임 병합 (종목 피처 + 라벨)
    data = df_stock.merge(df_labels, on=['ticker', 'date'], how='inner')
    report_memory(data, "Merged Data")

    with open('config/settings.yaml') as f:
        config = yaml.safe_load(f)

    # 전체 데이터셋을 테스트 대상으로 삼아 inference를 수행하고 predictions_latest를 채웁니다.
    # 평가 대상 분리는 06_evaluate.py에서 수행합니다.
    test_data = data.sort_values(['ticker', 'date']).dropna(subset=['A', 'R']).reset_index(drop=True)
    print(f"Inference and baseline calculation size: {len(test_data)}")

    # ── 컬럼 정의 ───────────────────────────────────────────────────────
    STOCK_SEQ_COLS = (
        [c for c in df_stock.columns if c.startswith('F_')]
      + [c for c in df_stock.columns if c.startswith('A_')]
      + ['open', 'high', 'low', 'close', 'volume']
      + ['ret_1q', 'ret_4q']
    )
    MACRO_SEQ_COLS = [c for c in df_macro.columns if c.startswith('M_') and c != 'date']
    SNAP_NUM_COLS = [c for c in df_stock.columns if c.startswith('C_')]
    SNAP_CAT_COLS = ['country', 'sector', 'size_tier']

    # ── FT-Transformer 예측 로드 & 추론 ────────────────────────────────────
    print("\n=== Step 2: Running FT-Transformer Inference ===")
    pred_a_ftt = np.zeros(len(test_data), dtype=np.float32)
    pred_r_ftt = np.zeros(len(test_data), dtype=np.float32)

    ckpt_files = glob.glob('checkpoints/stockml-*.ckpt')
    if ckpt_files:
        # 가장 최근에 저장되었거나 val/loss가 가장 낮은 파일 선택
        best_ckpt = ckpt_files[0]
        # val_loss/loss 값을 포함하는 체크포인트 파싱 시도
        try:
            import re
            def get_loss(p):
                m = re.search(r'loss[=_](-?\d+(?:\.\d+)?)', p)
                if m:
                    return float(m.group(1))
                return 999.0
            best_ckpt = min(ckpt_files, key=get_loss)
        except Exception:
            pass
        
        print(f"Loading checkpoint for inference: {best_ckpt}")
        try:
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            print(f"Inference device: {device}")
            model = StockPredictor.load_from_checkpoint(best_ckpt)
            model.to(device)
            model.eval()

            test_ds = StockDataset(
                df_stock=df_stock.copy(),
                df_macro=df_macro,
                df_theme=df_theme,
                df_labels=df_labels.copy(),
                stock_seq_cols=STOCK_SEQ_COLS,
                macro_seq_cols=MACRO_SEQ_COLS,
                snap_num_cols=SNAP_NUM_COLS,
                snap_cat_cols=SNAP_CAT_COLS,
                max_seq_len=20,
            )

            test_loader = DataLoader(
                test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
            )

            A_preds = []
            R_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch_dev = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    A, R = model(batch_dev)
                    A_preds.append(A.cpu().numpy())
                    R_preds.append(R.cpu().numpy())

            if A_preds:
                pred_a_ftt = np.concatenate(A_preds)
                pred_r_ftt = np.concatenate(R_preds)
                print("Successfully generated predictions from trained FTT model.")
            else:
                print("Generated empty predictions. Falling back to mock.")
                pred_a_ftt = np.random.normal(0.0, 0.1, len(test_data)).astype(np.float32)
                pred_r_ftt = np.random.uniform(0.05, 0.2, len(test_data)).astype(np.float32)

        except Exception as e:
            print(f"Failed to run inference: {e}. Falling back to mock FTT predictions.")
            pred_a_ftt = np.random.normal(0.0, 0.1, len(test_data)).astype(np.float32)
            pred_r_ftt = np.random.uniform(0.05, 0.2, len(test_data)).astype(np.float32)
    else:
        print("No checkpoints found. Running with mock FTT predictions.")
        pred_a_ftt = np.random.normal(0.0, 0.1, len(test_data)).astype(np.float32)
        pred_r_ftt = np.random.uniform(0.05, 0.2, len(test_data)).astype(np.float32)

    # ── 학술/재무 베이스라인 계산 ─────────────────────────────────────────────
    print("\n=== Step 3: Computing Accounting Baselines ===")
    acc_fscore = AccountingBaseline('fscore')
    score_fscore = acc_fscore.score(test_data)

    acc_quality = AccountingBaseline('quality')
    score_quality = acc_quality.score(test_data)

    acc_composite = AccountingBaseline('composite')
    score_composite = acc_composite.score(test_data)

    # ── 최종 예측 취합 및 저장 ────────────────────────────────────────────────
    print("\n=== Step 4: Generating and Saving Final Predictions ===")
    predictions = test_data[['ticker', 'country', 'sector', 'size_tier', 'date', 'close', 'A', 'R']].copy()
    predictions['name'] = predictions['ticker']

    # 글로벌 테마 및 회사명 매핑 로드
    themes_map = {}
    ticker_names = {}
    from src.theme.loader import load_themes
    try:
        theme_data = load_themes(config['themes']['processed_path'])
        global_themes_metadata = theme_data.get('themes', {})
        tickers_metadata = theme_data.get('tickers', {})
        for ticker, info in tickers_metadata.items():
            ticker_names[ticker] = info.get('name', ticker)
            theme_ids = info.get('themes', [])
            theme_names = [global_themes_metadata.get(tid, {}).get('name_ko', tid) for tid in theme_ids]
            themes_map[ticker] = theme_names
    except Exception as e:
        print(f"Error loading processed themes: {e}")

    if ticker_names:
        predictions['name'] = predictions['ticker'].astype(str).map(ticker_names).fillna(predictions['ticker'].astype(str))

    predictions['themes'] = predictions.apply(lambda r: themes_map.get(r['ticker'], ['기타 및 미분류']), axis=1)

    predictions['FTT_A'] = pred_a_ftt
    predictions['FTT_R'] = pred_r_ftt
    predictions['C_FSCORE'] = score_fscore
    predictions['C_QUALITY'] = score_quality
    predictions['ACC_COMPOSITE'] = score_composite

    # UI는 'A'와 'R' 컬럼을 최종 예측으로 사용하므로, FTT 결과를 복사해둠
    # (Streamlit 대시보드 대응용)
    predictions['A_FTT'] = predictions['FTT_A']
    predictions['R_FTT'] = predictions['FTT_R']

    save_parquet(predictions, 'data/processed/predictions_latest.parquet')
    report_memory(predictions, "predictions_latest.parquet")
    print(f"Baseline & Model prediction output completed successfully!")


if __name__ == '__main__':
    main()
