"""
scripts/04_train.py

실행: python scripts/04_train.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from src.data.dataset import StockDataset, collate_fn
from src.models.predictor import StockPredictor
from src.utils.device import get_device, get_optimal_batch_size, report_environment
from src.utils.io import load_parquet


def main():
    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    report_environment()

    # ── 데이터 로드 ────────────────────────────────────────────────────
    df_stock  = load_parquet('data/processed/features_stock.parquet')
    df_macro  = load_parquet('data/processed/features_macro.parquet')
    df_theme  = load_parquet('data/processed/theme_context.parquet')
    df_labels = load_parquet('data/processed/labels.parquet')

    # ── 컬럼 정의 ───────────────────────────────────────────────────────
    # 종목 재무 시계열 피처 (거시 제외)
    STOCK_SEQ_COLS = (
        [c for c in df_stock.columns if c.startswith('F_')]   # 재무 45
      + [c for c in df_stock.columns if c.startswith('A_')]   # 자산집중도 35
      + ['open', 'high', 'low', 'close', 'volume']
      + ['ret_1q', 'ret_4q']
    )
    # 거시 시계열 피처
    MACRO_SEQ_COLS = [c for c in df_macro.columns
                      if c.startswith('M_') and c != 'date']
    # 스냅샷 수치형 (계산형 지표)
    SNAP_NUM_COLS = [c for c in df_stock.columns if c.startswith('C_')]
    # 스냅샷 범주형
    SNAP_CAT_COLS = ['country', 'sector', 'size_tier']

    # 범주형 카디널리티 (미등록 카테고리를 위해 +1)
    cat_cardinalities = [
        int(df_stock[c].nunique()) for c in SNAP_CAT_COLS
    ]

    # ── 모델 설정 ────────────────────────────────────────────────────────
    mcfg = cfg['model']
    model_cfg = {
        'n_stock_features':  len(STOCK_SEQ_COLS),
        'n_macro_features':  len(MACRO_SEQ_COLS),
        'n_snap_num':        len(SNAP_NUM_COLS),
        'cat_cardinalities': cat_cardinalities,
        'lstm_stock_hidden': mcfg['lstm_stock_hidden'],
        'lstm_stock_layers': mcfg.get('lstm_stock_layers', 2),
        'lstm_macro_hidden': mcfg['lstm_macro_hidden'],
        'lstm_macro_layers': mcfg.get('lstm_macro_layers', 1),
        'theme_proj_dim':    mcfg['theme_proj_dim'],
        'd_token':           mcfg['d_token'],
        'n_heads':           mcfg['n_heads'],
        'n_layers':          mcfg['n_layers'],
        'ffn_factor':        mcfg.get('ffn_factor', 4/3),
        'dropout':           mcfg['dropout'],
        'attn_dropout':      mcfg.get('attn_dropout', 0.1),
        'lr':                mcfg['lr'],
        'weight_decay':      mcfg['weight_decay'],
        'max_epochs':        mcfg['max_epochs'],
    }

    # ── 데이터 분할 및 테마 비중 계산 (peer 오염 방지) ──────────────────────
    from src.utils.split import stratified_ticker_split, print_split_report

    split_cfg = cfg['split']
    train_tickers, val_tickers, test_tickers = stratified_ticker_split(
        processed_path = cfg['themes']['processed_path'],
        test_ratio     = split_cfg['test_ratio'],
        val_ratio      = split_cfg['val_ratio'],
        seed           = split_cfg['seed'],
        min_bucket_size= split_cfg['stratify']['min_bucket_size'],
        theme_level    = split_cfg['stratify']['theme_level'],
    )
    print_split_report(
        train_tickers, val_tickers, test_tickers,
        processed_path=cfg['themes']['processed_path'],
    )

    # 저장 (재현성)
    import json
    from pathlib import Path
    Path('data/splits').mkdir(exist_ok=True)
    json.dump({
        'train': train_tickers,
        'val':   val_tickers,
        'test':  test_tickers,
    }, open('data/splits/ticker_split.json', 'w'))

    # Dataset 분할
    train_df = df_stock[df_stock['ticker'].isin(train_tickers)]
    val_df   = df_stock[df_stock['ticker'].isin(val_tickers)]

    # ── 테마 비중: Train peer 오염 방지 ───────────────────────────────────
    # Train 종목의 theme_ctx는 Train peer만 참조
    # Val/Test 종목의 theme_ctx는 전체 peer 참조 (실운용과 동일)
    from src.theme.context import compute_theme_context

    df_theme_train = compute_theme_context(
        train_df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=set(train_tickers),   # Train peer만
    )
    df_theme_val = compute_theme_context(
        val_df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=None,                 # 전체 peer
    )
    # Validation은 별도 계산 후 concat
    df_theme = pd.concat([df_theme_train, df_theme_val], ignore_index=True)

    train_ds = StockDataset(
        df_stock=train_df,
        df_macro=df_macro,
        df_theme=df_theme,
        df_labels=df_labels[df_labels['ticker'].isin(train_tickers)],
        stock_seq_cols=STOCK_SEQ_COLS,
        macro_seq_cols=MACRO_SEQ_COLS,
        snap_num_cols=SNAP_NUM_COLS,
        snap_cat_cols=SNAP_CAT_COLS,
        max_seq_len=mcfg.get('lstm_stock_max_seq', 20),
    )
    val_ds = StockDataset(
        df_stock=val_df,
        df_macro=df_macro,
        df_theme=df_theme,
        df_labels=df_labels[df_labels['ticker'].isin(val_tickers)],
        stock_seq_cols=STOCK_SEQ_COLS,
        macro_seq_cols=MACRO_SEQ_COLS,
        snap_num_cols=SNAP_NUM_COLS,
        snap_cat_cols=SNAP_CAT_COLS,
        max_seq_len=mcfg.get('lstm_stock_max_seq', 20),
    )

    batch_size = get_optimal_batch_size(mcfg['batch_size'])
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=mcfg['num_workers'],
        persistent_workers=mcfg['persistent_workers'],
        pin_memory=False,   # MPS 미지원
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── 학습 ─────────────────────────────────────────────────────────────
    device      = get_device()
    accelerator = 'mps' if device.type == 'mps' else 'cpu'

    model = StockPredictor(model_cfg)

    # loggers 정의 (TensorBoard + 선택적 WandB)
    loggers = []
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger("lightning_logs", name="ftt_stock"))
    except Exception as e:
        print(f"[Logger Warning] Failed to initialize TensorBoard Logger: {e}")
    
    use_wandb = mcfg.get('use_wandb', False)
    if use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
            import wandb
            wandb_logger = WandbLogger(project="quant-ml-ftt", name="ftt_run")
            loggers.append(wandb_logger)
            print("[WandB] Weights & Biases Logger successfully initialized!")
        except Exception as e:
            print(f"[WandB Warning] Failed to initialize WandB Logger: {e}")

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=mcfg['patience'], mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='stockml-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        max_epochs=mcfg['max_epochs'],
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=mcfg['grad_clip'],
        callbacks=callbacks,
        logger=loggers if loggers else True,
        precision=mcfg.get('precision', '32-true'),
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    except RuntimeError as e:
        if 'MPS' in str(e):
            print(f"[MPS fallback] {e}")
            trainer = pl.Trainer(
                max_epochs=mcfg['max_epochs'],
                accelerator='cpu',
                gradient_clip_val=mcfg['grad_clip'],
                callbacks=callbacks,
                logger=loggers if loggers else True,
            )
            trainer.fit(model, train_loader, val_loader)
        else:
            raise

    print("학습 완료.")


if __name__ == '__main__':
    main()
