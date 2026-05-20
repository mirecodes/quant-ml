# scripts/06_evaluate.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import yaml
import pandas as pd
import numpy as np
from src.evaluation.metrics import spearman_ic, long_short_spread


def evaluate_all_models(test_data, predictions_dict, target='A'):
    """
    predictions_dict: {model_name: predictions_array}
    """
    realized = test_data[target].values
    
    results = []
    for model_name, preds in predictions_dict.items():
        ic = spearman_ic(preds, realized)
        ls = long_short_spread(preds, realized)
        rmse = np.sqrt(np.nanmean((preds - realized) ** 2))
        
        results.append({
            'Model': model_name,
            'Target': target,
            'Rank IC': ic,
            'L-S Spread': ls,
            'RMSE': rmse,
        })
    
    return pd.DataFrame(results).sort_values('Rank IC', ascending=False)


def main():
    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    print("=== Step 1: Loading Predictions & Splits ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Load splits
    with open('data/splits/ticker_split.json') as f:
        splits = json.load(f)
    train_tickers = set(splits['train'])
    val_tickers = set(splits['val'])
    test_tickers = set(splits['test'])

    # ── 1. Ticker-Stratified Split (Main Evaluation) ──────────────────────────
    test_data_main = predictions[predictions['ticker'].isin(test_tickers)].copy()
    
    print(f"\n=== Step 2: Main Evaluation (Ticker-Stratified Test Split, n={len(test_data_main)}) ===")
    if not test_data_main.empty:
        preds_a_main = {
            'FT-Transformer': test_data_main['FTT_A'].values,
            'F-Score': test_data_main['C_FSCORE'].values,
            'Quality-Score': test_data_main['C_QUALITY'].values,
            'Composite-Score': test_data_main['ACC_COMPOSITE'].values,
        }
        df_eval_a = evaluate_all_models(test_data_main, preds_a_main, target='A')
        print("--- Attractiveness (Target A) ---")
        print(df_eval_a.to_string(index=False))

        preds_r_main = {
            'FT-Transformer': test_data_main['FTT_R'].values,
        }
        df_eval_r = evaluate_all_models(test_data_main, preds_r_main, target='R')
        print("\n--- Risk (Target R) ---")
        print(df_eval_r.to_string(index=False))
    else:
        print("Warning: Ticker-Stratified test data is empty.")

    # ── 2. Time-Holdout (Auxiliary Evaluation) ──────────────────────────
    time_cfg = cfg['split']['time_holdout']
    if time_cfg['enabled']:
        cutoff = pd.Timestamp(time_cfg['cutoff'])
        
        # Filtering for time holdout: train_tickers (or train+val tickers) after cutoff
        eval_tickers = train_tickers if time_cfg['train_tickers'] == 'train_only' else (train_tickers | val_tickers)
        test_data_time = predictions[
            (predictions['ticker'].isin(eval_tickers)) & 
            (predictions['date'] > cutoff)
        ].copy()
        
        print(f"\n=== Step 3: Auxiliary Evaluation (Time Holdout Split, Date > {time_cfg['cutoff']}, n={len(test_data_time)}) ===")
        if not test_data_time.empty:
            preds_a_time = {
                'FT-Transformer': test_data_time['FTT_A'].values,
                'F-Score': test_data_time['C_FSCORE'].values,
                'Quality-Score': test_data_time['C_QUALITY'].values,
                'Composite-Score': test_data_time['ACC_COMPOSITE'].values,
            }
            df_eval_a_time = evaluate_all_models(test_data_time, preds_a_time, target='A')
            print("--- Attractiveness (Target A) ---")
            print(df_eval_a_time.to_string(index=False))

            preds_r_time = {
                'FT-Transformer': test_data_time['FTT_R'].values,
            }
            df_eval_r_time = evaluate_all_models(test_data_time, preds_r_time, target='R')
            print("\n--- Risk (Target R) ---")
            print(df_eval_r_time.to_string(index=False))
        else:
            print("Warning: No data available for Time Holdout Evaluation after cutoff.")


if __name__ == '__main__':
    main()
