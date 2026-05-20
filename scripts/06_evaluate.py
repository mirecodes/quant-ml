# scripts/06_evaluate.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    print("=== Step 1: Loading Predictions ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    
    # Target A (Attractiveness) 평가
    preds_a = {
        'FT-Transformer': predictions['FTT_A'].values,
        'F-Score': predictions['C_FSCORE'].values,
        'Quality-Score': predictions['C_QUALITY'].values,
        'Composite-Score': predictions['ACC_COMPOSITE'].values,
    }
    
    print("\n=== Step 2: Evaluating Attractiveness (Target A) ===")
    df_eval_a = evaluate_all_models(predictions, preds_a, target='A')
    print(df_eval_a.to_string(index=False))
    
    # Target R (Risk) 평가
    preds_r = {
        'FT-Transformer': predictions['FTT_R'].values,
    }
    
    print("\n=== Step 3: Evaluating Risk (Target R) ===")
    df_eval_r = evaluate_all_models(predictions, preds_r, target='R')
    print(df_eval_r.to_string(index=False))

if __name__ == '__main__':
    main()
