# evaluation/error/calculate_metrics_strictly.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def main():
    print("=== Step 1: Loading Dataset and Test Split ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Ticker split 로드
    with open('data/splits/ticker_split.json') as f:
        splits = json.load(f)
    test_tickers = set(splits['test'])

    # Test set에 속한 종목들만 필터링
    test_df = predictions[predictions['ticker'].isin(test_tickers)].copy()
    
    # 통계적 엄밀성을 위해 두 타겟(A, R) 및 두 예측치(FTT_A, FTT_R) 중 하나라도 결측치인 행은 엄격히 제외
    test_df = test_df.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()

    # 마이그레이션 출력 디렉토리 생성
    output_dir = 'evaluation/error'
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'test_set_error_metrics.txt')
    
    # 아티팩트 경로 변경
    artifact_txt_path = '/Users/mireflare/.gemini/antigravity-ide/brain/49637d44-18f3-47b1-8dd0-8f8a9c2c960c/evaluation/error/test_set_error_metrics.txt'
    os.makedirs(os.path.dirname(artifact_txt_path), exist_ok=True)

    print("=== Step 2: Calculating Strict Statistical Metrics ===")
    
    report_lines = []
    report_lines.append("=========================================================================")
    report_lines.append("       QUANT-ML MODEL STRICT EVALUATION METRICS (TEST SET)")
    report_lines.append("=========================================================================\n")
    report_lines.append("This report presents mathematically rigorous performance metrics")
    report_lines.append("of the FT-Transformer predictor model on the isolated test set.\n")

    def evaluate_target(df, actual_col, pred_col, target_label, country):
        y_true = df[actual_col].to_numpy()
        y_pred = df[pred_col].to_numpy()
        
        n = len(y_true)
        if n == 0:
            return
            
        # 1. MSE (Mean Squared Error)
        mse = mean_squared_error(y_true, y_pred)
        
        # 2. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # 3. R2 (Coefficient of Determination)
        r2 = r2_score(y_true, y_pred)
        
        # 4. Pearson Correlation & P-value
        corr, p_value = pearsonr(y_true, y_pred)
        
        # 5. MAE (보조 지표)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # SS_res 및 SS_tot 개별 연산
        y_bar = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_bar) ** 2)
        
        report_lines.append(f"▶ [{country} Market] Target: {target_label} ({actual_col})")
        report_lines.append(f"  - Total Observation Points (N)    : {n}")
        report_lines.append(f"  - Mean Squared Error (MSE)        : {mse:.8f}")
        report_lines.append(f"  - Root Mean Squared Error (RMSE)  : {rmse:.8f}")
        report_lines.append(f"  - Mean Absolute Error (MAE)       : {mae:.8f}")
        report_lines.append(f"  - R^2 Score (Determination Coeff) : {r2:.8f}")
        report_lines.append(f"  - Pearson Correlation Coeff (r)   : {corr:.8f}")
        report_lines.append(f"  - Correlation P-value (p-val)     : {p_value:.8e} (Significant if < 0.05)")
        report_lines.append("  [Mathematical Verification Details]")
        report_lines.append(f"    * True Mean (y_bar)             : {y_bar:.8f}")
        report_lines.append(f"    * Residual Sum of Squares (SS_res): {ss_res:.8f}")
        report_lines.append(f"    * Total Sum of Squares (SS_tot)   : {ss_tot:.8f}")
        report_lines.append(f"    * SS_res / SS_tot Ratio          : {ss_res / ss_tot:.8f}")
        report_lines.append(f"    * 1 - (SS_res / SS_tot)          : {1 - (ss_res / ss_tot):.8f}")
        report_lines.append("-" * 73 + "\n")

    for country in ['KR', 'US']:
        country_df = test_df[test_df['country'] == country]
        evaluate_target(country_df, 'A', 'FTT_A', 'Attractiveness A', country)
        evaluate_target(country_df, 'R', 'FTT_R', 'Robust IPR Risk R', country)

    # 텍스트 파일 저장
    content = "\n".join(report_lines)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    with open(artifact_txt_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Metrics strictly calculated and printed to: {txt_path}")
    print(f"Metrics also copied to artifact directory: {artifact_txt_path}")

if __name__ == '__main__':
    main()
