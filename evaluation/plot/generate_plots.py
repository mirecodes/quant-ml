# evaluation/plot/generate_plots.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Step 1: Loading Predictions & Splits ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Load splits
    with open('data/splits/ticker_split.json') as f:
        splits = json.load(f)
    test_tickers = set(splits['test'])

    # Test set 필터링
    test_df = predictions[predictions['ticker'].isin(test_tickers)].copy()
    test_df = test_df.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()

    if test_df.empty:
        print("오류: 비교 대상 test set 데이터가 비어 있습니다.")
        return

    # 오차 계산
    test_df['error_A'] = test_df['FTT_A'] - test_df['A']
    test_df['abs_error_A'] = test_df['error_A'].abs()
    test_df['error_R'] = test_df['FTT_R'] - test_df['R']
    test_df['abs_error_R'] = test_df['error_R'].abs()

    # 마이그레이션 저장 디렉토리 정의 및 생성 (프로젝트 내부 + 아티팩트 백업)
    save_dirs = [
        '/Users/mireflare/.gemini/antigravity-ide/brain/49637d44-18f3-47b1-8dd0-8f8a9c2c960c/evaluation/plot',
        'evaluation/plot'
    ]
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    print("=== Step 2: Aggregating Data by Date ===")
    # 국가별 분리하여 날짜별 평균 계산
    kr_df = test_df[test_df['country'] == 'KR'].copy()
    us_df = test_df[test_df['country'] == 'US'].copy()

    kr_grouped = kr_df.groupby('date')[['A', 'FTT_A', 'abs_error_A', 'R', 'FTT_R', 'abs_error_R']].mean().sort_index()
    us_grouped = us_df.groupby('date')[['A', 'FTT_A', 'abs_error_A', 'R', 'FTT_R', 'abs_error_R']].mean().sort_index()

    # 차트 스타일링 설정
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.figsize'] = (14, 6)

    # -------------------------------------------------------------------------
    # Chart 1: A 실제 vs 예측 비교
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # KR
    axes[0].plot(kr_grouped.index, kr_grouped['A'], label='Actual (GT) A', color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
    axes[0].plot(kr_grouped.index, kr_grouped['FTT_A'], label='Predicted A', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s', markersize=5)
    axes[0].set_title('Korea (KR) Attractiveness A: Actual vs Predicted', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Mean A Value')
    axes[0].legend(frameon=True)
    axes[0].tick_params(axis='x', rotation=30)
    
    # US
    axes[1].plot(us_grouped.index, us_grouped['A'], label='Actual (GT) A', color='#2ca02c', linewidth=2.5, marker='o', markersize=5)
    axes[1].plot(us_grouped.index, us_grouped['FTT_A'], label='Predicted A', color='#d62728', linewidth=2.5, linestyle='--', marker='s', markersize=5)
    axes[1].set_title('United States (US) Attractiveness A: Actual vs Predicted', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Mean A Value')
    axes[1].legend(frameon=True)
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_a_comparison.png'), dpi=150)
    plt.close()
    print("Saved: plot_a_comparison.png")

    # -------------------------------------------------------------------------
    # Chart 2: R 실제 vs 예측 비교
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # KR
    axes[0].plot(kr_grouped.index, kr_grouped['R'], label='Actual (GT) R', color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
    axes[0].plot(kr_grouped.index, kr_grouped['FTT_R'], label='Predicted R', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s', markersize=5)
    axes[0].set_title('Korea (KR) Risk R: Actual vs Predicted', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Mean R Value')
    axes[0].legend(frameon=True)
    axes[0].tick_params(axis='x', rotation=30)
    
    # US
    axes[1].plot(us_grouped.index, us_grouped['R'], label='Actual (GT) R', color='#2ca02c', linewidth=2.5, marker='o', markersize=5)
    axes[1].plot(us_grouped.index, us_grouped['FTT_R'], label='Predicted R', color='#d62728', linewidth=2.5, linestyle='--', marker='s', markersize=5)
    axes[1].set_title('United States (US) Risk R: Actual vs Predicted', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Mean R Value')
    axes[1].legend(frameon=True)
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_r_comparison.png'), dpi=150)
    plt.close()
    print("Saved: plot_r_comparison.png")

    # -------------------------------------------------------------------------
    # Chart 3: A 절대 오차 (MAE)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    
    # KR
    axes[0].plot(kr_grouped.index, kr_grouped['abs_error_A'], color='#e377c2', linewidth=2.5, marker='d', markersize=6)
    axes[0].fill_between(kr_grouped.index, kr_grouped['abs_error_A'], color='#e377c2', alpha=0.15)
    axes[0].set_title('Korea (KR) A Absolute Error (MAE) Trend', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Absolute Error')
    axes[0].tick_params(axis='x', rotation=30)
    
    # US
    axes[1].plot(us_grouped.index, us_grouped['abs_error_A'], color='#9467bd', linewidth=2.5, marker='d', markersize=6)
    axes[1].fill_between(us_grouped.index, us_grouped['abs_error_A'], color='#9467bd', alpha=0.15)
    axes[1].set_title('United States (US) A Absolute Error (MAE) Trend', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Absolute Error')
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_a_error.png'), dpi=150)
    plt.close()
    print("Saved: plot_a_error.png")

    # -------------------------------------------------------------------------
    # Chart 4: R 절대 오차 (MAE)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    
    # KR
    axes[0].plot(kr_grouped.index, kr_grouped['abs_error_R'], color='#17becf', linewidth=2.5, marker='d', markersize=6)
    axes[0].fill_between(kr_grouped.index, kr_grouped['abs_error_R'], color='#17becf', alpha=0.15)
    axes[0].set_title('Korea (KR) R Absolute Error (MAE) Trend', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Absolute Error')
    axes[0].tick_params(axis='x', rotation=30)
    
    # US
    axes[1].plot(us_grouped.index, us_grouped['abs_error_R'], color='#bcbd22', linewidth=2.5, marker='d', markersize=6)
    axes[1].fill_between(us_grouped.index, us_grouped['abs_error_R'], color='#bcbd22', alpha=0.15)
    axes[1].set_title('United States (US) R Absolute Error (MAE) Trend', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Absolute Error')
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_r_error.png'), dpi=150)
    plt.close()
    print("Saved: plot_r_error.png")
    
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()
