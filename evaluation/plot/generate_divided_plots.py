# evaluation/plot/generate_divided_plots.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Step 1: Loading Data ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    with open('data/splits/ticker_split.json') as f:
        splits = json.load(f)
    test_tickers = set(splits['test'])

    test_df = predictions[predictions['ticker'].isin(test_tickers)].copy()
    test_df = test_df.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()

    # 오차 계산
    test_df['abs_error_A'] = (test_df['FTT_A'] - test_df['A']).abs()
    test_df['abs_error_R'] = (test_df['FTT_R'] - test_df['R']).abs()

    # 대표 종목 정의
    kr_tickers = ['000660', '005380', '096770']
    kr_names = {
        '000660': 'SK Hynix (000660)',
        '005380': 'Hyundai Motor (005380)',
        '096770': 'SK Innovation (096770)'
    }
    
    us_tickers = ['NVDA', 'TSLA', 'KO']
    us_names = {
        'NVDA': 'Nvidia (NVDA)',
        'TSLA': 'Tesla (TSLA)',
        'KO': 'Coca-Cola (KO)'
    }

    # 마이그레이션 경로 정의 (plot 관련은 evaluation/plot, error 관련은 evaluation/error)
    artifact_base = '/Users/mireflare/.gemini/antigravity-ide/brain/49637d44-18f3-47b1-8dd0-8f8a9c2c960c/evaluation'
    scratch_base = 'evaluation'
    
    # 디렉토리 매핑
    dir_mappings = {
        'overall_average': ('error/overall_average', 'error/overall_average'),
        'korea_tickers': ('plot/korea_tickers', 'plot/korea_tickers'),
        'us_tickers': ('plot/us_tickers', 'plot/us_tickers')
    }

    for key, (s_path, a_path) in dir_mappings.items():
        os.makedirs(os.path.join(scratch_base, s_path), exist_ok=True)
        os.makedirs(os.path.join(artifact_base, a_path), exist_ok=True)

    # Matplotlib 스타일링 설정
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['grid.color'] = '#e9ecef'

    # --- helper function to save to both places ---
    def save_plot(fig, sub_dir, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(scratch_base, sub_dir, filename), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(artifact_base, sub_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {sub_dir}/{filename}")

    # -------------------------------------------------------------------------
    # 2. 전체 평균 세분화 그래프 (evaluation/error/overall_average/ 하위 8개 파일 개별 저장)
    # -------------------------------------------------------------------------
    print("=== Step 2: Generating Fine-grained Overall Average & Error Plots ===")
    kr_all = test_df[test_df['country'] == 'KR'].copy()
    us_all = test_df[test_df['country'] == 'US'].copy()

    kr_grouped = kr_all.groupby('date')[['A', 'FTT_A', 'R', 'FTT_R', 'abs_error_A', 'abs_error_R']].mean().sort_index()
    us_grouped = us_all.groupby('date')[['A', 'FTT_A', 'R', 'FTT_R', 'abs_error_A', 'abs_error_R']].mean().sort_index()

    # ① Korea Attractiveness A Mean Comparison (No X Marker)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kr_grouped.index, kr_grouped['A'], label='Actual (GT) A (Mean)', color='#0d6efd', linewidth=2.5, marker='o')
    ax.plot(kr_grouped.index, kr_grouped['FTT_A'], label='Predicted A (Mean)', color='#fd7e14', linewidth=2.5, linestyle='--')
    ax.set_title('Korea (KR) Attractiveness (A): Overall Mean Comparison', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Value')
    ax.legend(frameon=True, facecolor='white')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'overall_kr_a.png')

    # ② Korea Risk R Mean Comparison (No X Marker)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kr_grouped.index, kr_grouped['R'], label='Actual (GT) R (Mean)', color='#0d6efd', linewidth=2.5, marker='o')
    ax.plot(kr_grouped.index, kr_grouped['FTT_R'], label='Predicted R (Mean)', color='#fd7e14', linewidth=2.5, linestyle='--')
    ax.set_title('Korea (KR) Risk (R): Overall Mean Comparison', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Value')
    ax.legend(frameon=True, facecolor='white')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'overall_kr_r.png')

    # ③ United States Attractiveness A Mean Comparison (No X Marker)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(us_grouped.index, us_grouped['A'], label='Actual (GT) A (Mean)', color='#198754', linewidth=2.5, marker='o')
    ax.plot(us_grouped.index, us_grouped['FTT_A'], label='Predicted A (Mean)', color='#dc3545', linewidth=2.5, linestyle='--')
    ax.set_title('United States (US) Attractiveness (A): Overall Mean Comparison', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Value')
    ax.legend(frameon=True, facecolor='white')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'overall_us_a.png')

    # ④ United States Risk R Mean Comparison (No X Marker)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(us_grouped.index, us_grouped['R'], label='Actual (GT) R (Mean)', color='#198754', linewidth=2.5, marker='o')
    ax.plot(us_grouped.index, us_grouped['FTT_R'], label='Predicted R (Mean)', color='#dc3545', linewidth=2.5, linestyle='--')
    ax.set_title('United States (US) Risk (R): Overall Mean Comparison', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Value')
    ax.legend(frameon=True, facecolor='white')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'overall_us_r.png')

    # ⑤ Korea Attractiveness A Error (MAE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kr_grouped.index, kr_grouped['abs_error_A'], color='#e377c2', linewidth=2.5, marker='d', markersize=6)
    ax.fill_between(kr_grouped.index, kr_grouped['abs_error_A'], color='#e377c2', alpha=0.15)
    ax.set_title('Korea (KR) Attractiveness (A) Absolute Error (MAE) Trend', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Error')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'error_kr_a.png')

    # ⑥ Korea Risk R Error (MAE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kr_grouped.index, kr_grouped['abs_error_R'], color='#17becf', linewidth=2.5, marker='d', markersize=6)
    ax.fill_between(kr_grouped.index, kr_grouped['abs_error_R'], color='#17becf', alpha=0.15)
    ax.set_title('Korea (KR) Risk (R) Absolute Error (MAE) Trend', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Error')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'error_kr_r.png')

    # ⑦ United States Attractiveness A Error (MAE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(us_grouped.index, us_grouped['abs_error_A'], color='#9467bd', linewidth=2.5, marker='d', markersize=6)
    ax.fill_between(us_grouped.index, us_grouped['abs_error_A'], color='#9467bd', alpha=0.15)
    ax.set_title('United States (US) Attractiveness (A) Absolute Error (MAE) Trend', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Error')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'error_us_a.png')

    # ⑧ United States Risk R Error (MAE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(us_grouped.index, us_grouped['abs_error_R'], color='#bcbd22', linewidth=2.5, marker='d', markersize=6)
    ax.fill_between(us_grouped.index, us_grouped['abs_error_R'], color='#bcbd22', alpha=0.15)
    ax.set_title('United States (US) Risk (R) Absolute Error (MAE) Trend', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Error')
    ax.tick_params(axis='x', rotation=30)
    save_plot(fig, 'error/overall_average', 'error_us_r.png')

    # -------------------------------------------------------------------------
    # 3. 한국 3개 종목 A, R 그래프 각각 하나 (plot/korea_tickers/)
    # -------------------------------------------------------------------------
    print("=== Step 3: Generating Korea Tickers Plots ===")
    df_kr_sub = test_df[test_df['ticker'].isin(kr_tickers)].copy()

    # Korea A Plot (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, ticker in enumerate(kr_tickers):
        ax = axes[i]
        ticker_df = df_kr_sub[df_kr_sub['ticker'] == ticker].sort_values('date')
        ax.plot(ticker_df['date'], ticker_df['A'], label='Actual (GT) A', color='#0d6efd', linewidth=2.5, marker='o')
        ax.plot(ticker_df['date'], ticker_df['FTT_A'], label='Predicted A', color='#fd7e14', linewidth=2.5, linestyle='--')
        ax.set_title(kr_names[ticker], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('A Value')
        ax.legend(frameon=True, facecolor='white')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle("Korea Giants: Attractiveness (A) Time-Series Comparison", fontsize=15, fontweight='bold', y=0.98)
    save_plot(fig, 'plot/korea_tickers', 'korea_attractiveness_a.png')

    # Korea R Plot (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, ticker in enumerate(kr_tickers):
        ax = axes[i]
        ticker_df = df_kr_sub[df_kr_sub['ticker'] == ticker].sort_values('date')
        ax.plot(ticker_df['date'], ticker_df['R'], label='Actual (GT) R', color='#0d6efd', linewidth=2.5, marker='o')
        ax.plot(ticker_df['date'], ticker_df['FTT_R'], label='Predicted R', color='#fd7e14', linewidth=2.5, linestyle='--')
        ax.set_title(kr_names[ticker], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('R Value')
        ax.legend(frameon=True, facecolor='white')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle("Korea Giants: Robust IPR Risk (R) Time-Series Comparison", fontsize=15, fontweight='bold', y=0.98)
    save_plot(fig, 'plot/korea_tickers', 'korea_risk_r.png')

    # -------------------------------------------------------------------------
    # 4. 미국 3개 종목 A, R 그래프 각각 하나 (plot/us_tickers/)
    # -------------------------------------------------------------------------
    print("=== Step 4: Generating US Tickers Plots ===")
    df_us_sub = test_df[test_df['ticker'].isin(us_tickers)].copy()

    # US A Plot (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, ticker in enumerate(us_tickers):
        ax = axes[i]
        ticker_df = df_us_sub[df_us_sub['ticker'] == ticker].sort_values('date')
        ax.plot(ticker_df['date'], ticker_df['A'], label='Actual (GT) A', color='#198754', linewidth=2.5, marker='o')
        ax.plot(ticker_df['date'], ticker_df['FTT_A'], label='Predicted A', color='#dc3545', linewidth=2.5, linestyle='--')
        ax.set_title(us_names[ticker], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('A Value')
        ax.legend(frameon=True, facecolor='white')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle("United States Giants: Attractiveness (A) Time-Series Comparison", fontsize=15, fontweight='bold', y=0.98)
    save_plot(fig, 'plot/us_tickers', 'us_attractiveness_a.png')

    # US R Plot (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, ticker in enumerate(us_tickers):
        ax = axes[i]
        ticker_df = df_us_sub[df_us_sub['ticker'] == ticker].sort_values('date')
        ax.plot(ticker_df['date'], ticker_df['R'], label='Actual (GT) R', color='#198754', linewidth=2.5, marker='o')
        ax.plot(ticker_df['date'], ticker_df['FTT_R'], label='Predicted R', color='#dc3545', linewidth=2.5, linestyle='--')
        ax.set_title(us_names[ticker], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('R Value')
        ax.legend(frameon=True, facecolor='white')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle("United States Giants: Robust IPR Risk (R) Time-Series Comparison", fontsize=15, fontweight='bold', y=0.98)
    save_plot(fig, 'plot/us_tickers', 'us_risk_r.png')
    
    print("All tasks completed successfully!")

if __name__ == '__main__':
    main()
