# evaluation/plot/generate_representative_plots.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Step 1: Loading Predictions & Representative Tickers ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # 대표 종목 선정 (KR 3개, US 3개)
    # KR: SK Hynix (000660), Hyundai Motor (005380), SK Innovation (096770)
    # US: Nvidia (NVDA), Tesla (TSLA), Coca-Cola (KO)
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

    all_targets = kr_tickers + us_tickers
    df_sub = predictions[predictions['ticker'].isin(all_targets)].copy()
    df_sub = df_sub.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()

    # 마이그레이션 저장 경로 변경
    save_dirs = [
        '/Users/mireflare/.gemini/antigravity-ide/brain/49637d44-18f3-47b1-8dd0-8f8a9c2c960c/evaluation/plot',
        'evaluation/plot'
    ]
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    # Matplotlib 스타일링 설정
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['grid.color'] = '#e9ecef'

    # =========================================================================
    # 1. Attractiveness A Plot (2 rows x 3 columns)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=False, sharey=False)
    
    # --- Row 0: KR Tickers ---
    for i, ticker in enumerate(kr_tickers):
        ax = axes[0, i]
        ticker_df = df_sub[df_sub['ticker'] == ticker].sort_values('date')
        
        ax.plot(ticker_df['date'], ticker_df['A'], label='Actual (GT) A', 
                color='#0d6efd', linewidth=2.5, marker='o', markersize=6)
        ax.plot(ticker_df['date'], ticker_df['FTT_A'], label='Predicted A', 
                color='#fd7e14', linewidth=2.5, linestyle='--', marker='s', markersize=6)
        
        ax.set_title(f"KR: {kr_names[ticker]}", fontsize=13, fontweight='bold', pad=10, color='#212529')
        ax.set_xlabel('Date')
        ax.set_ylabel('Attractiveness (A)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- Row 1: US Tickers ---
    for i, ticker in enumerate(us_tickers):
        ax = axes[1, i]
        ticker_df = df_sub[df_sub['ticker'] == ticker].sort_values('date')
        
        ax.plot(ticker_df['date'], ticker_df['A'], label='Actual (GT) A', 
                color='#198754', linewidth=2.5, marker='o', markersize=6)
        ax.plot(ticker_df['date'], ticker_df['FTT_A'], label='Predicted A', 
                color='#dc3545', linewidth=2.5, linestyle='--', marker='s', markersize=6)
        
        ax.set_title(f"US: {us_names[ticker]}", fontsize=13, fontweight='bold', pad=10, color='#212529')
        ax.set_xlabel('Date')
        ax.set_ylabel('Attractiveness (A)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Attractiveness (A) Time-Series: Actual vs FTT-Predictor (Representative Giants)", 
                 fontsize=18, fontweight='bold', y=0.98, color='#1e293b')
    plt.tight_layout()
    
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_representative_a.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot_representative_a.png")

    # =========================================================================
    # 2. Risk R Plot (2 rows x 3 columns)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=False, sharey=False)
    
    # --- Row 0: KR Tickers ---
    for i, ticker in enumerate(kr_tickers):
        ax = axes[0, i]
        ticker_df = df_sub[df_sub['ticker'] == ticker].sort_values('date')
        
        ax.plot(ticker_df['date'], ticker_df['R'], label='Actual (GT) R', 
                color='#0d6efd', linewidth=2.5, marker='o', markersize=6)
        ax.plot(ticker_df['date'], ticker_df['FTT_R'], label='Predicted R', 
                color='#fd7e14', linewidth=2.5, linestyle='--', marker='s', markersize=6)
        
        ax.set_title(f"KR: {kr_names[ticker]}", fontsize=13, fontweight='bold', pad=10, color='#212529')
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk (R)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- Row 1: US Tickers ---
    for i, ticker in enumerate(us_tickers):
        ax = axes[1, i]
        ticker_df = df_sub[df_sub['ticker'] == ticker].sort_values('date')
        
        ax.plot(ticker_df['date'], ticker_df['R'], label='Actual (GT) R', 
                color='#198754', linewidth=2.5, marker='o', markersize=6)
        ax.plot(ticker_df['date'], ticker_df['FTT_R'], label='Predicted R', 
                color='#dc3545', linewidth=2.5, linestyle='--', marker='s', markersize=6)
        
        ax.set_title(f"US: {us_names[ticker]}", fontsize=13, fontweight='bold', pad=10, color='#212529')
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk (R)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Robust IPR Risk (R) Time-Series: Actual vs FTT-Predictor (Representative Giants)", 
                 fontsize=18, fontweight='bold', y=0.98, color='#1e293b')
    plt.tight_layout()
    
    for d in save_dirs:
        plt.savefig(os.path.join(d, 'plot_representative_r.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot_representative_r.png")
    
    print("Successfully generated representative giants plots!")

if __name__ == '__main__':
    main()
