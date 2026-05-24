# evaluation/error/generate_scatter_plots.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def main():
    print("=== Step 1: Loading Predictions & Splits ===")
    predictions = pd.read_parquet('data/processed/predictions_latest.parquet')
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Ticker split 로드
    with open('data/splits/ticker_split.json') as f:
        splits = json.load(f)
    test_tickers = set(splits['test'])

    # Test set 필터링 및 결측치 엄격 배제
    test_df = predictions[predictions['ticker'].isin(test_tickers)].copy()
    test_df = test_df.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()

    # 마이그레이션 저장 경로 정의
    base_artifact_dir = '/Users/mireflare/.gemini/antigravity-ide/brain/49637d44-18f3-47b1-8dd0-8f8a9c2c960c/evaluation/error/overall_average'
    base_scratch_dir = 'evaluation/error/overall_average'
    
    os.makedirs(base_scratch_dir, exist_ok=True)
    os.makedirs(base_artifact_dir, exist_ok=True)

    # Matplotlib 스타일링 설정
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['grid.color'] = '#e9ecef'

    # 1:1 동일 스케일 및 아웃라이어 자동 클리핑 산점도 설정 헬퍼
    def save_scatter_plot(df, actual_col, pred_col, title, filename, color_point, color_line):
        x = df[actual_col].to_numpy()
        y = df[pred_col].to_numpy()
        
        fig, ax = plt.subplots(figsize=(7.5, 7))
        
        # --- [엄밀한 아웃라이어 제거 및 1:1 동일 스케일 계산] ---
        # 실제값과 예측값의 1% ~ 99% 백분위수를 구하여 조밀 분포 영역 타겟팅
        x_min_q, x_max_q = np.percentile(x, 1), np.percentile(x, 99)
        y_min_q, y_max_q = np.percentile(y, 1), np.percentile(y, 99)
        
        # 양 변수의 최솟값과 최댓값을 대칭 결합하여 완벽한 정사각형 스케일 설계
        common_min = min(x_min_q, y_min_q)
        common_max = max(x_max_q, y_max_q)
        
        # 5% 수준의 약간의 여백 가드라인
        margin = 0.05 * (common_max - common_min)
        limit_min = common_min - margin
        limit_max = common_max + margin
        
        # 뷰포트 범위 내에 있는 포인트들만 필터링하여 회귀 분석을 수행 (아웃라이어 노이즈 배제)
        mask = (x >= limit_min) & (x <= limit_max) & (y >= limit_min) & (y <= limit_max)
        x_filtered = x[mask]
        y_filtered = y[mask]
        
        if len(x_filtered) < 10:  # 예외 가드
            x_filtered, y_filtered = x, y

        # 1. 2D 산점도 (투명도 조절로 밀도 고밀도 렌더링)
        ax.scatter(x, y, alpha=0.35, s=12, color=color_point, edgecolors='none', label='Test Data Points')
        
        # 2. 완벽 예측 대각선 (y = x)
        ax.plot([limit_min, limit_max], [limit_min, limit_max], color='#6c757d', linestyle='--', linewidth=1.5, label='Perfect Fit (y = x)')
        
        # 3. 회귀선 (필터링된 주 분포 영역 기준 피팅으로 왜곡 배제)
        slope, intercept = np.polyfit(x_filtered, y_filtered, 1)
        ax.plot(x_filtered, slope * x_filtered + intercept, color=color_line, linewidth=2.5, label=f'Model Trend (slope={slope:.3f})')
        
        # 4. 피어슨 상관계수 및 유의성 연산
        corr, p_value = pearsonr(x_filtered, y_filtered)
        
        # 통계 텍스트 박스
        stats_text = (
            f"Correlation (r) : {corr:.4f}\n"
            f"P-value         : {p_value:.2e}\n"
            f"Shown Points (N): {len(x_filtered)} / {len(x)}"
        )
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#ced4da'))

        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Actual (Ground Truth) Value', fontsize=11)
        ax.set_ylabel('Predicted Value', fontsize=11)
        ax.legend(loc='lower right', frameon=True, facecolor='white')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 축 한계 동기화 (1:1 정사각형 뷰포트 확보)
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_aspect('equal', adjustable='box')  # 1:1 Aspect Ratio 고정
        
        fig.tight_layout()
        fig.savefig(os.path.join(base_scratch_dir, filename), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(base_artifact_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Generated and Saved (Strict 1:1): {filename}")

    print("=== Step 2: Generating Standardized Scatter Plots ===")
    
    kr_df = test_df[test_df['country'] == 'KR'].copy()
    us_df = test_df[test_df['country'] == 'US'].copy()

    # ① KR Attractiveness A Scatter
    save_scatter_plot(
        df=kr_df,
        actual_col='A',
        pred_col='FTT_A',
        title='Korea (KR) Attractiveness (A): 1:1 Actual vs Pred Scatter',
        filename='scatter_kr_a.png',
        color_point='#0d6efd',
        color_line='#dc3545'
    )

    # ② KR Risk R Scatter
    save_scatter_plot(
        df=kr_df,
        actual_col='R',
        pred_col='FTT_R',
        title='Korea (KR) Risk (R): 1:1 Actual vs Pred Scatter',
        filename='scatter_kr_r.png',
        color_point='#0d6efd',
        color_line='#fd7e14'
    )

    # ③ US Attractiveness A Scatter
    save_scatter_plot(
        df=us_df,
        actual_col='A',
        pred_col='FTT_A',
        title='United States (US) Attractiveness (A): 1:1 Actual vs Pred Scatter',
        filename='scatter_us_a.png',
        color_point='#198754',
        color_line='#dc3545'
    )

    # ④ US Risk R Scatter
    save_scatter_plot(
        df=us_df,
        actual_col='R',
        pred_col='FTT_R',
        title='United States (US) Risk (R): 1:1 Actual vs Pred Scatter',
        filename='scatter_us_r.png',
        color_point='#198754',
        color_line='#fd7e14'
    )

    print("All standardized scatter plots generated successfully!")

if __name__ == '__main__':
    main()
