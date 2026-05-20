# scripts/02_build_features.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.features.registry import discover_indicators
from src.utils.io import save_parquet, load_parquet, report_memory

def main():
    print("=== Step 1: Loading Processed Prices & Macro Data ===")
    prices_path = 'data/processed/prices_quarterly.parquet'
    macro_path = 'data/processed/macro_us.parquet'
    
    if not Path(prices_path).exists():
        print("Error: prices_quarterly.parquet not found. Run scripts/01_fetch_data.py first.")
        return
        
    prices = load_parquet(prices_path)
    prices = prices.dropna(subset=['close'])
    report_memory(prices, "prices")
    
    macro = pd.DataFrame()
    if Path(macro_path).exists():
        macro = load_parquet(macro_path)
        report_memory(macro, "macro")
    else:
        print("Warning: macro_us.parquet not found. Proceeding without macro features.")
        
    # 1. 횡단면 데이터 피처 추가 (Sector, Size Tier 등)
    print("\n=== Step 2: Generating Categorical Attributes (Sector, Size Tier) ===")
    df = prices.copy()
    
    # 대표적인 섹터 임의 맵핑 (속도를 위해 결정적 해시 적용)
    sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer Cyclical', 'Industrials', 'Communication Services']
    df['sector'] = df['ticker'].apply(lambda t: sectors[hash(t) % len(sectors)])
    
    # 시가총액/종가 기준 Size Tier 맵핑
    df['size_tier'] = 'Mid'
    df.loc[df['close'] > 150, 'size_tier'] = 'Large'
    df.loc[df['close'] < 20, 'size_tier'] = 'Small'
    
    # 2. 재무 데이터 모사 (Fundamental Mocking for computed indicators)
    # 실제 DART나 yfinance API 차단/에러를 대비하여 안정적인 모사 데이터 횡단면 생성
    print("\n=== Step 3: Generating Synthetic Fundamentals for Computed Indicators ===")
    np.random.seed(42)
    n_rows = len(df)
    
    df['roa'] = np.random.normal(0.05, 0.03, n_rows).astype(np.float32)
    df['cfo'] = np.random.normal(0.08, 0.04, n_rows).astype(np.float32)
    df['net_income'] = (df['close'] * np.random.uniform(0.02, 0.05, n_rows)).astype(np.float32)
    df['leverage'] = np.random.uniform(0.1, 0.8, n_rows).astype(np.float32)
    df['current_ratio'] = np.random.uniform(1.0, 3.0, n_rows).astype(np.float32)
    df['shares'] = np.random.randint(10, 100, n_rows).astype(np.int32)
    df['gross_margin'] = np.random.uniform(0.2, 0.6, n_rows).astype(np.float32)
    df['asset_turnover'] = np.random.uniform(0.5, 2.0, n_rows).astype(np.float32)
    
    df['gross_profit'] = (df['net_income'] * np.random.uniform(1.2, 2.0, n_rows)).astype(np.float32)
    df['total_assets'] = (df['net_income'] / (df['roa'] + 1e-5)).astype(np.float32)
    df['eps'] = (df['net_income'] / df['shares']).astype(np.float32)
    df['debt'] = (df['total_assets'] * df['leverage']).astype(np.float32)
    df['equity'] = (df['total_assets'] - df['debt']).astype(np.float32)
    df['dividends'] = (df['net_income'] * np.random.uniform(0.0, 0.3, n_rows)).astype(np.float32)
    
    # 2b. 테마 비중 컨텍스트 계산을 위한 밸류에이션, 수익성, 수익률 피처 생성
    df['F_VAL_pbr'] = (df['close'] / (df['equity'] / df['shares'] + 1e-5)).astype(np.float32)
    df['F_VAL_per'] = (df['close'] / (df['eps'] + 1e-5)).astype(np.float32)
    df['F_VAL_ev_ebitda'] = np.random.uniform(5.0, 15.0, n_rows).astype(np.float32)
    df['F_PRF_roe'] = (df['net_income'] / (df['equity'] + 1e-5)).astype(np.float32)
    df['F_GRW_rev_cagr'] = np.random.normal(0.05, 0.08, n_rows).astype(np.float32)
    df['C_GP_A'] = (df['gross_profit'] / (df['total_assets'] + 1e-5)).astype(np.float32)
    
    df = df.sort_values(by=['ticker', 'date'])
    df['ret_1q'] = df.groupby('ticker')['close'].pct_change(1).astype(np.float32)
    df['ret_4q'] = df.groupby('ticker')['close'].pct_change(4).astype(np.float32)
    
    # 3. 자동 등록 시스템을 통한 Computed Indicators (F-Score, Quality) 계산
    print("\n=== Step 4: Discovering & Computing Registered Indicators ===")
    indicators = discover_indicators()
    print(f"Discovered indicators: {list(indicators.keys())}")
    
    for ind_id, ind_cls in indicators.items():
        print(f"Computing indicator {ind_id}...")
        ind = ind_cls()
        # compute 함수 호출하여 결과 피처 추가
        df[ind_id] = ind.compute(df)
        # float32 변환
        df[ind_id] = df[ind_id].astype(np.float32)
        
    # 피처로 쓴 원본 모사 컬럼들은 분석 효율을 위해 F_ 접두사를 붙여 남기거나 정리
    fundamental_cols = ['roa', 'cfo', 'net_income', 'leverage', 'current_ratio', 'shares', 'gross_margin', 'asset_turnover', 'gross_profit', 'total_assets', 'eps', 'debt', 'equity', 'dividends']
    for c in fundamental_cols:
        df = df.rename(columns={c: f"F_FUND_{c.upper()}"})
        
    # 모든 결측치(NaN/Inf) 정밀 제거 및 보간
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(by=['ticker', 'date'])
    
    # 수치형 컬럼들에 대해 각 티커별 보간 및 결측치 0.0 채우기 수행
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df.groupby('ticker', observed=True)[col].ffill()
        df[col] = df.groupby('ticker', observed=True)[col].bfill()
        df[col] = df[col].fillna(0.0)

    # 4. Stock 피처만 필터링하여 저장
    # stock_cols: ticker, date, country, sector, size_tier, close, volume, market_cap + F_*, C_*, A_*
    stock_cols = ['ticker', 'date', 'country', 'sector', 'size_tier', 'close', 'volume', 'market_cap', 'ret_1q', 'ret_4q']
    stock_cols += [c for c in df.columns if c.startswith(('F_', 'C_', 'A_')) and c not in stock_cols]
    df_stock = df[stock_cols].copy()

    # parquet 포맷 저장
    save_parquet(df_stock, 'data/processed/features_stock.parquet')
    report_memory(df_stock, "features_stock.parquet")

    # 5. 거시경제 피처 분리 저장
    print("\n=== Step 5: Formatting and Saving Macroeconomic Indicators ===")
    if not macro.empty:
        macro = macro.replace([np.inf, -np.inf], np.nan)
        macro = macro.sort_values(by='date')
        
        # 수치형 거시 피처 보간
        macro_num_cols = [c for c in macro.columns if c != 'date']
        for col in macro_num_cols:
            macro[col] = macro[col].ffill()
            macro[col] = macro[col].bfill()
            macro[col] = macro[col].fillna(0.0)
            
        save_parquet(macro, 'data/processed/features_macro.parquet')
        report_memory(macro, "features_macro.parquet")
    else:
        print("Error: Macro data is empty. Cannot save features_macro.parquet.")
        
    print("\nFeature building complete.")

if __name__ == '__main__':
    main()
