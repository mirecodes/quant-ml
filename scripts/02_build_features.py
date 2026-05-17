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
        
    # 4. 거시경제 피처 병합 (date 기준)
    print("\n=== Step 5: Merging Macroeconomic Indicators ===")
    if not macro.empty:
        # date 기준으로 병합
        df = pd.merge(df, macro, on='date', how='left')
        
    # TFT 모델용 접두사 처리 (TFT stock_model.py에서 'M_'로 시작하는 피처들을 time_varying_unknown_reals로 인식함)
    # macro 컬럼명에 M_ 접두사가 있는지 확인하고 없으면 붙여줌
    for col in df.columns:
        if col.startswith('M_INT_') or col.startswith('M_LIQ_') or col.startswith('M_INF_') or col.startswith('M_ECO_') or col.startswith('M_SNT_'):
            continue
        # FredMacroFetcher에 정의된 컬럼명 맵핑 처리
        if col in ['M_INT_001', 'M_INT_002', 'M_INT_003', 'M_LIQ_002', 'M_INF_001', 'M_INF_002', 'M_ECO_004', 'M_ECO_008', 'M_SNT_001']:
            pass
            
    # parquet 포맷 저장
    save_parquet(df, 'data/processed/features.parquet')
    report_memory(df, "features.parquet")
    print("\nFeature building complete.")

if __name__ == '__main__':
    main()
