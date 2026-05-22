# scratch/dataset_analysis_helper.py
import pandas as pd
import numpy as np

def main():
    print("=== Step 1: Analyzing Raw & Processed Prices ===")
    
    # 1. 분기 가격 데이터 로드
    prices_q = pd.read_parquet('data/processed/prices_quarterly.parquet')
    prices_q['date'] = pd.to_datetime(prices_q['date'])
    
    # 한국 / 미국 분리
    kr_q = prices_q[prices_q['country'] == 'KR']
    us_q = prices_q[prices_q['country'] == 'US']
    
    print(f"[분기 가격 데이터 종합]")
    print(f"  - 전체 행 수: {len(prices_q)}개")
    print(f"  - KR 시장 : {len(kr_q)}개 행 | 유니크 티커: {kr_q['ticker'].nunique()}개 | 기간: {kr_q['date'].min().strftime('%Y-%m-%d')} ~ {kr_q['date'].max().strftime('%Y-%m-%d')}")
    print(f"  - US 시장 : {len(us_q)}개 행 | 유니크 티커: {us_q['ticker'].nunique()}개 | 기간: {us_q['date'].min().strftime('%Y-%m-%d')} ~ {us_q['date'].max().strftime('%Y-%m-%d')}")
    
    # 2. 일별 가격 데이터 로드 (용량 확인용)
    print("\n[일별 가격 데이터 종합]")
    try:
        daily_kr = pd.read_parquet('data/processed/prices_daily_kr.parquet')
        daily_kr['date'] = pd.to_datetime(daily_kr['date'])
        print(f"  - KR 일별 행 수: {len(daily_kr)}개 | 유니크 티커: {daily_kr['ticker'].nunique()}개 | 기간: {daily_kr['date'].min().strftime('%Y-%m-%d')} ~ {daily_kr['date'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  - KR 일별 분석 실패: {e}")
        
    try:
        daily_us = pd.read_parquet('data/processed/prices_daily_us.parquet')
        daily_us['date'] = pd.to_datetime(daily_us['date'])
        print(f"  - US 일별 행 수: {len(daily_us)}개 | 유니크 티커: {daily_us['ticker'].nunique()}개 | 기간: {daily_us['date'].min().strftime('%Y-%m-%d')} ~ {daily_us['date'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  - US 일별 분석 실패: {e}")

    # 3. 라벨 분석 (labels.parquet)
    print("\n=== Step 2: Analyzing Target Labels (labels.parquet) ===")
    labels = pd.read_parquet('data/processed/labels.parquet')
    labels['date'] = pd.to_datetime(labels['date'])
    
    # 가격 데이터와 머지하여 국가 정보 얻기
    labels_with_meta = labels.merge(prices_q[['ticker', 'date', 'country']], on=['ticker', 'date'], how='left')
    
    kr_l = labels_with_meta[labels_with_meta['country'] == 'KR']
    us_l = labels_with_meta[labels_with_meta['country'] == 'US']
    
    print(f"[라벨 데이터 분포]")
    print(f"  - 전체 행 수: {len(labels)}개")
    print(f"  - KR 라벨 개수: {len(kr_l)}개 (A 비결측: {kr_l['A'].notna().sum()}개, R 비결측: {kr_l['R'].notna().sum()}개)")
    print(f"  - US 라벨 개수: {len(us_l)}개 (A 비결측: {us_l['A'].notna().sum()}개, R 비결측: {us_l['R'].notna().sum()}개)")
    
    # 2.0년 하한 패치가 적용된 A 라벨과 Robust IPR R 라벨의 통계
    print("\n[라벨 통계량 - Attractiveness A]")
    print(labels_with_meta.groupby('country')['A'].describe().to_string())
    
    print("\n[라벨 통계량 - Risk R]")
    print(labels_with_meta.groupby('country')['R'].describe().to_string())
    
    # 라벨링 적용 최종 날짜
    print("\n[시장별 라벨링 유효 최종 시점]")
    for country in ['KR', 'US']:
        sub = labels_with_meta[labels_with_meta['country'] == country]
        a_last = sub[sub['A'].notna()]['date'].max()
        r_last = sub[sub['R'].notna()]['date'].max()
        print(f"  - {country} 시장 ➡️ 매력도(A) 마지막: {a_last.strftime('%Y-%m-%d') if not pd.isna(a_last) else 'N/A'} | 위험도(R) 마지막: {r_last.strftime('%Y-%m-%d') if not pd.isna(r_last) else 'N/A'}")

    # 4. 최종 학습 피처 셋 분석
    print("\n=== Step 3: Analyzing Feature Sets ===")
    features_stock = pd.read_parquet('data/processed/features_stock.parquet')
    features_macro = pd.read_parquet('data/processed/features_macro.parquet')
    
    print(f"[피처 데이터 형상]")
    print(f"  - 개별 종목 피처 (features_stock.parquet): {features_stock.shape} (행, 열) | 유니크 티커: {features_stock['ticker'].nunique()}개")
    print(f"  - 거시 매크로 피처 (features_macro.parquet): {features_macro.shape} (행, 열) | 기간: {pd.to_datetime(features_macro['date']).min().strftime('%Y-%m-%d')} ~ {pd.to_datetime(features_macro['date']).max().strftime('%Y-%m-%d')}")

if __name__ == '__main__':
    main()
