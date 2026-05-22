# scratch/compare_test_set.py
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    if test_df.empty:
        print("오류: test set 데이터가 비어 있습니다. 파이프라인 학습을 먼저 완료해주세요.")
        return

    # NaN 제거 (A와 R이 존재하는 것만 비교)
    test_df = test_df.dropna(subset=['A', 'R', 'FTT_A', 'FTT_R']).copy()
    
    if len(test_df) == 0:
        print("비교할 유효 샘플이 없습니다. (NaN이거나 데이터가 부족함)")
        return

    # 오차 계산
    test_df['error_A'] = test_df['FTT_A'] - test_df['A']
    test_df['abs_error_A'] = test_df['error_A'].abs()

    test_df['error_R'] = test_df['FTT_R'] - test_df['R']
    test_df['abs_error_R'] = test_df['error_R'].abs()

    print(f"비교할 유효 Test Set 샘플 수: {len(test_df)}개")

    # 전체 오차 지표 계산
    mae_A = mean_absolute_error(test_df['A'], test_df['FTT_A'])
    rmse_A = np.sqrt(mean_squared_error(test_df['A'], test_df['FTT_A']))
    r2_A = r2_score(test_df['A'], test_df['FTT_A'])

    mae_R = mean_absolute_error(test_df['R'], test_df['FTT_R'])
    rmse_R = np.sqrt(mean_squared_error(test_df['R'], test_df['FTT_R']))
    r2_R = r2_score(test_df['R'], test_df['FTT_R'])

    print(f"\n[ Attractiveness (A) 오차 통계 ]")
    print(f"  - MAE (평균 절대 오차)   : {mae_A:.6f}")
    print(f"  - RMSE (평균 제곱 오근) : {rmse_A:.6f}")
    print(f"  - R² Score (결정계수)    : {r2_A:.6f}")

    print(f"\n[ Risk (R) 오차 통계 ]")
    print(f"  - MAE (평균 절대 오차)   : {mae_R:.6f}")
    print(f"  - RMSE (평균 제곱 오근) : {rmse_R:.6f}")
    print(f"  - R² Score (결정계수)    : {r2_R:.6f}")

    # 국가별(KR/US) 오차 분리
    for country in test_df['country'].unique():
        c_df = test_df[test_df['country'] == country]
        if c_df.empty: continue
        c_mae_A = mean_absolute_error(c_df['A'], c_df['FTT_A'])
        c_mae_R = mean_absolute_error(c_df['R'], c_df['FTT_R'])
        print(f"\n[ {country} 시장 오차 ] (샘플 수: {len(c_df)}개)")
        print(f"  - A (매력도) MAE: {c_mae_A:.6f}")
        print(f"  - R (위험도) MAE: {c_mae_R:.6f}")

    # 10개 랜덤 샘플 비교 테이블 출력
    print("\n[ 무작위 10개 샘플 상세 비교 ]")
    cols = ['ticker', 'name', 'date', 'A', 'FTT_A', 'abs_error_A', 'R', 'FTT_R', 'abs_error_R']
    sample_df = test_df.sample(min(10, len(test_df)), random_state=42)
    # 날짜 정렬 포맷팅
    sample_df['date'] = sample_df['date'].dt.strftime('%Y-%m-%d')
    print(sample_df[cols].to_string(index=False))

    # 오차가 가장 큰 샘플 Top 5
    print("\n[ A (매력도) 오차가 가장 큰 샘플 Top 5 ]")
    a_top = test_df.sort_values('abs_error_A', ascending=False).head(5).copy()
    a_top['date'] = a_top['date'].dt.strftime('%Y-%m-%d')
    print(a_top[cols].to_string(index=False))

    print("\n[ R (위험도) 오차가 가장 큰 샘플 Top 5 ]")
    r_top = test_df.sort_values('abs_error_R', ascending=False).head(5).copy()
    r_top['date'] = r_top['date'].dt.strftime('%Y-%m-%d')
    print(r_top[cols].to_string(index=False))

if __name__ == '__main__':
    main()
