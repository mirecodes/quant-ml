"""
src/features/fundamental/abs_values.py (v5.5)

비중 계산을 위한 절대값 컬럼 생성 및 KRW -> USD 환산 로직.
"""
import numpy as np
import pandas as pd


def append_absolute_usd_columns(
    df: pd.DataFrame,
    df_macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    종목별 절대값 컬럼을 생성하고, 통화가 KRW인 종목은 분기말 USD/KRW 환율(M_FX_001)을 적용해 USD로 환산한다.

    대상 절대값 컬럼:
      - F_GRW_rev_base         : 매출 절대값
      - F_PRF_ebitda_abs       : EBITDA 절대값
      - F_PRF_net_income_abs   : 순이익 절대값
      - F_FIN_total_assets     : 총자산 절대값

    Args:
        df       : features_stock 데이터프레임
        df_macro : features_macro 데이터프레임
    """
    df = df.copy()

    # 1. 환율 매핑 딕셔너리 구축 (date -> M_FX_001)
    rate_dict = {}
    if df_macro is not None and not df_macro.empty:
        # date를 index로 사용해 매핑 편리화
        macro_temp = df_macro.copy()
        if 'date' in macro_temp.columns:
            macro_temp = macro_temp.set_index('date')
        
        if 'M_FX_001' in macro_temp.columns:
            rate_dict = macro_temp['M_FX_001'].to_dict()

    # 2. 종목 통화 결정 (티커가 숫자 6자리이면 KRW, 그 외는 USD)
    is_krw = df['ticker'].str.match(r'^\d{6}$')
    df['currency'] = np.where(is_krw, 'KRW', 'USD')

    # 3. 절대값 컬럼 기본값 계산 (기존 synthetic fundamental 컬럼을 재활용)
    # 02_build_features.py에서 생성하는 net_income, gross_profit, total_assets, ebitda 등을 활용
    # features_stock에 rename 전후의 컬럼들이 다를 수 있으므로 안전하게 fallback/생성
    
    # 02_build_features.py에서 default로 생성하는 컬럼 명칭들
    raw_net_income = df['F_FUND_NET_INCOME'] if 'F_FUND_NET_INCOME' in df.columns else (df['net_income'] if 'net_income' in df.columns else df['close'] * 0.03)
    raw_gross_profit = df['F_FUND_GROSS_PROFIT'] if 'F_FUND_GROSS_PROFIT' in df.columns else (df['gross_profit'] if 'gross_profit' in df.columns else raw_net_income * 1.5)
    raw_total_assets = df['F_FUND_TOTAL_ASSETS'] if 'F_FUND_TOTAL_ASSETS' in df.columns else (df['total_assets'] if 'total_assets' in df.columns else raw_net_income / 0.05)
    
    # EBITDA가 없는 경우 모사 생성
    if 'F_VAL_ev_ebitda' in df.columns:
        # EV/EBITDA 피처가 있으므로, 대략적으로 ebitda_abs = market_cap / EV_EBITDA 모사
        ev_ebitda = df['F_VAL_ev_ebitda'].replace(0.0, 10.0)
        raw_ebitda = (df['market_cap'] / ev_ebitda).astype(np.float32)
    else:
        raw_ebitda = (raw_net_income * 1.2).astype(np.float32)

    # 4. 환산 적용
    # M_FX_001 (USD/KRW 환율, 예: 1300.0)
    # KRW 가격을 M_FX_001로 나누면 USD 환산 완료
    df['F_GRW_rev_base'] = raw_gross_profit.astype(np.float32)
    df['F_PRF_ebitda_abs'] = raw_ebitda.astype(np.float32)
    df['F_PRF_net_income_abs'] = raw_net_income.astype(np.float32)
    df['F_FIN_total_assets'] = raw_total_assets.astype(np.float32)

    # 각 행별로 환율 환산 수행
    def convert_to_usd(row, col_name):
        val = row[col_name]
        if row['currency'] == 'KRW':
            date_val = row['date']
            # macro에서 해당 date 환율 획득 (ECOS 연동 미완 시 fallback 1300.0)
            usd_rate = rate_dict.get(date_val, 1300.0)
            if pd.isna(usd_rate) or usd_rate <= 0:
                usd_rate = 1300.0
            return val / usd_rate
        return val

    # 절대값 컬럼들에 대해 KRW -> USD 변환 적용
    for col in ['F_GRW_rev_base', 'F_PRF_ebitda_abs', 'F_PRF_net_income_abs', 'F_FIN_total_assets']:
        # pandas apply 최적화
        df[col] = df.apply(lambda r: convert_to_usd(r, col), axis=1).astype(np.float32)

    # 불필요해진 임시 컬럼 제거
    if 'currency' in df.columns:
        df = df.drop(columns=['currency'])

    return df
