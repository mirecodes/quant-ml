# scripts/07_run_ui.py
import sys
import os
import glob
import yaml
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration with a broad 1-column layout and collapsed sidebar
st.set_page_config(
    page_title="QuantML - Portfolio Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Style Integration: Sofia Sans, Putty-Cream, Ink Black, Signal Orange, Pointer Cursors, and Expanded Font Scale
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Sofia+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap');
    
    /* Putty-Cream Canvas Background */
    .main {
        background-color: #f3f0ee !important;
        color: #141413 !important;
        font-family: "Sofia Sans", "Inter", sans-serif !important;
        padding: 40px 60px !important;
    }
    
    /* Bold and Sleek Headings (Ink Black & Signal Orange Highlights) */
    h1, h2, h3 {
        color: #141413 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    h4, h5, h6 {
        color: #cf4500 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Expanded Typography Scale */
    p, label, span, div, li, input, select {
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        color: #141413 !important;
    }
    
    /* Slate Gray Metadata Text */
    .slate-text {
        color: #696969 !important;
        font-size: 1.0rem !important;
    }
    
    /* Eyebrow Label dot and uppercase layout */
    .eyebrow-label {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #cf4500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 12px;
    }
    
    /* Lifted Cream Premium Stadium Cards */
    .premium-card {
        background-color: #fcfbfa !important;
        border-radius: 40px !important;
        padding: 36px !important;
        border: 2.5px solid #d1cdc7 !important;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 8px 32px 0px !important;
        margin-bottom: 30px !important;
        position: relative;
        overflow: hidden;
    }
    
    /* Pure White Metric blocks popping out from the Putty-Cream Canvas */
    .stMetric {
        background-color: #ffffff !important;
        padding: 26px !important;
        border-radius: 20px !important;
        border: 2.5px solid #d1cdc7 !important;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 8px 32px 0px !important;
    }
    .stMetric label {
        color: #696969 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #141413 !important;
        font-weight: 700 !important;
        font-size: 2.3rem !important;
    }
    
    .metric-caption {
        font-size: 1.0rem;
        color: #696969;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Primary Ink Black Pill Action Buttons */
    div.stButton > button {
        background-color: #141413 !important;
        color: #f3f0ee !important;
        border-radius: 999px !important;
        border: 2px solid #141413 !important;
        padding: 12px 40px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        background-color: #cf4500 !important;
        border-color: #cf4500 !important;
    }
    
    /* Streamlit Tab Buttons - Large, Bold, Muted to Signal Orange */
    button[data-baseweb="tab"] {
        color: #696969 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        padding: 18px 40px !important;
        cursor: pointer !important;
    }
    button[aria-selected="true"] {
        color: #cf4500 !important;
        border-bottom-color: #cf4500 !important;
        border-bottom-width: 3.5px !important;
    }
    
    /* Strict Text Selection I-beam Bypass for Dropdown Hovering */
    div[data-baseweb="select"] *, 
    div[data-baseweb="popover"] *,
    .stSelectbox *, 
    .stMultiSelect *,
    div[role="combobox"] *,
    div[role="listbox"] * {
        cursor: pointer !important;
    }
    .stSelectbox label, .stMultiSelect label {
        cursor: default !important;
        font-weight: 700 !important;
        color: #cf4500 !important;
    }
    div[data-baseweb="select"] input {
        cursor: pointer !important;
    }
    
    /* Custom Pill Rank Cards list */
    .pill-rank-card {
        background-color: #ffffff;
        border-radius: 999px;
        padding: 20px 40px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 2.5px solid #d1cdc7;
        box-shadow: rgba(0, 0, 0, 0.03) 0px 4px 16px 0px;
        transition: all 0.2s ease;
    }
    .pill-rank-card:hover {
        border-color: #cf4500;
        box-shadow: rgba(207, 69, 0, 0.08) 0px 8px 24px 0px;
    }
    .pill-rank-num {
        font-weight: 700;
        color: #cf4500;
        font-size: 1.4rem;
        width: 60px;
    }
    .pill-rank-name {
        font-weight: 700;
        color: #141413;
        font-size: 1.25rem;
        flex-grow: 1;
    }
    .pill-rank-sector {
        color: #696969;
        font-size: 1.1rem;
        margin-right: 40px;
    }
    .pill-rank-val {
        font-weight: 700;
        color: #cf4500;
        font-size: 1.25rem;
        margin-right: 30px;
    }
    .pill-rank-risk {
        font-weight: 700;
        color: #141413;
        font-size: 1.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Overview Card
st.title("QuantML 다차원 포트폴리오 분석 및 매크로 지능 엔진")

st.markdown("""
    <div class="premium-card">
        <!-- Ghost Watermark Background -->
        <div style="position: absolute; right: 20px; bottom: -20px; font-size: 110px; font-weight: 700; color: #d1cdc7; opacity: 0.15; pointer-events: none; user-select: none; font-family: 'Sofia Sans', sans-serif;">QUANTML</div>
        <div class="eyebrow-label">분석 프레임워크 명세</div>
        <h2 style="margin: 0 0 12px 0;">다차원 정량적 분석 및 기계학습 모니터링 시스템</h2>
        <p style="margin: 0; color: #141413; line-height: 1.6;">
            본 시스템은 S&P 500 및 KOSPI 유니버스를 대상으로 다기간 시계열 데이터를 분석하여 매력도(A)와 위험도(R)를 추정하고,
            글로벌 거시경제 변수와 피처 데이터의 연관관계 및 기계학습 최적화 수렴 곡선을 정밀 모니터링합니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# Data load handlers
@st.cache_data
def load_predictions():
    df = pd.read_parquet('data/processed/predictions_latest.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_raw_prices():
    try:
        df = pd.read_parquet('data/processed/prices_quarterly.parquet')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_features():
    try:
        df = pd.read_parquet('data/processed/features_stock.parquet')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_macro():
    try:
        df = pd.read_parquet('data/processed/macro_us.parquet')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()

# Load Datasets
try:
    df_pred = load_predictions()
    df_raw = load_raw_prices()
    df_feat = load_features()
    df_macro = load_macro()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}. 파이프라인을 먼저 순서대로 실행해주세요.")
    st.stop()

@st.cache_data
def load_ticker_names() -> dict:
    """Themes YAML 및 Wikipedia (S&P 500) 매핑 수집"""
    names = {}
    
    # 1. processed themes YAML에서 매핑 정보 추출
    try:
        yaml_path = 'data/themes/processed/themes.yaml'
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                theme_data = yaml.safe_load(f) or {}
                tickers_metadata = theme_data.get('tickers', {})
                for ticker, info in tickers_metadata.items():
                    name = info.get('name')
                    if name and name != ticker:
                        names[ticker] = name
    except Exception as e:
        print(f"Error loading names from themes.yaml: {e}")

    # 2. Wikipedia S&P 500 크롤링
    try:
        import urllib.request
        import ssl
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=5, context=context) as response:
            tables = pd.read_html(response.read())
            sp500 = tables[0]
            sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)
            for _, row in sp500.iterrows():
                symbol = row['Symbol']
                security = row['Security']
                if symbol not in names or names[symbol] == symbol:
                    names[symbol] = security
    except Exception as e:
        print(f"Error fetching S&P 500 names from Wikipedia: {e}")

    return names

# Load names dictionary
ticker_to_name = load_ticker_names()

# Merge missing names from predictions columns
if 'ticker' in df_pred.columns and 'name' in df_pred.columns:
    for t, n in zip(df_pred['ticker'].astype(str), df_pred['name'].astype(str)):
        if n and n != t:
            if t not in ticker_to_name or ticker_to_name[t] == t:
                ticker_to_name[t] = n

# Align names column in predictions
if 'name' in df_pred.columns and 'ticker' in df_pred.columns:
    df_pred['name'] = df_pred['ticker'].astype(str).map(ticker_to_name).fillna(df_pred['name'].astype(str))

def format_ticker_label(ticker):
    name = ticker_to_name.get(ticker, "")
    if name and name != ticker:
        return f"{ticker} ({name})"
    return ticker

# --- Inline Global Filters on Putty-Cream main canvas (Sidebar completely removed) ---
st.markdown("""
    <div class="premium-card">
        <div class="eyebrow-label">글로벌 필터 설정</div>
        <h3 style="margin: 0 0 15px 0; color: #cf4500;">분석 유니버스 설정</h3>
    </div>
""", unsafe_allow_html=True)

col_filt1, col_filt2, col_filt3 = st.columns(3)
with col_filt1:
    countries = df_pred['country'].unique()
    selected_countries = st.multiselect("분석 대상 국가", countries, default=list(countries), key="global_countries")
with col_filt2:
    sectors = df_pred['sector'].unique()
    selected_sectors = st.multiselect("분석 대상 섹터", sectors, default=list(sectors), key="global_sectors")
with col_filt3:
    all_themes = sorted(list(set([t for themes in df_pred['themes'].dropna() for t in themes])))
    selected_themes = st.multiselect("글로벌 매칭 테마", all_themes, key="global_themes")

# Apply filters to extract latest prediction quarterly metrics
latest_idx = df_pred.groupby('ticker', observed=True)['date'].idxmax()
df_latest = df_pred.loc[latest_idx]

filtered_latest = df_latest[df_latest['country'].isin(selected_countries) & df_latest['sector'].isin(selected_sectors)]
if selected_themes:
    filtered_latest = filtered_latest[filtered_latest['themes'].apply(lambda t: any(x in t for x in selected_themes))]

if filtered_latest.empty:
    st.warning("선택한 필터 조건에 부합하는 종목이 없습니다.")
    st.stop()

# Build 4 main tabs without emojis
tab_macro, tab_stock, tab_portfolio, tab_training = st.tabs([
    "거시경제지표 상관관계 분석",
    "주식 각 종목 지표 분석",
    "포트폴리오 관련 데이터 및 지표 예측 등",
    "트레이닝 결과 확인"
])

# -----------------------------------------------------------------------------
# 탭 1: 거시경제지표 상관관계 분석
# -----------------------------------------------------------------------------
with tab_macro:
    st.markdown("""
        <div style="position: relative;">
            <div style="position: absolute; right: 0; top: 0; font-size: 55px; font-weight: 700; color: #d1cdc7; opacity: 0.15; user-select: none; pointer-events: none;">MACRO</div>
            <h2>글로벌 거시경제 지표 연관관계 분석</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # FRED Macro labels mapping
    MACRO_LABELS = {
        'M_INT_001': 'DFF (Fed Funds Rate)',
        'M_INT_002': '2Y Treasury Yield (DGS2)',
        'M_INT_003': '10Y Treasury Yield (DGS10)',
        'M_LIQ_002': 'M2 Money Supply (M2SL)',
        'M_INF_001': 'CPI Inflation (CPIAUCSL)',
        'M_INF_002': 'Core CPI (CPILFESL)',
        'M_ECO_004': 'ISM Mfg PMI (NAPM)',
        'M_ECO_008': 'UNRATE (Unemployment)',
        'M_SNT_001': 'VIX Volatility (VIXCLS)'
    }
    
    if not df_macro.empty:
        macro_plot_df = df_macro.rename(columns=MACRO_LABELS).sort_values('date')
        macro_cols = [c for c in macro_plot_df.columns if c != 'date']
        
        # Historical Regime Filter Integration
        st.markdown("""
            <div class="premium-card">
                <h4 style="margin:0 0 10px 0; color:#cf4500;">거시경제 역사적 국면(Historical Regime) 필터</h4>
                <p class="slate-text" style="margin:0 0 15px 0;">금융 시장의 주요 역사적 변곡점들을 기준으로 거시 지표들의 연관관계 및 변동 국면을 정밀 탐색합니다.</p>
            </div>
        """, unsafe_allow_html=True)
        
        selected_regime = st.selectbox(
            "시계열 분석 국면 선택",
            [
                "전체 기간 (Full Period)",
                "COVID-19 유동성 팽창기 (2020 ~ 2021)",
                "연준 고금리 긴축기 (2022 ~ 현재)",
                "저금리 골디락스기 (2012 ~ 2019)"
            ],
            key="macro_regime_select"
        )
        
        # Filter dates based on regime
        regime_df = macro_plot_df.copy()
        if "COVID-19" in selected_regime:
            regime_df = regime_df[(regime_df['date'] >= '2020-01-01') & (regime_df['date'] <= '2021-12-31')]
        elif "고금리 긴축기" in selected_regime:
            regime_df = regime_df[regime_df['date'] >= '2022-01-01']
        elif "골디락스기" in selected_regime:
            regime_df = regime_df[(regime_df['date'] >= '2012-01-01') & (regime_df['date'] <= '2019-12-31')]
            
        if regime_df.empty:
            st.warning("선택한 국면 영역에 유효한 데이터가 존재하지 않습니다.")
        else:
            # 1. Pearson Correlation Heatmap using Mastercard Authentic Gradation (Ink Black -> Cream -> Signal Orange)
            st.markdown("### 1. 매크로 지표 피어슨 상관관계 (Pearson Correlation Heatmap)")
            corr_matrix = regime_df[macro_cols].corr()
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[[0.0, '#141413'], [0.5, '#f3f0ee'], [1.0, '#cf4500']],
                zmin=-1.0, zmax=1.0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hoverongaps=False
            ))
            fig_heat.update_layout(
                template="plotly_white",
                title=f"거시경제 변수 간 교차 상관관계 (분석 국면: {selected_regime})",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                width=1000,
                height=600,
                font=dict(color="#141413")
            )
            st.plotly_chart(fig_heat, width='stretch')
            
            # 2. Time Series Plotting
            st.markdown("### 2. 매크로 지표 역사적 추이 (Macro Indicators Over Time)")
            selected_macro_cols = st.multiselect(
                "비교할 매크로 지표 선택",
                macro_cols,
                default=list(macro_cols[:3]),
                key="macro_cols_selection"
            )
            
            if selected_macro_cols:
                col_macro_chart, col_macro_table = st.columns([2, 1])
                with col_macro_chart:
                    use_standardized = st.checkbox("데이터 표준화(Standardize)하여 그리기 (서로 다른 단위 비교용)", key="macro_standardize_cb")
                    
                    plot_data = regime_df[['date'] + selected_macro_cols].copy()
                    if use_standardized:
                        for col in selected_macro_cols:
                            mean = plot_data[col].mean()
                            std = plot_data[col].std()
                            if std > 0:
                                plot_data[col] = (plot_data[col] - mean) / std
                                
                    fig_macro_line = px.line(
                        plot_data,
                        x='date',
                        y=selected_macro_cols,
                        title=f"거시경제 시계열 추이 (분석 국면: {selected_regime})",
                        color_discrete_sequence=['#cf4500', '#141413', '#f37338', '#9a3a0a', '#696969']
                    )
                    fig_macro_line.update_layout(
                        template="plotly_white",
                        xaxis_title="날짜",
                        yaxis_title="값 (표준화)" if use_standardized else "지표 원본 값",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#d1cdc7', tickfont=dict(color="#141413")),
                        yaxis=dict(gridcolor='#d1cdc7', tickfont=dict(color="#141413")),
                        font=dict(color="#141413")
                    )
                    st.plotly_chart(fig_macro_line, width='stretch')
                with col_macro_table:
                    st.markdown("상세 매크로 데이터")
                    st.dataframe(
                        regime_df[['date'] + selected_macro_cols].sort_values('date', ascending=False),
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.info("비교 분석할 매크로 지표를 최소 1개 이상 선택해 주세요.")
    else:
        st.warning("FRED 매크로 지표 데이터가 존재하지 않습니다. scripts/01_fetch_data.py 를 먼저 가동해 주세요.")

# -----------------------------------------------------------------------------
# 탭 2: 주식 각 종목 지표 분석
# -----------------------------------------------------------------------------
with tab_stock:
    st.markdown("""
        <div style="position: relative;">
            <div style="position: absolute; right: 0; top: 0; font-size: 55px; font-weight: 700; color: #d1cdc7; opacity: 0.15; user-select: none; pointer-events: none;">STOCKS</div>
            <h2>원본 시세 및 가공 피처 탐색기</h2>
        </div>
    """, unsafe_allow_html=True)
    
    all_tickers = sorted(df_raw['ticker'].unique()) if not df_raw.empty else []
    if all_tickers:
        selected_raw_ticker = st.selectbox(
            "조회 대상 종목 선택", 
            all_tickers, 
            key="stock_analysis_ticker",
            format_func=format_ticker_label
        )
        
        # 1. Price Candlestick & Volume dual-axis with rolling SMA layering
        st.markdown("### 1. 원본 종가 및 거래량 (Raw Historical Data)")
        raw_stock_df = df_raw[df_raw['ticker'] == selected_raw_ticker].sort_values('date').copy()
        
        if not raw_stock_df.empty:
            st.markdown("""
                <div style="background-color: transparent; padding: 10px 0; display: flex; gap: 20px;">
                    <span style="font-weight: 700; color: #cf4500;">기술적 보조선 레이어링:</span>
                </div>
            """, unsafe_allow_html=True)
            col_sma1, col_sma2 = st.columns(2)
            with col_sma1:
                show_sma4 = st.checkbox("1년 이동평균선 (SMA 4분기)", value=False, key="show_sma4")
            with col_sma2:
                show_sma12 = st.checkbox("3년 이동평균선 (SMA 12분기)", value=False, key="show_sma12")
                
            col_raw_chart, col_raw_table = st.columns([2, 1])
            with col_raw_chart:
                fig_raw = go.Figure()
                fig_raw.add_trace(go.Candlestick(
                    x=raw_stock_df['date'],
                    open=raw_stock_df['open'],
                    high=raw_stock_df['high'],
                    low=raw_stock_df['low'],
                    close=raw_stock_df['close'],
                    name="주가 캔들"
                ))
                
                # Dynamic rolling SMA computations
                if show_sma4:
                    raw_stock_df['sma_4'] = raw_stock_df['close'].rolling(window=4, min_periods=1).mean()
                    fig_raw.add_trace(go.Scatter(
                        x=raw_stock_df['date'],
                        y=raw_stock_df['sma_4'],
                        mode='lines',
                        name='SMA 4 (1Y)',
                        line=dict(color='#cf4500', width=2)
                    ))
                if show_sma12:
                    raw_stock_df['sma_12'] = raw_stock_df['close'].rolling(window=12, min_periods=1).mean()
                    fig_raw.add_trace(go.Scatter(
                        x=raw_stock_df['date'],
                        y=raw_stock_df['sma_12'],
                        mode='lines',
                        name='SMA 12 (3Y)',
                        line=dict(color='#141413', width=2)
                    ))
                    
                fig_raw.add_trace(go.Bar(
                    x=raw_stock_df['date'],
                    y=raw_stock_df['volume'],
                    name="거래량",
                    yaxis="y2",
                    marker_color="rgba(207, 69, 0, 0.25)"
                ))
                
                fig_raw.update_layout(
                    template="plotly_white",
                    title=f"{selected_raw_ticker} 분기별 주가 변동 및 거래량 추이",
                    xaxis_title="날짜",
                    yaxis=dict(title="주가 (Price)", title_font=dict(color="#141413"), tickfont=dict(color="#141413")),
                    yaxis2=dict(title="거래량 (Volume)", title_font=dict(color="#cf4500"), tickfont=dict(color="#cf4500"), overlaying="y", side="right"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#d1cdc7'),
                    legend=dict(x=0.01, y=0.99),
                    font=dict(color="#141413")
                )
                st.plotly_chart(fig_raw, width='stretch')
            with col_raw_table:
                st.markdown("원시 시세 테이블")
                st.dataframe(
                    raw_stock_df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date', ascending=False),
                    column_config={
                        "date": "날짜",
                        "open": st.column_config.NumberColumn("시가", format="%.2f"),
                        "high": st.column_config.NumberColumn("고가", format="%.2f"),
                        "low": st.column_config.NumberColumn("저가", format="%.2f"),
                        "close": st.column_config.NumberColumn("종가", format="%.2f"),
                        "volume": st.column_config.NumberColumn("거래량", format="%,d"),
                    },
                    width='stretch',
                    hide_index=True
                )
        else:
            st.info("선택 종목의 원본 주가 정보가 존재하지 않습니다.")
            
        # 2. Engineered feature metrics with scale option
        st.markdown("### 2. 가공된 시계열 피처 추이 (Processed Engineered Features)")
        feat_stock_df = df_feat[df_feat['ticker'] == selected_raw_ticker].sort_values('date').copy()
        
        if not feat_stock_df.empty:
            excluded_cols = ['ticker', 'date', 'country', 'currency', 'sector', 'name', 'themes']
            feat_cols = [c for c in feat_stock_df.columns if c not in excluded_cols and pd.api.types.is_numeric_dtype(feat_stock_df[c])]
            
            selected_features = st.multiselect(
                "시각화할 피처 선택 (다중 선택 가능)", 
                feat_cols, 
                default=[c for c in ['ret_1q', 'vol_1y', 'F_SCORE', 'QUALITY_SCORE'] if c in feat_cols],
                key="feat_explorer_select"
            )
            
            if selected_features:
                col_feat_chart, col_feat_table = st.columns([2, 1])
                with col_feat_chart:
                    use_feat_std = st.checkbox("피처 Z-Score 표준화하여 그리기 (다변량 스케일 대조용)", value=False, key="use_feat_std")
                    
                    feat_plot_data = feat_stock_df[['date'] + selected_features].copy()
                    if use_feat_std:
                        for col in selected_features:
                            f_mean = feat_plot_data[col].mean()
                            f_std = feat_plot_data[col].std()
                            if f_std > 0:
                                feat_plot_data[col] = (feat_plot_data[col] - f_mean) / f_std
                                
                    fig_feat = px.line(
                        feat_plot_data,
                        x='date',
                        y=selected_features,
                        title=f"{selected_raw_ticker} 가공 시계열 피처 추이",
                        color_discrete_sequence=['#cf4500', '#141413', '#f37338', '#9a3a0a']
                    )
                    fig_feat.update_layout(
                        template="plotly_white",
                        xaxis_title="날짜",
                        yaxis_title="표준화 값" if use_feat_std else "원본 피처 값",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#d1cdc7'),
                        yaxis=dict(gridcolor='#d1cdc7'),
                        font=dict(color="#141413")
                    )
                    st.plotly_chart(fig_feat, width='stretch')
                with col_feat_table:
                    st.markdown("피처 값 상세")
                    st.dataframe(
                        feat_stock_df[['date'] + selected_features].sort_values('date', ascending=False),
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.info("시각화할 피처를 최소 1개 이상 선택해 주세요.")
        else:
            st.info("선택 종목의 가공 피처 정보가 존재하지 않습니다. 피처 빌드(Step 2)를 실행해 주세요.")
    else:
        st.info("조회할 수 있는 종목 시세 데이터가 없습니다.")

# -----------------------------------------------------------------------------
# 탭 3: 포트폴리오 관련 데이터 및 지표 예측 등
# -----------------------------------------------------------------------------
with tab_portfolio:
    st.markdown("""
        <div style="position: relative;">
            <div style="position: absolute; right: 0; top: 0; font-size: 55px; font-weight: 700; color: #d1cdc7; opacity: 0.15; user-select: none; pointer-events: none;">PORTFOLIO</div>
            <h2>포트폴리오 예측 매핑 및 자산 배분 지능</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col_chart, col_rank = st.columns([2, 1])
    
    with col_chart:
        st.subheader("매력도(A) vs 위험도(R) 2D 포트폴리오 맵")
        
        fig = px.scatter(
            filtered_latest,
            x='R', y='A',
            color='sector',
            size='close',
            hover_data=['ticker', 'name', 'country', 'close'],
            title="중장기 포트폴리오 스펙트럼 (원 크기: 종가 반영)",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="위험도 (R - 연환산 변동성)",
            yaxis_title="매력도 (A - 기대수익 파라미터)",
            legend_title="섹터",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#d1cdc7'),
            yaxis=dict(gridcolor='#d1cdc7'),
            hoverlabel=dict(bgcolor='#ffffff', font_size=13),
            font=dict(color="#141413")
        )
        st.plotly_chart(fig, width='stretch')
     
    with col_rank:
        st.subheader("매력도 Top 10 포트폴리오 유망주")
        top_stocks = filtered_latest.sort_values(by='A', ascending=False).head(10)
        
        # Genuine Mastercard Pill Ranks rendering
        rank_idx = 1
        for _, row in top_stocks.iterrows():
            t_code = row['ticker']
            t_name = row['name']
            t_sector = row['sector']
            t_attr = row['A']
            t_risk = row['R']
            
            st.markdown(f"""
                <div class="pill-rank-card">
                    <span class="pill-rank-num">#{rank_idx}</span>
                    <span class="pill-rank-name">{t_code} ({t_name})</span>
                    <span class="pill-rank-sector">{t_sector}</span>
                    <span class="pill-rank-val">매력도: {t_attr:.2f}</span>
                    <span class="pill-rank-risk">위험도: {t_risk:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
            rank_idx += 1

    st.markdown("---")
    
    # --- Advanced Quick MVO Allocation Engine ---
    st.markdown("""
        <div class="premium-card">
            <h3 style="color:#cf4500; margin-top:0;">간이 마코위츠 평균-분산 최적화 모니터 (Quick MVO Allocation Engine)</h3>
            <p class="slate-text">FT-Transformer가 예측한 매력도(A)를 기대수익률로, 위험도(R)를 변동성으로 대입하여 목표 허용 위험 범위에 최적화된 자산 배분 비중을 도출합니다.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_mvo1, col_mvo2 = st.columns([1, 2])
    with col_mvo1:
        target_risk = st.slider("목표 위험 수준 (Target Annual Risk Standard Deviation)", 0.05, 0.40, 0.18, 0.01)
        
        mvo_df = filtered_latest.copy()
        mvo_df['expected_return'] = mvo_df['A'] * 10.0
        
        # Filter assets within target risk threshold
        eligible = mvo_df[mvo_df['R'] <= target_risk].copy()
        
        if len(eligible) >= 3:
            # Select top 5 assets by expected return
            top_eligible = eligible.sort_values(by='expected_return', ascending=False).head(5).copy()
            # Allocation weight is proportional to Sharpe ratio: (E[R] - Risk_Free_Rate) / Vol
            rf_rate = 0.03
            top_eligible['sharpe'] = (top_eligible['expected_return'] - rf_rate) / top_eligible['R']
            top_eligible['sharpe'] = top_eligible['sharpe'].clip(lower=0.01)
            
            weight_sum = top_eligible['sharpe'].sum()
            top_eligible['weight'] = top_eligible['sharpe'] / weight_sum
            
            st.success(f"목표 위험 {target_risk:.1%} 이내에서 최적 분배 자산 5종이 선정되었습니다.")
        else:
            # Fallback if too few assets satisfy the risk constraint
            top_eligible = mvo_df.sort_values(by='expected_return', ascending=False).head(5).copy()
            top_eligible['weight'] = 0.20
            st.info(f"설정하신 목표 위험 {target_risk:.1%} 이내의 종목 수가 너무 적어, 유니버스 최상위 매력 종목들로 균등 배분(20%씩)을 산출했습니다.")
            
    with col_mvo2:
        fig_donut = px.pie(
            top_eligible,
            names='ticker',
            values='weight',
            hole=0.6,
            title="위험 조정 기반 최적 포트폴리오 자산 배분 비중",
            color_discrete_sequence=['#cf4500', '#141413', '#f37338', '#9a3a0a', '#696969']
        )
        fig_donut.update_layout(
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#141413"),
            hoverlabel=dict(bgcolor='#ffffff', font_size=13)
        )
        st.plotly_chart(fig_donut, width='stretch')

    st.markdown("---")
    st.subheader("개별 종목 정밀 진단")
    
    selected_ticker = st.selectbox(
        "진단할 종목 선택", 
        sorted(filtered_latest['ticker'].unique()), 
        format_func=format_ticker_label,
        key="deepdive_ticker"
    )
    stock = filtered_latest[filtered_latest['ticker'] == selected_ticker].iloc[0]
    
    col_detail1, col_detail2, col_detail3 = st.columns(3)
    
    with col_detail1:
        A_val = stock['A']
        multiple = 5 ** A_val
        st.metric(label="FTT 추정 매력도 (A)", value=f"{A_val:.2f}")
        st.markdown(f"<div class='metric-caption'>향후 최대 5년 내 약 <b>{multiple:.1f}배</b> 주가 상승 여력 내포</div>", unsafe_allow_html=True)
    
    with col_detail2:
        R_val = stock['R']
        st.metric(label="FTT 추정 위험도 (R)", value=f"{R_val:.1%}")
        st.markdown("<div class='metric-caption'>연환산 변동성(표준편차) 기준 위험 수준</div>", unsafe_allow_html=True)
    
    with col_detail3:
        st.metric(label="최종 종가", value=f"{stock['close']:.2f}")
        st.markdown(f"<div class='metric-caption'>분기 최종 가격 (국가: {stock['country']})</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("모델별 지표 비교 대조군 (Baselines)")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.info(f"Piotroski F-Score: **{stock['C_FSCORE']:.1f} / 9.0**")
    with col_b2:
        st.info(f"Asness Quality Score: **{stock['C_QUALITY']:.2f}**")
    with col_b3:
        st.info(f"Composite Score: **{stock['ACC_COMPOSITE']:.2f}**")
        
    # --- Multi-factor comparison plot ---
    st.markdown("#### 퀀트 팩터 지표 교차 검증 (Factor Cross-Validation)")
    factor_labels = ['Piotroski F-Score (scaled)', 'Asness Quality Score', 'FTT Attractiveness (A)']
    
    val_fscore_scaled = (stock['C_FSCORE'] / 9.0) * 3.0
    val_quality = stock['C_QUALITY']
    val_attr = stock['A']
    
    fig_factors = go.Figure(data=[
        go.Bar(
            name=selected_ticker,
            x=factor_labels,
            y=[val_fscore_scaled, val_quality, val_attr],
            marker_color=['#141413', '#cf4500', '#f37338']
        )
    ])
    fig_factors.update_layout(
        template="plotly_white",
        title=f"{selected_ticker} 전통 퀀트 팩터 대 AI 모델 예측치 교차 비교",
        yaxis_title="지표 스케일 수준",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#141413"),
        yaxis=dict(gridcolor='#d1cdc7')
    )
    st.plotly_chart(fig_factors, width='stretch')

# -----------------------------------------------------------------------------
# 탭 4: 트레이닝 결과 확인
# -----------------------------------------------------------------------------
with tab_training:
    st.markdown("""
        <div style="position: relative;">
            <div style="position: absolute; right: 0; top: 0; font-size: 55px; font-weight: 700; color: #d1cdc7; opacity: 0.15; user-select: none; pointer-events: none;">METRICS</div>
            <h2>모델 학습 오차 및 아키텍처 명세</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # --- YAML Hyperparameters Specification Parsing ---
    st.markdown("### 1. 모델 아키텍처 및 하이퍼파라미터 설정 사양 (Model Settings)")
    try:
        with open('config/settings.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
            
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            st.markdown("""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; border: 2.5px solid #d1cdc7;">
                    <h5 style="color:#cf4500; margin-top:0;">Stock Encoder (LSTM)</h5>
                    <p style="margin:5px 0;">Hidden Dim: <b>{}</b></p>
                    <p style="margin:5px 0;">Layers: <b>{}</b></p>
                    <p style="margin:5px 0;">Dropout: <b>{}</b></p>
                    <p style="margin:5px 0;">Max Seq: <b>{} quarters</b></p>
                </div>
            """.format(
                cfg.get('model', {}).get('lstm_stock_hidden', '128'),
                cfg.get('model', {}).get('lstm_stock_layers', '2'),
                cfg.get('model', {}).get('lstm_stock_dropout', '0.2'),
                cfg.get('model', {}).get('lstm_stock_max_seq', '20')
            ), unsafe_allow_html=True)
        with col_hp2:
            st.markdown("""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; border: 2.5px solid #d1cdc7;">
                    <h5 style="color:#cf4500; margin-top:0;">FT-Transformer</h5>
                    <p style="margin:5px 0;">Embedding Dim (d_token): <b>{}</b></p>
                    <p style="margin:5px 0;">Attention Heads: <b>{}</b></p>
                    <p style="margin:5px 0;">Layers Count: <b>{}</b></p>
                    <p style="margin:5px 0;">FFN Factor: <b>{}</b></p>
                </div>
            """.format(
                cfg.get('model', {}).get('d_token', '192'),
                cfg.get('model', {}).get('n_heads', '8'),
                cfg.get('model', {}).get('n_layers', '4'),
                cfg.get('model', {}).get('ffn_factor', '1.333')
            ), unsafe_allow_html=True)
        with col_hp3:
            st.markdown("""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; border: 2.5px solid #d1cdc7;">
                    <h5 style="color:#cf4500; margin-top:0;">Optimizer & Training</h5>
                    <p style="margin:5px 0;">Learning Rate: <b>{}</b></p>
                    <p style="margin:5px 0;">Weight Decay: <b>{}</b></p>
                    <p style="margin:5px 0;">Batch Size: <b>{}</b></p>
                    <p style="margin:5px 0;">Max Epochs: <b>{}</b></p>
                </div>
            """.format(
                cfg.get('model', {}).get('lr', '0.0001'),
                cfg.get('model', {}).get('weight_decay', '0.01'),
                cfg.get('model', {}).get('batch_size', '64'),
                cfg.get('model', {}).get('max_epochs', '60')
            ), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"설정 파일 config/settings.yaml 로드 중 에러가 발생했습니다: {e}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Search Training Logs
    log_dirs = glob.glob('lightning_logs/version_*')
    latest_log_dir = None
    df_metrics = pd.DataFrame()
    
    if log_dirs:
        log_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_log_dir = log_dirs[-1]
        csv_path = os.path.join(latest_log_dir, 'metrics.csv')
        if os.path.exists(csv_path):
            try:
                df_raw_metrics = pd.read_csv(csv_path)
                if 'epoch' in df_raw_metrics.columns:
                    df_metrics = df_raw_metrics.groupby('epoch').agg({
                        'train_loss': 'first',
                        'train_A': 'first',
                        'train_R': 'first',
                        'val_loss': 'first',
                        'val_A': 'first',
                        'val_R': 'first'
                    }).dropna(how='all').reset_index()
            except Exception as e:
                print(f"Error reading metrics.csv: {e}")
                
    if not df_metrics.empty:
        st.markdown(f"최근 학습 세션 정보: **{latest_log_dir}**")
        st.markdown("### 2. 학습 및 검증 오차 추이 (Training vs Validation Loss)")
        col_loss_chart, col_loss_table = st.columns([2, 1])
        with col_loss_chart:
            fig_loss = px.line(
                df_metrics,
                x='epoch',
                y=['train_loss', 'val_loss'],
                title="에폭별 종합 손실 추이",
                color_discrete_sequence=['#cf4500', '#141413']
            )
            fig_loss.update_layout(
                template="plotly_white",
                xaxis_title="에폭 (Epoch)",
                yaxis_title="손실 (Loss)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#d1cdc7'),
                yaxis=dict(gridcolor='#d1cdc7'),
                font=dict(color="#141413")
            )
            st.plotly_chart(fig_loss, width='stretch')
        with col_loss_table:
            st.markdown("최종 에폭 학습 요약")
            st.dataframe(
                df_metrics[['epoch', 'train_loss', 'val_loss']].sort_values('epoch', ascending=False), 
                hide_index=True, 
                width='stretch'
            )
            
        st.markdown("### 3. 타겟별 손실 분해 (Attractiveness vs Risk Sub-Losses)")
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            cols_to_plot = [c for c in ['train_A', 'val_A'] if c in df_metrics.columns]
            if cols_to_plot:
                fig_sub_a = px.line(
                    df_metrics,
                    x='epoch',
                    y=cols_to_plot,
                    title="매력도 (A) 학습 손실 추이",
                    color_discrete_sequence=['#cf4500', '#141413']
                )
                fig_sub_a.update_layout(
                    template="plotly_white",
                    xaxis_title="에폭 (Epoch)",
                    yaxis_title="손실 (Loss)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#d1cdc7'),
                    yaxis=dict(gridcolor='#d1cdc7'),
                    font=dict(color="#141413")
                )
                st.plotly_chart(fig_sub_a, width='stretch')
        with col_sub2:
            cols_to_plot_r = [c for c in ['train_R', 'val_R'] if c in df_metrics.columns]
            if cols_to_plot_r:
                fig_sub_r = px.line(
                    df_metrics,
                    x='epoch',
                    y=cols_to_plot_r,
                    title="위험도 (R) 학습 손실 추이",
                    color_discrete_sequence=['#cf4500', '#141413']
                )
                fig_sub_r.update_layout(
                    template="plotly_white",
                    xaxis_title="에폭 (Epoch)",
                    yaxis_title="손실 (Loss)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#d1cdc7'),
                    yaxis=dict(gridcolor='#d1cdc7'),
                    font=dict(color="#141413")
                )
                st.plotly_chart(fig_sub_r, width='stretch')
    else:
        # Fallback Simulation
        st.info("실시간 기계학습 로그(metrics.csv)를 수신 대기 중입니다. 로컬 파이프라인에서 ftt 모델 학습이 실행되면 여기에 오차 감쇄선이 출력됩니다.")
        
        dummy_epochs = list(range(30))
        dummy_train = [2.2 * (0.85**i) + 0.05 for i in dummy_epochs]
        dummy_val = [2.26 * (0.86**i) + 0.08 for i in dummy_epochs]
        df_dummy = pd.DataFrame({'epoch': dummy_epochs, 'train_loss': dummy_train, 'val_loss': dummy_val})
        
        st.markdown("### 2. 학습 및 검증 오차 추이 (예시 수렴 곡선)")
        col_loss_chart, col_loss_table = st.columns([2, 1])
        with col_loss_chart:
            fig_loss = px.line(
                df_dummy,
                x='epoch',
                y=['train_loss', 'val_loss'],
                title="에폭별 종합 손실 추이 (안정적 수렴 시의 예시)",
                color_discrete_sequence=['#cf4500', '#141413']
            )
            fig_loss.update_layout(
                template="plotly_white",
                xaxis_title="에폭 (Epoch)",
                yaxis_title="손실 (Loss)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#d1cdc7'),
                yaxis=dict(gridcolor='#d1cdc7'),
                font=dict(color="#141413")
            )
            st.plotly_chart(fig_loss, width='stretch')
        with col_loss_table:
            st.markdown("최종 에폭 학습 요약")
            st.dataframe(
                df_dummy[['epoch', 'train_loss', 'val_loss']].sort_values('epoch', ascending=False).head(5), 
                hide_index=True, 
                width='stretch'
            )

    st.markdown("### 4. 학습 모델 체크포인트 (Checkpoints) 현황")
    ckpt_files = glob.glob('checkpoints/*.ckpt')
    if ckpt_files:
        ckpt_data = []
        for fpath in ckpt_files:
            fname = os.path.basename(fpath)
            fsize = os.path.getsize(fpath) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
            ckpt_data.append({
                '파일명': fname,
                '용량 (MB)': f"{fsize:.1f} MB",
                '생성 일시': mtime
            })
        df_ckpt = pd.DataFrame(ckpt_data)
        st.dataframe(df_ckpt.sort_values('파일명', ascending=False), hide_index=True, width='stretch')
    else:
        st.warning("저장된 모델 체크포인트 파일(*.ckpt)이 checkpoints/ 폴더에 존재하지 않습니다.")

# Footer
st.markdown("---")
st.caption("QuantML v6.0 | Powered by FT-Transformer (FTT) on M1 Pro GPU accelerator")
