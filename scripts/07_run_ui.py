# scripts/07_run_ui.py
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
import json
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

# Set Streamlit Page Config
st.set_page_config(
    page_title="QuantML - Mastercard Putty-Cream Investment Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Mastercard Putty-Cream Design CSS Injection
st.markdown(
    """
    <style>
    /* Putty-Cream Canvas Background */
    .stApp {
        background-color: #F3F0EE !important;
        color: #141413 !important;
        font-family: 'Sofia Sans', 'Arial', sans-serif !important;
    }
    
    /* --- Force dark text globally on all standard text elements to avoid invisible white text on cream --- */
    .stApp, 
    .stApp p, 
    .stApp span, 
    .stApp label, 
    .stApp li, 
    .stApp h1, 
    .stApp h2, 
    .stApp h3, 
    .stApp h4, 
    .stApp h5, 
    .stApp h6,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp .stSlider label,
    .stApp .stRadio label,
    .stApp .stCheckbox label,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stMarkdownContainer"] li {
        color: #141413 !important;
    }

    /* Target standard Streamlit dataframes / tables to force dark text */
    .stApp .stDataFrame, 
    .stDataFrame div, 
    .stDataFrame span, 
    .stDataFrame table, 
    .stDataFrame th, 
    .stDataFrame td {
        color: #141413 !important;
    }

    /* Selectbox & Multiselect overrides: force white/light background and dark text */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #141413 !important;
        border-color: #E8E2DA !important;
    }
    
    /* Inside selectbox values */
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] div {
        color: #141413 !important;
    }

    /* Multiselect option pills (selected items) */
    div[data-baseweb="tag"] {
        background-color: #F3F0EE !important;
        color: #141413 !important;
        border: 1px solid #D1CDC7 !important;
    }
    div[data-baseweb="tag"] span {
        color: #141413 !important;
    }

    /* Dropdown popover list styling */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] span {
        background-color: #FFFFFF !important;
        color: #141413 !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li:hover div,
    div[data-baseweb="popover"] li:hover span {
        background-color: #E8E2DA !important;
        color: #141413 !important;
    }
    
    /* Alerts and messages (warnings, info, error) text */
    .stAlert div, 
    .stAlert p, 
    .stAlert span {
        color: #141413 !important;
    }
    
    /* Modify sidebar style to Lifted Cream */
    [data-testid="stSidebar"] {
        background-color: #FCFBFA !important;
        border-right: 1px solid #D1CDC7 !important;
    }
    
    /* Modify sidebar items to match Ink Black & Putty Cream */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #141413 !important;
        font-weight: 500 !important;
    }
    
    /* Mastercard Ink Black stadium buttons and CTAs */
    div.stButton > button {
        background-color: #141413 !important;
        color: #F3F0EE !important;
        border-radius: 20px !important;
        border: 2px solid #141413 !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        padding: 8px 24px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    div.stButton > button p, 
    div.stButton > button span {
        color: #F3F0EE !important;
    }
    
    div.stButton > button:hover {
        background-color: #CF4500 !important;
        border-color: #CF4500 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(207, 69, 0, 0.2) !important;
    }
    div.stButton > button:hover p, 
    div.stButton > button:hover span {
        color: #FFFFFF !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Streamlit tabs customization (Pill buttons style) */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #696969 !important;
        background-color: transparent !important;
        border-radius: 999px !important;
        padding: 10px 24px !important;
        margin-right: 12px !important;
        border: 1px solid transparent !important;
        transition: all 0.25s ease !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        color: #696969 !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #141413 !important;
        background-color: #E8E2DA !important;
    }
    button[data-baseweb="tab"]:hover p,
    button[data-baseweb="tab"]:hover span {
        color: #141413 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: #141413 !important;
        border-color: #141413 !important;
        box-shadow: 0 4px 12px rgba(20, 20, 19, 0.15) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
    }
    
    /* Styled container cards representing Mastercard raised panels */
    .mc-card {
        background-color: #FCFBFA;
        border: 1.5px solid #E8E2DA;
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 8px 32px 0px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .mc-card:hover {
        box-shadow: rgba(0, 0, 0, 0.08) 0px 12px 48px 0px;
    }
    
    /* Native container cards representation using st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 0px !important;
        margin-bottom: 24px !important;
        box-shadow: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FCFBFA !important;
        border: 1.5px solid #E8E2DA !important;
        border-radius: 24px !important;
        padding: 28px !important;
        box-shadow: rgba(0, 0, 0, 0.04) 0px 8px 32px 0px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        box-shadow: rgba(0, 0, 0, 0.08) 0px 12px 48px 0px !important;
    }
    
    /* Pill Rank Cards Leaderboard item */
    .pill-rank-card {
        background-color: #FFFFFF;
        border-radius: 999px;
        border: 1.5px solid #E8E2DA;
        padding: 12px 28px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: rgba(0, 0, 0, 0.03) 0px 2px 8px;
        transition: all 0.2s ease;
    }
    .pill-rank-card:hover {
        border-color: #141413;
        transform: scale(1.01);
        box-shadow: rgba(0, 0, 0, 0.06) 0px 4px 12px;
    }
    
    /* Standard Text & Headings */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 500 !important;
        color: #141413 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Sleek Section Eyebrow styling */
    .eyebrow-label {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #696969 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .eyebrow-dot {
        height: 6px;
        width: 6px;
        background-color: #CF4500;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    /* Custom Metric Elements */
    .mc-metric-val {
        font-size: 44px !important;
        font-weight: 700 !important;
        color: #CF4500 !important;
        line-height: 1.1 !important;
    }
    
    .mc-metric-sub {
        font-size: 13px !important;
        color: #696969 !important;
        font-weight: 500 !important;
        margin-top: 4px;
    }
    
    /* Beautiful divider matching Dust Taupe */
    .mc-divider {
        height: 1.5px;
        background-color: #D1CDC7;
        margin: 24px 0;
        opacity: 0.7;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
    
    /* Alert style overrides */
    .stAlert {
        border-radius: 16px !important;
        background-color: #FCFBFA !important;
        border: 1.5px solid #E8E2DA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# DATA LOADING UTILITIES (CACHED WITH FALLBACKS)
# =============================================================================

@st.cache_data
def load_predictions_data():
    path = Path("data/processed/predictions_latest.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_prices_data():
    path = Path("data/processed/prices_quarterly.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_features_stock():
    path = Path("data/processed/features_stock.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_macro_data():
    path = Path("data/processed/macro_us.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_themes_dict():
    path = Path("data/themes/processed/themes.yaml")
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

# Load datasets
predictions_df = load_predictions_data()
prices_df = load_prices_data()
features_stock_df = load_features_stock()
macro_df = load_macro_data()
themes_data = load_themes_dict()

# Error handling if essential data is missing
if predictions_df.empty:
    st.error("오류: predictions_latest.parquet 파일이 존재하지 않거나 비어 있습니다. 파이프라인(run.sh)을 먼저 실행해 주세요.")
    st.stop()

# =============================================================================
# GLOBAL SIDEBAR CONTROL PANEL
# =============================================================================

# Header Image & Logo
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <div style='display: inline-block; width: 48px; height: 32px; background: radial-gradient(circle, #CF4500 50%, transparent 50%) -16px 0/32px 32px no-repeat, radial-gradient(circle, #F37338 50%, transparent 50%) 16px 0/32px 32px no-repeat; opacity: 0.95;'></div>
        <h3 style='margin-top: 10px; color: #141413;'>QUANTML TERMINAL</h3>
        <p style='font-size: 11px; color: #696969; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;'>Mastercard Inspired UI</p>
    </div>
    <div class="mc-divider"></div>
    """,
    unsafe_allow_html=True
)

# Building clean ticker names list (Duplicate label guard)
tickers = sorted(predictions_df["ticker"].unique())
ticker_name_map = {}
for tick in tickers:
    tick_df = predictions_df[predictions_df["ticker"] == tick]
    name = tick_df["name"].iloc[0] if "name" in tick_df.columns and not pd.isna(tick_df["name"].iloc[0]) else ""
    
    # Fallback to themes_data if name is still empty
    if not name and themes_data and "tickers" in themes_data:
        name = themes_data["tickers"].get(tick, {}).get("name", "")
        
    if name and str(name) != str(tick):
        ticker_name_map[tick] = f"{tick} ({name})"
    else:
        ticker_name_map[tick] = tick

# Sidebar Ticker Selectbox
ticker_options = [ticker_name_map[t] for t in tickers]
default_ticker = "000020" if "000020" in ticker_name_map else tickers[0]
default_index = list(ticker_name_map.keys()).index(default_ticker) if default_ticker in ticker_name_map else 0

st.sidebar.markdown(
    """
    <div class="eyebrow-label"><span class="eyebrow-dot"></span>GLOBAL CONTROLS</div>
    """,
    unsafe_allow_html=True
)
selected_ticker_formatted = st.sidebar.selectbox(
    "분석 종목 선택 (Select Stock to Analyze)",
    options=ticker_options,
    index=default_index
)
selected_ticker = list(ticker_name_map.keys())[ticker_options.index(selected_ticker_formatted)]

# Global Date Filter
min_date = predictions_df["date"].min()
max_date = predictions_df["date"].max()
st.sidebar.markdown("<br>", unsafe_allow_html=True)
selected_date_range = st.sidebar.slider(
    "분석 기간 범위 (Select Date Range)",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM"
)
start_date = pd.Timestamp(selected_date_range[0])
end_date = pd.Timestamp(selected_date_range[1])

# Footer in Sidebar
st.sidebar.markdown(
    """
    <div style='position: fixed; bottom: 15px; left: 15px; font-size: 11px; color: #696969; font-weight: 500;'>
        QuantML System • 2026<br>
        Inspired by Mastercard Putty-Cream
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# MAIN INTERFACE HEADER
# =============================================================================

# Elegant editorial header with no emojis and large typography
st.markdown(
    f"""
    <div class="mc-card" style="margin-top: 10px; background-color: #FFFFFF; border-radius: 40px; padding: 36px 48px;">
        <div class="eyebrow-label"><span class="eyebrow-dot"></span>PORTFOLIO INTELLIGENCE ENGINE</div>
        <h1 style="font-size: 52px; font-weight: 500; margin: 0; line-height: 1.1; color: #141413;">QuantML Deep Investment Terminal</h1>
        <p style="font-size: 18px; color: #696969; margin: 12px 0 0 0; max-width: 800px; font-weight: 450; line-height: 1.4;">
            딥러닝 모델 FT-Transformer와 LSTM 거시경제 부호기를 결합한 다차원 자산 평가 플랫폼입니다. 
            정제된 데이터 시각화와 자산 배분 최적화 기능을 단일 캔버스에서 제공합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "거시경제지표 상관관계 분석 (Macro Correlation)",
    "주식 각 종목 지표 분석 (Stock Indicators)",
    "포트폴리오 예측 & MVO 최적화 (Predictions & Portfolio MVO)",
    "딥러닝 학습 성과 & 로그 (Training Metrics & Logs)"
])

# =============================================================================
# TAB 1: 거시경제지표 상관관계 분석 (MACRO CORRELATION)
# =============================================================================
with tab1:
    st.markdown(
        """
        <div class="mc-card">
            <div class="eyebrow-label"><span class="eyebrow-dot"></span>MACROECONOMIC INTELLIGENCE</div>
            <h3>글로벌 금융 시장 거시 변수 다차원 분석</h3>
            <p style="color: #696969; font-size: 15px;">
                거시 경제의 통화 정책, 인플레이션 흐름 및 채권 시장 리스크 간의 피어슨(Pearson) 동적 상관관계를 분석합니다. 
                특정 역사적 국면(Regime)을 필터링하여 시장 레짐 변화에 따른 금융 지표 간 동조성을 비교해 볼 수 있습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>REGIME FILTERS</div>', unsafe_allow_html=True)
            st.markdown('<h4>역사적 거시경제 국면 필터 선택</h4>', unsafe_allow_html=True)
            
            regime_options = {
                "전체 기간 (All Time)": "all",
                "COVID-19 유동성 팽창기 (2020~2021)": "covid",
                "연준 고금리 긴축기 (2022~현재)": "tightening",
                "저금리 골디락스기 (2012~2019)": "goldilocks"
            }
            
            selected_regime = st.radio(
                "시나리오별 필터 선택 (Select Macro Regime Limit) →",
                options=list(regime_options.keys()),
                index=0
            )
            
            st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
            
            # Filter macro data based on regime and slider
            df_macro_filtered = macro_df.copy()
            regime_code = regime_options[selected_regime]
            
            if regime_code == "covid":
                df_macro_filtered = df_macro_filtered[(df_macro_filtered["date"] >= "2020-01-01") & (df_macro_filtered["date"] <= "2021-12-31")]
                regime_desc = "COVID-19 대응 초유의 제로금리 및 대규모 양적완화(QE)로 글로벌 유동성이 극대화되고 인플레이션 압력이 서서히 잉태되던 국면입니다."
            elif regime_code == "tightening":
                df_macro_filtered = df_macro_filtered[df_macro_filtered["date"] >= "2022-01-01"]
                regime_desc = "러시아-우크라이나 전쟁발 인플레 충격과 연준의 공격적인 자이언트 스텝(자산 긴축 및 고금리 유지)이 이어진 통화 긴축 국면입니다."
            elif regime_code == "goldilocks":
                df_macro_filtered = df_macro_filtered[(df_macro_filtered["date"] >= "2012-01-01") & (df_macro_filtered["date"] <= "2019-12-31")]
                regime_desc = "저물가, 저금리 기조 속에서 미국 경제가 완만히 지속 성장하며 사상 최장기 강세장을 기록했던 안정적인 골디락스 국면입니다."
            else:
                df_macro_filtered = df_macro_filtered[(df_macro_filtered["date"] >= start_date) & (df_macro_filtered["date"] <= end_date)]
                regime_desc = "글로벌 전체 시계열 범위를 필터링하여 대세 거시 흐름을 통합 분석합니다."
                
            st.markdown(
                f"""
                <div style="background-color: #F3F0EE; border-radius: 16px; padding: 18px; border: 1px solid #D1CDC7;">
                    <span class="eyebrow-label" style="font-size: 11px; margin-bottom: 4px;">Regime Context</span>
                    <p style="font-size: 13.5px; color: #141413; line-height: 1.4; margin: 0;">{regime_desc}</p>
                    <div style="margin-top: 10px; font-size: 12px; color: #696969; font-weight: bold;">
                        추출된 데이터 개수: {len(df_macro_filtered)}분기 분량
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
    with col2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>PEARSON CORRELATION HEATMAP</div>', unsafe_allow_html=True)
            st.markdown('<h4>거시경제 변수 상관계수 매트릭스</h4>', unsafe_allow_html=True)
            
            if len(df_macro_filtered) > 1:
                # Drop date for correlation
                macro_cols = [c for c in df_macro_filtered.columns if c != "date"]
                
                MACRO_NAME_MAP = {
                    'M_INT_001': '금리 (DFF)',
                    'M_INT_002': '국채 2Y',
                    'M_INT_003': '국채 10Y',
                    'M_LIQ_002': '통화량 M2',
                    'M_INF_001': '소비자물가 CPI',
                    'M_INF_002': '근원물가 Core CPI',
                    'M_ECO_008': '실업률',
                    'M_SNT_001': '변동성 VIX',
                    'M_FX_001': '원·달러 환율'
                }
                
                corr_matrix = df_macro_filtered[macro_cols].corr()
                
                # Map index and columns to friendly names
                friendly_names = [MACRO_NAME_MAP.get(c, c) for c in macro_cols]
                
                # Draw beautiful customized Plotly Heatmap
                fig_heat = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=friendly_names,
                    y=friendly_names,
                    colorscale=[
                        [0.0, '#141413'],   # Ink Black for -1.0
                        [0.5, '#F3F0EE'],   # Putty Cream for 0.0
                        [1.0, '#CF4500']    # Signal Orange for +1.0
                    ],
                    zmin=-1.0,
                    zmax=1.0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    hoverongaps=False
                ))
                
                fig_heat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=10, b=40),
                    height=380,
                    yaxis=dict(autorange='reversed'),
                    font=dict(color='#141413')
                )
                
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("선택한 국면에 필터링된 데이터가 부족합니다.")

    # ── Macro Standardized Time Series & Raw Data Table ──────────────────────────
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>HISTORICAL MACRO TIME SERIES</div>', unsafe_allow_html=True)
        st.markdown('<h4>거시 지표 시계열 추이 모니터링</h4>', unsafe_allow_html=True)
        
        if not macro_df.empty:
            # Standardize toggle
            standardize = st.checkbox("동적 표준화(Z-Score Scaling) 적용 → 측정단위가 다른 다변량 지표의 흐름 대비용", value=True)
            
            all_macro_cols = [c for c in macro_df.columns if c != "date"]
            selected_macro_cols = st.multiselect(
                "시각화할 매크로 변수 다중 선택 (Select Indicators) →",
                options=all_macro_cols,
                default=['M_INT_001', 'M_INT_003', 'M_INF_001', 'M_SNT_001'],
                format_func=lambda x: f"{x} ({MACRO_NAME_MAP.get(x, x)})"
            )
            
            if selected_macro_cols:
                df_plot = macro_df.copy()
                df_plot = df_plot[(df_plot["date"] >= start_date) & (df_plot["date"] <= end_date)]
                
                if standardize:
                    for col in selected_macro_cols:
                        mean_val = df_plot[col].mean()
                        std_val = df_plot[col].std()
                        if std_val > 0:
                            df_plot[col] = (df_plot[col] - mean_val) / std_val
                        else:
                            df_plot[col] = 0.0
                
                fig_macro_ts = go.Figure()
                
                # Harmonious color palette for lines (Signal orange, light signal orange, ink black, slate gray)
                line_colors = ['#CF4500', '#F37338', '#141413', '#696969', '#9A3A0A', '#3860BE', '#9A9A9A']
                for idx, col in enumerate(selected_macro_cols):
                    color = line_colors[idx % len(line_colors)]
                    fig_macro_ts.add_trace(go.Scatter(
                        x=df_plot["date"],
                        y=df_plot[col],
                        mode='lines+markers',
                        name=MACRO_NAME_MAP.get(col, col),
                        line=dict(color=color, width=2.5),
                        marker=dict(size=6)
                    ))
                    
                fig_macro_ts.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                    xaxis=dict(showgrid=True, gridcolor='#E8E2DA'),
                    yaxis=dict(showgrid=True, gridcolor='#E8E2DA', title="Z-Score 스케일" if standardize else "원본 지표 스케일"),
                    hovermode='x unified',
                    font=dict(color='#141413')
                )
                
                st.plotly_chart(fig_macro_ts, use_container_width=True)
                
                st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>RAW MACRO DATA TABLE</div>', unsafe_allow_html=True)
                st.markdown('<h4>거시 지표 분기별 원본 데이터 수치</h4>', unsafe_allow_html=True)
                
                # Display sorted raw data table
                df_table = macro_df[['date'] + selected_macro_cols].copy()
                df_table = df_table[(df_table["date"] >= start_date) & (df_table["date"] <= end_date)].sort_values('date', ascending=False)
                df_table["date"] = df_table["date"].dt.strftime('%Y-%m-%d')
                st.dataframe(df_table, use_container_width=True, hide_index=True)
            else:
                st.warning("상단에서 시각화할 매크로 변수를 1개 이상 선택해 주세요.")
        else:
            st.warning("시각화할 거시경제 지표 데이터(macro_us.parquet)가 존재하지 않습니다.")

# =============================================================================
# TAB 2: 주식 각 종목 지표 분석 (INDIVIDUAL STOCK INDICATORS)
# =============================================================================
with tab2:
    # Title showing selected ticker name
    stock_full_name = ticker_name_map.get(selected_ticker, selected_ticker)
    
    st.markdown(
        f"""
        <div class="mc-card">
            <div class="eyebrow-label"><span class="eyebrow-dot"></span>MICROSECURITY INTELLIGENCE</div>
            <h3>개별 종목 정밀 시세 및 퀀트 팩터 분석 → <span style="color: #CF4500;">{stock_full_name}</span></h3>
            <p style="color: #696969; font-size: 15px;">
                선택한 기업의 분기별 정밀 가격 변동 추이와 매매 강도(거래량)의 이중축 시각화를 제공하며, 
                다변량 재무/가공 피처의 장기 추이를 관측할 수 있는 탐색형 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 1. Price Candlestick & Volume Chart
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>CANDLESTICK & VOLUME DUAL AXIS CHART</div>', unsafe_allow_html=True)
        st.markdown(f'<h4>{selected_ticker} 가격 캔들스틱 및 분기 거래량 추이</h4>', unsafe_allow_html=True)
        
        if not prices_df.empty:
            df_stock_price = prices_df[prices_df["ticker"] == selected_ticker].sort_values("date").copy()
            df_stock_price = df_stock_price[(df_stock_price["date"] >= start_date) & (df_stock_price["date"] <= end_date)]
            
            if not df_stock_price.empty:
                # Control checkboxes for SMAs
                col_sma1, col_sma2 = st.columns(2)
                with col_sma1:
                    show_sma4 = st.checkbox("1년 이동평균선(SMA 4분기) 레이어 중첩 표시", value=True)
                with col_sma2:
                    show_sma12 = st.checkbox("3년 이동평균선(SMA 12분기) 레이어 중첩 표시", value=False)
                    
                fig_candle = go.Figure()
                
                # 1. Candlestick
                fig_candle.add_trace(go.Candlestick(
                    x=df_stock_price["date"],
                    open=df_stock_price["open"],
                    high=df_stock_price["high"],
                    low=df_stock_price["low"],
                    close=df_stock_price["close"],
                    name="주가 캔들스틱",
                    increasing_line_color='#CF4500',  # Mastercard Signal Orange for rise
                    decreasing_line_color='#141413',  # Mastercard Ink Black for fall
                    increasing_fillcolor='#CF4500',
                    decreasing_fillcolor='#141413'
                ))
                
                # 2. SMA 4 (1 Year)
                if show_sma4:
                    df_stock_price["SMA4"] = df_stock_price["close"].rolling(4).mean()
                    fig_candle.add_trace(go.Scatter(
                        x=df_stock_price["date"],
                        y=df_stock_price["SMA4"],
                        line=dict(color='#F37338', width=2),  # Light Signal Orange
                        name="1년 이평선 (SMA 4)"
                    ))
                    
                # 3. SMA 12 (3 Years)
                if show_sma12:
                    df_stock_price["SMA12"] = df_stock_price["close"].rolling(12).mean()
                    fig_candle.add_trace(go.Scatter(
                        x=df_stock_price["date"],
                        y=df_stock_price["SMA12"],
                        line=dict(color='#3860BE', width=2, dash='dash'),  # Deep blue dashed
                        name="3년 이평선 (SMA 12)"
                    ))
                    
                # 4. Volume Bar on secondary axis
                fig_candle.add_trace(go.Bar(
                    x=df_stock_price["date"],
                    y=df_stock_price["volume"],
                    name="거래량",
                    yaxis="y2",
                    marker_color="#D1CDC7",
                    opacity=0.3
                ))
                
                # Formatting layout
                fig_candle.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=450,
                    xaxis=dict(rangeslider=dict(visible=False), showgrid=True, gridcolor='#E8E2DA'),
                    yaxis=dict(title="주가 (Price)", showgrid=True, gridcolor='#E8E2DA'),
                    yaxis2=dict(
                        title="거래량 (Volume)",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(color='#141413')
                )
                
                st.plotly_chart(fig_candle, use_container_width=True)
            else:
                st.warning("해당 분석 범위 내에 시세 데이터가 존재하지 않습니다.")
        else:
            st.warning("prices_quarterly.parquet 가격 데이터 파일이 존재하지 않습니다.")
    
    # 2. Engineered Feature Explorer
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>ENGINEERED FEATURE EXPLORER</div>', unsafe_allow_html=True)
        st.markdown(f'<h4>{selected_ticker} 다변량 계량 재무/가공 피처 탐색</h4>', unsafe_allow_html=True)
        
        if not features_stock_df.empty:
            df_stock_features = features_stock_df[features_stock_df["ticker"] == selected_ticker].sort_values("date").copy()
            df_stock_features = df_stock_features[(df_stock_features["date"] >= start_date) & (df_stock_features["date"] <= end_date)]
            
            if not df_stock_features.empty:
                exclude_cols = ['ticker', 'date', 'country', 'sector', 'size_tier', 'open', 'high', 'low', 'close', 'volume', 'market_cap']
                available_features = [c for c in df_stock_features.columns if c not in exclude_cols]
                
                FEATURE_DICT_MAP = {
                    'ret_1q': '1Q Return (1분기 수익률)',
                    'ret_4q': '4Q Return (1년 수익률)',
                    'F_FUND_ROA': 'ROA (자산수익률)',
                    'F_FUND_CFO': 'CFO (영업현금흐름)',
                    'F_FUND_NET_INCOME': 'Net Income (당기순이익)',
                    'F_FUND_LEVERAGE': 'Leverage (레버리지 비율)',
                    'F_FUND_CURRENT_RATIO': 'Current Ratio (유동비율)',
                    'F_VAL_pbr': 'PBR (주가순자산비율)',
                    'F_VAL_per': 'PER (주가수익비율)',
                    'F_VAL_ev_ebitda': 'EV/EBITDA',
                    'F_FUND_SHARES': '유통주식수',
                    'F_FUND_GROSS_MARGIN': 'Gross Margin (매출총이익률)',
                    'F_FUND_ASSET_TURNOVER': 'Asset Turnover (자산회전율)',
                    'F_FUND_GROSS_PROFIT': 'Gross Profit (매출총이익)',
                    'F_FUND_TOTAL_ASSETS': 'Total Assets (총자산)',
                    'F_FUND_EPS': 'EPS (주당순이익)',
                    'F_FUND_DEBT': 'Total Debt (총부채)',
                    'F_FUND_EQUITY': 'Total Equity (총자본)',
                    'F_FUND_DIVIDENDS': 'Dividends (배당금)'
                }
                
                # Select features to plot
                selected_features = st.multiselect(
                    "관측할 파생 피처 다중 선택 (Select Features to Plot) →",
                    options=available_features,
                    default=[f for f in ['ret_1q', 'F_FUND_ROA', 'F_VAL_pbr'] if f in available_features],
                    format_func=lambda x: f"{x} ({FEATURE_DICT_MAP.get(x, x)})"
                )
                
                if selected_features:
                    col_scale_f = st.checkbox("피처 동적 표준화(Z-Score Scaling) 레이아웃 적용", value=True)
                    
                    df_f_plot = df_stock_features.copy()
                    if col_scale_f:
                        for col in selected_features:
                            mean_f = df_f_plot[col].mean()
                            std_f = df_f_plot[col].std()
                            if std_f > 0:
                                df_f_plot[col] = (df_f_plot[col] - mean_f) / std_f
                            else:
                                df_f_plot[col] = 0.0
                                
                    fig_features = go.Figure()
                    feature_colors = ['#CF4500', '#F37338', '#141413', '#696969', '#3860BE', '#9A3A0A', '#9A9A9A']
                    
                    for idx, col in enumerate(selected_features):
                        color = feature_colors[idx % len(feature_colors)]
                        fig_features.add_trace(go.Scatter(
                            x=df_f_plot["date"],
                            y=df_f_plot[col],
                            mode="lines+markers",
                            name=FEATURE_DICT_MAP.get(col, col),
                            line=dict(color=color, width=2.5),
                            marker=dict(size=6)
                        ))
                        
                    fig_features.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        xaxis=dict(showgrid=True, gridcolor='#E8E2DA'),
                        yaxis=dict(showgrid=True, gridcolor='#E8E2DA', title="Z-Score 스케일" if col_scale_f else "피처 원본 값"),
                        hovermode="x unified",
                        font=dict(color='#141413')
                    )
                    
                    st.plotly_chart(fig_features, use_container_width=True)
                else:
                    st.warning("시각화할 파생 피처를 최소 1개 이상 선택해 주세요.")
            else:
                st.warning("선택한 분석 기간 내에 피처 데이터가 존재하지 않습니다.")
        else:
            st.warning("features_stock.parquet 피처 데이터 파일이 존재하지 않습니다.")

# =============================================================================
# TAB 3: 포트폴리오 관련 데이터 및 지표 예측 등 (PORTFOLIO & MVO)
# =============================================================================
with tab3:
    # Attractiveness expectation vs estimated Volatility (latest date)
    latest_date_pred = predictions_df["date"].max()
    df_latest_pred = predictions_df[predictions_df["date"] == latest_date_pred].copy()
    
    st.markdown(
        f"""
        <div class="mc-card">
            <div class="eyebrow-label"><span class="eyebrow-dot"></span>DEEP PREDICTIVE PORTFOLIO MANAGEMENT</div>
            <h3>딥러닝 FTT 추정 매력도 대비 위험도 포트폴리오 매핑</h3>
            <p style="color: #696969; font-size: 15px;">
                FT-Transformer 모델이 추정한 주가 기대 수익성(Attractiveness A, Y축)과 추정 변동성(Risk R, X축)을 다차원 공간 상에 맵핑합니다.
                기준일자: <b>{latest_date_pred.strftime('%Y-%m-%d')}</b> (최신 분기 스냅샷 데이터 기준)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_map, col_leader = st.columns([3, 2])
    
    with col_map:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>ATTRACTIVENESS VS RISK 2D MAP</div>', unsafe_allow_html=True)
            st.markdown('<h4>전 종목 위험 대비 매력도 스펙트럼</h4>', unsafe_allow_html=True)
            
            if not df_latest_pred.empty:
                # Highlight selected ticker
                df_latest_pred["highlight"] = "전체 유니버스 기업"
                df_latest_pred.loc[df_latest_pred["ticker"] == selected_ticker, "highlight"] = f"선택 기업 ({selected_ticker})"
                
                # Map columns to FTT expected ones
                # Use 'FTT_R' for Risk, 'FTT_A' for Attractiveness
                fig_scatter = px.scatter(
                    df_latest_pred,
                    x="FTT_R",
                    y="FTT_A",
                    color="highlight",
                    color_discrete_map={
                        "전체 유니버스 기업": "#696969",
                        f"선택 기업 ({selected_ticker})": "#CF4500"
                    },
                    hover_data=["ticker", "name", "sector", "close"],
                    labels={"FTT_R": "FTT 모델 추정 위험도 (Risk R, 변동성)", "FTT_A": "FTT 모델 예측 매력도 (Attractiveness A, 기대수익)"}
                )
                
                fig_scatter.update_traces(
                    marker=dict(
                        size=df_latest_pred["highlight"].apply(lambda x: 18 if "선택" in x else 8),
                        opacity=df_latest_pred["highlight"].apply(lambda x: 1.0 if "선택" in x else 0.5),
                        line=dict(width=1.5, color='#141413')
                    )
                )
                
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                    xaxis=dict(showgrid=True, gridcolor='#E8E2DA'),
                    yaxis=dict(showgrid=True, gridcolor='#E8E2DA'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    font=dict(color='#141413')
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("예측 데이터에 최신 스냅샷 일자의 데이터가 존재하지 않습니다.")
        
    with col_leader:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>TOP 10 ATTRACTIVENESS LEADERBOARD</div>', unsafe_allow_html=True)
            st.markdown('<h4>알약(Pill) 디자인 유망주 랭킹</h4>', unsafe_allow_html=True)
            
            # Pull top 10 tickers based on FTT_A
            if not df_latest_pred.empty:
                df_top10 = df_latest_pred.sort_values(by="FTT_A", ascending=False).head(10)
                
                for idx, row in df_top10.iterrows():
                    tick = row["ticker"]
                    name_str = row["name"] if not pd.isna(row["name"]) else ""
                    score = row["FTT_A"]
                    
                    # Truncate long names
                    if len(name_str) > 8:
                        name_str = name_str[:7] + ".."
                    name_label = f"({name_str})" if name_str else ""
                    
                    st.markdown(
                        f"""
                        <div class="pill-rank-card">
                            <span style="font-weight: 700; color: #CF4500; font-size: 13.5px; width: 45px;">• {row['FTT_A']:.4f}</span>
                            <span style="font-weight: 600; color: #141413; font-size: 14px;">{tick}</span>
                            <span style="color: #696969; font-size: 13px; text-align: left; flex-grow: 1; margin-left: 12px;">{name_label}</span>
                            <span style="font-size: 11px; background-color: #F3F0EE; color: #141413; padding: 3px 10px; border-radius: 999px; font-weight: 700;">Rank {len(df_top10) - df_top10['FTT_A'].gt(score).sum()}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown('<div class="mc-divider" style="margin: 16px 0;"></div>', unsafe_allow_html=True)
                st.markdown('<h4>전체 종목 지표 (Full Metrics Data)</h4>', unsafe_allow_html=True)
                
                # Full Metrics Table
                display_cols = ["ticker", "name", "sector", "FTT_A", "FTT_R"]
                if "C_FSCORE" in df_latest_pred.columns: display_cols.append("C_FSCORE")
                if "C_QUALITY" in df_latest_pred.columns: display_cols.append("C_QUALITY")
                
                df_display = df_latest_pred[[c for c in display_cols if c in df_latest_pred.columns]].copy()
                
                # Fallback mapping for names if missing
                if "name" in df_display.columns:
                    df_display["name"] = df_display.apply(
                        lambda r: themes_data.get("tickers", {}).get(r["ticker"], {}).get("name", "") 
                        if pd.isna(r["name"]) or r["name"] == "" else r["name"], axis=1
                    )
                
                # Formatting options for the dataframe
                column_config = {
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "name": st.column_config.TextColumn("Name"),
                    "sector": st.column_config.TextColumn("Sector"),
                    "FTT_A": st.column_config.NumberColumn("Attractiveness (A)", format="%.4f"),
                    "FTT_R": st.column_config.NumberColumn("Risk (R)", format="%.2f"),
                    "C_FSCORE": st.column_config.NumberColumn("F-Score", format="%d"),
                    "C_QUALITY": st.column_config.NumberColumn("Quality", format="%.2f")
                }
                
                st.dataframe(
                    df_display.sort_values(by="FTT_A", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                    height=300
                )
            else:
                st.warning("유망주 랭킹을 렌더링할 데이터가 부족합니다.")
        
    st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
    
    # 2. Markowitz MVO Engine with target risk slider and Donut chart
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>MEAN-VARIANCE OPTIMIZATION</div>', unsafe_allow_html=True)
        st.markdown('<h3>간이 마코위츠(Markowitz) 평균-분산 포트폴리오 최적화</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #696969; font-size: 14.5px; margin-bottom: 24px;">
                FT-Transformer 예측치(매력도 A → 기대 수익성 대용)와 위험도(Risk R → 개별 변동성 대용), 그리고 <b>실제 분기 가격 이력에서 산출된 공분산(Covariance)</b>을 
                바탕으로 목표 위험 수준 대비 최대 효율을 도출하는 MVO 모델입니다.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        if not df_latest_pred.empty and not prices_df.empty:
            # User multi-select for assets in MVO portfolio (Default to top 5 from leader board)
            top5_tickers = df_latest_pred.sort_values(by="FTT_A", ascending=False).head(5)["ticker"].tolist()
            
            mvo_tickers_selected = st.multiselect(
                "MVO 자산 포트폴리오 유니버스 구성 종목 선택 →",
                options=tickers,
                default=top5_tickers,
                format_func=lambda x: ticker_name_map.get(x, x)
            )
            
            if len(mvo_tickers_selected) >= 2:
                # Load expected returns (A) and risks (R) for selected assets
                df_mvo_assets = df_latest_pred[df_latest_pred["ticker"].isin(mvo_tickers_selected)].copy()
                
                # Historical covariance calculation using quarterly prices
                df_mvo_prices = prices_df[prices_df["ticker"].isin(mvo_tickers_selected)].copy()
                
                # Pivot by date
                df_mvo_pivot = df_mvo_prices.pivot(index="date", columns="ticker", values="close").sort_index()
                # Calculate quarterly returns
                df_mvo_returns = df_mvo_pivot.pct_change().dropna(how='all')
                
                # Compute covariance matrix (annualized by multiplying by 4)
                cov_matrix_mvo = df_mvo_returns.cov() * 4
                
                # Align lists of assets
                asset_tickers = cov_matrix_mvo.columns.tolist()
                # Make sure df_mvo_assets aligns with asset_tickers
                df_mvo_assets = df_mvo_assets.set_index("ticker").reindex(asset_tickers).reset_index()
                
                # Expected returns A and Risks R (annualized)
                expected_returns = df_mvo_assets["FTT_A"].values
                individual_risks = df_mvo_assets["FTT_R"].values / 100.0  # scale appropriately
                
                # Ensure covariance matrix is positive-semidefinite and valid
                if cov_matrix_mvo.isnull().any().any() or len(df_mvo_returns) < 2:
                    # Fallback: diagonal matrix of individual estimated risks
                    cov_matrix_mvo = pd.DataFrame(
                        np.diag(individual_risks ** 2),
                        index=asset_tickers,
                        columns=asset_tickers
                    )
                    
                cov_np = cov_matrix_mvo.values
                
                # Find min and max portfolios risk to set slider limits
                # Single asset risks as boundaries
                min_possible_risk = float(np.sqrt(np.min(np.diag(cov_np))))
                max_possible_risk = float(np.sqrt(np.max(np.diag(cov_np))))
                
                # Safe boundary check
                if min_possible_risk >= max_possible_risk:
                    max_possible_risk = min_possible_risk + 0.1
                    
                st.markdown("<br>", unsafe_allow_html=True)
                col_slider, col_donut = st.columns([1, 1])
                
                with col_slider:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>RISK BUDGET SLIDER</div>', unsafe_allow_html=True)
                        st.markdown('<h4>목표 위험 예산(Target Volatility) 조절</h4>', unsafe_allow_html=True)
                        
                        target_risk = st.slider(
                            "목표 연환산 변동성 (%) → 우측 비중 도넛 차트가 동적 연동됩니다",
                            min_value=float(min_possible_risk * 100),
                            max_value=float(max_possible_risk * 100),
                            value=float((min_possible_risk + max_possible_risk) / 2 * 100),
                            step=0.1
                        ) / 100.0
                        
                        # MVO Optimization using Scipy
                        # Minimize portfolio negative return: -w^T * E_r
                        # subject to w_i >= 0, sum(w) = 1
                        # and w^T * Cov * w <= target_risk^2
                        num_assets = len(asset_tickers)
                        
                        def obj_func(weights):
                            return -np.dot(weights, expected_returns)
                            
                        cons = [
                            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                            {'type': 'ineq', 'fun': lambda w: target_risk**2 - np.dot(w.T, np.dot(cov_np, w))}
                        ]
                        
                        bounds = [(0.0, 1.0) for _ in range(num_assets)]
                        initial_weights = np.ones(num_assets) / num_assets
                        
                        res_opt = minimize(obj_func, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
                        
                        if res_opt.success:
                            opt_weights = np.clip(res_opt.x, 0.0, 1.0)
                            opt_weights = opt_weights / np.sum(opt_weights) # normalize
                            opt_status = "최적 배분 달성 (Optimal Allocation Achieved)"
                        else:
                            # Fallback to Minimum Variance Portfolio
                            def min_var_obj(weights):
                                return np.dot(weights.T, np.dot(cov_np, weights))
                            cons_mv = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
                            res_mv = minimize(min_var_obj, initial_weights, method='SLSQP', bounds=bounds, constraints=cons_mv)
                            opt_weights = np.clip(res_mv.x, 0.0, 1.0)
                            opt_weights = opt_weights / np.sum(opt_weights)
                            opt_status = "허용 오차 범위 초과로 인한 최소분산 포트폴리오(MVP) 강제 대체"
                            
                        # Compute portfolio expected stats
                        port_return = np.dot(opt_weights, expected_returns)
                        port_risk = np.sqrt(np.dot(opt_weights.T, np.dot(cov_np, opt_weights)))
                        
                        st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
                        st.markdown(
                            f"""
                            <span class="eyebrow-label" style="font-size: 11px;">Optimizer Status</span>
                            <p style="font-weight: 700; color: #141413; font-size: 14.5px;">{opt_status}</p>
                            <div class="mc-divider" style="margin: 12px 0;"></div>
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <span class="eyebrow-label" style="font-size: 10px; margin-bottom: 2px;">Portfolio Expected Return</span>
                                    <div style="font-size: 26px; font-weight: 700; color: #CF4500;">{port_return:.4f}</div>
                                </div>
                                <div>
                                    <span class="eyebrow-label" style="font-size: 10px; margin-bottom: 2px;">Portfolio Realized Risk</span>
                                    <div style="font-size: 26px; font-weight: 700; color: #141413;">{port_risk*100:.2f}%</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                with col_donut:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>OPTIMAL PORTFOLIO WEIGHTS</div>', unsafe_allow_html=True)
                        st.markdown('<h4>MVO 자산 최적 배분 비중</h4>', unsafe_allow_html=True)
                        
                        # Show Donut chart
                        asset_labels_formatted = [ticker_name_map.get(t, t) for t in asset_tickers]
                        
                        fig_donut = go.Figure(data=[go.Pie(
                            labels=asset_labels_formatted,
                            values=opt_weights,
                            hole=.45,
                            marker=dict(colors=['#141413', '#CF4500', '#F37338', '#9A3A0A', '#696969', '#E8E2DA']),
                            textinfo='percent+label',
                            textposition='inside',
                            insidetextorientation='radial'
                        )])
                        
                        fig_donut.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=280,
                            showlegend=False,
                            font=dict(color='#141413')
                        )
                        
                        st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.warning("포트폴리오 구성을 위해 2개 이상의 종목을 상단에서 다중 선택해 주세요.")
        else:
            st.warning("포트폴리오 최적화 엔진 구동에 필요한 시세 및 예측 데이터가 부족합니다.")
    
    st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
    
    # 3. Individual stock deep dive and validation card
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>DEEP DIVE STOCK DIAGNOSIS</div>', unsafe_allow_html=True)
        st.markdown(f'<h3>{stock_full_name} 딥러닝 예측 및 회계 지표 교차 검증</h3>', unsafe_allow_html=True)
        
        if not df_latest_pred.empty:
            df_stock_diag = df_latest_pred[df_latest_pred["ticker"] == selected_ticker]
            
            if not df_stock_diag.empty:
                row_diag = df_stock_diag.iloc[0]
                
                # Calculating Multiplier (5^A)
                attractiveness_score = float(row_diag["FTT_A"])
                multiplier = 5.0 ** attractiveness_score
                
                # Composite and individual target metrics
                risk_score = float(row_diag["FTT_R"])
                composite_score = float(row_diag["ACC_COMPOSITE"]) if "ACC_COMPOSITE" in df_latest_pred.columns else 0.0
                
                col_mul, col_scores = st.columns([1, 2])
                
                with col_mul:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow-label" style="justify-content: center;"><span class="eyebrow-dot"></span>5-YEAR EXPECTED MULTIPLIER</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="mc-metric-val" style="text-align: center;">{multiplier:.2f}x</div>', unsafe_allow_html=True)
                        st.markdown('<div class="mc-metric-sub" style="text-align: center;">향후 5개년 주가 상승 잠재 여력 배수</div>', unsafe_allow_html=True)
                        st.markdown('<div class="mc-divider" style="margin: 16px 0;"></div>', unsafe_allow_html=True)
                        st.markdown(
                            f"""
                            <div style="font-size: 12.5px; color: #696969; line-height: 1.4; text-align: center;">
                                수학적 정량 함수($5^{{A}}$) 기준 산출.<br>
                                예측 매력도 A 스코어: <b>{attractiveness_score:.4f}</b><br>
                                추정 연환산 변동성 R: <b>{risk_score:.2f}%</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                with col_scores:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>FACTOR INTERACTION BAR CHART</div>', unsafe_allow_html=True)
                        st.markdown('<h4>딥러닝 판단(A)과 회계 팩터(F-Score, Quality) 비교 검증</h4>', unsafe_allow_html=True)
                        
                        # Grouped Bar chart
                        f_score = float(row_diag["C_FSCORE"]) if not pd.isna(row_diag["C_FSCORE"]) else 0.0
                        q_score = float(row_diag["C_QUALITY"]) if not pd.isna(row_diag["C_QUALITY"]) else 0.0
                        
                        # Piotroski F-Score ranges 0-9. To put on a unified visual scale, we can scale it or show absolute values.
                        # Showing absolute values on a grouped bar chart
                        fig_grouped_bar = go.Figure(data=[
                            go.Bar(
                                x=['Piotroski F-Score', 'Quality Score (Asness)', 'FTT 예측 매력도 A'],
                                y=[f_score, q_score, attractiveness_score],
                                marker_color=['#141413', '#696969', '#CF4500'],
                                text=[f"{f_score:.0f}", f"{q_score:.2f}", f"{attractiveness_score:.4f}"],
                                textposition='auto',
                            )
                        ])
                        
                        fig_grouped_bar.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=200,
                            yaxis=dict(showgrid=True, gridcolor='#E8E2DA'),
                            font=dict(color='#141413')
                        )
                        
                        st.plotly_chart(fig_grouped_bar, use_container_width=True)
                        
                        st.markdown(
                            """
                            <span class="eyebrow-label" style="font-size: 9px; margin-top: 8px;">Cross Validation Note</span>
                            <p style="font-size: 11.5px; color: #696969; margin: 0; line-height: 1.3;">
                                피오트로스키 F-스코어는 자산 건전성(0~9)을 나타내며, 아스네스 퀄리티 스코어는 초과 수익의 지속성(수익성/성장성 배분)을 나타냅니다.
                                딥러닝 예측 스코어 A가 양의 방향으로 크고 회계 지표들이 견고히 뒷받침할 때 가치-모멘텀 동조 우량주로 정의할 수 있습니다.
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Full width Historical Line Chart
                st.markdown('<div class="mc-divider" style="margin: 24px 0;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>HISTORICAL METRICS TRACKING</div>', unsafe_allow_html=True)
                st.markdown('<h4>지표 변천사 (Historical Attractiveness & Risk)</h4>', unsafe_allow_html=True)
                
                df_history = predictions_df[predictions_df["ticker"] == selected_ticker].sort_values("date")
                if not df_history.empty and "date" in df_history.columns:
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        x=df_history["date"], y=df_history["FTT_A"],
                        mode='lines+markers', name='Attractiveness (A)',
                        line=dict(color='#CF4500', width=2),
                        marker=dict(size=6)
                    ))
                    fig_history.add_trace(go.Scatter(
                        x=df_history["date"], y=df_history["FTT_R"],
                        mode='lines+markers', name='Risk (R)',
                        line=dict(color='#141413', width=2, dash='dot'),
                        marker=dict(size=6)
                    ))
                    fig_history.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=300,
                        xaxis=dict(showgrid=True, gridcolor='#E8E2DA', title='Date'),
                        yaxis=dict(showgrid=True, gridcolor='#E8E2DA', title='Score'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        font=dict(color='#141413')
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                else:
                    st.info("과거 시계열 예측 데이터가 부족하여 변천사를 표시할 수 없습니다.")
            else:

                st.warning("선택한 종목의 정밀 분석 데이터를 불러오지 못했습니다.")
        else:
            st.warning("스냅샷 예측 정보 파일이 부재합니다.")

# =============================================================================
# TAB 4: 트레이닝 결과 확인 (TRAINING METRICS & LOGS)
# =============================================================================
with tab4:
    st.markdown(
        """
        <div class="mc-card">
            <div class="eyebrow-label"><span class="eyebrow-dot"></span>MODEL OPTIMIZATION LAB</div>
            <h3>인공신경망 학습 성능 지표 및 하이퍼파라미터 조회</h3>
            <p style="color: #696969; font-size: 15px;">
                딥러닝 예측 모형 FT-Transformer와 시계열 피처 부호기들의 백엔드 학습 파라미터 설정을 파싱하고, 
                학습 과정(Loss Curve) 및 체크포인트 보관 상태를 투명하게 조회하는 정량적 분석 패널입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_params, col_log_plot = st.columns([2, 3])
    
    with col_params:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>HYPERPARAMETERS CONFIGURATION</div>', unsafe_allow_html=True)
            st.markdown('<h4>학습 아키텍처 설정 명세</h4>', unsafe_allow_html=True)
            
            # Load hyperparams from config/settings.yaml
            settings_path = Path("config/settings.yaml")
            if settings_path.exists():
                with open(settings_path, encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    
                model_cfg = cfg.get("model", {})
                universe_cfg = cfg.get("universe", {})
                split_cfg = cfg.get("split", {})
                
                st.markdown(
                    f"""
                    <div style="font-size: 13.5px; line-height: 1.6; color: #141413;">
                        <b style="color: #CF4500;">프로젝트 명칭</b>: {cfg.get('project', {}).get('name', 'stockml')}<br>
                        <b>국가 유니버스</b>: {", ".join(universe_cfg.get('countries', []))}<br>
                        <b>학습/검증/시험 비율</b>: {split_cfg.get('train_ratio')*100:.0f}% / {split_cfg.get('val_ratio')*100:.0f}% / {split_cfg.get('test_ratio')*100:.0f}%<br>
                        <div class="mc-divider" style="margin: 12px 0;"></div>
                        <b style="color: #141413; font-size: 14px;">[FT-Transformer] 아키텍처</b><br>
                        • Embedding Dim (d_token): <b>{model_cfg.get('d_token', 192)}</b> (Heads: {model_cfg.get('n_heads', 8)}, Layers: {model_cfg.get('n_layers', 4)})<br>
                        • Dropout / Attn Dropout: <b>{model_cfg.get('dropout', 0.2)} / {model_cfg.get('attn_dropout', 0.1)}</b><br>
                        • FFN Expansion Factor: <b>{model_cfg.get('ffn_factor', 1.333)}</b><br>
                        <div class="mc-divider" style="margin: 12px 0;"></div>
                        <b style="color: #141413; font-size: 14px;">[LSTM Encoders] 아키텍처</b><br>
                        • Stock sequence max len: <b>{model_cfg.get('lstm_stock_max_seq', 20)}분기</b> (Hidden Dim: {model_cfg.get('lstm_stock_hidden', 128)}, Layers: {model_cfg.get('lstm_stock_layers', 2)})<br>
                        • Macro sequence max len: <b>{model_cfg.get('lstm_macro_max_seq', 20)}분기</b> (Hidden Dim: {model_cfg.get('lstm_macro_hidden', 64)}, Layers: {model_cfg.get('lstm_macro_layers', 1)})<br>
                        <div class="mc-divider" style="margin: 12px 0;"></div>
                        <b style="color: #141413; font-size: 14px;">최적화 파라미터 (Optimization)</b><br>
                        • Batch Size: <b>{model_cfg.get('batch_size', 64)}</b><br>
                        • Initial Learning Rate: <b>{model_cfg.get('lr', 0.0001)}</b> (Weight Decay: {model_cfg.get('weight_decay', 0.01)})<br>
                        • Max Epochs: <b>{model_cfg.get('max_epochs', 60)}</b> (Patience: {model_cfg.get('patience', 10)})
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("config/settings.yaml 설정 파일을 불러오지 못했습니다.")
            
    with col_log_plot:
        with st.container(border=True):
            st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>TRAINING VS VALIDATION LOSS CURVE</div>', unsafe_allow_html=True)
            st.markdown('<h4>학습 에폭별 손실 추이 모니터링</h4>', unsafe_allow_html=True)
            
            # Scan lightning_logs directory for metrics.csv
            logs_dir = Path("lightning_logs")
            csv_files = []
            if logs_dir.exists():
                csv_files = sorted(list(logs_dir.glob("version_*/metrics.csv")), key=lambda x: int(x.parent.name.split("_")[1]), reverse=True)
                
            if csv_files:
                # Let the user select version, default to latest
                version_options = [c.parent.name for c in csv_files]
                selected_ver = st.selectbox("관측할 학습 세션 버전 선택 (Select Run Version) →", options=version_options, index=0)
                
                selected_csv_path = csv_files[version_options.index(selected_ver)]
                
                df_metrics = pd.read_csv(selected_csv_path)
                
                if not df_metrics.empty and "train_loss" in df_metrics.columns:
                    # Group by epoch to calculate average metrics per epoch
                    # Since Lightning writes step metrics, grouping by epoch cleans up noise
                    df_metrics_grouped = df_metrics.groupby("epoch").mean(numeric_only=True).reset_index()
                    
                    # Filter valid epoch values
                    df_metrics_grouped = df_metrics_grouped[df_metrics_grouped["epoch"].notna()]
                    
                    fig_loss = go.Figure()
                    
                    # Training Loss
                    fig_loss.add_trace(go.Scatter(
                        x=df_metrics_grouped["epoch"],
                        y=df_metrics_grouped["train_loss"],
                        mode="lines",
                        name="Train Loss",
                        line=dict(color='#CF4500', width=2.5)  # Signal Orange
                    ))
                    
                    # Validation Loss
                    if "val_loss" in df_metrics_grouped.columns:
                        # Drop NaN values for validation loss
                        df_val = df_metrics_grouped.dropna(subset=["val_loss"])
                        fig_loss.add_trace(go.Scatter(
                            x=df_val["epoch"],
                            y=df_val["val_loss"],
                            mode="lines+markers",
                            name="Validation Loss",
                            line=dict(color='#141413', width=2.5)  # Ink Black
                        ))
                        
                    fig_loss.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=280,
                        xaxis=dict(title="에폭 (Epoch)", showgrid=True, gridcolor='#E8E2DA'),
                        yaxis=dict(title="손실 값 (Loss)", showgrid=True, gridcolor='#E8E2DA'),
                        hovermode="x unified",
                        font=dict(color='#141413')
                    )
                    
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.info("해당 로그 파일에 유효한 학습 메트릭이 기록되지 않았습니다.")
            else:
                # Fallback guideline simulator (if empty/missing)
                st.info("백엔드 학습 로그(metrics.csv)가 감지되지 않아 시뮬레이션 가이드를 출력합니다.")
                epochs_sim = np.arange(25)
                train_loss_sim = 2.5 * np.exp(-epochs_sim/10) + 0.1 * np.random.randn(25) + 0.2
                val_loss_sim = 2.5 * np.exp(-epochs_sim/12) + 0.05 * np.random.randn(25) + 0.3
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs_sim, y=train_loss_sim, mode="lines", name="Simulated Train Loss", line=dict(color='#CF4500')))
                fig_loss.add_trace(go.Scatter(x=epochs_sim, y=val_loss_sim, mode="lines", name="Simulated Val Loss", line=dict(color='#141413')))
                fig_loss.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, font=dict(color='#141413'))
                st.plotly_chart(fig_loss, use_container_width=True)
        
    st.markdown('<div class="mc-divider"></div>', unsafe_allow_html=True)
    
    # ── Checkpoints directory scanner ──────────────────────────
    with st.container(border=True):
        st.markdown('<div class="eyebrow-label"><span class="eyebrow-dot"></span>SAVED CHECKPOINTS EXPLORER</div>', unsafe_allow_html=True)
        st.markdown('<h4>저장된 체크포인트 파일 정보 및 백엔드 안정성</h4>', unsafe_allow_html=True)
        
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
            
            if ckpt_files:
                ckpt_data = []
                for filepath in ckpt_files:
                    filename = filepath.name
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    
                    # Resilient epoch and val_loss extraction using multiple regex fallbacks
                    epoch_match = re.search(r'epoch[=:](\d+)', filename)
                    if not epoch_match:
                        epoch_match = re.search(r'epoch=(\d+)', filename)
                    if not epoch_match:
                        epoch_match = re.search(r'epoch:(\d+)', filename)
                        
                    val_loss_match = re.search(r'val_loss[=:]\s*([0-9\.]+)', filename)
                    if not val_loss_match:
                        val_loss_match = re.search(r'val_loss=([0-9\.]+)', filename)
                    if not val_loss_match:
                        val_loss_match = re.search(r'val_loss:([0-9\.]+)', filename)
                        
                    epoch_val = int(epoch_match.group(1)) if epoch_match else "N/A"
                    
                    val_loss_val = 999.0
                    if val_loss_match:
                        try:
                            # Strip any leading/trailing whitespace and trailing dots (such as those before .ckpt)
                            val_loss_str = val_loss_match.group(1).strip().rstrip('.')
                            # Handle potential double-dots or anomalies
                            if val_loss_str.count('.') > 1:
                                parts = val_loss_str.split('.')
                                val_loss_str = parts[0] + '.' + "".join(parts[1:])
                            val_loss_val = float(val_loss_str)
                        except Exception:
                            val_loss_val = 999.0
                    
                    ckpt_data.append({
                        "파일명 (Filename)": filename,
                        "학습 에폭 (Epoch)": epoch_val,
                        "검증 오차 (Val Loss)": val_loss_val if val_loss_val != 999.0 else "N/A",
                        "파일 크기 (Size)": f"{size_mb:.1f} MB",
                        "수정 시각 (Modified At)": pd.Timestamp(filepath.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
                    })
                    
                df_ckpt = pd.DataFrame(ckpt_data)
                
                # Sort by Val Loss
                # Filter rows with numeric validation loss to sort correctly
                df_ckpt_numeric = df_ckpt[df_ckpt["검증 오차 (Val Loss)"] != "N/A"].copy()
                df_ckpt_numeric["검증 오차 (Val Loss)"] = df_ckpt_numeric["검증 오차 (Val Loss)"].astype(float)
                df_ckpt_sorted = df_ckpt_numeric.sort_values(by="검증 오차 (Val Loss)", ascending=True)
                
                # Append non-numeric rows if any
                df_ckpt_na = df_ckpt[df_ckpt["검증 오차 (Val Loss)"] == "N/A"]
                df_ckpt_final = pd.concat([df_ckpt_sorted, df_ckpt_na], ignore_index=True)
                
                # Styled Display
                st.dataframe(
                    df_ckpt_final,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown(
                    """
                    <span class="eyebrow-label" style="font-size: 9px; margin-top: 8px;">Checkpoint Tracker Guide</span>
                    <p style="font-size: 11.5px; color: #696969; margin: 0; line-height: 1.3;">
                        가장 오차가 낮은 체크포인트는 최적화 가중치 저장을 통해 예측 단계의 가중치로 적재됩니다.
                        에폭 증가에 따른 Val Loss의 수렴 여부를 확인하여 모델 오버피팅을 방지할 수 있습니다.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("checkpoints/ 디렉토리는 존재하나 저장된 체크포인트(*.ckpt) 파일이 없습니다.")
        else:
            st.warning("checkpoints/ 디렉토리가 감지되지 않았습니다.")
