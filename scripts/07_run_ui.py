# scripts/07_run_ui.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration with a premium dark theme feel
st.set_page_config(
    page_title="QuantML - Portfolio Intelligence & Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via Markdown HTML
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .metric-caption {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 5px;
    }
    .premium-card {
        background: rgba(31, 41, 55, 0.4);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .correlation-label {
        font-size: 0.9rem;
        font-weight: bold;
        color: #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 QuantML 다차원 포트폴리오 분석 및 매크로 지능 엔진")

st.markdown("""
    <div class="premium-card">
        <h4>💡 다차원 정량적 분석 프레임워크 (v5.2)</h4>
        <p style="margin: 0; color: #d1d5db;">
            본 시스템은 <b>S&P 500</b> 및 <b>KOSPI</b> 유니버스를 대상으로 다기간 시계열 데이터를 분석하여 매력도(A)와 위험도(R)를 추정하고,
            글로벌 <b>거시경제(Macro)</b> 및 <b>피처(Features)</b> 데이터의 다차원 연관관계를 실시간 심층 분석합니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# 데이터 로드 캐싱 함수
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
        df = pd.read_parquet('data/processed/features.parquet')
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

# 로드 실행
try:
    df = load_predictions()
    df_raw = load_raw_prices()
    df_feat = load_features()
    df_macro = load_macro()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}. 파이프라인을 먼저 순서대로 실행해주세요.")
    st.stop()

# 사이드바 필터
with st.sidebar:
    st.header("🔍 글로벌 필터 설정")
    st.markdown("---")
    
    # 국가 필터
    countries = df['country'].unique()
    selected_countries = st.multiselect("분석 대상 국가", countries, default=list(countries))
    
    # 섹터 필터
    sectors = df['sector'].unique()
    selected_sectors = st.multiselect("분석 대상 섹터", sectors, default=list(sectors))
    
    # 테마 필터 (explode 대응)
    all_themes = sorted(list(set([t for themes in df['themes'].dropna() for t in themes])))
    selected_themes = st.multiselect("글로벌 매칭 테마", all_themes)

# 필터 적용 전 최신 날짜 정보 추출
latest_idx = df.groupby('ticker', observed=True)['date'].idxmax()
df_latest = df.loc[latest_idx]

filtered_latest = df_latest[df_latest['country'].isin(selected_countries) & df_latest['sector'].isin(selected_sectors)]
if selected_themes:
    filtered_latest = filtered_latest[filtered_latest['themes'].apply(lambda t: any(x in t for x in selected_themes))]

if filtered_latest.empty:
    st.warning("선택한 필터 조건에 부합하는 종목이 없습니다.")
    st.stop()

# st.tabs를 통해 다차원 탭 생성
tab_portfolio, tab_data, tab_macro = st.tabs([
    "🎯 포트폴리오 분석 (Portfolio Map)", 
    "📊 데이터 탐색기 (Raw & Processed Data)", 
    "📉 거시경제 상관관계 (Macro Correlation)"
])

# -----------------------------------------------------------------------------
# TAB 1: 포트폴리오 분석 (Portfolio Map)
# -----------------------------------------------------------------------------
with tab_portfolio:
    col_chart, col_rank = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📈 매력도(A) vs 위험도(R) 2D 포트폴리오 맵")
        
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
            template="plotly_dark",
            xaxis_title="위험도 (R - 연환산 변동성)",
            yaxis_title="매력도 (A - 단일 기대수익 값)",
            legend_title="섹터",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#2d3748'),
            yaxis=dict(gridcolor='#2d3748'),
            hoverlabel=dict(bgcolor='#1f2937', font_size=13)
        )
        st.plotly_chart(fig, use_container_width=True)
     
    with col_rank:
        st.subheader("🏆 매력도 Top 종목")
        top_stocks = filtered_latest.sort_values(by='A', ascending=False).head(10)
        
        st.dataframe(
            top_stocks[['ticker', 'name', 'sector', 'A', 'R']],
            column_config={
                "ticker": "티커",
                "name": "종목명",
                "sector": "섹터",
                "A": st.column_config.NumberColumn("매력도 (A)", format="%.2f"),
                "R": st.column_config.NumberColumn("위험도 (R)", format="%.1%"),
            },
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.subheader("🔍 개별 종목 정밀 진단")
    
    selected_ticker = st.selectbox("진단할 종목 선택", sorted(filtered_latest['ticker'].unique()), key="deepdive_ticker")
    stock = filtered_latest[filtered_latest['ticker'] == selected_ticker].iloc[0]
    
    col_detail1, col_detail2, col_detail3 = st.columns(3)
    
    with col_detail1:
        A_val = stock['A']
        multiple = 5 ** A_val
        st.metric(label="🌟 TFT 추정 매력도 (A)", value=f"{A_val:.2f}")
        st.markdown(f"<div class='metric-caption'>향후 최대 5년 내 약 <b>{multiple:.1f}배</b> 주가 상승 여력 내포</div>", unsafe_allow_html=True)
    
    with col_detail2:
        R_val = stock['R']
        st.metric(label="⚠️ TFT 추정 위험도 (R)", value=f"{R_val:.1%}")
        st.markdown("<div class='metric-caption'>연환산 변동성(표준편차) 기준 위험 수준</div>", unsafe_allow_html=True)
    
    with col_detail3:
        st.metric(label="💵 최종 종가", value=f"{stock['close']:.2f}")
        st.markdown(f"<div class='metric-caption'>분기 최종 가격 (국가: {stock['country']})</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚙️ 모델별 지표 비교 대조군 (Baselines)")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.info(f"Piotroski F-Score: **{stock['C_FSCORE']:.1f} / 9.0**")
    with col_b2:
        st.info(f"Asness Quality Score: **{stock['C_QUALITY']:.2f}**")
    with col_b3:
        st.info(f"Composite Score: **{stock['ACC_COMPOSITE']:.2f}**")

# -----------------------------------------------------------------------------
# TAB 2: 데이터 탐색기 (Raw & Processed Data)
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("📊 원본(Raw) 시세 & 가공(Processed) 피처 탐색기")
    
    # 돋보이는 종목 선택
    all_tickers = sorted(df_raw['ticker'].unique()) if not df_raw.empty else []
    if all_tickers:
        selected_raw_ticker = st.selectbox("조회 대상 종목 선택", all_tickers, index=all_tickers.index(selected_ticker) if selected_ticker in all_tickers else 0)
        
        # 1. 원본 데이터 섹션
        st.markdown("### 📈 1. 원본 종가 및 거래량 (Raw Historical Data)")
        raw_stock_df = df_raw[df_raw['ticker'] == selected_raw_ticker].sort_values('date')
        
        if not raw_stock_df.empty:
            col_raw_chart, col_raw_table = st.columns([2, 1])
            with col_raw_chart:
                # Plotly 이중 축 차트 (종가 & 거래량)
                fig_raw = go.Figure()
                fig_raw.add_trace(go.Scatter(x=raw_stock_df['date'], y=raw_stock_df['close'], name="종가", line=dict(color="#3b82f6", width=2.5)))
                fig_raw.add_trace(go.Bar(x=raw_stock_df['date'], y=raw_stock_df['volume'], name="거래량", yaxis="y2", marker_color="rgba(147, 51, 234, 0.3)"))
                
                fig_raw.update_layout(
                    template="plotly_dark",
                    title=f"{selected_raw_ticker} 분기별 종가 및 거래량 추이",
                    xaxis_title="날짜",
                    yaxis=dict(title="종가 (Close)", titlefont=dict(color="#3b82f6"), tickfont=dict(color="#3b82f6")),
                    yaxis2=dict(title="거래량 (Volume)", titlefont=dict(color="#9333ea"), tickfont=dict(color="#9333ea"), overlaying="y", side="right"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#2d3748'),
                    legend=dict(x=0.01, y=0.99)
                )
                st.plotly_chart(fig_raw, use_container_width=True)
            with col_raw_table:
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    raw_stock_df[['date', 'close', 'volume']].sort_values('date', ascending=False),
                    column_config={
                        "date": "날짜",
                        "close": st.column_config.NumberColumn("종가", format="%.2f"),
                        "volume": st.column_config.NumberColumn("거래량", format="%,d"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("선택 종목의 원본 주가 정보가 존재하지 않습니다.")
            
        # 2. 가공 피처 데이터 섹션
        st.markdown("### ⚙️ 2. 가공된 시계열 피처 추이 (Processed Engineered Features)")
        feat_stock_df = df_feat[df_feat['ticker'] == selected_raw_ticker].sort_values('date')
        
        if not feat_stock_df.empty:
            # 수치형 피처 컬럼 자동 추출
            excluded_cols = ['ticker', 'date', 'country', 'currency', 'sector', 'name', 'themes']
            feat_cols = [c for c in feat_stock_df.columns if c not in excluded_cols and pd.api.types.is_numeric_dtype(feat_stock_df[c])]
            
            selected_features = st.multiselect(
                "시각화할 피처 선택 (다중 선택 가능)", 
                feat_cols, 
                default=[c for c in ['ret_1q', 'vol_1y', 'F_SCORE', 'QUALITY_SCORE'] if c in feat_cols]
            )
            
            if selected_features:
                col_feat_chart, col_feat_table = st.columns([2, 1])
                with col_feat_chart:
                    fig_feat = px.line(
                        feat_stock_df,
                        x='date',
                        y=selected_features,
                        title=f"{selected_raw_ticker} 가공 시계열 피처 시계열",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig_feat.update_layout(
                        template="plotly_dark",
                        xaxis_title="날짜",
                        yaxis_title="피처 값 (Scaled/Raw)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#2d3748'),
                        yaxis=dict(gridcolor='#2d3748')
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)
                with col_feat_table:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(
                        feat_stock_df[['date'] + selected_features].sort_values('date', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("시각화할 피처를 최소 1개 이상 선택해 주세요.")
        else:
            st.info("선택 종목의 가공 피처 정보가 존재하지 않습니다. 피처 빌드(Step 2)를 실행해 주세요.")
    else:
        st.info("조회할 수 있는 종목 시세 데이터가 없습니다.")

# -----------------------------------------------------------------------------
# TAB 3: 거시경제 상관관계 (Macro Correlation)
# -----------------------------------------------------------------------------
with tab_macro:
    st.subheader("📉 글로벌 거시경제(Macro) 지표 연관관계 분석")
    
    # FRED 매크로 지표 라벨명 매핑
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
        # 데이터프레임 컬럼 리네임 및 날짜 정렬
        macro_plot_df = df_macro.rename(columns=MACRO_LABELS).sort_values('date')
        macro_cols = [c for c in macro_plot_df.columns if c != 'date']
        
        # 1. 상관관계 매트릭스 히트맵
        st.markdown("### 🔥 1. 매크로 지표 피어슨 상관관계 (Pearson Correlation Heatmap)")
        corr_matrix = macro_plot_df[macro_cols].corr()
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmin=-1.0, zmax=1.0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            title="거시경제 변수 간 교차 상관관계 (레드: 양의 상관 / 블루: 음의 상관)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=1000,
            height=600
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # 2. 매크로 지표 시계열 추이
        st.markdown("### 📉 2. 매크로 지표 역사적 추이 (Macro Indicators Over Time)")
        
        selected_macro_cols = st.multiselect(
            "비교할 매크로 지표 선택",
            macro_cols,
            default=list(macro_cols[:3])
        )
        
        if selected_macro_cols:
            col_macro_chart, col_macro_table = st.columns([2, 1])
            with col_macro_chart:
                # 개별 지표 스케일이 달라 표준화 플롯 옵션 제공
                use_standardized = st.checkbox("데이터 표준화(Standardize)하여 그리기 (서로 다른 단위 비교용)")
                
                plot_data = macro_plot_df[['date'] + selected_macro_cols].copy()
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
                    title="거시경제 시계열 모니터링",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_macro_line.update_layout(
                    template="plotly_dark",
                    xaxis_title="날짜",
                    yaxis_title="값 (표준화)" if use_standardized else "지표 원본 값",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#2d3748'),
                    yaxis=dict(gridcolor='#2d3748')
                )
                st.plotly_chart(fig_macro_line, use_container_width=True)
            with col_macro_table:
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    macro_plot_df[['date'] + selected_macro_cols].sort_values('date', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("비교 분석할 매크로 지표를 최소 1개 이상 선택해 주세요.")
    else:
        st.warning("FRED 매크로 지표 데이터가 존재하지 않습니다. scripts/01_fetch_data.py 를 먼저 가동하여 FRED 데이터를 적재해 주세요.")

# 하단 푸터
st.markdown("---")
st.caption("QuantML v5.2 | Powered by Temporal Fusion Transformer (TFT) on M1 Pro GPU accelerator")

