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
    page_title="QuantML - Stock Attractiveness & Risk Dashboard",
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
    </style>
""", unsafe_allow_html=True)

st.title("📊 QuantML 종목별 매력도 · 위험도 분석")
st.markdown("""
    <div class="premium-card">
        <h4>💡 다차원 정량적 분석 프레임워크 (v5.1)</h4>
        <p style="margin: 0; color: #d1d5db;">
            본 시스템은 <b>S&P 500</b> 및 <b>KOSPI</b> 유니버스를 대상으로 다기간 시계열 데이터를 분석하여 매력도(A)와 위험도(R)를 추정합니다.
            각 수치는 독립적으로 계산되며 투자자의 선호도에 맞춰 매칭할 수 있습니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_predictions():
    df = pd.read_parquet('data/processed/predictions_latest.parquet')
    # datetime 형식 변환
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = load_predictions()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}. 파이프라인을 먼저 순서대로 실행해주세요.")
    st.stop()

# 사이드바 필터
with st.sidebar:
    st.header("🔍 필터링 옵션")
    st.markdown("---")
    
    # 국가 필터
    countries = df['country'].unique()
    selected_countries = st.multiselect("국가", countries, default=list(countries))
    
    # 섹터 필터
    sectors = df['sector'].unique()
    selected_sectors = st.multiselect("섹터", sectors, default=list(sectors))
    
    # 테마 필터 (explode 대응)
    all_themes = sorted(list(set([t for themes in df['themes'].dropna() for t in themes])))
    selected_themes = st.multiselect("테마 (Naver)", all_themes)

# 필터 적용
filtered = df[df['country'].isin(selected_countries) & df['sector'].isin(selected_sectors)]
if selected_themes:
    filtered = filtered[filtered['themes'].apply(lambda t: any(x in t for x in selected_themes))]

if filtered.empty:
    st.warning("선택한 필터 조건에 부합하는 종목이 없습니다.")
    st.stop()

# 레이아웃 구성
col_chart, col_rank = st.columns([2, 1])

with col_chart:
    st.subheader("📈 매력도(A) vs 위험도(R) 2D 포트폴리오 맵")
    
    # 2D 산점도 플롯
    fig = px.scatter(
        filtered,
        x='R', y='A',
        color='sector',
        size='close',
        hover_data=['ticker', 'name', 'country', 'close'],
        title="최대 5년 전망: 매력도 vs 위험도 분포",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="위험도 (R - 연환산 변동성)",
        yaxis_title="매력도 (A - 단일 지표 값)",
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
    top_stocks = filtered.sort_values(by='A', ascending=False).head(10)
    
    # 세련된 표 표시
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

# 3. 상세 종목 딥다이브
st.markdown("---")
st.subheader("🔍 개별 종목 정밀 진단")

selected_ticker = st.selectbox("종목 선택", sorted(filtered['ticker'].unique()))
stock = filtered[filtered['ticker'] == selected_ticker].iloc[0]

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

# 학술/재무 베이스라인 비교 레이아웃
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("⚙️ 모델별 지표 비교 대조군 (Baselines)")

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    st.info(f"Piotroski F-Score: **{stock['C_FSCORE']:.1f}**")
with col_b2:
    st.info(f"Asness Quality Score: **{stock['C_QUALITY']:.2f}**")
with col_b3:
    st.info(f"Composite Score: **{stock['ACC_COMPOSITE']:.2f}**")

# 하단 푸터
st.markdown("---")
st.caption("QuantML v5.1 | Powered by Temporal Fusion Transformer (TFT) on M1 Pro GPU accelerator")
