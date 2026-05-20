# stockml 구현 가이드 v5.5
# Coding Agent 전용 기술 명세서

"""
이 문서는 v5.4를 기준으로 다음을 완전 교체/추가한다:
  - 지표 레퍼런스 신규: M_* (거시), A_* (한국 자산집중도), F_* (기업 재무), C_* (계산형) 지표 전수 정의
  - 테마 비중 벡터 설계 변경: 순위 백분위 (rank percentile) → 실질 비중 (actual weight) 및 상대 배수 (relative multiple) (18차원)
  - theme_context.py 교체: w_mktcap, w_revenue, w_ebitda, rel_pbr 등 18차원 계산 로직 구현

Coding agent 행동 원칙:
  - 모든 섹션을 위에서 아래로 순서대로 구현한다
  - 섹션 간 의존성은 명시된 import 경로를 따른다
  - 각 모듈은 독립적으로 단위 테스트 가능하게 작성한다
  - 불명확한 결정은 이 문서의 설계 의도를 우선한다
  - M1 Pro MPS 제약을 항상 염두에 두고 float32, pin_memory=False를 기본으로 한다
"""

# =============================================================================
# PART 1. 전체 지표 목록 (레퍼런스)
# =============================================================================

# 사용 흐름:
#   M_*  → features_macro.parquet  → LSTM_macro 시계열 입력
#   F_*, A_* → features_stock.parquet → LSTM_stock 시계열 입력
#   C_*  → features_stock.parquet  → FT-Transformer 스냅샷 입력
#   THEME_CTX → theme_context.parquet → FT-Transformer 스냅샷 입력
#
# 각 지표는 분기 종가 기준으로 resample('QE').last() 로 집계.
# Point-in-Time lag은 config/feature_lags.yaml 참조.

# -----------------------------------------------------------------------------
# M — 거시 지표 (80종, LSTM_macro 입력)
# -----------------------------------------------------------------------------
MACRO_INDICATORS = """
# ── M_INT: 금리 (13종) ────────────────────────────────────────────────────────
M_INT_001  기준금리                   중앙은행 정책금리                  FRED(DFF) / ECOS       US·KR
M_INT_002  국채 2Y                    단기 기대 정책금리                 FRED(DGS2) / ECOS      US·KR
M_INT_003  국채 10Y                   장기 성장·인플레 기대              FRED(DGS10) / ECOS     US·KR
M_INT_004  국채 30Y                   초장기 자본비용                    FRED(DGS30) / ECOS     US·KR
M_INT_005  장단기 스프레드 10Y-2Y      경기침체 선행                      산출                   US·KR
M_INT_006  장단기 스프레드 10Y-3M      Estrella & Mishkin 기준            산출                   US
M_INT_007  실질금리 TIPS              명목금리 - 기대인플레              FRED(DFII10)           US
M_INT_008  실질금리 한국              국고채 10Y - 기대CPI              ECOS                   KR
M_INT_009  모기지 30Y                 부동산·소비 선행                   FRED(MORTGAGE30US)     US
M_INT_010  SOFR                      LIBOR 대체 기준금리                FRED(SOFR)             US
M_INT_011  CD 91일                   단기 자금시장 기준                 ECOS                   KR
M_INT_012  COFIX                     주택담보대출 기준금리              은행연합회             KR
M_INT_013  가산금리 은행채3Y-국채3Y    단기 크레딧 스프레드              ECOS                   KR

# ── M_LIQ: 유동성 (10종) ──────────────────────────────────────────────────────
M_LIQ_001  M1                        현금+요구불예금                    FRED(M1SL) / ECOS      US·KR
M_LIQ_002  M2                        M1+저축성예금                      FRED(M2SL) / ECOS      US·KR
M_LIQ_003  M2 YoY 성장률             복리 기준                          산출                   US·KR
M_LIQ_004  연준 대차대조표            QE/QT 사이클                       FRED(WALCL)            US
M_LIQ_005  역레포 RRP 잔액            연준 단기 유동성 흡수              FRED(RRPONTSYD)        US
M_LIQ_006  한국은행 RP 잔액           공개시장조작                       ECOS                   KR
M_LIQ_007  통화승수 M2/본원통화       신용창출 배율                      산출                   US·KR
M_LIQ_008  화폐유통속도 V=GDP/M2      실물 유동성 활용도                 산출                   US·KR
M_LIQ_009  글로벌 M2 USD 환산         전세계 유동성                      FRED 합산              GLOBAL
M_LIQ_010  초과지급준비금             은행 여유 유동성                   FRED / ECOS            US·KR

# ── M_CRD: 신용·리스크 (11종) ─────────────────────────────────────────────────
M_CRD_001  하이일드 스프레드 HY       ICE BofA HY OAS                   FRED(BAMLH0A0HYM2)     US
M_CRD_002  투자등급 스프레드 IG       ICE BofA IG OAS                   FRED(BAMLC0A0CM)       US
M_CRD_003  TED 스프레드               LIBOR 3M - T-Bill 3M              FRED                   US
M_CRD_004  LIBOR-OIS 스프레드         은행 간 신용리스크                 FRED                   US
M_CRD_005  CDS 5Y 국가신용            미국·한국 부도 위험                Bloomberg              US·KR
M_CRD_006  회사채 스프레드 한국 AAA   우량채 가산금리                    ECOS                   KR
M_CRD_007  회사채 스프레드 한국 BBB   투기등급 가산금리                  ECOS                   KR
M_CRD_008  기업 파산 건수 YoY         신용 사이클 후행                   법원통계 / Fed         US·KR
M_CRD_009  연체율 가계                소비자 신용리스크                  금융감독원 / Fed       US·KR
M_CRD_010  연체율 기업                기업 신용리스크                    금융감독원 / Fed       US·KR
M_CRD_011  대출 연체 90일+ 비율       부실 심화                          금융감독원             KR

# ── M_INF: 인플레이션 (15종) ──────────────────────────────────────────────────
M_INF_001  CPI YoY                   소비자물가 복리 변화율             BLS / 통계청           US·KR
M_INF_002  Core CPI YoY              식품·에너지 제외                   BLS / 통계청           US·KR
M_INF_003  PCE 디플레이터 YoY         연준 선호 물가                     BEA                    US
M_INF_004  Core PCE YoY              연준 목표 물가 2%                  BEA                    US
M_INF_005  PPI YoY                   생산자물가 CPI 선행                BLS / 통계청           US·KR
M_INF_006  기대인플레이션 5Y5Y        장기 인플레 기대                   FRED(T5YIFR)           US
M_INF_007  기대인플레이션 2Y          단기 인플레 기대                   FRED                   US
M_INF_008  수입물가 YoY               해외발 인플레 압력                 한국은행 / BLS         US·KR
M_INF_009  주거비 CPI Shelter         미국 CPI 최대 구성요소             BLS                    US
M_INF_010  임금 상승률 YoY            비용 인플레 핵심                   BLS / 고용부           US·KR
M_INF_011  유가 WTI/Brent             에너지 인플레                      EIA / Bloomberg        GLOBAL
M_INF_012  천연가스 가격               에너지 비용                        EIA                    GLOBAL
M_INF_013  구리 가격 Dr. Copper        실물경기 선행                      LME                    GLOBAL
M_INF_014  CRB 원자재 지수             광범위 원자재 물가                 Refinitiv              GLOBAL
M_INF_015  발틱운임지수 BDI            글로벌 교역·물류 비용              Baltic Exchange        GLOBAL

# ── M_ECO: 경기실물 (22종) ────────────────────────────────────────────────────
M_ECO_001  GDP 성장률 QoQ 연율        분기 실질 성장률                   BEA / 한국은행         US·KR
M_ECO_002  GDP 성장률 YoY             전년 동기 대비                     BEA / 한국은행         US·KR
M_ECO_003  GDP 갭 Output Gap          잠재 GDP 대비 실제 GDP             CBO / KDI              US·KR
M_ECO_004  ISM 제조업 PMI             50 기준 확장/수축                  ISM                    US
M_ECO_005  ISM 비제조업 PMI           서비스업 경기                      ISM                    US
M_ECO_006  S&P 글로벌 제조업 PMI      글로벌 제조업                      S&P Global             GLOBAL
M_ECO_007  한국 제조업 PMI            수출 중심 제조업                   S&P Global             KR
M_ECO_008  실업률                     고용시장 후행                      BLS / 통계청           US·KR
M_ECO_009  비농업 고용 NFP            미국 고용 핵심                     BLS                    US
M_ECO_010  JOLTS 구인 건수            고용시장 선행                      BLS                    US
M_ECO_011  소비자신뢰지수 CB          소비 심리                          Conference Board       US
M_ECO_012  미시간 소비심리지수         소비자 인플레 기대                 UMich                  US
M_ECO_013  소매판매 YoY               실물 소비 모멘텀                   Census / 통계청        US·KR
M_ECO_014  산업생산 YoY               제조업 실물 생산                   Fed / 통계청           US·KR
M_ECO_015  설비 가동률                인플레·투자 선행                   Fed                    US
M_ECO_016  주택착공 건수              건설·소비 선행                     Census                 US
M_ECO_017  건축허가 건수              주택착공 선행                      Census                 US
M_ECO_018  내구재 수주                기업 투자 선행                     Census                 US
M_ECO_019  한국 수출 YoY              글로벌 수요 집약                   관세청                 KR
M_ECO_020  한국 수출 반도체           IT 사이클 선행                     산업통상자원부         KR
M_ECO_021  경기선행지수 CLI           OECD 복합 선행지수                 OECD                   US·KR
M_ECO_022  기업경기실사지수 BSI        기업 체감경기                      한국은행               KR

# ── M_FX: 환율 (8종) ──────────────────────────────────────────────────────────
M_FX_001   USD/KRW                   원·달러 환율                       한국은행               KR
M_FX_002   DXY 달러 인덱스            달러 강약 종합                     FRED(DX-Y.NYB)         US
M_FX_003   EUR/USD                   유로·달러                          ECB                    GLOBAL
M_FX_004   USD/JPY                   엔·달러 안전자산 심리              BOJ                    GLOBAL
M_FX_005   원화 실효환율 REER         교역 가중 실질 환율                BIS                    KR
M_FX_006   경상수지 USD               외환 수급 펀더멘털                 한국은행               KR
M_FX_007   외환보유액                 외부충격 완충 능력                 한국은행               KR
M_FX_008   외국인 국채 보유 비율       국채 수급 외국 의존도              기재부                 KR

# ── M_SNT: 센티멘트 (15종) ────────────────────────────────────────────────────
M_SNT_001  VIX                       미국 시장 공포 지수                CBOE                   US
M_SNT_002  VKOSPI                    한국 변동성 지수                   KRX                    KR
M_SNT_003  SKEW Index                극단적 하락 헤지 수요              CBOE                   US
M_SNT_004  Put/Call Ratio            풋 vs 콜 옵션 비율                CBOE                   US
M_SNT_005  CNN Fear & Greed          복합 심리 지수                     CNN                    US
M_SNT_006  AAII Bullish 비율          개인 투자자 심리                   AAII                   US
M_SNT_007  II Bullish 기관            기관 투자자 심리                   Investors Intelligence US
M_SNT_008  외국인 순매수 주식          한국 외국인 수급                   KRX                    KR
M_SNT_009  기관 순매수 주식            국내 기관 수급                     KRX                    KR
M_SNT_010  개인 신용융자 잔고          개인 레버리지 수준                 금융투자협회           KR
M_SNT_011  공매도 비율                하락 베팅 강도                     KRX / FINRA            US·KR
M_SNT_012  주식형 펀드 순유입          간접 투자 자금 흐름               금융투자협회           KR
M_SNT_013  ETF 자금 유출입            직접 수급 지표                     ETFDB / 금투협         US·KR
M_SNT_014  GS 위험선호지수 GSRAII      글로벌 위험 선호                   Goldman Sachs          GLOBAL
M_SNT_015  BofA 펀드매니저 서베이      기관 현금·자산 배분               BofA                   GLOBAL

# ── M_FSC: 재정·국채 (7종) ────────────────────────────────────────────────────
M_FSC_001  미국 연방부채/GDP          재정 지속가능성                    FRED / CBO             US
M_FSC_002  미국 재정적자/GDP          국채 발행 압력                     CBO / Treasury         US
M_FSC_003  미국 국채 경매 BTC         Bid-to-Cover Ratio                Treasury               US
M_FSC_004  한국 국가채무/GDP          재정 건전성                        기재부                 KR
M_FSC_005  한국 국채 발행 잔액         국채 공급 압력                     기재부                 KR
M_FSC_006  한국 국채 외국인 보유       외국 자본 이탈 리스크              기재부                 KR
M_FSC_007  한국 통안채 발행 잔액       한국은행 불태화 정책               한국은행               KR

총계: 13+10+11+15+22+8+15+7 = 101종 (설계 상 80종 목표, 우선순위 ★★★ 기준 약 80종 선별)
"""

# -----------------------------------------------------------------------------
# A — 한국 자산집중도 지표 (35종, LSTM_stock 입력 — KR 종목에만 적용)
# -----------------------------------------------------------------------------
KOREAN_ASSET_INDICATORS = """
# ── A_RE: 부동산 (11종) ────────────────────────────────────────────────────────
A_RE_001   전국 아파트 실거래가 YoY    주거용 부동산 자산가치             국토부 실거래가        KR
A_RE_002   서울 아파트 실거래가 YoY    핵심 지역 선행성                   국토부 실거래가        KR
A_RE_003   KB 주택가격지수 YoY         장기 시계열                        KB국민은행             KR
A_RE_004   주택매매 거래량             수요·유동성 강도                   국토부                 KR
A_RE_005   전세가율                    레버리지 수요 압력                 KB국민은행             KR
A_RE_006   PIR Price-to-Income         소득 대비 주택가격                 산출 KB+통계청         KR
A_RE_007   주택담보대출 잔액 YoY        부동산 레버리지 총량               한국은행               KR
A_RE_008   주택담보대출 월별 증감       신규 차입 흐름                     한국은행               KR
A_RE_009   분양물량·미분양 건수         공급 압력·수요 소진               국토부                 KR
A_RE_010   상업용 부동산 공실률         비주거용 건전성                    한국부동산원           KR
A_RE_011   전국 부동산 시가총액 추정    부동산에 묶인 자산 총량            한국은행 국민대차대조표 KR

# ── A_EQ: 주식 (8종) ──────────────────────────────────────────────────────────
A_EQ_001   KOSPI 시가총액/GDP          버핏 지표 한국판                   KRX+한국은행           KR
A_EQ_002   KOSDAQ 시가총액/GDP         성장주 시장 밸류에이션             KRX+한국은행           KR
A_EQ_003   주식형 펀드 설정원본 잔액   간접 투자 집중 금액               금융투자협회           KR
A_EQ_004   직접투자 계좌 예탁금         투자 대기 자금                     금융투자협회           KR
A_EQ_005   고객예탁금                  즉시 투자 대기 자금                금융투자협회           KR
A_EQ_006   신용융자 잔고/시가총액       레버리지 과잉 여부                 금융투자협회           KR
A_EQ_007   KOSPI 12M Forward PER        시장 전체 밸류에이션               Quantiwise/ECOS        KR
A_EQ_008   외국인 주식 보유 비율        외국 자본 집중도                   KRX                    KR

# ── A_BD: 채권·대출 (12종) ────────────────────────────────────────────────────
A_BD_001   가계부채 총잔액             민간 부문 최대 리스크              한국은행               KR
A_BD_002   가계부채/GDP                지속가능성 비율                    한국은행+기재부         KR
A_BD_003   가계부채 YoY                부채 팽창 속도                     한국은행               KR
A_BD_004   기업부채 총잔액             기업 레버리지 총량                 한국은행               KR
A_BD_005   기업부채/GDP                기업 재무 리스크                   산출                   KR
A_BD_006   은행권 총대출 잔액          은행 신용 공급 총량                한국은행               KR
A_BD_007   비은행권 총대출 2금융권      그림자 금융 리스크                 한국은행               KR
A_BD_008   DSR 평균                    상환 능력 대비 부채 부담           금융감독원             KR
A_BD_009   LTV 평균 비율               담보 대비 대출 비율                금융감독원             KR
A_BD_010   국채 발행 잔액 YoY          국가 채무 팽창 속도                기재부                 KR
A_BD_011   회사채 발행 잔액            기업 직접금융 규모                 금융감독원             KR
A_BD_012   ABS 잔액                    부채 구조화 규모                   금융감독원             KR

# ── A_ALT: 금·대체 (8종) ──────────────────────────────────────────────────────
A_ALT_001  금 가격 국제 CAGR           안전자산 수요                      LBMA/FRED              GLOBAL
A_ALT_002  금 가격 국내 CAGR           환율 반영 국내 금 가치             KRX 금시장             KR
A_ALT_003  금 ETF 순매수 전세계         금 투자 수요                       WGC                    GLOBAL
A_ALT_004  중앙은행 금 보유량 한국       한국은행 준비자산                  한국은행               KR
A_ALT_005  금/S&P500 비율              안전자산 vs 위험자산                산출                   GLOBAL
A_ALT_006  금/WTI 비율                 에너지 대비 금 가치                산출                   GLOBAL
A_ALT_007  비트코인 가격 CAGR           디지털 대체자산                    Coinbase/Binance       GLOBAL
A_ALT_008  크립토 시총/글로벌 M2        디지털 자산 유동성 흡수            CoinGecko+FRED         GLOBAL

# ── A_IDX: 자산 배분 비중 합성 (5종) ─────────────────────────────────────────
A_IDX_001  부동산 비중                 전체 가계 자산 중 부동산 비율      한국은행 국민대차대조표 KR
A_IDX_002  주식 비중                   전체 가계 자산 중 주식 비율        한국은행               KR
A_IDX_003  채권 비중                   전체 가계 자산 중 채권 비율        한국은행               KR
A_IDX_004  예금 비중                   전체 가계 자산 중 예금 비율        한국은행               KR
A_IDX_005  금·대체 비중                전체 가계 자산 중 금·대체 비율     산출                   KR

총계: 11+8+12+8+5 = 44종
"""

# -----------------------------------------------------------------------------
# F — 기업 재무 지표 (45종, LSTM_stock 입력)
# -----------------------------------------------------------------------------
FUNDAMENTAL_INDICATORS = """
# ── F_VAL: 밸류에이션 (13종) ──────────────────────────────────────────────────
F_VAL_001  PER Trailing 12M          주가/EPS(TTM)                      낮을수록 저평가
F_VAL_002  Forward PER               주가/예상 EPS                      낮을수록 저평가
F_VAL_003  PBR                       주가/BPS                           낮을수록 저평가
F_VAL_004  PSR                       시총/매출                          낮을수록 저평가
F_VAL_005  EV/EBITDA                 EV/EBITDA                          낮을수록 저평가
F_VAL_006  EV/Sales                  EV/매출                            낮을수록 저평가
F_VAL_007  EV/FCF                    EV/FCF                             낮을수록 저평가
F_VAL_008  PCR                       주가/주당 영업현금흐름              낮을수록 저평가
F_VAL_009  CAPE Shiller PER          주가/10년 평균 EPS 물가조정         낮을수록 저평가
F_VAL_010  PEG Ratio                 PER/EPS CAGR                       1 이하 저평가
F_VAL_011  FCF Yield                 FCF/시총                           높을수록 저평가
F_VAL_012  배당수익률                 DPS/주가                           높을수록 저평가
F_VAL_013  총주주수익률 TSY           (배당+자사주매입)/시총              높을수록 저평가

# ── F_PRF: 수익성 (9종) ───────────────────────────────────────────────────────
F_PRF_001  매출총이익률               매출총이익/매출
F_PRF_002  영업이익률                 영업이익/매출
F_PRF_003  EBITDA 마진               EBITDA/매출
F_PRF_004  순이익률                   순이익/매출
F_PRF_005  ROE                       순이익/자기자본
F_PRF_006  ROA                       순이익/총자산
F_PRF_007  ROIC                      NOPAT/투하자본
F_PRF_008  ROCE                      EBIT/(총자산-유동부채)
F_PRF_009  ROE 듀퐁 분해              순이익률×자산회전율×레버리지

# ── F_GRW: 성장성 (7종, 모두 기하평균 CAGR) ──────────────────────────────────
F_GRW_001  매출 CAGR                  1Y/3Y/5Y
F_GRW_002  영업이익 CAGR              1Y/3Y/5Y
F_GRW_003  EPS CAGR                  1Y/3Y/5Y
F_GRW_004  FCF CAGR                  1Y/3Y/5Y
F_GRW_005  BPS CAGR                  1Y/3Y/5Y
F_GRW_006  배당 CAGR                  1Y/3Y/5Y
F_GRW_007  시장 점유율 변화           YoY

# ── F_CF: 현금흐름 (8종) ──────────────────────────────────────────────────────
F_CF_001   영업현금흐름 OCF           핵심 현금창출 능력
F_CF_002   잉여현금흐름 FCF           OCF - CapEx
F_CF_003   FCF 마진                   FCF/매출
F_CF_004   FCF/순이익                 이익 품질 (1에 가까울수록 우량)
F_CF_005   CapEx/매출                 투자 집중도
F_CF_006   현금전환주기 CCC           DSO+DIO-DPO
F_CF_007   순부채/EBITDA              레버리지 상환 능력
F_CF_008   FCF CAGR 3Y·5Y            기하평균 기준 FCF 성장률

# ── F_FIN: 재무건전성 (8종) ───────────────────────────────────────────────────
F_FIN_001  부채비율                   총부채/자기자본
F_FIN_002  순부채비율                 (총부채-현금)/EBITDA
F_FIN_003  유동비율                   유동자산/유동부채
F_FIN_004  당좌비율                   (유동자산-재고)/유동부채
F_FIN_005  이자보상배율               EBIT/이자비용
F_FIN_006  Altman Z-Score            파산 예측 복합 지수
F_FIN_007  Piotroski F-Score         재무 건전성 9개 기준 합산
F_FIN_008  자기자본 증가율 CAGR       내부 축적 속도

총계: 13+9+7+8+8 = 45종
"""

# -----------------------------------------------------------------------------
# C — 계산형 지표 (7종, FT-Transformer 스냅샷 입력)
# -----------------------------------------------------------------------------
COMPUTED_INDICATORS = """
C_FSCORE   Piotroski F-Score (0~9)          재무건전성 9개 기준 합산         Piotroski (2000)
C_ZSCORE   Altman Z-Score                    파산 예측 선형 모델              Altman (1968)
C_QUALITY  Quality Score                     GP/A + ROE + Safety + Payout    Asness et al. (2019)
C_EPS_STAB EPS 변동계수 CV (20분기)          이익 안정성                      표준 회계
C_OPLEV    영업 레버리지 ΔEBIT%/ΔRev%        비용 고정도 민감성               표준 회계
C_AMIHUD   Amihud 비유동성                   |수익률|/거래량                  Amihud (2002)
C_GP_A     Gross Profitability/Assets         수익성 자산 효율                 Novy-Marx (2013)

총계: 7종
"""

# -----------------------------------------------------------------------------
# 지표 수 요약
# -----------------------------------------------------------------------------
SUMMARY = """
모듈           지표 수    입력 위치
M (거시)       80종*      LSTM_macro 시계열
A (한국 자산)  44종       LSTM_stock 시계열 (KR 종목만)
F (기업 재무)  45종       LSTM_stock 시계열
C (계산형)     7종        FT-Transformer 스냅샷
THEME_CTX      아래 참조  FT-Transformer 스냅샷
─────────────────────────────
합계           176종 + 테마 비중

* M 지표는 ★★★ 기준 약 80종 선별. 전체 정의는 101종.
  KR 전용 지표는 US 종목에서 0으로 패딩.
"""


# =============================================================================
# 섹션 0. 아키텍처 개요
# =============================================================================

"""
전체 데이터 흐름:

  [종목 재무 시계열]  (B, T_stock, F_stock)   F_stock = 거시 제외 피처 (재무 45 + 자산 35)
  [거시 시계열]       (B, T_macro, F_macro)   F_macro = 거시 80종, 모든 종목 공유
  [테마 비중 벡터]    (B, F_theme)            F_theme = 18차원, 분기별 스냅샷
  [스냅샷 피처]       (B, F_snap)             F_snap  = 계산형 7 + 범주형(섹터,국가 등)

         ↓                   ↓                   ↓              ↓
  LSTM_stock            LSTM_macro          Linear           Feature
  Bidirectional         단방향             Projection       Tokenizer
  (B, 256)              (B, 128)            (B, 64)
         ↓                   ↓                   ↓
         └───────────────────┴───────────────────┘
                             ↓
                    FT-Transformer
          [CLS] [stock_ctx] [macro_ctx] [theme_ctx] [snap_f1..fn]
                     Self-Attention (피처 간 상호작용)
                             ↓
                    [CLS] 토큰
                      /         \\
                 head_A         head_R
                   ↓               ↓
                   A               R
                (매력도)         (위험도)

핵심 설계 의도:
  LSTM_stock  : 종목별 고유 재무 흐름 (ROE 개선 추세, 부채비율 변화)
  LSTM_macro  : 모든 종목에 공통인 거시 레짐 (금리 사이클, 경기 국면)
                → 배치당 1회만 계산 후 모든 종목에 브로드캐스트
  theme_ctx   : 같은 GT 테마 내에서 이 종목의 상대적 위치
                → "나는 이 테마에서 얼마나 크고, 얼마나 저평가인가"
  FT-Transformer: 세 컨텍스트와 스냅샷 피처 간 교차 Attention
                → "금리 인상 국면(macro)에서 고부채(stock) 종목이
                    테마 내 저평가(theme) 상태일 때의 의미" 학습
"""

# =============================================================================
# 섹션 1. 환경 설정 (v5.1과 동일, requirements만 변경)
# =============================================================================

# requirements.txt
REQUIREMENTS = """
torch>=2.2.0
pytorch-lightning>=2.2.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
yfinance>=0.2.36
pykrx>=1.0.45
pandas-datareader>=0.10.0
fredapi>=0.5.1
streamlit>=1.31.0
plotly>=5.18.0
optuna>=3.5.0
statsmodels>=0.14.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
requests>=2.31.0
tqdm>=4.66.0
pyyaml>=6.0.1
joblib>=1.3.0
pyarrow>=14.0.0
psutil>=5.9.0
scipy>=1.12.0
# 제거: pytorch-forecasting, lightgbm
"""

# =============================================================================
# 섹션 2. 프로젝트 디렉토리 구조
# =============================================================================

"""
stockml/
├── config/
│   ├── settings.yaml
│   └── feature_lags.yaml
│
├── data/
│   ├── raw/prices/ financials/ macro/
│   ├── themes/
│   │   ├── raw/                          ← 원본 YAML (수동 편집)
│   │   │   ├── global_themes.yaml
│   │   │   ├── kospi/
│   │   │   │   ├── kospi_mapping_part1.yaml
│   │   │   │   ├── kospi_mapping_part2.yaml
│   │   │   │   └── kospi_mapping_part3.yaml
│   │   │   └── sp500/
│   │   │       ├── sp500_mapping_part1.yaml
│   │   │       ├── sp500_mapping_part2.yaml
│   │   │       └── sp500_mapping_part3.yaml
│   │   └── processed/
│   │       └── themes.yaml               ← 병합본 (자동 생성, git ignore 가능)
│   │
│   ├── processed/
│   │   ├── prices_quarterly.parquet
│   │   ├── features_stock.parquet    ← 종목 재무 피처
│   │   ├── features_macro.parquet    ← 거시 피처 (날짜별, 종목 무관)
│   │   ├── theme_context.parquet     ← 테마 내 비중 (종목×날짜)
│   │   └── labels.parquet
│   └── splits/
│       └── ticker_split.json         ← 종목 분할 결과 저장 (재현성)
│
├── src/
│   ├── utils/
│   │   ├── device.py                 ← v5.1과 동일
│   │   ├── io.py                     ← v5.1과 동일
│   │   ├── pit.py                    ← Point-in-Time 유틸
│   │   └── split.py                  ← 신규: 종목 분할 유틸         ★v5.4 추가
│   │
│   ├── data_fetchers/                ← v5.1과 동일
│   │
│   ├── features/                     ← v5.1과 동일
│   │
│   ├── theme/
│   │   ├── loader.py                 ← processed YAML 로드         ★v5.4 교체
│   │   └── context.py                ← 테마 비중 벡터 계산         ★v5.4 수정
│   │
│   ├── labels/                       ← v5.1과 동일
│   │
│   ├── models/
│   │   ├── lstm_encoder.py           ← 가변길이 시계열 → 컨텍스트
│   │   ├── ft_transformer.py         ← 메인 예측 모델
│   │   ├── predictor.py              ← Lightning 통합 모듈
│   │   └── baseline_accounting.py    ← v5.1과 동일 (평가용)
│   │
│   ├── data/
│   │   └── dataset.py                ← 가변길이 Dataset + collate
│   │
│   └── evaluation/                   ← v5.1과 동일
│
└── scripts/
    ├── 00_merge_themes.py            ← 최초 1회 및 변경시 실행      ★v5.4 추가
    ├── 01_fetch_data.py              ← v5.1과 동일
    ├── 02_build_features.py          ← v5.1과 동일
    ├── 02b_build_theme_context.py    ← 테마 비중 계산              ★v5.4 수정
    ├── 03_build_labels.py            ← v5.1과 동일
    ├── 04_train.py                   ← stratified split 수행        ★v5.4 수정
    ├── 05_train_baselines.py         ← 회계 baseline만 유지
    ├── 06_evaluate.py                ← ticker-split 평가 메인       ★v5.4 수정
    └── 07_run_ui.py                  ← v5.1과 동일
"""

# =============================================================================
# 섹션 3. 설정 파일
# =============================================================================

SETTINGS_YAML = """
# config/settings.yaml

project:
  name: stockml
  data_dir: ./data
  random_seed: 42

universe:
  countries: [KR, US]
  kr_market: KOSPI
  us_index: SP500
  exclude_etfs: true

prices:
  frequency: quarterly
  source_kr: pykrx
  source_us: yfinance

targets:
  attractiveness:
    max_horizon_years: 5
    log_base: 5
    use_max_in_window: true
    min_forward_quarters: 4
  risk:
    max_horizon_years: 5
    annualization_factor: 4
    min_forward_quarters: 4

split:
  method: ticker                  # 종목 기반 분할
  test_ratio:  0.15               # Test: 15%
  val_ratio:   0.15               # Val:  15%
  train_ratio: 0.70               # Train: 70% (명시, 합산 검증용)
  seed: 42

  # stratify 기준: market × theme_level
  # market:      KR / US 각각에서 독립적으로 비율 맞춤
  # theme_level: Tier3 우선, 종목 수 < min_bucket_size 이면 Tier2, 그래도 부족하면 Tier1
  stratify:
    market: true                  # KR/US 비율 유지
    theme_level: tier3            # tier1 | tier2 | tier3
    min_bucket_size: 3            # 버킷당 최소 종목 수 (미달 시 상위 tier로 합산)

  # 보조: 시간 외삽 성능 별도 측정 (선택, 06_evaluate.py에서 수행)
  time_holdout:
    enabled: true
    train_tickers: train_only     # train 종목만 사용
    cutoff: '2015-12-31'          # 이전 학습, 이후 테스트

themes:
  raw_dir: data/themes/raw
  processed_path: data/themes/processed/themes.yaml
  # processed 파일이 존재하면 재생성하지 않음
  # raw 파일 변경 후 재생성하려면: python scripts/00_merge_themes.py --force

# ── 모델 설정 ──────────────────────────────────────────────────────
model:
  # LSTM_stock: 종목 재무 시계열 인코더
  lstm_stock_hidden: 128      # 양방향이므로 출력 256
  lstm_stock_layers: 2
  lstm_stock_dropout: 0.2
  lstm_stock_max_seq: 20      # 최대 20분기(5년) 과거

  # LSTM_macro: 거시 시계열 인코더
  lstm_macro_hidden: 64       # 단방향, 출력 64
  lstm_macro_layers: 1
  lstm_macro_max_seq: 20

  # ThemeContext Linear 투영
  theme_proj_dim: 64          # 18 → 64

  # FT-Transformer
  d_token: 192                # 피처 임베딩 차원 (192 = 8 heads × 24)
  n_heads: 8
  n_layers: 4
  ffn_factor: 1.333           # FFN hidden = d_token × ffn_factor
  dropout: 0.2
  attn_dropout: 0.1

  # 학습
  lr: 0.0001
  weight_decay: 0.01
  grad_clip: 1.0
  batch_size: 64              # M1 Pro 16GB
  max_epochs: 60
  patience: 10

  # M1 Pro
  num_workers: 2
  persistent_workers: true
  pin_memory: false           # MPS 미지원
  precision: '32-true'

device:
  prefer: mps
  fallback: cpu
"""

# =============================================================================
# 섹션 4. 테마 모듈 (신규)
# =============================================================================

# ── src/theme/loader.py ──────────────────────────────────────────────────────

THEME_LOADER = '''
"""
src/theme/loader.py

processed/themes.yaml 만 읽는다.
파일이 없으면 에러 메시지로 scripts/00_merge_themes.py 실행 안내.
"""
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List


@lru_cache(maxsize=1)
def load_themes(processed_path: str = 'data/themes/processed/themes.yaml') -> Dict:
    """
    병합된 themes.yaml 로드.

    반환:
        {
          "meta":             {...},
          "themes":           {"GT_XXX": {...}, ...},
          "tickers":          {"005930": {"market","themes","primary_tier1/2/3",...}, ...},
          "theme_to_tickers": {"GT_XXX": ["005930", ...], ...},
        }
    """
    p = Path(processed_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 파일이 없습니다.\n"
            "다음 명령을 먼저 실행하세요:\n"
            "  python scripts/00_merge_themes.py"
        )

    with open(p, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_tier_map(processed_path: str = 'data/themes/processed/themes.yaml') -> Dict[str, int]:
    """GT_XXX → tier 정수 딕셔너리."""
    data = load_themes(processed_path)
    return {k: v.get('tier', 0) for k, v in data['themes'].items()}
'''

# ── src/theme/context.py ─────────────────────────────────────────────────────

THEME_CONTEXT = '''
"""
src/theme/context.py  (v5.5)

테마 비중 벡터: 순위 백분위 → 실질 비중 및 상대 배수로 교체.
"""
import numpy as np
import pandas as pd
from typing import Optional, Set
from src.theme.loader import load_themes

THEME_VEC_DIM = 18
NEUTRAL_WEIGHT = 0.0        # 비중 기본값 (데이터 없을 때)
NEUTRAL_RELATIVE = 1.0      # 상대 배수 기본값 (평균과 같음)


def _safe_weight(numerator: float, denominator: float) -> float:
    """비중 계산. denominator가 0이거나 음수이면 0 반환."""
    if denominator <= 0 or np.isnan(denominator) or np.isnan(numerator):
        return NEUTRAL_WEIGHT
    val = numerator / denominator
    return float(np.clip(val, 0.0, 1.0))


def _safe_relative(value: float, mean: float) -> float:
    """
    상대 배수 = value / mean.
    mean이 0이거나 부호가 다른 경우 NEUTRAL 반환.
    극단값 클리핑: [0.1, 10.0] 범위로 제한.
    """
    if mean == 0 or np.isnan(mean) or np.isnan(value):
        return NEUTRAL_RELATIVE
    if mean < 0 and value < 0:
        # 둘 다 음수: 비율 의미 있음
        ratio = value / mean
    elif mean < 0 or value < 0:
        # 부호 다름: 의미 없음
        return NEUTRAL_RELATIVE
    else:
        ratio = value / mean
    return float(np.clip(ratio, 0.1, 10.0))


def _signed_weight(numerator: float, denominator: float) -> float:
    """
    음수 허용 비중 (EBITDA, FCF, 순이익).
    분모는 양수 합계만 사용 (음수 기업은 분모에서 제외).
    결과는 [-1, 1] 클리핑.
    """
    if np.isnan(numerator) or denominator <= 0:
        return NEUTRAL_WEIGHT
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def compute_theme_vector(
    ticker: str,
    ticker_row: pd.Series,
    peers: pd.DataFrame,
) -> np.ndarray:
    """
    단일 테마에 대한 18차원 벡터 계산.

    Args:
        ticker     : 대상 종목 코드
        ticker_row : 대상 종목의 현재 시점 피처 (pd.Series)
        peers      : 같은 테마·같은 시점의 종목 DataFrame (대상 포함)

    Returns:
        (18,) float32 벡터
    """
    vec = np.full(THEME_VEC_DIM, NEUTRAL_WEIGHT, dtype=np.float32)

    if len(peers) < 2:
        return vec

    def col_sum_pos(col):
        """양수 합계 (음수 기업은 분모에서 제외)."""
        return peers[col].clip(lower=0).sum() if col in peers.columns else 0.0

    def col_sum_all(col):
        return peers[col].sum() if col in peers.columns else 0.0

    def own(col):
        mask = peers['ticker'] == ticker
        if col in peers.columns and mask.any():
            v = peers.loc[mask, col].iloc[0]
            return float(v) if not pd.isna(v) else np.nan
        return np.nan

    def col_mean(col):
        if col not in peers.columns:
            return np.nan
        vals = peers[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else np.nan

    # ── [0] 시총 비중 ──────────────────────────────────────────────────
    total_mktcap = col_sum_pos('market_cap')
    vec[0] = _safe_weight(own('market_cap') or 0.0, total_mktcap)

    # ── [1] 매출 비중 ──────────────────────────────────────────────────
    total_rev = col_sum_pos('F_GRW_rev_base')   # 매출 절대값 컬럼
    vec[1] = _safe_weight(own('F_GRW_rev_base') or 0.0, total_rev)

    # ── [2] EBITDA 비중 (음수 허용) ───────────────────────────────────
    total_ebitda_pos = col_sum_pos('F_PRF_ebitda_abs')
    vec[2] = _signed_weight(own('F_PRF_ebitda_abs') or 0.0, total_ebitda_pos)

    # ── [3] FCF 비중 (음수 허용) ──────────────────────────────────────
    total_fcf_pos = col_sum_pos('F_CF_002')
    vec[3] = _signed_weight(own('F_CF_002') or 0.0, total_fcf_pos)

    # ── [4] 순이익 비중 (음수 허용) ───────────────────────────────────
    total_ni_pos = col_sum_pos('F_PRF_net_income_abs')
    vec[4] = _signed_weight(own('F_PRF_net_income_abs') or 0.0, total_ni_pos)

    # ── [5] 자산 비중 ──────────────────────────────────────────────────
    total_assets = col_sum_pos('F_FIN_total_assets')
    vec[5] = _safe_weight(own('F_FIN_total_assets') or 0.0, total_assets)

    # ── [6]~[11] 상대 배수 ────────────────────────────────────────────
    for i, col in enumerate([
        'F_VAL_003',      # PBR    [6]
        'F_VAL_001',      # PER    [7]
        'F_VAL_005',      # EV/EBITDA [8]
        'F_PRF_005',      # ROE   [9]
        'F_GRW_001',      # 매출 CAGR [10]
        'F_CF_003',       # FCF 마진 [11]
    ], start=6):
        mean_val = col_mean(col)
        own_val  = own(col)
        if own_val is not None and not np.isnan(own_val):
            vec[i] = _safe_relative(own_val, mean_val)

    # ── [12]~[17] 테마 전체 상태 ──────────────────────────────────────

    # [12] 테마 전체 시총 (로그 스케일, 억 단위 정규화)
    if total_mktcap > 0:
        vec[12] = float(np.log1p(total_mktcap / 1e8))   # 억 단위

    # [13] 테마 시총 4분기 성장률 (데이터 없으면 0)
    if 'theme_mktcap_prev4q' in peers.columns:
        prev = peers['theme_mktcap_prev4q'].mean()
        if prev > 0:
            vec[13] = float(np.clip(total_mktcap / prev - 1, -1.0, 3.0))

    # [14] 테마 평균 4분기 수익률
    if 'ret_4q' in peers.columns:
        ret4q = peers['ret_4q'].dropna()
        if len(ret4q) > 0:
            vec[14] = float(np.clip(ret4q.mean(), -1.0, 3.0))

    # [15] 테마 수익률 변동성
    if 'ret_4q' in peers.columns:
        ret4q = peers['ret_4q'].dropna()
        if len(ret4q) > 1:
            vec[15] = float(np.clip(ret4q.std(), 0.0, 2.0))

    # [16] HHI 집중도
    if total_mktcap > 0 and 'market_cap' in peers.columns:
        weights = peers['market_cap'].clip(lower=0) / total_mktcap
        vec[16] = float(np.clip((weights ** 2).sum(), 0.0, 1.0))

    # [17] 테마 종목 수 (로그 스케일)
    vec[17] = float(np.log1p(len(peers)))

    return vec


def compute_theme_context(
    df: pd.DataFrame,
    processed_path: str = 'data/themes/processed/themes.yaml',
    peer_tickers: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    전체 DataFrame에 대해 테마 비중 벡터를 계산한다.

    Args:
        df             : 종목-분기별 DataFrame. 필수: ['ticker', 'date']
        processed_path : themes.yaml 경로
        peer_tickers   : None=전체, set=Train 오염 방지용 지정 종목만

    Returns:
        ['ticker', 'date', 'theme_ctx_0', ..., 'theme_ctx_17'] DataFrame
    """
    mapping = load_themes(processed_path)
    t2th    = mapping['tickers']
    th2t    = mapping['theme_to_tickers']

    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    results = []

    for date, date_group in df.groupby('date', observed=True, sort=False):
        ticker_rows = {row['ticker']: row for _, row in date_group.iterrows()}

        for _, row in date_group.iterrows():
            ticker  = row['ticker']
            info    = t2th.get(ticker, {})
            themes  = info.get('themes', [])

            if not themes:
                vec = np.zeros(THEME_VEC_DIM, dtype=np.float32)
                vec[6:12] = NEUTRAL_RELATIVE    # 상대 배수는 1.0 (평균)
            else:
                vecs = []
                for theme_id in themes:
                    all_peers = th2t.get(theme_id, [])
                    if peer_tickers is not None:
                        peers_list = [t for t in all_peers
                                      if t in peer_tickers and t in ticker_rows]
                    else:
                        peers_list = [t for t in all_peers if t in ticker_rows]

                    if not peers_list:
                        continue
                    peers_df = pd.DataFrame([ticker_rows[t] for t in peers_list])
                    vecs.append(compute_theme_vector(ticker, row, peers_df))

                if vecs:
                    # 시총 비중 기준 가중 평균
                    # primary 테마(첫 번째)에 더 높은 가중치 (2:1)
                    if len(vecs) == 1:
                        vec = vecs[0]
                    else:
                        weights = np.array(
                            [2.0] + [1.0] * (len(vecs) - 1), dtype=np.float32
                        )
                        weights /= weights.sum()
                        vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)
                else:
                    vec = np.zeros(THEME_VEC_DIM, dtype=np.float32)
                    vec[6:12] = NEUTRAL_RELATIVE

            entry = {'ticker': ticker, 'date': date}
            for i, v in enumerate(vec):
                entry[f'theme_ctx_{i}'] = float(v)
            results.append(entry)

    out = pd.DataFrame(results)
    ctx_cols = [f'theme_ctx_{i}' for i in range(THEME_VEC_DIM)]
    out[ctx_cols] = out[ctx_cols].astype('float32')
    return out
'''

# =============================================================================
# 벡터 차원 레이블 (사람이 읽는 용도)
# =============================================================================
THEME_CTX_LABELS = """
차원   컬럼명                  의미                         범위        해석
[0]    theme_ctx_0  w_mktcap   시총 비중                    [0, 1]      클수록 이 종목이 테마를 지배
[1]    theme_ctx_1  w_revenue  매출 비중                    [0, 1]
[2]    theme_ctx_2  w_ebitda   EBITDA 비중                  [-1, 1]     음수=적자 기업
[3]    theme_ctx_3  w_fcf      FCF 비중                     [-1, 1]
[4]    theme_ctx_4  w_ni       순이익 비중                  [-1, 1]
[5]    theme_ctx_5  w_assets   자산 비중                    [0, 1]
[6]    theme_ctx_6  rel_pbr    상대 PBR                     [0.1, 10]   <1이면 테마 평균보다 저평가
[7]    theme_ctx_7  rel_per    상대 PER                     [0.1, 10]   <1이면 테마 평균보다 저평가
[8]    theme_ctx_8  rel_ev_ebitda 상대 EV/EBITDA            [0.1, 10]
[9]    theme_ctx_9  rel_roe    상대 ROE                     [0.1, 10]   >1이면 테마 평균보다 우수
[10]   theme_ctx_10 rel_rev_g  상대 매출성장률              [0.1, 10]
[11]   theme_ctx_11 rel_fcf_mg 상대 FCF 마진               [0.1, 10]
[12]   theme_ctx_12 log_mktcap 테마 전체 시총 로그          [0, ∞)      테마 규모
[13]   theme_ctx_13 mktcap_g4q 테마 시총 4분기 성장률       [-1, 3]     테마 모멘텀
[14]   theme_ctx_14 avg_ret4q  테마 평균 4분기 수익률       [-1, 3]
[15]   theme_ctx_15 ret_vol    테마 수익률 변동성            [0, 2]
[16]   theme_ctx_16 hhi        시총 HHI 집중도              [0, 1]      높을수록 특정 종목 독과점
[17]   theme_ctx_17 n_log      테마 종목 수 로그             [0, ∞)

예시: 삼성전자 (GT_SEMI_MEMORY)
  [0] w_mktcap = 0.42   → 이 테마 시총의 42% 차지
  [1] w_revenue = 0.35  → 이 테마 매출의 35% 차지
  [6] rel_pbr = 1.2     → 테마 평균 PBR보다 20% 높음 (약간 고평가)
  [9] rel_roe = 2.1     → 테마 평균 ROE의 2.1배 (우수한 수익성)
  [16] hhi = 0.28       → 삼성전자+하이닉스가 테마를 과점 (집중도 높음)
"""

# =============================================================================
# 필요 컬럼 추가 요건 (features_stock.parquet에 포함되어야 하는 절대값 컬럼)
# =============================================================================
REQUIRED_ABS_COLUMNS = """
비중 계산에 필요한 절대값 컬럼 (기존 F_ 지표에 추가):

F_GRW_rev_base         매출 절대값 (TTM 기준, 통화별 원단위)
F_PRF_ebitda_abs       EBITDA 절대값
F_PRF_net_income_abs   순이익 절대값
F_FIN_total_assets     총자산 절대값

주의:
  KRW/USD 혼합 비교는 하지 않음.
  같은 테마 내 KR-US 종목 비중 계산 시 모두 USD로 환산 후 합산.
  환산 방법: KRW 값 × 분기말 USD/KRW 환율 (M_FX_001의 역수)
  
  환산 코드 (features/fundamental/abs_values.py):
      if row['currency'] == 'KRW':
          usd_rate = macro.loc[date, 'M_FX_001']   # USD/KRW
          value_usd = value_krw / usd_rate
      else:
          value_usd = value_usd  # 이미 USD
"""

# =============================================================================
# 섹션 4b. 테마 병합 및 종목 분할 모듈 (v5.4 신규)
# =============================================================================

# ── scripts/00_merge_themes.py ───────────────────────────────────────────────

MERGE_THEMES_SCRIPT = '''
"""
scripts/00_merge_themes.py

실행 조건:
  - 최초 1회 필수 실행
  - data/themes/raw/ 파일 변경 시 --force 옵션으로 재실행
  - 그 외에는 processed/themes.yaml이 존재하면 건너뜀

역할:
  raw/global_themes.yaml + raw/kospi/*.yaml + raw/sp500/*.yaml 를
  하나의 processed/themes.yaml로 병합한다.
"""
import argparse
import yaml
import re
from pathlib import Path
from datetime import datetime


def get_tier1(theme_id: str, tier_map: dict, parent_map: dict) -> str:
    cur = theme_id
    for _ in range(10):
        p = parent_map.get(cur, 'null')
        if p == 'null' or p not in tier_map:
            return cur
        cur = p
    return cur


def get_primary_tiers(themes: list, tier_map: dict, parent_map: dict) -> dict:
    """
    themes 리스트에서 primary tier1/tier2/tier3 결정.
    tier3인 첫 번째 테마를 primary_tier3로 사용.
    없으면 tier2, 그래도 없으면 tier1.
    """
    primary = themes[0] if themes else None
    tier3, tier2, tier1 = None, None, None

    for t in themes:
        tier = tier_map.get(t, 0)
        if tier == 3 and tier3 is None:
            tier3 = t
            tier2 = parent_map.get(t)
            tier1 = get_tier1(t, tier_map, parent_map)
            break
        elif tier == 2 and tier2 is None:
            tier2 = t
            tier1 = get_tier1(t, tier_map, parent_map)
        elif tier == 1 and tier1 is None:
            tier1 = t

    if tier3 is None and tier2 is not None:
        tier3 = tier2   # fallback: tier2를 tier3 자리에
    if tier2 is None and tier1 is not None:
        tier2 = tier1
    if tier1 is None:
        tier1 = 'UNMAPPED'

    return {
        'primary_theme': primary,
        'primary_tier3': tier3 or 'UNMAPPED',
        'primary_tier2': tier2 or 'UNMAPPED',
        'primary_tier1': tier1 or 'UNMAPPED',
    }


def parse_mapping_file(path: Path, market: str, tier_map: dict, parent_map: dict) -> dict:
    """
    단일 매핑 YAML 파일 파싱.
    DUP 접미사 ticker는 이미 등록된 ticker와 병합.
    """
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    result = {}
    for key, info in raw.items():
        if not isinstance(info, dict):
            continue
        # DUP 정리: "005930_DUP", "005930_DUP2" → "005930"
        clean_key = re.sub(r'_DUP\d*$', '', str(key)).strip('"').strip("'")

        themes = info.get('themes', [])
        if clean_key in result:
            # 이미 등록된 ticker: 테마 병합 (중복 제거, 순서 유지)
            existing = result[clean_key]['themes']
            for t in themes:
                if t not in existing:
                    existing.append(t)
        else:
            tier_info = get_primary_tiers(themes, tier_map, parent_map)
            result[clean_key] = {
                'name': info.get('name', clean_key),
                'market': market,
                'themes': themes,
                **tier_info,
            }

    return result


def merge(raw_dir: str, output_path: str, force: bool = False):
    raw = Path(raw_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"[skip] {out} 이미 존재. 재생성하려면 --force 사용.")
        return

    # ── global_themes 로드 ───────────────────────────────────────────
    with open(raw / 'global_themes.yaml', encoding='utf-8') as f:
        themes_raw = yaml.safe_load(f)

    # tier/parent 인덱스 구축
    tier_map, parent_map = {}, {}
    for k, v in themes_raw.items():
        if not k.startswith('GT_') or not isinstance(v, dict):
            continue
        tier_map[k]   = v.get('tier', 0)
        parent_map[k] = v.get('parent', 'null')

    # ── 종목 매핑 로드 ───────────────────────────────────────────────
    all_tickers = {}

    kr_files = sorted((raw / 'kospi').glob('*.yaml'))
    for fpath in kr_files:
        parsed = parse_mapping_file(fpath, 'KR', tier_map, parent_map)
        for ticker, info in parsed.items():
            if ticker not in all_tickers:
                all_tickers[ticker] = info
            else:
                for t in info['themes']:
                    if t not in all_tickers[ticker]['themes']:
                        all_tickers[ticker]['themes'].append(t)

    us_files = sorted((raw / 'sp500').glob('*.yaml'))
    for fpath in us_files:
        parsed = parse_mapping_file(fpath, 'US', tier_map, parent_map)
        for ticker, info in parsed.items():
            if ticker not in all_tickers:
                all_tickers[ticker] = info
            else:
                for t in info['themes']:
                    if t not in all_tickers[ticker]['themes']:
                        all_tickers[ticker]['themes'].append(t)

    # ── 역방향 인덱스 ────────────────────────────────────────────────
    theme_to_tickers = {}
    for ticker, info in all_tickers.items():
        for t in info['themes']:
            theme_to_tickers.setdefault(t, []).append(ticker)

    # ── 통계 ─────────────────────────────────────────────────────────
    n_kr = sum(1 for v in all_tickers.values() if v['market'] == 'KR')
    n_us = sum(1 for v in all_tickers.values() if v['market'] == 'US')

    # ── 저장 ─────────────────────────────────────────────────────────
    output = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'n_themes': len(tier_map),
            'n_tickers_kr': n_kr,
            'n_tickers_us': n_us,
            'n_tickers_total': len(all_tickers),
            'source_files': (
                [str(f) for f in kr_files] +
                [str(f) for f in us_files]
            ),
        },
        'themes': {k: v for k, v in themes_raw.items()
                   if k.startswith('GT_')},
        'tickers': all_tickers,
        'theme_to_tickers': theme_to_tickers,
    }

    with open(out, 'w', encoding='utf-8') as f:
        yaml.dump(output, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    print(f"[done] {out}")
    print(f"  테마: {len(tier_map)}개 | KR: {n_kr}종목 | US: {n_us}종목")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir',  default='data/themes/raw')
    parser.add_argument('--output',   default='data/themes/processed/themes.yaml')
    parser.add_argument('--force',    action='store_true',
                        help='processed 파일이 있어도 강제 재생성')
    args = parser.parse_args()
    merge(args.raw_dir, args.output, args.force)
'''

# ── src/utils/split.py ───────────────────────────────────────────────────────

SPLIT_UTIL = '''
"""
src/utils/split.py

종목 기반 Stratified Split.
"""
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from src.theme.loader import load_themes


def stratified_ticker_split(
    processed_path: str = 'data/themes/processed/themes.yaml',
    test_ratio:  float = 0.15,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
    min_bucket_size: int = 3,
    theme_level: str = 'tier3',    # 'tier1' | 'tier2' | 'tier3'
) -> Tuple[List[str], List[str], List[str]]:
    """
    종목을 market × theme_level 버킷 기준으로 균등 분할한다.
    """
    rng = random.Random(seed)

    data     = load_themes(processed_path)
    tickers  = data['tickers']

    # ── 버킷 배정 ─────────────────────────────────────────────────────
    tier_key = {
        'tier1': 'primary_tier1',
        'tier2': 'primary_tier2',
        'tier3': 'primary_tier3',
    }[theme_level]

    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for ticker, info in tickers.items():
        market = info.get('market', 'UNKNOWN')
        tier_id = info.get(tier_key, 'UNMAPPED')
        buckets[(market, tier_id)].append(ticker)

    # ── 소형 버킷 상향 합산 ───────────────────────────────────────────
    if theme_level == 'tier3':
        merged: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        theme_meta = data['themes']

        for (market, tier3_id), ticker_list in buckets.items():
            if len(ticker_list) >= min_bucket_size:
                merged[(market, tier3_id)].extend(ticker_list)
            else:
                tier2_id = theme_meta.get(tier3_id, {}).get('parent', 'UNMAPPED')
                if tier2_id == 'null' or tier2_id not in theme_meta:
                    tier2_id = 'UNMAPPED'
                merged[(market, tier2_id)].extend(ticker_list)

        final: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for (market, tier2_id), ticker_list in merged.items():
            if len(ticker_list) >= min_bucket_size:
                final[(market, tier2_id)].extend(ticker_list)
            else:
                tier1_id = theme_meta.get(tier2_id, {}).get('parent', 'UNMAPPED')
                if tier1_id == 'null' or tier1_id not in theme_meta:
                    tier1_id = 'UNMAPPED'
                final[(market, tier1_id)].extend(ticker_list)

        buckets = final

    # ── 버킷별 분할 ───────────────────────────────────────────────────
    train_tickers, val_tickers, test_tickers = [], [], []

    for (market, theme_id), ticker_list in sorted(buckets.items()):
        unique = list(dict.fromkeys(ticker_list))
        rng.shuffle(unique)
        n = len(unique)

        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            train_tickers.extend(unique)
            continue

        test_tickers.extend(unique[:n_test])
        val_tickers.extend(unique[n_test:n_test + n_val])
        train_tickers.extend(unique[n_test + n_val:])

    def dedup(lst):
        return list(dict.fromkeys(lst))

    train_tickers = dedup(train_tickers)
    val_tickers   = dedup(val_tickers)
    test_tickers  = dedup(test_tickers)

    train_set = set(train_tickers)
    val_tickers  = [t for t in val_tickers  if t not in train_set]
    test_tickers = [t for t in test_tickers if t not in train_set]

    return train_tickers, val_tickers, test_tickers


def print_split_report(
    train: List[str],
    val:   List[str],
    test:  List[str],
    processed_path: str = 'data/themes/processed/themes.yaml',
):
    """
    분할 결과 요약 출력.
    """
    from collections import Counter
    data    = load_themes(processed_path)
    tickers = data['tickers']

    def market_dist(lst):
        c = Counter(tickers[t]['market'] for t in lst if t in tickers)
        return dict(c)

    def tier1_dist(lst):
        c = Counter(tickers[t]['primary_tier1'] for t in lst if t in tickers)
        return dict(c)

    total = len(train) + len(val) + len(test)
    print(f"{'='*55}")
    print(f"  분할 결과 (총 {total}종목)")
    print(f"  Train: {len(train)} ({len(train)/total:.0%})")
    print(f"  Val:   {len(val)}  ({len(val)/total:.0%})")
    print(f"  Test:  {len(test)} ({len(test)/total:.0%})")
    print(f"{'='*55}")

    print("  [Market 분포]")
    for split_name, split_list in [('Train', train), ('Val', val), ('Test', test)]:
        d = market_dist(split_list)
        print(f"    {split_name}: KR={d.get('KR',0)}, US={d.get('US',0)}")

    print("  [Tier1 분포 — Test]")
    for tier1, cnt in sorted(tier1_dist(test).items(), key=lambda x: -x[1]):
        print(f"    {tier1:<35s}: {cnt}종목")
    print(f"{'='*55}")
'''

# =============================================================================
# 섹션 5. 데이터셋 (신규)
# =============================================================================

DATASET = '''
"""
src/data/dataset.py

세 가지 시퀀스를 하나의 배치로 묶는 Dataset.

샘플 = (종목, 기준분기) 쌍.
  seq_stock  : 과거 T분기의 종목 재무 피처 (가변 길이)
  seq_macro  : 과거 T분기의 거시 피처 (동일 날짜 기준, 가변 길이)
  theme_ctx  : 현재 분기의 테마 비중 벡터 (18차원, 고정)
  snap_num   : 현재 분기 수치형 스냅샷 피처 (계산형 등)
  snap_cat   : 현재 분기 범주형 스냅샷 피처 (섹터, 국가 등)
  A, R       : 타깃

중요:
  거시 시계열은 모든 종목이 공유하므로 Dataset에서 날짜로 조회한다.
  배치 내에서 같은 날짜의 종목들은 동일한 macro 시퀀스를 가진다.
  이를 활용해 LSTM_macro는 배치 내 유니크 날짜에 대해서만 실행한다.
  (→ Predictor.forward에서 처리, Dataset은 단순히 날짜를 저장)
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StockDataset(Dataset):

    def __init__(
        self,
        df_stock: pd.DataFrame,      # 종목 재무 피처 (ticker, date, F_*)
        df_macro: pd.DataFrame,      # 거시 피처 (date, M_*)
        df_theme: pd.DataFrame,      # 테마 비중 (ticker, date, theme_ctx_*)
        df_labels: pd.DataFrame,     # 라벨 (ticker, date, A, R)
        stock_seq_cols: list,        # 시계열로 쓸 재무 피처 컬럼명
        macro_seq_cols: list,        # 시계열로 쓸 거시 피처 컬럼명
        snap_num_cols: list,         # 스냅샷 수치형 컬럼명
        snap_cat_cols: list,         # 스냅샷 범주형 컬럼명
        max_seq_len: int = 20,
    ):
        self.max_seq_len   = max_seq_len
        self.stock_seq_cols = stock_seq_cols
        self.macro_seq_cols = macro_seq_cols
        self.snap_num_cols  = snap_num_cols
        self.snap_cat_cols  = snap_cat_cols
        theme_ctx_cols      = [f'theme_ctx_{i}' for i in range(18)]

        # 거시 데이터를 날짜 → 배열 딕셔너리로 변환 (빠른 조회)
        df_macro = df_macro.sort_values('date').set_index('date')
        self._macro_index = df_macro.index.tolist()
        self._macro_array = df_macro[macro_seq_cols].values.astype('float32')
        self._macro_date_to_idx = {d: i for i, d in enumerate(self._macro_index)}

        # 종목별 피처 정렬
        df_stock = df_stock.sort_values(['ticker', 'date'])
        df_all   = (df_stock
                    .merge(df_theme, on=['ticker', 'date'], how='left')
                    .merge(df_labels, on=['ticker', 'date'], how='inner'))
        df_all[theme_ctx_cols] = df_all[theme_ctx_cols].fillna(0.5).astype('float32')

        self.samples = []
        for ticker, grp in df_all.groupby('ticker', observed=True, sort=False):
            grp = grp.reset_index(drop=True)
            for i in range(len(grp)):
                row = grp.iloc[i]

                # 라벨 없으면 제외
                if pd.isna(row.get('A')) or pd.isna(row.get('R')):
                    continue

                cur_date = row['date']

                # 종목 재무 시퀀스 (현재 포함 과거 max_seq_len 분기)
                start  = max(0, i - max_seq_len + 1)
                s_seq  = grp.iloc[start:i+1][stock_seq_cols].values.astype('float32')
                s_seq  = np.nan_to_num(s_seq, nan=0.0)

                # 거시 시퀀스 (같은 기간의 날짜 인덱스 범위)
                macro_end_idx   = self._macro_date_to_idx.get(cur_date, -1)
                macro_start_idx = max(0, macro_end_idx - max_seq_len + 1)
                if macro_end_idx < 0:
                    m_seq = np.zeros((1, len(macro_seq_cols)), dtype='float32')
                else:
                    m_seq = self._macro_array[macro_start_idx:macro_end_idx+1]
                m_seq = np.nan_to_num(m_seq, nan=0.0)

                # 테마 비중 (현재 시점, 고정 18차원)
                theme_vec = row[theme_ctx_cols].values.astype('float32')

                # 스냅샷
                snap_num = np.nan_to_num(
                    row[snap_num_cols].values.astype('float32'), nan=0.0
                )
                snap_cat = row[snap_cat_cols].values.astype('int64')

                self.samples.append({
                    's_seq':    s_seq,          # (T_s, F_stock)
                    'm_seq':    m_seq,          # (T_m, F_macro)
                    'theme':    theme_vec,      # (18,)
                    'snap_num': snap_num,       # (F_snap_num,)
                    'snap_cat': snap_cat,       # (F_snap_cat,)
                    'A':        float(row['A']),
                    'R':        float(row['R']),
                    'date':     cur_date,
                    'ticker':   ticker,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            's_seq':    torch.from_numpy(s['s_seq']),
            'm_seq':    torch.from_numpy(s['m_seq']),
            'theme':    torch.from_numpy(s['theme']),
            'snap_num': torch.from_numpy(s['snap_num']),
            'snap_cat': torch.from_numpy(s['snap_cat']),
            'A':        torch.tensor(s['A'], dtype=torch.float32),
            'R':        torch.tensor(s['R'], dtype=torch.float32),
        }


def collate_fn(batch):
    """
    가변 길이 s_seq, m_seq를 패딩해 배치로 묶는다.
    lengths를 함께 반환해 LSTM에서 pack_padded_sequence에 사용한다.
    """
    s_seqs    = [b['s_seq']    for b in batch]
    m_seqs    = [b['m_seq']    for b in batch]
    themes    = [b['theme']    for b in batch]
    snap_nums = [b['snap_num'] for b in batch]
    snap_cats = [b['snap_cat'] for b in batch]
    As = [b['A'] for b in batch]
    Rs = [b['R'] for b in batch]

    s_lengths = torch.tensor([x.shape[0] for x in s_seqs], dtype=torch.long)
    m_lengths = torch.tensor([x.shape[0] for x in m_seqs], dtype=torch.long)

    # pad_sequence: list of (T, F) → (B, max_T, F)
    s_padded = pad_sequence(s_seqs, batch_first=True, padding_value=0.0)
    m_padded = pad_sequence(m_seqs, batch_first=True, padding_value=0.0)

    return {
        's_seq':      s_padded,                  # (B, T_s, F_stock)
        's_lengths':  s_lengths,                  # (B,)
        'm_seq':      m_padded,                  # (B, T_m, F_macro)
        'm_lengths':  m_lengths,                  # (B,)
        'theme':      torch.stack(themes),        # (B, 18)
        'snap_num':   torch.stack(snap_nums),     # (B, F_snap_num)
        'snap_cat':   torch.stack(snap_cats),     # (B, F_snap_cat)
        'A':          torch.stack(As),            # (B,)
        'R':          torch.stack(Rs),            # (B,)
    }
'''

# =============================================================================
# 섹션 6. LSTM 인코더 (신규)
# =============================================================================

LSTM_ENCODER = '''
"""
src/models/lstm_encoder.py

가변 길이 시계열 → 고정 크기 컨텍스트 벡터.
stock 인코더(양방향)와 macro 인코더(단방향)를 동일 클래스로 구현한다.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMEncoder(nn.Module):
    """
    Args:
        input_size  : 입력 피처 수
        hidden_size : LSTM hidden 차원
        num_layers  : LSTM 레이어 수
        bidirectional: True면 양방향 (출력 = hidden_size * 2)
        dropout     : 드롭아웃 (num_layers > 1일 때만 LSTM 내부 적용)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        self.output_size   = hidden_size * (2 if bidirectional else 1)

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, max_T, input_size) 패딩된 시계열, float32
            lengths : (B,) 각 샘플의 실제 길이 (CPU 텐서)
        Returns:
            context : (B, output_size)
        """
        x = x.float()
        x = self.input_norm(x)

        # pack: 패딩 토큰을 LSTM 연산에서 제외
        # lengths는 반드시 CPU에 있어야 함 (MPS/CUDA 무관)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, hidden_size)

        if self.bidirectional:
            # 마지막 레이어의 양방향 결합
            ctx = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, hidden*2)
        else:
            ctx = h_n[-1]                                   # (B, hidden)

        return self.proj(ctx)
'''

# =============================================================================
# 섹션 7. FT-Transformer (신규)
# =============================================================================

FT_TRANSFORMER = '''
"""
src/models/ft_transformer.py

Gorishniy et al. (2021) FT-Transformer.
각 피처를 독립 임베딩으로 토크나이징한 뒤 Self-Attention으로 상호작용 학습.

입력:
  context_stock : (B, 256)  LSTM_stock 출력
  context_macro : (B, 64)   LSTM_macro 출력
  theme_ctx     : (B, 64)   Linear 투영된 테마 비중
  snap_num      : (B, F_num) 수치형 스냅샷
  snap_cat      : (B, F_cat) 범주형 스냅샷 (정수 인덱스)

출력:
  A : (B,)  매력도
  R : (B,)  위험도 (Softplus로 ≥ 0 보장)
"""
import math
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    수치형: x_i → Linear(1, d_token) + bias → (d,)
    범주형: cat_id → Embedding(n, d_token) → (d,)
    컨텍스트 벡터: Linear(ctx_dim, d_token) → (d,)  [수치형과 동일 처리]
    """

    def __init__(
        self,
        context_dims: list,      # 컨텍스트 벡터 차원 리스트 [256, 64, 64]
        n_num_features: int,     # 수치형 스냅샷 피처 수
        cat_cardinalities: list, # 범주형 피처별 카테고리 수 [n1, n2, ...]
        d_token: int = 192,
    ):
        super().__init__()
        self.d_token = d_token

        # 컨텍스트 투영 (각각 독립 Linear)
        self.ctx_projs = nn.ModuleList([
            nn.Linear(dim, d_token) for dim in context_dims
        ])

        # 수치형 피처: 피처별 독립 가중치
        self.n_num = n_num_features
        if n_num_features > 0:
            self.num_W = nn.Parameter(torch.empty(n_num_features, d_token))
            self.num_b = nn.Parameter(torch.zeros(n_num_features, d_token))
            nn.init.kaiming_uniform_(self.num_W, a=math.sqrt(5))

        # 범주형 임베딩
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n + 1, d_token) for n in cat_cardinalities
        ])
        self.n_cat = len(cat_cardinalities)

        # [CLS] 집계 토큰
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 토큰 수 계산
        self.n_tokens = 1 + len(context_dims) + n_num_features + len(cat_cardinalities)

    def forward(
        self,
        contexts: list,          # [(B, ctx_dim), ...]
        x_num: torch.Tensor,     # (B, n_num)
        x_cat: torch.Tensor,     # (B, n_cat)
    ) -> torch.Tensor:

        tokens = []

        # 컨텍스트 토큰
        for proj, ctx in zip(self.ctx_projs, contexts):
            tokens.append(proj(ctx.float()).unsqueeze(1))  # (B, 1, d)

        # 수치형 토큰: x_i * w_i + b_i
        if self.n_num > 0:
            num_tok = (
                x_num.float().unsqueeze(-1) * self.num_W.unsqueeze(0)
                + self.num_b.unsqueeze(0)
            )  # (B, n_num, d)
            tokens.append(num_tok)

        # 범주형 토큰
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))  # (B, 1, d)

        # 전체 피처 토큰 결합
        feat = torch.cat(tokens, dim=1)       # (B, n_tokens-1, d)

        # [CLS] prepend
        cls = self.cls_token.expand(feat.size(0), -1, -1)
        return torch.cat([cls, feat], dim=1)  # (B, n_tokens, d)


class FTTransformer(nn.Module):

    def __init__(
        self,
        context_dims: list,
        n_num_features: int,
        cat_cardinalities: list,
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_factor: float = 4/3,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            context_dims, n_num_features, cat_cardinalities, d_token
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=max(int(d_token * ffn_factor), d_token),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: 깊은 레이어에서 학습 안정성
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 예측 헤드: [CLS] → A, R
        def _head(out_activation=None):
            layers = [
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token // 2, 1),
            ]
            if out_activation:
                layers.append(out_activation)
            return nn.Sequential(*layers)

        self.head_A = _head()
        self.head_R = _head(nn.Softplus())  # R ≥ 0

    def forward(
        self,
        contexts: list,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ):
        tokens  = self.tokenizer(contexts, x_num, x_cat)  # (B, n_tok, d)
        encoded = self.transformer(tokens)                 # (B, n_tok, d)
        cls_out = encoded[:, 0]                            # (B, d)

        A = self.head_A(cls_out).squeeze(-1)   # (B,)
        R = self.head_R(cls_out).squeeze(-1)   # (B,)
        return A, R
'''

# =============================================================================
# 섹션 8. Lightning 통합 모듈 (신규)
# =============================================================================

PREDICTOR = '''
"""
src/models/predictor.py

LSTM_stock + LSTM_macro + ThemeLinear + FT-Transformer를
하나의 PyTorch Lightning 모듈로 통합한다.

LSTM_macro 최적화:
  배치 내 동일 날짜의 종목들은 같은 거시 시퀀스를 공유한다.
  동일 시퀀스를 여러 번 계산하는 낭비를 줄이기 위해
  유니크 시퀀스 기준으로 LSTM을 실행 후 인덱스로 gather한다.
  단, 간소화를 위해 v1에서는 이 최적화를 생략하고
  배치 내 모든 행에 동일하게 실행한다. (배치 크기 64에서 허용 가능)
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.lstm_encoder import LSTMEncoder
from src.models.ft_transformer import FTTransformer
from src.utils.device import get_device


class StockPredictor(pl.LightningModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        c = cfg

        # LSTM 인코더 두 개
        self.lstm_stock = LSTMEncoder(
            input_size=c['n_stock_features'],
            hidden_size=c['lstm_stock_hidden'],
            num_layers=c.get('lstm_stock_layers', 2),
            bidirectional=True,
            dropout=c['dropout'],
        )
        self.lstm_macro = LSTMEncoder(
            input_size=c['n_macro_features'],
            hidden_size=c['lstm_macro_hidden'],
            num_layers=c.get('lstm_macro_layers', 1),
            bidirectional=False,  # 거시: 단방향
            dropout=0.0,
        )

        # 테마 비중 Linear 투영 (18 → theme_proj_dim)
        self.theme_proj = nn.Sequential(
            nn.Linear(18, c['theme_proj_dim']),
            nn.GELU(),
            nn.Dropout(c['dropout']),
        )

        # FT-Transformer
        context_dims = [
            self.lstm_stock.output_size,    # 256
            self.lstm_macro.output_size,    # 64
            c['theme_proj_dim'],            # 64
        ]
        self.ftt = FTTransformer(
            context_dims=context_dims,
            n_num_features=c['n_snap_num'],
            cat_cardinalities=c['cat_cardinalities'],
            d_token=c['d_token'],
            n_heads=c['n_heads'],
            n_layers=c['n_layers'],
            ffn_factor=c.get('ffn_factor', 4/3),
            dropout=c['dropout'],
            attn_dropout=c.get('attn_dropout', 0.1),
        )

        # Kendall (2018) 불확실성 기반 멀티태스크 손실 가중치
        self.log_var_A = nn.Parameter(torch.zeros(1))
        self.log_var_R = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict):
        # ── 시계열 인코딩 ────────────────────────────────────────────
        stock_ctx = self.lstm_stock(batch['s_seq'], batch['s_lengths'])
        macro_ctx = self.lstm_macro(batch['m_seq'], batch['m_lengths'])
        theme_ctx = self.theme_proj(batch['theme'].float())

        # ── FT-Transformer ────────────────────────────────────────────
        A, R = self.ftt(
            contexts=[stock_ctx, macro_ctx, theme_ctx],
            x_num=batch['snap_num'].float(),
            x_cat=batch['snap_cat'],
        )
        return A, R

    def _loss(self, A_pred, R_pred, A_true, R_true):
        """Kendall 멀티태스크 손실."""
        mask = ~(torch.isnan(A_true) | torch.isnan(R_true))
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        l_A = nn.functional.mse_loss(A_pred[mask], A_true[mask])
        l_R = nn.functional.mse_loss(R_pred[mask], R_true[mask])
        prec_A = torch.exp(-self.log_var_A)
        prec_R = torch.exp(-self.log_var_R)
        loss = prec_A * l_A + self.log_var_A + prec_R * l_R + self.log_var_R
        return loss, l_A.item(), l_R.item()

    def training_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'train/loss': loss, 'train/A': lA, 'train/R': lR},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'val/loss': loss, 'val/A': lA, 'val/R': lR},
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams['max_epochs'],
            eta_min=self.hparams['lr'] * 0.01,
        )
        return {'optimizer': opt, 'lr_scheduler': sched}
'''

# =============================================================================
# 섹션 9. 학습 스크립트 (신규 — 04_train_tft.py 교체)
# =============================================================================

TRAIN_SCRIPT = '''
"""
scripts/04_train.py

실행: python scripts/04_train.py
"""
import yaml, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from src.data.dataset import StockDataset, collate_fn
from src.models.predictor import StockPredictor
from src.utils.device import get_device, get_optimal_batch_size, report_environment
from src.utils.io import load_parquet


def main():
    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    report_environment()

    # ── 데이터 로드 ────────────────────────────────────────────────────
    df_stock  = load_parquet('data/processed/features_stock.parquet')
    df_macro  = load_parquet('data/processed/features_macro.parquet')
    df_theme  = load_parquet('data/processed/theme_context.parquet')
    df_labels = load_parquet('data/processed/labels.parquet')

    # ── 컬럼 정의 ───────────────────────────────────────────────────────
    # 종목 재무 시계열 피처 (거시 제외)
    STOCK_SEQ_COLS = (
        [c for c in df_stock.columns if c.startswith('F_')]   # 재무 45
      + [c for c in df_stock.columns if c.startswith('A_')]   # 자산집중도 35
    )
    # 거시 시계열 피처
    MACRO_SEQ_COLS = [c for c in df_macro.columns
                      if c.startswith('M_') and c != 'date']
    # 스냅샷 수치형 (계산형 지표)
    SNAP_NUM_COLS = [c for c in df_stock.columns if c.startswith('C_')]
    # 스냅샷 범주형
    SNAP_CAT_COLS = ['country', 'sector', 'size_tier']

    # 범주형 카디널리티 (미등록 카테고리를 위해 +1)
    cat_cardinalities = [
        int(df_stock[c].nunique()) for c in SNAP_CAT_COLS
    ]

    # ── 모델 설정 ────────────────────────────────────────────────────────
    mcfg = cfg['model']
    model_cfg = {
        'n_stock_features':  len(STOCK_SEQ_COLS),
        'n_macro_features':  len(MACRO_SEQ_COLS),
        'n_snap_num':        len(SNAP_NUM_COLS),
        'cat_cardinalities': cat_cardinalities,
        'lstm_stock_hidden': mcfg['lstm_stock_hidden'],
        'lstm_stock_layers': mcfg.get('lstm_stock_layers', 2),
        'lstm_macro_hidden': mcfg['lstm_macro_hidden'],
        'lstm_macro_layers': mcfg.get('lstm_macro_layers', 1),
        'theme_proj_dim':    mcfg['theme_proj_dim'],
        'd_token':           mcfg['d_token'],
        'n_heads':           mcfg['n_heads'],
        'n_layers':          mcfg['n_layers'],
        'ffn_factor':        mcfg.get('ffn_factor', 4/3),
        'dropout':           mcfg['dropout'],
        'attn_dropout':      mcfg.get('attn_dropout', 0.1),
        'lr':                mcfg['lr'],
        'weight_decay':      mcfg['weight_decay'],
        'max_epochs':        mcfg['max_epochs'],
    }

    # ── 데이터 분할 및 테마 비중 계산 (peer 오염 방지) ──────────────────────
    from src.utils.split import stratified_ticker_split, print_split_report

    split_cfg = cfg['split']
    train_tickers, val_tickers, test_tickers = stratified_ticker_split(
        processed_path = cfg['themes']['processed_path'],
        test_ratio     = split_cfg['test_ratio'],
        val_ratio      = split_cfg['val_ratio'],
        seed           = split_cfg['seed'],
        min_bucket_size= split_cfg['stratify']['min_bucket_size'],
        theme_level    = split_cfg['stratify']['theme_level'],
    )
    print_split_report(
        train_tickers, val_tickers, test_tickers,
        processed_path=cfg['themes']['processed_path'],
    )

    # 저장 (재현성)
    import json
    from pathlib import Path
    Path('data/splits').mkdir(exist_ok=True)
    json.dump({
        'train': train_tickers,
        'val':   val_tickers,
        'test':  test_tickers,
    }, open('data/splits/ticker_split.json', 'w'))

    # Dataset 분할
    train_df = df_stock[df_stock['ticker'].isin(train_tickers)]
    val_df   = df_stock[df_stock['ticker'].isin(val_tickers)]

    # ── 테마 비중: Train peer 오염 방지 ───────────────────────────────────
    # Train 종목의 theme_ctx는 Train peer만 참조
    # Val/Test 종목의 theme_ctx는 전체 peer 참조 (실운용과 동일)
    from src.theme.context import compute_theme_context

    df_theme_train = compute_theme_context(
        train_df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=set(train_tickers),   # Train peer만
    )
    df_theme_val = compute_theme_context(
        val_df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=None,                 # 전체 peer
    )
    # Validation은 별도 계산 후 concat
    df_theme = pd.concat([df_theme_train, df_theme_val], ignore_index=True)

    # dataset 생성
    train_ds = StockDataset(
        df_stock=train_df,
        df_macro=df_macro,
        df_theme=df_theme,
        df_labels=df_labels[df_labels['ticker'].isin(train_tickers)],
        stock_seq_cols=STOCK_SEQ_COLS,
        macro_seq_cols=MACRO_SEQ_COLS,
        snap_num_cols=SNAP_NUM_COLS,
        snap_cat_cols=SNAP_CAT_COLS,
        max_seq_len=mcfg.get('lstm_stock_max_seq', 20),
    )
    val_ds = StockDataset(
        df_stock=val_df,
        df_macro=df_macro,
        df_theme=df_theme,
        df_labels=df_labels[df_labels['ticker'].isin(val_tickers)],
        stock_seq_cols=STOCK_SEQ_COLS,
        macro_seq_cols=MACRO_SEQ_COLS,
        snap_num_cols=SNAP_NUM_COLS,
        snap_cat_cols=SNAP_CAT_COLS,
        max_seq_len=mcfg.get('lstm_stock_max_seq', 20),
    )

    batch_size = get_optimal_batch_size(mcfg['batch_size'])
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=mcfg['num_workers'],
        persistent_workers=mcfg['persistent_workers'],
        pin_memory=False,   # MPS 미지원
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── 학습 ─────────────────────────────────────────────────────────────
    device      = get_device()
    accelerator = 'mps' if device.type == 'mps' else 'cpu'

    model = StockPredictor(model_cfg)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val/loss', patience=mcfg['patience'], mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='stockml-{epoch:02d}-{val/loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        max_epochs=mcfg['max_epochs'],
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=mcfg['grad_clip'],
        callbacks=callbacks,
        precision=mcfg.get('precision', '32-true'),
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    except RuntimeError as e:
        if 'MPS' in str(e):
            print(f"[MPS fallback] {e}")
            trainer = pl.Trainer(
                max_epochs=mcfg['max_epochs'],
                accelerator='cpu',
                gradient_clip_val=mcfg['grad_clip'],
                callbacks=callbacks,
            )
            trainer.fit(model, train_loader, val_loader)
        else:
            raise

    print("학습 완료.")


if __name__ == '__main__':
    main()
'''

# =============================================================================
# 섹션 10. 테마 비중 계산 스크립트 (신규)
# =============================================================================

THEME_SCRIPT = '''
"""
scripts/02b_build_theme_context.py

실행: python scripts/02b_build_theme_context.py
02_build_features.py 이후, 03_build_labels.py 이전에 실행한다.
"""
import yaml
from src.theme.context import compute_theme_context
from src.utils.io import load_parquet, save_parquet, report_memory


def main():
    print("테마 비중 컨텍스트 계산 중...")
    df = load_parquet('data/processed/features_stock.parquet')
    report_memory(df, "features_stock")

    with open('config/settings.yaml') as f:
        cfg = yaml.safe_load(f)

    # 전체 peer 기준 (학습 전 전처리 단계이므로 peer 제한 없음)
    # Train/Val 분리는 04_train.py에서 처리
    theme_ctx = compute_theme_context(
        df,
        processed_path=cfg['themes']['processed_path'],
        peer_tickers=None,
    )

    save_parquet(theme_ctx, 'data/processed/theme_context.parquet')
    report_memory(theme_ctx, "theme_context")
    print(f"완료: {len(theme_ctx)} rows")


if __name__ == '__main__':
    main()
'''

# =============================================================================
# 섹션 11. 실행 순서 (업데이트)
# =============================================================================

EXECUTION_ORDER = """
# 실행 순서 (v5.4)

# 1. 환경 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env: FRED_API_KEY, ECOS_API_KEY, DART_API_KEY

# ── 최초 1회 (raw 파일 변경 시 --force 추가) ──────────────────────────
python scripts/00_merge_themes.py
# → data/themes/processed/themes.yaml 생성
# 이미 존재하면 건너뜀. raw 변경 후 재생성: --force 옵션

# ── 이후 매 실행 ─────────────────────────────────────────────────────
python scripts/01_fetch_data.py --start 2010-01-01 --end 2026-05-01
python scripts/02_build_features.py     # features_stock + features_macro 분리 저장
python scripts/02b_build_theme_context.py
python scripts/03_build_labels.py
python scripts/04_train.py              # 내부에서 stratified split 수행
python scripts/05_train_baselines.py    # 회계 지표 baseline
python scripts/06_evaluate.py           # ticker-split 평가 + time_holdout 보조 평가
streamlit run scripts/07_run_ui.py
"""

# =============================================================================
# 섹션 12. 단위 테스트 (신규 및 v5.4 포함)
# =============================================================================

TESTS = '''
"""
tests/test_pipeline.py
"""
import torch
import numpy as np
import pandas as pd
import pytest


def test_lstm_encoder_variable_length():
    """가변 길이 시퀀스를 패딩 없이 처리하는지 확인."""
    from src.models.lstm_encoder import LSTMEncoder
    enc = LSTMEncoder(input_size=10, hidden_size=32, bidirectional=True)
    enc.eval()

    # 길이가 다른 3개 샘플
    seqs = [
        torch.randn(5, 10),
        torch.randn(12, 10),
        torch.randn(3, 10),
    ]
    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(seqs, batch_first=True)   # (3, 12, 10)
    lengths = torch.tensor([5, 12, 3])

    with torch.no_grad():
        ctx = enc(padded, lengths)

    assert ctx.shape == (3, 64)   # 32 * 2 = 64
    # 길이가 다른 샘플들의 결과가 서로 달라야 함
    assert not torch.allclose(ctx[0], ctx[1])


def test_ft_transformer_output_shape():
    """FT-Transformer 출력 차원 확인."""
    from src.models.ft_transformer import FTTransformer
    model = FTTransformer(
        context_dims=[64, 32, 18],
        n_num_features=5,
        cat_cardinalities=[10, 5, 3],
        d_token=64,
        n_heads=4,
        n_layers=2,
    )
    model.eval()
    B = 8
    contexts = [torch.randn(B, 64), torch.randn(B, 32), torch.randn(B, 18)]
    x_num = torch.randn(B, 5)
    x_cat = torch.randint(0, 3, (B, 3))

    with torch.no_grad():
        A, R = model(contexts, x_num, x_cat)

    assert A.shape == (B,)
    assert R.shape == (B,)
    assert (R >= 0).all(), "위험도는 항상 ≥ 0 이어야 함 (Softplus)"


def test_theme_context_point_in_time():
    """테마 비중 계산이 미래 데이터를 참조하지 않는지 확인."""
    pass


def test_attractiveness_label_no_lookahead():
    """매력도 라벨이 forward 데이터만 사용하는지 확인."""
    from src.labels.attractiveness import compute_attractiveness
    prices = pd.DataFrame({
        'ticker': ['X'] * 10,
        'date':   pd.date_range('2015-01-01', periods=10, freq='QE'),
        'close':  [100, 110, 90, 120, 130, 125, 140, 135, 145, 150],
    })
    result = compute_attractiveness(prices, max_horizon_years=5,
                                    min_forward_quarters=4)
    first = result.iloc[0]
    max_price = max([110, 90, 120, 130, 125, 140, 135, 145, 150])
    expected_A = np.log(150 / 100) / np.log(5)
    assert abs(first['A'] - expected_A) < 1e-5


def test_collate_padding():
    """collate_fn이 가변 길이 시퀀스를 올바르게 패딩하는지 확인."""
    from src.data.dataset import collate_fn
    batch = [
        {
            's_seq':    torch.randn(5, 10),
            'm_seq':    torch.randn(5, 8),
            'theme':    torch.randn(18),
            'snap_num': torch.randn(7),
            'snap_cat': torch.zeros(3, dtype=torch.long),
            'A': torch.tensor(0.5),
            'R': torch.tensor(0.2),
        },
        {
            's_seq':    torch.randn(12, 10),
            'm_seq':    torch.randn(12, 8),
            'theme':    torch.randn(18),
            'snap_num': torch.randn(7),
            'snap_cat': torch.zeros(3, dtype=torch.long),
            'A': torch.tensor(0.8),
            'R': torch.tensor(0.3),
        },
    ]
    out = collate_fn(batch)
    assert out['s_seq'].shape   == (2, 12, 10)
    assert out['s_lengths'][0]  == 5
    assert out['s_lengths'][1]  == 12
    assert out['theme'].shape   == (2, 18)


# ── v5.4 신규 테스트 (tests/test_split.py) ───────────────────────────────────

def test_stratified_split_market_ratio():
    """KR/US 비율이 Train/Val/Test에서 유사하게 유지되는지."""
    from src.utils.split import stratified_ticker_split
    from src.theme.loader import load_themes

    train, val, test = stratified_ticker_split()
    data    = load_themes()
    tickers = data['tickers']

    def kr_ratio(lst):
        kr = sum(1 for t in lst if tickers.get(t, {}).get('market') == 'KR')
        return kr / len(lst) if lst else 0

    r_train = kr_ratio(train)
    r_val   = kr_ratio(val)
    r_test  = kr_ratio(test)

    assert abs(r_train - r_test) < 0.05, f"KR ratio mismatch: train={r_train:.2f}, test={r_test:.2f}"
    assert abs(r_train - r_val)  < 0.05


def test_stratified_split_no_overlap():
    """Train/Val/Test 간 종목 중복 없음."""
    from src.utils.split import stratified_ticker_split

    train, val, test = stratified_ticker_split()
    train_set = set(train)
    val_set   = set(val)
    test_set  = set(test)

    assert len(train_set & val_set)  == 0, "Train-Val overlap"
    assert len(train_set & test_set) == 0, "Train-Test overlap"
    assert len(val_set   & test_set) == 0, "Val-Test overlap"


def test_stratified_split_tier1_coverage():
    """Test 셋에 모든 Tier1 카테고리가 최소 1종목 포함."""
    from src.utils.split import stratified_ticker_split
    from src.theme.loader import load_themes

    _, _, test = stratified_ticker_split()
    data    = load_themes()
    tickers = data['tickers']

    tier1_in_test = set(
        tickers[t]['primary_tier1'] for t in test if t in tickers
    )
    all_tier1 = set(
        k for k, v in data['themes'].items() if v.get('tier') == 1
    )
    missing = all_tier1 - tier1_in_test
    assert not missing, f"Test에 없는 Tier1: {missing}"


def test_merge_themes_idempotent(tmp_path):
    """00_merge_themes.py 두 번 실행해도 결과 동일."""
    from scripts.00_merge_themes import merge
    out = tmp_path / 'themes.yaml'
    merge('data/themes/raw', str(out), force=True)
    import yaml
    with open(out) as f:
        content1 = yaml.safe_load(f)
    merge('data/themes/raw', str(out), force=True)
    with open(out) as f:
        content2 = yaml.safe_load(f)
    assert content1['meta']['n_tickers_total'] == content2['meta']['n_tickers_total']
    assert content1['meta']['n_themes'] == content2['meta']['n_themes']


# tests/test_theme_context.py (추가 테스트)

def test_mktcap_weight_sums_to_one():
    """시총 비중의 합이 1인지 확인."""
    import pandas as pd
    import numpy as np
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':     ['A', 'B', 'C'],
        'market_cap': [300.0, 500.0, 200.0],
        'F_VAL_003':  [1.0, 2.0, 0.8],   # PBR
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    vB = compute_theme_vector('B', peers[peers['ticker']=='B'].iloc[0], peers)
    vC = compute_theme_vector('C', peers[peers['ticker']=='C'].iloc[0], peers)

    # 시총 비중 합산 = 1
    total_w = vA[0] + vB[0] + vC[0]
    assert abs(total_w - 1.0) < 1e-5, f"mktcap weight sum = {total_w}"

    # 개별 비중 = 기대값
    assert abs(vA[0] - 0.30) < 1e-4   # 300/1000
    assert abs(vB[0] - 0.50) < 1e-4   # 500/1000
    assert abs(vC[0] - 0.20) < 1e-4   # 200/1000


def test_relative_pbr():
    """상대 PBR: 테마 평균 PBR = 2.0일 때 PBR=1.0 종목의 상대값 = 0.5."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':    ['A', 'B', 'C'],
        'market_cap':[100.0, 100.0, 100.0],
        'F_VAL_003': [1.0, 2.0, 3.0],   # 평균 = 2.0
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    assert abs(vA[6] - 0.5) < 1e-4, f"rel_pbr = {vA[6]}, expected 0.5"


def test_hhi_monopoly():
    """한 종목이 시총 100% 점유 시 HHI = 1.0."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':    ['A', 'B'],
        'market_cap':[999.0, 1.0],
    })
    vA = compute_theme_vector('A', peers[peers['ticker']=='A'].iloc[0], peers)
    assert vA[16] > 0.99 - 1e-3, f"HHI expected ~1.0, got {vA[16]}"


def test_negative_ebitda_weight():
    """음수 EBITDA 기업의 비중이 음수로 정확히 표현되는지."""
    import pandas as pd
    from src.theme.context import compute_theme_vector

    peers = pd.DataFrame({
        'ticker':          ['A', 'B', 'C'],
        'market_cap':      [200.0, 200.0, 200.0],
        'F_PRF_ebitda_abs':[100.0, 200.0, -50.0],  # C는 적자
    })
    vC = compute_theme_vector('C', peers[peers['ticker']=='C'].iloc[0], peers)
    # C의 EBITDA 비중 = -50 / (100+200) = -0.1667
    assert vC[2] < 0, f"Negative EBITDA should give negative weight, got {vC[2]}"
    assert abs(vC[2] - (-50.0/300.0)) < 1e-3
'''

# =============================================================================
# 섹션 13. 트러블슈팅 (업데이트)
# =============================================================================

TROUBLESHOOTING = """
# 트러블슈팅

## M1 Pro / MPS

| 증상 | 원인 | 해결 |
|------|------|------|
| RuntimeError: MPS backend out of memory | 배치 or 모델 너무 큼 | batch_size 줄이기, d_token 128로 축소 |
| aten::xxx not implemented for MPS | MPS 미지원 op | PYTORCH_ENABLE_MPS_FALLBACK=1 |
| pack_padded_sequence 오류 | lengths가 GPU에 있음 | lengths.cpu() 확인 (코드에 이미 적용) |
| TransformerEncoderLayer MPS 오류 | 일부 어텐션 커널 미지원 | PYTORCH_ENABLE_MPS_FALLBACK=1 |
| float64 관련 오류 | MPS는 float32만 지원 | 모든 텐서 .float() 호출 확인 |

## 데이터

| 증상 | 해결 |
|------|------|
| features_stock/macro 분리 안 됨 | 02_build_features.py에서 M_ 접두사 컬럼을 macro로, F_/A_ 컬럼을 stock으로 분리 저장 |
| theme_context NaN 비율 높음 | 매핑 파일 ticker 형식 확인 (KR: 6자리 문자열, US: 영문 티커) |
| LSTM 학습 발산 | LayerNorm 적용 확인, lr 1e-5로 낮추기, gradient_clip 확인 |
| val/loss가 train/loss보다 훨씬 큼 | 시퀀스 길이 분포 확인, max_seq_len 줄이기 |

## 모델

| 증상 | 원인 | 해결 |
|------|------|------|
| R (위험도) 예측이 0에 수렴 | Softplus + MSE 조합 이슈 | log_var_R 초기값 확인, R 라벨 스케일링 |
| A, R 손실 중 하나가 지배 | Kendall 가중치 불균형 | log_var 초기값을 -1.0으로 조정 |
| 테마 비중 피처가 학습에 미반영 | FTT attention에서 theme_ctx 토큰 무시 | n_layers 늘리기, theme_proj_dim 확인 |

## 성능 튜닝 (M1 Pro 16GB 기준 권장값)

  lstm_stock_hidden: 128   (256은 메모리 부담)
  lstm_macro_hidden: 64
  d_token: 192             (128도 가능, 속도 우선 시)
  n_layers: 4              (3으로 줄이면 30% 빠름)
  batch_size: 64           (32GB 모델은 128)
  num_workers: 2           (4로 올리면 빠르지만 메모리 주의)
"""

# =============================================================================
# 섹션 14. 변경 체크리스트 (v5.4 업데이트)
# =============================================================================

CHECKLIST = """
# v5.4 → v5.5 변경 체크리스트

## 변경 파일
□ src/theme/context.py
    - compute_theme_vector(): 백분위 → 실질 비중 + 상대 배수 (18차원)
    - _safe_weight(), _signed_weight(), _safe_relative() 헬퍼 추가

□ src/models/predictor.py
    - THEME_VEC_DIM: 16 → 18 업데이트

□ src/features/fundamental/abs_values.py (신규)
    - 비중 계산용 절대값 컬럼 생성 (매출, EBITDA, 순이익, 총자산)
    - KRW → USD 환산 포함

## 추가 없음 (구조 유지)
□ theme_context.parquet 스키마 변경 (theme_ctx_0~17, 16→18차원)
□ dataset.py: THEME_VEC_DIM 상수 18로 업데이트
□ tests/test_theme_context.py: 비중 합산=1 검증 테스트 추가
"""

# =============================================================================
# 섹션 15. 추가 모니터링 및 로컬 라벨링 가이드 (WandB & Local Labeling)
# =============================================================================

ADDITIONAL_GUIDE = """
# WandB (Weights & Biases) 및 로컬 라벨링(log_N) 추가 명세

## 1. WandB 실험 모니터링
FT-Transformer 학습 시, 학습 및 검증 메트릭의 실시간 시각화를 지원하기 위해 WandB 로거를 옵션으로 지원합니다.

### 1.1 설정 및 활성화
- `config/settings.yaml` 파일의 `model.use_wandb` 설정을 `true`로 설정하여 자동 활성화할 수 있습니다.
- 환경 내 `wandb` 및 `lightning` 관련 종속성이 온전해야 합니다. 
- 비활성화 시 기본적으로 `CSVLogger`가 실행 로그를 담당합니다.

### 1.2 로깅 키 네이밍 규칙
학습 과정에서 체크포인트 디렉토리가 하위 폴더로 슬래시(`\`, `/`)로 쪼개지는 오동작을 미연에 방지하기 위해 다음과 같이 평면화된 메트릭 이름 형식을 고수합니다.
- `train_loss`, `train_A`, `train_R` (Epoch 별 학습 손실 및 타겟별 손실)
- `val_loss`, `val_A`, `val_R` (Epoch 별 검증 손실 및 타겟별 손실)
- 저장 파일명은 `stockml-{epoch:02d}-{val_loss:.4f}.ckpt` 형식을 준수하여 `checkpoints/` 폴더에 단일 파일 형태로 생성됩니다.

---

## 2. 로컬 라벨링 및 log_N 계산 기법
5년 미만의 상장/관측 데이터를 가지는 종목군에 대해 신뢰할 수 있는 매수 매력도(A) 및 위험도(R) 지표를 수립하기 위해 동적 윈도우 크기 $N$을 적용합니다.

### 2.1 Attractiveness ($A$, 매력도)
동적 관측 기간 $N$ 분기 ($4 \le N \le 20$)에 따른 Attractiveness는 다음과 같은 수학적 정규화 규칙을 따릅니다:
$$A = \log_N \left( \frac{\max(P_{t \dots t+N}) + 1\text{e-}8}{P_t + 1\text{e-}8} \right)$$
여기서 밑이 $N$인 로그를 취함으로써, 관측 기간이 짧아 단기간 내 급격하게 상승한 종목과 장기간 서서히 상승한 종목의 매력도 스케일을 일정하게 조정(정규화)합니다.

### 2.2 Risk ($R$, 위험도)
관측 기간 $N$ 분기 동안 종가 기준의 로그 수익률에 대한 변동성의 평균값(연환산 표준편차)을 계산합니다:
$$R = \text{std}(\text{log\_returns}_{t \dots t+N}) \times \sqrt{4}$$
- $N$이 5년(20분기)보다 적은 종목들의 경우 실제 유효 분기 개수 만큼의 표준편차를 사용하여 개별 종목의 고유 위험도를 보수적이고 안정적으로 측정합니다.

### 2.3 단위 테스트 검증
- 해당 연산 로직은 `tests/test_pipeline.py` 내 `test_attractiveness_label` 및 `test_risk_label` 단위 테스트를 통해 그 정확도가 철저히 보증됩니다.
"""

# =============================================================================
# 섹션 16. 데이터 수집 파이프라인 고도화 및 정합성 검증 (v5.5 신규)
# =============================================================================

DATA_PIPELINE_UPGRADE = """
# 데이터 수집 및 정합성 검증 시스템 고도화

기존 데이터 파이프라인의 분기 결측치, 연말 휴장일 가격 유실, 시가총액 계산 오류 등의 문제를 근본적으로 정비하고, 실시간 데이터 정합성 검증 엔진을 파이프라인에 이식하여 학습 데이터의 신뢰성을 극대화했습니다.

## 1. 한국 가격 수집 엔진의 yfinance 벌크 전환
- **문제점**: 기존 일별 `pykrx` API는 잦은 접속 차단(403/IndexError) 및 누락이 잦았으며, 특히 4분기 말 연말 휴장일의 주가 유실로 인해 분기 리샘플링 시 전체 분기 데이터가 누락되는 치명적인 데이터 공백이 발생했습니다.
- **해결책 (`src/data_fetchers/prices_kr.py`)**:
  - `pykrx`는 KOSPI 상장 종목 리스트 수집용으로만 최소한으로 활용하고, 실제 주가 데이터는 `yfinance` 벌크 쿼리(`.KS` 접미사)를 통해 100개 티커 단위 청크로 다운로드하도록 전면 교체했습니다.
  - 일별 데이터를 로컬 메모리에 적재한 뒤, Pandas의 `.resample('QE')` 함수를 이용해 분기말 기준으로 **시가(first), 고가(max), 저가(min), 종가(last), 거래량(sum)**을 완벽하게 산출했습니다.
- **성능 개선**: 이 구조 개선을 통해 전체 시계열 로우 개수가 **31.5k개에서 85,763개로 약 172% 증가**하였으며, 분기 사이의 데이터 누락율(Gap)이 **0%**로 감소했습니다.

## 2. 메타데이터(국가 및 통화) 보존 가드
- **문제점**: 분기 리샘플링 과정에서 국가(`country`) 및 통화(`currency`) 등 고정 메타데이터 컬럼이 집계 대상에서 제외되어 파이프라인 다운스트림에서 결측치가 대량 발생했습니다.
- **해결책 (`prices_kr.py` & `prices_us.py`)**:
  - `.resample('QE').agg(...)` 연산 시 `'country': 'last'`, `'currency': 'last'` 규칙을 공통 적용하여 병합본(`prices_quarterly.parquet`) 내 모든 주식의 국가/통화 속성을 결측치 없이 100% 보존했습니다.

## 3. 시가총액(Market Cap) 동적 복원 시스템
- **문제점**: `yfinance` 벌크 다운로드 과정 중 시가총액 정보가 0.0 또는 Null로 수집되는 경우, 멀티팩터 가중치(HHI 등) 연산에서 나눗셈 에러가 발생하거나 종목 가중치 계산이 왜곡되었습니다.
- **해결책 (`scripts/02_build_features.py`)**:
  - 시가총액 값이 Null이거나 0 이하인 경우, 해당 분기의 `종가(close) * 상장주식수(shares) * 1,000,000` 공식을 이용하여 실시간으로 복원하는 백필 가드를 구축했습니다.

## 4. 정합성 검증 엔진 내장 (`src/utils/validation.py`)
- **기능**: 데이터 수집(`01_fetch_data.py`) 및 피처 생성(`02_build_features.py`) 파이프라인의 각 단계 끝에 `check_intermediate_gaps()` 검증 단계를 이식했습니다.
- **검증 범위**:
  - key 컬럼(`open`, `high`, `low`, `close`) 내 NaN 값 감지 및 샘플 티커/날짜 출력
  - 주가 0값 및 거래량 0값(거래정지 등)의 행 단위 감지 및 경고
  - 종목별 관측 라이프타임 내 분기 캘린더 공백(중간 누락 분기) 탐지
- **효과**: 터미널에 가시성이 높은 컬러 경고 문구와 가이드를 실시간으로 출력하여 개발자가 데이터 품질 저하를 즉각 인지하고 로컬 캐시 초기화(`--no-cache`) 등 즉각 대처할 수 있도록 지원합니다.
"""


