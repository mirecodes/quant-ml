# Quant-ML: M1 Pro 가속 시계열 예측 및 글로벌 포트폴리오 리스크 최적화 엔진

Quant-ML은 **Apple Silicon M1 Pro(MPS)** 하드웨어 성능을 최대로 활용하여, 국내외 대형주(KOSPI 200 & S&P 500)의 중장기 수익률(Attractiveness)과 변동성 리스크(Risk)를 다차원 예측하고 최적의 포트폴리오를 발굴·시각화하는 엔드투엔드(End-to-End) 머신러닝 시스템입니다.

이 프로젝트는 최첨단 딥러닝 아키텍처인 **Temporal Fusion Transformer (TFT)**와 정교한 트리 모델인 **LightGBM**, 그리고 **재무적 베이스라인(F-Score, Quality Score)**을 고도로 결합하여 신뢰도 높은 예측 성과를 도출합니다.

---

## 🚀 Key Features

*   **Apple Silicon (MPS) 하드웨어 가속**: PyTorch Lightning 기반의 TFT 모델 학습 시 Apple M1 Pro(14-16 Core GPU)의 `mps` 가속을 완벽하게 활용하여 고속 분산 학습을 수행합니다.
*   **다차원 라벨 Consolidated 단일화 (v5.1 패치)**: 복잡한 3년/5년 만기별 라벨 분리를 직관적인 **단일 연환산 매력도(A)**와 **위험도(R)** 값으로 병합 및 일원화하여 모델의 학습 효율과 성능을 극대화했습니다.
*   **하이브리드 테마 엔진**: 실시간 크롤링 정보가 부재할 시에도 한국 KOSPI 200 반도체/2차전지/저PBR 수혜주 및 미국 S&P 500 핵심 빅테크 테마를 매핑해주는 고도화된 Fallback 테마 파이프라인을 지원합니다.
*   **2D 포트폴리오 다차원 맵 (Streamlit)**: 매력도(A)와 위험도(R), 시가총액, 업종, 테마 정보를 한눈에 동적으로 필터링하고 분석하는 프리미엄 다크 테마 대시보드를 제공합니다.

---

## 📂 Project Structure

```
quant-ml/
├── config/
│   └── settings.yaml          # 모델 하이퍼파라미터 및 하드웨어 가속 설정
├── src/
│   ├── data_fetchers/         # KRX, yfinance, FRED, 네이버 테마 수집 엔진
│   ├── models/                # TFT 딥러닝 모델 및 LightGBM/재무 베이스라인 모델
│   └── utils/                 # 데이터 I/O, 결측치 자동 보간 및 메모리 최적화 유틸
├── scripts/
│   ├── 01_fetch_data.py       # 데이터 수집 (KRX / yfinance 폴백)
│   ├── 02_build_features.py   # 시계열 피처 생성 및 누락 데이터 정밀 보간
│   ├── 03_build_labels.py     # 매력도(A) / 위험도(R) 타겟 라벨링 생성
│   ├── 04_train_tft.py        # TFT 딥러닝 학습 및 체크포인트 세이빙
│   ├── 05_train_baselines.py  # LightGBM 및 재무 베이스라인 학습 & 예측 병합
│   ├── 06_evaluate.py         # 백테스트 성과 지표(Rank IC, L-S Spread) 평가
│   └── 07_run_ui.py           # 프리미엄 Streamlit 대시보드 구동
├── pipeline.sh                # ★ 전체 파이프라인 원클릭 제어 스크립트
└── README.md                  # 본 문서
```

---

## 🔑 KRX(한국거래소) 로그인 계정 설정 가이드 (필수)

**2026년 2월 한국거래소(KRX)의 정보데이터시스템 보안 및 접근 정책 변경**에 따라, 회원제 기반의 데이터 조회 권한이 엄격히 적용되기 시작했습니다. 이에 따라 비로그인 상태로 직접적인 API 크롤링을 시도할 경우 `LOGOUT` 또는 JSON 파싱 에러가 발생하게 됩니다.

`pykrx` 라이브러리의 최신 패치(v1.2.8)는 이 로그인 변경 사항을 완벽히 수용하여 환경 변수를 통한 자동 로그인을 지원합니다. 1,300여 개 한국 주식 데이터를 100% 누락 없이 안전하게 수집하려면 아래 설정을 필수로 완료해 주세요.

### **설정 순서:**
1. **한국거래소 정보데이터시스템 회원가입**:
   * [https://data.krx.co.kr](https://data.krx.co.kr) 에 접속하여 무료 회원가입을 완료해 주세요 (1분 소요).
2. **`.env` 파일에 계정 정보 등록**:
   * 프로젝트 루트 디렉토리의 `.env` 파일을 열고 다음과 같이 계정 정보를 추가해 주세요:
     ```env
     # KRX 로그인 계정 정보 (한국 데이터 100% 수집용)
     KRX_ID=본인의_KRX_아이디
     KRX_PW=본인의_KRX_비밀번호
     ```
3. **장애복구 모드 (Fallback)**:
   * `.env` 파일에 계정 정보가 기록되지 않았거나 로그인에 실패하더라도, 시스템은 자동으로 **`yfinance` 및 KOSPI 대형주 대표 110여 개 종목 대상의 제한적 다운그레이드 수집 모드**로 전환되어 전체 파이프라인의 안전한 지속 구동을 보장합니다.

---

## ⚡ Quick Start: 쉘 스크립트로 1분 만에 파이프라인 정복하기

제공되는 `./pipeline.sh` 스크립트를 사용하여 전 과정 또는 특정 단계만 즉시 가동할 수 있습니다.

### 1. 도움말 및 사용법 확인
```bash
./pipeline.sh help
```

### 2. 역할별 부분 명령어 제어

#### **A. 데이터 수집 (Fetch)**
*   기본 캐시를 사용하여 수집 (최고 속도):
    ```bash
    ./pipeline.sh fetch
    ```
*   **[전수 수집]** 캐시를 우회하고 1,400여 개 전 종목에 대해 실시간 데이터를 새로 가져오기:
    ```bash
    ./pipeline.sh fetch --no-cache
    ```
*   가볍게 마켓별 50개 종목씩만 수집하여 테스트하기:
    ```bash
    ./pipeline.sh fetch --no-cache --limit 50
    ```

#### **B. 피처 빌드 및 모델 학습/평가 (Train)**
*   시계열 데이터 전처리, 라벨링, TFT 및 LightGBM 모델의 학습부터 포트폴리오 백테스트 평가까지 순차 실행:
    ```bash
    ./pipeline.sh train
    ```

#### **C. 대시보드 시각화 (Display)**
*   학습된 최종 예측 데이터를 바탕으로 Streamlit 다차원 포트폴리오 맵을 띄웁니다:
    ```bash
    ./pipeline.sh display
    ```

#### **D. 원클릭 전체 프로세스 실행 (All)**
*   수집 $\rightarrow$ 전처리 $\rightarrow$ 학습 $\rightarrow$ UI 기동까지 논스톱으로 실행합니다:
    ```bash
    ./pipeline.sh all --no-cache --limit 100
    ```

---

## 📊 Model Performance

종목 유니버스를 확장하고 데이터 결측치 보간 엔진을 탑재하여 횡단면 데이터의 질을 극대화한 결과, 백테스트 성능 지표가 혁신적으로 양전(Positive) 상승했습니다:

| 모델 아키텍처 | Target | Rank IC (예측 방향성 지표) | L-S Spread (롱-숏 스프레드 수익률) |
|---|---|---|---|
| **LightGBM** | 매력도 (A) | **+0.0671** | **+6.67%** |
| **TFT (AI 시계열)** | 매력도 (A) | **+0.0575** | **+7.76%** |
| **LightGBM** | 위험도 (R) | **+0.0855** | **+3.76%** |

---

## 🎨 Streamlit Premium UI Dashboard

대시보드 기동 시 다음과 같은 세련된 다크 테마 시각화 분석을 누리실 수 있습니다:

1.  **매력도 vs 위험도 2D 포트폴리오 맵**: 종목들을 위험도(R, 변동성) 대비 매력도(A, 수익률) 상에 분산하여 한눈에 Risk-Reward 효율이 뛰어난 종목을 스캔합니다.
2.  **🏆 매력도 Top 종목 랭킹**: 최신 분기 예측 기준 중복되지 않는 최우량 종목 10선을 고유 랭킹 테이블로 실시간 뷰어링합니다.
3.  **🔍 개별 종목 정밀 진단**: 선택한 종목의 TFT 예측치, LightGBM 예측치, 롱-숏 전적, F-Score, Quality Score, 그리고 테마 정보까지 마이크로 분석을 제공합니다.
