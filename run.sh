#!/bin/bash

# ==============================================================================
# QuantML Pipeline Control Script
# OS Support: macOS / Linux
# ==============================================================================

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 헤더 출력 함수
print_header() {
    echo -e "\n${BLUE}${BOLD}======================================================================${NC}"
    echo -e "${BLUE}${BOLD}  $1 ${NC}"
    echo -e "${BLUE}${BOLD}======================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}${BOLD}✔ $1${NC}"
}

print_error() {
    echo -e "${RED}${BOLD}✘ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}⚠ $1${NC}"
}

# 가상환경 확인 및 활성화
VENV_PATH=".venv"
if [ -d "$VENV_PATH" ]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
    STREAMLIT_BIN="$VENV_PATH/bin/streamlit"
else
    print_warning "로컬 가상환경(.venv)을 찾을 수 없습니다. 시스템 기본 python을 사용합니다."
    PYTHON_BIN="python"
    STREAMLIT_BIN="streamlit"
fi

# 도움말 표시 함수
show_help() {
    echo -e "${BOLD}사용법:${NC} ./run.sh [명령어] [옵션]"
    echo ""
    echo -e "${BOLD}명령어:${NC}"
    echo -e "  ${GREEN}fetch${NC}    : 데이터 및 네이버 증권 테마 수집"
    echo -e "  ${GREEN}train${NC}    : 피처 생성, 테마 비중 계산, 라벨 생성, FT-Transformer & 베이스라인 모델 학습 및 백테스트 평가"
    echo -e "  ${GREEN}display${NC}  : Streamlit 대시보드 UI 기동"
    echo -e "  ${GREEN}all${NC}      : [fetch] -> [train] -> [display] 전 과정 순차 자동 실행 (기본값)"
    echo -e "  ${GREEN}help${NC}     : 도움말 표시"
    echo ""
    echo -e "${BOLD}옵션 (fetch / all 명령어 전용):${NC}"
    echo -e "  ${YELLOW}--no-cache${NC}   : 기존 캐시 파일을 무시하고 실시간 API 가격 데이터 전수 수집 강제"
    echo -e "  ${YELLOW}--limit <N>${NC}  : 마켓별(KR/US) 수집할 최대 종목 수 제한 (예: --limit 50)"
    echo ""
    echo -e "${BOLD}예시:${NC}"
    echo "  ./run.sh fetch --no-cache --limit 100  # 100종목씩 실시간 수집"
    echo "  ./run.sh train                         # 피처 빌드, 테마 비중 계산, 라벨링 및 AI 모델 학습"
    echo "  ./run.sh display                       # UI 기동"
    echo "  ./run.sh all --limit 30                # 30종목 제한으로 전체 파이프라인 구동"
}

# 파라미터 파싱
CMD=$1
if [ -z "$CMD" ]; then
    CMD="all"
else
    shift # 첫 번째 인자(명령어) 제거하여 옵션들만 남김
fi

# 옵션들 수집
FETCH_OPTS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)
            FETCH_OPTS="$FETCH_OPTS --no-cache"
            shift
            ;;
        --limit)
            FETCH_OPTS="$FETCH_OPTS --limit $2"
            shift 2
            ;;
        *)
            print_error "알 수 없는 옵션입니다: $1"
            show_help
            exit 1
            ;;
    esac
done

# 역할별 실행 로직
run_fetch() {
    print_header "Step 1: Data & Theme Fetching"
    echo -e "실행 중: $PYTHON_BIN scripts/01_fetch_data.py $FETCH_OPTS\n"
    $PYTHON_BIN scripts/01_fetch_data.py $FETCH_OPTS
    if [ $? -ne 0 ]; then
        print_error "데이터 수집 중 오류가 발생했습니다. 파이프라인을 중단합니다."
        exit 1
    fi
    print_success "데이터 및 테마 수집 완료!"
}

run_train() {
    print_header "Step 2: Building Features"
    $PYTHON_BIN scripts/02_build_features.py
    if [ $? -ne 0 ]; then
        print_error "피처 생성 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "시계열 피처 빌드 완료!"

    print_header "Step 3: Building Theme Context"
    $PYTHON_BIN scripts/02b_build_theme_context.py
    if [ $? -ne 0 ]; then
        print_error "테마 비중 컨텍스트 계산 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "테마 비중 컨텍스트 계산 완료!"

    print_header "Step 4: Creating Target Labels"
    $PYTHON_BIN scripts/03_build_labels.py
    if [ $? -ne 0 ]; then
        print_error "라벨 생성 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "학습 타겟 라벨링 생성 완료!"

    print_header "Step 5: Training FT-Transformer Model"
    $PYTHON_BIN scripts/04_train.py
    if [ $? -ne 0 ]; then
        print_error "FT-Transformer 모델 학습 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "FT-Transformer 모델 학습 완료!"

    print_header "Step 6: Training Baseline Models"
    $PYTHON_BIN scripts/05_train_baselines.py
    if [ $? -ne 0 ]; then
        print_error "베이스라인 모델 학습 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "LightGBM 제거 및 회계 베이스라인 학습 완료!"

    print_header "Step 7: Evaluating Models"
    $PYTHON_BIN scripts/06_evaluate.py
    if [ $? -ne 0 ]; then
        print_error "모델 평가 중 오류가 발생했습니다."
        exit 1
    fi
    print_success "모델 평가 리포트 생성 완료!"
}

run_display() {
    print_header "Step 8: Launching QuantML Dashboard"
    echo -e "${YELLOW}대시보드를 로컬 호스트에 기동합니다...${NC}"
    $STREAMLIT_BIN run scripts/07_run_ui.py
}

# 메인 분기 처리
case "$CMD" in
    fetch)
        run_fetch
        ;;
    train)
        run_train
        ;;
    display)
        run_display
        ;;
    all)
        run_fetch
        run_train
        run_display
        ;;
    help|* )
        show_help
        ;;
esac
