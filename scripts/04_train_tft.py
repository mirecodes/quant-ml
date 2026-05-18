# scripts/04_train_tft.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet

# PyTorch Lightning 2.0+ 버전 대응 경고 차단
import warnings
warnings.filterwarnings("ignore")

from src.models.tft_model import TFTStockModel
from src.utils.device import get_device

def main():
    pl.seed_everything(42)
    
    with open('config/settings.yaml') as f:
        config = yaml.safe_load(f)
    
    features = pd.read_parquet('data/processed/features.parquet')
    labels = pd.read_parquet('data/processed/labels.parquet')
    
    # 두 데이터프레임 병합
    data = features.merge(labels, on=['ticker', 'date'], how='inner')
    
    # TFT 모델은 PyTorch Forecasting Dataset 생성 시 타겟 라벨(A, R)의 결측치를 허용하지 않습니다.
    # 미래 수익률(최대 5년)을 알 수 없는 최근 데이터들의 결측치 로우를 학습 대상에서 안전하게 제거합니다.
    data = data.dropna(subset=['A', 'R']).copy()
    
    if data.empty:
        print("Error: Merged features and labels dataset is empty.")
        return
        
    # quarter_idx를 분할 및 데이터셋 준비 전에 원본 데이터에 추가 (KeyError 방지)
    data['quarter_idx'] = (data['date'].dt.year * 4 + data['date'].dt.quarter).astype(int)
    data['quarter_idx'] -= data['quarter_idx'].min()
    
    print(f"Total rows merged: {len(data)}")
    print(f"Unique Tickers: {data['ticker'].nunique()}")
    print(f"Date range: {data['date'].min()} ~ {data['date'].max()}")
    print(f"Computation Device: {get_device()}")
    
    # 학습 시작
    model = TFTStockModel(config['model'])
    
    # train/validation 분할 설정
    val_cutoff = pd.Timestamp(config['train_split']['train_end'])
    train_data = data[data['date'] <= val_cutoff]
    val_data = data[(data['date'] > val_cutoff) & 
                    (data['date'] <= pd.Timestamp(config['train_split']['val_end']))]
    
    if train_data.empty:
        print("Warning: Train data split is empty. Adjusting train_end date to have some training data.")
        median_date = data['date'].sort_values().iloc[int(len(data)*0.7)]
        train_data = data[data['date'] <= median_date]
        val_data = data[data['date'] > median_date]
        print(f"Adjusted Train Data size: {len(train_data)}, Validation Data size: {len(val_data)}")
        
    # 데이터셋 준비
    dataset = model.prepare_dataset(
        train_data,
        targets=['A', 'R']
    )
    
    # DataLoader 생성 (pin_memory=False 필수)
    train_loader = dataset.to_dataloader(
        train=True, 
        batch_size=config['model']['batch_size'], 
        num_workers=0, # M1 Mac 환경에서 다중 프로세스 교착 예방
        pin_memory=False
    )
    
    val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_data, predict=False, stop_randomization=True)
    val_loader = val_dataset.to_dataloader(
        train=False, 
        batch_size=config['model']['batch_size'], 
        num_workers=0, 
        pin_memory=False
    )
    
    print("Building Temporal Fusion Transformer model...")
    model.build_model()
    
    print("Starting training...")
    # 데모/안정성을 위해 최대 에폭을 5 또는 yaml에 정의된 에폭으로 실행
    max_epochs = min(config['model']['max_epochs'], 5) # 테스트 속도를 위해 5에폭으로 제한
    trainer = model.train(train_loader, val_loader, max_epochs=max_epochs)
    
    print("Training complete! Model saved under ./checkpoints/")

if __name__ == '__main__':
    main()
