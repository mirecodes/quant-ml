# src/models/tft_model.py
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE, MultiLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.device import get_device

class TFTStockModel:
    """TFT 래퍼 — M1 Mac 호환."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()
        self.model = None
        self.dataset = None
    
    def prepare_dataset(self, df: pd.DataFrame, 
                        targets: list = ['A', 'R']):
        """
        df: 종목·분기별 데이터 (피처 + 라벨)
        """
        df = df.copy()
        # quarter index를 정수로 변환
        df['quarter_idx'] = (df['date'].dt.year * 4 + df['date'].dt.quarter).astype(int)
        df['quarter_idx'] -= df['quarter_idx'].min()
        
        # NaN 라벨이 있는 행 제거 (A, R 둘 다 NaN인 행 제거)
        df = df.dropna(subset=targets, how='all')
        
        # static categoricals
        static_cats = ['country', 'sector', 'size_tier']
        for col in static_cats:
            df[col] = df[col].astype(str).fillna('unknown')
            
        # 모든 거시·재무·계산 피처 자동 인식
        feature_cols = [c for c in df.columns 
                        if c.startswith(('M_', 'F_', 'C_'))]
        
        # 데모/소규모 데이터셋에서의 정상 동작을 위해 인코더 길이 축소 최적화
        max_encoder = 2   # 과거 2분기
        max_pred = 1
        
        training_cutoff = df['quarter_idx'].max() - max_pred
        
        # v5.1 패치: 멀티 타겟인 경우 MultiNormalizer 필수 적용
        target_normalizer = MultiNormalizer([
            GroupNormalizer(groups=["ticker"], transformation="softplus"), # Target A용
            GroupNormalizer(groups=["ticker"], transformation="softplus")  # Target R용
        ])
        
        self.dataset = TimeSeriesDataSet(
            df[df['quarter_idx'] <= training_cutoff],
            time_idx="quarter_idx",
            target=targets,
            group_ids=["ticker"],
            min_encoder_length=1,    # 최소 1분기 과거만 있어도 가능하게 완화
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_pred,
            
            static_categoricals=static_cats,
            static_reals=[],
            
            time_varying_known_reals=["quarter_idx"],
            time_varying_unknown_reals=feature_cols,
            
            target_normalizer=target_normalizer,
            
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        return self.dataset
    
    def build_model(self):
        """TFT 모델 인스턴스화."""
        assert self.dataset is not None, "Call prepare_dataset first"
        
        loss = MultiLoss(
            [QuantileLoss(quantiles=[0.1, 0.5, 0.9])] * 2
        )
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            hidden_size=self.config.get('hidden_size', 64),
            attention_head_size=self.config.get('attention_head_size', 4),
            dropout=self.config.get('dropout', 0.2),
            hidden_continuous_size=self.config.get('hidden_continuous_size', 32),
            loss=loss,
            learning_rate=self.config.get('learning_rate', 0.001),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        return self.model
    
    def train(self, train_loader, val_loader, max_epochs: int = 50):
        """학습 실행."""
        # MPS 가속 우선 사용
        accelerator = 'mps' if self.device.type == 'mps' else 'cpu'
        
        early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min')
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath='./checkpoints',
            filename='tft-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
        )
        
        try:
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[early_stop, checkpoint],
                enable_progress_bar=True,
            )
            trainer.fit(self.model, train_loader, val_loader)
        except Exception as e:
            # MPS 에러 발생 시 CPU 자동 폴백
            print(f"[Device Failover] MPS encountered error: {e}. Falling back to CPU...")
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator='cpu',
                devices=1,
                gradient_clip_val=0.1,
                callbacks=[early_stop, checkpoint],
                enable_progress_bar=True,
            )
            trainer.fit(self.model, train_loader, val_loader)
            
        return trainer
    
    def predict(self, dataloader):
        predictions = self.model.predict(
            dataloader, 
            mode='quantiles',
            return_x=True,
        )
        return predictions
