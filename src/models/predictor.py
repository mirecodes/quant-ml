"""
src/models/predictor.py

LSTM_stock + LSTM_macro + ThemeLinear + FT-Transformer를
하나의 PyTorch Lightning 모듈로 통합한다.
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

        # 테마 비중 Linear 투영 (16 → theme_proj_dim)
        self.theme_proj = nn.Sequential(
            nn.Linear(16, c['theme_proj_dim']),
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
        # R (위험도) 예측이 0에 수렴하는 문제를 방지하기 위해 초기값을 -1.0 등으로 조정 가능 (Research Plan v5.3 참고)
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
            return torch.tensor(0.0, requires_grad=True), 0.0, 0.0

        l_A = nn.functional.mse_loss(A_pred[mask], A_true[mask])
        l_R = nn.functional.mse_loss(R_pred[mask], R_true[mask])
        prec_A = torch.exp(-self.log_var_A)
        prec_R = torch.exp(-self.log_var_R)
        loss = prec_A * l_A + self.log_var_A + prec_R * l_R + self.log_var_R
        return loss, l_A.item(), l_R.item()

    def training_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'train_loss': loss, 'train_A': lA, 'train_R': lR},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        A, R   = self(batch)
        loss, lA, lR = self._loss(A, R, batch['A'], batch['R'])
        self.log_dict({'val_loss': loss, 'val_A': lA, 'val_R': lR},
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
