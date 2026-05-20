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
