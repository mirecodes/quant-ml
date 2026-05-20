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
