"""全文脈の因果 Transformer: 対照モデル。フレーム t はフレーム 0..t に
注意を向けるので、何も忘れない。誤るとすれば 500 フレームにわたる注意の
限界（例: lost-in-the-middle）であり、圧縮の限界ではない。

Pre-LayerNorm ブロック、学習する絶対位置、注意は
scaled_dot_product_attention(is_causal=True)。既定は RESEARCH.md どおり
4層、d=128、4ヘッド。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import StreamModel


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, L, d)
        B, L, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, L, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.unbind(2)                                 # それぞれ (B, L, H, dh)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))        # (B, H, L, dh)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalEncoder(nn.Module):
    """埋め込んだフレーム (B, L, d) -> 状態 (B, L, d)。窓つきモデルと共有する。"""
    def __init__(self, d: int, n_layers: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        x = x + self.pos_emb(torch.arange(L, device=x.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class CausalTransformer(StreamModel):
    def __init__(self, n_balls: int, max_len: int, d: int = 128, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.0, velocity: bool = False,
                 current_frame: bool = True):
        super().__init__(n_balls, d, h_dim=d, velocity=velocity, current_frame=current_frame)
        self.encoder = CausalEncoder(d, n_layers, n_heads, max_len, dropout)

    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        return self.encoder(emb)
