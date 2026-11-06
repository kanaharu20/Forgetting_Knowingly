"""窓つき Transformer: フレーム t の状態は文字どおり直近 W フレームである。

スライディング窓の注意マスクではそうならない: L 層あると受容野は L*W まで
広がる。そこで代わりに、質問ごとに窓 pos[t-W+1 .. t] を切り出し、小さな
因果 Transformer をそれに走らせ、最後の位置を読み出す。ストリーム開始前の
フレームは零でパディングし、学習する「パディング」埋め込みを足して印をつける。
したがって W-1 を超える参照距離は状態からは答えられない: これが
RESEARCH.md が校正の基準点として使う崖である。

計算するのは質問フレームの窓だけ（W フレームの窓が B*Q 個）で、
500 フレームにわたる全注意より安い。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data import T_COL
from .common import StreamModel
from .transformer import CausalEncoder


class WindowTransformer(StreamModel):
    def __init__(self, n_balls: int, window: int = 32, d: int = 128, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.0, velocity: bool = False,
                 current_frame: bool = True):
        super().__init__(n_balls, d, h_dim=d, velocity=velocity, current_frame=current_frame)
        self.window = window
        self.pad_emb = nn.Parameter(torch.zeros(d))
        self.encoder = CausalEncoder(d, n_layers, n_heads, max_len=window, dropout=dropout)

    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("the windowed model has no per-frame state; use forward")

    def _window_states(self, emb: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """埋め込んだフレーム (B, T, d) -> (B, Q, d): 各質問の t で終わる W フレームを
        符号化した最後の位置。"""
        B, T, d = emb.shape
        W = self.window
        t = q[..., T_COL]                                               # (B, Q)
        offsets = torch.arange(-W + 1, 1, device=emb.device)            # (W,)
        frames = t[..., None] + offsets                                 # (B, Q, W)、負になりうる
        valid = frames >= 0
        frames = frames.clamp(min=0)
        idx = frames.reshape(B, -1, 1).expand(-1, -1, d)                # (B, Q*W, d)
        x = torch.gather(emb, 1, idx).reshape(B, -1, W, d)              # (B, Q, W, d)
        x = torch.where(valid[..., None], x, self.pad_emb.expand_as(x))
        h = self.encoder(x.reshape(-1, W, d))[:, -1]                    # (B*Q, d): t の状態
        return h.reshape(B, -1, d)

    def states(self, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._window_states(self.embed(pos), q)

    def forward(self, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        emb = self.embed(pos)                                           # (B, T, d)、フレームごと
        return self.read(self._window_states(emb, q), emb, q)
