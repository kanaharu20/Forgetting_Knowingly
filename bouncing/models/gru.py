"""GRU: 固定サイズの状態（隠れ 256）をフレームごとに1回更新する。忘却は
なだらか: ゲートが保持しなかったものは時間とともに減衰する。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import StreamModel


class GRUModel(StreamModel):
    def __init__(self, n_balls: int, d: int = 128, hidden: int = 256, n_layers: int = 1,
                 velocity: bool = False, current_frame: bool = True):
        super().__init__(n_balls, d, h_dim=hidden, velocity=velocity, current_frame=current_frame)
        self.rnn = nn.GRU(d, hidden, num_layers=n_layers, batch_first=True)

    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(emb)                                    # (B, T, hidden)
        return h
