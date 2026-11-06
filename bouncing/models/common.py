"""3モデルが共有する部品。違いが「過去をどう保持するか」だけになるように
する。

  InputEmbedding      1フレームの全玉の座標 -> 大きさ d のベクトル
  QuestionEmbedding   (qtype, ball, cond) -> 大きさ d のベクトル。3つの
                      埋め込み表（整数ごとに1つ）の和
  Readout             [フレーム t の状態 ; 質問ベクトル] -> MLP -> MAX_OPTIONS 個
                      の答えに対するロジット。この質問に存在しない答え
                      （n_options < MAX_OPTIONS）には -1e9 を入れる

StreamModel が基底クラス。サブクラスは `encode(pos)` を実装し、ストリーム
全体 (B, T, N*2) をフレームごとの状態 (B, T, h_dim) に写す。基底クラスは
各質問のフレームの状態を gather して読み出す。フレームごとの状態を出せない
モデル（窓つき）は代わりに `forward` を上書きする。

読み出しは *現在フレーム* も直接見る（`current_frame=True`）:
[t の状態 ; フレーム t の埋め込み ; 質問]。これがないと、玉が半分の境界から
0.05 以内にある NOW_HALF の質問が 20-40% 誤っていた: 座標は4層と LayerNorm
を通ると粗くしか残らない。記憶はあくまで状態の中だけにある。現在の観測は
固定状態モデルが忘れるべきものではない。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data import T_COL, QTYPE, BALL, COND, NOPT

N_QTYPES = 4
N_CONDS = 5            # 0..3 は壁/半分、4 = どの壁でも（LAST_HIT のみ）
MAX_OPTIONS = 3        # = 玉の数。はい/いいえの質問は先頭の2つを使う


class InputEmbedding(nn.Module):
    """1フレーム -> ベクトル。velocity=True ならフレーム間の差分を付け足す
    （t=0 では零、典型的な速度が O(1) になるように拡大）。壁ヒットは
    3フレームにわたるパターンではなく、連続する2フレーム間の符号反転
    として現れるようになる。"""
    VEL_SCALE = 30.0

    def __init__(self, n_balls: int, d: int, velocity: bool = False):
        super().__init__()
        self.velocity = velocity
        self.proj = nn.Linear(n_balls * 2 * (2 if velocity else 1), d)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:   # (B, T, N*2) -> (B, T, d)
        x = pos * 2 - 1                                        # [0,1] を [-1,1] に写す
        if self.velocity:
            v = torch.zeros_like(pos)
            v[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * self.VEL_SCALE
            x = torch.cat([x, v], dim=-1)
        return self.proj(x)

"""
      X          ·      Wᵀ        +      b        =       H
  (B·T, 12)          (12, 128)          (128)          (B·T, 128)

  ┌──────────┐     ┌───────────┐                     ┌───────────┐
  │ x0 … vy2 │     │           │                     │           │
  │ x0 … vy2 │  ·  │  12 × 128 │  +  [b0 … b127]  =  │ B·T × 128 │
  │    ⋮     │     │           │                     │           │
  │ x0 … vy2 │     └───────────┘                     │     ⋮     │
  └──────────┘                                       └───────────┘
   フレームが                 行列積の内側の 12 が
   1行ずつ                    消えて 128 になる
"""


class QuestionEmbedding(nn.Module):
    def __init__(self, n_balls: int, d: int):
        super().__init__()
        self.qtype = nn.Embedding(N_QTYPES, d)
        self.ball = nn.Embedding(n_balls, d)
        self.cond = nn.Embedding(N_CONDS, d)

    def forward(self, q: torch.Tensor) -> torch.Tensor:       # (B, Q, 8) -> (B, Q, d)
        return self.qtype(q[..., QTYPE]) + self.ball(q[..., BALL]) + self.cond(q[..., COND])


class Readout(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_out: int = MAX_OPTIONS):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        self.n_out = n_out

    def forward(self, parts: list[torch.Tensor], n_options: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(torch.cat(parts, dim=-1))                  # (B, Q, n_out)
        option = torch.arange(self.n_out, device=logits.device)
        invalid = option[None, None, :] >= n_options[..., None]     # (B, Q, n_out)
        return logits.masked_fill(invalid, -1e9)


class StreamModel(nn.Module):
    def __init__(self, n_balls: int, d: int, h_dim: int, velocity: bool = False,
                 current_frame: bool = True):
        super().__init__()
        self.n_balls, self.d, self.h_dim, self.current_frame = n_balls, d, h_dim, current_frame
        self.embed = InputEmbedding(n_balls, d, velocity=velocity)
        self.q_embed = QuestionEmbedding(n_balls, d)
        self.readout = Readout(h_dim + d + (d if current_frame else 0), hidden=2 * d)

    def gather_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x (B, T, D), t (B, Q) -> x[b, t[b, j]] を (B, Q, D) として返す。"""
        return torch.gather(x, 1, t[..., None].expand(-1, -1, x.shape[-1]))

    def read(self, h_t: torch.Tensor, emb: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """gather した状態 h_t (B, Q, h_dim)、埋め込んだストリーム emb (B, T, d)、
        質問行 q (B, Q, 8) からロジットを出す。"""
        parts = [h_t, self.q_embed(q)]
        if self.current_frame:
            parts.append(self.gather_t(emb, q[..., T_COL]))
        return self.readout(parts, q[..., NOPT])

    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        """埋め込んだフレーム (B, T, d) -> (B, T, h_dim): 各フレーム後の状態。"""
        raise NotImplementedError

    def states(self, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """(B, Q, h_dim): 各質問が読み出される状態。probe.py の線形プローブが
        見るのはこれだけ (質問と現在フレームを除けば、モデルが覚えている全て)。"""
        emb = self.embed(pos)
        return self.gather_t(self.encode(emb), q[..., T_COL])

    def forward(self, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """全ストリームの全質問に対するロジット (B, Q, MAX_OPTIONS)。"""
        emb = self.embed(pos)                                          # (B, T, d)
        h = self.encode(emb)                                           # (B, T, h_dim)
        h_t = self.gather_t(h, q[..., T_COL])                          # 各質問のフレームの状態
        return self.read(h_t, emb, q)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
