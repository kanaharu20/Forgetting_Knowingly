"""questions.py が書き出したデータセットを読み込み、PyTorch に渡す。

バッチが *質問* ではなく *ストリーム* 単位である理由。因果モデルは1回の
パスで全フレームの状態を計算するので、1本のストリームの 256 問に答える
最も安い方法は、ストリームを一度符号化し、各質問のフレーム t の状態を
拾うことである。したがってバッチは B 本のストリームとその全質問であり、
拾う処理はモデル側が行う（models/common.py: StreamModel.forward を参照）。

バッチの配置（collate が返すもの）:
  pos   float32 (B, T, N*2)   全フレームにおける全玉の座標
  q     int64   (B, Q, 8)     各ストリームの質問表の行、パディングつき
  qmask bool    (B, Q)        q が実際の質問なら True、False = パディング

q の列番号は questions.COLUMNS に従う。下の定数がその名前である。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 質問表の列番号（questions.COLUMNS）
STREAM, T_COL, QTYPE, BALL, COND, ANSWER, EVIDENCE, NOPT = range(8)


class StreamDataset(Dataset):
    def __init__(self, path: str | Path):
        npz = np.load(path)
        pos = npz["pos"]                                   # (S, T, N, 2) float32
        q = npz["questions"]                               # (rows, 8) int64
        self.n_streams, self.T, self.n_balls, _ = pos.shape
        self.pos = torch.from_numpy(pos.reshape(self.n_streams, self.T, -1))
        self.speed_range = tuple(npz["speed_range"]) if "speed_range" in npz else None

        # 質問行をストリーム id でまとめる -> (S, Qmax, 8) とマスク
        order = np.argsort(q[:, STREAM], kind="stable")
        q = q[order]
        counts = np.bincount(q[:, STREAM], minlength=self.n_streams)
        self.q_max = int(counts.max())
        table = np.zeros((self.n_streams, self.q_max, q.shape[1]), dtype=np.int64)
        mask = np.zeros((self.n_streams, self.q_max), dtype=bool)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for s in range(self.n_streams):
            n = counts[s]
            table[s, :n] = q[starts[s]:starts[s] + n]
            mask[s, :n] = True
        self.q = torch.from_numpy(table)
        self.qmask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return self.n_streams

    def __getitem__(self, i: int):
        return self.pos[i], self.q[i], self.qmask[i]


def collate(batch):
    pos, q, m = zip(*batch)
    return torch.stack(pos), torch.stack(q), torch.stack(m)


def make_loader(path: str | Path, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = StreamDataset(path)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=g, drop_last=False)


if __name__ == "__main__":
    import sys
    ds = StreamDataset(sys.argv[1] if len(sys.argv) > 1 else "data/val.npz")
    print(f"streams={len(ds)} T={ds.T} balls={ds.n_balls} q_max={ds.q_max} speed_range={ds.speed_range}")
    pos, q, m = ds[0]
    print("pos", tuple(pos.shape), "q", tuple(q.shape), "real questions", int(m.sum()))
    print("first question row", q[0].tolist())
