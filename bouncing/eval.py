"""チェックポイントをある分割で採点し、全質問の確率を保存する。

    python eval.py runs/transformer/best.pt --split test

<run>/<split>_probs.npz に質問の表、質問ごとの確率ベクトル、そこから導いた
correct / conf / dist の配列を書き出し、層別した表を印字する。対照モデルの
保存された確率は、段階3 (蒸留、切り替え) の教師信号になる。

<run>/<split>_curves.json も書く: 質問の種類ごとに、25 フレーム幅の参照距離の
区間での正解率と平均確信 (EVER_HIT の「いいえ」群も加える)。数 KB なので、
run の主図のデータを npz なしで持ち運べる (例: Colab から Drive 経由)。
plot_results.py はどちらのファイルも受け付ける。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import make_loader
from models import build
from metrics import format_table
from train import evaluate, pick_device


def curves(arrays: dict, width: int = 25) -> dict:
    """qtype ごとの細かい区間での正解率 / 確信。plot_results.py の裏にあるデータ。"""
    from data import QTYPE
    from questions import QTYPE_NAMES
    q, dist, correct, conf = arrays["questions"], arrays["dist"], arrays["correct"], arrays["conf"]
    T = int(q[:, 1].max()) + 1
    out = {"width": width, "T": T, "qtypes": {}}
    for k, name in enumerate(QTYPE_NAMES):
        m_k = q[:, QTYPE] == k
        bins = []
        for lo in range(0, T, width):
            m = m_k & (dist >= lo) & (dist < lo + width)
            if m.sum():
                bins.append(dict(lo=lo, hi=lo + width, n=int(m.sum()),
                                 acc=float(correct[m].mean()), conf=float(conf[m].mean())))
        mn = m_k & (dist < 0)
        out["qtypes"][name] = dict(bins=bins, no_evidence=(
            dict(n=int(mn.sum()), acc=float(correct[mn].mean()), conf=float(conf[mn].mean())) if mn.any() else None))
    return out


def load_checkpoint(path: str | Path, device: torch.device):
    ck = torch.load(path, map_location=device)
    model = build(ck["model"], **ck["kw"]).to(device)
    model.load_state_dict(ck["state"])
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--streams", type=int, default=0, help="score only the first n streams (smoke tests)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = pick_device(a.device)
    model, ck = load_checkpoint(a.checkpoint, device)
    loader = make_loader(Path(a.data) / f"{a.split}.npz", a.batch, shuffle=False)
    if a.streams:
        ds = loader.dataset
        ds.pos, ds.q, ds.qmask = (x[:a.streams] for x in (ds.pos, ds.q, ds.qmask))
        ds.n_streams = a.streams
        loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=loader.collate_fn)
    rows, arrays = evaluate(model, loader, device, ck["kw"]["T"])
    print(f"{ck['model']} (epoch {ck['epoch']}, val acc {ck['val_acc']:.4f}) on {a.split}")
    print(format_table(rows))
    run = Path(a.checkpoint).parent
    np.savez_compressed(run / f"{a.split}_probs.npz", **arrays)
    json.dump(rows, open(run / f"{a.split}_metrics.json", "w"), indent=1)
    json.dump(curves(arrays), open(run / f"{a.split}_curves.json", "w"))
    print("saved", run / f"{a.split}_probs.npz", "and", run / f"{a.split}_curves.json")


if __name__ == "__main__":
    main()
