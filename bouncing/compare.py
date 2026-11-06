"""同じ分割上で複数の run を並べて比較する。

    python compare.py runs/transformer runs/gru runs/window [--split test]

質問の種類と参照距離の区間ごとに、run 別の正解率、平均確信、
過信の差（確信 − 正解率）を出力する。
H2 の予想では、固定状態の run は距離とともに差が広がり、
対照の差はゼロ近くにとどまる。さらに固定状態の run については、
正解に対する（対照の確率 − run の確率）の平均も出力する。これは
「読み直せば消える不確かさ」であり、段階3の切り替え信号になる。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import QTYPE, ANSWER
from metrics import distance_bins, DIST_NO_EVIDENCE
from questions import QTYPE_NAMES, EVER_HIT


def load(run: Path, split: str):
    d = np.load(run / f"{split}_probs.npz")
    return {k: d[k] for k in d.files}


def groups(q, dist, T):
    out = []
    for k, name in enumerate(QTYPE_NAMES):
        m_k = q[:, QTYPE] == k
        if k == EVER_HIT:
            out.append((name, "no-evidence", m_k & (dist == DIST_NO_EVIDENCE)))
        for lo, hi in distance_bins(T):
            out.append((name, f"[{lo},{hi})", m_k & (dist >= lo) & (dist < hi)))
        out.append((name, "all", m_k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    data = {r.name: load(r, a.split) for r in runs}
    ref = data[runs[0].name]
    q = ref["questions"]; dist = ref["dist"]; T = int(q[:, 1].max()) + 1
    for name, d in data.items():
        assert np.array_equal(d["questions"], q), f"{name}: question table differs from {runs[0].name}"

    names = [r.name for r in runs]
    print(f"{'':30s} |" + "".join(f" {n[:19]:^19s} |" for n in names))
    print(f"{'qtype':9s} {'bin':12s} {'n':>7s} |" + "".join(f" {'acc':>5s} {'conf':>5s} {'gap':>6s} |" for n in names))
    for qname, label, m in groups(q, dist, T):
        if m.sum() == 0:
            continue
        line = f"{qname:9s} {label:12s} {int(m.sum()):7d} |"
        for n in names:
            d = data[n]
            acc, conf = d["correct"][m].mean(), d["conf"][m].mean()
            line += f" {acc:5.3f} {conf:5.3f} {conf - acc:+6.3f} |"
        print(line)

    if len(runs) > 1:
        print("\nmean(control prob on correct answer − run prob on correct answer), i.e. what re-reading would recover:")
        p_ctrl = ref["probs"][np.arange(len(q)), q[:, ANSWER]]
        print(f"{'qtype':9s} {'bin':12s} |" + "".join(f" {n[:12]:>12s} |" for n in names[1:]))
        for qname, label, m in groups(q, dist, T):
            if m.sum() == 0:
                continue
            line = f"{qname:9s} {label:12s} |"
            for n in names[1:]:
                p = data[n]["probs"][np.arange(len(q)), q[:, ANSWER]]
                line += f" {np.mean(p_ctrl[m] - p[m]):+12.3f} |"
            print(line)


if __name__ == "__main__":
    main()
