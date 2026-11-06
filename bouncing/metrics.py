"""正解率・確信・校正を、質問の種類と参照距離で層別して集計する。
train.py（検証）と eval.py（テスト）が使う。

  conf     = モデルが自分の選んだ答えに置いた確率
  correct  = その答えが正しかったか
  ECE      = expected calibration error（期待校正誤差）: 質問を確信の
             区間に振り分け、|平均確信 - 正解率| を区間の大きさで
             重みづけして区間にわたり平均したもの。0 = 完全に校正されている。

「正直な忘却」とは、距離区間の中で平均確信が正解率に追従すること
（ECE が小さい）。「気づかない忘却」とは、確信が高いまま正解率が落ちること。
"""
from __future__ import annotations

import numpy as np

from questions import BIN_EDGES, QTYPE_NAMES, EVER_HIT

DIST_NO_EVIDENCE = -1   # EVER_HIT の答えが「いいえ」: 独立したグループ


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def distance_bins(T: int) -> list[tuple[int, int]]:
    edges = list(BIN_EDGES) + [T]
    return list(zip(edges[:-1], edges[1:]))


def summarize(qtype: np.ndarray, dist: np.ndarray, correct: np.ndarray, conf: np.ndarray,
              T: int) -> list[dict]:
    """(qtype, 距離区間) ごとに1行。加えて qtype ごとの合計と全体の合計。"""
    rows = []
    for k, name in enumerate(QTYPE_NAMES):
        m_k = qtype == k
        groups = []
        if k == EVER_HIT:
            groups.append(("no-evidence", m_k & (dist == DIST_NO_EVIDENCE)))
        for lo, hi in distance_bins(T):
            groups.append((f"[{lo},{hi})", m_k & (dist >= lo) & (dist < hi)))
        groups.append(("all", m_k))
        for label, m in groups:
            if m.sum() == 0:
                continue
            rows.append(dict(qtype=name, bin=label, n=int(m.sum()), acc=float(correct[m].mean()),
                             conf=float(conf[m].mean()), ece=ece(conf[m], correct[m])))
    rows.append(dict(qtype="ALL", bin="all", n=len(correct), acc=float(correct.mean()),
                     conf=float(conf.mean()), ece=ece(conf, correct)))
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'qtype':10s} {'ref-distance':13s} {'n':>8s} {'acc':>7s} {'conf':>7s} {'ECE':>7s}"]
    for r in rows:
        lines.append(f"{r['qtype']:10s} {r['bin']:13s} {r['n']:8d} {r['acc']:7.4f} {r['conf']:7.4f} {r['ece']:7.4f}")
    return "\n".join(lines)


def per_qtype_accuracy(rows: list[dict]) -> dict[str, float]:
    return {r["qtype"]: r["acc"] for r in rows if r["bin"] == "all" and r["qtype"] != "ALL"}
