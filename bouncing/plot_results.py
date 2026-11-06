"""主要な図：参照距離に対する正解率と平均確信。
記憶を問う質問の種類ごとに1パネル、モデルごとに1色。

    python plot_results.py runs/transformer/test_probs.npz runs/gru/test_probs.npz --out results.png

入力は <run>/<split>_probs.npz か、eval.py がその隣に書く小さな
<run>/<split>_curves.json（曲線は同じで、質問ごとのデータはない）。

上の行：正解率（実線）と平均確信（破線）。2本が一緒に落ちれば
正直な忘却、開いていけば気づかない忘却。
下の行：差そのもの、すなわちモデルごとの平均確信 − 正解率。ゼロより
上なら過信。これが仮説 H2 をそのまま読んだものである。EVER_HIT の
「いいえ」質問（根拠フレームなし）は右端にマーカーで示す。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import QTYPE
from questions import QTYPE_NAMES, LAST_HIT, EVER_HIT, MOST_HITS

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # 順序固定：run ごとに1色
# --ordered: 大きさの順に並んだ run（状態サイズ、λ など）を青の単色で薄→濃に塗る
ORDERED = {4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
           5: ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]}


import json

MIN_N = 30


def load_curves(path: str, width: int):
    """npz か curves.json から -> (T, {qtype_name: (x, acc, conf, no_evidence_or_None)}) を返す。"""
    if path.endswith(".json"):
        c = json.load(open(path))
        assert c["width"] == width, f"{path} was binned with width {c['width']}, asked for {width}"
        out = {}
        for name, d in c["qtypes"].items():
            b = [x for x in d["bins"] if x["n"] >= MIN_N]
            out[name] = (np.array([(x["lo"] + x["hi"]) / 2 for x in b]), np.array([x["acc"] for x in b]),
                         np.array([x["conf"] for x in b]), d["no_evidence"])
        return c["T"], out
    d = np.load(path)
    q = d["questions"]; T = int(q[:, 1].max()) + 1
    out = {}
    for k, name in enumerate(QTYPE_NAMES):
        m = (q[:, QTYPE] == k) & (d["dist"] >= 0)
        x, acc = binned(d["dist"][m], d["correct"][m], width, T)
        _, conf = binned(d["dist"][m], d["conf"][m], width, T)
        mn = (q[:, QTYPE] == k) & (d["dist"] < 0)
        ne = dict(n=int(mn.sum()), acc=float(d["correct"][mn].mean()), conf=float(d["conf"][mn].mean())) if mn.any() else None
        out[name] = (x, acc, conf, ne)
    return T, out


def binned(dist, y, width, T, min_n=MIN_N):
    edges = np.arange(0, T + width, width)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() >= min_n:
            xs.append((lo + hi) / 2); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="<run>/<split>_probs.npz, one per model")
    ap.add_argument("--out", default="results.png")
    ap.add_argument("--width", type=int, default=25, help="distance bin width in frames")
    ap.add_argument("--ordered", action="store_true",
                    help="runs are given in increasing order of one setting; colour them light -> dark")
    a = ap.parse_args()
    colors = ORDERED.get(len(a.files)) if a.ordered else COLORS
    if colors is None or len(a.files) > len(colors):
        raise SystemExit(f"{len(a.files)} runs: at most {len(COLORS)} (or 4-5 with --ordered)")

    runs = [(Path(f).parent.name, load_curves(f, a.width)) for f in a.files]
    T = max(t for _, (t, _) in runs)
    qtypes = [LAST_HIT, EVER_HIT, MOST_HITS]
    fig, axes = plt.subplots(2, len(qtypes), figsize=(5 * len(qtypes), 7.5), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12, wspace=0.18))
    for col, k in enumerate(qtypes):
        top, bot = axes[0, col], axes[1, col]
        for (name, (_, cv)), c in zip(runs, colors):
            x, acc, conf, ne = cv[QTYPE_NAMES[k]]
            top.plot(x, acc, "-", color=c, lw=2, label=f"{name}: accuracy")
            top.plot(x, conf, color=c, lw=1.6, ls=(0, (4, 2)), label=f"{name}: mean confidence")
            bot.plot(x, conf - acc, "-", color=c, lw=2, label=name)
            if k == EVER_HIT and ne:
                top.plot([T * 1.03], [ne["acc"]], "o", color=c, ms=7)
                top.plot([T * 1.03], [ne["conf"]], "o", color=c, ms=7, mfc="none")
                bot.plot([T * 1.03], [ne["conf"] - ne["acc"]], "o", color=c, ms=7)
        top.set_title(QTYPE_NAMES[k] + ("   (dots at right: 'no' questions)" if k == EVER_HIT else ""))
        top.set_ylim(0.0, 1.02)
        bot.axhline(0, color="#52514e", lw=1)
        bot.set_ylim(-0.3, 0.8)
        bot.set_xlabel("reference distance (frames since the evidence)")
        for ax in (top, bot):
            ax.grid(color="#e6e5e1", lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("accuracy (solid) / mean confidence (dashed)")
    axes[1, 0].set_ylabel("overconfidence\n(confidence − accuracy)")
    axes[0, 0].legend(fontsize=7.5, loc="lower left", frameon=False, ncol=1)
    axes[1, 0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.suptitle("Does confidence fall with accuracy as the evidence gets older?", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.08)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
