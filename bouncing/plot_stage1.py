"""学習済みモデル1つ（既定は対照）の段階1まとめ図。

    python plot_stage1.py runs/transformer --out runs/transformer/stage1.png

パネル：
  A  質問の種類ごとの検証正解率のエポック推移（train.log から）、
     0.98 の合格線つき
  B  質問の種類ごとのテスト正解率と ECE
  C  参照距離の区間ごとの校正：テスト集合の全 (qtype, bin) について
     平均確信 vs 正解率；対角線上なら正直
  D  記憶を問う3種の質問について、参照距離に対する正解率（実線）と
     平均確信（破線）、テスト集合

質問の種類ごとに1色、全パネルで固定。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import QTYPE
from questions import QTYPE_NAMES, LAST_HIT, EVER_HIT, MOST_HITS

# カテゴリ用パレット、順序固定：NOW_HALF, LAST_HIT, EVER_HIT, MOST_HITS
COLORS = {"NOW_HALF": "#2a78d6", "LAST_HIT": "#eb6834", "EVER_HIT": "#1baf7a", "MOST_HITS": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PASS = 0.98


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def read_epochs(log: Path) -> dict[str, list[float]]:
    acc = {k: [] for k in QTYPE_NAMES}
    for line in log.read_text().splitlines():
        if line.startswith("per-qtype acc:"):
            for k, v in re.findall(r"'(\w+)': ([\d.]+)", line):
                acc[k].append(float(v))
    return acc


def panel_epochs(ax, acc):
    n = len(next(iter(acc.values())))
    x = np.arange(1, n + 1)
    for k in QTYPE_NAMES:
        ax.plot(x, acc[k], "-", color=COLORS[k], lw=2, label=k)
    ax.axhline(PASS, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(n, PASS - 0.012, f"pass line {PASS}", color=INK2, fontsize=8, ha="right")
    ax.set_xlim(1, n); ax.set_ylim(0.75, 1.005)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title("A  learning curve per question type")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def panel_per_qtype(ax, rows):
    tot = [r for r in rows if r["bin"] == "all" and r["qtype"] != "ALL"]
    y = np.arange(len(tot))[::-1]
    ax.axvline(PASS, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(PASS - 0.002, y[0] + 0.5, f"pass line {PASS}", color=INK2, fontsize=8, ha="right")
    for yi, r in zip(y, tot):
        ax.scatter(r["acc"], yi, s=90, color=COLORS[r["qtype"]], zorder=3, edgecolor="white", linewidth=1)
        ax.text(r["acc"] + 0.003, yi, f"{r['acc']:.3f}   ECE {r['ece']:.3f}   n={r['n']:,}",
                color=INK, fontsize=8.5, va="center")
    ax.set_yticks(y); ax.set_yticklabels([r["qtype"] for r in tot], color=INK)
    ax.set_xlim(0.95, 1.03); ax.set_ylim(-0.7, len(tot) - 0.3)
    ax.set_xticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0]); ax.set_xticklabels(["0.95", "", "0.97", "", "0.99", "1.00"])
    ax.set_xlabel("test accuracy")
    ax.set_title("B  test accuracy per question type")
    ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")


def panel_calibration(ax, rows):
    pts = [r for r in rows if r["qtype"] != "ALL" and r["bin"] != "all"]
    lo = min(min(r["acc"], r["conf"]) for r in pts) - 0.01
    ax.plot([lo, 1], [lo, 1], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(lo + 0.003, lo + 0.012, "calibrated: confidence = accuracy", color=INK2, fontsize=8, rotation=45,
            rotation_mode="anchor")
    for r in pts:
        c = COLORS[r["qtype"]]
        ax.scatter(r["conf"], r["acc"], s=12 + 1.2 * np.sqrt(r["n"]), color=c, alpha=0.85,
                   edgecolor="white", linewidth=1, zorder=3)
        if abs(r["acc"] - r["conf"]) > 0.015:
            ax.annotate(f"{r['qtype']} {r['bin']}", (r["conf"], r["acc"]), xytext=(6, -4),
                        textcoords="offset points", fontsize=7.5, color=INK)
    for k in QTYPE_NAMES:
        ax.scatter([], [], color=COLORS[k], s=40, label=k)
    ax.set_xlim(lo, 1.002); ax.set_ylim(lo, 1.002); ax.set_aspect("equal")
    ax.set_xlabel("mean confidence in the chosen answer"); ax.set_ylabel("accuracy")
    ax.set_title("C  calibration per distance bin (dot size = n)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def binned(dist, y, width, T, min_n=30):
    edges = np.arange(0, T + width, width)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dist >= a) & (dist < b)
        if m.sum() >= min_n:
            xs.append((a + b) / 2); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def panel_distance(axes, d, width=25):
    q = d["questions"]; T = int(q[:, 1].max()) + 1
    for ax, k in zip(axes, (LAST_HIT, EVER_HIT, MOST_HITS)):
        name = QTYPE_NAMES[k]; c = COLORS[name]
        m = (q[:, QTYPE] == k) & (d["dist"] >= 0)
        x, acc = binned(d["dist"][m], d["correct"][m], width, T)
        _, conf = binned(d["dist"][m], d["conf"][m], width, T)
        ax.plot(x, acc, "-", color=c, lw=2, label="accuracy")
        ax.plot(x, conf, color=c, lw=1.6, ls=(0, (4, 2)), label="mean confidence")
        ax.text(x[-1] + 6, acc[-1], "acc", color=INK, fontsize=8, va="center")
        ax.text(x[-1] + 6, conf[-1], "conf", color=INK2, fontsize=8, va="center")
        if k == EVER_HIT:
            mn = (q[:, QTYPE] == k) & (d["dist"] < 0)
            ax.scatter([T * 1.04], [d["correct"][mn].mean()], color=c, s=50, zorder=3)
            ax.text(T * 1.04, d["correct"][mn].mean() - 0.04, "'no'\nquestions", ha="center", fontsize=7.5, color=INK2)
        ax.set_title(f"D  {name}")
        ax.set_xlabel("reference distance (frames since the evidence)")
        ax.set_ylim(0.8, 1.01)
        style(ax)
    axes[0].set_ylabel("test accuracy (solid) / mean confidence (dashed)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/transformer")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Path(a.run)
    out = a.out or run / "stage1.png"

    acc = read_epochs(run / "train.log")
    rows = json.load(open(run / "test_metrics.json"))
    probs = np.load(run / "test_probs.npz")

    fig = plt.figure(figsize=(15, 9), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    axD = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in (axA, axB, axC):
        style(ax)
    panel_epochs(axA, acc)
    panel_per_qtype(axB, rows)
    panel_calibration(axC, rows)
    panel_distance(axD, probs)
    total = [r for r in rows if r["qtype"] == "ALL"][0]
    fig.suptitle(f"Stage 1 — control model ({run.name}): test accuracy {total['acc']:.3f}, ECE {total['ece']:.3f}",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
