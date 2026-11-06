"""段階3、切り替え機構: 一部の質問を固定状態モデルから全文脈の対照に回し、
読み直し 1 単位あたりでどれだけ正解率が買えるかを見る。

    python switch.py runs/window runs/gru_h16 --control runs/transformer --out figures/switch.png

どの質問も固定状態モデルが答えるが、*信号* が読み直せと言った質問だけは
対照の答えを使う。予算は読み直した質問の割合 (読み直し 1 回に全文脈の
1 パスがかかるので、コストはこれに比例する)。4 つの信号を比べる:

  confidence  固定状態モデルの確信が最も低い質問を読み直す。
              実運用で使える: モデル自身の出力しか要らない。モデルが
              正直に忘れている範囲でちょうど効く。
  distance    根拠が最も古い質問を読み直す。そのままでは実運用で使えない
              (モデルは距離を見ていない) が、薄れていく痕跡を保つ状態なら
              近似できるはずのもの。
  oracle      対照が正解に最も多く確率を上乗せする所を読み直す。対照が
              要るので、あらゆる信号の上限であって方針ではない。
  random      基準線: 一様乱数で読み直す。

出力: 固定状態の run ごとに 1 行。列 = 全質問と、同じ大域的な方針の下での
記憶を問う各質問の種類。印字する表: 予算 10 / 25 / 50 % での正解率と、
対照の 0.01 以内に入るのに必要な予算。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import QTYPE, ANSWER
from questions import LAST_HIT, EVER_HIT, MOST_HITS

SIGNALS = ["confidence", "distance", "oracle", "random"]
COLORS = {"confidence": "#2a78d6", "distance": "#1baf7a", "oracle": "#eb6834", "random": "#9b9a96"}
BUDGETS = [0.1, 0.25, 0.5]


def load(run: str, split: str):
    d = np.load(Path(run) / f"{split}_probs.npz")
    return {k: d[k] for k in d.files}


def signals(m: dict, c: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """値が高いほど先に読み直す。"""
    ans = m["questions"][:, ANSWER]
    idx = np.arange(len(ans))
    rng = np.random.default_rng(seed)
    return {
        "confidence": -m["conf"],
        "distance": m["dist"].astype(np.float64),               # -1 (本当の「いいえ」) は最後
        "oracle": c["probs"][idx, ans] - m["probs"][idx, ans],
        "random": rng.random(len(ans)),
    }


def curve(order: np.ndarray, correct_m: np.ndarray, correct_c: np.ndarray, subset: np.ndarray,
          grid: np.ndarray) -> np.ndarray:
    """`order` (大域) の先頭 f·n 問を読み直したときの `subset` 上の正解率。"""
    n = len(order)
    rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)
    out = []
    for f in grid:
        reread = rank < int(round(f * n))
        acc = np.where(reread, correct_c, correct_m)[subset].mean()
        out.append(acc)
    return np.array(out)


def budget_to_reach(grid: np.ndarray, accs: np.ndarray, target: float) -> float:
    hit = np.nonzero(accs >= target)[0]
    return float(grid[hit[0]]) if len(hit) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="fixed-state run directories")
    ap.add_argument("--control", default="runs/transformer")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="figures/switch.png")
    a = ap.parse_args()

    c = load(a.control, a.split)
    grid = np.linspace(0, 1, 101)
    cols = [("ALL", None), ("LAST_HIT", LAST_HIT), ("EVER_HIT", EVER_HIT), ("MOST_HITS", MOST_HITS)]
    fig, axes = plt.subplots(len(a.runs), len(cols), figsize=(4.2 * len(cols), 3.4 * len(a.runs) + 0.6),
                             squeeze=False, sharex=True)
    print(f"{'run':<14}{'signal':<12}{'acc@0':>7}{'@10%':>7}{'@25%':>7}{'@50%':>7}{'budget to control-0.01':>24}")
    residual = []
    for r, run in enumerate(a.runs):
        m = load(run, a.split)
        assert (m["questions"] == c["questions"]).all(), f"{run} and {a.control} were scored on different questions"
        qt = m["questions"][:, QTYPE]
        sig = signals(m, c)
        # confidence 信号には見えないもの: 高い確信のまま犯した誤り
        wrong = m["correct"] == 0
        for label, k in cols:
            sub = wrong if k is None else wrong & (qt == k)
            residual.append((Path(run).name, label, int(sub.sum()), float((m["conf"][sub] >= 0.9).mean()) if sub.any() else 0.0))
        for col, (label, k) in enumerate(cols):
            ax = axes[r, col]
            subset = np.ones(len(qt), bool) if k is None else qt == k
            acc_m, acc_c = m["correct"][subset].mean(), c["correct"][subset].mean()
            for s in SIGNALS:
                order = np.argsort(-sig[s], kind="stable")
                accs = curve(order, m["correct"], c["correct"], subset, grid)
                ax.plot(grid, accs, color=COLORS[s], lw=2 if s != "random" else 1.4,
                        ls="-" if s != "random" else (0, (3, 2)), label=s)
                if k is None:
                    at = [accs[int(round(b * 100))] for b in BUDGETS]
                    print(f"{Path(run).name:<14}{s:<12}{acc_m:7.3f}{at[0]:7.3f}{at[1]:7.3f}{at[2]:7.3f}"
                          f"{budget_to_reach(grid, accs, acc_c - 0.01):24.2f}")
            ax.axhline(acc_c, color="#52514e", lw=1, ls=":")
            ax.text(0.99, acc_c, "control", ha="right", va="bottom", fontsize=7.5, color="#52514e")
            ax.set_ylim(min(0.5, acc_m - 0.05), 1.01)
            ax.grid(color="#e6e5e1", lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{Path(run).name}\naccuracy")
            if r == len(a.runs) - 1:
                ax.set_xlabel("fraction of questions re-read by the control")
    print("\nconfident errors (confidence >= 0.9) as a share of all errors: the part no confidence threshold can route")
    print(f"{'run':<14}{'subset':<11}{'errors':>8}{'share confident':>17}")
    for name, label, n, share in residual:
        print(f"{name:<14}{label:<11}{n:8d}{share:17.2f}")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Switching: which questions should be handed to the full-context model?", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
