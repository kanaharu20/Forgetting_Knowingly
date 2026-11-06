"""1本のストリームを描く: 壁ヒットを印した軌跡と、参照距離のヒストグラム。

python render.py --seed 0 --out stream0.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import generate_stream, WALL_NAMES
from questions import generate_dataset, QTYPE_NAMES

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


def draw(seed: int, out: str, T: int = 500, n_balls: int = 3):
    s = generate_stream(n_balls, T, seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    for b in range(n_balls):
        xy = s.pos[:, b]
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[b], lw=0.8, alpha=0.7, label=f"ball {b}")
        ax.plot(xy[0, 0], xy[0, 1], "o", color=COLORS[b], ms=8)
    for t, b, w in s.events:
        ax.plot(s.pos[t, b, 0], s.pos[t, b, 1], "x", color=COLORS[b], ms=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"seed {seed}: {T} frames, {len(s.events)} wall hits (x = hit, o = start)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ds = generate_dataset(n_streams=100, T=T, n_balls=n_balls, frames_per_stream=32, per_frame=8, seed=seed + 1)
    q = ds["questions"]
    for k in (1, 2, 3):
        sub = q[(q[:, 2] == k) & (q[:, 6] >= 0)]
        ax.hist(sub[:, 1] - sub[:, 6], bins=50, range=(0, T), histtype="step", lw=1.5, label=QTYPE_NAMES[k])
    ax.set_xlabel("reference distance (frames since the evidence)")
    ax.set_ylabel("questions")
    ax.set_yscale("log")
    ax.set_title("how far back the questions reach (100 streams)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="stream0.png")
    a = ap.parse_args()
    draw(a.seed, a.out)
