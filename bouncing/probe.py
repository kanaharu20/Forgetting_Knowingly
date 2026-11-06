"""Linear probes on the state a question is read from: what does the model
still *know* about the evidence, whether or not it uses it?

    python probe.py runs/transformer runs/gru_h16 runs/window --out figures/probe.png

For each run, the state h_t (models' `states()`, the only thing the readout
sees besides the question and the current frame) is collected for every
question of the fit split (val) and the test split. Ridge-regression probes
are fit on val and scored on test:

  time        h_t -> t / T                        all questions        R^2
  EVER_HIT    h_t -> log(1 + distance)            per (ball, wall), "yes" only   R^2,
              and the mean predicted distance in 25-frame bins (figure)
  LAST_HIT    h_t -> log(1 + distance)            per wall
  ever index  h_t -> has `ball` hit `wall` yet?   per (ball, wall)     accuracy (sign of ridge)

The intended reading: a probe that can read the distance means the state
carries a trace of how old the evidence is. What it found on the original
data (2026-11-04) was the opposite of the prediction: the windowed state
reads the distance best (R^2 0.80), the GRU barely (0.15). The reason is
the simulator, not the models: with specular bounces the distance is a
function of the current position and velocity (see leak_check.py), so any
state that holds the current kinematics "knows" it. Re-run on data made
with --resample before drawing conclusions about memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_loader, T_COL, QTYPE, BALL, COND, ANSWER, EVIDENCE
from eval import load_checkpoint
from questions import EVER_HIT, LAST_HIT
from train import pick_device

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9b9a96", "#7b5ea7"]


@torch.no_grad()
def collect(model, loader, device):
    hs, qs = [], []
    for pos, q, qmask in loader:
        h = model.states(pos.to(device), q.to(device))
        hs.append(h[qmask.to(device)].cpu().numpy()); qs.append(q[qmask].numpy())
    return np.concatenate(hs).astype(np.float64), np.concatenate(qs)


class Ridge:
    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        A = Z.T @ Z + self.lam * len(X) * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, X):
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.w


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def grouped_probe(h_fit, q_fit, h_te, q_te, select, key, target):
    """Fit one ridge per group `key(q)` on rows `select(q)`; return test predictions
    (NaN where no probe) and the mask of scored rows."""
    pred = np.full(len(q_te), np.nan)
    kf, kt = key(q_fit), key(q_te)
    for g in np.unique(kt[select(q_te)]):
        mf = select(q_fit) & (kf == g)
        mt = select(q_te) & (kt == g)
        if mf.sum() < 50 or mt.sum() == 0:
            continue
        pred[mt] = Ridge().fit(h_fit[mf], target(q_fit[mf])).predict(h_te[mt])
    return pred, ~np.isnan(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fit", default="val")
    ap.add_argument("--test", default="test")
    ap.add_argument("--data", default="data")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="figures/probe.png")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = pick_device(a.device)

    dist = lambda q: np.where(q[:, EVIDENCE] >= 0, q[:, T_COL] - q[:, EVIDENCE], -1)
    logd = lambda q: np.log1p(dist(q).astype(np.float64))
    results, curves = {}, {}
    print(f"{'run':<14}{'h_dim':>6}{'R2 time':>9}{'R2 EVER':>9}{'<32':>7}{'32-128':>8}{'128+':>7}{'R2 LAST':>9}{'ever idx':>10}")
    for run in a.runs:
        model, ck = load_checkpoint(Path(run) / "best.pt", device)
        T = ck["kw"]["T"]
        model.eval()
        h_fit, q_fit = collect(model, make_loader(Path(a.data) / f"{a.fit}.npz", a.batch, shuffle=False), device)
        h_te, q_te = collect(model, make_loader(Path(a.data) / f"{a.test}.npz", a.batch, shuffle=False), device)
        row = dict(h_dim=int(h_fit.shape[1]))

        # time
        p = Ridge().fit(h_fit, q_fit[:, T_COL] / T).predict(h_te)
        row["r2_time"] = r2(q_te[:, T_COL] / T, p)

        # EVER_HIT "yes": distance per (ball, wall)
        yes = lambda q: (q[:, QTYPE] == EVER_HIT) & (q[:, ANSWER] == 1)
        p, m = grouped_probe(h_fit, q_fit, h_te, q_te, yes, lambda q: q[:, BALL] * 4 + q[:, COND], logd)
        d_te = dist(q_te)
        row["r2_ever"] = r2(logd(q_te)[m], p[m])
        for name, lo, hi in (("<32", 0, 32), ("32-128", 32, 128), ("128+", 128, T)):
            mm = m & (d_te >= lo) & (d_te < hi)
            row[f"r2_ever_{name}"] = r2(logd(q_te)[mm], p[mm]) if mm.sum() > 50 else float("nan")
        # curve: mean predicted distance per true-distance bin
        xs, ys = [], []
        for lo in range(0, T, 25):
            mm = m & (d_te >= lo) & (d_te < lo + 25)
            if mm.sum() >= 30:
                xs.append(lo + 12.5); ys.append(float(np.expm1(p[mm]).mean()))
        curves[run] = (xs, ys)

        # LAST_HIT: distance per wall
        last = lambda q: q[:, QTYPE] == LAST_HIT
        p, m = grouped_probe(h_fit, q_fit, h_te, q_te, last, lambda q: q[:, COND], logd)
        row["r2_last"] = r2(logd(q_te)[m], p[m])

        # ever index: yes/no per (ball, wall) from h alone
        ever = lambda q: q[:, QTYPE] == EVER_HIT
        p, m = grouped_probe(h_fit, q_fit, h_te, q_te, ever, lambda q: q[:, BALL] * 4 + q[:, COND],
                             lambda q: q[:, ANSWER].astype(np.float64) * 2 - 1)
        row["acc_ever_index"] = float(((p[m] > 0) == (q_te[m, ANSWER] == 1)).mean())

        results[run] = row
        print(f"{Path(run).name:<14}{row['h_dim']:>6}{row['r2_time']:>9.3f}{row['r2_ever']:>9.3f}"
              f"{row['r2_ever_<32']:>7.2f}{row['r2_ever_32-128']:>8.2f}{row['r2_ever_128+']:>7.2f}"
              f"{row['r2_last']:>9.3f}{row['acc_ever_index']:>10.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(Path(a.out).with_suffix(".json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    names = [Path(r).name for r in a.runs]
    for i, (run, c) in enumerate(zip(a.runs, COLORS)):
        r = results[run]
        ax1.bar(i - 0.2, max(r["r2_time"], 0), 0.38, color=c, alpha=0.55)
        ax1.bar(i + 0.2, max(r["r2_ever"], 0), 0.38, color=c)
        xs, ys = curves[run]
        ax2.plot(xs, ys, "-o", color=c, ms=3.5, lw=2, label=names[i])
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=20, fontsize=8)
    ax1.set_ylim(0, 1.02); ax1.set_ylabel("R² of a linear probe on h_t")
    ax1.set_title("faint = time t,  solid = EVER_HIT distance", fontsize=10)
    T = max(ck["kw"]["T"] for _ in [0])
    ax2.plot([0, T], [0, T], ls=(0, (4, 3)), color="#52514e", lw=1, label="perfect")
    ax2.set_xlabel("true reference distance (EVER_HIT 'yes')"); ax2.set_ylabel("distance read from h_t by the probe")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.grid(color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Does the state still carry how old the evidence is?", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2, wspace=0.25)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out, "and", Path(a.out).with_suffix(".json"))


if __name__ == "__main__":
    main()
