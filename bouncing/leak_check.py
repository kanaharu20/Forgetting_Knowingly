"""Does the *present* give away the past? A memoryless check on a dataset.

    python leak_check.py data/test.npz

With specular bounces the per-axis speed of a ball never changes, so the
time since its last hit on a wall is a function of its current position
and velocity: gap / |v| if it is moving away from that wall, (2 - gap) / |v|
if it is moving towards it (it last left that wall, crossed the box and
bounced back). This script evaluates three memoryless predictors built
from frames t-1 and t only:

  EVER_HIT "yes"   reference distance from the formula above (median error)
  LAST_HIT         the ball with the smallest formula time       (accuracy)
  MOST_HITS        the ball with the largest |v| along that axis (accuracy)

If these are good, the reference distance is not a memory demand but a
kinematic quantity, and a model with no memory can look like it "remembers".
Found on 2026-11-04 by the linear probes (probe.py). The fix is a simulator
whose bounces resample the velocity (sim.generate_stream(..., resample=True)).
"""
from __future__ import annotations

import argparse

import numpy as np

from data import QTYPE, ANSWER, T_COL, BALL, COND, EVIDENCE
from questions import LAST_HIT, EVER_HIT, MOST_HITS


def formula_time(pos, s, t, b, axis, wall):
    """Frames since ball b last hit `wall` (0.0 or 1.0 on `axis`), from frames t-1, t."""
    x = pos[s, t, b, axis]
    v = pos[s, t, b, axis] - pos[s, t - 1, b, axis]
    away = np.sign(v) == np.sign(x - wall)
    gap, sp = np.abs(x - wall), np.abs(v) + 1e-9
    return np.where(away, gap / sp, (2 - gap) / sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", default="data/test.npz", nargs="?")
    a = ap.parse_args()
    d = np.load(a.data)
    pos, q = d["pos"], d["questions"]
    N = pos.shape[2]
    if "resample" in d:
        print("dataset resample-at-bounce:", bool(d["resample"]))

    m = (q[:, QTYPE] == EVER_HIT) & (q[:, ANSWER] == 1) & (q[:, T_COL] >= 1)
    qq = q[m]; s, t, b, c = qq[:, 0], qq[:, T_COL], qq[:, BALL], qq[:, COND]
    dist = (t - qq[:, EVIDENCE]).astype(float)
    pred = formula_time(pos, s, t, b, np.where(c < 2, 0, 1), np.where(c % 2 == 0, 0.0, 1.0))
    print(f"EVER_HIT 'yes' distance from kinematics (n={len(dist)}), median |error| per true-distance bin:")
    for lo, hi in [(0, 32), (32, 100), (100, 200), (200, 500)]:
        k = (dist >= lo) & (dist < hi)
        if k.any():
            print(f"  [{lo:3d},{hi:3d}): n={k.sum():6d}  median |err| = {np.median(np.abs(pred[k] - dist[k])):6.1f} frames"
                  f"   (within 10%: {(np.abs(pred[k] - dist[k]) <= 0.1 * np.maximum(dist[k], 1)).mean():.2f})")

    for name, kind in (("LAST_HIT", LAST_HIT), ("MOST_HITS", MOST_HITS)):
        m = (q[:, QTYPE] == kind) & (q[:, COND] < 4) & (q[:, T_COL] >= 1)
        qq = q[m]; s, t, c = qq[:, 0], qq[:, T_COL], qq[:, COND]
        axis, wall = np.where(c < 2, 0, 1), np.where(c % 2 == 0, 0.0, 1.0)
        cols = []
        for bb in range(N):
            if kind == LAST_HIT:
                cols.append(formula_time(pos, s, t, bb, axis, wall))
            else:
                cols.append(-np.abs(pos[s, t, bb, axis] - pos[s, t - 1, bb, axis]))
        guess = np.stack(cols, 1).argmin(1)
        dl = t - qq[:, EVIDENCE]
        print(f"{name}: memoryless kinematic guess accuracy = {(guess == qq[:, ANSWER]).mean():.3f} (n={len(qq)}, chance {1 / N:.2f})")
        for lo, hi in [(0, 32), (32, 100), (100, 200), (200, 500)]:
            k = (dl >= lo) & (dl < hi)
            if k.any():
                print(f"  dist [{lo:3d},{hi:3d}): acc = {(guess[k] == qq[k, ANSWER]).mean():.3f}  n={k.sum()}")


if __name__ == "__main__":
    main()
