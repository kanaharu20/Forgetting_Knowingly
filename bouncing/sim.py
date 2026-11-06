"""跳ねる玉のストリームのシミュレータ。

ストリームはフレームの列。各フレームは単位正方形の中にある N 個の玉の
(x, y) 座標を持つ。玉は直線で動き、4 つの壁で反射する。玉同士の衝突は
無視する。

壁の id:  0 = 左 (x=0), 1 = 右 (x=1), 2 = 下 (y=0), 3 = 上 (y=1)

壁へのヒットはすべて出来事 (t, ball, wall) として記録する。長期記憶を
問う質問はこの出来事について尋ねる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3
WALL_NAMES = ["left", "right", "bottom", "top"]


@dataclass
class Stream:
    pos: np.ndarray        # (T, N, 2) 各フレームの座標、値は [0, 1]
    events: np.ndarray     # (E, 3) の整数行 (t, ball, wall)。t は玉がすでに反射した後のフレーム
    n_balls: int
    seed: int

    @property
    def T(self) -> int:
        return self.pos.shape[0]


def generate_stream(n_balls: int, T: int, seed: int,
                    radius: float = 0.03,
                    speed_range: tuple[float, float] = (0.01, 0.04),
                    resample: bool = False) -> Stream:
    """ストリームを 1 本シミュレートする。

    各玉は正方形内のランダムな位置から始まり、速さが speed_range に入る
    ランダムな速度を持つ（単位: 1 フレームあたりの正方形に対する割合）。
    速さ 0.02 なら玉は約 50 フレームで正方形を横断するので、500 フレームの
    ストリームでは玉 1 個あたりおよそ 10 回の壁ヒットが起きる。

    resample=True だと、壁に当たるたびに速さと向きを引き直す（向きは壁から
    離れる半平面で一様）。鏡面反射だけだと軸方向の速さが変わらないので、
    「最後にこの壁に当たってからの時間」が現在の座標と速度から計算できて
    しまう（leak_check.py、2026-11-04 に発見）。引き直すと、最後の反射より
    前の履歴は現在のフレームから復元できず、質問が本当に記憶を要する。
    resample=False のときの乱数の消費は従来どおりで、既存データは再現できる。
    """
    rng = np.random.default_rng(seed)
    lo, hi = radius, 1.0 - radius

    p = rng.uniform(lo, hi, size=(n_balls, 2))
    speed = rng.uniform(*speed_range, size=n_balls)
    angle = rng.uniform(0, 2 * np.pi, size=n_balls)
    v = np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=1)

    pos = np.empty((T, n_balls, 2))
    events: list[tuple[int, int, int]] = []
    pos[0] = p

    def bounce(b: int, axis: int, sign: float) -> None:
        """Reflect ball b on `axis`; `sign` = +1 leaving a low wall, -1 a high wall."""
        if not resample:
            v[b, axis] = -v[b, axis]
            return
        s = rng.uniform(*speed_range)
        a = rng.uniform(-np.pi / 2, np.pi / 2)          # half-plane away from the wall
        v[b, axis] = sign * s * np.cos(a)
        v[b, 1 - axis] = s * np.sin(a)

    for t in range(1, T):
        p = p + v
        for b in range(n_balls):
            for axis, (wall_lo, wall_hi) in enumerate([(LEFT, RIGHT), (BOTTOM, TOP)]):
                if p[b, axis] < lo:
                    p[b, axis] = 2 * lo - p[b, axis]
                    bounce(b, axis, +1.0)
                    events.append((t, b, wall_lo))
                elif p[b, axis] > hi:
                    p[b, axis] = 2 * hi - p[b, axis]
                    bounce(b, axis, -1.0)
                    events.append((t, b, wall_hi))
        pos[t] = p

    ev = np.array(events, dtype=np.int64).reshape(-1, 3)
    return Stream(pos=pos, events=ev, n_balls=n_balls, seed=seed)


if __name__ == "__main__":
    s = generate_stream(n_balls=3, T=500, seed=0)
    print("frames", s.T, "balls", s.n_balls, "events", len(s.events))
    print("first events (t, ball, wall):")
    for t, b, w in s.events[:8]:
        print(f"  t={t:4d} ball={b} wall={WALL_NAMES[w]}")
