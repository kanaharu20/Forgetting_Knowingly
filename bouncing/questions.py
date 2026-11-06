"""跳ねる玉のストリームに対する型つき質問と、その答えと根拠。

質問は3つの整数 (qtype, ball, cond) である。フレーム t で問われ、
`n_options` 個の答えを持つ。自然言語で表現するものはなく、モデルは
この3つの整数を埋め込む。

qtype  意味                                             ball   cond        選択肢
-----  -----------------------------------------------  -----  ----------  -------
0      NOW_HALF   `ball` は今、半面 `cond` にいるか      0..N-1 0..3 半面   2 (no/yes)
1      LAST_HIT   `cond` に最後にヒットした玉はどれか    未使用 0..3 壁,    N
                                                                 4 = 任意の壁
2      EVER_HIT   `ball` は壁 `cond` にヒットしたことがあるか 0..N-1 0..3 壁 2 (no/yes)
3      MOST_HITS  壁 `cond` に最も多くヒットした玉はどれか 未使用 0..3 壁   N

qtype 0 の半面:  0 = 左 (x < .5), 1 = 右, 2 = 下 (y < .5), 3 = 上。

evidence_t は、答えるのに十分な最新の出来事のフレームである。
参照距離 = t - evidence_t が、この研究で x 軸に置く量である。約束事:
  NOW_HALF   evidence_t = t（距離 0）
  LAST_HIT   evidence_t = その最新のヒットのフレーム
  EVER_HIT   「はい」なら evidence_t = 条件に合う最新のヒットのフレーム。
             「いいえ」なら -1（単一のフレームでは足りない。モデルは履歴
             全体にそのヒットがないことを知っていなければならない — 別扱い）
  MOST_HITS  evidence_t = その壁にいずれかの玉が最後にヒットしたフレーム
             （集計が変わり得た最後の瞬間）

フレーム t で答えが定まらない質問（まだヒットがない、MOST_HITS で同数）
はサンプラーが飛ばす。

サンプラーは2つ:
  sample_questions             一様なフレーム、一様な質問（自然な分布。
                               遠い質問は稀）
  sample_questions_stratified  先に目標の参照距離の区間を引き、次にその
                               距離になるフレーム t を構成する。固定した
                               train/val/test 分割はこれを使う（2026-10-24
                               に決定、README 参照）。
"""
from __future__ import annotations

import numpy as np

from sim import Stream, generate_stream

NOW_HALF, LAST_HIT, EVER_HIT, MOST_HITS = 0, 1, 2, 3
QTYPE_NAMES = ["NOW_HALF", "LAST_HIT", "EVER_HIT", "MOST_HITS"]
ANY_WALL = 4

COLUMNS = ["stream", "t", "qtype", "ball", "cond", "answer", "evidence_t", "n_options"]

# 層別サンプラーと分析で使う参照距離の区間。
# 半開区間 [lo, hi)。最後の区間はストリーム長まで。
BIN_EDGES = [0, 1, 50, 100, 200]
# 各区間を狙う確率（LAST_HIT / EVER_HIT の「はい」 / MOST_HITS 用）。
# 200 までの区間が同程度の量になるように選んだ。200+ の区間はストリームの
# 中身に制限され、空ならひとつ下の区間に退避する。
BIN_WEIGHTS = [0.05, 0.30, 0.25, 0.25, 0.15]
# EVER_HIT で答えが「いいえ」(evidence_t = -1) になる質問の割合。
EVER_HIT_NO_FRAC = 0.35

DEFAULT_SPEED_RANGE = (0.003, 0.03)


def answer_question(s: Stream, t: int, qtype: int, ball: int, cond: int):
    """(answer, evidence_t, n_options) を返す。t で定まらなければ None。"""
    N = s.n_balls
    ev = s.events[s.events[:, 0] <= t]  # フレーム t（含む）までに分かっている出来事

    if qtype == NOW_HALF:
        x, y = s.pos[t, ball]
        inside = [x < 0.5, x >= 0.5, y < 0.5, y >= 0.5][cond]
        return int(inside), t, 2

    if qtype == LAST_HIT:
        m = ev if cond == ANY_WALL else ev[ev[:, 2] == cond]
        if len(m) == 0:
            return None
        last_t = m[-1, 0]
        at_last = m[m[:, 0] == last_t]
        if len(np.unique(at_last[:, 1])) > 1:   # 同じフレームで2つの玉がヒット
            return None
        return int(at_last[-1, 1]), int(last_t), N

    if qtype == EVER_HIT:
        m = ev[(ev[:, 1] == ball) & (ev[:, 2] == cond)]
        if len(m) == 0:
            return 0, -1, 2
        return 1, int(m[-1, 0]), 2

    if qtype == MOST_HITS:
        m = ev[ev[:, 2] == cond]
        if len(m) == 0:
            return None
        counts = np.bincount(m[:, 1], minlength=N)
        top = counts.max()
        if (counts == top).sum() > 1:
            return None
        return int(counts.argmax()), int(m[-1, 0]), N

    raise ValueError(qtype)


# ---------------------------------------------------------------- 一様サンプラー

def sample_questions(s: Stream, frames: np.ndarray, per_frame: int,
                     rng: np.random.Generator, stream_id: int) -> np.ndarray:
    """`frames` の各フレームで、定まった質問を `per_frame` 個サンプルする。

    int 配列 (rows, len(COLUMNS)) を返す。qtype は一様に引き、
    ball と cond はそれぞれの有効範囲から一様に引く。
    """
    N = s.n_balls
    rows = []
    for t in frames:
        got, tries = 0, 0
        while got < per_frame and tries < per_frame * 20:
            tries += 1
            qtype = int(rng.integers(4))
            ball = int(rng.integers(N)) if qtype in (NOW_HALF, EVER_HIT) else 0
            cond = int(rng.integers(5)) if qtype == LAST_HIT else int(rng.integers(4))
            out = answer_question(s, int(t), qtype, ball, cond)
            if out is None:
                continue
            ans, evidence_t, n_opt = out
            rows.append((stream_id, int(t), qtype, ball, cond, ans, evidence_t, n_opt))
            got += 1
    return np.array(rows, dtype=np.int64).reshape(-1, len(COLUMNS))


# ------------------------------------------------------------- 層別サンプラー

def _key_frames(s: Stream, qtype: int, ball: int, cond: int) -> np.ndarray:
    """この質問の根拠になり得る出来事のフレーム。"""
    ev = s.events
    if qtype == LAST_HIT:
        m = ev if cond == ANY_WALL else ev[ev[:, 2] == cond]
    elif qtype == EVER_HIT:
        m = ev[(ev[:, 1] == ball) & (ev[:, 2] == cond)]
    elif qtype == MOST_HITS:
        m = ev[ev[:, 2] == cond]
    else:
        raise ValueError(qtype)
    return np.unique(m[:, 0])


def _frame_at_distance(keys: np.ndarray, T: int, lo: int, hi: int,
                       rng: np.random.Generator) -> int | None:
    """直前のキーフレームまでの距離が [lo, hi) に入るフレームの中から
    t を一様に選ぶ。該当フレームがなければ None。"""
    if len(keys) == 0:
        return None
    ends = np.append(keys[1:] - 1, T - 1)          # 次のキーの直前のフレーム
    starts = np.maximum(keys + lo, 1)
    stops = np.minimum(np.minimum(keys + hi - 1, ends), T - 1)
    lengths = np.maximum(stops - starts + 1, 0)
    total = int(lengths.sum())
    if total == 0:
        return None
    i = int(rng.choice(len(keys), p=lengths / total))
    return int(rng.integers(starts[i], stops[i] + 1))


def sample_questions_stratified(s: Stream, n_questions: int, rng: np.random.Generator,
                                stream_id: int,
                                bin_edges: list[int] = BIN_EDGES,
                                bin_weights: list[float] = BIN_WEIGHTS,
                                ever_hit_no_frac: float = EVER_HIT_NO_FRAC) -> np.ndarray:
    """参照距離が `bin_weights` に従って `bin_edges` 上に散るように
    `n_questions` 個の質問をサンプルする。

    各質問について: qtype を一様に引き、(ball, cond) を一様に引き、
    目標の区間を引き、質問の根拠がその距離だけ前にあるフレーム t を選ぶ。
    ストリームにそのようなフレームがなければ、ひとつ下の区間に退避する。
    NOW_HALF は常に距離 0 で、フレームを一様に取る。EVER_HIT は確率
    `ever_hit_no_frac` で「いいえ」の質問（条件に合う最初のヒットより前の t）
    を引く。

    行は一意（(t, qtype, ball, cond) の重複はない）。
    """
    N, T = s.n_balls, s.T
    edges = list(bin_edges) + [T]
    w = np.asarray(bin_weights, dtype=float)
    w = w / w.sum()
    rows, seen = [], set()
    tries = 0
    while len(rows) < n_questions and tries < n_questions * 20:
        tries += 1
        qtype = int(rng.integers(4))
        ball = int(rng.integers(N)) if qtype in (NOW_HALF, EVER_HIT) else 0
        cond = int(rng.integers(5)) if qtype == LAST_HIT else int(rng.integers(4))

        if qtype == NOW_HALF:
            t = int(rng.integers(1, T))
        elif qtype == EVER_HIT and rng.random() < ever_hit_no_frac:
            keys = _key_frames(s, qtype, ball, cond)
            first = int(keys[0]) if len(keys) else T
            if first <= 1:
                continue
            t = int(rng.integers(1, first))
        else:
            keys = _key_frames(s, qtype, ball, cond)
            b = int(rng.choice(len(w), p=w))
            t = None
            while b >= 0 and t is None:
                t = _frame_at_distance(keys, T, edges[b], edges[b + 1], rng)
                b -= 1
            if t is None:
                continue

        key = (t, qtype, ball, cond)
        if key in seen:
            continue
        out = answer_question(s, t, qtype, ball, cond)
        if out is None:
            continue
        seen.add(key)
        ans, evidence_t, n_opt = out
        rows.append((stream_id, t, qtype, ball, cond, ans, evidence_t, n_opt))
    return np.array(rows, dtype=np.int64).reshape(-1, len(COLUMNS))


# ------------------------------------------------------------------- データセット

def generate_dataset(n_streams: int, T: int, n_balls: int,
                     frames_per_stream: int, per_frame: int, seed: int,
                     speed_range: tuple[float, float] = DEFAULT_SPEED_RANGE,
                     stratified: bool = True, resample: bool = False) -> dict:
    """データセットを作る: 全ストリームの位置と、1つの質問テーブル。

    各ストリームは frames_per_stream * per_frame 個の質問を持つ。
    `stratified=True` なら sample_questions_stratified から（参照距離が
    BIN_EDGES 上に散る）、そうでなければ `frames_per_stream` 個の一様な
    フレームでの sample_questions から得る。ストリームの seed は seed*10^6 + i
    とし、基底 seed の異なる train / val / test 分割が重ならないようにする。
    """
    rng = np.random.default_rng(seed)
    pos = np.empty((n_streams, T, n_balls, 2), dtype=np.float32)
    tables = []
    for i in range(n_streams):
        s = generate_stream(n_balls=n_balls, T=T, seed=seed * 1_000_000 + i, speed_range=speed_range,
                            resample=resample)
        pos[i] = s.pos
        if stratified:
            tables.append(sample_questions_stratified(s, frames_per_stream * per_frame, rng, stream_id=i))
        else:
            frames = np.sort(rng.choice(np.arange(1, T), size=frames_per_stream, replace=False))
            tables.append(sample_questions(s, frames, per_frame, rng, stream_id=i))
    q = np.concatenate(tables, axis=0)
    return {"pos": pos, "questions": q, "columns": np.array(COLUMNS),
            "speed_range": np.array(speed_range, dtype=np.float64),
            "stratified": np.array(stratified), "resample": np.array(resample)}


def describe(ds: dict) -> str:
    q = ds["questions"]
    T = ds["pos"].shape[1]
    edges = BIN_EDGES + [T]
    lines = [f"streams={ds['pos'].shape[0]}  T={T}  balls={ds['pos'].shape[2]}  questions={len(q)}"
             f"  speed_range={ds.get('speed_range', '?')}  stratified={ds.get('stratified', '?')}"]
    head = "  " + " " * 32 + "ref-distance bins " + " ".join(f"[{a},{b})".rjust(9) for a, b in zip(edges[:-1], edges[1:]))
    lines.append(head)
    for k, name in enumerate(QTYPE_NAMES):
        sub = q[q[:, 2] == k]
        if len(sub) == 0:
            continue
        ans = sub[:, 5]
        frac = np.bincount(ans, minlength=int(sub[0, 7])) / len(sub)
        dist = sub[:, 1] - sub[:, 6]
        dist = dist[sub[:, 6] >= 0]
        h = np.histogram(dist, bins=edges)[0] if len(dist) else np.zeros(len(edges) - 1, int)
        extra = f"  evidence=-1: {int((sub[:, 6] < 0).sum())}" if k == EVER_HIT else ""
        lines.append(
            f"  {name:9s} n={len(sub):7d} answer dist={str(np.round(frac, 2).tolist()):18s} "
            + " ".join(f"{x:9d}" for x in h) + extra
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=200)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--balls", type=int, default=3)
    ap.add_argument("--frames", type=int, default=32, help="frames sampled per stream (uniform mode); frames*per-frame = questions per stream")
    ap.add_argument("--per-frame", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, nargs=2, default=list(DEFAULT_SPEED_RANGE), metavar=("LO", "HI"),
                    help="ball speed range per frame (fraction of the square)")
    ap.add_argument("--uniform", action="store_true", help="uniform frames instead of distance-stratified sampling")
    ap.add_argument("--resample", action="store_true",
                    help="redraw speed and direction at every wall hit, so the past cannot be read off the present (leak_check.py)")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    ds = generate_dataset(a.streams, a.T, a.balls, a.frames, a.per_frame, a.seed,
                          speed_range=tuple(a.speed), stratified=not a.uniform, resample=a.resample)
    print(describe(ds))
    if a.out:
        np.savez_compressed(a.out, **ds)
        print("saved", a.out)
