"""物理と答えが、独立した再計算と一致することを確かめる。

実行:  python -m pytest test_bouncing.py -q     (または単に: python test_bouncing.py)
"""
import numpy as np

from sim import generate_stream, LEFT, RIGHT, BOTTOM, TOP
from questions import (answer_question, generate_dataset, sample_questions_stratified,
                       _frame_at_distance, _key_frames, NOW_HALF, LAST_HIT,
                       EVER_HIT, MOST_HITS, ANY_WALL, BIN_EDGES)

RADIUS = 0.03


def test_positions_stay_inside():
    for seed in range(20):
        s = generate_stream(3, 400, seed, radius=RADIUS)
        assert s.pos.min() >= RADIUS - 1e-9 and s.pos.max() <= 1 - RADIUS + 1e-9


def test_events_match_independent_resimulation():
    """最初の2フレームから、別に書いたベクトル化積分器で再シミュレーションし、
    位置と出来事の集合が厳密に一致することを比べる。

    初速は pos[1] - pos[0] として復元するが、これはフレーム 1 でどの玉も
    壁にヒットしていない場合にしか正しくないので、そのような seed は飛ばす。
    """
    checked = 0
    for seed in range(40):
        s = generate_stream(3, 400, seed, radius=RADIUS)
        if (s.events[:, 0] == 1).any():
            continue
        lo, hi = RADIUS, 1 - RADIUS
        p = s.pos[0].copy()
        v = s.pos[1] - s.pos[0]
        pos = [p.copy()]
        events = set()
        for t in range(1, s.T):
            p = p + v
            under, over = p < lo, p > hi
            p = np.where(under, 2 * lo - p, p)
            p = np.where(over, 2 * hi - p, p)
            v = np.where(under | over, -v, v)
            for b, axis in zip(*np.nonzero(under)):
                events.add((t, int(b), [LEFT, BOTTOM][axis]))
            for b, axis in zip(*np.nonzero(over)):
                events.add((t, int(b), [RIGHT, TOP][axis]))
            pos.append(p.copy())
        assert np.allclose(np.array(pos), s.pos, atol=1e-9), seed
        assert events == {tuple(r) for r in s.events.tolist()}, seed
        checked += 1
    assert checked >= 20


def test_answers_against_bruteforce():
    rng = np.random.default_rng(0)
    for seed in range(10):
        s = generate_stream(3, 300, seed)
        for _ in range(200):
            t = int(rng.integers(1, s.T))
            # NOW_HALF
            b, c = int(rng.integers(3)), int(rng.integers(4))
            ans, ev_t, n = answer_question(s, t, NOW_HALF, b, c)
            x, y = s.pos[t, b]
            truth = [x < .5, x >= .5, y < .5, y >= .5][c]
            assert ans == int(truth) and ev_t == t and n == 2
            # EVER_HIT
            hits = [e for e in s.events.tolist() if e[0] <= t and e[1] == b and e[2] == c]
            ans, ev_t, n = answer_question(s, t, EVER_HIT, b, c)
            assert ans == int(bool(hits))
            assert ev_t == (hits[-1][0] if hits else -1)
            # LAST_HIT、任意の壁
            past = [e for e in s.events.tolist() if e[0] <= t]
            out = answer_question(s, t, LAST_HIT, 0, ANY_WALL)
            if past:
                last_t = past[-1][0]
                same = {e[1] for e in past if e[0] == last_t}
                if len(same) == 1:
                    assert out[0] == past[-1][1] and out[1] == last_t
                else:
                    assert out is None
            else:
                assert out is None
            # MOST_HITS
            m = [e for e in past if e[2] == c]
            out = answer_question(s, t, MOST_HITS, 0, c)
            if m:
                counts = np.bincount([e[1] for e in m], minlength=3)
                if (counts == counts.max()).sum() == 1:
                    assert out[0] == int(counts.argmax()) and out[1] == m[-1][0]
                else:
                    assert out is None
            else:
                assert out is None


def test_dataset_shapes_and_distances():
    for stratified in (False, True):
        ds = generate_dataset(n_streams=5, T=200, n_balls=3, frames_per_stream=10, per_frame=4,
                              seed=7, stratified=stratified)
        q = ds["questions"]
        assert ds["pos"].shape == (5, 200, 3, 2)
        assert q.shape[1] == 8
        assert (q[:, 1] >= 1).all() and (q[:, 1] < 200).all()
        defined = q[:, 6] >= 0
        assert ((q[defined, 1] - q[defined, 6]) >= 0).all()      # 根拠が未来にあることはない
        assert (q[:, 5] < q[:, 7]).all()                           # 答えの添字 < n_options


def test_speed_range_is_respected():
    for seed in range(5):
        s = generate_stream(3, 100, seed, speed_range=(0.003, 0.03))
        v = s.pos[1] - s.pos[0]
        if (s.events[:, 0] == 1).any():
            continue
        speed = np.linalg.norm(v, axis=1)
        assert (speed >= 0.003 - 1e-9).all() and (speed <= 0.03 + 1e-9).all()


def test_frame_at_distance_lands_in_bin():
    rng = np.random.default_rng(1)
    T = 300
    for _ in range(300):
        keys = np.unique(rng.integers(1, T, size=int(rng.integers(1, 12))))
        lo = int(rng.choice([0, 1, 50, 100, 200]))
        hi = int(lo + rng.integers(1, 150))
        t = _frame_at_distance(keys, T, lo, hi, rng)
        # 総当たり: 直前のキーまでの距離が [lo, hi) に入るフレームの集合
        ok = set()
        for f in range(1, T):
            prev = keys[keys <= f]
            if len(prev) and lo <= f - prev[-1] < hi:
                ok.add(f)
        if t is None:
            assert not ok, (keys, lo, hi)
        else:
            assert t in ok, (keys, lo, hi, t)


def test_stratified_rows_are_correct_unique_and_reach_far():
    rng = np.random.default_rng(3)
    far = {LAST_HIT: 0, EVER_HIT: 0, MOST_HITS: 0}
    n_streams = 60
    for i in range(n_streams):
        s = generate_stream(3, 500, 500 + i, speed_range=(0.003, 0.03))
        q = sample_questions_stratified(s, 256, rng, stream_id=i)
        assert len(q) == 256, len(q)
        assert len({tuple(r[1:5]) for r in q.tolist()}) == len(q)       # 行が一意
        for _, t, qt, b, c, ans, ev_t, n in q.tolist():
            assert answer_question(s, t, qt, b, c) == (ans, ev_t, n)     # 検証済みの同じ経路
            if ev_t >= 0 and t - ev_t >= BIN_EDGES[-1]:
                far[qt] = far.get(qt, 0) + 1
    # このサンプラーの眼目: 記憶を問う全ての qtype に遠い質問が存在する
    assert far[LAST_HIT] >= 10 and far[MOST_HITS] >= 10 and far[EVER_HIT] >= 50, far


def test_resample_keeps_balls_inside_and_speeds_in_range():
    from sim import generate_stream
    lo, hi = 0.004, 0.02
    s = generate_stream(n_balls=3, T=400, seed=5, speed_range=(lo, hi), resample=True)
    assert (s.pos >= RADIUS - 1e-9).all() and (s.pos <= 1 - RADIUS + 1e-9).all()
    assert len(s.events) > 0
    # between two consecutive events of a ball, its velocity is constant and its speed in range
    for b in range(3):
        ev_t = sorted(set(int(t) for t, bb, _ in s.events if bb == b))
        bounds = [1] + ev_t + [s.T - 1]
        for a, z in zip(bounds[:-1], bounds[1:]):
            if z - a < 3:
                continue
            v = s.pos[a + 1:z, b] - s.pos[a:z - 1, b]
            assert np.allclose(v, v[0], atol=1e-9), (b, a, z)
            assert lo - 1e-9 <= np.linalg.norm(v[0]) <= hi + 1e-9
    # resample=False still reproduces the original streams exactly
    a0 = generate_stream(n_balls=3, T=100, seed=7)
    a1 = generate_stream(n_balls=3, T=100, seed=7, resample=False)
    assert np.array_equal(a0.pos, a1.pos) and np.array_equal(a0.events, a1.events)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
