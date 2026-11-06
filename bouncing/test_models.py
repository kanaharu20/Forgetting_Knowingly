"""学習なしで確認できるモデル配線のテスト。CPU のみ。

実行:  python test_models.py
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from data import StreamDataset, collate, T_COL, ANSWER, NOPT
from models import build, MAX_OPTIONS
from models.common import QuestionEmbedding
from questions import generate_dataset
from metrics import ece

torch.manual_seed(0)


def small_batch(T=64, n_streams=4):
    ds = generate_dataset(n_streams=n_streams, T=T, n_balls=3, frames_per_stream=4, per_frame=4, seed=11)
    path = Path(tempfile.mkdtemp()) / "small.npz"
    np.savez(path, **ds)
    d = StreamDataset(path)
    return d, collate([d[i] for i in range(len(d))])


def test_shapes_and_option_mask():
    d, (pos, q, m) = small_batch()
    for name in ("transformer", "gru", "window"):
        model = build(name, n_balls=3, T=d.T, **({"window": 8} if name == "window" else {}))
        logits = model(pos, q)
        assert logits.shape == (pos.shape[0], d.q_max, MAX_OPTIONS), (name, logits.shape)
        # n_options を超える選択肢は（実質）-inf にマスクされる
        invalid = torch.arange(MAX_OPTIONS)[None, None] >= q[..., NOPT][..., None]
        assert (logits[invalid] < -1e8).all(), name
        assert (logits[~invalid] > -1e8).all(), name
        print("ok shapes", name)


def test_transformer_is_causal():
    """t より後のフレームを変えても t での答えは変わってはならない。"""
    d, (pos, q, m) = small_batch()
    model = build("transformer", n_balls=3, T=d.T).eval()
    with torch.no_grad():
        base = model(pos, q)
        t = q[..., T_COL]
        pos2 = pos.clone()
        for b in range(pos.shape[0]):
            tmin = int(t[b].min())
            pos2[b, tmin + 1:] = torch.rand_like(pos2[b, tmin + 1:])
        alt = model(pos2, q)
        # 各ストリームの最初のフレームの質問は摂動の影響を受けない
        first = t == t.min(dim=1, keepdim=True).values
        assert torch.allclose(base[first], alt[first], atol=1e-5)
        # 後の質問のいくつかは変わる（モデルがストリームを読んでいることの確認）
        assert not torch.allclose(base[~first], alt[~first], atol=1e-3)


def test_gru_is_causal():
    d, (pos, q, m) = small_batch()
    model = build("gru", n_balls=3, T=d.T).eval()
    with torch.no_grad():
        base = model(pos, q)
        t = q[..., T_COL]
        pos2 = pos.clone()
        for b in range(pos.shape[0]):
            pos2[b, int(t[b].min()) + 1:] = 0.5
        alt = model(pos2, q)
        first = t == t.min(dim=1, keepdim=True).values
        assert torch.allclose(base[first], alt[first], atol=1e-5)


def test_window_sees_only_last_W_frames():
    """窓つきモデルの t での答えは pos[t-W+1 .. t] だけに依存する。"""
    d, (pos, q, m) = small_batch()
    W = 8
    model = build("window", n_balls=3, T=d.T, window=W).eval()
    with torch.no_grad():
        base = model(pos, q)
        t = q[..., T_COL]
        pos2 = pos.clone()
        # 各ストリームの*最初の*質問について、t-W+1 より厳密に前をすべて摂動する
        for b in range(pos.shape[0]):
            cut = int(t[b].min()) - W + 1
            if cut > 0:
                pos2[b, :cut] = torch.rand_like(pos2[b, :cut])
        alt = model(pos2, q)
        first = t == t.min(dim=1, keepdim=True).values
        assert torch.allclose(base[first], alt[first], atol=1e-5)
        # 窓の中を摂動すれば答えは変わる
        pos3 = pos.clone()
        for b in range(pos.shape[0]):
            pos3[b, int(t[b].min())] = 0.5
        assert not torch.allclose(base[first], model(pos3, q)[first], atol=1e-3)


def test_question_embedding_is_sum_of_tables():
    qe = QuestionEmbedding(n_balls=3, d=16)
    q = torch.tensor([[[0, 5, 1, 2, 3, 0, 0, 2]]])      # qtype 1、ball 2、cond 3
    out = qe(q)[0, 0]
    expect = qe.qtype.weight[1] + qe.ball.weight[2] + qe.cond.weight[3]
    assert torch.allclose(out, expect)


def test_ece_extremes():
    conf = np.array([0.9, 0.9, 0.9, 0.9]); correct = np.array([1., 1., 1., 1.])
    assert abs(ece(conf, correct) - 0.1) < 1e-9          # 確信 90%、正解 100%
    correct = np.array([1., 1., 0., 0.])
    assert abs(ece(conf, correct) - 0.4) < 1e-9          # 確信 90%、正解 50%


def test_one_training_step_reduces_loss():
    from train import loss_fn
    d, (pos, q, m) = small_batch()
    model = build("transformer", n_balls=3, T=d.T)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    l0 = loss_fn(model(pos, q), q, m)
    for _ in range(20):
        loss = loss_fn(model(pos, q), q, m)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < l0.item()


def test_stage3_losses():
    from train import loss_fn, calibration_term
    d, (pos, q, m) = small_batch()
    model = build("transformer", n_balls=3, T=d.T)
    logits = model(pos, q)
    ce, brier = loss_fn(logits, q, m), loss_fn(logits, q, m, kind="brier")
    assert ce.item() > 0 and 0 <= brier.item() <= 2
    # モデルを自分自身に蒸留しても何も加わらない；別の教師なら正の KL が加わる
    p = torch.softmax(logits, -1).detach()
    assert torch.allclose(loss_fn(logits, q, m, teacher=p), ce, atol=1e-5)
    other = torch.softmax(logits.detach() * 3, -1)
    assert loss_fn(logits, q, m, teacher=other).item() > ce.item()
    # 校正項が 0 になるのは、すべての (qtype, bin) グループで平均 conf == 正解率のとき、かつそのときに限る
    c = calibration_term(logits, q, m)
    assert c.item() >= 0 and c.requires_grad
    assert loss_fn(logits, q, m, calib_weight=1.0).item() >= ce.item()


def test_teacher_table_matches_dataset_order():
    from train import teacher_table, evaluate
    import json
    d, (pos, q, m) = small_batch()
    model = build("gru", n_balls=3, T=d.T, hidden=8)
    loader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=False, collate_fn=collate)
    _, arrays = evaluate(model, loader, torch.device("cpu"), d.T)
    path = Path(tempfile.mkdtemp()) / "train_probs.npz"
    np.savez(path, **arrays)
    table = teacher_table(str(path), d)
    assert table.shape == (len(d), d.q_max, MAX_OPTIONS)
    assert not torch.isnan(table[m]).any()
    # 表の行 (s, i) はストリーム s の質問 i の確率である
    probs = torch.softmax(model(pos, q), -1).detach()
    assert torch.allclose(table[m], probs[m], atol=1e-5)


def test_states_feed_the_readout():
    """states() must be exactly the h_t that forward() reads from (probe.py relies on it)."""
    d, (pos, q, m) = small_batch()
    for name, kw in (("transformer", {}), ("gru", {"hidden": 8}), ("window", {"window": 8})):
        model = build(name, n_balls=3, T=d.T, **kw).eval()
        h = model.states(pos, q)
        assert h.shape == (pos.shape[0], d.q_max, model.h_dim), (name, h.shape)
        logits = model.read(h, model.embed(pos), q)
        assert torch.allclose(logits, model(pos, q), atol=1e-5), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
