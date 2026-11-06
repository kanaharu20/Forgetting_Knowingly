"""data/train.npz でモデルを 1 つ学習し、data/val.npz で検証し、最良のものを残す。

    python train.py --model transformer --epochs 4 --out runs/transformer

損失: (マスク済みの) 正解ロジットに対する交差エントロピーを、バッチ内の
実在する質問で平均したもの。最適化: AdamW、ウォームアップの後にコサイン減衰。
各エポックの終わりに検証分割を、質問の種類 × 参照距離の区間ごとの正解率・
平均確信・ECE で採点する (metrics.py)。検証正解率が最良のチェックポイントを
<out>/best.pt に、指標の全表を <out>/val_metrics.json に残す。

段階1の合格条件 (CLAUDE.md): transformer、T=500、4 種の質問すべてで
正解率 >= 0.98。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import make_loader, StreamDataset, STREAM, QTYPE, ANSWER, EVIDENCE, T_COL
from models import build
from metrics import summarize, format_table, per_qtype_accuracy


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_mean(x: torch.Tensor, qmask: torch.Tensor) -> torch.Tensor:
    m = qmask.float()
    return (x * m).sum() / m.sum()


def loss_fn(logits: torch.Tensor, q: torch.Tensor, qmask: torch.Tensor, kind: str = "ce",
            teacher: torch.Tensor | None = None, distill_weight: float = 1.0,
            calib_weight: float = 0.0) -> torch.Tensor:
    """段階1/2 の既定は素の交差エントロピー。段階3 (H3) では同じロジットの上に
    1 行ずつの変更として、次の 3 つの代替案を加える:

      kind="brier"     確率ベクトルと one-hot の正解との二乗誤差
                       (適切なスコア関数で、確信を持った誤りを対数ではなく
                       二乗で罰する)
      teacher=...      蒸留: 確率ベクトル全体に対する KL(control || model) を
                       distill_weight で重み付けして加える。対照の確率は
                       eval.py --split train で作る
      calib_weight>0   バッチ内で (qtype, 参照距離の区間) ごとに計算する校正項:
                       (平均確信 − 正解率)^2 を群の大きさで重み付けしたもの。
                       距離は学習時にだけ分かるもので、モデルは決して見ない。
    """
    ans = q[..., ANSWER]
    if kind == "ce":
        base = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ans.reshape(-1), reduction="none").reshape(ans.shape)
    elif kind == "brier":
        p = torch.softmax(logits, dim=-1)
        onehot = F.one_hot(ans, logits.shape[-1]).float()
        base = ((p - onehot) ** 2).sum(-1)
    else:
        raise ValueError(kind)
    loss = masked_mean(base, qmask)
    if teacher is not None:
        logp = torch.log_softmax(logits, dim=-1)
        kl = (teacher * (torch.log(teacher.clamp_min(1e-8)) - logp)).sum(-1)
        loss = loss + distill_weight * masked_mean(kl, qmask)
    if calib_weight > 0:
        loss = loss + calib_weight * calibration_term(logits, q, qmask)
    return loss


def calibration_term(logits: torch.Tensor, q: torch.Tensor, qmask: torch.Tensor) -> torch.Tensor:
    """(qtype, 距離区間) の群ごとの (平均確信 − 正解率)^2 を、群の大きさで重み付けして平均したもの。
    勾配を持つのは確信だけで、群の正解率に向けて引き寄せられる。"""
    from questions import BIN_EDGES
    p = torch.softmax(logits, dim=-1)
    conf, pred = p.max(-1)
    correct = (pred == q[..., ANSWER]).float()
    dist = torch.where(q[..., EVIDENCE] >= 0, q[..., T_COL] - q[..., EVIDENCE], torch.full_like(q[..., T_COL], -1))
    edges = torch.tensor(list(BIN_EDGES[1:]), device=logits.device)
    dbin = torch.where(dist < 0, torch.full_like(dist, -1), torch.bucketize(dist, edges, right=True))
    group = q[..., QTYPE] * (len(BIN_EDGES) + 1) + (dbin + 1)
    total, n_all = logits.new_zeros(()), qmask.float().sum()
    for g in torch.unique(group[qmask]):
        m = (group == g) & qmask
        n = m.float().sum()
        total = total + n * (conf[m].mean() - correct[m].mean()) ** 2
    return total / n_all


def teacher_table(path: str, ds: StreamDataset) -> torch.Tensor:
    """eval.py --split train の npz から作る、ds.q と並びを揃えた (S, Qmax, options) の
    対照の確率。npz に無いストリームは NaN にしておき、使えば黙って 0 になるのではなく
    エラーになるようにする。"""
    d = np.load(path)
    qs, probs = d["questions"], d["probs"]
    table = torch.full((ds.n_streams, ds.q_max, probs.shape[1]), float("nan"))
    sid = qs[:, 0]
    starts = np.searchsorted(sid, np.arange(ds.n_streams + 1))
    for s in range(ds.n_streams):
        lo, hi = starts[s], starts[s + 1]
        if hi > lo:
            assert (torch.from_numpy(qs[lo:hi]) == ds.q[s, :hi - lo]).all(), f"teacher rows differ from data for stream {s}"
            table[s, :hi - lo] = torch.from_numpy(probs[lo:hi])
    return table


@torch.no_grad()
def evaluate(model, loader, device, T: int):
    """分割全体にモデルを走らせ、(指標の行, 質問ごとの配列) を返す。"""
    model.eval()
    out = {k: [] for k in ("q", "probs")}
    for pos, q, qmask in loader:
        pos, q = pos.to(device), q.to(device)
        probs = torch.softmax(model(pos, q), dim=-1)
        out["q"].append(q[qmask].cpu().numpy())
        out["probs"].append(probs[qmask.to(device)].cpu().numpy())
    q = np.concatenate(out["q"]); probs = np.concatenate(out["probs"])
    pred = probs.argmax(-1)
    correct = (pred == q[:, ANSWER]).astype(np.float64)
    conf = probs.max(-1)
    dist = np.where(q[:, EVIDENCE] >= 0, q[:, T_COL] - q[:, EVIDENCE], -1)
    rows = summarize(q[:, QTYPE], dist, correct, conf, T)
    model.train()
    return rows, dict(questions=q, probs=probs, correct=correct, conf=conf, dist=dist)


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer", choices=["transformer", "gru", "window"])
    ap.add_argument("--window", type=int, default=32, help="W for --model window")
    ap.add_argument("--hidden", type=int, default=256, help="state size for --model gru")
    ap.add_argument("--layers", type=int, default=0, help="--model transformer: n_layers (0 = model default 4)")
    ap.add_argument("--d", type=int, default=0, help="--model transformer: width d (0 = model default 128)")
    ap.add_argument("--velocity", action="store_true", help="also feed frame-to-frame differences")
    ap.add_argument("--no-current-frame", action="store_true", help="readout sees only the state, not frame t")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for smoke tests)")
    ap.add_argument("--train-streams", type=int, default=0, help="use only the first n streams (smoke tests)")
    ap.add_argument("--questions-per-step", type=int, default=0,
                    help="train on a random subset of k questions per stream each step (0 = all); "
                         "the windowed model's cost is per question, so use ~32 there")
    ap.add_argument("--loss", default="ce", choices=["ce", "brier"], help="stage 3: base loss")
    ap.add_argument("--calib-weight", type=float, default=0.0,
                    help="stage 3: weight of the per-(qtype, distance bin) calibration term")
    ap.add_argument("--distill", default="",
                    help="stage 3: <run>/train_probs.npz of the control (eval.py --split train); adds KL(control || model)")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    out = Path(a.out or f"runs/{a.model}")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(*parts):
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    train_ds = StreamDataset(Path(a.data) / "train.npz")
    if a.train_streams:
        train_ds.pos, train_ds.q, train_ds.qmask = (x[:a.train_streams] for x in (train_ds.pos, train_ds.q, train_ds.qmask))
        train_ds.n_streams = a.train_streams
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                                               collate_fn=__import__("data").collate,
                                               generator=torch.Generator().manual_seed(a.seed))
    val_loader = make_loader(Path(a.data) / "val.npz", a.batch, shuffle=False)
    T, N = train_ds.T, train_ds.n_balls

    kw = dict(velocity=a.velocity, current_frame=not a.no_current_frame)
    if a.model == "window":
        kw["window"] = a.window
    if a.model == "gru":
        kw["hidden"] = a.hidden
    if a.model == "transformer":
        if a.layers: kw["n_layers"] = a.layers
        if a.d: kw["d"] = a.d
    model = build(a.model, n_balls=N, T=T, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    total = a.epochs * len(train_loader) if not a.max_steps else a.max_steps
    teacher = teacher_table(a.distill, train_ds) if a.distill else None
    say(f"model={a.model} {kw} params={model.n_params():,} device={device} streams={len(train_ds)} "
        f"T={T} N={N} steps={total} batch={a.batch} lr={a.lr} questions/step={a.questions_per_step or 'all'} "
        f"loss={a.loss} calib_weight={a.calib_weight} distill={a.distill or '-'} x{a.distill_weight}")

    step, best = 0, -1.0
    t0 = time.time()
    for epoch in range(a.epochs):
        run_loss, run_n = 0.0, 0
        for pos, q, qmask in train_loader:
            qi = torch.arange(q.shape[1]).expand(q.shape[0], -1)       # ストリーム内での質問の番号
            if a.questions_per_step and a.questions_per_step < q.shape[1]:
                pick = torch.rand(q.shape[:2]).argsort(dim=1)[:, :a.questions_per_step]
                q, qmask, qi = (q.gather(1, pick[..., None].expand(-1, -1, q.shape[-1])),
                                qmask.gather(1, pick), qi.gather(1, pick))
            t_probs = None
            if teacher is not None:
                t_probs = teacher[q[..., STREAM], qi]
                assert not torch.isnan(t_probs[qmask]).any(), "teacher probabilities missing for some training questions"
                t_probs = t_probs.to(device)
            pos, q, qmask = pos.to(device), q.to(device), qmask.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr, a.warmup)
            loss = loss_fn(model(pos, q), q, qmask, kind=a.loss, teacher=t_probs,
                           distill_weight=a.distill_weight, calib_weight=a.calib_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += loss.item(); run_n += 1
            if step % a.log_every == 0:
                say(f"epoch {epoch} step {step}/{total} loss {run_loss / run_n:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s")
                run_loss, run_n = 0.0, 0
            if a.max_steps and step >= a.max_steps:
                break

        rows, _ = evaluate(model, val_loader, device, T)
        accs = per_qtype_accuracy(rows)
        say(f"== epoch {epoch} validation ({time.time() - t0:.0f}s)")
        say(format_table(rows))
        overall = rows[-1]["acc"]
        if overall > best:
            best = overall
            torch.save(dict(model=a.model, kw=dict(n_balls=N, T=T, **kw), state=model.state_dict(),
                            epoch=epoch, step=step, val_acc=overall), out / "best.pt")
            json.dump(rows, open(out / "val_metrics.json", "w"), indent=1)
            say(f"saved best (val acc {overall:.4f})")
        say("per-qtype acc:", {k: round(v, 4) for k, v in accs.items()},
            "PASS" if all(v >= 0.98 for v in accs.values()) else "below 0.98")
        if a.max_steps and step >= a.max_steps:
            break
    say("done")


if __name__ == "__main__":
    main()
