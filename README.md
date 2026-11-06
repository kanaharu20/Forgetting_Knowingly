# Forgetting, Knowingly / 忘れたと分かっているか

*Do fixed-state models know what they have forgotten?*

過去を固定サイズの状態に圧縮して持つモデルは、圧縮で失った情報による誤りを、答えの確率に正直に反映するか、を調べる研究。作業中のリポジトリで、結果のまとめは完成時に追加する。

## 設定

2D の正方形の中を跳ねる玉 3 個のストリーム（長さ 500 フレーム）を見せ、各フレームで過去についての質問に答えさせる。質問は整数の組 `(qtype, ball, cond)` で、自然言語にはしない。

| qtype | 質問 |
|---|---|
| NOW_HALF | 玉 b は今、半面 c にいるか |
| LAST_HIT | 壁 c に最後に当たった玉はどれか |
| EVER_HIT | 玉 b は壁 c に当たったことがあるか |
| MOST_HITS | 壁 c に最も多く当たった玉はどれか |

質問が参照する出来事までのフレーム数（参照距離）ごとに、正解率と確信を比べる。

## モデル

入力の埋め込みと読み出しは共通で、過去の持ち方だけが違う。

| モデル | 過去の持ち方 |
|---|---|
| 因果 Transformer | 全フレームを保持する（対照） |
| GRU | 固定サイズの状態に圧縮する |
| 窓つき Transformer | 直近 W フレームだけを見る |

## ファイル

コードはすべて `bouncing/` にある。

- `sim.py`, `questions.py`: ストリームの生成と質問の付与
- `data.py`, `models/`, `train.py`, `eval.py`, `metrics.py`: 学習と評価
- `probe.py`, `switch.py`, `leak_check.py`, `plot_*.py`: 分析と図
- `test_bouncing.py`, `test_models.py`: テスト

## 動かし方

Python 3.14 と [uv](https://docs.astral.sh/uv/) を使う。データセットと学習済みの重みは含めていないので、生成してから学習する。

```
cd bouncing
uv run python questions.py --streams 20000 --seed 1 --resample --out data_resample/train.npz
uv run python questions.py --streams 2000  --seed 2 --resample --out data_resample/val.npz
uv run python questions.py --streams 2000  --seed 3 --resample --out data_resample/test.npz
uv run python train.py --model gru --hidden 16 --velocity --epochs 12 --lr 1e-3 --warmup 500 \
    --data data_resample --out runs_resample/gru_h16
uv run python eval.py runs_resample/gru_h16/best.pt --split test --data data_resample
```

`--model` は `transformer` / `gru` / `window` から選ぶ。
