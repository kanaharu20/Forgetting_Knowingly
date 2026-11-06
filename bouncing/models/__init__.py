"""同じインターフェース（common.py 参照）を持つ3つのモデル: 全文脈の因果
Transformer（対照）、GRU（なだらかな忘却）、窓つき Transformer（W フレーム
での崖）。`build(name, ...)` が名前で1つを選ぶ。"""
from .common import StreamModel, MAX_OPTIONS
from .transformer import CausalTransformer
from .gru import GRUModel
from .window import WindowTransformer


def build(name: str, n_balls: int, T: int, **kw) -> StreamModel:
    if name == "transformer":
        return CausalTransformer(n_balls=n_balls, max_len=T, **kw)
    if name == "gru":
        return GRUModel(n_balls=n_balls, **kw)
    if name == "window":
        return WindowTransformer(n_balls=n_balls, **kw)
    raise ValueError(f"unknown model {name!r}; choose transformer / gru / window")
