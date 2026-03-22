from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import pandas as pd
from backtesting import Backtest

from strategies import ClassificationStrategy, RegressionStrategy, WalkForwardAnchored, WalkForwardUnanchored


def to_backtesting_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    bt = df.copy()
    bt = bt.set_index("Date")
    # backtesting.py needs OHLCV columns present; extra features can remain in the frame.
    return bt


def run_backtest(prepared, strategy_name: str, cash: float = 10000, commission: float = 0.002, optimize: bool = False):
    strategy_map = {
        "classification": ClassificationStrategy,
        "regression": RegressionStrategy,
        "walk_forward_anchored": WalkForwardAnchored,
        "walk_forward_unanchored": WalkForwardUnanchored,
    }
    strategy_cls = strategy_map[strategy_name]

    bt_df = to_backtesting_frame(prepared.df, prepared.feature_columns)
    strategy_cls._full_data = prepared
    bt = Backtest(bt_df, strategy_cls, cash=cash, commission=commission, exclusive_orders=True)

    if optimize and strategy_name == "classification":
        stats, heatmap = bt.optimize(
            prob_buy=[0.52, 0.55, 0.58, 0.6],
            prob_sell=[0.4, 0.42, 0.45, 0.48],
            maximize="Return [%]",
            return_heatmap=True,
        )
        return bt, stats, heatmap

    if optimize and strategy_name in {"regression", "walk_forward_anchored", "walk_forward_unanchored"}:
        stats, heatmap = bt.optimize(
            limit_buy=range(0, 6),
            limit_sell=range(-6, 0),
            maximize="Return [%]",
            method="skopt",
            max_tries=60,
            random_state=42,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    stats = bt.run()
    return bt, stats, None
