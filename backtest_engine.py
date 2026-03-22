from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest

from strategies import (
    GenericClassificationStrategy,
    GenericClassificationWalkForwardAnchored,
    GenericClassificationWalkForwardUnanchored,
    GenericRegressionStrategy,
    GenericRegressionWalkForwardAnchored,
    GenericRegressionWalkForwardUnanchored,
)


STRATEGY_MAP = {
    ("classification", "static"): GenericClassificationStrategy,
    ("classification", "walk_forward_anchored"): GenericClassificationWalkForwardAnchored,
    ("classification", "walk_forward_unanchored"): GenericClassificationWalkForwardUnanchored,
    ("regression", "static"): GenericRegressionStrategy,
    ("regression", "walk_forward_anchored"): GenericRegressionWalkForwardAnchored,
    ("regression", "walk_forward_unanchored"): GenericRegressionWalkForwardUnanchored,
}


def to_backtesting_frame(df: pd.DataFrame) -> pd.DataFrame:
    bt = df.copy()
    bt = bt.set_index("Date")
    return bt


def _bind_prepared(strategy_cls, prepared):
    class BoundStrategy(strategy_cls):
        _full_data = prepared

    BoundStrategy.__name__ = strategy_cls.__name__
    return BoundStrategy


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def objective_score_from_stats(stats, objective: str) -> float:
    if objective == "Return [%]":
        return _safe_float(stats.get("Return [%]"), -np.inf)

    if objective == "Sharpe Ratio":
        return _safe_float(stats.get("Sharpe Ratio"), -np.inf)

    if objective == "Balanced":
        sharpe = _safe_float(stats.get("Sharpe Ratio"), -9999.0)
        ret = _safe_float(stats.get("Return [%]"), -9999.0)
        drawdown = abs(_safe_float(stats.get("Max. Drawdown [%]"), 9999.0))
        win_rate = _safe_float(stats.get("Win Rate [%]"), 0.0)
        trades = _safe_float(stats.get("# Trades"), 0.0)
        trade_penalty = 0.0 if trades >= 5 else 2.0
        return sharpe * 5.0 + ret * 0.08 + win_rate * 0.02 - drawdown * 0.05 - trade_penalty

    raise ValueError(f"Unknown objective: {objective}")


def run_backtest(
    prepared,
    cash: float = 10000,
    commission: float = 0.002,
    optimize: bool = True,
    objective: str = "Balanced",
):
    strategy_key = (prepared.task, prepared.training_style)
    if strategy_key not in STRATEGY_MAP:
        raise ValueError(f"Unsupported strategy key: {strategy_key}")

    raw_strategy_cls = STRATEGY_MAP[strategy_key]
    strategy_cls = _bind_prepared(raw_strategy_cls, prepared)

    bt_df = to_backtesting_frame(prepared.df)
    bt = Backtest(
        bt_df,
        strategy_cls,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )

    maximize_metric = "Sharpe Ratio" if objective in {"Sharpe Ratio", "Balanced"} else "Return [%]"

    if optimize and prepared.task == "classification":
        stats, heatmap = bt.optimize(
            prob_buy=[0.52, 0.55, 0.58, 0.60, 0.62, 0.65],
            prob_sell=[0.35, 0.38, 0.40, 0.42, 0.45, 0.48],
            maximize=maximize_metric,
            method="sambo",
            max_tries=35,
            constraint=lambda p: p.prob_buy > p.prob_sell,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    if optimize and prepared.task == "regression":
        stats, heatmap = bt.optimize(
            limit_buy=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            limit_sell=[-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0],
            maximize=maximize_metric,
            method="sambo",
            max_tries=35,
            constraint=lambda p: p.limit_buy > p.limit_sell,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    stats = bt.run()
    return bt, stats, None
