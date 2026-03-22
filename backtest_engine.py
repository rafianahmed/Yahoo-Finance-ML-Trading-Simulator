from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest

from strategies import (
    ClassificationStrategy,
    RegressionStrategy,
    WalkForwardAnchored,
    WalkForwardUnanchored,
)


def to_backtesting_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    bt = df.copy()
    bt = bt.set_index("Date")
    return bt


def _bind_prepared(strategy_cls, prepared):
    class BoundStrategy(strategy_cls):
        _full_data = prepared
    BoundStrategy.__name__ = strategy_cls.__name__
    return BoundStrategy


def _safe_metric(stats: Any, metric_name: str) -> float:
    value = stats.get(metric_name, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan


def run_backtest(
    prepared,
    strategy_name: str,
    cash: float = 10000,
    commission: float = 0.002,
    optimize: bool = False,
    objective: str = "Sharpe Ratio",
):
    strategy_map = {
        "classification": ClassificationStrategy,
        "regression": RegressionStrategy,
        "walk_forward_anchored": WalkForwardAnchored,
        "walk_forward_unanchored": WalkForwardUnanchored,
    }

    raw_strategy_cls = strategy_map[strategy_name]
    strategy_cls = _bind_prepared(raw_strategy_cls, prepared)

    bt_df = to_backtesting_frame(prepared.df, prepared.feature_columns)
    bt = Backtest(
        bt_df,
        strategy_cls,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )

    if optimize and strategy_name == "classification":
        stats, heatmap = bt.optimize(
            prob_buy=[0.52, 0.55, 0.58, 0.60, 0.62],
            prob_sell=[0.38, 0.40, 0.42, 0.45, 0.48],
            maximize=objective,
            method="sambo",
            max_tries=50,
            constraint=lambda p: p.prob_buy > p.prob_sell,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    if optimize and strategy_name in {"regression", "walk_forward_anchored", "walk_forward_unanchored"}:
        stats, heatmap = bt.optimize(
            limit_buy=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            limit_sell=[-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0],
            maximize=objective,
            method="sambo",
            max_tries=50,
            constraint=lambda p: p.limit_buy > p.limit_sell,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    stats = bt.run()
    return bt, stats, None


def auto_optimize_model_and_strategy(
    prepare_data_fn,
    ticker: str,
    start: str,
    end: str | None,
    interval: str,
    strategy_name: str,
    cash: float = 10000,
    commission: float = 0.002,
    objective: str = "Sharpe Ratio",
):
    """
    Joint search over:
    - n_train
    - reg_depth
    - clf_depth
    - thresholds via run_backtest(..., optimize=True)

    Returns:
    {
        "best_prepared": ...,
        "best_bt": ...,
        "best_stats": ...,
        "best_heatmap": ...,
        "best_params": {...},
        "search_results": DataFrame
    }
    """

    n_train_grid = [300, 400, 500, 600, 800]
    reg_depth_grid = [4, 6, 8, 10, 12, 15]
    clf_depth_grid = [3, 5, 7, 8, 10, 12]

    results: list[dict[str, Any]] = []

    if strategy_name == "classification":
        grid = product(n_train_grid, [8], clf_depth_grid)
    elif strategy_name in {"regression", "walk_forward_anchored", "walk_forward_unanchored"}:
        grid = product(n_train_grid, reg_depth_grid, [8])
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    best_score = -np.inf
    best_bundle = None

    for n_train, reg_depth, clf_depth in grid:
        try:
            prepared = prepare_data_fn(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                n_train=n_train,
                reg_depth=reg_depth,
                clf_depth=clf_depth,
            )

            bt, stats, heatmap = run_backtest(
                prepared=prepared,
                strategy_name=strategy_name,
                cash=cash,
                commission=commission,
                optimize=True,
                objective=objective,
            )

            score = _safe_metric(stats, objective)
            total_return = _safe_metric(stats, "Return [%]")
            win_rate = _safe_metric(stats, "Win Rate [%]")
            trades = _safe_metric(stats, "# Trades")

            row = {
                "strategy": strategy_name,
                "n_train": n_train,
                "reg_depth": reg_depth,
                "clf_depth": clf_depth,
                "objective": objective,
                "objective_value": score,
                "return_pct": total_return,
                "win_rate_pct": win_rate,
                "n_trades": trades,
            }

            if strategy_name == "classification":
                row["prob_buy"] = stats.get("_strategy").prob_buy
                row["prob_sell"] = stats.get("_strategy").prob_sell
            else:
                row["limit_buy"] = stats.get("_strategy").limit_buy
                row["limit_sell"] = stats.get("_strategy").limit_sell

            results.append(row)

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_bundle = {
                    "best_prepared": prepared,
                    "best_bt": bt,
                    "best_stats": stats,
                    "best_heatmap": heatmap,
                    "best_params": row,
                }

        except Exception as e:
            results.append(
                {
                    "strategy": strategy_name,
                    "n_train": n_train,
                    "reg_depth": reg_depth,
                    "clf_depth": clf_depth,
                    "objective": objective,
                    "objective_value": np.nan,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results).sort_values(
        by="objective_value",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if best_bundle is None:
        raise ValueError("Auto-optimization failed for all parameter combinations.")

    best_bundle["search_results"] = results_df
    return best_bundle
