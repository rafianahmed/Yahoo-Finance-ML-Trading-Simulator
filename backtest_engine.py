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


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _objective_score(stats, objective: str) -> float:
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

    maximize_metric = "Sharpe Ratio" if objective in {"Sharpe Ratio", "Balanced"} else "Return [%]"

    if optimize and strategy_name == "classification":
        stats, heatmap = bt.optimize(
            prob_buy=[0.52, 0.55, 0.58, 0.60, 0.62, 0.65],
            prob_sell=[0.35, 0.38, 0.40, 0.42, 0.45, 0.48],
            maximize=maximize_metric,
            method="sambo",
            max_tries=40,
            constraint=lambda p: p.prob_buy > p.prob_sell,
            return_heatmap=True,
        )
        return bt, stats, heatmap

    if optimize and strategy_name in {"regression", "walk_forward_anchored", "walk_forward_unanchored"}:
        stats, heatmap = bt.optimize(
            limit_buy=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            limit_sell=[-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0],
            maximize=maximize_metric,
            method="sambo",
            max_tries=40,
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
    n_train_grid = [300, 400, 500, 600, 800]
    reg_depth_grid = [4, 6, 8, 10, 12, 15]
    clf_depth_grid = [3, 5, 7, 8, 10, 12]

    results: list[dict[str, Any]] = []
    best_score = -np.inf
    best_bundle = None

    if strategy_name == "classification":
        grid = product(n_train_grid, [8], clf_depth_grid)
    else:
        grid = product(n_train_grid, reg_depth_grid, [8])

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

            score = _objective_score(stats, objective)

            row = {
                "strategy": strategy_name,
                "n_train": n_train,
                "reg_depth": reg_depth,
                "clf_depth": clf_depth,
                "objective": objective,
                "objective_score": score,
                "sharpe_ratio": _safe_float(stats.get("Sharpe Ratio")),
                "return_pct": _safe_float(stats.get("Return [%]")),
                "max_drawdown_pct": _safe_float(stats.get("Max. Drawdown [%]")),
                "win_rate_pct": _safe_float(stats.get("Win Rate [%]")),
                "n_trades": _safe_float(stats.get("# Trades")),
            }

            strategy_obj = stats.get("_strategy")
            if strategy_name == "classification":
                row["prob_buy"] = getattr(strategy_obj, "prob_buy", np.nan)
                row["prob_sell"] = getattr(strategy_obj, "prob_sell", np.nan)
            else:
                row["limit_buy"] = getattr(strategy_obj, "limit_buy", np.nan)
                row["limit_sell"] = getattr(strategy_obj, "limit_sell", np.nan)

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
                    "objective_score": np.nan,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results).sort_values(
        by="objective_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if best_bundle is None:
        raise ValueError("Auto-optimization failed for all tested parameter combinations.")

    best_bundle["search_results"] = results_df
    return best_bundle


def auto_select_best_strategy(
    prepare_data_fn,
    ticker: str,
    start: str,
    end: str | None,
    interval: str,
    cash: float = 10000,
    commission: float = 0.002,
    objective: str = "Balanced",
):
    candidate_strategies = [
        "classification",
        "regression",
        "walk_forward_anchored",
        "walk_forward_unanchored",
    ]

    overall_rows: list[dict[str, Any]] = []
    best_global_score = -np.inf
    best_global_bundle = None

    for strategy_name in candidate_strategies:
        try:
            bundle = auto_optimize_model_and_strategy(
                prepare_data_fn=prepare_data_fn,
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                strategy_name=strategy_name,
                cash=cash,
                commission=commission,
                objective=objective,
            )

            stats = bundle["best_stats"]
            score = _objective_score(stats, objective)

            row = {
                "strategy": strategy_name,
                "objective": objective,
                "objective_score": score,
                "sharpe_ratio": _safe_float(stats.get("Sharpe Ratio")),
                "return_pct": _safe_float(stats.get("Return [%]")),
                "max_drawdown_pct": _safe_float(stats.get("Max. Drawdown [%]")),
                "win_rate_pct": _safe_float(stats.get("Win Rate [%]")),
                "n_trades": _safe_float(stats.get("# Trades")),
            }

            row.update(
                {
                    k: v
                    for k, v in bundle["best_params"].items()
                    if k
                    not in {
                        "strategy",
                        "objective",
                        "objective_score",
                        "sharpe_ratio",
                        "return_pct",
                        "max_drawdown_pct",
                        "win_rate_pct",
                        "n_trades",
                    }
                }
            )

            overall_rows.append(row)

            if np.isfinite(score) and score > best_global_score:
                best_global_score = score
                best_global_bundle = bundle.copy()
                best_global_bundle["best_strategy_name"] = strategy_name

        except Exception as e:
            overall_rows.append(
                {
                    "strategy": strategy_name,
                    "objective": objective,
                    "objective_score": np.nan,
                    "error": str(e),
                }
            )

    overall_df = pd.DataFrame(overall_rows).sort_values(
        by="objective_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if best_global_bundle is None:
        raise ValueError("Automatic strategy selection failed for all strategies.")

    best_global_bundle["strategy_ranking"] = overall_df
    return best_global_bundle
