from __future__ import annotations

import numpy as np
import pandas as pd


def attach_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Trend_Regime" not in out.columns:
        out["Trend_Regime"] = (out["SMA_5_Ratio"] > out["SMA_10_Ratio"]).astype(int)

    vol_cut = out["Volatility_20"].median()

    out["Vol_Regime"] = np.where(out["Volatility_20"] >= vol_cut, "high_vol", "low_vol")
    out["Trend_Label"] = np.where(out["Trend_Regime"] == 1, "uptrend", "downtrend")
    out["Market_Regime"] = out["Trend_Label"] + "_" + out["Vol_Regime"]

    return out


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Market_Regime" not in df.columns:
        df = attach_regime_labels(df)

    summary = (
        df.groupby("Market_Regime")
        .agg(
            count=("Market_Regime", "size"),
            avg_return_1d=("Return_1d", "mean"),
            avg_vol_20=("Volatility_20", "mean"),
            avg_gap=("Gap_Pct", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return summary


def adjust_classification_thresholds(
    row: pd.Series,
    base_prob_buy: float,
    base_prob_sell: float,
) -> tuple[float, float]:
    prob_buy = base_prob_buy
    prob_sell = base_prob_sell

    trend_regime = int(row.get("Trend_Regime", 1))
    vol_20 = float(row.get("Volatility_20", 0.0))
    vol_10 = float(row.get("Volatility_10", 0.0))

    # High volatility → require stronger confidence
    if vol_20 > vol_10 and vol_20 > 1.5:
        prob_buy += 0.03
        prob_sell += 0.02

    # Downtrend → harder to open longs, easier to close them
    if trend_regime == 0:
        prob_buy += 0.05
        prob_sell += 0.03

    prob_buy = min(prob_buy, 0.95)
    prob_sell = min(prob_sell, 0.95)
    return prob_buy, prob_sell


def adjust_regression_thresholds(
    row: pd.Series,
    base_limit_buy: float,
    base_limit_sell: float,
) -> tuple[float, float]:
    limit_buy = base_limit_buy
    limit_sell = base_limit_sell

    trend_regime = int(row.get("Trend_Regime", 1))
    vol_20 = float(row.get("Volatility_20", 0.0))
    vol_10 = float(row.get("Volatility_10", 0.0))

    # High volatility → require stronger positive forecast and tighter exits
    if vol_20 > vol_10 and vol_20 > 1.5:
        limit_buy += 0.25
        limit_sell += 0.25

    # Downtrend → make long entries harder
    if trend_regime == 0:
        limit_buy += 0.50
        limit_sell += 0.25

    return limit_buy, limit_sell
