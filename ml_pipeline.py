from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from strategies import BASE_FEATURES


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_columns: List[str]
    regression_target: str
    classification_target: str
    model_reg: DecisionTreeRegressor
    model_clf: DecisionTreeClassifier
    n_train: int


def _normalize_date_input(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _clamp_end_date(end: str | date | datetime | None) -> str | None:
    end_str = _normalize_date_input(end)
    if end_str is None or end_str == "":
        return None

    try:
        parsed = pd.to_datetime(end_str).date()
        return min(parsed, date.today()).isoformat()
    except Exception:
        return end_str


@st.cache_data(ttl=3600, show_spinner=False)
def download_yahoo_data(
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    ticker = str(ticker).strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    start_str = _normalize_date_input(start)
    end_str = _clamp_end_date(end)

    last_error: Exception | None = None

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                raise ValueError(
                    f"No Yahoo Finance data returned for ticker: {ticker}. "
                    f"Check the ticker symbol, date range, or try again later."
                )

            df = df.reset_index()

            required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Downloaded data is missing required columns: {sorted(missing)}")

            return df

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2)

    raise ValueError(f"Yahoo Finance download failed for ticker {ticker}: {last_error}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Return_1d"] = data["Close"].pct_change() * 100
    data["Return_5d"] = data["Close"].pct_change(5) * 100
    data["Range_Pct"] = ((data["High"] - data["Low"]) / data["Close"].replace(0, np.nan)) * 100
    data["Gap_Pct"] = ((data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1).replace(0, np.nan)) * 100
    data["Volume_Change"] = data["Volume"].pct_change() * 100
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_5_Ratio"] = data["Close"] / data["SMA_5"].replace(0, np.nan)
    data["SMA_10_Ratio"] = data["Close"] / data["SMA_10"].replace(0, np.nan)
    data["Volatility_5"] = data["Return_1d"].rolling(5).std()

    data["change_tomorrow"] = ((data["Close"].shift(-1) - data["Close"]) / data["Close"].replace(0, np.nan)) * 100
    data["change_tomorrow_direction"] = (data["change_tomorrow"] > 0).astype(int)

    keep_cols = ["Date", *BASE_FEATURES, "change_tomorrow", "change_tomorrow_direction"]

    missing_features = [col for col in keep_cols if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing engineered columns: {missing_features}")

    data = data[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable rows remain after feature engineering.")

    return data


def prepare_data(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    n_train: int = 600,
    reg_depth: int = 15,
    clf_depth: int = 8,
) -> PreparedData:
    raw = download_yahoo_data(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
    )

    df = engineer_features(raw)

    if len(df) <= n_train + 30:
        raise ValueError(
            f"Not enough usable rows after feature engineering. "
            f"Got {len(df)} rows; need more than {n_train + 30}. "
            f"Try a longer date range or a smaller training window."
        )

    model_reg = DecisionTreeRegressor(
        max_depth=reg_depth,
        random_state=42,
    )
    model_clf = DecisionTreeClassifier(
        max_depth=clf_depth,
        random_state=42,
    )

    return PreparedData(
        df=df,
        feature_columns=BASE_FEATURES,
        regression_target="change_tomorrow",
        classification_target="change_tomorrow_direction",
        model_reg=model_reg,
        model_clf=model_clf,
        n_train=n_train,
    )


def time_series_scores(prepared: PreparedData, n_splits: int = 5) -> Dict[str, float]:
    df = prepared.df.copy()

    X = df[prepared.feature_columns]
    y_reg = df[prepared.regression_target]
    y_clf = df[prepared.classification_target]

    if len(df) <= n_splits:
        raise ValueError("Not enough rows to run time-series cross-validation.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores: List[float] = []
    acc_scores: List[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        reg = clone(prepared.model_reg)
        clf = clone(prepared.model_clf)

        reg.fit(X_train, y_train_reg)
        clf.fit(X_train, y_train_clf)

        pred_reg = reg.predict(X_test)
        pred_clf = clf.predict(X_test)

        mse_scores.append(mean_squared_error(y_test_reg, pred_reg))
        acc_scores.append(accuracy_score(y_test_clf, pred_clf))

    return {
        "regression_cv_mse": float(np.mean(mse_scores)),
        "classification_cv_accuracy": float(np.mean(acc_scores)),
        "rows_used": int(len(df)),
    }
