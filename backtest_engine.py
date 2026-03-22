from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from strategies import BASE_FEATURES

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    model: Any
    task: str
    model_family: str
    model_name: str
    training_style: str
    n_train: int
    model_params: dict[str, Any]


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
    start_str = _normalize_date_input(start)
    end_str = _clamp_end_date(end)

    last_error = None
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
                raise ValueError(f"No Yahoo Finance data returned for ticker: {ticker}")

            df = df.reset_index()
            required = {"Date", "Open", "High", "Low", "Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing required Yahoo columns: {sorted(missing)}")
            return df

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2)

    raise ValueError(f"Yahoo Finance download failed for {ticker}: {last_error}")


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

    keep_cols = ["Date", "Close", *BASE_FEATURES, "change_tomorrow", "change_tomorrow_direction"]
    data = data[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable rows remain after feature engineering.")

    return data


def get_model_registry() -> dict[str, dict[str, Any]]:
    registry = {
        "classification": {
            "logistic_regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
            "decision_tree_classifier": DecisionTreeClassifier(random_state=42),
            "random_forest_classifier": RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
            "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
        },
        "regression": {
            "linear_regression": LinearRegression(),
            "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
            "random_forest_regressor": RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
        },
    }

    if XGB_AVAILABLE:
        registry["classification"]["xgboost_classifier"] = XGBClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        )
        registry["regression"]["xgboost_regressor"] = XGBRegressor(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
        )

    return registry


def build_model(task: str, model_name: str, model_params: dict[str, Any] | None = None):
    registry = get_model_registry()
    if task not in registry or model_name not in registry[task]:
        raise ValueError(f"Unknown model for task={task}: {model_name}")

    model = clone(registry[task][model_name])
    if model_params:
        model.set_params(**model_params)
    return model


def prepare_data(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    n_train: int = 600,
    task: str = "classification",
    model_name: str = "logistic_regression",
    training_style: str = "static",
    model_params: dict[str, Any] | None = None,
) -> PreparedData:
    raw = download_yahoo_data(ticker=ticker, start=start, end=end, interval=interval)
    df = engineer_features(raw)

    if len(df) <= n_train + 30:
        raise ValueError(
            f"Not enough usable rows after feature engineering. Got {len(df)} rows; need more than {n_train + 30}."
        )

    if task == "classification":
        target_column = "change_tomorrow_direction"
        model_family = "classification"
    elif task == "regression":
        target_column = "change_tomorrow"
        model_family = "regression"
    else:
        raise ValueError(f"Unsupported task: {task}")

    model = build_model(task=task, model_name=model_name, model_params=model_params or {})

    return PreparedData(
        df=df,
        feature_columns=BASE_FEATURES,
        target_column=target_column,
        model=model,
        task=task,
        model_family=model_family,
        model_name=model_name,
        training_style=training_style,
        n_train=n_train,
        model_params=model_params or {},
    )


def assess_model(prepared: PreparedData, n_splits: int = 5) -> dict[str, float]:
    df = prepared.df
    X = df[prepared.feature_columns]
    y = df[prepared.target_column]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if prepared.task == "regression":
        mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = clone(prepared.model)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            mse = mean_squared_error(y_test, pred)
            mse_scores.append(mse)
            rmse_scores.append(np.sqrt(mse))
            mae_scores.append(mean_absolute_error(y_test, pred))
            r2_scores.append(r2_score(y_test, pred))

        return {
            "rows_used": float(len(df)),
            "cv_mse": float(np.mean(mse_scores)),
            "cv_rmse": float(np.mean(rmse_scores)),
            "cv_mae": float(np.mean(mae_scores)),
            "cv_r2": float(np.mean(r2_scores)),
        }

    acc_scores, prec_scores, rec_scores, f1_scores, roc_scores = [], [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(prepared.model)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, pred))
        prec_scores.append(precision_score(y_test, pred, zero_division=0))
        rec_scores.append(recall_score(y_test, pred, zero_division=0))
        f1_scores.append(f1_score(y_test, pred, zero_division=0))

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            try:
                roc_scores.append(roc_auc_score(y_test, prob))
            except Exception:
                pass

    result = {
        "rows_used": float(len(df)),
        "cv_accuracy": float(np.mean(acc_scores)),
        "cv_precision": float(np.mean(prec_scores)),
        "cv_recall": float(np.mean(rec_scores)),
        "cv_f1": float(np.mean(f1_scores)),
    }
    if roc_scores:
        result["cv_roc_auc"] = float(np.mean(roc_scores))
    return result


def get_hyperparameter_grid(task: str, model_name: str) -> list[dict[str, Any]]:
    if model_name in {"linear_regression", "logistic_regression"}:
        if model_name == "logistic_regression":
            return [{"C": c} for c in [0.1, 1.0, 3.0]]
        return [{}]

    if "decision_tree" in model_name:
        return [{"max_depth": d} for d in [3, 5, 8, 12]]

    if "random_forest" in model_name:
        return [
            {"n_estimators": 200, "max_depth": d}
            for d in [4, 8, None]
        ]

    if "gradient_boosting" in model_name:
        return [
            {"n_estimators": 150, "learning_rate": lr, "max_depth": d}
            for lr in [0.05, 0.1]
            for d in [2, 3]
        ]

    if "xgboost" in model_name:
        return [
            {"n_estimators": 150, "learning_rate": lr, "max_depth": d}
            for lr in [0.05, 0.1]
            for d in [3, 5]
        ]

    return [{}]


def evaluate_time_series_suite(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    ts = df[["Date", "Close"]].copy()
    ts["naive_next_close_pred"] = ts["Close"].shift(1)
    ts["ma5_next_close_pred"] = ts["Close"].rolling(5).mean().shift(1)

    eval_df = ts.dropna().copy()
    if not eval_df.empty:
        for model_name, pred_col in [
            ("naive_last_close", "naive_next_close_pred"),
            ("rolling_ma5", "ma5_next_close_pred"),
        ]:
            actual = eval_df["Close"]
            pred = eval_df[pred_col]
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)
            mape = float(np.mean(np.abs((actual - pred) / actual.replace(0, np.nan))) * 100)

            rows.append(
                {
                    "model_family": "time_series",
                    "model_name": model_name,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape_pct": mape,
                }
            )

    if PROPHET_AVAILABLE and len(ts) > 80:
        try:
            prophet_df = ts.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]].copy()
            split = int(len(prophet_df) * 0.8)
            train_df = prophet_df.iloc[:split].copy()
            test_df = prophet_df.iloc[split:].copy()

            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(train_df)

            future = test_df[["ds"]].copy()
            forecast = model.predict(future)
            pred = forecast["yhat"].values
            actual = test_df["y"].values

            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)
            mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100)

            rows.append(
                {
                    "model_family": "time_series",
                    "model_name": "prophet",
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape_pct": mape,
                }
            )
        except Exception:
            pass

    return pd.DataFrame(rows)


def evaluate_clustering_suite(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[feature_columns].copy()
    rows = []
    labeled = df.copy()

    for k in [2, 3, 4]:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)

            rows.append(
                {
                    "model_family": "clustering",
                    "model_name": f"kmeans_k_{k}",
                    "silhouette_score": float(sil),
                }
            )

            if k == 3:
                labeled["regime_cluster"] = labels
        except Exception:
            rows.append(
                {
                    "model_family": "clustering",
                    "model_name": f"kmeans_k_{k}",
                    "silhouette_score": np.nan,
                }
            )

    return pd.DataFrame(rows), labeled


def evaluate_outlier_suite(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns].copy()

    rows = []
    try:
        iso = IsolationForest(random_state=42, contamination=0.05)
        preds = iso.fit_predict(X)
        anomaly_rate = float((preds == -1).mean() * 100)

        rows.append(
            {
                "model_family": "outlier",
                "model_name": "isolation_forest",
                "anomaly_rate_pct": anomaly_rate,
            }
        )
    except Exception:
        rows.append(
            {
                "model_family": "outlier",
                "model_name": "isolation_forest",
                "anomaly_rate_pct": np.nan,
            }
        )

    return pd.DataFrame(rows)
