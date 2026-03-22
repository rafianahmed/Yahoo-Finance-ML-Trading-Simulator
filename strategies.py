from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


BASE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return_1d",
    "Return_5d",
    "Range_Pct",
    "Gap_Pct",
    "Volume_Change",
    "SMA_5_Ratio",
    "SMA_10_Ratio",
    "Volatility_5",
]


class RegressionStrategy(Strategy):
    limit_buy = 1.0
    limit_sell = -5.0
    max_depth = 15

    def init(self):
        self.model = clone(self._full_data.model_reg)
        self.feature_columns = self._full_data.feature_columns
        self.target_column = self._full_data.regression_target
        self.already_bought = False
        self._fit_initial_model()

    def _fit_initial_model(self):
        train_df = self.data.df.iloc[: self._full_data.n_train]
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def next(self):
        row = self.data.df.iloc[[-1]]
        forecast_tomorrow = float(self.model.predict(row[self.feature_columns])[0])

        if forecast_tomorrow > self.limit_buy and not self.position:
            self.buy()
            self.already_bought = True
        elif forecast_tomorrow < self.limit_sell and self.position:
            self.position.close()
            self.already_bought = False


class WalkForwardAnchored(RegressionStrategy):
    coef_retrain = 200

    def next(self):
        if len(self.data) < self._full_data.n_train:
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[: len(self.data)]
            X_train = train_df[self.feature_columns]
            y_train = train_df[self.target_column]
            self.model.fit(X_train, y_train)

        super().next()


class WalkForwardUnanchored(RegressionStrategy):
    coef_retrain = 200

    def next(self):
        if len(self.data) < self._full_data.n_train:
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[-self._full_data.n_train :]
            X_train = train_df[self.feature_columns]
            y_train = train_df[self.target_column]
            self.model.fit(X_train, y_train)

        super().next()


class ClassificationStrategy(Strategy):
    prob_buy = 0.55
    prob_sell = 0.45

    def init(self):
        self.model = clone(self._full_data.model_clf)
        self.feature_columns = self._full_data.feature_columns
        self.target_column = self._full_data.classification_target
        self._fit_initial_model()

    def _fit_initial_model(self):
        train_df = self.data.df.iloc[: self._full_data.n_train]
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def next(self):
        if len(self.data) < self._full_data.n_train:
            return
        row = self.data.df.iloc[[-1]]
        if hasattr(self.model, "predict_proba"):
            prob_up = float(self.model.predict_proba(row[self.feature_columns])[0][1])
        else:
            prob_up = float(self.model.predict(row[self.feature_columns])[0])

        if prob_up >= self.prob_buy and not self.position:
            self.buy()
        elif prob_up <= self.prob_sell and self.position:
            self.position.close()
