from __future__ import annotations

from backtesting import Strategy
from sklearn.base import clone


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
    "Volatility_10",
    "Volatility_20",
    "Lag_Return_1",
    "Lag_Return_2",
    "Lag_Return_3",
    "Rolling_Mean_5",
    "Rolling_Max_5",
    "Rolling_Min_5",
    "Volume_SMA_5",
    "Volume_Ratio",
    "Close_Position",
    "SMA_5_Slope",
    "SMA_10_Slope",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Trend_Regime",
]


class BaseMLStrategy(Strategy):
    _full_data = None
    coef_retrain = 200

    def _check_bound_data(self):
        if self._full_data is None:
            raise ValueError(f"{self.__class__.__name__} requires bound prepared data via _full_data.")

    def _enough_data(self):
        return len(self.data) >= self._full_data.n_train

    def _get_latest_row(self):
        return self.data.df.iloc[[-1]]


class GenericRegressionStrategy(BaseMLStrategy):
    limit_buy = 0.5
    limit_sell = -0.5

    def init(self):
        self._check_bound_data()
        self.model = clone(self._full_data.model)
        self.feature_columns = self._full_data.feature_columns
        self.target_column = self._full_data.target_column
        self._fit_initial_model()

    def _fit_initial_model(self):
        train_df = self.data.df.iloc[: self._full_data.n_train]
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def _retrain_model(self, train_df):
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def next(self):
        if not self._enough_data():
            return

        row = self._get_latest_row()
        pred = float(self.model.predict(row[self.feature_columns])[0])

        if pred >= self.limit_buy and not self.position:
            self.buy()
        elif pred <= self.limit_sell and self.position:
            self.position.close()


class GenericRegressionWalkForwardAnchored(GenericRegressionStrategy):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[: len(self.data)]
            self._retrain_model(train_df)

        super().next()


class GenericRegressionWalkForwardUnanchored(GenericRegressionStrategy):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[-self._full_data.n_train :]
            self._retrain_model(train_df)

        super().next()


class GenericClassificationStrategy(BaseMLStrategy):
    prob_buy = 0.55
    prob_sell = 0.45

    def init(self):
        self._check_bound_data()
        self.model = clone(self._full_data.model)
        self.feature_columns = self._full_data.feature_columns
        self.target_column = self._full_data.target_column
        self._fit_initial_model()

    def _fit_initial_model(self):
        train_df = self.data.df.iloc[: self._full_data.n_train]
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def _retrain_model(self, train_df):
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        self.model.fit(X_train, y_train)

    def _predict_prob_up(self, row):
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(row[self.feature_columns])[0][1])
        return float(self.model.predict(row[self.feature_columns])[0])

    def next(self):
        if not self._enough_data():
            return

        row = self._get_latest_row()
        prob_up = self._predict_prob_up(row)

        if prob_up >= self.prob_buy and not self.position:
            self.buy()
        elif prob_up <= self.prob_sell and self.position:
            self.position.close()


class GenericClassificationWalkForwardAnchored(GenericClassificationStrategy):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[: len(self.data)]
            self._retrain_model(train_df)

        super().next()


class GenericClassificationWalkForwardUnanchored(GenericClassificationStrategy):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[-self._full_data.n_train :]
            self._retrain_model(train_df)

        super().next()
