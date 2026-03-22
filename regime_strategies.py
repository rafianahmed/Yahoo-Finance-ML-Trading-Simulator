from __future__ import annotations

from strategies import (
    GenericClassificationStrategy,
    GenericClassificationWalkForwardAnchored,
    GenericClassificationWalkForwardUnanchored,
    GenericRegressionStrategy,
    GenericRegressionWalkForwardAnchored,
    GenericRegressionWalkForwardUnanchored,
)
from regime_extensions import (
    adjust_classification_thresholds,
    adjust_regression_thresholds,
)


class RegimeAwareClassificationStrategy(GenericClassificationStrategy):
    def next(self):
        if not self._enough_data():
            return

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        prob_up = self._predict_prob_up(row_df)

        prob_buy, prob_sell = adjust_classification_thresholds(
            row=row,
            base_prob_buy=self.prob_buy,
            base_prob_sell=self.prob_sell,
        )

        if prob_up >= prob_buy and not self.position:
            self.buy()
        elif prob_up <= prob_sell and self.position:
            self.position.close()


class RegimeAwareClassificationWalkForwardAnchored(GenericClassificationWalkForwardAnchored):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[: len(self.data)]
            self._retrain_model(train_df)

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        prob_up = self._predict_prob_up(row_df)

        prob_buy, prob_sell = adjust_classification_thresholds(
            row=row,
            base_prob_buy=self.prob_buy,
            base_prob_sell=self.prob_sell,
        )

        if prob_up >= prob_buy and not self.position:
            self.buy()
        elif prob_up <= prob_sell and self.position:
            self.position.close()


class RegimeAwareClassificationWalkForwardUnanchored(GenericClassificationWalkForwardUnanchored):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[-self._full_data.n_train :]
            self._retrain_model(train_df)

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        prob_up = self._predict_prob_up(row_df)

        prob_buy, prob_sell = adjust_classification_thresholds(
            row=row,
            base_prob_buy=self.prob_buy,
            base_prob_sell=self.prob_sell,
        )

        if prob_up >= prob_buy and not self.position:
            self.buy()
        elif prob_up <= prob_sell and self.position:
            self.position.close()


class RegimeAwareRegressionStrategy(GenericRegressionStrategy):
    def next(self):
        if not self._enough_data():
            return

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        pred = float(self.model.predict(row_df[self.feature_columns])[0])

        limit_buy, limit_sell = adjust_regression_thresholds(
            row=row,
            base_limit_buy=self.limit_buy,
            base_limit_sell=self.limit_sell,
        )

        if pred >= limit_buy and not self.position:
            self.buy()
        elif pred <= limit_sell and self.position:
            self.position.close()


class RegimeAwareRegressionWalkForwardAnchored(GenericRegressionWalkForwardAnchored):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[: len(self.data)]
            self._retrain_model(train_df)

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        pred = float(self.model.predict(row_df[self.feature_columns])[0])

        limit_buy, limit_sell = adjust_regression_thresholds(
            row=row,
            base_limit_buy=self.limit_buy,
            base_limit_sell=self.limit_sell,
        )

        if pred >= limit_buy and not self.position:
            self.buy()
        elif pred <= limit_sell and self.position:
            self.position.close()


class RegimeAwareRegressionWalkForwardUnanchored(GenericRegressionWalkForwardUnanchored):
    def next(self):
        if not self._enough_data():
            return

        if len(self.data) % self.coef_retrain == 0:
            train_df = self.data.df.iloc[-self._full_data.n_train :]
            self._retrain_model(train_df)

        row_df = self._get_latest_row()
        row = row_df.iloc[0]
        pred = float(self.model.predict(row_df[self.feature_columns])[0])

        limit_buy, limit_sell = adjust_regression_thresholds(
            row=row,
            base_limit_buy=self.limit_buy,
            base_limit_sell=self.limit_sell,
        )

        if pred >= limit_buy and not self.position:
            self.buy()
        elif pred <= limit_sell and self.position:
            self.position.close()
