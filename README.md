# Yahoo Finance ML Trading Simulator

A fully automated Streamlit-based machine learning trading simulator that downloads stock data from Yahoo Finance, engineers advanced features, runs multiple predictive models, optimizes trading strategies, and ranks all results automatically.

The user only needs to:

1. Select a ticker  
2. Click **Run simulation**

Everything else is handled automatically.

---

## Overview

This project integrates:

- Data collection from Yahoo Finance
- Advanced feature engineering
- Multiple machine learning models
- Walk-forward retraining
- Strategy optimization
- Backtesting simulation
- Model comparison and ranking
- Time-series, clustering, and anomaly diagnostics

---

## What this simulator does

After selecting a ticker, the app:

- downloads OHLCV data from Yahoo Finance
- engineers advanced predictive features
- builds classification and regression targets
- trains multiple ML models
- evaluates predictive performance
- runs trading strategies via backtesting
- optimizes trading thresholds automatically
- compares all models and ranks them
- displays results in an interactive dashboard

---

## Data source

Data is fetched from **Yahoo Finance**.

Example tickers:
- AAPL
- MSFT
- TSLA
- NVDA
- AMZN
- ^GSPC

Supported intervals:
- 1d
- 1wk
- 1mo

---

## Feature engineering

The simulator uses a rich feature set combining price, momentum, volatility, and structure.

### Base price features
- Open
- High
- Low
- Close
- Volume

---

## Engineered features

### Returns & price structure
- 1-day return
- 5-day return
- intraday range %
- overnight gap %
- close position within range

---

### Trend features
- SMA(5), SMA(10)
- SMA ratios
- SMA slopes (trend strength)

---

### Volatility features
- 5-day volatility
- 10-day volatility
- 20-day volatility

---

### Momentum indicators
- RSI (14)
- MACD
- MACD signal
- MACD histogram

---

### Lag features (autoregressive)
- Lag return 1
- Lag return 2
- Lag return 3

---

### Rolling statistics
- rolling mean (5)
- rolling max (5)
- rolling min (5)

---

### Volume features
- volume change %
- volume moving average
- volume ratio

---

### Market regime feature
- trend regime (SMA5 > SMA10)

---

## Targets

### Regression target
- `change_tomorrow`
- predicts next-day % return

---

### Classification target
- `change_tomorrow_direction`
- predicts next-day up/down movement

---

## Models used

## 1. Classification models

Used to predict direction.

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost *(optional)*

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 2. Regression models

Used to predict return magnitude.

- Multiple Linear Regression ✅
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor *(optional)*

### Important clarification
The simulator uses:

**Multiple Linear Regression (not single-variable regression)**

Because it uses multiple input features simultaneously:
- price
- volume
- momentum
- volatility
- lag features
- technical indicators

---

### Metrics
- MSE
- RMSE
- MAE
- R²

---

## Training styles

Each model is tested under:

### 1. Static
- trained once on initial data

---

### 2. Walk-forward anchored
- retrained using all historical data

---

### 3. Walk-forward unanchored
- retrained using rolling recent window

---

## Strategy logic

### Classification strategy
- Buy when probability ≥ `prob_buy`
- Sell when probability ≤ `prob_sell`

---

### Regression strategy
- Buy when predicted return ≥ `limit_buy`
- Sell when predicted return ≤ `limit_sell`

---

## Optimization

The simulator automatically optimizes:

### Model hyperparameters
- tree depth
- learning rate
- number of estimators
- regularization

---

### Strategy thresholds

Classification:
- `prob_buy`
- `prob_sell`

Regression:
- `limit_buy`
- `limit_sell`

---

## Backtesting

Uses `backtesting.py` to simulate trading.

### Metrics reported
- Return %
- Sharpe Ratio
- Max Drawdown %
- Win Rate %
- Number of trades
- Equity curve

---

## Ranking system

Models are ranked using:

### 1. Return [%]
Maximizes profit

---

### 2. Sharpe Ratio
Maximizes risk-adjusted return

---

### 3. Balanced (recommended)
Combines:
- Sharpe
- Return
- Drawdown penalty
- Win rate
- trade count penalty

---

## Additional diagnostics

### Time-series models
- naive forecast
- moving average forecast
- Prophet *(optional)*

Metrics:
- MSE
- RMSE
- MAE
- MAPE

---

### Clustering (market regimes)
- KMeans

Metric:
- silhouette score

---

### Outlier detection
- Isolation Forest

Metric:
- anomaly rate %

---

## Project structure
