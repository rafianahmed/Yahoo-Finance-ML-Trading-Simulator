# Yahoo Finance ML Trading Simulator

A fully automated machine learning trading simulation system that downloads financial data, engineers predictive features, trains multiple models, evaluates performance, optimizes trading strategies, and ranks results — all from a single click.

---

## Overview

This project is an end-to-end quantitative trading research environment built using:

* **Yahoo Finance data**
* **Machine Learning (classification + regression)**
* **Time-series validation**
* **Walk-forward retraining**
* **Strategy optimization**
* **Backtesting simulation**

The system is designed to simulate how ML models behave in realistic trading conditions.

---

## How the simulator works (Pipeline)

When you select a ticker and click **Run simulation**, the system performs:

### 1. Data Collection

* Downloads OHLCV data from Yahoo Finance
* Supports multiple tickers and time intervals

---

### 2. Feature Engineering

Transforms raw price data into predictive signals.

#### Price & Returns

* 1-day return
* 5-day return
* intraday range %
* overnight gap %

#### Trend Features

* SMA (5, 10)
* SMA ratios
* SMA slopes

#### Volatility

* rolling volatility (5, 10, 20)

#### Momentum Indicators

* RSI (14)
* MACD
* MACD signal & histogram

#### Lag Features (Time Dependency)

* lag returns (1, 2, 3)

#### Rolling Statistics

* rolling mean / max / min

#### Volume Signals

* volume change
* volume moving average
* volume ratio

#### Market Structure

* close position within daily range
* trend regime indicator

---

### 3. Target Creation

#### Regression Target

* Predict next-day % return

#### Classification Target

* Predict next-day direction (up/down)

---

## Models Used

The simulator runs **multiple models automatically**

### Regression Models

* Multiple Linear Regression ✅
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost (optional)

### Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost (optional)

---

## What is Multiple Linear Regression?

The system uses:

👉 **Multiple Linear Regression (not single-variable regression)**

Because predictions are based on **many features simultaneously**:

* price
* momentum
* volatility
* lag signals
* technical indicators

---

## Training Styles

Each model is tested under different learning strategies:

### Static

* Model trained once and used throughout

### Walk-Forward Anchored

* Model retrained using all past data

### Walk-Forward Unanchored

* Model retrained using recent rolling window only

👉 This simulates real-world adaptive learning

---

## Strategy Logic

### Classification Strategy

* Buy if probability of price increase ≥ threshold
* Sell if probability falls below threshold

### Regression Strategy

* Buy if predicted return ≥ buy threshold
* Sell if predicted return ≤ sell threshold

---

## Optimization

The system automatically optimizes:

### Model Hyperparameters

* tree depth
* learning rate
* number of estimators

### Trading Strategy Parameters

* buy/sell thresholds

---

## Ranking Objectives

The simulator evaluates strategies using:

### 1. Balanced (Recommended)

Combines:

* Sharpe Ratio
* Return %
* Drawdown penalty
* Win rate
* trade frequency

👉 Best for realistic trading

---

### 2. Sharpe Ratio

* Measures risk-adjusted return
* Prefers stable strategies

---

### 3. Return [%]

* Maximizes profit only
* Ignores risk

---

## Backtesting Engine

Uses **backtesting.py** to simulate trading.

### Outputs:

* Total Return %
* Sharpe Ratio
* Max Drawdown %
* Win Rate %
* Number of trades
* Equity curve

---

## Model Evaluation

### Regression Metrics

* MSE
* RMSE
* MAE
* R²

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Additional Analysis Modules

### Time-Series Baselines

* naive prediction
* moving average prediction
* Prophet (optional)

---

### Clustering (Market Regimes)

* KMeans
* Silhouette score

---

### Outlier Detection

* Isolation Forest
* anomaly rate %

---

## Automation

The simulator is **fully automated**:

👉 Runs ALL models
👉 Evaluates ALL strategies
👉 Optimizes parameters
👉 Compares results
👉 Ranks best model

User only selects:

* ticker
* date range

---

## Project Structure

```
Yahoo-Finance-ML-Trading-Simulator/
├── app.py
├── ml_pipeline.py
├── strategies.py
├── backtest_engine.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Run app

```
streamlit run app.py
```

---

## Key Insights

* High prediction accuracy ≠ profitable trading
* Walk-forward testing prevents overfitting
* Feature engineering is critical
* Optimization improves execution, not prediction
* Balanced objective gives best real-world performance

---

## Limitations

* Not a live trading system
* No slippage or market impact modeling
* Yahoo Finance data limitations
* Some models optional (XGBoost, Prophet)

---

## Future Improvements

* Portfolio optimization
* multi-asset trading
* deep learning (LSTM, Transformers)
* feature importance (SHAP)
* regime-switching strategies

---

## Summary

This system is a complete ML trading research framework that:

* builds advanced features
* runs multiple ML models
* applies realistic training methods
* optimizes trading strategies
* evaluates performance
* ranks models automatically

All from a single click.

---

## Author

Rafian Ahmed Raad

---

## Disclaimer

This project is for educational and research purposes only.
It is not financial advice.
