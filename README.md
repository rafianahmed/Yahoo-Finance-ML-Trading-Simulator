# Yahoo Finance ML Trading Simulator

A Streamlit-based stock market simulation app that downloads historical market data from Yahoo Finance, builds machine learning features, trains predictive models, and evaluates trading strategies through backtesting.

This project combines **data preprocessing**, **predictive modeling**, **walk-forward retraining**, **parameter optimization**, and **strategy simulation** into one interactive app.

---

## What this simulator does

This simulator lets you test how machine learning models might be used for **next-day stock movement prediction** and **rule-based trading decisions**.

It does the following:

- fetches historical OHLCV stock data from **Yahoo Finance**
- creates engineered features from price and volume data
- predicts **next-day return** using a regression model
- predicts **next-day direction** using a classification model
- converts model predictions into simulated buy/sell signals
- runs a historical backtest using `backtesting.py`
- supports **anchored** and **unanchored** walk-forward retraining
- optionally optimizes decision thresholds for better backtest performance
- displays model metrics, price charts, strategy performance, and equity curves in a Streamlit dashboard

---

## Main features

### 1. Yahoo Finance data download
The app downloads stock data for any valid Yahoo Finance ticker, such as:

- `AAPL`
- `MSFT`
- `TSLA`
- `NVDA`
- `^GSPC`

Supported intervals include:

- `1d`
- `1wk`
- `1mo`

---

### 2. Feature engineering
The simulator transforms raw stock data into machine learning inputs, including:

- 1-day return
- 5-day return
- intraday price range percentage
- gap percentage
- volume change
- 5-day moving average ratio
- 10-day moving average ratio
- short-term volatility

These features are used to predict the next trading day.

---

### 3. Predictive modeling
The simulator includes two machine learning approaches:

#### Regression
Uses a **DecisionTreeRegressor** to predict the **next-day percentage return**.

#### Classification
Uses a **DecisionTreeClassifier** to predict whether the **next-day price direction** will be up or down.

---

### 4. Trading strategy simulation
The predicted outputs are converted into trading rules:

- **Regression strategy**
  - buy when predicted return is above a buy threshold
  - close the position when predicted return falls below a sell threshold

- **Classification strategy**
  - buy when predicted probability of upward movement is high enough
  - close when the probability falls below a sell threshold

---

### 5. Walk-forward testing
The simulator supports more realistic time-series retraining methods:

#### Anchored walk-forward
The model is retrained on all available data up to the current point.

#### Unanchored walk-forward
The model is retrained only on the most recent rolling training window.

This helps compare static training vs adaptive retraining behavior.

---

### 6. Parameter optimization
The app can optimize strategy thresholds using `backtesting.py`:

- classification buy/sell probabilities
- regression buy/sell limits

This helps test whether different parameter values improve return.

---

### 7. Interactive dashboard
The Streamlit interface allows you to:

- choose a ticker
- select date range and interval
- set training window size
- set model depth
- choose strategy type
- enable optimization
- run the simulation instantly

The app then shows:

- prepared dataset preview
- cross-validation metrics
- price chart
- backtest statistics
- equity curve
- optimization results

---

## Project structure

```text
trading_simulator/
│
├── app.py
├── ml_pipeline.py
├── strategies.py
├── backtest_engine.py
├── requirements.txt
└── README.md
