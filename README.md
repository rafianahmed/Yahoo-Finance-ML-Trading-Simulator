# Yahoo Finance ML Trading Simulator

A Streamlit-based machine learning trading simulator that downloads historical market data from Yahoo Finance, engineers predictive features, runs a full suite of models, performs model assessment, optimizes trading thresholds, and compares all results automatically.

This version is designed so that the user only needs to:

1. choose a Yahoo Finance ticker  
2. click **Run simulation**

The app then runs the full pipeline automatically.

---

## Overview

This project combines:

- data preprocessing
- feature engineering
- classification modeling
- regression modeling
- walk-forward retraining
- threshold optimization
- backtesting
- time-series diagnostics
- clustering / market regime analysis
- outlier / anomaly analysis

into one automated dashboard.

---

## What this simulator does

When you enter a ticker and run the app, it automatically:

- downloads OHLCV data from Yahoo Finance
- engineers next-day predictive features
- creates both regression and classification targets
- trains multiple machine learning models
- evaluates predictive performance for each model
- runs static, walk-forward anchored, and walk-forward unanchored training styles
- optimizes trading thresholds for all backtestable models
- backtests every model/strategy combination
- ranks all models using a selected objective
- reports time-series, clustering, and anomaly diagnostics

---

## Fully automated workflow

After selecting a ticker and clicking **Run simulation**, the app runs:

### Backtestable predictive model suite
For each supported model, the app tests:

- **static**
- **walk_forward_anchored**
- **walk_forward_unanchored**

This is done for both:

- **classification**
- **regression**

### Additional diagnostics
The app also evaluates:

- **time-series models**
- **clustering / market regime models**
- **outlier / anomaly models**

---

## Data source

Historical market data is pulled from **Yahoo Finance**.

Supported examples:

- `AAPL`
- `MSFT`
- `TSLA`
- `NVDA`
- `AMZN`
- `^GSPC`

Supported intervals:

- `1d`
- `1wk`
- `1mo`

---

## Feature engineering

The simulator transforms raw Yahoo Finance data into machine-learning-ready features.

### Base features used
- Open
- High
- Low
- Close
- Volume

### Engineered features
- 1-day return
- 5-day return
- intraday price range percentage
- overnight gap percentage
- volume change percentage
- 5-day simple moving average ratio
- 10-day simple moving average ratio
- 5-day rolling volatility

These features are used to predict the next trading day.

---

## Targets

### Regression target
- `change_tomorrow`
- predicts the **next-day percentage return**

### Classification target
- `change_tomorrow_direction`
- predicts whether the **next-day movement is up or down**

---

## Models included

## 1. Classification models

These models predict **direction**: whether the next day will go up or down.

Supported classification models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier *(if installed)*

### Role
Classification models are used as directional signal generators.

### Assessment metrics
The app reports:

- CV Accuracy
- CV Precision
- CV Recall
- CV F1
- CV ROC-AUC *(when probability output is available)*

### Optimization
For each classification model, the app optimizes:

- `prob_buy`
- `prob_sell`

### Trading logic
- Buy when probability of upward movement is above `prob_buy`
- Close position when probability falls below `prob_sell`

---

## 2. Regression models

These models predict **magnitude**: how much the next day is expected to move.

Supported regression models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor *(if installed)*

### Role
Regression models are used as return forecasters and signal generators.

### Assessment metrics
The app reports:

- CV MSE
- CV RMSE
- CV MAE
- CV R²

### Optimization
For each regression model, the app optimizes:

- `limit_buy`
- `limit_sell`

### Trading logic
- Buy when predicted return is above `limit_buy`
- Close position when predicted return is below `limit_sell`

---

## Training styles

Each backtestable model is evaluated under multiple training styles.

### 1. Static
The model is trained once on the initial training window and then used without retraining.

**Use case:**  
baseline comparison

---

### 2. Walk-forward anchored
The model is retrained over time using **all available historical data up to the current point**.

### Example
If the app is at day 1200:

- training uses day 1 through day 1200

### Pros
- uses the full available history
- more stable
- lower variance

### Cons
- slower to adapt to regime changes
- may retain outdated market behavior

---

### 3. Walk-forward unanchored
The model is retrained over time using only the **most recent rolling training window**.

### Example
If the training window is 600 and the app is at day 1200:

- training uses only the last 600 rows

### Pros
- more adaptive
- better for changing market regimes
- often more realistic in non-stationary financial data

### Cons
- less historical context
- noisier estimates

---

## Time-series diagnostics

The app also evaluates time-series forecasting baselines.

### Included time-series models
- Naive last-close forecast
- Rolling MA(5) forecast
- Prophet *(if installed)*

### Purpose
These models are used to assess direct forecasting quality for price series.

### Metrics
The app reports:

- MSE
- RMSE
- MAE
- MAPE

These are diagnostic forecasting models and are not the main threshold-optimized backtesting engines.

---

## Clustering / market regime diagnostics

The app evaluates clustering models to identify market regimes.

### Included clustering model
- K-Means

### Purpose
Used to identify latent market states such as:
- trending
- range-bound
- high-volatility
- low-volatility

### Metrics
The app reports:

- silhouette score

### Use in the system
This is a diagnostic / regime analysis tool rather than a direct threshold-based buy/sell model.

---

## Outlier / anomaly diagnostics

The app also evaluates outlier detection to identify abnormal market conditions.

### Included outlier model
- Isolation Forest

### Purpose
Used to flag unusual or anomalous periods that may correspond to:
- volatility spikes
- abnormal returns
- unstable market conditions

### Metric
The app reports:

- anomaly rate percentage

This is diagnostic and can be extended later for risk filtering.

---

## What gets optimized

The app performs optimization at two levels.

## 1. Model-level search
For each backtestable model, the app searches across:

- training window size (`n_train`)
- model hyperparameters

Examples:
- logistic regression `C`
- decision tree `max_depth`
- random forest `n_estimators`, `max_depth`
- gradient boosting `learning_rate`, `max_depth`
- xgboost `learning_rate`, `max_depth`, `n_estimators`

### Goal
Improve predictive performance and downstream trading performance.

---

## 2. Strategy-level threshold optimization
After the model is trained, the app optimizes trading thresholds.

### Classification thresholds
- `prob_buy`
- `prob_sell`

### Regression thresholds
- `limit_buy`
- `limit_sell`

### Goal
Maximize trading performance under backtesting.

---

## Model assessment vs trading optimization

This is one of the most important concepts in the project.

### Model assessment
Measures how well the predictive model performs on unseen time-series folds.

Examples:
- classification accuracy
- regression MSE

### Trading optimization
Measures how well the predictions translate into profitable trading decisions.

Examples:
- Sharpe ratio
- Return %
- drawdown

These are different.
A model with lower error does not always produce the best trading strategy, and a profitable strategy does not always come from the model with the lowest raw prediction error.

---

## Backtesting metrics reported

For every backtestable model/strategy combination, the app reports:

- Return [%]
- Sharpe Ratio
- Max Drawdown [%]
- Win Rate [%]
- Number of trades
- Equity curve

---

## Ranking objectives

The app can rank the full model suite using one of three objectives.

### 1. Return [%]
Ranks models by raw profit only.

**Use when:**  
you want the most aggressive profit-first ranking.

---

### 2. Sharpe Ratio
Ranks models by risk-adjusted performance.

**Use when:**  
you want smoother and more realistic strategy selection.

---

### 3. Balanced
A custom combined score that uses:

- Sharpe ratio
- Return %
- drawdown penalty
- win rate
- trade count penalty

### Purpose
This is designed to avoid selecting unrealistic or unstable high-return strategies.

Balanced is usually the recommended default for overall ranking.

---

## What the app shows

After running the simulation, the app displays:

### Automated leaderboard
A ranked comparison of all model/strategy combinations.

### Per-model details
For each model and training style:
- predictive assessment metrics
- optimized parameters
- backtest summary
- optimization search results
- equity curve

### Time-series diagnostics
Forecast accuracy results for time-series models.

### Clustering diagnostics
Silhouette scores and recent market regime plots.

### Outlier diagnostics
Anomaly rates for abnormal market behavior.

---

## Project structure

```text
your-repo/
├── app.py
├── ml_pipeline.py
├── strategies.py
├── backtest_engine.py
├── requirements.txt
├── runtime.txt
└── README.md
