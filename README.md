# Yahoo Finance ML Trading Simulator

A fully automated machine learning trading simulation platform that integrates data collection, feature engineering, predictive modeling, walk-forward retraining, strategy optimization, backtesting, feature selection, and diagnostics into a single interactive system.

The user only needs to:

1. Select a ticker
2. Click **Run Simulation**

The system performs everything else automatically.

---

# 1. System Overview

This simulator is designed as an **end-to-end quantitative research pipeline** that mimics real-world ML-based trading workflows.

It combines:

* Financial data processing
* Predictive modeling (classification + regression)
* Time-series validation
* Strategy optimization
* Walk-forward retraining
* Backtesting simulation
* Feature selection
* Model comparison and ranking

---

# 2. End-to-End Pipeline

The simulator follows this workflow:

```text
Yahoo Finance Data
→ Data Cleaning
→ Feature Engineering
→ Target Construction
→ Model Training
→ Model Evaluation (CV)
→ Strategy Generation
→ Threshold Optimization
→ Backtesting
→ Model Ranking
→ Backward Feature Selection (Best Model)
→ Final Refined Results
```

---

# 3. Data Collection

Data is fetched from **Yahoo Finance** using:

* Ticker symbol (e.g., AAPL, MSFT, TSLA)
* Time range
* Interval (1d, 1wk, 1mo)

### Data fields used:

* Open
* High
* Low
* Close
* Volume
* Date

---

# 4. Data Cleaning & Preprocessing

The system performs:

* Data type conversion (numeric enforcement)
* Missing value handling (drop invalid rows)
* Date normalization and sorting
* Removal of infinite values
* Rolling window stabilization (removes early NaNs)

👉 Ensures the dataset is fully usable for ML models

---

# 5. Feature Engineering

The simulator builds a **multi-dimensional feature space** capturing:

---

## 5.1 Price & Return Features

* 1-day return
* 5-day return
* Intraday range (%)
* Overnight gap (%)
* Close position within daily range

---

## 5.2 Trend Features

* SMA (5, 10)
* SMA ratios
* SMA slopes (trend strength)

---

## 5.3 Volatility Features

* 5-day volatility
* 10-day volatility
* 20-day volatility

---

## 5.4 Momentum Indicators

* RSI (14)
* MACD
* MACD signal
* MACD histogram

---

## 5.5 Lag Features (Time Dependency)

* Lag return (1, 2, 3)

---

## 5.6 Rolling Statistics

* Rolling mean (5)
* Rolling max (5)
* Rolling min (5)

---

## 5.7 Volume Features

* Volume change (%)
* Volume moving average
* Volume ratio

---

## 5.8 Market Regime Feature

* Trend regime (SMA5 > SMA10)

---

# 6. Target Construction

Two prediction tasks are created:

---

## 6.1 Regression Target

Predicts:

* Next-day return (%)

---

## 6.2 Classification Target

Predicts:

* Direction (Up / Down)

---

# 7. Models Used

The simulator runs a **full model suite automatically**

---

## 7.1 Regression Models

* Multiple Linear Regression ✅
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor (optional)

### Important

This is **Multiple Linear Regression**, meaning:

* many features → one prediction
* not a single-variable model

---

## 7.2 Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier (optional)

---

# 8. Model Evaluation

Uses **Time Series Cross Validation**

---

## 8.1 Regression Metrics

* MSE
* RMSE
* MAE
* R²

---

## 8.2 Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 9. Training Styles

Each model is tested under:

---

## Static

* Train once, no updates

---

## Walk-Forward Anchored

* Retrain using all past data

---

## Walk-Forward Unanchored

* Retrain using rolling recent window

---

👉 This simulates real market adaptation

---

# 10. Strategy Generation

Predictions are converted into trading rules:

---

## Classification Strategy

* Buy if probability ≥ threshold
* Sell if probability ≤ threshold

---

## Regression Strategy

* Buy if predicted return ≥ threshold
* Sell if predicted return ≤ threshold

---

# 11. Optimization

The system performs **two levels of optimization**

---

## 11.1 Hyperparameter Optimization

Searches across:

* tree depth
* learning rate
* number of estimators
* regularization

---

## 11.2 Strategy Threshold Optimization

Classification:

* prob_buy
* prob_sell

Regression:

* limit_buy
* limit_sell

Uses:

* `backtesting.py` optimizer
* `sambo` optimization engine

---

# 12. Backtesting

Simulates trading using historical data.

---

## Metrics reported:

* Return (%)
* Sharpe Ratio
* Max Drawdown (%)
* Win Rate (%)
* Number of trades
* Equity curve

---

# 13. Ranking System

Models are ranked using:

---

## Balanced (Recommended)

Combines:

* Sharpe Ratio
* Return
* Drawdown penalty
* Win rate
* trade count penalty

---

## Sharpe Ratio

* Focuses on risk-adjusted performance

---

## Return (%)

* Focuses only on profit

---

# 14. Backward Feature Selection

After ranking:

👉 The best model is refined using **Backward Selection**

---

## Process

* Start with all features
* Remove one feature at a time
* Keep removal if performance improves
* Stop when no improvement

---

## Why it is used

* Removes noisy variables
* Reduces overfitting
* Improves stability
* Finds minimal optimal feature set

---

## Output

* Selected features
* Selection history
* Improved model performance
* Refined backtest

---

# 15. Additional Diagnostics

---

## Time-Series Models

* Naive prediction
* Moving average
* Prophet (optional)

Metrics:

* MSE, RMSE, MAE, MAPE

---

## Clustering (Market Regimes)

* KMeans
* Silhouette score

---

## Outlier Detection

* Isolation Forest
* Anomaly rate

---

# 16. Visualization Outputs

The app provides:

* Price chart
* Equity curves
* Optimization heatmaps
* Model leaderboard
* Feature selection history
* Clustering scatter plots
* Performance tables

---

# 17. What You Get (Final Output)

When you run the simulation, you receive:

* Best model
* Best training style
* Best hyperparameters
* Best trading thresholds
* Full leaderboard
* Model evaluation metrics
* Backtest performance
* Equity curve
* Feature importance (via selection)
* Final selected feature subset

---

# 18. Automation

The system automatically:

* runs all models
* evaluates all models
* optimizes all strategies
* compares all results
* ranks models
* selects best model
* refines best model

---

# 19. Project Structure

```text
your-repo/
├── app.py
├── ml_pipeline.py
├── strategies.py
├── backtest_engine.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 20. How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 21. Key Insights

* Prediction accuracy ≠ trading profitability
* Walk-forward testing is essential
* Feature engineering is critical
* Optimization improves execution
* Feature selection reduces noise

---

# 22. Limitations

* Not live trading
* No slippage modeling
* Yahoo Finance limitations
* Diagnostics are not trading signals

---

# 23. Future Improvements

* Portfolio optimization
* Deep learning models
* SHAP explainability
* Regime-based strategies
* Multi-asset trading

---

# 24. Summary

This project is a complete ML trading system that:

* builds advanced features
* trains multiple models
* applies realistic validation
* optimizes strategies
* evaluates performance
* selects best model
* refines it using feature selection

All fully automated.

---

# 25. Author

**Rafian Ahmed Raad**

---

# 26. Disclaimer

This project is for educational and research purposes only.
It is not financial advice.
