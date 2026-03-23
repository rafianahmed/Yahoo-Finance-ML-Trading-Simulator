cat > README.md << 'EOF'
# Yahoo Finance ML Trading Simulator

A fully automated machine learning trading research application built with Streamlit. The app downloads historical market data from Yahoo Finance, cleans and transforms it, engineers predictive features, trains multiple machine learning models, evaluates them using time-series validation, optimizes trading thresholds, backtests strategies, ranks all models, refines the best model using backward feature selection, and supports advanced extensions such as regime diagnostics, deep learning, and SHAP explainability.

The user only needs to:
1. Choose a ticker  
2. Click Run simulation

Everything else is automated.

---

## 6. Feature Engineering

The simulator builds a rich feature space.

### Price & Returns
- 1-day return  
- 5-day return  
- intraday range (%)  
- overnight gap (%)  

### Trend Features
- SMA(5), SMA(10)  
- SMA ratios  
- SMA slopes  

### Volatility
- 5, 10, 20-day volatility  

### Momentum Indicators
- RSI  
- MACD  
- MACD signal  
- MACD histogram  

### Lag Features
- lag returns (1,2,3)  

### Rolling Stats
- rolling mean / max / min  

### Volume Features
- volume change  
- volume ratio  

### Market Regime
- trend indicator (SMA5 > SMA10)

---

## 7. Target Construction

### Regression
Predicts:
- next-day return (%)

### Classification
Predicts:
- direction (up/down)

---

## 8. Models Used

### Regression Models
- Multiple Linear Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost (optional)  

### Classification Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost (optional)  

**Important:**  
Multiple Linear Regression uses multiple features, not a single variable.

---

## 9. Training Styles

### Static
Train once, no updates.

### Walk-Forward Anchored
Retrain using all past data.

### Walk-Forward Unanchored
Retrain using rolling recent window.

---

## 10. Model Evaluation

### Regression Metrics
- MSE  
- RMSE  
- MAE  
- R²  

### Classification Metrics
- Accuracy  
- Precision  
- Recall  
- F1  

Uses Time Series Cross Validation to avoid data leakage.

---

## 11. Strategy Generation

### Classification Strategy
- Buy if probability ≥ threshold  
- Sell if probability ≤ threshold  

### Regression Strategy
- Buy if predicted return ≥ threshold  
- Sell if predicted return ≤ threshold  

---

## 12. Optimization

### Hyperparameter Optimization
- depth  
- estimators  
- learning rate  

### Threshold Optimization
- classification: prob_buy / prob_sell  
- regression: limit_buy / limit_sell  

Uses:
- backtesting.py  
- sambo optimization  

---

## 13. Backtesting

Simulates trading performance.

### Metrics
- Return (%)  
- Sharpe Ratio  
- Max Drawdown  
- Win Rate  
- Trades  
- Equity Curve  

---

## 14. Model Ranking

### Balanced (Recommended)
Combines:
- Sharpe  
- Return  
- Drawdown penalty  
- Win rate  

### Other Options
- Sharpe only  
- Return only  

---

## 15. Backward Feature Selection

### Method
- start with all features  
- remove weakest feature  
- keep removal if performance improves  

### Purpose
- reduce overfitting  
- remove noise  
- improve stability  

### Output
- selected features  
- refined model  
- improved backtest  

---

## 16. Additional Diagnostics

### Time-Series Models
- naive  
- moving average  
- Prophet  

### Clustering
- KMeans (market regimes)  

### Outliers
- Isolation Forest  

---

## 17. Extensions (Additive Layer)

These do NOT replace the system — they extend it.

### Regime Diagnostics
- market regimes (trend + volatility)  
- regime visualization  

### Deep Learning
- LSTM sequence models  
- time-series prediction  

### SHAP Explainability
- feature importance  
- model interpretability  

---

## 18. Visualizations

- price chart  
- equity curve  
- leaderboard  
- optimization results  
- feature selection results  
- clustering plots  
- SHAP plots  

---

## 19. Outputs

After running simulation:

- best model  
- best strategy  
- best parameters  
- best feature set  
- full leaderboard  
- performance metrics  
- backtest results  
- visualizations  

---

## 20. Automation

The system automatically:

- downloads data  
- cleans data  
- engineers features  
- trains all models  
- evaluates models  
- optimizes strategies  
- backtests results  
- ranks models  
- refines best model  

User only selects ticker.

---

## 21. Project Structure

your-repo/
├── app.py  
├── ml_pipeline.py  
├── strategies.py  
├── backtest_engine.py  
├── regime_extensions.py  
├── deep_extensions.py  
├── shap_extensions.py  
├── requirements.txt  
├── runtime.txt  
└── README.md  

---

## Author

Rafian Ahmed Raad

---

## Disclaimer

This project is for educational purposes only.  
Not financial advice.
EOF
