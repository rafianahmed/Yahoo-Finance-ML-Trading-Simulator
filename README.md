# Yahoo Finance ML Trading Simulator

This project turns your uploaded notebook workflow into a reusable simulation app.

## What it covers
- Yahoo Finance data download for any valid ticker
- Preprocessing and next-day target construction
- Decision Tree regression and classification models
- Backtesting with `backtesting.py`
- Walk-forward anchored and unanchored retraining
- Optional threshold optimization
- Streamlit UI for running experiments interactively

## Files
- `app.py` - Streamlit interface
- `ml_pipeline.py` - data download, feature engineering, model prep, CV scoring
- `strategies.py` - regression, classification, anchored, and unanchored strategies
- `backtest_engine.py` - backtest/optimization runner
- `requirements.txt` - dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
This app is based on the patterns clearly visible in your notebooks:
- `01A_Data Preprocessing.ipynb`
- `02A_Machine Learning Classification Model.ipynb`
- `03A_Backtesting ML Classification-Based.ipynb`
- `04B_Machine Learning Regression Model.ipynb`
- `07A_Optimizing Strategy Parameters.ipynb`
- `08A_Smart Optimization to Save Computing Time.ipynb`
- `09A_The Overfitting Problem.ipynb`
- `10A_Walk Forward Regression.ipynb`

Some notebook cells contained placeholders like `???`, so those parts were completed conservatively using the structure already present in the uploaded files.
