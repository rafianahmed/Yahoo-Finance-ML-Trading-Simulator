from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backtest_engine import run_backtest
from ml_pipeline import prepare_data, time_series_scores

st.set_page_config(page_title="Yahoo Finance ML Trading Simulator", layout="wide")
st.title("Yahoo Finance ML Trading Simulator")
st.caption("Built from the uploaded notebook workflow: preprocessing, classification, regression, overfitting checks, parameter tuning, and anchored/unanchored walk-forward testing.")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Yahoo Finance ticker", value="MSFT")
    start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    n_train = st.slider("Initial training window", min_value=200, max_value=1000, value=600, step=50)
    reg_depth = st.slider("Regression tree depth", min_value=2, max_value=25, value=15)
    clf_depth = st.slider("Classification tree depth", min_value=2, max_value=25, value=8)
    strategy_name = st.selectbox(
        "Strategy",
        ["classification", "regression", "walk_forward_anchored", "walk_forward_unanchored"],
    )
    optimize = st.checkbox("Optimize thresholds", value=False)
    run_btn = st.button("Run simulation", type="primary")

if run_btn:
    try:
        prepared = prepare_data(
            ticker=ticker.strip().upper(),
            start=str(start),
            end=str(end),
            interval=interval,
            n_train=n_train,
            reg_depth=reg_depth,
            clf_depth=clf_depth,
        )

        scores = time_series_scores(prepared)
        bt, stats, heatmap = run_backtest(prepared, strategy_name=strategy_name, optimize=optimize)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows used", scores["rows_used"])
        k2.metric("CV regression MSE", round(scores["regression_cv_mse"], 4))
        k3.metric("CV classification accuracy", f"{scores['classification_cv_accuracy']:.2%}")
        k4.metric("Backtest return", f"{float(stats['Return [%]']):.2f}%")

        st.subheader("Prepared dataset preview")
        st.dataframe(prepared.df.tail(20), use_container_width=True)

        st.subheader("Price chart")
        fig_price = px.line(prepared.df, x="Date", y="Close", title=f"{ticker.upper()} close price")
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Backtest summary")
        summary_series = stats.to_frame(name="Value") if hasattr(stats, "to_frame") else pd.DataFrame(stats, index=[0]).T.rename(columns={0: "Value"})
        st.dataframe(summary_series, use_container_width=True)

        st.subheader("Equity curve")
        if "_equity_curve" in stats:
            eq = stats["_equity_curve"].reset_index()
            y_col = "Equity" if "Equity" in eq.columns else eq.columns[-1]
            fig_eq = px.line(eq, x=eq.columns[0], y=y_col, title="Equity curve")
            st.plotly_chart(fig_eq, use_container_width=True)

        if heatmap is not None:
            st.subheader("Optimization results")
            heatmap_df = heatmap.reset_index().sort_values("Return [%]", ascending=False)
            st.dataframe(heatmap_df.head(20), use_container_width=True)

        st.subheader("What this app integrates from your notebooks")
        st.markdown(
            """
            - Data preprocessing and next-day target creation
            - Decision tree **classification** and **regression** modeling
            - In-sample vs walk-forward style evaluation
            - Parameter optimization with `backtesting.py`
            - Anchored and unanchored retraining logic based on your `strategies.py`
            - Yahoo Finance ticker input so you can test any supported symbol
            """
        )
    except Exception as e:
        st.error(str(e))
else:
    st.info("Set a ticker and click **Run simulation**.")
