from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backtest_engine import (
    auto_optimize_model_and_strategy,
    auto_select_best_strategy,
    run_backtest,
)
from ml_pipeline import prepare_data, time_series_scores

st.set_page_config(page_title="Yahoo Finance ML Trading Simulator", layout="wide")
st.title("Yahoo Finance ML Trading Simulator")
st.caption(
    "Built from the uploaded notebook workflow: preprocessing, classification, "
    "regression, overfitting checks, parameter tuning, and anchored/unanchored "
    "walk-forward testing."
)


def make_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str)
    return out


with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Yahoo Finance ticker", value="MSFT").strip().upper()
    start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    mode = st.radio(
        "Mode",
        ["Manual", "Auto-optimize selected strategy", "Auto-select best strategy"],
        index=0,
    )

    objective = st.selectbox(
        "Optimization target",
        ["Balanced", "Sharpe Ratio", "Return [%]"],
        index=0,
    )

    if mode == "Manual":
        strategy_name = st.selectbox(
            "Strategy",
            ["classification", "regression", "walk_forward_anchored", "walk_forward_unanchored"],
        )
        n_train = st.slider("Initial training window", min_value=200, max_value=1000, value=600, step=50)
        reg_depth = st.slider("Regression tree depth", min_value=2, max_value=25, value=15)
        clf_depth = st.slider("Classification tree depth", min_value=2, max_value=25, value=8)
        optimize = st.checkbox("Optimize thresholds", value=False)

    elif mode == "Auto-optimize selected strategy":
        strategy_name = st.selectbox(
            "Strategy",
            ["classification", "regression", "walk_forward_anchored", "walk_forward_unanchored"],
        )
        st.info("Training window, tree depth, and thresholds will be selected automatically.")
        n_train = None
        reg_depth = None
        clf_depth = None
        optimize = False

    else:
        strategy_name = None
        st.info("The app will test all strategies and choose the best one automatically.")
        n_train = None
        reg_depth = None
        clf_depth = None
        optimize = False

    run_btn = st.button("Run simulation", type="primary")


if run_btn:
    try:
        chosen_strategy_name = strategy_name

        if mode == "Auto-select best strategy":
            bundle = auto_select_best_strategy(
                prepare_data_fn=prepare_data,
                ticker=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                cash=10000,
                commission=0.002,
                objective=objective,
            )

            prepared = bundle["best_prepared"]
            bt = bundle["best_bt"]
            stats = bundle["best_stats"]
            heatmap = bundle["best_heatmap"]
            best_params = bundle["best_params"]
            strategy_ranking = bundle["strategy_ranking"]
            chosen_strategy_name = bundle["best_strategy_name"]

            st.success(f"Best strategy selected automatically: {chosen_strategy_name}")

            st.subheader("Strategy ranking")
            st.dataframe(make_display_df(strategy_ranking), width="stretch")

            st.subheader("Chosen best parameter set")
            best_params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in best_params.items()]
            )
            st.dataframe(best_params_df, width="stretch")

        elif mode == "Auto-optimize selected strategy":
            bundle = auto_optimize_model_and_strategy(
                prepare_data_fn=prepare_data,
                ticker=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                strategy_name=strategy_name,
                cash=10000,
                commission=0.002,
                objective=objective,
            )

            prepared = bundle["best_prepared"]
            bt = bundle["best_bt"]
            stats = bundle["best_stats"]
            heatmap = bundle["best_heatmap"]
            best_params = bundle["best_params"]
            search_results = bundle["search_results"]

            st.success(f"Auto-optimization complete for: {strategy_name}")

            st.subheader("Best parameter set")
            best_params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in best_params.items()]
            )
            st.dataframe(best_params_df, width="stretch")

            st.subheader("Search leaderboard")
            st.dataframe(make_display_df(search_results.head(20)), width="stretch")

        else:
            prepared = prepare_data(
                ticker=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                n_train=n_train,
                reg_depth=reg_depth,
                clf_depth=clf_depth,
            )

            bt, stats, heatmap = run_backtest(
                prepared=prepared,
                strategy_name=strategy_name,
                optimize=optimize,
                objective=objective,
            )

        scores = time_series_scores(prepared)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows used", scores["rows_used"])
        k2.metric("CV regression MSE", round(scores["regression_cv_mse"], 4))
        k3.metric("CV classification accuracy", f"{scores['classification_cv_accuracy']:.2%}")
        k4.metric("Backtest return", f"{float(stats['Return [%]']):.2f}%")

        st.subheader("Selected strategy")
        st.write(chosen_strategy_name)

        st.subheader("Prepared dataset preview")
        st.dataframe(prepared.df.tail(20), width="stretch")

        st.subheader("Price chart")
        fig_price = px.line(prepared.df, x="Date", y="Close", title=f"{ticker} close price")
        st.plotly_chart(fig_price, width="stretch")

        st.subheader("Cross-validation scores")
        scores_df = pd.DataFrame(
            [{"Metric": k, "Value": str(v)} for k, v in scores.items()]
        )
        st.dataframe(scores_df, width="stretch")

        st.subheader("Backtest summary")
        if hasattr(stats, "to_frame"):
            summary_df = stats.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"})
        else:
            summary_df = pd.DataFrame(stats, index=[0]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.dataframe(summary_df, width="stretch")

        st.subheader("Equity curve")
        if "_equity_curve" in stats:
            eq = stats["_equity_curve"].reset_index()
            y_col = "Equity" if "Equity" in eq.columns else eq.columns[-1]
            fig_eq = px.line(eq, x=eq.columns[0], y=y_col, title="Equity curve")
            st.plotly_chart(fig_eq, width="stretch")

        if heatmap is not None:
            st.subheader("Optimization results")
            heatmap_df = heatmap.reset_index()
            sort_col = "Return [%]" if "Return [%]" in heatmap_df.columns else heatmap_df.columns[-1]
            heatmap_df = heatmap_df.sort_values(sort_col, ascending=False)
            st.dataframe(make_display_df(heatmap_df.head(20)), width="stretch")

        st.subheader("How the automatic selection works")
        st.markdown(
            """
            - **Sharpe Ratio** favors stable, risk-adjusted performance
            - **Return [%]** favors raw profit
            - **Balanced** combines Sharpe, return, drawdown, win rate, and trade count
            - **Walk Forward Anchored** keeps all past data when retraining
            - **Walk Forward Unanchored** uses only the most recent rolling window
            """
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Set a ticker and click **Run simulation**.")
