from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backtest_engine import objective_score_from_stats, run_backtest
from ml_pipeline import (
    BASE_FEATURES,
    assess_model,
    backward_feature_selection,
    evaluate_clustering_suite,
    evaluate_outlier_suite,
    evaluate_time_series_suite,
    get_hyperparameter_grid,
    get_model_registry,
    prepare_data,
)
from regime_extensions import attach_regime_labels, regime_summary
from deep_extensions import evaluate_lstm_models
from shap_extensions import compute_shap_importance

st.set_page_config(page_title="Yahoo Finance ML Trading Simulator", layout="wide")
st.title("Yahoo Finance ML Trading Simulator")
st.caption(
    "Pick a ticker and click Run simulation. The app automatically runs the full model suite, "
    "assesses predictive quality, optimizes trading thresholds, applies backward feature selection "
    "to the best model, and ranks the results."
)


def make_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str)
    return out


def metric_card_values(best_row: pd.Series) -> tuple[str, str, str, str]:
    best_model = str(best_row.get("model_name", "N/A"))
    best_style = str(best_row.get("training_style", "N/A"))
    best_return = f"{float(best_row.get('return_pct', float('nan'))):.2f}%"
    best_sharpe = f"{float(best_row.get('sharpe_ratio', float('nan'))):.4f}"
    return best_model, best_style, best_return, best_sharpe


def run_full_suite(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    ranking_objective: str,
):
    registry = get_model_registry()

    training_styles = [
        "static",
        "walk_forward_anchored",
        "walk_forward_unanchored",
    ]
    n_train_grid = [300, 500, 700]

    leaderboard_rows = []
    details = {}

    for task, model_dict in registry.items():
        for model_name in model_dict.keys():
            for training_style in training_styles:
                best_bundle = None
                best_score = float("-inf")
                search_rows = []

                for n_train in n_train_grid:
                    for model_params in get_hyperparameter_grid(task, model_name):
                        try:
                            prepared = prepare_data(
                                ticker=ticker,
                                start=start,
                                end=end,
                                interval=interval,
                                n_train=n_train,
                                task=task,
                                model_name=model_name,
                                training_style=training_style,
                                model_params=model_params,
                            )

                            assess = assess_model(prepared)
                            bt, stats, heatmap = run_backtest(
                                prepared=prepared,
                                optimize=True,
                                objective=ranking_objective,
                            )

                            score = objective_score_from_stats(stats, ranking_objective)

                            row = {
                                "task": task,
                                "model_name": model_name,
                                "training_style": training_style,
                                "n_train": n_train,
                                "model_params": str(model_params),
                                "objective_score": score,
                                "return_pct": float(stats.get("Return [%]", float("nan"))),
                                "sharpe_ratio": float(stats.get("Sharpe Ratio", float("nan"))),
                                "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", float("nan"))),
                                "win_rate_pct": float(stats.get("Win Rate [%]", float("nan"))),
                                "n_trades": float(stats.get("# Trades", float("nan"))),
                            }

                            row.update(assess)

                            strategy_obj = stats.get("_strategy")
                            if task == "classification":
                                row["prob_buy"] = getattr(strategy_obj, "prob_buy", None)
                                row["prob_sell"] = getattr(strategy_obj, "prob_sell", None)
                            else:
                                row["limit_buy"] = getattr(strategy_obj, "limit_buy", None)
                                row["limit_sell"] = getattr(strategy_obj, "limit_sell", None)

                            search_rows.append(row)

                            if score > best_score:
                                best_score = score
                                best_bundle = {
                                    "prepared": prepared,
                                    "assess": assess,
                                    "bt": bt,
                                    "stats": stats,
                                    "heatmap": heatmap,
                                    "best_row": row,
                                    "search_df": None,
                                }

                        except Exception as e:
                            search_rows.append(
                                {
                                    "task": task,
                                    "model_name": model_name,
                                    "training_style": training_style,
                                    "n_train": n_train,
                                    "model_params": str(model_params),
                                    "objective_score": float("nan"),
                                    "error": str(e),
                                }
                            )

                search_df = pd.DataFrame(search_rows).sort_values(
                    by="objective_score",
                    ascending=False,
                    na_position="last",
                ).reset_index(drop=True)

                if best_bundle is not None:
                    best_bundle["search_df"] = search_df
                    key = f"{task} | {model_name} | {training_style}"
                    details[key] = best_bundle
                    leaderboard_rows.append(best_bundle["best_row"])

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by="objective_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    raw = prepare_data(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
        n_train=300,
        task="classification",
        model_name="logistic_regression",
        training_style="static",
        model_params={},
    ).df.copy()

    ts_df = evaluate_time_series_suite(raw)
    cluster_df, clustered_data = evaluate_clustering_suite(raw, BASE_FEATURES)
    outlier_df = evaluate_outlier_suite(raw, BASE_FEATURES)

    return {
        "leaderboard": leaderboard,
        "details": details,
        "time_series_df": ts_df,
        "cluster_df": cluster_df,
        "outlier_df": outlier_df,
        "clustered_data": clustered_data,
        "raw_df": raw,
    }


with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Yahoo Finance ticker", value="MSFT").strip().upper()
    start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    ranking_objective = st.selectbox(
        "Ranking objective",
        ["Balanced", "Sharpe Ratio", "Return [%]"],
        index=0,
    )

    st.markdown("---")
    st.subheader("Extensions")
    enable_regime_extension = st.checkbox("Enable regime-aware diagnostics", value=True)
    enable_dl_extension = st.checkbox("Enable deep learning extension", value=False)
    enable_shap_extension = st.checkbox("Enable SHAP explainability", value=True)

    run_btn = st.button("Run simulation", type="primary")


if run_btn:
    try:
        results = run_full_suite(
            ticker=ticker,
            start=str(start),
            end=str(end),
            interval=interval,
            ranking_objective=ranking_objective,
        )

        leaderboard = results["leaderboard"]
        details = results["details"]
        time_series_df = results["time_series_df"]
        cluster_df = results["cluster_df"]
        outlier_df = results["outlier_df"]
        clustered_data = results["clustered_data"]
        raw_df = results["raw_df"]

        if leaderboard.empty:
            st.error("No backtestable models completed successfully.")
            st.stop()

        if enable_regime_extension:
            raw_df = attach_regime_labels(raw_df)
            regime_df = regime_summary(raw_df)
        else:
            regime_df = pd.DataFrame()

        if enable_dl_extension:
            dl_df = evaluate_lstm_models(raw_df, BASE_FEATURES)
        else:
            dl_df = pd.DataFrame()

        best = leaderboard.iloc[0]
        best_model, best_style, best_return, best_sharpe = metric_card_values(best)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best model", best_model)
        k2.metric("Best style", best_style)
        k3.metric("Best return", best_return)
        k4.metric("Best Sharpe", best_sharpe)

        st.subheader("Automated leaderboard")
        st.dataframe(make_display_df(leaderboard), width="stretch")

        # Backward selection on the top-ranked model
        best_key = None
        for key, bundle in details.items():
            row = bundle["best_row"]
            if (
                row["task"] == best["task"]
                and row["model_name"] == best["model_name"]
                and row["training_style"] == best["training_style"]
                and str(row["model_params"]) == str(best["model_params"])
                and row["n_train"] == best["n_train"]
            ):
                best_key = key
                break

        shap_df = pd.DataFrame()

        if best_key is not None:
            best_bundle = details[best_key]
            best_prepared = best_bundle["prepared"]

            selected_features, selection_history = backward_feature_selection(best_prepared)

            refined_prepared = prepare_data(
                ticker=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                n_train=best_prepared.n_train,
                task=best_prepared.task,
                model_name=best_prepared.model_name,
                training_style=best_prepared.training_style,
                model_params=best_prepared.model_params,
                feature_columns=selected_features,
            )

            refined_assess = assess_model(refined_prepared)
            _, refined_stats, refined_heatmap = run_backtest(
                prepared=refined_prepared,
                optimize=True,
                objective=ranking_objective,
            )

            if enable_shap_extension:
                try:
                    X_best = refined_prepared.df[refined_prepared.feature_columns]
                    y_best = refined_prepared.df[refined_prepared.target_column]

                    model = refined_prepared.model
                    model.fit(X_best, y_best)

                    sample_n = min(len(X_best), 300)
                    shap_df = compute_shap_importance(
                        model,
                        X_best.sample(sample_n, random_state=42),
                    )
                except Exception:
                    shap_df = pd.DataFrame()

            st.markdown("---")
            st.subheader("Best model refinement with backward feature selection")

            c1, c2, c3 = st.columns(3)
            c1.metric("Original feature count", len(best_prepared.feature_columns))
            c2.metric("Selected feature count", len(selected_features))
            c3.metric(
                "Refined objective score",
                f"{objective_score_from_stats(refined_stats, ranking_objective):.4f}",
            )

            st.write("Selected features")
            st.dataframe(
                pd.DataFrame({"Selected Feature": selected_features}),
                width="stretch",
            )

            st.write("Backward selection history")
            st.dataframe(make_display_df(selection_history), width="stretch")

            st.write("Refined model assessment")
            refined_assess_df = pd.DataFrame(
                [{"Metric": k, "Value": str(v)} for k, v in refined_assess.items()]
            )
            st.dataframe(refined_assess_df, width="stretch")

            st.write("Refined backtest summary")
            if hasattr(refined_stats, "to_frame"):
                refined_summary = refined_stats.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"})
            else:
                refined_summary = pd.DataFrame(refined_stats, index=[0]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
            refined_summary["Value"] = refined_summary["Value"].astype(str)
            st.dataframe(refined_summary, width="stretch")

            if "_equity_curve" in refined_stats:
                eq = refined_stats["_equity_curve"].reset_index()
                y_col = "Equity" if "Equity" in eq.columns else eq.columns[-1]
                fig_eq = px.line(eq, x=eq.columns[0], y=y_col, title="Equity curve - refined best model")
                st.plotly_chart(fig_eq, width="stretch")

            if refined_heatmap is not None:
                st.write("Refined optimization heatmap results")
                heatmap_df = refined_heatmap.reset_index()
                sort_col = "Return [%]" if "Return [%]" in heatmap_df.columns else heatmap_df.columns[-1]
                heatmap_df = heatmap_df.sort_values(sort_col, ascending=False)
                st.dataframe(make_display_df(heatmap_df.head(20)), width="stretch")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "Backtestable Models",
                "Time Series",
                "Clustering / Regimes",
                "Outliers",
                "Regime Extension",
                "Deep Learning Extension",
                "SHAP Explainability",
            ]
        )

        with tab1:
            for name, bundle in details.items():
                st.markdown("---")
                st.subheader(name)

                prepared = bundle["prepared"]
                assess = bundle["assess"]
                stats = bundle["stats"]
                heatmap = bundle["heatmap"]
                best_row = bundle["best_row"]
                search_df = bundle["search_df"]

                c1, c2, c3, c4 = st.columns(4)
                if prepared.task == "classification":
                    c1.metric("CV Accuracy", f"{assess['cv_accuracy']:.2%}")
                    c2.metric("CV Precision", f"{assess['cv_precision']:.2%}")
                    c3.metric("CV Recall", f"{assess['cv_recall']:.2%}")
                    c4.metric("CV F1", f"{assess['cv_f1']:.2%}")
                else:
                    c1.metric("CV MSE", f"{assess['cv_mse']:.6f}")
                    c2.metric("CV RMSE", f"{assess['cv_rmse']:.6f}")
                    c3.metric("CV MAE", f"{assess['cv_mae']:.6f}")
                    c4.metric("CV R²", f"{assess['cv_r2']:.4f}")

                st.write("Best configuration")
                best_df = pd.DataFrame(
                    [{"Parameter": k, "Value": str(v)} for k, v in best_row.items()]
                )
                st.dataframe(best_df, width="stretch")

                st.write("Feature set used")
                st.dataframe(
                    pd.DataFrame({"Feature": prepared.feature_columns}),
                    width="stretch",
                )

                st.write("Backtest summary")
                if hasattr(stats, "to_frame"):
                    summary_df = stats.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"})
                else:
                    summary_df = pd.DataFrame(stats, index=[0]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
                summary_df["Value"] = summary_df["Value"].astype(str)
                st.dataframe(summary_df, width="stretch")

                if "_equity_curve" in stats:
                    eq = stats["_equity_curve"].reset_index()
                    y_col = "Equity" if "Equity" in eq.columns else eq.columns[-1]
                    fig_eq = px.line(eq, x=eq.columns[0], y=y_col, title=f"Equity curve - {name}")
                    st.plotly_chart(fig_eq, width="stretch")

                if heatmap is not None:
                    st.write("Optimization heatmap results")
                    heatmap_df = heatmap.reset_index()
                    sort_col = "Return [%]" if "Return [%]" in heatmap_df.columns else heatmap_df.columns[-1]
                    heatmap_df = heatmap_df.sort_values(sort_col, ascending=False)
                    st.dataframe(make_display_df(heatmap_df.head(20)), width="stretch")

                st.write("Search results for this model/style")
                st.dataframe(make_display_df(search_df.head(20)), width="stretch")

        with tab2:
            st.subheader("Time-series model assessment")
            if time_series_df.empty:
                st.info("No time-series models completed.")
            else:
                st.dataframe(make_display_df(time_series_df), width="stretch")

            fig_price = px.line(raw_df, x="Date", y="Close", title=f"{ticker} close price")
            st.plotly_chart(fig_price, width="stretch")

        with tab3:
            st.subheader("Clustering / market regime diagnostics")
            if cluster_df.empty:
                st.info("No clustering models completed.")
            else:
                st.dataframe(make_display_df(cluster_df), width="stretch")

            if "regime_cluster" in clustered_data.columns:
                fig_regime = px.scatter(
                    clustered_data.tail(300),
                    x="Date",
                    y="Close",
                    color=clustered_data.tail(300)["regime_cluster"].astype(str),
                    title="Recent market regime clusters",
                )
                st.plotly_chart(fig_regime, width="stretch")

        with tab4:
            st.subheader("Outlier / anomaly diagnostics")
            if outlier_df.empty:
                st.info("No outlier models completed.")
            else:
                st.dataframe(make_display_df(outlier_df), width="stretch")

        with tab5:
            st.subheader("Regime-aware extension")
            if not enable_regime_extension or regime_df.empty:
                st.info("Regime extension not enabled.")
            else:
                st.dataframe(make_display_df(regime_df), width="stretch")
                fig_regime_ext = px.scatter(
                    raw_df.tail(300),
                    x="Date",
                    y="Close",
                    color="Market_Regime",
                    title="Recent market regimes",
                )
                st.plotly_chart(fig_regime_ext, width="stretch")

        with tab6:
            st.subheader("Deep learning extension")
            if not enable_dl_extension or dl_df.empty:
                st.info("Deep learning extension not enabled or torch not installed.")
            else:
                st.dataframe(make_display_df(dl_df), width="stretch")

        with tab7:
            st.subheader("SHAP explainability")
            if not enable_shap_extension or shap_df.empty:
                st.info("SHAP extension not enabled or unavailable.")
            else:
                st.dataframe(make_display_df(shap_df.head(20)), width="stretch")
                fig_shap = px.bar(
                    shap_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top SHAP feature importances",
                )
                st.plotly_chart(fig_shap, width="stretch")

        st.markdown("---")
        st.subheader("How this automation works")
        st.markdown(
            """
            - Runs all classification and regression models automatically  
            - Runs each with static, walk-forward anchored, and walk-forward unanchored training  
            - Performs predictive model assessment for every model  
            - Optimizes backtest thresholds for every backtestable model  
            - Ranks all models using the selected objective  
            - Applies backward feature selection to the top-ranked model  
            - Reports the final selected feature subset  
            - Adds regime-aware diagnostics as an extension  
            - Adds deep learning evaluation as an extension  
            - Adds SHAP explainability for the refined best model  
            - Also reports time-series, clustering, and outlier diagnostics  
            """
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Choose a ticker and click Run simulation.")
