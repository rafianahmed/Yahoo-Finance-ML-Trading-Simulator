from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def compute_shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    if not SHAP_AVAILABLE:
        return pd.DataFrame(
            [{"feature": "SHAP not installed", "importance": np.nan}]
        )

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                vals = np.abs(shap_values[1]).mean(axis=0)
            else:
                vals = np.abs(shap_values).mean(axis=0)

        elif hasattr(model, "coef_"):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            vals = np.abs(shap_values.values).mean(axis=0)

        else:
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X)
            vals = np.abs(shap_values.values).mean(axis=0)

        out = pd.DataFrame({"feature": X.columns, "importance": vals})
        out = out.sort_values("importance", ascending=False).reset_index(drop=True)
        return out
    except Exception as e:
        return pd.DataFrame([{"feature": f"SHAP failed: {e}", "importance": np.nan}])
