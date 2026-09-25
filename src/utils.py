"""
utils.py — Shared helpers.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def plot_feature_importance(model, feature_names, top_n=30, title="Feature Importance"):
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi.nlargest(top_n).plot(kind="barh", figsize=(8, 10))
    plt.title(title)
    plt.tight_layout()
    plt.show()


def shap_summary(model, X_sample, max_display=20):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    shap.summary_plot(shap_vals, X_sample, max_display=max_display)


def is_noise(new_score, best_score, std, direction="lower_is_better"):
    """Return True if improvement is within 1 std (likely noise)."""
    delta = best_score - new_score if direction == "lower_is_better" else new_score - best_score
    is_n = delta < std
    print(f"Delta={delta:.5f}, Std={std:.5f} → {'NOISE' if is_n else 'REAL improvement'}")
    return is_n
