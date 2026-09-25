"""
features.py — Feature engineering pipelines.

v1  — Baseline (raw numeric, median impute, one-hot low-card cats)
v2  — Target encoding for all categoricals (CV, k=5, smoothing=10)
v3  — Lag / rolling features for time-series
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# ── v1: Baseline ─────────────────────────────────────────────────────────────
def v1_baseline(train: pd.DataFrame, test: pd.DataFrame, num_cols, cat_cols):
    """Median impute numerics; one-hot encode cats with < 10 levels."""
    # Impute
    for col in num_cols:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col]  = test[col].fillna(median)

    # One-hot encode low-cardinality cats
    low_card = [c for c in cat_cols if train[c].nunique() < 10]
    train = pd.get_dummies(train, columns=low_card, drop_first=True)
    test  = pd.get_dummies(test,  columns=low_card, drop_first=True)
    train, test = train.align(test, join="left", axis=1, fill_value=0)
    return train, test


# ── v2: Target Encoding ───────────────────────────────────────────────────────
def target_encode_cv(train: pd.DataFrame, col: str, target: str,
                     n_folds: int = 5, smoothing: float = 10) -> pd.Series:
    """Out-of-fold target encoding to avoid leakage."""
    from sklearn.model_selection import KFold
    global_mean = train[target].mean()
    encoded = pd.Series(np.nan, index=train.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train):
        stats = train.iloc[tr_idx].groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = train.iloc[val_idx][col].map(smooth).fillna(global_mean)
    return encoded


# ── v3: Lag / Rolling (time-series) ──────────────────────────────────────────
def add_lags(df: pd.DataFrame, col: str, lags=(7, 14, 28)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling(df: pd.DataFrame, col: str, windows=(7, 14, 28)):
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_roll_std_{w}"]  = df[col].shift(1).rolling(w).std()
    return df
