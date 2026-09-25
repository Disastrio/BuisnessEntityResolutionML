"""
validate.py — CV logic.
Implements all strategies from the Validation Strategy table.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
)


def get_cv(strategy: str = "stratified", n_splits: int = 5, **kwargs):
    """
    strategy options:
        'stratified'   → StratifiedKFold (classification)
        'kfold'        → KFold (regression)
        'timeseries'   → TimeSeriesSplit
        'group'        → GroupKFold (pass groups=... to split())
    """
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    elif strategy == "group":
        return GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")


def cv_score(model, X, y, cv, metric_fn, groups=None):
    """Run CV and return (mean, std) of scores."""
    scores = []
    for fold, (tr, val) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        s = metric_fn(y_val, preds)
        scores.append(s)
        print(f"  Fold {fold+1}: {s:.5f}")
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"  CV: {mean:.5f} ± {std:.5f}")
    return mean, std
