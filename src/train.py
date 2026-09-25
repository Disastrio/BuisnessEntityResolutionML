"""
train.py — Training pipeline.
Follows the Model Progression from hackathon_basics Section 4.2.
"""
import lightgbm as lgb
import xgboost  as xgb
import catboost as cb
import optuna
import numpy as np
import pandas as pd
from validate import get_cv, cv_score
from config  import SEED, N_FOLDS, TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── LightGBM default ─────────────────────────────────────────────────────────
def lgbm_default_params():
    return dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )


# ── Optuna tuning ─────────────────────────────────────────────────────────────
def tune_lgbm(X, y, metric_fn, n_trials=100, cv_strategy="stratified"):
    cv = get_cv(cv_strategy, N_FOLDS)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 2000),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 20, 300),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state      = SEED,
            n_jobs            = -1,
        )
        model = lgb.LGBMRegressor(**params)
        mean, _ = cv_score(model, X, y, cv, metric_fn)
        return mean

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest params: {study.best_params}")
    return study.best_params


# ── Ensemble / blend ──────────────────────────────────────────────────────────
def blend_oof(oof_preds: dict, y, weights=None):
    """Weighted average of OOF predictions. Weights default to equal."""
    names = list(oof_preds.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    blend = sum(oof_preds[n] * weights[n] for n in names)
    return blend
