"""
train.py — LightGBM Pairwise Classifier & Optuna Tuning for Business Entity Resolution.

Trains a binary classifier on (S1, S2/S3) candidate pairs:
    - Positive: true matches from ground truth
    - Negative: blocking candidates that are NOT true matches
    - Hard negatives: same name diff address, same address diff business

Supports:
    - Default LightGBM training with class weights
    - Optuna hyperparameter tuning (50-200 trials)
    - Train/validation split by S1 entity (no leakage)
    - Model serialization to models/ directory
"""
from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import SEED, MODELS_DIR, BETA
from src.evaluate import macro_fbeta, threshold_sweep, apply_threshold
from src.features import FEATURE_NAMES


# ═════════════════════════════════════════════════════════════════════════════
# Default LightGBM Parameters
# ═════════════════════════════════════════════════════════════════════════════

def default_lgbm_params() -> dict:
    """
    Conservative default params optimized for precision-heavy F0.5.
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.0,  # Will be adjusted based on class balance
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Train/Validation Split by S1 Entity
# ═════════════════════════════════════════════════════════════════════════════

def split_by_s1_entity(
    s1_ids: List[str],
    target_ids: List[str],
    features: pd.DataFrame,
    labels: np.ndarray,
    val_frac: float = 0.2,
    seed: int = SEED,
) -> Tuple:
    """
    Split data by S1 entity to avoid leakage.
    All pairs for a given S1 go into either train or val, never both.

    Returns:
        (X_train, X_val, y_train, y_val,
         s1_ids_train, s1_ids_val, target_ids_train, target_ids_val,
         val_s1_set)
    """
    rng = np.random.RandomState(seed)
    unique_s1 = sorted(set(s1_ids))
    rng.shuffle(unique_s1)

    n_val = max(1, int(len(unique_s1) * val_frac))
    val_s1_set = set(unique_s1[:n_val])
    train_s1_set = set(unique_s1[n_val:])

    train_mask = np.array([sid in train_s1_set for sid in s1_ids])
    val_mask = ~train_mask

    X_train = features.iloc[train_mask].reset_index(drop=True)
    X_val = features.iloc[val_mask].reset_index(drop=True)
    y_train = labels[train_mask]
    y_val = labels[val_mask]
    s1_train = [s1_ids[i] for i in range(len(s1_ids)) if train_mask[i]]
    s1_val = [s1_ids[i] for i in range(len(s1_ids)) if val_mask[i]]
    t_train = [target_ids[i] for i in range(len(target_ids)) if train_mask[i]]
    t_val = [target_ids[i] for i in range(len(target_ids)) if val_mask[i]]

    return (X_train, X_val, y_train, y_val,
            s1_train, s1_val, t_train, t_val, val_s1_set)


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    auto_class_weight: bool = True,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier for pair matching.

    Args:
        X_train: Training features.
        y_train: Binary labels (1=match, 0=no-match).
        X_val: Validation features (for early stopping).
        y_val: Validation labels.
        params: LightGBM parameters (defaults used if None).
        auto_class_weight: Auto-compute scale_pos_weight from class ratio.

    Returns:
        Trained LGBMClassifier.
    """
    if params is None:
        params = default_lgbm_params()
    else:
        params = {**default_lgbm_params(), **params}

    # Auto class weight: ratio of negatives to positives
    if auto_class_weight:
        n_pos = max(1, int(y_train.sum()))
        n_neg = len(y_train) - n_pos
        params['scale_pos_weight'] = n_neg / n_pos
        print(f"  Class balance: {n_pos} pos / {n_neg} neg "
              f"(scale_pos_weight={params['scale_pos_weight']:.2f})")

    model = lgb.LGBMClassifier(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['callbacks'] = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Optuna Hyperparameter Tuning
# ═════════════════════════════════════════════════════════════════════════════

def tune_lgbm_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    s1_ids_val: List[str],
    target_ids_val: List[str],
    ground_truth: Dict[str, Set[str]],
    n_trials: int = 100,
    beta: float = BETA,
) -> dict:
    """
    Optuna hyperparameter search optimizing macro F0.5 on validation set.

    Returns best params dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1,
        }

        # Auto class weight
        n_pos = max(1, int(y_train.sum()))
        n_neg = len(y_train) - n_pos
        params['scale_pos_weight'] = n_neg / n_pos

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ],
        )

        # Predict probabilities on validation
        probs = model.predict_proba(X_val)[:, 1]

        # Find best threshold
        best_thresh, best_score, _ = threshold_sweep(
            s1_ids_val, target_ids_val, probs, ground_truth,
            thresholds=[round(t, 2) for t in np.arange(0.40, 0.90, 0.05)],
            beta=beta,
        )

        return best_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best trial: {study.best_trial.number}")
    print(f"  Best macro F0.5: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


# ═════════════════════════════════════════════════════════════════════════════
# Model Persistence
# ═════════════════════════════════════════════════════════════════════════════

def save_model(
    model: lgb.LGBMClassifier,
    name: str = "lgbm_pair_classifier",
    threshold: float = 0.70,
    metadata: Optional[dict] = None,
) -> Path:
    """Save trained model and metadata to models/ directory."""
    model_path = MODELS_DIR / f"{name}.txt"
    model.booster_.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    # Save metadata
    meta = {
        'name': name,
        'threshold': threshold,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
    }
    if metadata:
        meta.update(metadata)

    meta_path = MODELS_DIR / f"{name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")

    return model_path


def load_model(name: str = "lgbm_pair_classifier") -> Tuple[lgb.LGBMClassifier, dict]:
    """Load model and metadata from models/ directory."""
    model_path = MODELS_DIR / f"{name}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    # Wrap in classifier for predict_proba
    model = lgb.LGBMClassifier()
    model._Booster = booster
    model._n_classes = 2
    model.fitted_ = True

    meta_path = MODELS_DIR / f"{name}_meta.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, metadata


# ═════════════════════════════════════════════════════════════════════════════
# Feature Importance
# ═════════════════════════════════════════════════════════════════════════════

def print_feature_importance(model: lgb.LGBMClassifier, top_n: int = 28) -> pd.DataFrame:
    """Print and return feature importance DataFrame."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        'feature': FEATURE_NAMES[:len(importance)],
        'importance': importance,
    }).sort_values('importance', ascending=False)

    print(f"\n  Top {min(top_n, len(fi))} features by importance:")
    for i, (_, row) in enumerate(fi.head(top_n).iterrows()):
        print(f"    {i+1:2d}. {row['feature']:30s} {row['importance']:6.0f}")

    return fi


if __name__ == '__main__':
    print("Train module loaded. Run via pipeline.py for full execution.")
    print(f"Default params: {default_lgbm_params()}")
