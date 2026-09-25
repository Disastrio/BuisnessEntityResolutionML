"""
Business Entity Resolution Pipeline — Amazon ML Challenge.

Modules:
    config      Paths, seeds, constants, thresholds, feature flags
    io          Streaming TSV ingestion, ground-truth & aligned sampling
    normalize   Name & address canonicalization
    blocking    Multi-pass candidate generation (name/phonetic/postal/numeric/address)
    features    Pairwise similarity feature engineering (28 features)
    train       LightGBM pair classifier, hard negatives & Optuna tuning
    evaluate    Macro-F0.5, threshold / per-source / singleton-barrier sweeps
    predict     Test inference, conservative rules & per-S1 output assembly
    pipeline    End-to-end deterministic execution CLI (train/predict/smoke)
"""
