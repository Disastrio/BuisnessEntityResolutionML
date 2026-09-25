"""
Business Entity Resolution Pipeline — Amazon ML Challenge.

Modules:
    config      Paths, seeds, constants, thresholds
    io          TSV ingestion & ground-truth parsers
    normalize   Name & address canonicalization
    blocking    Multi-pass candidate generation (7-tier inverted index)
    features    Pairwise similarity feature engineering (~28 features)
    train       LightGBM pair classifier & Optuna tuning
    evaluate    Macro-F0.5 calculator, threshold sweep, singleton accuracy
    predict     Test inference & per-S1 thresholding
    pipeline    End-to-end deterministic execution CLI
"""
