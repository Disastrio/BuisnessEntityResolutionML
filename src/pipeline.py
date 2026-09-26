"""
pipeline.py — End-to-End Deterministic Execution CLI for Business Entity Resolution.

Usage:
    # Full training pipeline (on a sample for rapid iteration)
    python -m src.pipeline --mode train --sample 50000

    # Full training on all data
    python -m src.pipeline --mode train

    # Test inference (after model is trained)
    python -m src.pipeline --mode predict

    # Quick smoke test (10k sample)
    python -m src.pipeline --mode smoke

Phases:
    1. Load & Normalize data
    2. Generate candidates via blocking
    3. Compute pairwise features
    4. Train LightGBM classifier (train mode)
    5. Sweep thresholds for optimal F0.5
    6. Generate predictions and output files (predict mode)
"""
import argparse
import time
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    SEED, MODELS_DIR, ID_COL, DEFAULT_THRESH, MAX_CANDIDATES,
    MATCHING_OUT, CANDIDATE_OUT,
    PREDICT_CHUNK_SIZE, PREDICT_LIMIT_S1, FEATURE_WORKERS,
    USE_HARD_NEGATIVES, HARD_NEGATIVE_WEIGHT,
    USE_DUAL_MODELS, USE_SOURCE_THRESHOLDS, USE_CONSERVATIVE_RULES,
    SINGLETON_BARRIER, DEFAULT_MODEL_NAME, DUAL_MODEL_NAME,
)
from src.io import load_source, load_ground_truth, load_aligned_sample
from src.normalize import normalize_all_sources, benchmark_normalization
from src.blocking import (
    generate_all_candidates, evaluate_blocking,
)
from src.features import (
    build_feature_matrix, assign_labels, restrict_to_core_columns,
)
from src.train import (
    train_lgbm, split_by_s1_entity,
    save_model, load_model, print_feature_importance,
    compute_sample_weights, train_dual_models,
    save_dual_models, load_dual_models, predict_dual_probabilities,
)
from src.evaluate import (
    detailed_evaluation, threshold_sweep,
    assemble_predictions, threshold_sweep_by_source, sweep_singleton_barrier,
)
from src.predict import (
    predict_probabilities, run_chunked_inference, conservative_reject_mask,
)


def _print_header(msg: str):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def _elapsed(start: float) -> str:
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed/60:.1f}m"


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    sample_size: Optional[int] = None,
    dual_model: Optional[bool] = None,
    source_thresholds: Optional[bool] = None,
    use_rules: Optional[bool] = None,
    hard_negatives: Optional[bool] = None,
    barrier: Optional[float] = None,
    max_candidates: Optional[int] = None,
    workers: Optional[int] = None,
):
    """
    Full training pipeline:
    1. Load data
    2. Normalize
    3. Block
    4. Feature engineering
    5. Train LightGBM (with hard-negative upweighting / optional dual models)
    6. Tune threshold, per-source thresholds, conservative rules, barrier
    7. Save model + report

    Optional arguments override the corresponding `src.config` toggles for this
    run (used by the CLI flags).
    """
    global USE_DUAL_MODELS, USE_SOURCE_THRESHOLDS, USE_CONSERVATIVE_RULES
    global USE_HARD_NEGATIVES, SINGLETON_BARRIER, FEATURE_WORKERS
    if workers is not None:
        FEATURE_WORKERS = max(1, int(workers))
    if dual_model is not None:
        USE_DUAL_MODELS = dual_model
    if source_thresholds is not None:
        USE_SOURCE_THRESHOLDS = source_thresholds
    if use_rules is not None:
        USE_CONSERVATIVE_RULES = use_rules
    if hard_negatives is not None:
        USE_HARD_NEGATIVES = hard_negatives
    if barrier is not None:
        SINGLETON_BARRIER = barrier
    max_candidates = MAX_CANDIDATES if max_candidates is None else int(max_candidates)

    total_start = time.time()

    # ── Phase 1: Load Data ────────────────────────────────────────────────
    _print_header("Phase 1: Loading Training Data")
    t = time.time()

    if sample_size is not None:
        # Aligned sampling: ensures ground truth S2/S3 IDs exist in loaded data
        s1, s2, s3, gt = load_aligned_sample(n_s1=sample_size, seed=SEED)
    else:
        s1 = load_source(1, 'train')
        s2 = load_source(2, 'train')
        s3 = load_source(3, 'train')
        gt = load_ground_truth()

    print(f"  S1: {len(s1):,} records")
    print(f"  S2: {len(s2):,} records")
    print(f"  S3: {len(s3):,} records")
    print(f"  Ground truth: {len(gt):,} S1 entities")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 2: Normalize ────────────────────────────────────────────────
    _print_header("Phase 2: Normalizing Records")
    t = time.time()

    s1_norm, s2_norm, s3_norm = normalize_all_sources(
        s1, s2, s3, n_workers=FEATURE_WORKERS)

    # Quick benchmark
    bench = benchmark_normalization(s1)
    print(f"  Normalization speed: {bench['records_per_second']:,.0f} records/sec")
    print(f"  Countries found: {bench['unique_countries']}")
    print(f"  Has postal code: {bench['has_postal_pct']}%")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 3: Blocking / Candidate Generation ─────────────────────────
    _print_header("Phase 3: Candidate Generation (Blocking)")
    t = time.time()

    print(f"  Max candidates per S1: {max_candidates}")
    candidates = generate_all_candidates(
        s1_norm, s2_norm, s3_norm,
        max_candidates=max_candidates,
        show_progress=True,
        n_workers=FEATURE_WORKERS,
    )

    # Evaluate blocking quality
    blocking_metrics = evaluate_blocking(candidates, gt)
    print(f"\n  Blocking Results:")
    print(f"    Candidate Recall: {blocking_metrics['candidate_recall']:.4f} "
          f"({'✅ PASS' if blocking_metrics['gate_passed'] else '❌ FAIL — need >= 0.92'})")
    print(f"    Total candidates: {blocking_metrics['total_candidates']:,}")
    print(f"    Avg per S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")
    print(f"    Max per S1: {blocking_metrics['max_candidates_per_s1']}")
    print(f"    True matches found: {blocking_metrics['found_true_matches']:,} / {blocking_metrics['total_true_matches']:,}")
    print(f"    Singletons: {blocking_metrics['singletons']:,}")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 4: Pairwise Feature Engineering ────────────────────────────
    _print_header("Phase 4: Feature Engineering")
    t = time.time()

    # Combine S2 and S3 for feature lookup
    target_df = pd.concat([s2_norm, s3_norm], ignore_index=True)

    print(f"  Feature workers: {FEATURE_WORKERS}")
    features, s1_ids, target_ids = build_feature_matrix(
        s1_norm, target_df, candidates, show_progress=True,
        n_workers=FEATURE_WORKERS,
    )
    labels = assign_labels(s1_ids, target_ids, gt)

    # Halve feature memory; LightGBM trains on float32 natively. The candidate
    # dict and target frame are no longer needed once features are built.
    features = features.astype(np.float32)
    del candidates, target_df

    print(f"\n  Feature matrix: {features.shape}")
    print(f"  Positive pairs: {int(labels.sum()):,}")
    print(f"  Negative pairs: {int(len(labels) - labels.sum()):,}")
    print(f"  Positive ratio: {labels.mean():.4f}")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 5: Train/Val Split & Training ──────────────────────────────
    _print_header("Phase 5: Training LightGBM Pair Classifier")
    t = time.time()

    (X_train, X_val, y_train, y_val,
     s1_train, s1_val, t_train, t_val,
     val_s1_set) = split_by_s1_entity(
        s1_ids, target_ids, features, labels,
        val_frac=0.2, seed=SEED,
    )

    print(f"  Train: {len(X_train):,} pairs ({int(y_train.sum()):,} pos)")
    print(f"  Val:   {len(X_val):,} pairs ({int(y_val.sum()):,} pos)")
    print(f"  Val S1 entities: {len(val_s1_set):,}")

    # The unsplit frames are superseded by the train/val copies.
    del features, labels, s1_ids, target_ids

    # E4: hard-negative upweighting
    weights_train = compute_sample_weights(
        y_train, X_train, use_hard_negatives=USE_HARD_NEGATIVES)
    n_hard = int((weights_train > 1.0).sum())
    print(f"  [E4] Hard negatives upweighted: {n_hard:,} (x{HARD_NEGATIVE_WEIGHT})")

    dual_models = None
    if USE_DUAL_MODELS:
        print("  [E6] Training dual source-specific models (S2 / S3)...")
        dual_models = train_dual_models(
            X_train, y_train, t_train,
            X_val, y_val, t_val,
            weights_train=weights_train,
        )
        model = None
        for src, m in dual_models.items():
            print(f"\n  Feature importance (S1<->{src}):")
            print_feature_importance(m)
    else:
        model = train_lgbm(X_train, y_train, X_val, y_val,
                           sample_weight=weights_train)
        print_feature_importance(model)
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 6: Threshold / Rule Optimization ───────────────────────────
    _print_header("Phase 6: Threshold / Rule Optimization")
    t = time.time()

    if dual_models is not None:
        val_probs = predict_dual_probabilities(dual_models, X_val, t_val)
    else:
        val_probs = predict_probabilities(model, X_val)

    # Build val ground truth (only S1 entities in validation set)
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1_set}

    # E7: conservative rules (kept only when they improve validation F0.5)
    val_reject = None
    use_rules = False
    if USE_CONSERVATIVE_RULES:
        candidate_reject = conservative_reject_mask(X_val)
        _, score_without, _ = threshold_sweep(
            s1_val, t_val, val_probs, val_gt, n_workers=FEATURE_WORKERS)
        _, score_with, _ = threshold_sweep(
            s1_val, t_val, val_probs, val_gt, reject_mask=candidate_reject,
            n_workers=FEATURE_WORKERS)
        if score_with > score_without + 1e-9:
            val_reject = candidate_reject
            use_rules = True
            print(f"  [E7] Conservative rules KEPT "
                  f"({score_with:.6f} > {score_without:.6f})")
        else:
            print(f"  [E7] Conservative rules rejected "
                  f"({score_with:.6f} <= {score_without:.6f})")

    # E5: per-source thresholds vs a single global threshold
    thresholds_by_source = None
    if USE_SOURCE_THRESHOLDS:
        thresholds_by_source, best_score, _ = threshold_sweep_by_source(
            s1_val, t_val, val_probs, val_gt, reject_mask=val_reject)
        best_thresh = max(thresholds_by_source.values())
        print(f"  [E5] Per-source thresholds: {thresholds_by_source}")
    else:
        best_thresh, best_score, sweep = threshold_sweep(
            s1_val, t_val, val_probs, val_gt, reject_mask=val_reject,
            n_workers=FEATURE_WORKERS)
    print(f"  Best threshold: {best_thresh}")
    print(f"  Best macro F0.5: {best_score:.6f}")

    # E8: singleton confidence barrier (tuned; only improves the score)
    tuned_barrier, barrier_score, _ = sweep_singleton_barrier(
        s1_val, t_val, val_probs, val_gt,
        threshold=None if thresholds_by_source else best_thresh,
        thresholds_by_source=thresholds_by_source,
        reject_mask=val_reject,
        n_workers=FEATURE_WORKERS,
    )
    if tuned_barrier > 0:
        print(f"  [E8] Singleton barrier adopted: {tuned_barrier} "
              f"(F0.5 {barrier_score:.6f})")
        barrier = tuned_barrier
    else:
        barrier = SINGLETON_BARRIER

    # Detailed evaluation at the final tuned configuration
    val_preds = assemble_predictions(
        s1_val, t_val, val_probs,
        threshold=None if thresholds_by_source else best_thresh,
        thresholds_by_source=thresholds_by_source,
        all_s1_ids=val_s1_set,
        barrier=barrier,
        reject_mask=val_reject,
    )
    details = detailed_evaluation(val_gt, val_preds)
    print(f"\n  Detailed Metrics (t={best_thresh}, "
          f"rules={use_rules}, barrier={barrier}):")
    for k, v in details.items():
        print(f"    {k}: {v}")

    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 7: Save Model ──────────────────────────────────────────────
    _print_header("Phase 7: Saving Model")

    metadata = {
        'val_macro_f05': details['macro_fbeta'],
        'val_pair_precision': details['pair_precision'],
        'val_pair_recall': details['pair_recall'],
        'val_singleton_accuracy': details['singleton_accuracy'],
        'sample_size': sample_size,
        'best_threshold': best_thresh,
        'thresholds_by_source': thresholds_by_source,
        'barrier': barrier,
        'use_conservative_rules': use_rules,
        'use_hard_negatives': USE_HARD_NEGATIVES,
        'dual_model': dual_models is not None,
        'max_candidates': max_candidates,
        'tuned_macro_f05': best_score,
    }

    if dual_models is not None:
        save_dual_models(dual_models, name=DUAL_MODEL_NAME, metadata=metadata)
    else:
        save_model(model, threshold=best_thresh, metadata=metadata)

    total_elapsed = time.time() - total_start
    _print_header(f"Training Complete - Total time: {_elapsed(total_start)}")
    print(f"  Best Val Macro F0.5: {details['macro_fbeta']:.6f}")
    print(f"  Best Threshold: {best_thresh}")

    return (dual_models if dual_models is not None else model), best_thresh, details['macro_fbeta']


# ═════════════════════════════════════════════════════════════════════════════
# TEST INFERENCE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_prediction(
    threshold: Optional[float] = None,
    chunk_size: Optional[int] = None,
    limit_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    barrier: Optional[float] = None,
    workers: Optional[int] = None,
):
    """
    Full test inference pipeline (chunked / streaming):

    1. Load the most recently trained model
    2. Load + normalize test data, keeping only core columns
    3. Build blocking indices once
    4. Score S1 in chunks (candidates -> features -> probabilities -> output)
    5. Stream matching_results.tsv and candidate_pairs.tsv to disk
    6. Report lightweight integrity checks

    Memory scales with `chunk_size` (not the ~1.7M test S1 entities), so this
    runs within 16-32 GB instead of the 64-128 GB an all-in-memory pass needs.
    """
    global FEATURE_WORKERS
    if workers is not None:
        FEATURE_WORKERS = max(1, int(workers))

    total_start = time.time()
    chunk_size = chunk_size or PREDICT_CHUNK_SIZE
    limit_s1 = PREDICT_LIMIT_S1 if limit_s1 is None else limit_s1

    # ── Load Model ────────────────────────────────────────────────────────
    _print_header("Loading Model")
    # Choose the most recently trained artifact so a newer single-model run is
    # not shadowed by a stale dual-model bundle (or vice versa).
    dual_meta_path = MODELS_DIR / f"{DUAL_MODEL_NAME}_meta.json"
    single_meta_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}_meta.json"
    use_dual = dual_meta_path.exists() and (
        not single_meta_path.exists()
        or dual_meta_path.stat().st_mtime >= single_meta_path.stat().st_mtime
    )

    dual_models = None
    model = None
    if use_dual:
        dual_models, metadata = load_dual_models()
        print(f"  Loaded dual source-specific models: {sorted(dual_models.keys())}")

        def score_fn(feats, tids, _models=dual_models):
            return predict_dual_probabilities(_models, feats, tids)
    else:
        model, metadata = load_model()

        def score_fn(feats, tids, _model=model):
            return predict_probabilities(_model, feats)

    if threshold is None:
        threshold = metadata.get('best_threshold', DEFAULT_THRESH)
    thresholds_by_source = metadata.get('thresholds_by_source')
    if barrier is None:
        barrier = metadata.get('barrier', SINGLETON_BARRIER) or 0.0
    barrier = float(barrier)
    use_rules = metadata.get('use_conservative_rules', USE_CONSERVATIVE_RULES)
    if max_candidates is None:
        max_candidates = int(metadata.get('max_candidates', MAX_CANDIDATES))
    max_candidates = int(max_candidates)

    print(f"  Threshold: {threshold}")
    print(f"  Max candidates per S1: {max_candidates}")
    if thresholds_by_source:
        print(f"  Per-source thresholds: {thresholds_by_source}")
    if barrier:
        print(f"  Singleton barrier: {barrier}")
    if use_rules:
        print("  Conservative decision rules: ON")
    print(f"  Chunk size: {chunk_size:,} S1 entities")
    if limit_s1 is not None:
        print(f"  Limit: first {limit_s1:,} S1 entities")

    # ── Load Test Data ────────────────────────────────────────────────────
    _print_header("Loading Test Data")
    t = time.time()
    s1 = load_source(1, 'test')
    s2 = load_source(2, 'test')
    s3 = load_source(3, 'test')
    print(f"  S1: {len(s1):,} records")
    print(f"  S2: {len(s2):,} records")
    print(f"  S3: {len(s3):,} records")
    print(f"  {_elapsed(t)}")

    # ── Normalize (core columns only) ─────────────────────────────────────
    _print_header("Normalizing Test Data")
    t = time.time()
    s1_norm, s2_norm, s3_norm = normalize_all_sources(
        s1, s2, s3, n_workers=FEATURE_WORKERS)
    del s1, s2, s3
    s1_norm = restrict_to_core_columns(s1_norm)
    s2_norm = restrict_to_core_columns(s2_norm)
    s3_norm = restrict_to_core_columns(s3_norm)
    print(f"  {_elapsed(t)}")

    # ── Chunked Streaming Inference ───────────────────────────────────────
    _print_header("Streaming Inference (chunked)")
    t = time.time()
    summary = run_chunked_inference(
        s1_norm, s2_norm, s3_norm, score_fn,
        threshold=threshold,
        thresholds_by_source=thresholds_by_source,
        barrier=barrier,
        use_rules=use_rules,
        max_candidates=max_candidates,
        chunk_size=chunk_size,
        limit_s1=limit_s1,
        n_workers=FEATURE_WORKERS,
    )
    print(f"  {_elapsed(t)}")

    # ── Summary & Lightweight Validation ──────────────────────────────────
    _print_header("Prediction Summary")
    expected = len(s1_norm) if limit_s1 is None else min(limit_s1, len(s1_norm))
    print(f"  S1 rows written:       {summary['s1_written']:,} (expected {expected:,})")
    print(f"  Entities with matches: {summary['entities_with_matches']:,}")
    print(f"  Total matched pairs:   {summary['total_matches']:,}")
    print(f"  Avg candidates per S1: {summary['avg_candidates_per_s1']}")
    print(f"  Wrote {MATCHING_OUT}")
    print(f"  Wrote {CANDIDATE_OUT}")

    ok = True
    if summary['s1_written'] != expected:
        print(f"  [FAIL] expected {expected:,} rows, wrote {summary['s1_written']:,}")
        ok = False
    if summary['superset_violations']:
        print(f"  [FAIL] {summary['superset_violations']:,} rows have matches "
              f"missing from candidate_pairs.tsv")
        ok = False

    if limit_s1 is None:
        print("\n  Final gate (run before submitting):")
        print("    python utils/validate_submission.py "
              "--matching output/matching_results.tsv "
              "--candidate output/candidate_pairs.tsv --test-dir dataset/test")

    total_elapsed = time.time() - total_start
    _print_header(f"Prediction Complete - Total time: {_elapsed(total_start)}")
    if ok:
        print("  Streaming integrity checks passed.")
    else:
        print("  Integrity checks FAILED - review before submitting.")


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        '--mode', choices=['train', 'predict', 'smoke'],
        default='train',
        help='Pipeline mode: train, predict, or smoke (quick 10k test)'
    )
    parser.add_argument(
        '--sample', type=int, default=None,
        help='Number of rows to sample from each source (for rapid iteration)'
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Override decision threshold for prediction'
    )
    parser.add_argument(
        '--dual-model', action='store_true', default=None,
        help='E6: train separate S1<->S2 and S1<->S3 models'
    )
    parser.add_argument(
        '--source-thresholds', action='store_true', default=None,
        help='E5: tune separate decision thresholds for S2 and S3'
    )
    parser.add_argument(
        '--no-rules', action='store_true', default=None,
        help='E7: disable conservative decision rules'
    )
    parser.add_argument(
        '--no-hard-negatives', action='store_true', default=None,
        help='E4: disable hard-negative upweighting'
    )
    parser.add_argument(
        '--barrier', type=float, default=None,
        help='E8: singleton confidence barrier floor (tuned upward)'
    )
    parser.add_argument(
        '--max-candidates', type=int, default=None,
        help='Train: max candidate pairs retained per S1 (blocking recall vs cost)'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=None,
        help='Predict: S1 entities scored per streaming chunk (default 20000)'
    )
    parser.add_argument(
        '--limit-s1', type=int, default=None,
        help='Predict: score only the first N S1 entities (smoke / staged runs)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Process pool size for normalize/blocking/features/sweeps '
             '(default: CPU count; lower it if RAM is tight)'
    )
    args = parser.parse_args()

    train_kwargs = dict(
        dual_model=args.dual_model,
        source_thresholds=args.source_thresholds,
        use_rules=(False if args.no_rules else None),
        hard_negatives=(False if args.no_hard_negatives else None),
        barrier=args.barrier,
        max_candidates=args.max_candidates,
        workers=args.workers,
    )

    if args.mode == 'smoke':
        print("[SMOKE] Smoke Test Mode (10k sample)")
        run_training(sample_size=10000, **train_kwargs)

    elif args.mode == 'train':
        sample = args.sample
        if sample:
            print(f"[TRAIN] Training Mode (sample={sample:,})")
        else:
            print("[TRAIN] Training Mode (full data)")
        run_training(sample_size=sample, **train_kwargs)

    elif args.mode == 'predict':
        print("[PREDICT] Prediction Mode")
        run_prediction(
            threshold=args.threshold,
            chunk_size=args.chunk_size,
            limit_s1=args.limit_s1,
            max_candidates=args.max_candidates,
            barrier=args.barrier,
            workers=args.workers,
        )


if __name__ == '__main__':
    main()
