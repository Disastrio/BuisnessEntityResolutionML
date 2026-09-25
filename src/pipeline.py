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
    SEED, TRAIN_DIR, TEST_DIR,
    ID_COL, DEFAULT_THRESH, MAX_CANDIDATES,
)
from src.io import load_source, load_ground_truth, load_aligned_sample
from src.normalize import normalize_all_sources, benchmark_normalization
from src.blocking import (
    generate_all_candidates, evaluate_blocking,
)
from src.features import build_feature_matrix, assign_labels, FEATURE_NAMES
from src.train import (
    train_lgbm, split_by_s1_entity,
    save_model, load_model, print_feature_importance,
)
from src.evaluate import (
    macro_fbeta, detailed_evaluation, threshold_sweep, apply_threshold,
)
from src.predict import (
    predict_probabilities, assemble_matches, assemble_candidates,
    save_matching_results, save_candidate_pairs, validate_output,
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

def run_training(sample_size: Optional[int] = None):
    """
    Full training pipeline:
    1. Load data
    2. Normalize
    3. Block
    4. Feature engineering
    5. Train LightGBM
    6. Threshold sweep
    7. Save model + report
    """
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

    s1_norm, s2_norm, s3_norm = normalize_all_sources(s1, s2, s3)

    # Quick benchmark
    bench = benchmark_normalization(s1)
    print(f"  Normalization speed: {bench['records_per_second']:,.0f} records/sec")
    print(f"  Countries found: {bench['unique_countries']}")
    print(f"  Has postal code: {bench['has_postal_pct']}%")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 3: Blocking / Candidate Generation ─────────────────────────
    _print_header("Phase 3: Candidate Generation (Blocking)")
    t = time.time()

    candidates = generate_all_candidates(
        s1_norm, s2_norm, s3_norm,
        max_candidates=MAX_CANDIDATES,
        show_progress=True,
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

    features, s1_ids, target_ids = build_feature_matrix(
        s1_norm, target_df, candidates, show_progress=True,
    )
    labels = assign_labels(s1_ids, target_ids, gt)

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

    model = train_lgbm(X_train, y_train, X_val, y_val)

    # Feature importance
    print_feature_importance(model)
    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 6: Threshold Sweep ─────────────────────────────────────────
    _print_header("Phase 6: Threshold Optimization")
    t = time.time()

    val_probs = predict_probabilities(model, X_val)

    # Build val ground truth (only S1 entities in validation set)
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1_set}

    best_thresh, best_score, sweep = threshold_sweep(
        s1_val, t_val, val_probs, val_gt,
    )

    print(f"  Best threshold: {best_thresh}")
    print(f"  Best macro F0.5: {best_score:.6f}")

    # Detailed evaluation at best threshold
    val_preds = apply_threshold(s1_val, t_val, val_probs, best_thresh, val_s1_set)
    details = detailed_evaluation(val_gt, val_preds)
    print(f"\n  Detailed Metrics at t={best_thresh}:")
    for k, v in details.items():
        print(f"    {k}: {v}")

    print(f"  ⏱ {_elapsed(t)}")

    # ── Phase 7: Save Model ──────────────────────────────────────────────
    _print_header("Phase 7: Saving Model")

    save_model(model, threshold=best_thresh, metadata={
        'val_macro_f05': best_score,
        'val_pair_precision': details['pair_precision'],
        'val_pair_recall': details['pair_recall'],
        'val_singleton_accuracy': details['singleton_accuracy'],
        'sample_size': sample_size,
        'best_threshold': best_thresh,
    })

    total_elapsed = time.time() - total_start
    _print_header(f"Training Complete — Total time: {_elapsed(total_start)}")
    print(f"  Best Val Macro F0.5: {best_score:.6f}")
    print(f"  Best Threshold: {best_thresh}")

    return model, best_thresh, best_score


# ═════════════════════════════════════════════════════════════════════════════
# TEST INFERENCE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_prediction(threshold: Optional[float] = None):
    """
    Full test inference pipeline:
    1. Load model and test data
    2. Normalize test data
    3. Generate candidates
    4. Compute features
    5. Predict and threshold
    6. Save output files
    7. Validate
    """
    total_start = time.time()

    # ── Load Model ────────────────────────────────────────────────────────
    _print_header("Loading Model")
    model, metadata = load_model()
    if threshold is None:
        threshold = metadata.get('best_threshold', DEFAULT_THRESH)
    print(f"  Threshold: {threshold}")

    # ── Load Test Data ────────────────────────────────────────────────────
    _print_header("Loading Test Data")
    t = time.time()

    s1 = load_source(1, 'test')
    s2 = load_source(2, 'test')
    s3 = load_source(3, 'test')

    print(f"  S1: {len(s1):,} records")
    print(f"  S2: {len(s2):,} records")
    print(f"  S3: {len(s3):,} records")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Normalize ─────────────────────────────────────────────────────────
    _print_header("Normalizing Test Data")
    t = time.time()

    s1_norm, s2_norm, s3_norm = normalize_all_sources(s1, s2, s3)
    print(f"  ⏱ {_elapsed(t)}")

    # ── Block ─────────────────────────────────────────────────────────────
    _print_header("Candidate Generation (Blocking)")
    t = time.time()

    candidates = generate_all_candidates(
        s1_norm, s2_norm, s3_norm,
        max_candidates=MAX_CANDIDATES,
    )
    total_cands = sum(len(c) for c in candidates.values())
    print(f"  Total candidate pairs: {total_cands:,}")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Features ──────────────────────────────────────────────────────────
    _print_header("Computing Features")
    t = time.time()

    target_df = pd.concat([s2_norm, s3_norm], ignore_index=True)
    features, s1_ids, target_ids = build_feature_matrix(
        s1_norm, target_df, candidates,
    )
    print(f"  Feature matrix: {features.shape}")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Predict ───────────────────────────────────────────────────────────
    _print_header("Predicting")
    t = time.time()

    probs = predict_probabilities(model, features)
    all_s1_ids = set(s1[ID_COL])

    matches = assemble_matches(s1_ids, target_ids, probs, threshold, all_s1_ids)
    all_candidates = assemble_candidates(s1_ids, target_ids, all_s1_ids)

    print(f"  Threshold: {threshold}")
    print(f"  Entities with matches: {sum(1 for m in matches.values() if m):,}")
    print(f"  Total matched pairs: {sum(len(m) for m in matches.values()):,}")
    print(f"  ⏱ {_elapsed(t)}")

    # ── Save Output ───────────────────────────────────────────────────────
    _print_header("Saving Output Files")

    save_matching_results(matches)
    save_candidate_pairs(all_candidates)

    # ── Validate ──────────────────────────────────────────────────────────
    _print_header("Validating Output")

    test_s1_ids = set(s1[ID_COL])
    test_s2_ids = set(s2[ID_COL])
    test_s3_ids = set(s3[ID_COL])

    errors = validate_output(matches, all_candidates, test_s1_ids, test_s2_ids, test_s3_ids)

    total_elapsed = time.time() - total_start
    _print_header(f"Prediction Complete — Total time: {_elapsed(total_start)}")

    if not errors:
        print("  ✅ All checks passed. Ready for submission!")
    else:
        print(f"  ❌ {len(errors)} validation errors — fix before submitting.")


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
    args = parser.parse_args()

    if args.mode == 'smoke':
        print("[SMOKE] Smoke Test Mode (10k sample)")
        run_training(sample_size=10000)

    elif args.mode == 'train':
        sample = args.sample
        if sample:
            print(f"[TRAIN] Training Mode (sample={sample:,})")
        else:
            print("[TRAIN] Training Mode (full data)")
        run_training(sample_size=sample)

    elif args.mode == 'predict':
        print("[PREDICT] Prediction Mode")
        run_prediction(threshold=args.threshold)


if __name__ == '__main__':
    main()
