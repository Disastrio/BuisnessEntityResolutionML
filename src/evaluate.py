"""
evaluate.py — Macro F0.5 Calculator, Threshold Sweeper, and Singleton Accuracy.

Implements the exact scoring logic for the Amazon ML Challenge:
- Per-S1 entity F0.5 (β=0.5, precision-weighted)
- Singletons: correctly empty = 1.0, any false positive = 0.0
- Macro average across all S1 entities
- Threshold sweep to find optimal decision boundary
"""
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
from src.config import BETA


# ═════════════════════════════════════════════════════════════════════════════
# Per-Entity F-beta Score
# ═════════════════════════════════════════════════════════════════════════════

def entity_fbeta(
    true_matches: Set[str],
    predicted_matches: Set[str],
    beta: float = BETA,
) -> float:
    """
    Compute F-beta score for a single S1 entity.

    Special cases (per challenge specification):
    - Singleton (true_matches empty):
        - Predicted empty → 1.0 (correct)
        - Any prediction → 0.0 (false positive on singleton)
    - Non-singleton (true_matches non-empty):
        - Standard precision/recall/F-beta
    """
    # Singleton case
    if not true_matches:
        return 1.0 if not predicted_matches else 0.0

    # Non-singleton case
    if not predicted_matches:
        # Predicted empty but had true matches → recall = 0
        return 0.0

    tp = len(true_matches & predicted_matches)
    fp = len(predicted_matches - true_matches)
    fn = len(true_matches - predicted_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return fbeta


# ═════════════════════════════════════════════════════════════════════════════
# Macro F0.5 Score
# ═════════════════════════════════════════════════════════════════════════════

def macro_fbeta(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = BETA,
) -> float:
    """
    Compute macro-averaged F-beta across all S1 entities.

    Args:
        ground_truth: {s1_id: set(true_matched_ids)}
        predictions: {s1_id: set(predicted_matched_ids)}
        beta: Beta parameter (0.5 for this challenge)

    Returns:
        Macro-averaged F-beta score.
    """
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        score = entity_fbeta(true_matches, pred_matches, beta)
        scores.append(score)

    return np.mean(scores) if scores else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Detailed Evaluation Report
# ═════════════════════════════════════════════════════════════════════════════

def detailed_evaluation(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = BETA,
) -> dict:
    """
    Compute detailed evaluation metrics.

    Returns dict with:
    - macro_fbeta: Main metric
    - pair_precision, pair_recall: Aggregate pair-level metrics
    - singleton_accuracy: % of true singletons correctly predicted as empty
    - non_singleton_avg_fbeta: Average F-beta for non-singleton entities
    - entity_count breakdown
    """
    entity_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_total = 0
    singleton_correct = 0
    non_singleton_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        score = entity_fbeta(true_matches, pred_matches, beta)
        entity_scores.append(score)

        if not true_matches:
            # Singleton
            singleton_total += 1
            if not pred_matches:
                singleton_correct += 1
        else:
            # Non-singleton
            non_singleton_scores.append(score)
            tp = len(true_matches & pred_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    macro_score = np.mean(entity_scores) if entity_scores else 0.0
    pair_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pair_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    non_singleton_avg = np.mean(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        'macro_fbeta': round(macro_score, 6),
        'pair_precision': round(pair_precision, 6),
        'pair_recall': round(pair_recall, 6),
        'singleton_accuracy': round(singleton_acc, 6),
        'singleton_total': singleton_total,
        'singleton_correct': singleton_correct,
        'non_singleton_avg_fbeta': round(non_singleton_avg, 6),
        'non_singleton_count': len(non_singleton_scores),
        'total_entities': len(entity_scores),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Threshold Sweep
# ═════════════════════════════════════════════════════════════════════════════

def threshold_sweep(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    beta: float = BETA,
) -> Tuple[float, float, List[dict]]:
    """
    Sweep decision thresholds to find optimal macro F0.5.

    Args:
        s1_ids: List of S1 entity IDs for each pair.
        target_ids: List of target (S2/S3) entity IDs for each pair.
        probabilities: Model output probabilities for each pair.
        ground_truth: {s1_id: set(true_matched_ids)}
        thresholds: List of thresholds to try. Default: 0.30 to 0.95 step 0.01.
        beta: Beta parameter.

    Returns:
        (best_threshold, best_score, sweep_results)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.01)]

    sweep_results = []
    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        # Build predictions at this threshold
        predictions: Dict[str, Set[str]] = {}
        for s1_id in ground_truth:
            predictions[s1_id] = set()

        for s1_id, t_id, prob in zip(s1_ids, target_ids, probabilities):
            if prob >= thresh:
                if s1_id not in predictions:
                    predictions[s1_id] = set()
                predictions[s1_id].add(t_id)

        score = macro_fbeta(ground_truth, predictions, beta)
        sweep_results.append({
            'threshold': thresh,
            'macro_fbeta': round(score, 6),
        })

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, sweep_results


def apply_threshold(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float,
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Apply a decision threshold to produce per-S1 predictions.

    Args:
        s1_ids: List of S1 entity IDs for each pair.
        target_ids: List of target IDs for each pair.
        probabilities: Model probabilities.
        threshold: Decision threshold.
        all_s1_ids: Complete set of S1 IDs (to ensure singletons get empty sets).

    Returns:
        {s1_id: set(matched_ids)}
    """
    predictions: Dict[str, Set[str]] = {}

    # Initialize all S1 IDs with empty sets
    if all_s1_ids:
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()

    for s1_id, t_id, prob in zip(s1_ids, target_ids, probabilities):
        if prob >= threshold:
            if s1_id not in predictions:
                predictions[s1_id] = set()
            predictions[s1_id].add(t_id)

    return predictions


if __name__ == '__main__':
    # Quick test
    gt = {
        'S1-001': {'S2-001', 'S3-001'},
        'S1-002': set(),  # singleton
        'S1-003': {'S2-003'},
    }
    pred = {
        'S1-001': {'S2-001'},  # got 1 of 2
        'S1-002': set(),       # correctly empty
        'S1-003': {'S2-003', 'S2-999'},  # got 1 right, 1 wrong
    }

    result = detailed_evaluation(gt, pred)
    print("=== Evaluation Test ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
