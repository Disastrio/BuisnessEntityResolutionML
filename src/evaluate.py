"""
evaluate.py — Macro F0.5 Calculator, Threshold Sweeper, and Singleton Accuracy.

Implements the exact scoring logic for the Amazon ML Challenge:
- Per-S1 entity F0.5 (β=0.5, precision-weighted)
- Singletons: correctly empty = 1.0, any false positive = 0.0
- Macro average across all S1 entities
- Threshold sweep to find optimal decision boundary
"""
import os
import multiprocessing as mp
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
from src.config import BETA


def infer_source(entity_id: str) -> str:
    """
    Map a target entity ID to its source label.

    Source is encoded in the ID prefix (``S2-`` / ``S3-``); never inferred from
    a hard-coded list of sources, so it works for any future source.
    """
    s = str(entity_id)
    if s.startswith("S2"):
        return "S2"
    if s.startswith("S3"):
        return "S3"
    return "UNK"


def _resolve_threshold(target_id: str, threshold: Optional[float],
                       thresholds_by_source: Optional[Dict[str, float]]) -> float:
    """Pick the decision threshold for one target ID (per-source overrides scalar)."""
    if thresholds_by_source:
        return thresholds_by_source.get(
            infer_source(target_id),
            threshold if threshold is not None else 0.5,
        )
    return threshold if threshold is not None else 0.5


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
# Prediction Assembly (threshold / per-source thresholds / barrier / rules)
# ═════════════════════════════════════════════════════════════════════════════

def assemble_predictions(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    threshold: Optional[float] = None,
    thresholds_by_source: Optional[Dict[str, float]] = None,
    all_s1_ids: Optional[Set[str]] = None,
    barrier: float = 0.0,
    reject_mask: Optional[np.ndarray] = None,
) -> Dict[str, Set[str]]:
    """
    Core per-S1 prediction assembler shared by tuning and inference.

    Args:
        threshold: scalar global decision threshold.
        thresholds_by_source: {'S2': t2, 'S3': t3}; overrides `threshold`.
        all_s1_ids: complete S1 set, so singletons receive empty sets.
        barrier: singleton guard — an entity keeps its matches only if its single
            best accepted probability is >= barrier (0.0 disables the guard).
        reject_mask: boolean array; True rejects the pair outright (decision rules).

    Returns:
        {s1_id: set(matched_ids)}
    """
    predictions: Dict[str, Set[str]] = {}
    if all_s1_ids:
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()

    best_prob: Dict[str, float] = {}
    probs_arr = np.asarray(probabilities)

    for i, (s1_id, t_id) in enumerate(zip(s1_ids, target_ids)):
        if reject_mask is not None and bool(reject_mask[i]):
            continue
        prob = float(probs_arr[i])
        if prob < _resolve_threshold(t_id, threshold, thresholds_by_source):
            continue
        predictions.setdefault(s1_id, set()).add(t_id)
        if prob > best_prob.get(s1_id, -1.0):
            best_prob[s1_id] = prob

    if barrier and barrier > 0.0:
        for s1_id, matched in predictions.items():
            if matched and best_prob.get(s1_id, 0.0) < barrier:
                predictions[s1_id] = set()

    return predictions


# ── Parallel sweep helpers (fork): each candidate value is independent ────────
_SWEEP_CTX: dict = {}


def _sweep_worker(value: float):
    c = _SWEEP_CTX
    if c['mode'] == 'threshold':
        threshold = value
        by_source = None
        barrier = c.get('barrier', 0.0)
    else:  # 'barrier'
        threshold = c.get('threshold')
        by_source = c.get('by_source')
        barrier = value
    preds = assemble_predictions(
        c['s1_ids'], c['target_ids'], c['probs'],
        threshold=threshold,
        thresholds_by_source=by_source,
        all_s1_ids=c['all_s1'],
        barrier=barrier,
        reject_mask=c.get('reject'),
    )
    return value, macro_fbeta(c['gt'], preds, c['beta'])


def _parallel_sweep(values: List[float], ctx: dict, n_workers: int) -> Dict[float, float]:
    global _SWEEP_CTX
    _SWEEP_CTX = ctx
    scores: Dict[float, float] = {}
    mpctx = mp.get_context('fork')
    with mpctx.Pool(processes=int(n_workers)) as pool:
        for value, score in pool.imap_unordered(_sweep_worker, list(values)):
            scores[value] = score
    return scores


def threshold_sweep(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    beta: float = BETA,
    reject_mask: Optional[np.ndarray] = None,
    barrier: float = 0.0,
    n_workers: int = 1,
) -> Tuple[float, float, List[dict]]:
    """
    Sweep a scalar decision threshold to find optimal macro F0.5.

    Args:
        thresholds: List of thresholds to try. Default: 0.30 to 0.95 step 0.01.
        reject_mask: pairs rejected by conservative decision rules.
        barrier: singleton confidence barrier applied at every threshold.
        n_workers: evaluate independent thresholds across processes (fork only).

    Returns:
        (best_threshold, best_score, sweep_results)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.01)]
    thresholds = [float(t) for t in thresholds]

    if n_workers and n_workers > 1 and hasattr(os, 'fork') and len(thresholds) >= 8:
        ctx = {
            'mode': 'threshold', 's1_ids': s1_ids, 'target_ids': target_ids,
            'probs': np.asarray(probabilities), 'all_s1': set(ground_truth.keys()),
            'barrier': barrier, 'reject': reject_mask,
            'gt': ground_truth, 'beta': beta,
        }
        scores = _parallel_sweep(thresholds, ctx, n_workers)
        sweep_results = [{'threshold': t, 'macro_fbeta': round(scores[t], 6)}
                         for t in thresholds]
        # First max in ascending-threshold order == serial tie-breaking.
        best_threshold = max(thresholds, key=lambda t: scores[t])
        return best_threshold, scores[best_threshold], sweep_results

    sweep_results = []
    best_threshold = 0.5
    best_score = -1.0

    for thresh in thresholds:
        predictions = assemble_predictions(
            s1_ids, target_ids, probabilities,
            threshold=thresh,
            all_s1_ids=set(ground_truth.keys()),
            barrier=barrier,
            reject_mask=reject_mask,
        )
        score = macro_fbeta(ground_truth, predictions, beta)
        sweep_results.append({'threshold': thresh, 'macro_fbeta': round(score, 6)})
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, sweep_results


def threshold_sweep_by_source(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    beta: float = BETA,
    reject_mask: Optional[np.ndarray] = None,
    barrier: float = 0.0,
    max_iters: int = 3,
) -> Tuple[Dict[str, float], float, List[dict]]:
    """
    Jointly tune separate S2 / S3 decision thresholds for macro F0.5.

    S2 and S3 have different noise profiles, so one global threshold is often
    suboptimal. The two thresholds interact inside the per-entity score, so we
    use deterministic coordinate ascent from the global optimum: alternately
    optimise one source while holding the other fixed until no further
    improvement (or `max_iters`).

    Returns:
        ({'S2': t2, 'S3': t3}, best_score, history)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
    thresholds = [float(t) for t in thresholds]

    global_t, global_score, _ = threshold_sweep(
        s1_ids, target_ids, probabilities, ground_truth,
        thresholds=thresholds, beta=beta, reject_mask=reject_mask, barrier=barrier,
    )
    global_t = float(global_t)

    best = {'S2': global_t, 'S3': global_t}
    best_score = global_score
    history: List[dict] = [
        {'iteration': 0, 'S2': global_t, 'S3': global_t,
         'macro_fbeta': round(global_score, 6)}
    ]

    for it in range(1, max_iters + 1):
        improved = False
        for src in ('S2', 'S3'):
            current = best[src]
            for t in thresholds:
                if t == current:
                    continue
                candidate = dict(best)
                candidate[src] = t
                predictions = assemble_predictions(
                    s1_ids, target_ids, probabilities,
                    thresholds_by_source=candidate,
                    all_s1_ids=set(ground_truth.keys()),
                    barrier=barrier,
                    reject_mask=reject_mask,
                )
                score = macro_fbeta(ground_truth, predictions, beta)
                if score > best_score + 1e-12:
                    best_score = score
                    best = candidate
                    improved = True
        history.append({
            'iteration': it, 'S2': best['S2'], 'S3': best['S3'],
            'macro_fbeta': round(best_score, 6),
        })
        if not improved:
            break

    return best, best_score, history


def sweep_singleton_barrier(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold: Optional[float] = None,
    thresholds_by_source: Optional[Dict[str, float]] = None,
    barriers: Optional[List[float]] = None,
    beta: float = BETA,
    reject_mask: Optional[np.ndarray] = None,
    n_workers: int = 1,
) -> Tuple[float, float, List[dict]]:
    """
    Tune the singleton confidence barrier at a fixed threshold.

    The barrier is a second-level gate: an entity only emits matches when its
    best accepted probability clears the barrier. It is a cheap, targeted way to
    protect the ~5.6% singleton entities without hurting confident matches.
    n_workers parallelises the independent barrier evaluations (fork only).

    Returns:
        (best_barrier, best_score, sweep_results)
    """
    if barriers is None:
        barriers = [0.0] + [round(b, 2) for b in np.arange(0.50, 0.99, 0.02)]
    barriers = [float(b) for b in barriers]

    if n_workers and n_workers > 1 and hasattr(os, 'fork') and len(barriers) >= 8:
        ctx = {
            'mode': 'barrier', 's1_ids': s1_ids, 'target_ids': target_ids,
            'probs': np.asarray(probabilities), 'all_s1': set(ground_truth.keys()),
            'threshold': threshold, 'by_source': thresholds_by_source,
            'reject': reject_mask, 'gt': ground_truth, 'beta': beta,
        }
        scores = _parallel_sweep(barriers, ctx, n_workers)
        sweep_results = [{'barrier': b, 'macro_fbeta': round(scores[b], 6)}
                         for b in barriers]
        best_barrier = max(barriers, key=lambda b: scores[b])
        return best_barrier, scores[best_barrier], sweep_results

    sweep_results = []
    best_barrier = 0.0
    best_score = -1.0

    for barrier in barriers:
        predictions = assemble_predictions(
            s1_ids, target_ids, probabilities,
            threshold=threshold,
            thresholds_by_source=thresholds_by_source,
            all_s1_ids=set(ground_truth.keys()),
            barrier=barrier,
            reject_mask=reject_mask,
        )
        score = macro_fbeta(ground_truth, predictions, beta)
        sweep_results.append({'barrier': barrier, 'macro_fbeta': round(score, 6)})
        if score > best_score:
            best_score = score
            best_barrier = barrier

    return best_barrier, best_score, sweep_results


def apply_threshold(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float,
    all_s1_ids: Optional[Set[str]] = None,
    thresholds_by_source: Optional[Dict[str, float]] = None,
    barrier: float = 0.0,
    reject_mask: Optional[np.ndarray] = None,
) -> Dict[str, Set[str]]:
    """Backward-compatible threshold application (see `assemble_predictions`)."""
    return assemble_predictions(
        s1_ids, target_ids, probabilities,
        threshold=threshold,
        thresholds_by_source=thresholds_by_source,
        all_s1_ids=all_s1_ids,
        barrier=barrier,
        reject_mask=reject_mask,
    )


def apply_threshold_by_source(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    thresholds_by_source: Dict[str, float],
    all_s1_ids: Optional[Set[str]] = None,
    barrier: float = 0.0,
    reject_mask: Optional[np.ndarray] = None,
) -> Dict[str, Set[str]]:
    """Apply per-source decision thresholds (see `assemble_predictions`)."""
    return assemble_predictions(
        s1_ids, target_ids, probabilities,
        thresholds_by_source=thresholds_by_source,
        all_s1_ids=all_s1_ids,
        barrier=barrier,
        reject_mask=reject_mask,
    )


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
