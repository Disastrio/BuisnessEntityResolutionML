"""
predict.py — Test Inference & Per-S1 Thresholding for Business Entity Resolution.

Handles:
- Full test set candidate generation, feature computation, and scoring
- Per-S1 entity match assembly with threshold
- Output file generation (matching_results.tsv and candidate_pairs.tsv)
- Format validation before save
"""
from typing import Dict, Set, List, Optional
import csv

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import (
    ID_COL, GT_S1_COL, GT_MATCH_COL, CAND_MATCH_COL,
    MATCHING_OUT, CANDIDATE_OUT, DEFAULT_THRESH, MAX_CANDIDATES,
    NAME_VERY_HIGH_THRESHOLD, ADDR_WEAK_THRESHOLD,
)
from src.features import FEATURE_NAMES, build_feature_matrix, TargetLookup
from src.evaluate import (
    assemble_predictions, infer_source, dynamic_threshold_array,
)
from src.blocking import build_all_indices, generate_candidates_from_bundle


# ═════════════════════════════════════════════════════════════════════════════
# Prediction from Model
# ═════════════════════════════════════════════════════════════════════════════

def predict_probabilities(
    model: lgb.LGBMClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Predict match probabilities for candidate pairs.

    Returns array of P(match) for each pair.
    """
    # Use booster directly if model was loaded from file
    if hasattr(model, '_Booster') and model._Booster is not None:
        probs = model._Booster.predict(features)
        return probs
    else:
        probs = model.predict_proba(features)[:, 1]
        return probs


def predict_probabilities_by_source(
    models: Dict[str, lgb.LGBMClassifier],
    features: pd.DataFrame,
    target_ids: List[str],
    fallback_model: Optional[lgb.LGBMClassifier] = None,
) -> np.ndarray:
    """
    Score pairs with the source-specific model that matches each target ID.

    Wraps `src.train.predict_dual_probabilities` so inference can stay in this
    module without importing training internals at call time.
    """
    from src.train import predict_dual_probabilities
    return predict_dual_probabilities(models, features, target_ids,
                                      fallback_model=fallback_model)


# ═════════════════════════════════════════════════════════════════════════════
# E7 — Conservative Decision Rules
# ═════════════════════════════════════════════════════════════════════════════

def conservative_reject_mask(
    features: pd.DataFrame,
    name_thresh: float = NAME_VERY_HIGH_THRESHOLD,
    addr_weak_thresh: float = ADDR_WEAK_THRESHOLD,
) -> np.ndarray:
    """
    Flag "dangerous" pairs to reject: near-identical name but zero usable address
    evidence (no token overlap, no postal match, no exact address). These are the
    classic common-name / chain-branch false merges. Returning a mask (rather
    than mutating probabilities) keeps the rule auditable during tuning.
    """
    name_very_high = features['name_levenshtein'].to_numpy() >= name_thresh
    no_addr_evidence = (
        (features['addr_token_jaccard'].to_numpy() < addr_weak_thresh)
        & (features['addr_postal_match'].to_numpy() < 0.5)
        & (features['addr_exact_match'].to_numpy() < 0.5)
    )
    return name_very_high & no_addr_evidence


# ═════════════════════════════════════════════════════════════════════════════
# Per-S1 Match Assembly
# ═════════════════════════════════════════════════════════════════════════════

def assemble_matches(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESH,
    all_s1_ids: Optional[Set[str]] = None,
    thresholds_by_source: Optional[Dict[str, float]] = None,
    barrier: float = 0.0,
    reject_mask: Optional[np.ndarray] = None,
    pair_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Set[str]]:
    """
    Assemble per-S1 entity matches by applying threshold(s) and guards.

    Args:
        s1_ids: S1 entity IDs for each candidate pair.
        target_ids: Target (S2/S3) entity IDs for each pair.
        probabilities: Match probabilities.
        threshold: Scalar decision threshold.
        all_s1_ids: Complete set of S1 IDs (ensures every S1 has an entry).
        thresholds_by_source: {'S2': t2, 'S3': t3} overrides `threshold`.
        barrier: singleton confidence barrier (0.0 disables).
        reject_mask: pairs rejected by conservative decision rules.

    Returns:
        {s1_id: set(matched_s2_s3_ids)}
    """
    return assemble_predictions(
        s1_ids, target_ids, probabilities,
        threshold=threshold,
        thresholds_by_source=thresholds_by_source,
        all_s1_ids=all_s1_ids,
        barrier=barrier,
        reject_mask=reject_mask,
        pair_thresholds=pair_thresholds,
    )


def assemble_candidates(
    s1_ids: List[str],
    target_ids: List[str],
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Assemble ALL candidates per S1 (no threshold — full candidate list).
    This is required for candidate_pairs.tsv output.
    """
    candidates: Dict[str, Set[str]] = {}

    if all_s1_ids:
        for s1_id in all_s1_ids:
            candidates[s1_id] = set()

    for s1_id, t_id in zip(s1_ids, target_ids):
        if s1_id not in candidates:
            candidates[s1_id] = set()
        candidates[s1_id].add(t_id)

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Output File Generation
# ═════════════════════════════════════════════════════════════════════════════

def save_matching_results(
    matches: Dict[str, Set[str]],
    output_path=MATCHING_OUT,
) -> None:
    """
    Save matching results in the required TSV format.

    Format:
        source1_entity_id\tmatched_entity_ids
        S1-00001\tS2-00047,S2-00193,S3-00812
        S1-00002\tS3-00004
        S1-00003\t                              ← empty for singletons
    """
    rows = []
    for s1_id in sorted(matches.keys()):
        matched = matches[s1_id]
        matched_str = ','.join(sorted(matched)) if matched else ''
        rows.append({GT_S1_COL: s1_id, GT_MATCH_COL: matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    print(f"  Matching results saved → {output_path}")
    print(f"    Total S1 entities: {len(df)}")
    print(f"    With matches: {(df[GT_MATCH_COL] != '').sum()}")
    print(f"    Singletons: {(df[GT_MATCH_COL] == '').sum()}")


def save_candidate_pairs(
    candidates: Dict[str, Set[str]],
    output_path=CANDIDATE_OUT,
) -> None:
    """
    Save candidate pairs in the required TSV format.

    Format same as matching_results but with candidate_entity_ids column
    containing ALL candidates (superset of matched IDs).
    """
    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = candidates[s1_id]
        cands_str = ','.join(sorted(cands)) if cands else ''
        rows.append({GT_S1_COL: s1_id, CAND_MATCH_COL: cands_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    print(f"  Candidate pairs saved → {output_path}")
    print(f"    Total S1 entities: {len(df)}")
    print(f"    Avg candidates per S1: {df[CAND_MATCH_COL].apply(lambda x: len(x.split(',')) if x else 0).mean():.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# Streaming (Chunked) Inference
# ═════════════════════════════════════════════════════════════════════════════

class TsvListWriter:
    """
    Streaming writer for the two-column submission format::

        source1_entity_id<TAB>id1,id2,...

    IDs are sorted for deterministic output; an empty list is written as an
    empty field (the required singleton representation).
    """

    def __init__(self, path, second_column: str, append: bool = False):
        self.path = path
        self._f = open(path, 'a' if append else 'w', encoding='utf-8', newline='')
        if not append:
            self._f.write(f"{GT_S1_COL}\t{second_column}\n")

    def write(self, s1_id: str, ids: Set[str]) -> None:
        payload = ','.join(sorted(ids)) if ids else ''
        self._f.write(f"{s1_id}\t{payload}\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_chunked_inference(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    score_fn,
    threshold: float = DEFAULT_THRESH,
    thresholds_by_source: Optional[Dict[str, float]] = None,
    barrier: float = 0.0,
    use_rules: bool = True,
    max_candidates: int = MAX_CANDIDATES,
    chunk_size: int = 20_000,
    matching_path=MATCHING_OUT,
    candidate_path=CANDIDATE_OUT,
    limit_s1: Optional[int] = None,
    n_workers: int = 1,
    resume: bool = False,
    competitive: bool = False,
    tiered: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    Chunked, streaming test-set inference.

    Peak memory stays proportional to `chunk_size` (plus the fixed cost of the
    target frame and blocking indices) because candidates, features and
    predictions are materialised one S1 chunk at a time and written straight to
    disk, rather than holding every candidate pair for all ~1.7M S1 entities.

    Args:
        s1_df/s2_df/s3_df: normalized frames (see
            `src.features.restrict_to_core_columns`).
        score_fn: callable(features_df, target_ids) -> np.ndarray of P(match).
        chunk_size: number of S1 entities processed per chunk.
        limit_s1: optionally score only the first N S1 entities (smoke runs).
        matching_path/candidate_path: output TSV destinations.

    Returns:
        Summary dict with counts and lightweight integrity checks.
    """
    import os
    import tempfile
    from tqdm import tqdm

    s1 = s1_df.reset_index(drop=True)
    if limit_s1 is not None:
        s1 = s1.iloc[:limit_s1]

    # Resume: outputs are written in sorted-S1 order, so the number of data rows
    # already present in matching_results.tsv tells us where to continue.
    append = False
    if resume and os.path.exists(matching_path):
        try:
            with open(matching_path, 'r', encoding='utf-8') as f:
                done = max(0, sum(1 for _ in f) - 1)
        except OSError:
            done = 0
        if done:
            s1 = s1.iloc[done:]
            append = True
            print(f"  [resume] {done:,} S1 already written; continuing", flush=True)

    if len(s1) == 0:
        print("  [resume] nothing left to score.", flush=True)
        return {'s1_written': 0, 'entities_with_matches': 0, 'total_matches': 0,
                'total_candidates': 0, 'avg_candidates_per_s1': 0.0,
                'superset_violations': 0}

    # Fixed cost: target frame + blocking indices, built exactly once.
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    lookup = TargetLookup(target_df)
    bundle = build_all_indices(s2_df, s3_df)

    total_s1 = 0
    entities_with_matches = 0
    total_matches = 0
    total_candidates = 0
    superset_violations = 0
    all_s1_order = s1[ID_COL].tolist()

    # Competitive mode buffers above-threshold pairs, then assigns each target to
    # its single best S1 (enforces the injective target->S1 mapping).
    acc_f = None
    acc_path = None
    if competitive:
        fd, acc_path = tempfile.mkstemp(prefix="er_accept_", suffix=".tsv")
        acc_f = os.fdopen(fd, "w", encoding="utf-8", newline="")

    starts = list(range(0, len(s1), chunk_size))
    iterator = tqdm(starts, desc="Inference chunks") if show_progress else starts

    mw = None if competitive else TsvListWriter(
        matching_path, GT_MATCH_COL, append=append)
    cw = TsvListWriter(candidate_path, CAND_MATCH_COL, append=append)
    try:
        for start in iterator:
            chunk = s1.iloc[start:start + chunk_size]
            chunk_ids = set(chunk[ID_COL])

            candidates = generate_candidates_from_bundle(
                chunk, bundle, max_candidates=max_candidates,
                show_progress=False, n_workers=n_workers)
            features, s1_ids, target_ids = build_feature_matrix(
                chunk, target_df, candidates,
                show_progress=False, target_lookup=lookup, n_workers=n_workers)

            if len(features) > 0:
                features = features.astype(np.float32, copy=False)
                probs = np.asarray(score_fn(features, target_ids), dtype=np.float64)
                reject = conservative_reject_mask(features) if use_rules else None
                thr = dynamic_threshold_array(features) if tiered else None
            else:
                probs = np.array([], dtype=np.float64)
                reject = None
                thr = None

            cand_lists = assemble_candidates(s1_ids, target_ids, chunk_ids)

            if competitive:
                if len(features) > 0:
                    for i in range(len(s1_ids)):
                        if reject is not None and bool(reject[i]):
                            continue
                        p = probs[i]
                        t = float(thr[i]) if thr is not None else float(threshold)
                        if p >= t:
                            acc_f.write(f"{s1_ids[i]}\t{target_ids[i]}\t{p:.6f}\n")
            else:
                matches = assemble_matches(
                    s1_ids, target_ids, probs, threshold, chunk_ids,
                    thresholds_by_source=thresholds_by_source,
                    barrier=barrier, reject_mask=reject, pair_thresholds=thr)

            for s1_id in sorted(chunk_ids):
                c = cand_lists.get(s1_id, set())
                total_candidates += len(c)
                cw.write(s1_id, c)
                if not competitive:
                    m = matches.get(s1_id, set())
                    if not m <= c:
                        superset_violations += 1
                    mw.write(s1_id, m)
                    total_s1 += 1
                    total_matches += len(m)
                    if m:
                        entities_with_matches += 1

            del candidates, features, cand_lists, probs, target_ids, s1_ids

    finally:
        if mw is not None:
            mw.close()
        cw.close()

    if competitive:
        acc_f.close()
        # Pass 2: each target goes to its single highest-probability S1.
        best: Dict[str, tuple] = {}
        with open(acc_path, "r", encoding="utf-8") as f:
            for line in f:
                s1_id, t_id, p = line.rstrip("\n").split("\t")
                p = float(p)
                cur = best.get(t_id)
                if cur is None or p > cur[1]:
                    best[t_id] = (s1_id, p)
        owned: Dict[str, Set[str]] = {}
        best_prob: Dict[str, float] = {}
        for t_id, (s1_id, p) in best.items():
            owned.setdefault(s1_id, set()).add(t_id)
            if p > best_prob.get(s1_id, 0.0):
                best_prob[s1_id] = p
        if barrier and barrier > 0.0:
            for s1_id in list(owned):
                if best_prob.get(s1_id, 0.0) < barrier:
                    owned[s1_id] = set()

        with TsvListWriter(matching_path, GT_MATCH_COL, append=append) as mw2:
            for s1_id in all_s1_order:
                m = owned.get(s1_id, set())
                mw2.write(s1_id, m)
                total_s1 += 1
                total_matches += len(m)
                if m:
                    entities_with_matches += 1
        try:
            os.remove(acc_path)
        except OSError:
            pass

    return {
        's1_written': total_s1,
        'entities_with_matches': entities_with_matches,
        'total_matches': total_matches,
        'total_candidates': total_candidates,
        'avg_candidates_per_s1': (
            round(total_candidates / total_s1, 2) if total_s1 else 0.0
        ),
        'superset_violations': superset_violations,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Validation Checks
# ═════════════════════════════════════════════════════════════════════════════

def validate_output(
    matches: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    test_s1_ids: Set[str],
    test_s2_ids: Set[str],
    test_s3_ids: Set[str],
) -> List[str]:
    """
    Pre-submission validation checks.

    Returns list of error messages (empty = all good).
    """
    errors = []

    # Check 1: Every test S1 entity has exactly one row
    missing_s1 = test_s1_ids - set(matches.keys())
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} S1 entities in matching_results")

    extra_s1 = set(matches.keys()) - test_s1_ids
    if extra_s1:
        errors.append(f"Extra {len(extra_s1)} S1 entities in matching_results (not in test)")

    # Check 2: All matched IDs are valid S2/S3 test IDs
    valid_target_ids = test_s2_ids | test_s3_ids
    for s1_id, matched in matches.items():
        invalid = matched - valid_target_ids
        if invalid:
            errors.append(f"S1 {s1_id} has invalid matched IDs: {invalid}")

    # Check 3: No duplicate IDs per row (sets guarantee this)

    # Check 4: Every matched ID appears in candidate_pairs
    for s1_id, matched in matches.items():
        cands = candidates.get(s1_id, set())
        missing = matched - cands
        if missing:
            errors.append(f"S1 {s1_id}: matched IDs {missing} not in candidate_pairs")

    # Check 5: Candidate pairs also cover all test S1 entities
    missing_cand_s1 = test_s1_ids - set(candidates.keys())
    if missing_cand_s1:
        errors.append(f"Missing {len(missing_cand_s1)} S1 entities in candidate_pairs")

    if not errors:
        print("  ✅ Output validation PASSED")
    else:
        print(f"  ❌ Output validation FAILED ({len(errors)} errors)")
        for e in errors:
            print(f"    - {e}")

    return errors


if __name__ == '__main__':
    print("Predict module loaded. Run via pipeline.py for full execution.")
