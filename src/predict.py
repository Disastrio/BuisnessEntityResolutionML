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
    MATCHING_OUT, CANDIDATE_OUT, DEFAULT_THRESH,
)
from src.features import FEATURE_NAMES


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


# ═════════════════════════════════════════════════════════════════════════════
# Per-S1 Match Assembly
# ═════════════════════════════════════════════════════════════════════════════

def assemble_matches(
    s1_ids: List[str],
    target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESH,
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Assemble per-S1 entity matches by applying threshold.

    Args:
        s1_ids: S1 entity IDs for each candidate pair.
        target_ids: Target (S2/S3) entity IDs for each pair.
        probabilities: Match probabilities.
        threshold: Decision threshold.
        all_s1_ids: Complete set of S1 IDs (ensures every S1 has an entry).

    Returns:
        {s1_id: set(matched_s2_s3_ids)}
    """
    matches: Dict[str, Set[str]] = {}

    # Initialize all S1 IDs with empty sets
    if all_s1_ids:
        for s1_id in all_s1_ids:
            matches[s1_id] = set()

    # Apply threshold
    for s1_id, t_id, prob in zip(s1_ids, target_ids, probabilities):
        if s1_id not in matches:
            matches[s1_id] = set()
        if prob >= threshold:
            matches[s1_id].add(t_id)

    return matches


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
