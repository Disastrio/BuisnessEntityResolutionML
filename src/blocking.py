"""
blocking.py — Multi-Pass Candidate Generation (7-Tier Inverted Index Union).

Sets the RECALL CEILING for the entire pipeline. Uses union of multiple
blocking strategies to maximize candidate recall while keeping candidate
set manageable (capped at MAX_CANDIDATES per S1 entity).

Blocking Strategies:
    Index 1: Exact normalized name (country + name_clean)
    Index 2: Name prefix (country + name_prefix[:6])
    Index 3: Postal/PIN code (country + postal_code)
    Index 4: Rare name tokens (TF-IDF > threshold, inverted index)
    Index 5: Shared numerics + country (addr numbers + country)
    Index 6: Character n-grams of name (3-char shingles)
    Index 7: Relaxed — first 4 chars of name + country (catch-all)
"""
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
import re
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import MAX_CANDIDATES, ID_COL


# ═════════════════════════════════════════════════════════════════════════════
# Inverted Index Builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_inverted_index(
    df: pd.DataFrame,
    key_fn,
    id_col: str = ID_COL,
) -> Dict[str, List[str]]:
    """
    Build an inverted index: blocking_key → [entity_ids].

    Args:
        df: Source DataFrame (S2 or S3) with normalized columns.
        key_fn: Function(row) → list of blocking keys for that row.
        id_col: Entity ID column name.
    """
    index = defaultdict(list)
    for _, row in df.iterrows():
        keys = key_fn(row)
        eid = row[id_col]
        for k in keys:
            if k:  # Skip empty keys
                index[k].append(eid)
    return dict(index)


def _build_inverted_index_fast(
    df: pd.DataFrame,
    key_col: str,
    id_col: str = ID_COL,
) -> Dict[str, List[str]]:
    """
    Build an inverted index from a pre-computed key column (faster, vectorized).
    """
    index = defaultdict(list)
    for eid, key in zip(df[id_col], df[key_col]):
        if key and str(key).strip():
            index[str(key).strip()].append(eid)
    return dict(index)


# ═════════════════════════════════════════════════════════════════════════════
# Blocking Key Generators
# ═════════════════════════════════════════════════════════════════════════════

def _key_exact_name(row) -> List[str]:
    """Index 1: country + exact normalized name."""
    name = str(row.get('name_clean', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    if name:
        return [f"{country}|{name}"]
    return []


def _key_name_prefix(row, length: int = 6) -> List[str]:
    """Index 2: country + name prefix (first N chars)."""
    name = str(row.get('name_clean', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    # Strip leading 'the ' for blocking
    name = re.sub(r'^the\s+', '', name)
    if len(name) >= 3:
        return [f"{country}|pfx|{name[:length]}"]
    return []


def _key_postal(row) -> List[str]:
    """Index 3: country + postal/PIN/ZIP code."""
    postal = str(row.get('addr_postal', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    keys = []
    if postal:
        for code in postal.split(','):
            code = code.strip()
            if code:
                keys.append(f"{country}|post|{code}")
    return keys


def _key_name_tokens_rare(row, rare_tokens: Set[str]) -> List[str]:
    """Index 4: Rare name tokens (only tokens with low document frequency)."""
    tokens_str = str(row.get('name_tokens', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    if not tokens_str:
        return []
    tokens = tokens_str.split()
    keys = []
    for tok in tokens:
        if tok in rare_tokens and len(tok) >= 3:
            keys.append(f"{country}|rtok|{tok}")
    return keys


def _key_addr_numbers_country(row) -> List[str]:
    """Index 5: Shared address numeric tokens + country."""
    nums = str(row.get('addr_numeric', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    if not nums:
        return []
    # Use sorted numeric tokens as a combined key
    num_list = sorted(set(nums.split()))
    if len(num_list) >= 2:
        # Take first 3 numeric tokens for key
        key = '_'.join(num_list[:3])
        return [f"{country}|nums|{key}"]
    return []


def _key_name_ngrams(row, n: int = 3) -> List[str]:
    """Index 6: Character n-grams (shingles) of the name."""
    name = str(row.get('name_clean', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    if len(name) < n:
        return []
    # Generate character n-grams
    ngrams = set()
    for i in range(len(name) - n + 1):
        gram = name[i:i+n]
        if gram.strip():  # Skip whitespace-only grams
            ngrams.add(gram)
    # Only use a subset of distinct n-grams to avoid index explosion
    # Pick at most 5 n-grams that contain non-space characters
    selected = sorted(ngrams)[:5]
    return [f"{country}|ng|{g}" for g in selected]


def _key_relaxed_name(row, length: int = 4) -> List[str]:
    """Index 7: Relaxed — first 4 chars of name + country (catch-all)."""
    name = str(row.get('name_clean', '')).strip()
    country = str(row.get('country_clean', '')).strip()
    name = re.sub(r'^the\s+', '', name)
    if len(name) >= 3:
        return [f"{country}|rel|{name[:length]}"]
    return []


# ═════════════════════════════════════════════════════════════════════════════
# Rare Token Detection (for Index 4)
# ═════════════════════════════════════════════════════════════════════════════

def compute_rare_tokens(
    df: pd.DataFrame,
    min_freq: int = 2,
    max_doc_frac: float = 0.01,
) -> Set[str]:
    """
    Find tokens that appear in at least min_freq documents but in at most
    max_doc_frac fraction of all documents. These are distinctive tokens
    useful for blocking.
    """
    doc_freq = defaultdict(int)
    n_docs = len(df)

    for tokens_str in df['name_tokens']:
        if not tokens_str or str(tokens_str).strip() == '':
            continue
        unique_tokens = set(str(tokens_str).split())
        for tok in unique_tokens:
            doc_freq[tok] += 1

    max_count = max(1, int(n_docs * max_doc_frac))
    rare = {
        tok for tok, freq in doc_freq.items()
        if min_freq <= freq <= max_count and len(tok) >= 3
    }
    return rare


# ═════════════════════════════════════════════════════════════════════════════
# Main Blocking Engine
# ═════════════════════════════════════════════════════════════════════════════

def build_candidate_indices(
    target_df: pd.DataFrame,
    rare_tokens: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build all 7 inverted indices for a target source (S2 or S3).

    Args:
        target_df: Normalized S2 or S3 DataFrame.
        rare_tokens: Set of rare tokens for Index 4.

    Returns:
        Dict with index names as keys and inverted indices as values.
    """
    indices = {}

    # Index 1: Exact name
    print("  Building Index 1 (exact name)...")
    idx1 = defaultdict(list)
    for eid, name, country in zip(
        target_df[ID_COL], target_df['name_clean'], target_df['country_clean']
    ):
        name = str(name).strip()
        country = str(country).strip()
        if name:
            idx1[f"{country}|{name}"].append(eid)
    indices['exact_name'] = dict(idx1)

    # Index 2: Name prefix
    print("  Building Index 2 (name prefix)...")
    idx2 = defaultdict(list)
    for eid, name, country in zip(
        target_df[ID_COL], target_df['name_clean'], target_df['country_clean']
    ):
        name = re.sub(r'^the\s+', '', str(name).strip())
        country = str(country).strip()
        if len(name) >= 3:
            idx2[f"{country}|pfx|{name[:6]}"].append(eid)
    indices['name_prefix'] = dict(idx2)

    # Index 3: Postal code
    print("  Building Index 3 (postal code)...")
    idx3 = defaultdict(list)
    for eid, postal, country in zip(
        target_df[ID_COL], target_df['addr_postal'], target_df['country_clean']
    ):
        postal = str(postal).strip()
        country = str(country).strip()
        if postal:
            for code in postal.split(','):
                code = code.strip()
                if code:
                    idx3[f"{country}|post|{code}"].append(eid)
    indices['postal'] = dict(idx3)

    # Index 4: Rare name tokens
    if rare_tokens:
        print(f"  Building Index 4 (rare tokens, {len(rare_tokens)} rare tokens)...")
        idx4 = defaultdict(list)
        for eid, tokens_str, country in zip(
            target_df[ID_COL], target_df['name_tokens'], target_df['country_clean']
        ):
            tokens_str = str(tokens_str).strip()
            country = str(country).strip()
            if tokens_str:
                for tok in tokens_str.split():
                    if tok in rare_tokens and len(tok) >= 3:
                        idx4[f"{country}|rtok|{tok}"].append(eid)
        indices['rare_tokens'] = dict(idx4)

    # Index 5: Address numerics + country
    print("  Building Index 5 (address numerics)...")
    idx5 = defaultdict(list)
    for eid, nums, country in zip(
        target_df[ID_COL], target_df['addr_numeric'], target_df['country_clean']
    ):
        nums = str(nums).strip()
        country = str(country).strip()
        if nums:
            num_list = sorted(set(nums.split()))
            if len(num_list) >= 2:
                key = '_'.join(num_list[:3])
                idx5[f"{country}|nums|{key}"].append(eid)
    indices['addr_numerics'] = dict(idx5)

    # Index 7: Relaxed name
    print("  Building Index 7 (relaxed name)...")
    idx7 = defaultdict(list)
    for eid, name, country in zip(
        target_df[ID_COL], target_df['name_clean'], target_df['country_clean']
    ):
        name = re.sub(r'^the\s+', '', str(name).strip())
        country = str(country).strip()
        if len(name) >= 3:
            idx7[f"{country}|rel|{name[:4]}"].append(eid)
    indices['relaxed_name'] = dict(idx7)

    return indices


def generate_candidates_for_s1(
    s1_row: pd.Series,
    indices: Dict[str, Dict[str, List[str]]],
    rare_tokens: Optional[Set[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> Set[str]:
    """
    Generate candidate IDs for a single S1 entity by querying all indices.

    Returns union of all candidates, capped at max_candidates.
    """
    candidates = set()

    name = str(s1_row.get('name_clean', '')).strip()
    country = str(s1_row.get('country_clean', '')).strip()
    postal = str(s1_row.get('addr_postal', '')).strip()
    nums = str(s1_row.get('addr_numeric', '')).strip()
    tokens_str = str(s1_row.get('name_tokens', '')).strip()

    # Index 1: Exact name
    key1 = f"{country}|{name}"
    if key1 in indices.get('exact_name', {}):
        candidates.update(indices['exact_name'][key1])

    # Index 2: Name prefix
    clean_name = re.sub(r'^the\s+', '', name)
    if len(clean_name) >= 3:
        key2 = f"{country}|pfx|{clean_name[:6]}"
        if key2 in indices.get('name_prefix', {}):
            candidates.update(indices['name_prefix'][key2])

    # Index 3: Postal code
    if postal:
        for code in postal.split(','):
            code = code.strip()
            if code:
                key3 = f"{country}|post|{code}"
                if key3 in indices.get('postal', {}):
                    candidates.update(indices['postal'][key3])

    # Index 4: Rare name tokens
    if rare_tokens and tokens_str:
        for tok in tokens_str.split():
            if tok in rare_tokens and len(tok) >= 3:
                key4 = f"{country}|rtok|{tok}"
                if key4 in indices.get('rare_tokens', {}):
                    candidates.update(indices['rare_tokens'][key4])

    # Index 5: Address numerics
    if nums:
        num_list = sorted(set(nums.split()))
        if len(num_list) >= 2:
            key5 = f"{country}|nums|{'_'.join(num_list[:3])}"
            if key5 in indices.get('addr_numerics', {}):
                candidates.update(indices['addr_numerics'][key5])

    # Index 7: Relaxed name
    if len(clean_name) >= 3:
        key7 = f"{country}|rel|{clean_name[:4]}"
        if key7 in indices.get('relaxed_name', {}):
            candidates.update(indices['relaxed_name'][key7])

    # Cap at max_candidates
    if len(candidates) > max_candidates:
        candidates = set(list(candidates)[:max_candidates])

    return candidates


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int = MAX_CANDIDATES,
    show_progress: bool = True,
) -> Dict[str, Set[str]]:
    """
    Generate candidate pairs for all S1 entities against S2 and S3.

    Args:
        s1_df: Normalized S1 DataFrame.
        s2_df: Normalized S2 DataFrame.
        s3_df: Normalized S3 DataFrame.
        max_candidates: Max candidate pairs per S1 entity.

    Returns:
        Dict: {s1_id: set(candidate_s2_s3_ids)}
    """
    print("Computing rare tokens for S2...")
    rare_tokens_s2 = compute_rare_tokens(s2_df)
    print(f"  Found {len(rare_tokens_s2)} rare tokens in S2")

    print("Computing rare tokens for S3...")
    rare_tokens_s3 = compute_rare_tokens(s3_df)
    print(f"  Found {len(rare_tokens_s3)} rare tokens in S3")

    # Combine rare tokens
    all_rare = rare_tokens_s2 | rare_tokens_s3

    print("\nBuilding S2 indices...")
    s2_indices = build_candidate_indices(s2_df, rare_tokens=rare_tokens_s2)

    print("\nBuilding S3 indices...")
    s3_indices = build_candidate_indices(s3_df, rare_tokens=rare_tokens_s3)

    # Generate candidates for each S1 entity
    print(f"\nGenerating candidates for {len(s1_df)} S1 entities...")
    all_candidates: Dict[str, Set[str]] = {}

    iterator = s1_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(s1_df), desc="Blocking")

    for _, s1_row in iterator:
        s1_id = s1_row[ID_COL]

        # Query S2 indices
        s2_cands = generate_candidates_for_s1(
            s1_row, s2_indices,
            rare_tokens=rare_tokens_s2,
            max_candidates=max_candidates,
        )

        # Query S3 indices
        s3_cands = generate_candidates_for_s1(
            s1_row, s3_indices,
            rare_tokens=rare_tokens_s3,
            max_candidates=max_candidates,
        )

        combined = s2_cands | s3_cands
        # Final cap
        if len(combined) > max_candidates:
            combined = set(list(combined)[:max_candidates])

        all_candidates[s1_id] = combined

    return all_candidates


# ═════════════════════════════════════════════════════════════════════════════
# Blocking Quality Metrics
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_blocking(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """
    Evaluate blocking quality:
    - Candidate Recall: fraction of true matches found in candidates
    - Reduction Ratio: how much blocking reduces the search space
    - Average/Max candidates per S1

    This is a critical gate: candidate recall must be >= 92%.
    """
    total_true_matches = 0
    found_true_matches = 0
    total_candidates = 0
    max_cands = 0
    singleton_count = 0
    correct_singleton_count = 0

    for s1_id, true_matches in ground_truth.items():
        cands = candidates.get(s1_id, set())
        total_candidates += len(cands)
        max_cands = max(max_cands, len(cands))

        if not true_matches:
            singleton_count += 1
            if not cands:
                correct_singleton_count += 1
        else:
            total_true_matches += len(true_matches)
            found_true_matches += len(true_matches & cands)

    candidate_recall = (
        found_true_matches / total_true_matches
        if total_true_matches > 0 else 1.0
    )
    avg_candidates = (
        total_candidates / len(ground_truth)
        if ground_truth else 0
    )

    return {
        'candidate_recall': round(candidate_recall, 4),
        'total_true_matches': total_true_matches,
        'found_true_matches': found_true_matches,
        'missed_true_matches': total_true_matches - found_true_matches,
        'avg_candidates_per_s1': round(avg_candidates, 2),
        'max_candidates_per_s1': max_cands,
        'total_candidates': total_candidates,
        'singletons': singleton_count,
        'gate_passed': candidate_recall >= 0.92,
    }


if __name__ == '__main__':
    print("Blocking module loaded. Run via pipeline.py for full execution.")
