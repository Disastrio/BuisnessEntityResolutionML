"""
features.py — Pairwise Similarity Feature Engineering for Business Entity Resolution.

Generates ~28 dense similarity features for every candidate pair (S1, S2/S3):

Name Features (~12):
    1.  name_exact_match         Exact normalized name match (bool)
    2.  name_levenshtein         Levenshtein ratio (rapidfuzz)
    3.  name_jaro_winkler        Jaro-Winkler similarity
    4.  name_token_sort          Token Sort Ratio
    5.  name_token_set           Token Set Ratio
    6.  name_jaccard             Token Jaccard similarity
    7.  name_char_ngram_sim      Character 3-gram Jaccard
    8.  name_containment         Containment ratio (min overlap)
    9.  name_token_overlap       Count of overlapping tokens
    10. name_len_diff            Absolute length difference ratio
    11. name_prefix_match        First 6 chars match (bool)
    12. name_common_token_ratio  Ratio of shared tokens to total tokens

Address Features (~10):
    13. addr_exact_match         Exact normalized address match (bool)
    14. addr_levenshtein         Levenshtein ratio
    15. addr_token_jaccard       Token Jaccard
    16. addr_token_overlap       Token overlap count
    17. addr_char_ngram_sim      Character 3-gram Jaccard
    18. addr_len_diff            Length difference ratio
    19. addr_numeric_jaccard     Shared numeric token Jaccard
    20. addr_numeric_overlap     Count of shared numeric tokens
    21. addr_postal_match        Postal code exact match (bool)
    22. addr_containment         Address containment ratio

Cross-Field Features (~6):
    23. same_country             Country match (bool)
    24. source_is_s2             Source is S2 (bool) — noise profile feature
    25. name_addr_both_exact     Name AND address both exact match
    26. name_high_addr_high      name_sim > 0.85 AND addr_sim > 0.70
    27. name_high_addr_mismatch  name_sim > 0.90 AND addr numeric mismatch (precision guard)
    28. name_high_postal_match   name_sim > 0.85 AND postal code match
"""
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

from src.config import ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL


# ═════════════════════════════════════════════════════════════════════════════
# Low-Level Similarity Functions
# ═════════════════════════════════════════════════════════════════════════════

def _safe_str(val) -> str:
    """Safely convert to string, handling NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of whitespace-split token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def token_overlap_count(a: str, b: str) -> int:
    """Count of tokens shared between two strings."""
    if not a or not b:
        return 0
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb)


def containment_ratio(a: str, b: str) -> float:
    """
    Containment: |A ∩ B| / min(|A|, |B|).
    Measures if the smaller set is contained in the larger.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    min_len = min(len(sa), len(sb))
    if min_len == 0:
        return 0.0
    return len(sa & sb) / min_len


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def ngrams(s):
        s = s.replace(' ', '')
        return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}

    ga, gb = ngrams(a), ngrams(b)
    intersection = len(ga & gb)
    union = len(ga | gb)
    return intersection / union if union > 0 else 0.0


def length_diff_ratio(a: str, b: str) -> float:
    """Absolute length difference divided by max length."""
    la, lb = len(a), len(b)
    max_len = max(la, lb)
    if max_len == 0:
        return 0.0
    return abs(la - lb) / max_len


def numeric_token_jaccard(nums_a: str, nums_b: str) -> float:
    """Jaccard similarity of numeric tokens extracted from addresses."""
    if not nums_a and not nums_b:
        return 1.0
    if not nums_a or not nums_b:
        return 0.0
    sa, sb = set(nums_a.split()), set(nums_b.split())
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def numeric_token_overlap(nums_a: str, nums_b: str) -> int:
    """Count of shared numeric tokens."""
    if not nums_a or not nums_b:
        return 0
    sa, sb = set(nums_a.split()), set(nums_b.split())
    return len(sa & sb)


def postal_match(postal_a: str, postal_b: str) -> float:
    """Check if any postal code matches between two entities."""
    if not postal_a or not postal_b:
        return 0.0
    sa = set(postal_a.split(','))
    sb = set(postal_b.split(','))
    return 1.0 if sa & sb else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Feature Vector Generation for a Single Pair
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Name features
    'name_exact_match',
    'name_levenshtein',
    'name_jaro_winkler',
    'name_token_sort',
    'name_token_set',
    'name_jaccard',
    'name_char_ngram_sim',
    'name_containment',
    'name_token_overlap',
    'name_len_diff',
    'name_prefix_match',
    'name_common_token_ratio',
    # Address features
    'addr_exact_match',
    'addr_levenshtein',
    'addr_token_jaccard',
    'addr_token_overlap',
    'addr_char_ngram_sim',
    'addr_len_diff',
    'addr_numeric_jaccard',
    'addr_numeric_overlap',
    'addr_postal_match',
    'addr_containment',
    # Cross-field features
    'same_country',
    'source_is_s2',
    'name_addr_both_exact',
    'name_high_addr_high',
    'name_high_addr_mismatch',
    'name_high_postal_match',
]


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    s1_name_tokens: str,
    s1_addr_numeric: str,
    s1_addr_postal: str,
    s1_name_prefix: str,
    t_name: str,
    t_addr: str,
    t_country: str,
    t_name_tokens: str,
    t_addr_numeric: str,
    t_addr_postal: str,
    t_name_prefix: str,
    t_source: str,
) -> List[float]:
    """
    Compute all ~28 pairwise features for one (S1, S2/S3) candidate pair.

    Returns list of floats in FEATURE_NAMES order.
    """
    # ── Name Features ─────────────────────────────────────────────────────
    name_exact = 1.0 if s1_name == t_name and s1_name else 0.0

    # RapidFuzz similarities (0-100 scale, normalize to 0-1)
    name_lev = fuzz.ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_name, t_name) if s1_name and t_name else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    name_tset = fuzz.token_set_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0

    name_jacc = token_jaccard(s1_name_tokens, t_name_tokens)
    name_ngram = char_ngram_jaccard(s1_name, t_name)
    name_contain = containment_ratio(s1_name_tokens, t_name_tokens)
    name_tok_overlap = float(token_overlap_count(s1_name_tokens, t_name_tokens))
    name_len_d = length_diff_ratio(s1_name, t_name)
    name_pfx_match = 1.0 if s1_name_prefix == t_name_prefix and s1_name_prefix else 0.0

    # Common token ratio
    if s1_name_tokens and t_name_tokens:
        sa = set(s1_name_tokens.split())
        sb = set(t_name_tokens.split())
        total = len(sa | sb)
        name_common_ratio = len(sa & sb) / total if total > 0 else 0.0
    else:
        name_common_ratio = 0.0

    # ── Address Features ──────────────────────────────────────────────────
    addr_exact = 1.0 if s1_addr == t_addr and s1_addr else 0.0
    addr_lev = fuzz.ratio(s1_addr, t_addr) / 100.0 if s1_addr and t_addr else 0.0
    addr_tok_jacc = token_jaccard(s1_addr, t_addr)
    addr_tok_overlap = float(token_overlap_count(s1_addr, t_addr))
    addr_ngram = char_ngram_jaccard(s1_addr, t_addr)
    addr_len_d = length_diff_ratio(s1_addr, t_addr)
    addr_num_jacc = numeric_token_jaccard(s1_addr_numeric, t_addr_numeric)
    addr_num_overlap = float(numeric_token_overlap(s1_addr_numeric, t_addr_numeric))
    addr_post_match = postal_match(s1_addr_postal, t_addr_postal)
    addr_contain = containment_ratio(s1_addr, t_addr)

    # ── Cross-Field Features ──────────────────────────────────────────────
    same_ctry = 1.0 if s1_country == t_country and s1_country else 0.0
    is_s2 = 1.0 if t_source == 'S2' else 0.0

    # Combined signals
    both_exact = 1.0 if name_exact and addr_exact else 0.0
    name_high_sim = name_lev > 0.85
    addr_high_sim = addr_lev > 0.70
    name_high_addr_h = 1.0 if name_high_sim and addr_high_sim else 0.0

    # Precision guard: high name sim but address numeric mismatch
    has_addr_numeric_mismatch = (
        s1_addr_numeric and t_addr_numeric
        and not (set(s1_addr_numeric.split()) & set(t_addr_numeric.split()))
    )
    name_high_addr_mis = 1.0 if name_lev > 0.90 and has_addr_numeric_mismatch else 0.0

    # High name + postal match
    name_high_postal = 1.0 if name_high_sim and addr_post_match else 0.0

    return [
        name_exact, name_lev, name_jw, name_tsort, name_tset,
        name_jacc, name_ngram, name_contain, name_tok_overlap,
        name_len_d, name_pfx_match, name_common_ratio,
        addr_exact, addr_lev, addr_tok_jacc, addr_tok_overlap,
        addr_ngram, addr_len_d, addr_num_jacc, addr_num_overlap,
        addr_post_match, addr_contain,
        same_ctry, is_s2, both_exact,
        name_high_addr_h, name_high_addr_mis, name_high_postal,
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Batch Feature Generation
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_pairs: Dict[str, Set[str]],
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Build feature matrix for all candidate pairs.

    Args:
        s1_df: Normalized S1 DataFrame.
        target_df: Normalized S2 or S3 DataFrame (concatenated).
        candidate_pairs: {s1_id: set(candidate_ids)}

    Returns:
        (features_df, labels placeholder, s1_ids list, target_ids list)
        Labels are empty — fill from ground truth externally.
    """
    from tqdm import tqdm

    # Index target_df by entity_id for fast lookup
    target_lookup = target_df.set_index(ID_COL)

    # Index s1_df by entity_id
    s1_lookup = s1_df.set_index(ID_COL)

    all_features = []
    all_s1_ids = []
    all_target_ids = []

    s1_ids_with_candidates = [
        s1_id for s1_id in candidate_pairs
        if candidate_pairs[s1_id]
    ]

    iterator = s1_ids_with_candidates
    if show_progress:
        iterator = tqdm(iterator, desc="Computing features")

    for s1_id in iterator:
        if s1_id not in s1_lookup.index:
            continue

        s1_row = s1_lookup.loc[s1_id]
        s1_name = _safe_str(s1_row.get('name_clean'))
        s1_addr = _safe_str(s1_row.get('addr_clean'))
        s1_country = _safe_str(s1_row.get('country_clean'))
        s1_name_tokens = _safe_str(s1_row.get('name_tokens'))
        s1_addr_numeric = _safe_str(s1_row.get('addr_numeric'))
        s1_addr_postal = _safe_str(s1_row.get('addr_postal'))
        s1_name_pfx = _safe_str(s1_row.get('name_prefix'))

        for t_id in candidate_pairs[s1_id]:
            if t_id not in target_lookup.index:
                continue

            t_row = target_lookup.loc[t_id]
            # Handle case where index lookup returns DataFrame (duplicates)
            if isinstance(t_row, pd.DataFrame):
                t_row = t_row.iloc[0]

            t_name = _safe_str(t_row.get('name_clean'))
            t_addr = _safe_str(t_row.get('addr_clean'))
            t_country = _safe_str(t_row.get('country_clean'))
            t_name_tokens = _safe_str(t_row.get('name_tokens'))
            t_addr_numeric = _safe_str(t_row.get('addr_numeric'))
            t_addr_postal = _safe_str(t_row.get('addr_postal'))
            t_name_pfx = _safe_str(t_row.get('name_prefix'))
            t_source = _safe_str(t_row.get('source'))

            feats = compute_pair_features(
                s1_name, s1_addr, s1_country,
                s1_name_tokens, s1_addr_numeric, s1_addr_postal, s1_name_pfx,
                t_name, t_addr, t_country,
                t_name_tokens, t_addr_numeric, t_addr_postal, t_name_pfx,
                t_source,
            )
            all_features.append(feats)
            all_s1_ids.append(s1_id)
            all_target_ids.append(t_id)

    features_df = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    return features_df, all_s1_ids, all_target_ids


def assign_labels(
    s1_ids: List[str],
    target_ids: List[str],
    ground_truth: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Assign binary labels (match=1, no-match=0) to candidate pairs
    using ground truth.
    """
    labels = np.zeros(len(s1_ids), dtype=np.float32)
    for i, (s1_id, t_id) in enumerate(zip(s1_ids, target_ids)):
        if s1_id in ground_truth and t_id in ground_truth[s1_id]:
            labels[i] = 1.0
    return labels


if __name__ == '__main__':
    print(f"Feature module loaded. {len(FEATURE_NAMES)} features defined:")
    for i, name in enumerate(FEATURE_NAMES, 1):
        print(f"  {i:2d}. {name}")
