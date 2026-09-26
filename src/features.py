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
import os
import multiprocessing as mp
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


def _char_trigrams(s: str) -> Set[str]:
    """Character trigrams of a space-stripped string (matches char_ngram_jaccard)."""
    s2 = s.replace(' ', '')
    if len(s2) >= 3:
        return {s2[i:i + 3] for i in range(len(s2) - 2)}
    return {s2}


def _pair_jaccard(ga: Set[str], gb: Set[str], a_nonempty: bool, b_nonempty: bool) -> float:
    """Jaccard of two prebuilt sets with the same empty-handling as the helpers."""
    if not a_nonempty and not b_nonempty:
        return 1.0
    if not a_nonempty or not b_nonempty:
        return 0.0
    union = len(ga | gb)
    return len(ga & gb) / union if union else 0.0


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
    # Each token/ngram set is built once and reused (was rebuilt per feature).
    name_exact = 1.0 if s1_name == t_name and s1_name else 0.0

    if s1_name and t_name:
        name_lev = fuzz.ratio(s1_name, t_name) / 100.0
        name_jw = distance.JaroWinkler.similarity(s1_name, t_name)
        name_tsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
        name_tset = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    else:
        name_lev = name_jw = name_tsort = name_tset = 0.0

    if s1_name_tokens and t_name_tokens:
        sa = set(s1_name_tokens.split())
        sb = set(t_name_tokens.split())
        n_inter = len(sa & sb)
        n_union = len(sa | sb)
        name_jacc = n_inter / n_union if n_union else 0.0
        n_min = min(len(sa), len(sb))
        name_contain = n_inter / n_min if n_min else 0.0
        name_tok_overlap = float(n_inter)
        name_common_ratio = n_inter / n_union if n_union > 0 else 0.0
    else:
        name_jacc = 1.0 if (not s1_name_tokens and not t_name_tokens) else 0.0
        name_contain = 0.0
        name_tok_overlap = 0.0
        name_common_ratio = 0.0

    name_ngram = _pair_jaccard(_char_trigrams(s1_name), _char_trigrams(t_name),
                               bool(s1_name), bool(t_name))
    name_len_d = length_diff_ratio(s1_name, t_name)
    name_pfx_match = 1.0 if s1_name_prefix == t_name_prefix and s1_name_prefix else 0.0

    # ── Address Features ──────────────────────────────────────────────────
    addr_exact = 1.0 if s1_addr == t_addr and s1_addr else 0.0
    addr_lev = fuzz.ratio(s1_addr, t_addr) / 100.0 if s1_addr and t_addr else 0.0

    if s1_addr and t_addr:
        aa = set(s1_addr.split())
        ab = set(t_addr.split())
        a_inter = len(aa & ab)
        a_union = len(aa | ab)
        addr_tok_jacc = a_inter / a_union if a_union else 0.0
        a_min = min(len(aa), len(ab))
        addr_contain = a_inter / a_min if a_min else 0.0
        addr_tok_overlap = float(a_inter)
    else:
        addr_tok_jacc = 1.0 if (not s1_addr and not t_addr) else 0.0
        addr_contain = 0.0
        addr_tok_overlap = 0.0

    addr_ngram = _pair_jaccard(_char_trigrams(s1_addr), _char_trigrams(t_addr),
                               bool(s1_addr), bool(t_addr))
    addr_len_d = length_diff_ratio(s1_addr, t_addr)

    if s1_addr_numeric and t_addr_numeric:
        na = set(s1_addr_numeric.split())
        nb = set(t_addr_numeric.split())
        nu_inter = len(na & nb)
        nu_union = len(na | nb)
        addr_num_jacc = nu_inter / nu_union if nu_union else 0.0
        addr_num_overlap = float(nu_inter)
        has_addr_numeric_mismatch = nu_inter == 0
    else:
        addr_num_jacc = 1.0 if (not s1_addr_numeric and not t_addr_numeric) else 0.0
        addr_num_overlap = 0.0
        has_addr_numeric_mismatch = False

    if s1_addr_postal and t_addr_postal:
        addr_post_match = 1.0 if (set(s1_addr_postal.split(',')) &
                                  set(t_addr_postal.split(','))) else 0.0
    else:
        addr_post_match = 0.0

    # ── Cross-Field Features ──────────────────────────────────────────────
    same_ctry = 1.0 if s1_country == t_country and s1_country else 0.0
    is_s2 = 1.0 if t_source == 'S2' else 0.0

    # Combined signals
    both_exact = 1.0 if name_exact and addr_exact else 0.0
    name_high_sim = name_lev > 0.85
    addr_high_sim = addr_lev > 0.70
    name_high_addr_h = 1.0 if name_high_sim and addr_high_sim else 0.0

    name_high_addr_mis = 1.0 if (name_lev > 0.90 and has_addr_numeric_mismatch) else 0.0

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
# Efficient Lookups & Large-Scale Batch Feature Generation
# ═════════════════════════════════════════════════════════════════════════════

# Columns required to compute pairwise features. Keeping only these after
# normalization releases the (much larger) raw name/address strings and cuts
# peak memory substantially at test scale.
LOOKUP_FIELDS = (
    'name_clean', 'addr_clean', 'country_clean', 'name_tokens',
    'addr_numeric', 'addr_postal', 'name_prefix', 'source',
)
CORE_COLUMNS = (ID_COL,) + LOOKUP_FIELDS


def restrict_to_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only the columns needed for blocking/features."""
    cols = [c for c in CORE_COLUMNS if c in df.columns]
    return df[cols].copy()


class TargetLookup:
    """
    Fast positional lookup over a normalized target (S2/S3) DataFrame.

    Resolves many entity IDs to row positions with a single vectorized
    ``Index.get_indexer`` call and exposes each field as a NumPy array, avoiding
    the per-pair ``DataFrame.loc`` overhead that dominates at test scale.
    """

    def __init__(self, target_df: pd.DataFrame):
        self.index = pd.Index(target_df[ID_COL].to_numpy())
        self.id_array = self.index.to_numpy()
        self.arrays = {c: target_df[c].to_numpy() for c in LOOKUP_FIELDS}

    def positions(self, entity_ids: List[str]) -> np.ndarray:
        """Return row positions for entity_ids (-1 where not found)."""
        return self.index.get_indexer(entity_ids)

    def values(self, pos: int) -> List[str]:
        """Return the 8 feature fields for a row position, in LOOKUP_FIELDS order."""
        return [_safe_str(self.arrays[c][pos]) for c in LOOKUP_FIELDS]


# Module-level state for forked feature workers (populated in the parent before
# the Pool is created; fork inherits it copy-on-write, so nothing large is pickled).
_FEAT_CTX: dict = {}


def _feature_worker(bounds: Tuple[int, int]):
    """Compute features for a contiguous slice of the flattened pair list."""
    start, end = bounds
    s1_arrays = _FEAT_CTX['s1_arrays']
    t_arrays = _FEAT_CTX['t_arrays']
    s1_pos = _FEAT_CTX['s1_pos']
    t_pos = _FEAT_CTX['t_pos']

    out = np.empty((end - start, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(start, end):
        a = int(s1_pos[i])
        b = int(t_pos[i])
        s1v = [_safe_str(s1_arrays[c][a]) for c in LOOKUP_FIELDS]
        tv = [_safe_str(t_arrays[c][b]) for c in LOOKUP_FIELDS]
        out[i - start] = compute_pair_features(
            s1v[0], s1v[1], s1v[2], s1v[3], s1v[4], s1v[5], s1v[6],
            tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7],
        )
    return start, out


def _build_features_parallel(
    s1_df: pd.DataFrame,
    target_lookup: TargetLookup,
    candidate_pairs: Dict[str, Set[str]],
    n_workers: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Multiprocess feature build over a flattened, resolved pair list."""
    s1 = s1_df.reset_index(drop=True)
    s1_ids_arr = s1[ID_COL].to_numpy()
    s1_index = pd.Index(s1_ids_arr)
    s1_arrays = {c: s1[c].to_numpy() for c in LOOKUP_FIELDS}

    ids = [sid for sid in candidate_pairs if candidate_pairs[sid]]
    s1_pos_of = s1_index.get_indexer(ids)
    keep = s1_pos_of >= 0
    ids = [sid for sid, k in zip(ids, keep) if k]
    s1_pos_of = s1_pos_of[keep]

    s1_chunks, t_chunks = [], []
    for sid, a in zip(ids, s1_pos_of):
        tpos = target_lookup.positions(sorted(candidate_pairs[sid]))
        m = tpos >= 0
        if not m.any():
            continue
        tpi = tpos[m].astype(np.int32)
        s1_chunks.append(np.full(len(tpi), a, dtype=np.int32))
        t_chunks.append(tpi)

    if s1_chunks:
        s1_pos = np.concatenate(s1_chunks)
        t_pos = np.concatenate(t_chunks)
    else:
        s1_pos = np.empty(0, np.int32)
        t_pos = np.empty(0, np.int32)
    n_total = len(s1_pos)

    global _FEAT_CTX
    _FEAT_CTX = {
        's1_arrays': s1_arrays,
        't_arrays': target_lookup.arrays,
        's1_pos': s1_pos,
        't_pos': t_pos,
    }

    bounds = np.linspace(0, n_total, n_workers + 1).astype(int)
    ranges = [(int(bounds[i]), int(bounds[i + 1]))
              for i in range(n_workers) if bounds[i] < bounds[i + 1]]

    results: Dict[int, np.ndarray] = {}
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=len(ranges)) as pool:
        for start, arr in pool.imap_unordered(_feature_worker, ranges):
            results[start] = arr

    if ranges:
        feats = np.vstack([results[s] for s, _ in ranges])
    else:
        feats = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    features_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    all_s1 = [str(x) for x in s1_ids_arr[s1_pos]]
    all_target = [str(x) for x in target_lookup.id_array[t_pos]]
    return features_df, all_s1, all_target


def build_feature_matrix(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_pairs: Dict[str, Set[str]],
    show_progress: bool = True,
    target_lookup: Optional[TargetLookup] = None,
    n_workers: int = 1,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the feature matrix for candidate pairs.

    Args:
        s1_df: Normalized S1 DataFrame.
        target_df: Normalized S2/S3 DataFrame (concatenated).
        candidate_pairs: {s1_id: set(candidate_ids)}.
        target_lookup: Optionally reuse a prebuilt TargetLookup (avoids
            rebuilding the index every chunk during streaming inference).
        n_workers: Process count for the embarrassingly parallel per-pair
            feature loop. >1 requires ``os.fork`` (Linux/macOS).

    Returns:
        (features_df, s1_ids, target_ids); target ids are iterated in sorted
        order for determinism.
    """
    from tqdm import tqdm

    if target_lookup is None:
        target_lookup = TargetLookup(target_df)

    # Multiprocess path (fork only); keeps row order deterministic because the
    # contiguous ranges are reassembled by ascending start offset.
    if n_workers and n_workers > 1 and hasattr(os, 'fork') and len(candidate_pairs) >= 2000:
        return _build_features_parallel(
            s1_df, target_lookup, candidate_pairs, int(n_workers))

    s1 = s1_df.reset_index(drop=True)
    s1_index = pd.Index(s1[ID_COL].to_numpy())
    s1_arrays = {c: s1[c].to_numpy() for c in LOOKUP_FIELDS}

    all_features: List[List[float]] = []
    all_s1_ids: List[str] = []
    all_target_ids: List[str] = []

    s1_ids_with_candidates = [sid for sid in candidate_pairs if candidate_pairs[sid]]
    iterator = s1_ids_with_candidates
    if show_progress:
        iterator = tqdm(iterator, desc="Computing features")

    for s1_id in iterator:
        pos = s1_index.get_indexer([s1_id])[0]
        if pos < 0:
            continue

        s1_vals = [_safe_str(s1_arrays[c][pos]) for c in LOOKUP_FIELDS]

        t_ids = sorted(candidate_pairs[s1_id])
        positions = target_lookup.positions(t_ids)
        for t_id, t_pos in zip(t_ids, positions):
            if t_pos < 0:
                continue
            t_vals = target_lookup.values(t_pos)
            feats = compute_pair_features(
                s1_vals[0], s1_vals[1], s1_vals[2], s1_vals[3],
                s1_vals[4], s1_vals[5], s1_vals[6],
                t_vals[0], t_vals[1], t_vals[2], t_vals[3],
                t_vals[4], t_vals[5], t_vals[6], t_vals[7],
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
