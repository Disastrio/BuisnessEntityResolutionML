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
    Index 6: Soundex / phonetic prefix of the leading name token
    Index 7: Relaxed — first 4 chars of name + country (catch-all)

Candidates are accumulated in index-reliability order and truncated at
MAX_CANDIDATES, so the retained set is deterministic (stable across runs and
PYTHONHASHSEED values) and keeps the highest-precision blocks first.
"""
from collections import defaultdict
from itertools import zip_longest
from typing import Dict, Set, List, Tuple, Optional
import re
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    MAX_CANDIDATES, ID_COL, USE_PHONETIC_BLOCK, SOUNDEX_LENGTH, BLOCK_FETCH_CAP,
)


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
# Phonetic (Soundex) Helpers — Index 6
# ═════════════════════════════════════════════════════════════════════════════

# Standard American Soundex letter → digit grouping.
_SOUNDEX_GROUPS = [
    ('bfpv', '1'),
    ('cgjkqsxz', '2'),
    ('dt', '3'),
    ('l', '4'),
    ('mn', '5'),
    ('r', '6'),
]
_SOUNDEX_MAP: Dict[str, str] = {
    ch: digit for letters, digit in _SOUNDEX_GROUPS for ch in letters
}


def soundex(token: str, length: int = SOUNDEX_LENGTH) -> str:
    """
    Compute the Soundex code of a token (ASCII letters only).

    Collapses adjacent identical codes and drops vowels / h / w (which reset the
    previous code), which makes it robust to transliteration and small typos
    (e.g. 'kalyan' and 'calian' both encode to 'K450').

    Returns an uppercase code of exactly `length` characters, or '' for tokens
    with no alphabetic characters.
    """
    if not token:
        return ""
    letters = re.sub(r'[^a-z]', '', str(token).lower())
    if not letters:
        return ""

    first = letters[0]
    out = first
    prev = _SOUNDEX_MAP.get(first, '')
    for ch in letters[1:]:
        code = _SOUNDEX_MAP.get(ch, '')
        if code and code != prev:
            out += code
        if ch not in 'hw':   # h/w are transparent; vowels reset the chain (code='')
            prev = code
    return (out.upper() + '0' * length)[:length]


def _first_name_token(name: str) -> str:
    """Return the leading name token after dropping a leading 'the'."""
    cleaned = re.sub(r'^the\s+', '', str(name).strip())
    tokens = cleaned.split()
    return tokens[0] if tokens else ''


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
    col: str = 'name_tokens',
) -> Set[str]:
    """
    Find tokens that appear in at least min_freq documents but in at most
    max_doc_frac fraction of all documents. These are distinctive tokens
    useful for blocking.

    `col` selects the token column ('name_tokens' or 'addr_clean'); address
    blocking reuses this to find distinctive street/area tokens.
    """
    doc_freq = defaultdict(int)
    n_docs = len(df)

    for tokens_str in df[col]:
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
    rare_addr_tokens: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build all inverted indices for a target source (S2 or S3).

    Args:
        target_df: Normalized S2 or S3 DataFrame.
        rare_tokens: Distinctive name tokens (Index 4).
        rare_addr_tokens: Distinctive address tokens (Index 8) — lets pairs
            connect on address alone, which matters when the noisy name is blank.

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

    # Index 6: Phonetic (Soundex) prefix of leading name token
    if USE_PHONETIC_BLOCK:
        print("  Building Index 6 (phonetic / soundex)...")
        idx6 = defaultdict(list)
        for eid, name, country in zip(
            target_df[ID_COL], target_df['name_clean'], target_df['country_clean']
        ):
            country = str(country).strip()
            token = _first_name_token(name)
            if token:
                code = soundex(token, SOUNDEX_LENGTH)
                if code:
                    idx6[f"{country}|sndx|{code}"].append(eid)
        indices['soundex'] = dict(idx6)

    # Index 5b: Single numeric address token (>= 3 digits) + country.
    # Catches pairs that share exactly one meaningful number (plot/house/PIN
    # fragment); Index 5 requires >= 2 shared numbers and misses these.
    print("  Building Index 5b (single numeric tokens)...")
    idx5b = defaultdict(list)
    for eid, nums, country in zip(
        target_df[ID_COL], target_df['addr_numeric'], target_df['country_clean']
    ):
        nums = str(nums).strip()
        country = str(country).strip()
        if nums:
            for tok in set(nums.split()):
                if len(tok) >= 3:
                    idx5b[f"{country}|num|{tok}"].append(eid)
    indices['num_single'] = dict(idx5b)

    # Index 8: Rare address tokens (street/area names) + country.
    # Primary connect signal for records whose business name is blank.
    if rare_addr_tokens:
        print(f"  Building Index 8 (rare address tokens, "
              f"{len(rare_addr_tokens)} tokens)...")
        idx8 = defaultdict(list)
        for eid, addr, country in zip(
            target_df[ID_COL], target_df['addr_clean'], target_df['country_clean']
        ):
            addr = str(addr).strip()
            country = str(country).strip()
            if addr:
                for tok in set(addr.split()):
                    if tok in rare_addr_tokens and len(tok) >= 3:
                        idx8[f"{country}|atok|{tok}"].append(eid)
        indices['addr_rare'] = dict(idx8)

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


def _ranked_candidates_for_s1(
    s1_row: pd.Series,
    indices: Dict[str, Dict[str, List[str]]],
    rare_tokens: Optional[Set[str]] = None,
    rare_addr_tokens: Optional[Set[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> List[str]:
    """
    Generate candidate IDs for a single S1 entity by querying all indices.

    Each block contributes its own list; the final set is built by round-robin
    interleaving across blocks, so a huge high-precision block cannot starve
    later blocks (relaxed name, phonetics, single numerics, address tokens) out
    of the candidate cap — the failure mode that previously dropped true
    matches. Deterministic and bounded: blocks are fetched in high-precision
    order and each block is capped at BLOCK_FETCH_CAP before interleaving.
    """
    name = str(s1_row.get('name_clean', '')).strip()
    country = str(s1_row.get('country_clean', '')).strip()
    postal = str(s1_row.get('addr_postal', '')).strip()
    nums = str(s1_row.get('addr_numeric', '')).strip()
    tokens_str = str(s1_row.get('name_tokens', '')).strip()
    addr = str(s1_row.get('addr_clean', '')).strip()
    clean_name = re.sub(r'^the\s+', '', name)

    def fetch(index_name: str, key: str) -> List[str]:
        if not key:
            return []
        return indices.get(index_name, {}).get(key, [])

    block_lists: List[List[str]] = []

    # 1: Exact name
    block_lists.append(fetch('exact_name', f"{country}|{name}"))

    # 3: Postal code
    post_ids: List[str] = []
    if postal:
        for code in postal.split(','):
            code = code.strip()
            post_ids.extend(fetch('postal', f"{country}|post|{code}"))
    block_lists.append(post_ids)

    # 2: Name prefix
    if len(clean_name) >= 3:
        block_lists.append(fetch('name_prefix', f"{country}|pfx|{clean_name[:6]}"))
    else:
        block_lists.append([])

    # 4: Rare name tokens
    rare_ids: List[str] = []
    if rare_tokens and tokens_str:
        for tok in tokens_str.split():
            if tok in rare_tokens and len(tok) >= 3:
                rare_ids.extend(fetch('rare_tokens', f"{country}|rtok|{tok}"))
    block_lists.append(rare_ids)

    # 6: Phonetic (Soundex) prefix of leading token
    sndx_ids: List[str] = []
    if USE_PHONETIC_BLOCK:
        token = _first_name_token(name)
        if token:
            sndx_ids = fetch('soundex', f"{country}|sndx|{soundex(token, SOUNDEX_LENGTH)}")
    block_lists.append(sndx_ids)

    # 5: Combined address numerics (>= 2 shared)
    comb_ids: List[str] = []
    if nums:
        num_list = sorted(set(nums.split()))
        if len(num_list) >= 2:
            comb_ids = fetch('addr_numerics',
                             f"{country}|nums|{'_'.join(num_list[:3])}")
    block_lists.append(comb_ids)

    # 5b: Single meaningful numeric tokens (>= 3 digits)
    single_ids: List[str] = []
    if nums:
        for tok in set(nums.split()):
            if len(tok) >= 3:
                single_ids.extend(fetch('num_single', f"{country}|num|{tok}"))
    block_lists.append(single_ids)

    # 8: Rare address tokens (connect blank-name records by address)
    addr_ids: List[str] = []
    if rare_addr_tokens and addr:
        for tok in set(addr.split()):
            if tok in rare_addr_tokens and len(tok) >= 3:
                addr_ids.extend(fetch('addr_rare', f"{country}|atok|{tok}"))
    block_lists.append(addr_ids)

    # 7: Relaxed name (lowest precision catch-all)
    if len(clean_name) >= 3:
        block_lists.append(fetch('relaxed_name', f"{country}|rel|{clean_name[:4]}"))
    else:
        block_lists.append([])

    # Bound per-block work, then round-robin interleave (dedup across blocks).
    block_lists = [b[:BLOCK_FETCH_CAP] for b in block_lists]
    ordered: List[str] = []
    seen: Set[str] = set()
    max_len = max((len(b) for b in block_lists), default=0)
    for i in range(max_len):
        for block in block_lists:
            if i < len(block):
                eid = block[i]
                if eid not in seen:
                    seen.add(eid)
                    ordered.append(eid)
                    if len(ordered) >= max_candidates:
                        return ordered
    return ordered


def generate_candidates_for_s1(
    s1_row: pd.Series,
    indices: Dict[str, Dict[str, List[str]]],
    rare_tokens: Optional[Set[str]] = None,
    rare_addr_tokens: Optional[Set[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> Set[str]:
    """Set-returning wrapper around `_ranked_candidates_for_s1`."""
    return set(_ranked_candidates_for_s1(
        s1_row, indices, rare_tokens=rare_tokens,
        rare_addr_tokens=rare_addr_tokens, max_candidates=max_candidates,
    ))


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
    print("Computing distinctive tokens for S2...")
    rare_tokens_s2 = compute_rare_tokens(s2_df, col='name_tokens')
    rare_addr_s2 = compute_rare_tokens(s2_df, col='addr_clean')
    print(f"  S2: {len(rare_tokens_s2)} name tokens, {len(rare_addr_s2)} address tokens")

    print("Computing distinctive tokens for S3...")
    rare_tokens_s3 = compute_rare_tokens(s3_df, col='name_tokens')
    rare_addr_s3 = compute_rare_tokens(s3_df, col='addr_clean')
    print(f"  S3: {len(rare_tokens_s3)} name tokens, {len(rare_addr_s3)} address tokens")

    print("\nBuilding S2 indices...")
    s2_indices = build_candidate_indices(
        s2_df, rare_tokens=rare_tokens_s2, rare_addr_tokens=rare_addr_s2)

    print("\nBuilding S3 indices...")
    s3_indices = build_candidate_indices(
        s3_df, rare_tokens=rare_tokens_s3, rare_addr_tokens=rare_addr_s3)

    # Generate candidates for each S1 entity
    print(f"\nGenerating candidates for {len(s1_df)} S1 entities...")
    all_candidates: Dict[str, Set[str]] = {}

    iterator = s1_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(s1_df), desc="Blocking")

    for _, s1_row in iterator:
        s1_id = s1_row[ID_COL]

        # Ranked candidate lists per source (highest-precision blocks first)
        s2_ranked = _ranked_candidates_for_s1(
            s1_row, s2_indices,
            rare_tokens=rare_tokens_s2,
            rare_addr_tokens=rare_addr_s2,
            max_candidates=max_candidates,
        )
        s3_ranked = _ranked_candidates_for_s1(
            s1_row, s3_indices,
            rare_tokens=rare_tokens_s3,
            rare_addr_tokens=rare_addr_s3,
            max_candidates=max_candidates,
        )

        # Interleave S2/S3 by block rank so neither source is starved when the
        # global cap is reached (deterministic, no set-order dependence).
        combined: List[str] = []
        seen: Set[str] = set()
        for s2_id, s3_id in zip_longest(s2_ranked, s3_ranked):
            for eid in (s2_id, s3_id):
                if eid is None or eid in seen:
                    continue
                seen.add(eid)
                combined.append(eid)
                if len(combined) >= max_candidates:
                    break
            if len(combined) >= max_candidates:
                break

        all_candidates[s1_id] = set(combined)

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
