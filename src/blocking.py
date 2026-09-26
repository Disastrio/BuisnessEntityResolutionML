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
from functools import lru_cache
from itertools import zip_longest
from typing import Dict, Set, List, Tuple, Optional
import os
import re
import math
import zlib
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    MAX_CANDIDATES, ID_COL, USE_PHONETIC_BLOCK, SOUNDEX_LENGTH, BLOCK_FETCH_CAP,
    USE_TRIGRAM_BLOCK, MAX_BUCKET_IDS,
    USE_MINHASH_LSH, MINHASH_K, MINHASH_BANDS, MINHASH_ROWS, MINHASH_SHINGLE,
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


@lru_cache(maxsize=2_000_000)
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


def _strip_the(name: str) -> str:
    """Drop a leading 'the ' (cheap string op instead of a regex)."""
    return name[4:] if name.startswith('the ') else name


def _first_name_token(name: str) -> str:
    """Return the leading name token after dropping a leading 'the'."""
    tokens = _strip_the(str(name).strip()).split()
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
    rare_trigrams: Optional[Set[str]] = None,
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

    idx = {name: defaultdict(list) for name in (
        'exact_name', 'name_prefix', 'postal', 'rare_tokens', 'addr_numerics',
        'soundex', 'num_single', 'addr_rare', 'trigram', 'relaxed_name',
        'tokenset', 'postal3', 'minhash')}

    ids = target_df[ID_COL].to_numpy()
    names = target_df['name_clean'].to_numpy()
    countries = target_df['country_clean'].to_numpy()
    postals = target_df['addr_postal'].to_numpy()
    numerics = target_df['addr_numeric'].to_numpy()
    tokens_col = target_df['name_tokens'].to_numpy()
    addrs = target_df['addr_clean'].to_numpy()
    n = len(ids)
    use_rt = bool(rare_tokens)
    use_ra = bool(rare_addr_tokens)
    use_tri = bool(rare_trigrams)
    print(f"  Building all indices in a single pass over {n:,} rows...")

    for i in range(n):
        eid = ids[i]
        country = str(countries[i]).strip()
        name = str(names[i]).strip()
        if name:
            idx['exact_name'][f"{country}|{name}"].append(eid)
            cname = _strip_the(name)
            if len(cname) >= 3:
                idx['name_prefix'][f"{country}|pfx|{cname[:6]}"].append(eid)
                idx['relaxed_name'][f"{country}|rel|{cname[:4]}"].append(eid)
            if USE_PHONETIC_BLOCK:
                code = soundex(cname.split()[0], SOUNDEX_LENGTH) if cname else ''
                if code:
                    idx['soundex'][f"{country}|sndx|{code}"].append(eid)
            if use_tri:
                for gram in _trigrams(name):
                    if gram in rare_trigrams:
                        idx['trigram'][f"{country}|tri|{gram}"].append(eid)
            # Token-set key: sorted unique name tokens (word-order invariant).
            ts = str(tokens_col[i]).strip()
            if ts:
                idx['tokenset'][
                    f"{country}|tset|{'_'.join(sorted(ts.split()))}"].append(eid)
            if USE_MINHASH_LSH:
                sig = _minhash_signature(_shingle_hashes(name))
                if sig is not None:
                    for bk in _band_keys(sig):
                        idx['minhash'][f"{country}|mh|{bk}"].append(eid)
        if use_rt:
            ts = str(tokens_col[i]).strip()
            if ts:
                for tok in ts.split():
                    if tok in rare_tokens and len(tok) >= 3:
                        idx['rare_tokens'][f"{country}|rtok|{tok}"].append(eid)
        postal = str(postals[i]).strip()
        if postal:
            for code in postal.split(','):
                code = code.strip()
                if code:
                    idx['postal'][f"{country}|post|{code}"].append(eid)
                    if len(code) >= 3:
                        idx['postal3'][f"{country}|post3|{code[:3]}"].append(eid)
        nums = str(numerics[i]).strip()
        if nums:
            numset = set(nums.split())
            num_list = sorted(numset)
            if len(num_list) >= 2:
                idx['addr_numerics'][
                    f"{country}|nums|{'_'.join(num_list[:3])}"].append(eid)
            for tok in numset:
                if len(tok) >= 3:
                    idx['num_single'][f"{country}|num|{tok}"].append(eid)
        if use_ra:
            addr = str(addrs[i]).strip()
            if addr:
                for tok in set(addr.split()):
                    if tok in rare_addr_tokens and len(tok) >= 3:
                        idx['addr_rare'][f"{country}|atok|{tok}"].append(eid)

    # Freeze and drop super-keys: buckets larger than MAX_BUCKET_IDS are too
    # generic to discriminate and dominate ranking cost / noise.
    indices = {}
    dropped = 0
    for name, d in idx.items():
        frozen = {}
        for key, lst in d.items():
            if len(lst) <= MAX_BUCKET_IDS:
                frozen[key] = lst
            else:
                dropped += 1
        indices[name] = frozen
    if dropped:
        print(f"  Dropped {dropped} super-keys (> {MAX_BUCKET_IDS} ids)")
    return indices


def _rank_blocks(block_lists: List[List[str]], max_candidates: int) -> List[str]:
    """
    Rank candidate ids gathered per block.

    1) Multi-block candidates first (true matches usually co-occur across
       exact/prefix/postal/rare/phonetic/numeric/trigram blocks).
    2) Then round-robin across blocks so a single-block true match is not starved.
    Each block is pre-truncated to BLOCK_FETCH_CAP. Deterministic.
    """
    block_lists = [b[:BLOCK_FETCH_CAP] for b in block_lists]
    hits: Dict[str, int] = {}
    best_rank: Dict[str, int] = {}
    for bi, block in enumerate(block_lists):
        for eid in block:
            hits[eid] = hits.get(eid, 0) + 1
            if eid not in best_rank:
                best_rank[eid] = bi

    ordered: List[str] = [eid for eid in hits if hits[eid] >= 2]
    ordered.sort(key=lambda e: (-hits[e], best_rank[e], e))
    ordered = ordered[:max_candidates]

    # Exact-normalized-name candidates always keep top priority: a true match
    # with an exact name but no address evidence otherwise risks losing its slot
    # to distractors that merely accumulate incidental multi-block hits.
    if block_lists and block_lists[0]:
        ordered = list(dict.fromkeys(block_lists[0] + ordered))[:max_candidates]

    if len(ordered) < max_candidates:
        seen: Set[str] = set(ordered)
        max_len = max((len(b) for b in block_lists), default=0)
        for i in range(max_len):
            for block in block_lists:
                if i < len(block):
                    eid = block[i]
                    if eid in seen or hits[eid] >= 2:
                        continue
                    seen.add(eid)
                    ordered.append(eid)
                    if len(ordered) >= max_candidates:
                        return ordered
    return ordered


def _s1_components(name: str, country: str, postal: str, nums: str,
                   tokens_str: str, addr: str) -> dict:
    """Precompute all blocking components for one S1 record (once)."""
    country = country.strip()
    name = name.strip()
    cname = _strip_the(name)
    sndx = ''
    if USE_PHONETIC_BLOCK and cname:
        sndx = soundex(cname.split()[0], SOUNDEX_LENGTH)
    return {
        'country': country,
        'name': name,
        'cname': cname,
        'tokens': tokens_str.split() if tokens_str else [],
        'nums': sorted(set(nums.split())) if nums else [],
        'postals': [c.strip() for c in postal.split(',') if c.strip()] if postal else [],
        'sndx': sndx,
        'trigrams': _trigrams(name) if name else set(),
        'addr_tokens': set(addr.split()) if addr else set(),
    }


def _candidates_from_components(comp: dict, indices, rare_tokens,
                                rare_addr_tokens, rare_trigrams,
                                max_candidates: int) -> List[str]:
    """Query S2 or S3 indices for one S1 record using its precomputed components."""
    country = comp['country']

    def fetch(index_name: str, key: str) -> List[str]:
        if not key:
            return []
        return indices.get(index_name, {}).get(key, [])

    cname = comp['cname']
    block_lists: List[List[str]] = [
        fetch('exact_name', f"{country}|{comp['name']}"),
    ]
    post: List[str] = []
    for code in comp['postals']:
        post.extend(fetch('postal', f"{country}|post|{code}"))
    block_lists.append(post)
    post3: List[str] = []
    for code in comp['postals']:
        if len(code) >= 3:
            post3.extend(fetch('postal3', f"{country}|post3|{code[:3]}"))
    block_lists.append(post3)
    block_lists.append(
        fetch('name_prefix', f"{country}|pfx|{cname[:6]}") if len(cname) >= 3 else [])
    block_lists.append(
        fetch('tokenset', f"{country}|tset|{'_'.join(sorted(comp['tokens']))}")
        if comp['tokens'] else [])
    if USE_MINHASH_LSH and comp['name']:
        sig = _minhash_signature(_shingle_hashes(comp['name']))
        if sig is not None:
            mh: List[str] = []
            for bk in _band_keys(sig):
                mh.extend(fetch('minhash', f"{country}|mh|{bk}"))
            block_lists.append(mh)
        else:
            block_lists.append([])
    else:
        block_lists.append([])
    if rare_tokens:
        rare: List[str] = []
        for tok in comp['tokens']:
            if tok in rare_tokens and len(tok) >= 3:
                rare.extend(fetch('rare_tokens', f"{country}|rtok|{tok}"))
        block_lists.append(rare)
    else:
        block_lists.append([])
    if rare_trigrams:
        tri: List[str] = []
        for gram in comp['trigrams']:
            if gram in rare_trigrams:
                tri.extend(fetch('trigram', f"{country}|tri|{gram}"))
        block_lists.append(tri)
    else:
        block_lists.append([])
    block_lists.append(fetch('soundex', f"{country}|sndx|{comp['sndx']}"))
    nums = comp['nums']
    block_lists.append(
        fetch('addr_numerics', f"{country}|nums|{'_'.join(nums[:3])}")
        if len(nums) >= 2 else [])
    single: List[str] = []
    for tok in nums:
        if len(tok) >= 3:
            single.extend(fetch('num_single', f"{country}|num|{tok}"))
    block_lists.append(single)
    if rare_addr_tokens:
        addr_ids: List[str] = []
        for tok in comp['addr_tokens']:
            if tok in rare_addr_tokens and len(tok) >= 3:
                addr_ids.extend(fetch('addr_rare', f"{country}|atok|{tok}"))
        block_lists.append(addr_ids)
    else:
        block_lists.append([])
    block_lists.append(
        fetch('relaxed_name', f"{country}|rel|{cname[:4]}") if len(cname) >= 3 else [])

    return _rank_blocks(block_lists, max_candidates)


def _s1_components_iter(s1_df: pd.DataFrame):
    """Yield (entity_id, components) over an S1 DataFrame using arrays (no iterrows)."""
    ids = s1_df[ID_COL].to_numpy()
    names = s1_df['name_clean'].to_numpy()
    countries = s1_df['country_clean'].to_numpy()
    postals = s1_df['addr_postal'].to_numpy()
    numerics = s1_df['addr_numeric'].to_numpy()
    tokens = s1_df['name_tokens'].to_numpy()
    addrs = s1_df['addr_clean'].to_numpy()
    for i in range(len(ids)):
        yield ids[i], _s1_components(
            str(names[i]), str(countries[i]), str(postals[i]),
            str(numerics[i]), str(tokens[i]), str(addrs[i]))


def _ranked_candidates_for_s1(
    s1_row: pd.Series,
    indices: Dict[str, Dict[str, List[str]]],
    rare_tokens: Optional[Set[str]] = None,
    rare_addr_tokens: Optional[Set[str]] = None,
    rare_trigrams: Optional[Set[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> List[str]:
    """Single-record wrapper (kept for callers/tests)."""
    comp = _s1_components(
        str(s1_row.get('name_clean', '')), str(s1_row.get('country_clean', '')),
        str(s1_row.get('addr_postal', '')), str(s1_row.get('addr_numeric', '')),
        str(s1_row.get('name_tokens', '')), str(s1_row.get('addr_clean', '')))
    return _candidates_from_components(
        comp, indices, rare_tokens, rare_addr_tokens, rare_trigrams, max_candidates)


def generate_candidates_for_s1(
    s1_row: pd.Series,
    indices: Dict[str, Dict[str, List[str]]],
    rare_tokens: Optional[Set[str]] = None,
    rare_addr_tokens: Optional[Set[str]] = None,
    rare_trigrams: Optional[Set[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> Set[str]:
    """Set-returning wrapper around `_ranked_candidates_for_s1`."""
    return set(_ranked_candidates_for_s1(
        s1_row, indices, rare_tokens=rare_tokens,
        rare_addr_tokens=rare_addr_tokens, rare_trigrams=rare_trigrams,
        max_candidates=max_candidates,
    ))


def _trigrams(text: str, n: int = 3) -> Set[str]:
    """Character n-grams of a name with whitespace removed."""
    s = re.sub(r'\s+', '', str(text))
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def compute_rare_trigrams(
    df: pd.DataFrame,
    col: str = 'name_clean',
    min_freq: int = 2,
    max_doc_frac: float = 0.02,
) -> Set[str]:
    """
    Distinctive character trigrams (document frequency in [min_freq, max_frac]).

    Common trigrams ('the', 'ing') are excluded so the inverted index stays
    small; rare trigrams are highly discriminative and recover name variants
    (typos, transliterations) that prefix/soundex blocks miss.
    """
    doc_freq: Dict[str, int] = defaultdict(int)
    n_docs = len(df)
    for name in df[col]:
        if not name:
            continue
        for gram in _trigrams(name):
            doc_freq[gram] += 1
    max_count = max(1, int(n_docs * max_doc_frac))
    return {g for g, f in doc_freq.items() if min_freq <= f <= max_count}


# ── MinHash-LSH over name character shingles (typo/reorder recall) ────────────
_LSH_P = (1 << 31) - 1


def _lsh_params(k: int, seed: int = 1234):
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _LSH_P, size=k, dtype=np.int64)
    b = rng.randint(0, _LSH_P, size=k, dtype=np.int64)
    return a, b


_MH_A, _MH_B = _lsh_params(MINHASH_K)


def _shingle_hashes(text: str, n: int = MINHASH_SHINGLE):
    s = re.sub(r'\s+', '', text)
    if not s:
        return None
    grams = ({s} if len(s) < n else {s[i:i + n] for i in range(len(s) - n + 1)})
    return np.array([zlib.crc32(g.encode('utf-8')) & 0x7fffffff for g in grams],
                    dtype=np.int64)


def _minhash_signature(hashes):
    if hashes is None or len(hashes) == 0:
        return None
    return ((_MH_A[:, None] * hashes[None, :] + _MH_B[:, None]) % _LSH_P).min(axis=1)


def _band_keys(sig, bands: int = MINHASH_BANDS, rows: int = MINHASH_ROWS):
    return [zlib.crc32(sig[i * rows:(i + 1) * rows].tobytes()) & 0xffffffff
            for i in range(bands)]


def build_all_indices(s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute distinctive tokens and build every inverted index for S2 and S3 once.

    Returns a bundle that can be reused across many S1 chunks, so streaming
    inference pays the (expensive) index-build cost a single time instead of
    per chunk.
    """
    print("Computing distinctive tokens for S2...")
    rare_tokens_s2 = compute_rare_tokens(s2_df, col='name_tokens')
    rare_addr_s2 = compute_rare_tokens(s2_df, col='addr_clean')
    rare_tri_s2 = compute_rare_trigrams(s2_df) if USE_TRIGRAM_BLOCK else set()
    print(f"  S2: {len(rare_tokens_s2)} name tokens, {len(rare_addr_s2)} address "
          f"tokens, {len(rare_tri_s2)} trigrams")

    print("Computing distinctive tokens for S3...")
    rare_tokens_s3 = compute_rare_tokens(s3_df, col='name_tokens')
    rare_addr_s3 = compute_rare_tokens(s3_df, col='addr_clean')
    rare_tri_s3 = compute_rare_trigrams(s3_df) if USE_TRIGRAM_BLOCK else set()
    print(f"  S3: {len(rare_tokens_s3)} name tokens, {len(rare_addr_s3)} address "
          f"tokens, {len(rare_tri_s3)} trigrams")

    print("\nBuilding S2 indices...")
    s2_indices = build_candidate_indices(
        s2_df, rare_tokens=rare_tokens_s2, rare_addr_tokens=rare_addr_s2,
        rare_trigrams=rare_tri_s2)

    print("\nBuilding S3 indices...")
    s3_indices = build_candidate_indices(
        s3_df, rare_tokens=rare_tokens_s3, rare_addr_tokens=rare_addr_s3,
        rare_trigrams=rare_tri_s3)

    return {
        's2_indices': s2_indices,
        's3_indices': s3_indices,
        'rare_tokens_s2': rare_tokens_s2,
        'rare_tokens_s3': rare_tokens_s3,
        'rare_addr_s2': rare_addr_s2,
        'rare_addr_s3': rare_addr_s3,
        'rare_tri_s2': rare_tri_s2,
        'rare_tri_s3': rare_tri_s3,
    }


def _merge_ranked_sources(
    s2_ranked: List[str], s3_ranked: List[str], max_candidates: int,
) -> Set[str]:
    """
    Interleave S2/S3 ranked candidate lists so neither source is starved when the
    global cap is reached (deterministic; no set-order dependence).
    """
    combined: List[str] = []
    seen: Set[str] = set()
    for s2_id, s3_id in zip_longest(s2_ranked, s3_ranked):
        for eid in (s2_id, s3_id):
            if eid is None or eid in seen:
                continue
            seen.add(eid)
            combined.append(eid)
            if len(combined) >= max_candidates:
                return set(combined)
    return set(combined)


# Module-level state for forked blocking workers (populated in the parent).
_BLOCK_CTX: dict = {}


def _block_worker(bounds: Tuple[int, int]) -> Dict[str, Set[str]]:
    """Build candidates for a contiguous slice of S1 rows."""
    start, end = bounds
    s1_df = _BLOCK_CTX['s1_df']
    bundle = _BLOCK_CTX['bundle']
    cap = _BLOCK_CTX['cap']

    s2_i = bundle['s2_indices']
    s3_i = bundle['s3_indices']
    out: Dict[str, Set[str]] = {}
    for s1_id, comp in _s1_components_iter(s1_df.iloc[start:end]):
        s2_ranked = _candidates_from_components(
            comp, s2_i, bundle['rare_tokens_s2'], bundle['rare_addr_s2'],
            bundle['rare_tri_s2'], cap)
        s3_ranked = _candidates_from_components(
            comp, s3_i, bundle['rare_tokens_s3'], bundle['rare_addr_s3'],
            bundle['rare_tri_s3'], cap)
        out[s1_id] = _merge_ranked_sources(s2_ranked, s3_ranked, cap)
    return out


def _generate_candidates_parallel(
    s1_df: pd.DataFrame,
    bundle: Dict[str, object],
    max_candidates: int,
    n_workers: int,
) -> Dict[str, Set[str]]:
    s1 = s1_df.reset_index(drop=True)
    n = len(s1)
    global _BLOCK_CTX
    _BLOCK_CTX = {'s1_df': s1, 'bundle': bundle, 'cap': max_candidates}

    bounds = np.linspace(0, n, n_workers + 1).astype(int)
    ranges = [(int(bounds[i]), int(bounds[i + 1]))
              for i in range(n_workers) if bounds[i] < bounds[i + 1]]

    result: Dict[str, Set[str]] = {}
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=len(ranges)) as pool:
        for part in pool.imap_unordered(_block_worker, ranges):
            result.update(part)
    return result


def generate_candidates_from_bundle(
    s1_df: pd.DataFrame,
    bundle: Dict[str, object],
    max_candidates: int = MAX_CANDIDATES,
    show_progress: bool = True,
    n_workers: int = 1,
) -> Dict[str, Set[str]]:
    """
    Generate candidates for a set of S1 rows using a prebuilt index bundle.

    This is the chunk-friendly entry point: call `build_all_indices` once, then
    invoke this per S1 chunk during streaming inference. Pass n_workers > 1 to
    parallelise the per-S1 query loop across processes (fork only).
    """
    if n_workers and n_workers > 1 and hasattr(os, 'fork') and len(s1_df) >= 2000:
        return _generate_candidates_parallel(s1_df, bundle, max_candidates, int(n_workers))

    s2_indices = bundle['s2_indices']
    s3_indices = bundle['s3_indices']

    all_candidates: Dict[str, Set[str]] = {}
    iterator = _s1_components_iter(s1_df)
    if show_progress:
        iterator = tqdm(iterator, total=len(s1_df), desc="Blocking")

    for s1_id, comp in iterator:
        s2_ranked = _candidates_from_components(
            comp, s2_indices, bundle['rare_tokens_s2'], bundle['rare_addr_s2'],
            bundle['rare_tri_s2'], max_candidates)
        s3_ranked = _candidates_from_components(
            comp, s3_indices, bundle['rare_tokens_s3'], bundle['rare_addr_s3'],
            bundle['rare_tri_s3'], max_candidates)
        all_candidates[s1_id] = _merge_ranked_sources(
            s2_ranked, s3_ranked, max_candidates)

    return all_candidates


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int = MAX_CANDIDATES,
    show_progress: bool = True,
    n_workers: int = 1,
) -> Dict[str, Set[str]]:
    """
    Generate candidate pairs for all S1 entities against S2 and S3.

    Convenience wrapper: builds the indices, then delegates to
    `generate_candidates_from_bundle`.

    Returns:
        Dict: {s1_id: set(candidate_s2_s3_ids)}
    """
    bundle = build_all_indices(s2_df, s3_df)
    print(f"\nGenerating candidates for {len(s1_df)} S1 entities "
          f"(workers={n_workers})...")
    return generate_candidates_from_bundle(
        s1_df, bundle, max_candidates=max_candidates,
        show_progress=show_progress, n_workers=n_workers,
    )


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
