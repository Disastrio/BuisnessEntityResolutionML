"""
normalize.py — Multilingual Name & Address Normalization for Business Entity Resolution.

Handles:
- Unicode NFKD normalization (French accents: é → e)
- Legal suffix canonicalization (pvt ltd → private limited, etc.)
- Ampersand normalization (& → and)
- Address component extraction (postal codes, numeric tokens, city/state)
- Abbreviation expansion (rd → road, st → street, etc.)
- Whitespace and punctuation cleanup
- Maintains both raw and normalized representations to avoid over-normalization
"""
import os
import re
import unicodedata
import multiprocessing as mp
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.config import NAME_COL, ADDR_COL, COUNTRY_COL, ID_COL

# ── Legal Suffix Canonicalization ─────────────────────────────────────────────
# Ordered longest-first to prevent partial matches
LEGAL_SUFFIXES = [
    # Full → canonical
    (r'\bprivate\s+limited\b', 'private limited'),
    (r'\bpvt\s*\.?\s*ltd\s*\.?\b', 'private limited'),
    (r'\bpvt\b', 'private'),
    (r'\bltd\s*\.?\b', 'limited'),
    (r'\blimited\b', 'limited'),
    (r'\bcorp\s*\.?\b', 'corporation'),
    (r'\bcorporation\b', 'corporation'),
    (r'\binc\s*\.?\b', 'incorporated'),
    (r'\bincorporated\b', 'incorporated'),
    (r'\bllc\s*\.?\b', 'limited liability company'),
    (r'\bl\.?l\.?c\.?\b', 'limited liability company'),
    (r'\bllp\s*\.?\b', 'limited liability partnership'),
    (r'\bl\.?l\.?p\.?\b', 'limited liability partnership'),
    (r'\bco\s*\.?\b', 'company'),
    (r'\bcompany\b', 'company'),
    (r'\benterprises?\b', 'enterprise'),
    (r'\bintl\s*\.?\b', 'international'),
    (r'\binternational\b', 'international'),
    (r'\bsvc\s*\.?\b', 'services'),
    (r'\bservices?\b', 'services'),
    (r'\btechnolog(?:y|ies)\b', 'technology'),
    (r'\btech\s*\.?\b', 'technology'),
    (r'\bsoln\s*\.?\b', 'solutions'),
    (r'\bsolutions?\b', 'solutions'),
    (r'\bindust(?:ry|ries)\b', 'industries'),
    (r'\bgrp\s*\.?\b', 'group'),
    (r'\bgroup\b', 'group'),
    (r'\bassoc\s*\.?\b', 'associates'),
    (r'\bassociates?\b', 'associates'),
    (r'\bfound\s*\.?\b', 'foundation'),
    (r'\bfoundation\b', 'foundation'),
    (r'\bmfg\s*\.?\b', 'manufacturing'),
    (r'\bmanufacturing\b', 'manufacturing'),
    (r'\bdist\s*\.?\b', 'distributors'),
    (r'\bdistributors?\b', 'distributors'),
    (r'\bhldg[s]?\s*\.?\b', 'holdings'),
    (r'\bholdings?\b', 'holdings'),
]

# ── Address Abbreviation Expansion ────────────────────────────────────────────
ADDRESS_ABBREVS = [
    (r'\brd\s*\.?\b', 'road'),
    (r'\bst\s*\.?\b', 'street'),
    (r'\bave?\s*\.?\b', 'avenue'),
    (r'\bblvd\s*\.?\b', 'boulevard'),
    (r'\bdr\s*\.?\b', 'drive'),
    (r'\bln\s*\.?\b', 'lane'),
    (r'\bct\s*\.?\b', 'court'),
    (r'\bpl\s*\.?\b', 'place'),
    (r'\bpkwy\s*\.?\b', 'parkway'),
    (r'\bhwy\s*\.?\b', 'highway'),
    (r'\bcir\s*\.?\b', 'circle'),
    (r'\bsq\s*\.?\b', 'square'),
    (r'\bfwy\s*\.?\b', 'freeway'),
    (r'\bapt\s*\.?\b', 'apartment'),
    (r'\bste\s*\.?\b', 'suite'),
    (r'\bfl\s*\.?\b', 'floor'),
    (r'\bbldg\s*\.?\b', 'building'),
    (r'\bdist\s*\.?\b', 'district'),
    (r'\bnr\s*\.?\b', 'near'),
    (r'\bopp\s*\.?\b', 'opposite'),
]

# ── Postal Code Patterns ─────────────────────────────────────────────────────
# India: 6-digit PIN (e.g. 110001)
# US: 5-digit ZIP or ZIP+4 (e.g. 90210, 90210-1234)
# France: 5-digit (e.g. 75001)
POSTAL_PATTERNS = [
    re.compile(r'\b(\d{6})\b'),        # India PIN
    re.compile(r'\b(\d{5}(?:-\d{4})?)\b'),  # US ZIP / France
]

# ── Country Normalization ─────────────────────────────────────────────────────
COUNTRY_MAP = {
    'us': 'us', 'usa': 'us', 'united states': 'us', 'united states of america': 'us',
    'u.s.': 'us', 'u.s.a.': 'us', 'america': 'us',
    'in': 'india', 'ind': 'india', 'india': 'india', 'bharat': 'india',
    'fr': 'france', 'fra': 'france', 'france': 'france',
}


# ═════════════════════════════════════════════════════════════════════════════
# Core Normalization Functions
# ═════════════════════════════════════════════════════════════════════════════

def unicode_normalize(text: str) -> str:
    """NFKD Unicode normalization → strip accents → ASCII-safe lowercase."""
    if not text:
        return ""
    # Decompose characters, then strip combining marks (accents)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_text.lower().strip()


def normalize_country(country: str) -> str:
    """Normalize country to canonical form. Treats as open string — never hardcodes."""
    if not country:
        return ""
    cleaned = unicode_normalize(country).strip()
    # Remove punctuation for lookup
    lookup = re.sub(r'[^a-z\s]', '', cleaned).strip()
    return COUNTRY_MAP.get(lookup, cleaned)


def normalize_name(name: str) -> str:
    """
    Normalize business name:
    1. Unicode NFKD → lowercase
    2. Ampersand → 'and'
    3. Legal suffix canonicalization
    4. Remove punctuation (keep alphanumeric + spaces)
    5. Collapse whitespace
    """
    if not name:
        return ""

    text = unicode_normalize(name)

    # Ampersand → and
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Legal suffix canonicalization
    for pattern, replacement in LEGAL_SUFFIXES:
        text = re.sub(pattern, replacement, text)

    # Remove punctuation but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_address(address: str) -> str:
    """
    Normalize address:
    1. Unicode NFKD → lowercase
    2. Expand abbreviations
    3. Remove punctuation (keep alphanumeric + spaces)
    4. Collapse whitespace
    """
    if not address:
        return ""

    text = unicode_normalize(address)

    # Ampersand → and
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Expand address abbreviations
    for pattern, replacement in ADDRESS_ABBREVS:
        text = re.sub(pattern, replacement, text)

    # Remove punctuation but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_postal_codes(address: str) -> str:
    """
    Extract all postal/PIN/ZIP codes from address.
    Returns comma-separated string of found codes, or empty string.
    """
    if not address:
        return ""
    codes = []
    for pattern in POSTAL_PATTERNS:
        codes.extend(pattern.findall(address))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return ','.join(unique)


def extract_numeric_tokens(text: str) -> str:
    """
    Extract all numeric tokens (house numbers, floor, ward, plot, etc.)
    Returns space-separated string of numeric tokens.
    """
    if not text:
        return ""
    nums = re.findall(r'\b\d+\b', text)
    return ' '.join(nums)


def extract_name_tokens(name: str) -> str:
    """Extract alphabetic tokens from normalized name, sorted for set comparison."""
    if not name:
        return ""
    tokens = re.findall(r'[a-z]+', name.lower())
    return ' '.join(sorted(tokens))


def name_prefix(name: str, length: int = 6) -> str:
    """Extract first N characters of normalized name for blocking."""
    if not name:
        return ""
    # Strip common prefixes like 'the'
    cleaned = re.sub(r'^the\s+', '', name)
    return cleaned[:length]


# ═════════════════════════════════════════════════════════════════════════════
# DataFrame-Level Normalization
# ═════════════════════════════════════════════════════════════════════════════

def normalize_dataframe(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
    """
    Apply all normalizations to a source DataFrame.

    Adds columns:
        - name_clean:          Normalized business name
        - name_tokens:         Sorted alphabetic tokens from name
        - name_prefix:         First 6 chars of cleaned name (for blocking)
        - addr_clean:          Normalized address
        - addr_postal:         Extracted postal/PIN/ZIP codes
        - addr_numeric:        Extracted numeric tokens from address
        - country_clean:       Normalized country
        - source:              Source label ('S1', 'S2', 'S3')

    Preserves original columns for feature engineering (raw vs normalized).
    """
    out = df.copy()

    # Ensure string types and fill NaN
    for col in [NAME_COL, ADDR_COL, COUNTRY_COL]:
        if col in out.columns:
            out[col] = out[col].fillna('').astype(str)

    # Normalize name
    out['name_clean'] = out[NAME_COL].apply(normalize_name)
    out['name_tokens'] = out['name_clean'].apply(extract_name_tokens)
    out['name_prefix'] = out['name_clean'].apply(name_prefix)

    # Normalize address
    out['addr_clean'] = out[ADDR_COL].apply(normalize_address)
    out['addr_postal'] = out[ADDR_COL].apply(
        lambda x: extract_postal_codes(unicode_normalize(x))
    )
    out['addr_numeric'] = out['addr_clean'].apply(extract_numeric_tokens)

    # Normalize country (open-string; no hardcoding!)
    out['country_clean'] = out[COUNTRY_COL].apply(normalize_country)

    # Source label
    if source_label:
        out['source'] = source_label

    return out


# Module-level state for forked normalization workers (populated in the parent).
_NORM_CTX: dict = {}


def _normalize_worker(label: str) -> pd.DataFrame:
    return normalize_dataframe(_NORM_CTX[label], source_label=label)


def normalize_all_sources(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    n_workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize all three source DataFrames, tagging each with source label.

    With n_workers > 1 (fork only), S1/S2/S3 are normalized concurrently in
    separate processes so the three independent sources use multiple cores.
    """
    if n_workers and n_workers > 1 and hasattr(os, 'fork'):
        global _NORM_CTX
        _NORM_CTX = {'S1': s1, 'S2': s2, 'S3': s3}
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=3) as pool:
            s1_norm, s2_norm, s3_norm = pool.map(_normalize_worker, ['S1', 'S2', 'S3'])
        return s1_norm, s2_norm, s3_norm

    s1_norm = normalize_dataframe(s1, source_label='S1')
    s2_norm = normalize_dataframe(s2, source_label='S2')
    s3_norm = normalize_dataframe(s3, source_label='S3')
    return s1_norm, s2_norm, s3_norm


# ═════════════════════════════════════════════════════════════════════════════
# Benchmarking & Diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_normalization(df: pd.DataFrame, n_samples: int = 10000) -> dict:
    """
    Quick benchmark of normalization speed and quality on a sample.
    Returns dict with timing and example results.
    """
    import time

    sample = df.head(n_samples).copy()

    start = time.time()
    normalized = normalize_dataframe(sample, source_label='BENCH')
    elapsed = time.time() - start

    records_per_sec = n_samples / elapsed if elapsed > 0 else float('inf')

    results = {
        'n_samples': n_samples,
        'elapsed_seconds': round(elapsed, 3),
        'records_per_second': round(records_per_sec, 1),
        'empty_names_pct': round(
            (normalized['name_clean'] == '').mean() * 100, 2
        ),
        'empty_addrs_pct': round(
            (normalized['addr_clean'] == '').mean() * 100, 2
        ),
        'has_postal_pct': round(
            (normalized['addr_postal'] != '').mean() * 100, 2
        ),
        'unique_countries': sorted(
            normalized['country_clean'].unique().tolist()
        ),
    }

    return results


if __name__ == '__main__':
    # Quick smoke test
    print("=== Normalization Smoke Test ===\n")

    # Test name normalization
    test_names = [
        "ABC Pvt. Ltd.",
        "XYZ Corp.",
        "Smith & Sons Inc",
        "THE Global Tech Solutions LLP",
        "RÉSEAU Française de Commerce",
    ]
    for name in test_names:
        print(f"  '{name}' → '{normalize_name(name)}'")

    print()

    # Test address normalization
    test_addrs = [
        "123 Main St., Apt. 4B, New York, NY 10001",
        "Plot No. 42, Industrial Area, Phase-II, Chandigarh 160002",
        "15 Rue de la Paix, 75002 Paris, France",
        "Near SBI ATM, MG Rd, Bangalore 560001",
    ]
    for addr in test_addrs:
        clean = normalize_address(addr)
        postal = extract_postal_codes(unicode_normalize(addr))
        numeric = extract_numeric_tokens(clean)
        print(f"  '{addr}'")
        print(f"    → clean:   '{clean}'")
        print(f"    → postal:  '{postal}'")
        print(f"    → numeric: '{numeric}'")
        print()

    # Test country normalization
    test_countries = ["US", "USA", "united states", "India", "IN", "France", "FR", "BRASIL"]
    for c in test_countries:
        print(f"  '{c}' → '{normalize_country(c)}'")
