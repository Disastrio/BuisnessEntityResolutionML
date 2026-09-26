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
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.config import NAME_COL, ADDR_COL, COUNTRY_COL, ID_COL

# ── Legal Suffix Canonicalization ─────────────────────────────────────────────
# Ordered longest-first to prevent partial matches
LEGAL_SUFFIXES = [
    # ── Continental European forms (France/Europe appear in the test set) ──
    (r'\bsociete\s+a\s+responsabilite\s+limitee\b', 'sarl'),
    (r'\bsociete\s+par\s+actions\s+simplifiee\b', 'sas'),
    (r'\bsociete\s+anonyme\b', 'sa'),
    (r'\bs\.?a\.?r\.?l\.?\b', 'sarl'),
    (r'\bs\.?a\.?s\.?u\b', 'sasu'),
    (r'\bs\.?a\.?s\.?\b', 'sas'),
    (r'\bs\.?c\.?i\.?\b', 'sci'),
    (r'\be\.?u\.?r\.?l\.?\b', 'eurl'),
    (r'\bgmbh\b', 'gmbh'),
    (r'\bs\.?a\.?\b', 'sa'),
    # ── Anglo-American / Indian forms (longest-first) ──
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
    # Regional (India / France) urban tokens
    (r'\bsec(?:t)?\s*\.?\b', 'sector'),
    (r'\bph(?:s)?\s*\.?\b', 'phase'),
    (r'\bflr\s*\.?\b', 'floor'),
    (r'\bbeh\s*\.?\b', 'behind'),
    (r'\bblog\s*\.?\b', 'block'),
    (r'\bbk\s*\.?\b', 'block'),
    (r'\bboul\s*\.?\b', 'boulevard'),
    # French street types (accents already folded by NFKD)
    (r'\bav\s*\.?\b', 'avenue'),
    (r'\bbd\s*\.?\b', 'boulevard'),
    (r'\brue\b', 'rue'),
    (r'\ballee\b', 'allee'),
    (r'\bchemin\b', 'chemin'),
    (r'\bimpasse\b', 'impasse'),
    (r'\bquai\b', 'quai'),
]

# ── Postal Code Patterns ─────────────────────────────────────────────────────
# India: 6-digit PIN (e.g. 110001)
# US: 5-digit ZIP or ZIP+4 (e.g. 90210, 90210-1234)
# France: 5-digit (e.g. 75001)
# India: 6-digit PIN (first digit 1-9). US/France: 5-digit ZIP (ZIP+4 allowed).
_INDIA_PIN_RE = re.compile(r'\b[1-9]\d{5}\b')
_FIVE_DIGIT_RE = re.compile(r'\b\d{5}(?:-\d{4})?\b')
_ANY_POSTAL_RE = re.compile(r'\b\d{5,6}\b')

# ── Precompiled replacement passes ────────────────────────────────────────────
# NOTE: these must stay SEQUENTIAL — a single combined alternation changes the
# result because later patterns would no longer see earlier replacements (e.g.
# 'co intl' -> 'company intl' vs 'companyinternational'). Precompiling avoids
# per-call pattern compilation while preserving exact semantics.
def _compile_with_literal(pairs):
    """
    Compile each pattern together with a literal keyword that MUST appear for a
    match. Guarding on it skips the vast majority of regex passes per row while
    preserving exact results (if the literal is absent, the regex cannot match).
    """
    out = []
    for pat, rep in pairs:
        bare = re.sub(r'\\[a-zA-Z]', ' ', pat)   # drop \b, \s, \d, ... escapes
        m = re.search(r'[a-z]{2,}', bare)
        out.append(((m.group(0) if m else ''), re.compile(pat), rep))
    return out


_SUFFIX_COMPILED = _compile_with_literal(LEGAL_SUFFIXES)
_ABBREV_COMPILED = _compile_with_literal(ADDRESS_ABBREVS)


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


def normalize_name_pre(text: str) -> str:
    """Name canonicalization for already NFKD-normalized (lowercase) text."""
    if not text:
        return ""
    text = re.sub(r'\s*&\s*', ' and ', text)
    for lit, rx, replacement in _SUFFIX_COMPILED:
        if not lit or lit in text:
            text = rx.sub(replacement, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_name(name: str) -> str:
    """
    Normalize business name: NFKD/lowercase, '&'->'and', legal-suffix
    canonicalization (incl. French/EU forms), punctuation/whitespace cleanup.
    """
    return normalize_name_pre(unicode_normalize(name))


def normalize_address_pre(text: str) -> str:
    """Address canonicalization for already NFKD-normalized (lowercase) text."""
    if not text:
        return ""
    text = re.sub(r'\s*&\s*', ' and ', text)
    # Split digit<->letter runs so '42b' -> '42 b' (recovers house numbers for
    # numeric blocking/features; '42b' previously yielded no numeric token).
    text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)
    text = re.sub(r'([a-z]+)(\d+)', r'\1 \2', text)
    for lit, rx, replacement in _ABBREV_COMPILED:
        if not lit or lit in text:
            text = rx.sub(replacement, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_address(address: str) -> str:
    """Normalize address: NFKD/lowercase, abbreviation expansion, cleanup."""
    return normalize_address_pre(unicode_normalize(address))


def extract_postal_codes(address: str, country: str = "") -> str:
    """
    Country-aware postal/PIN/ZIP extraction (input already NFKD-normalized).

    Conditioning on country prevents incidental 5-digit street/plot numbers in
    Indian addresses from being treated as US/French ZIPs (false-positive merges).
    """
    if not address:
        return ""
    if country == 'india':
        codes = _INDIA_PIN_RE.findall(address)
    elif country in ('us', 'france'):
        codes = _FIVE_DIGIT_RE.findall(address)
    else:
        codes = _ANY_POSTAL_RE.findall(address)
    return ','.join(dict.fromkeys(codes))   # order-preserving dedupe


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

    # NFKD-normalize each raw column exactly ONCE (was re-decomposed per field).
    name_nfkd = [unicode_normalize(x) for x in out[NAME_COL]]
    addr_nfkd = [unicode_normalize(x) for x in out[ADDR_COL]]
    country_nfkd = [unicode_normalize(x) for x in out[COUNTRY_COL]]

    # Names (list comprehensions are ~2x faster than Series.apply)
    names_clean = [normalize_name_pre(n) for n in name_nfkd]
    out['name_clean'] = names_clean
    out['name_tokens'] = [' '.join(sorted(re.findall(r'[a-z]+', n)))
                          for n in names_clean]
    out['name_prefix'] = [(n[4:] if n.startswith('the ') else n)[:6]
                          for n in names_clean]

    # Countries (open-string; no hardcoding)
    countries = []
    for c in country_nfkd:
        lookup = re.sub(r'[^a-z\s]', '', c).strip()
        countries.append(COUNTRY_MAP.get(lookup, c))
    out['country_clean'] = countries

    # Addresses
    addrs_clean = [normalize_address_pre(a) for a in addr_nfkd]
    out['addr_clean'] = addrs_clean
    out['addr_numeric'] = [' '.join(re.findall(r'\b\d+\b', a)) if a else ''
                           for a in addrs_clean]

    # Postal codes conditioned on the record's country
    out['addr_postal'] = [extract_postal_codes(a, c)
                          for a, c in zip(addr_nfkd, countries)]

    # Source label
    if source_label:
        out['source'] = source_label

    return out


# Module-level state for forked normalization workers (populated in the parent).
_NORM_CTX: dict = {}


def _normalize_chunk(task):
    label, start, end = task
    chunk = _NORM_CTX[label].iloc[start:end]
    return label, start, normalize_dataframe(chunk, source_label=label)


def _normalize_chunk_spawn(task):
    """Windows path: the chunk is passed in (spawn has no inherited globals)."""
    label, chunk = task
    return label, normalize_dataframe(chunk, source_label=label)


def normalize_all_sources(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    n_workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize all three source DataFrames, tagging each with source label.

    With n_workers > 1 (fork only), each source is split into chunks and
    normalized across a process pool sized to `n_workers`, so normalization
    scales with CPU cores (previously fixed at 3, one process per source).
    Results are reassembled in original row order.
    """
    per = max(1, int(n_workers) // 3) if n_workers else 1

    if n_workers and n_workers > 1 and hasattr(os, 'fork'):
        global _NORM_CTX
        _NORM_CTX = {'S1': s1, 'S2': s2, 'S3': s3}
        tasks = []
        for label, df in (('S1', s1), ('S2', s2), ('S3', s3)):
            n = len(df)
            if n == 0:
                continue
            bounds = np.linspace(0, n, min(per, n) + 1).astype(int)
            for i in range(len(bounds) - 1):
                tasks.append((label, int(bounds[i]), int(bounds[i + 1])))
        parts = {'S1': [], 'S2': [], 'S3': []}
        with mp.get_context('fork').Pool(
                processes=min(int(n_workers), len(tasks))) as pool:
            for label, start, out in pool.imap_unordered(_normalize_chunk, tasks):
                parts[label].append((start, out))
        return tuple(
            pd.concat([df for _, df in sorted(parts[l], key=lambda t: t[0])],
                      ignore_index=True) if parts[l] else pd.DataFrame()
            for l in ('S1', 'S2', 'S3'))

    if n_workers and n_workers > 1:   # Windows/spawn: pass chunks explicitly
        tasks = []
        for label, df in (('S1', s1), ('S2', s2), ('S3', s3)):
            n = len(df)
            if n == 0:
                continue
            bounds = np.linspace(0, n, min(per, n) + 1).astype(int)
            for i in range(len(bounds) - 1):
                tasks.append((label, df.iloc[int(bounds[i]):int(bounds[i + 1])]))
        parts = {'S1': [], 'S2': [], 'S3': []}
        with ProcessPoolExecutor(
                max_workers=min(int(n_workers), len(tasks))) as ex:
            for label, out in ex.map(_normalize_chunk_spawn, tasks):
                parts[label].append(out)
        return tuple(
            pd.concat(parts[l], ignore_index=True) if parts[l] else pd.DataFrame()
            for l in ('S1', 'S2', 'S3'))

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
