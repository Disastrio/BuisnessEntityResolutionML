"""
io.py — Robust TSV Data Ingestion and Serialization for Business Entity Resolution.

Handles:
- Tab-separated values (sep="\\t") strictly to avoid comma collisions in addresses.
- Ground truth parsing into fast set lookups.
- Chunked processing for memory scalability across 24M records.
"""
from typing import Dict, Set, List, Optional, Generator, Tuple
import pandas as pd
import numpy as np
import csv
from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL,
    GT_S1_COL, GT_MATCH_COL
)


def load_tsv(
    filepath,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
):
    """
    Safely load TSV file with tab separation and whitespace trimming.
    """
    dtypes = {
        ID_COL: "string",
        NAME_COL: "string",
        ADDR_COL: "string",
        COUNTRY_COL: "string",
        GT_S1_COL: "string",
        GT_MATCH_COL: "string",
    }
    present_dtypes = {k: v for k, v in dtypes.items() if usecols is None or k in usecols}

    if chunksize is not None:
        return pd.read_csv(
            filepath,
            sep="\t",
            dtype=present_dtypes,
            usecols=usecols,
            nrows=nrows,
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
            chunksize=chunksize,
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=present_dtypes,
        usecols=usecols,
        nrows=nrows,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    for col in df.columns:
        if df[col].dtype == "string" or df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def load_source(source_num: int, split: str = "train", nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load Source 1, 2, or 3 for either 'train' or 'test'.
    """
    paths = {
        ("train", 1): TRAIN_S1,
        ("train", 2): TRAIN_S2,
        ("train", 3): TRAIN_S3,
        ("test", 1): TEST_S1,
        ("test", 2): TEST_S2,
        ("test", 3): TEST_S3,
    }
    path = paths.get((split.lower(), source_num))
    if not path:
        raise ValueError(f"Invalid source_num {source_num} or split {split}")
    return load_tsv(path, nrows=nrows)


def load_ground_truth(nrows: Optional[int] = None) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into {s1_id: {matched_s2_or_s3_ids}}.
    Singletons will map to empty sets: {s1_id: set()}.
    """
    gt_df = load_tsv(TRAIN_GT, nrows=nrows)
    gt_map: Dict[str, Set[str]] = {}

    for _, row in gt_df.iterrows():
        s1_id = str(row[GT_S1_COL]).strip()
        matched_str = str(row[GT_MATCH_COL]).strip()
        if matched_str and matched_str.lower() != "nan":
            matches = {x.strip() for x in matched_str.split(",") if x.strip()}
        else:
            matches = set()
        gt_map[s1_id] = matches

    return gt_map


def save_tsv(df: pd.DataFrame, filepath, index: bool = False) -> None:
    """
    Save DataFrame strictly with tab separators and UTF-8 encoding.
    """
    df.to_csv(filepath, sep="\t", index=index, quoting=csv.QUOTE_NONE, escapechar="\\")


def _count_lines(path) -> int:
    """Fast binary line count, excluding the header row."""
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            n += block.count(b"\n")
    return max(0, n - 1)


def _parse_matches(matched_str: str) -> Set[str]:
    matched_str = str(matched_str).strip()
    if matched_str and matched_str.lower() != "nan":
        return {x.strip() for x in matched_str.split(",") if x.strip()}
    return set()


def _stream_keep(path, keep_ids: Set[str], chunk_size: int) -> pd.DataFrame:
    """Stream a source TSV, keeping only rows whose entity_id is in keep_ids."""
    parts = []
    for chunk in load_tsv(path, chunksize=chunk_size):
        mask = chunk[ID_COL].isin(keep_ids)
        if mask.any():
            parts.append(chunk[mask])
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=[ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL])


def _stream_keep_plus_background(
    path, keep_ids: Set[str], target_background: int, seed: int, chunk_size: int,
) -> pd.DataFrame:
    """
    Stream a source TSV, keeping (a) every row referenced by ground truth and
    (b) a bounded random background sample for negatives. The full file is never
    materialised, so peak memory stays proportional to the sample size.
    """
    rng = np.random.RandomState(seed)
    total = _count_lines(path)
    p_bg = min(1.0, target_background / total) if total > 0 else 0.0

    kept, bg = [], []
    for chunk in load_tsv(path, chunksize=chunk_size):
        keep_mask = chunk[ID_COL].isin(keep_ids)
        if keep_mask.any():
            kept.append(chunk[keep_mask])
        rest = chunk[~keep_mask]
        if len(rest) and p_bg > 0.0:
            mask = rng.rand(len(rest)) < p_bg
            if mask.any():
                bg.append(rest[mask])

    frames = []
    if kept:
        frames.append(pd.concat(kept, ignore_index=True))
    if bg:
        bg_df = pd.concat(bg, ignore_index=True)
        if len(bg_df) > target_background:
            bg_df = bg_df.sample(n=target_background, random_state=seed)
        frames.append(bg_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=[ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL])


def load_aligned_sample(
    n_s1: int = 10000,
    seed: int = 42,
    chunk_size: int = 200_000,
    background_multiplier: int = 3,
) -> tuple:
    """
    Load an aligned sample where ground-truth matches are guaranteed present in
    the loaded S2/S3 data — using only bounded, streaming memory.

    All three sources are read in chunks and the full multi-GB files are never
    materialised, so this runs on low-RAM machines where the previous full-file
    loader would OOM.

    Strategy:
    1. Uniformly sample ~n_s1 ground-truth rows (streamed, no full-file load).
    2. Collect the S2/S3 IDs referenced by those sampled S1 entities.
    3. Stream S1/S2/S3 in chunks, keeping referenced rows plus a bounded random
       background sample per noisy source for negatives.

    Returns:
        (s1_df, s2_df, s3_df, ground_truth_dict)
    """
    print(f"  Loading aligned sample (n_s1={n_s1:,}, streaming)...")

    rng = np.random.RandomState(seed)

    # ── Step 1: sample ~n_s1 ground-truth rows without loading the whole file ──
    gt_total = _count_lines(TRAIN_GT)
    p_row = min(1.0, n_s1 / gt_total) if gt_total > 0 else 1.0
    sampled: List[Tuple[str, str]] = []
    for chunk in load_tsv(TRAIN_GT, chunksize=chunk_size):
        mask = rng.rand(len(chunk)) < p_row
        if mask.any():
            sub = chunk[mask]
            sampled.extend(zip(sub[GT_S1_COL].astype(str), sub[GT_MATCH_COL].astype(str)))
    if len(sampled) > n_s1:
        idx = rng.choice(len(sampled), size=n_s1, replace=False)
        sampled = [sampled[i] for i in idx]

    # ── Step 2: collect referenced S2/S3 IDs ──────────────────────────────────
    sampled_gt: Dict[str, Set[str]] = {}
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()
    for s1_id, matched_str in sampled:
        matches = _parse_matches(matched_str)
        sampled_gt[s1_id] = matches
        for m_id in matches:
            if m_id.startswith("S2"):
                needed_s2.add(m_id)
            elif m_id.startswith("S3"):
                needed_s3.add(m_id)

    print(f"  Sampled {len(sampled_gt):,} S1 entities")
    print(f"  Need {len(needed_s2):,} S2 IDs + {len(needed_s3):,} S3 IDs from ground truth")

    # ── Step 3: stream each source, keeping referenced rows + background ──────
    s1_df = _stream_keep(TRAIN_S1, set(sampled_gt.keys()), chunk_size)

    n_bg_s2 = max(len(needed_s2) * background_multiplier, n_s1)
    n_bg_s3 = max(len(needed_s3) * background_multiplier, n_s1)
    print(f"  Streaming S2 (keep {len(needed_s2):,} + <= {n_bg_s2:,} background)...")
    s2_df = _stream_keep_plus_background(TRAIN_S2, needed_s2, n_bg_s2, seed, chunk_size)
    print(f"  Streaming S3 (keep {len(needed_s3):,} + <= {n_bg_s3:,} background)...")
    s3_df = _stream_keep_plus_background(TRAIN_S3, needed_s3, n_bg_s3, seed + 1, chunk_size)

    # ── Step 4: align ground truth to what was actually loaded ────────────────
    loaded_s2 = set(s2_df[ID_COL]) if len(s2_df) else set()
    loaded_s3 = set(s3_df[ID_COL]) if len(s3_df) else set()
    for s1_id, matches in sampled_gt.items():
        sampled_gt[s1_id] = {
            m for m in matches if (m in loaded_s2) or (m in loaded_s3)
        }

    print(f"  Loaded S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}")
    n_with_matches = sum(1 for v in sampled_gt.values() if v)
    print(f"  S1 with matches: {n_with_matches:,}, singletons: {len(sampled_gt) - n_with_matches:,}")

    return s1_df, s2_df, s3_df, sampled_gt

