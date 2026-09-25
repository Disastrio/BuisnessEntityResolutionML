"""
io.py — Robust TSV Data Ingestion and Serialization for Business Entity Resolution.

Handles:
- Tab-separated values (sep="\\t") strictly to avoid comma collisions in addresses.
- Ground truth parsing into fast set lookups.
- Chunked processing for memory scalability across 24M records.
"""
from typing import Dict, Set, List, Optional, Generator
import pandas as pd
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


def load_aligned_sample(
    n_s1: int = 10000,
    seed: int = 42,
) -> tuple:
    """
    Load an aligned sample of training data where ground truth matches
    are guaranteed to exist in the loaded S2/S3 data.

    Strategy:
    1. Load full ground truth (small file — only IDs)
    2. Sample n_s1 S1 entity IDs
    3. Collect all referenced S2/S3 IDs from ground truth
    4. Load S1 rows for sampled IDs
    5. Load S2/S3 rows that include all referenced IDs + random extras

    Returns:
        (s1_df, s2_df, s3_df, ground_truth_dict)
    """
    import numpy as np

    print(f"  Loading aligned sample (n_s1={n_s1:,})...")

    # Step 1: Load full ground truth (just IDs, ~120MB but fast)
    gt_df = load_tsv(TRAIN_GT)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row[GT_S1_COL]).strip()
        matched_str = str(row[GT_MATCH_COL]).strip()
        if matched_str and matched_str.lower() != "nan":
            matches = {x.strip() for x in matched_str.split(",") if x.strip()}
        else:
            matches = set()
        gt_map[s1_id] = matches

    # Step 2: Sample S1 entity IDs
    all_s1_ids = sorted(gt_map.keys())
    rng = np.random.RandomState(seed)
    sampled_s1 = set(rng.choice(all_s1_ids, size=min(n_s1, len(all_s1_ids)), replace=False))

    # Step 3: Collect all S2/S3 IDs referenced by sampled S1 entities
    needed_s2_ids: Set[str] = set()
    needed_s3_ids: Set[str] = set()
    sampled_gt: Dict[str, Set[str]] = {}
    for s1_id in sampled_s1:
        matches = gt_map.get(s1_id, set())
        sampled_gt[s1_id] = matches
        for m_id in matches:
            if m_id.startswith("S2"):
                needed_s2_ids.add(m_id)
            elif m_id.startswith("S3"):
                needed_s3_ids.add(m_id)

    print(f"  Sampled {len(sampled_s1):,} S1 entities")
    print(f"  Need {len(needed_s2_ids):,} S2 IDs + {len(needed_s3_ids):,} S3 IDs from ground truth")

    # Step 4: Load S1 rows for sampled IDs
    s1_full = load_tsv(TRAIN_S1)
    s1_df = s1_full[s1_full[ID_COL].isin(sampled_s1)].reset_index(drop=True)
    del s1_full

    # Step 5: Load S2 — include all needed IDs + random extras for blocking diversity
    s2_full = load_tsv(TRAIN_S2)
    s2_needed = s2_full[s2_full[ID_COL].isin(needed_s2_ids)]
    # Add random extras (up to 3x the needed count for hard negatives)
    s2_extra_pool = s2_full[~s2_full[ID_COL].isin(needed_s2_ids)]
    n_extra_s2 = min(len(s2_extra_pool), max(len(needed_s2_ids) * 3, n_s1))
    if n_extra_s2 > 0:
        s2_extras = s2_extra_pool.sample(n=n_extra_s2, random_state=seed)
        s2_df = pd.concat([s2_needed, s2_extras], ignore_index=True)
    else:
        s2_df = s2_needed.reset_index(drop=True)
    del s2_full, s2_extra_pool

    # Step 6: Load S3 — same strategy
    s3_full = load_tsv(TRAIN_S3)
    s3_needed = s3_full[s3_full[ID_COL].isin(needed_s3_ids)]
    s3_extra_pool = s3_full[~s3_full[ID_COL].isin(needed_s3_ids)]
    n_extra_s3 = min(len(s3_extra_pool), max(len(needed_s3_ids) * 3, n_s1))
    if n_extra_s3 > 0:
        s3_extras = s3_extra_pool.sample(n=n_extra_s3, random_state=seed)
        s3_df = pd.concat([s3_needed, s3_extras], ignore_index=True)
    else:
        s3_df = s3_needed.reset_index(drop=True)
    del s3_full, s3_extra_pool

    print(f"  Loaded S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}")
    n_with_matches = sum(1 for v in sampled_gt.values() if v)
    print(f"  S1 with matches: {n_with_matches:,}, singletons: {len(sampled_gt) - n_with_matches:,}")

    return s1_df, s2_df, s3_df, sampled_gt

