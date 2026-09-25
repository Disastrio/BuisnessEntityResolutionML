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
