"""
config.py — Master Configuration for Business Entity Resolution.
Consolidates paths, seeds, thresholds, and execution hyperparameters.
"""
from pathlib import Path
import random
import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

# ── Project Directories ───────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATASET_DIR  = ROOT / "dataset"
TRAIN_DIR    = DATASET_DIR / "train"
TEST_DIR     = DATASET_DIR / "test"
MODELS_DIR   = ROOT / "models"
REPORTS_DIR  = ROOT / "reports"
OUTPUT_DIR   = ROOT / "output"
SUBS_DIR     = ROOT / "submissions"

for _d in [MODELS_DIR, REPORTS_DIR, OUTPUT_DIR, SUBS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset File Paths (Tab-Separated .tsv) ───────────────────────────────────
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1  = TEST_DIR / "test_source1.tsv"
TEST_S2  = TEST_DIR / "test_source2.tsv"
TEST_S3  = TEST_DIR / "test_source3.tsv"

# ── Output Submission File Paths ──────────────────────────────────────────────
MATCHING_OUT   = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT  = OUTPUT_DIR / "candidate_pairs.tsv"

# ── Schema Constants ──────────────────────────────────────────────────────────
ID_COL         = "entity_id"
NAME_COL       = "business_name"
ADDR_COL       = "business_address"
COUNTRY_COL    = "country"

GT_S1_COL      = "source1_entity_id"
GT_MATCH_COL   = "matched_entity_ids"
CAND_MATCH_COL = "candidate_entity_ids"

# ── Evaluation & Modeling ─────────────────────────────────────────────────────
METRIC         = "macro_f0_5"
BETA           = 0.5                     # Precision weight in F-beta
DEFAULT_THRESH = 0.70                    # Conservative starting threshold for F0.5
MAX_CANDIDATES = 100                     # Max candidate pairs to retain per S1 record

if __name__ == "__main__":
    print(f"Config initialized successfully. Root: {ROOT}")
    print(f"Train dir exists: {TRAIN_DIR.exists()}")
    print(f"Test dir exists:  {TEST_DIR.exists()}")
