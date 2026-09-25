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
BLOCK_FETCH_CAP = 500                    # Max IDs pulled per block per S1 (bounds work)

# ── Model Improvement Hyperparameters (Experiments E4-E8) ─────────────────────
# Thresholds used to recognise boundary cases.
NAME_HIGH_THRESHOLD = 0.85               # "high name similarity" cutoff
NAME_VERY_HIGH_THRESHOLD = 0.90          # "very high name similarity" cutoff
ADDR_HIGH_THRESHOLD = 0.60               # "strong address evidence" cutoff
ADDR_WEAK_THRESHOLD = 0.30               # "no usable address evidence" cutoff

# E4 - Hard-negative mining: pairs that look plausible by one field but are not
# true matches (same name/different address, same address/different name, chain
# branches). These are upweighted during training to sharpen the precision guard.
USE_HARD_NEGATIVES      = True
HARD_NEGATIVE_WEIGHT    = 3.0            # sample weight applied to hard negatives

# E5 - Source-aware thresholds: S2 and S3 have different noise profiles, so the
# F0.5-optimal decision threshold can differ per source. Learned on validation.
USE_SOURCE_THRESHOLDS   = False

# E6 - Dual source-specific models: train separate S1<->S2 and S1<->S3 models.
USE_DUAL_MODELS         = False
DEFAULT_MODEL_NAME      = "lgbm_pair_classifier"       # single unified model
DUAL_MODEL_NAME         = "lgbm_pair_classifier_dual"  # per-source bundle

# E7 - Conservative decision rules: raise the bar for dangerous pairs (near
# identical name but zero address/postal confirmation) instead of a hard reject.
USE_CONSERVATIVE_RULES  = True
RULE_THRESHOLD_BOOST    = 0.10           # extra confidence required for risky pairs

# E8 - Singleton-aware confidence barrier: an S1 entity emits matches only if its
# single best candidate probability reaches `barrier`. 0.0 disables the guard.
SINGLETON_BARRIER       = 0.0

# Blocking: phonetic (Soundex) index (Index 6) - transliteration / typo recovery.
USE_PHONETIC_BLOCK      = True
SOUNDEX_LENGTH          = 4

if __name__ == "__main__":
    print(f"Config initialized successfully. Root: {ROOT}")
    print(f"Train dir exists: {TRAIN_DIR.exists()}")
    print(f"Test dir exists:  {TEST_DIR.exists()}")
