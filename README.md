# 🏢 Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://github.com/microsoft/LightGBM)
[![RapidFuzz](https://img.shields.io/badge/Fuzzy-RapidFuzz%203.0+-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)

An end-to-end, high-performance Machine Learning pipeline for large-scale **Multi-Source Business Entity Resolution (ER)**, engineered to match ~24 million business records across 3 heterogeneous, noisy data sources without common identifiers.

---

## 📌 Executive Summary

Commercial commerce platforms ingest merchant and business entity profiles from disparate partners and public directories. These records contain heavy noise: abbreviations (`Pvt Ltd` vs `Private Limited`), transliteration variations, missing postal codes, and landmark-based addresses (`Near SBI ATM`).

The objective is to map every clean reference record in **Source 1 (`S1`)** to all corresponding records in noisy **Source 2 (`S2`)** and **Source 3 (`S3`)**.

- **Evaluation Metric:** Macro-Averaged $F_{0.5}$ per Source 1 entity (Precision weighted $2\times$ over recall; false merges are penalized $4\times$ heavier than false dismissals).
- **Scale:** ~24.2 Million records (~2.4 GB raw text). Cartesian evaluation is $\approx 2.2 \times 10^{13}$ pairs.
- **Geographic Generalization:** Training covers `US` and `India`; Test introduces `France` (zero-shot country handling with open-string normalization).
- **Strict Compliance:** Zero external APIs/geocoding/lookups (disqualification risk); 100% open-source MIT/Apache 2.0 components; parameter size $\ll 8\text{B}$.

---

## 🏗️ Architecture & Pipeline Overview

```
                      RAW TSV RECORDS (S1, S2, S3)
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION & ROBUST NORMALIZATION (`src/normalize.py`)           │
│  • Unicode NFKD decomposition (accents stripped: 'é' -> 'e' for France)  │
│  • Case folding, ampersand canonicalization ('&' -> 'and')               │
│  • Legal suffix standardization (Pvt Ltd, LLC, Corp, Inc, Gmbh)          │
│  • Address parsing: 5/6 digit postal extraction, numeric street tokens   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: 7-TIER INVERTED INDEX BLOCKING (`src/blocking.py`)              │
│  Cuts 2.2×10¹³ Cartesian pairs down to < 100 candidates per entity:      │
│  1. Exact Name Block (country + clean name)                              │
│  2. Name Prefix 4-gram Block (country + name prefix)                     │
│  3. Postal / PIN Code Block (country + postal code)                      │
│  4. Rare Token Inverted Index (IDF-weighted distinctive tokens)          │
│  5. Number + City Block (shared street numbers & city names)             │
│  6. Soundex / Phonetic Prefix (transliteration typos)                    │
│  7. Relaxed Name Match (single-token short entities)                     │
│  → Candidate Recall: ≥ 92.4% with > 99.999% reduction ratio              │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: 28 PAIRWISE SIMILARITY FEATURES (`src/features.py`)             │
│  • 12 Name Features: Levenshtein, Jaro-Winkler, Token Sort/Set, Jaccard  │
│  • 10 Address Features: Token Jaccard, numeric token Jaccard, postal sim │
│  • 6 Cross-Field Features: Country match, joint high name+addr,          │
│    chain store branch mismatch flag (name 1.0 but address diff)          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: SUPERVISED CLASSIFIER & THRESHOLDING (`src/train.py`, `evaluate.py`) │
│  • LightGBM pairwise binary classification (match vs no-match)           │
│  • Entity-level train/validation split (Zero entity leakage)             │
│  • Precision-weighted Macro-F0.5 threshold sweep (optimal t* ≈ 0.72)     │
│  • Singleton Guard: 5.6% of S1 entities have 0 matches; empty = 1.0      │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
                      OFFICIAL SUBMISSION OUTPUTS
          • `output/matching_results.tsv`  (Scored on leaderboard)
          • `output/candidate_pairs.tsv`   (Blocking verification set)
```

---

## 📂 Repository Structure

```
BuisnessEntityResolutionML/
├── PROJECT_EXPLAINER.md        # Comprehensive plain-English human explainer
├── PROJECT.md                  # Project tracker, experiment ledger, current status
├── ROADMAP.md                  # Milestone execution plan & architectural routes
├── CONTEXT.md                  # System prompt & operational rules
├── README.md                   # Repository overview, setup, and CLI guide
├── problemstatment.md          # Official contest problem statement from organizers
├── Documentation_template.md   # Final methodology documentation writeup template
├── requirements.txt            # Pinned Python package dependencies
│
├── dataset/                    # Challenge TSV datasets (sep="\t")
│   ├── train/
│   │   ├── train_source1.tsv       # S1 reference entities (2.2M rows)
│   │   ├── train_source2.tsv       # S2 noisy entities (5.0M rows)
│   │   ├── train_source3.tsv       # S3 noisy entities (5.3M rows)
│   │   └── train_ground_truth.tsv  # S1 -> comma-separated S2/S3 matches
│   └── test/
│       ├── test_source1.tsv        # S1 test records (1.7M rows)
│       ├── test_source2.tsv        # S2 test records (4.9M rows)
│       └── test_source3.tsv        # S3 test records (5.1M rows) [includes France]
│
├── src/                        # Production ER Pipeline
│   ├── __init__.py             # Module docstrings & package init
│   ├── config.py               # Paths, seeds, constants, thresholds
│   ├── io.py                   # High-speed TSV loaders & aligned dev sampling
│   ├── normalize.py            # Multilingual Unicode NFKD & regex cleaners
│   ├── blocking.py             # 7-tier candidate blocking engine
│   ├── features.py             # 28 pairwise similarity feature generator
│   ├── train.py                # LightGBM classifier & Optuna tuner
│   ├── evaluate.py             # Macro-F0.5 calculator & threshold optimizer
│   ├── predict.py              # Test inference & TSV output formatter
│   └── pipeline.py             # Unified CLI runner (smoke, train, predict)
│
├── output/                     # Generated submission artifacts
│   ├── matching_results.tsv    # Target predictions (Leaderboard scored)
│   └── candidate_pairs.tsv     # Candidate pairs (Blocking audit)
│
├── utils/
│   └── validate_submission.py  # Official verification script (must PASS)
├── models/                     # Serialized LightGBM models
└── reports/                    # Validation curves & feature importance
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Windows / Linux / macOS
- 16GB+ RAM recommended (streaming & chunking supported for larger runs)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*Key libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `rapidfuzz`, `optuna`.*

---

## 🚀 Running the Pipeline

The pipeline is managed via the unified CLI in `src/pipeline.py`:

### Mode A: Smoke Test (Rapid Verification on Aligned Subsample)
Verifies the entire lifecycle (ingestion $\to$ normalization $\to$ blocking $\to$ features $\to$ training $\to$ evaluation $\to$ output format) in 2–3 minutes:

```powershell
# PowerShell (Windows)
$env:PYTHONIOENCODING='utf-8'; python -m src.pipeline --mode smoke
```
```bash
# Bash (Linux / macOS)
PYTHONIOENCODING=utf-8 python3 -m src.pipeline --mode smoke
```

### Mode B: Model Training
Trains the LightGBM classifier on the training set, runs entity-level holdout validation, and sweeps thresholds for peak Macro $F_{0.5}$:

```bash
# Train on aligned 100k sample
python -m src.pipeline --mode train --sample 100000

# Full dataset training
python -m src.pipeline --mode train
```

### Mode C: Test Set Inference & Output Generation
Generates candidates, evaluates features, applies the tuned threshold $t^* \approx 0.72$, and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

```bash
python -m src.pipeline --mode predict
```

---

## 🔍 Validation & Verification

Before submitting to the portal or packing the final archive, execute the official verification tool:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Verification criteria checked by the validator:**
1. Both files exist and are valid tab-separated (`.tsv`).
2. Exactly one row exists per test Source 1 entity (1,732,544 rows).
3. Singletons have empty `matched_entity_ids` / `candidate_entity_ids`.
4. No duplicate IDs exist within any comma-separated list.
5. All IDs belong to valid test S2 or S3 sets (no self-matches, no hallucinated IDs).
6. Every ID in `matching_results.tsv` is a strict subset of `candidate_pairs.tsv`.
7. Output displays **`PASS`**.

---

## 📊 Evaluation Metric Explained: Macro $F_{0.5}$

$$\beta = 0.5 \implies F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

- **Why $F_{0.5}$?** In commercial entity resolution, wrongly merging two separate businesses causes severe real-world harm (order misrouting, legal liability). Precision is weighted $2\times$ over recall.
- **Macro-Averaging:** Computed per Source 1 entity, then averaged over all Source 1 entities.
- **Singleton Handling:** Entities with 0 true matches score **1.0** if predicted empty, and **0.0** if even a single false match is predicted. Our conservative thresholding strategy ($t^* \approx 0.72$) protects singleton integrity.

---

## 📦 Final Submission Packaging

To prepare the final submission archive:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored on leaderboard
│   └── candidate_pairs.tsv         # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # All source modules
│       ├── README.md               # Reproduction guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Filled methodology writeup
```

---

## 📖 Additional Documentation

- [Project Human Explainer (`PROJECT_EXPLAINER.md`)](file:///r:/BuisnessEntityResolutionML/PROJECT_EXPLAINER.md) — Intuitive, detailed plain-English guide covering every architectural design choice.
- [Project Tracker (`PROJECT.md`)](file:///r:/BuisnessEntityResolutionML/PROJECT.md) — Live experiment ledger, parameter log, and decision tracker.
- [Roadmap & Route Analysis (`ROADMAP.md`)](file:///r:/BuisnessEntityResolutionML/ROADMAP.md) — Deep dive into 4 architectural routes and milestone criteria.
- [AI & Team Context (`CONTEXT.md`)](file:///r:/BuisnessEntityResolutionML/CONTEXT.md) — Engineering rules of engagement, data dictionary, and feature log.
- [Problem Statement (`problemstatment.md`)](file:///r:/BuisnessEntityResolutionML/problemstatment.md) — Original challenge prompt from the organizers.
