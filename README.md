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
├── README.md                   # Repository overview, setup, and CLI guide
├── requirements.txt            # Pinned Python package dependencies
├── Documentation_template.md   # Official methodology template (filled for submission)
├── .gitignore
│
├── docs/                       # All long-form documentation
│   ├── PROJECT.md              # Project tracker, experiment ledger, current status
│   ├── ROADMAP.md              # Milestone execution plan & architectural routes
│   ├── CONTEXT.md              # Rules of engagement, data dictionary, feature log
│   ├── PROJECT_EXPLAINER.md    # Plain-English guide to every design choice
│   ├── problem_statement.md    # Official problem statement (expanded guide)
│   └── organizer_readme.md     # Official organizer README
│
├── dataset/                    # Challenge TSV datasets (sep="\t", gitignored)
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
│   ├── __init__.py             # Package init & module map
│   ├── config.py               # Paths, seeds, constants, thresholds, feature flags
│   ├── io.py                   # Streaming TSV loaders & aligned dev sampling
│   ├── normalize.py            # Multilingual Unicode NFKD & regex cleaners
│   ├── blocking.py             # Multi-pass candidate blocking engine
│   ├── features.py             # 28 pairwise similarity feature generator
│   ├── train.py                # LightGBM classifier, hard negatives, Optuna tuner
│   ├── evaluate.py             # Macro-F0.5, threshold/barrier optimizer
│   ├── predict.py              # Test inference & TSV output formatter
│   └── pipeline.py             # Unified CLI runner (smoke, train, predict)
│
├── scripts/
│   ├── test_model_improvements.py  # 34 synthetic checks (no dataset needed)
│   └── diagnose_blocking.py        # Why true matches are missed by the blocker
│
├── utils/
│   └── validate_submission.py  # Official verification script (must PASS)
│
├── output/                     # Generated submission artifacts (gitignored)
│   ├── matching_results.tsv    # Target predictions (leaderboard scored)
│   └── candidate_pairs.tsv     # Candidate pairs (blocking audit)
├── models/                     # Serialized LightGBM models (gitignored)
└── reports/                    # Generated logs, curves, EDA output (gitignored)
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

### Mode C: Test Set Inference & Output Generation (streaming)
Scores S1 in **chunks**, writing `output/matching_results.tsv` and
`output/candidate_pairs.tsv` incrementally, so peak memory scales with
`--chunk-size` rather than with the full ~1.7M S1 / ~11.7M total test records:

```bash
python -m src.pipeline --mode predict
```

| Flag | Default | Effect |
|---|---|---|
| `--chunk-size N` | `20000` | S1 entities scored (and written) per chunk |
| `--limit-s1 N` | all | Score only the first N S1 entities (staged / smoke runs) |

```bash
# Conservative 16 GB box: smaller chunks
python -m src.pipeline --mode predict --chunk-size 10000

# Quick staged check that streams every phase without a full run
python -m src.pipeline --mode predict --limit-s1 5000
```

> **Full test-set inference** needs ~10 GB free disk. The blocking indices and the
> in-memory S2/S3 target frame are the fixed cost; if that exceeds RAM, either
> raise `--chunk-size`-independent memory by adding RAM or shard the target sources.
> After a full run, always validate:
> ```bash
> python utils/validate_submission.py --matching output/matching_results.tsv \
>     --candidate output/candidate_pairs.tsv --test-dir dataset/test
> ```

### Mode D: Advanced Model Options (Experiments E4–E8)
The model improvements are opt-in so the validated E3 baseline stays reproducible.

| Flag | Experiment | Effect |
|---|---|---|
| `--no-hard-negatives` | E4 | Disable x3 upweighting of same-name / same-address negatives |
| `--source-thresholds` | E5 | Tune separate decision thresholds for S2 and S3 |
| `--dual-model` | E6 | Train separate `S1↔S2` and `S1↔S3` LightGBM models |
| `--no-rules` | E7 | Disable the conservative high-name/no-address rejection rule |
| `--barrier <p>` | E8 | Singleton confidence-barrier floor (tuned upward on validation) |

```bash
# Full model: hard negatives + per-source thresholds + rules + dual models
python -m src.pipeline --mode train --sample 100000 --dual-model --source-thresholds
```

```bash
# Force a conservative singleton barrier of 0.80
python -m src.pipeline --mode train --barrier 0.80
```

> **Note:** the conservative rule is only kept when it improves held-out F0.5, and the
> singleton barrier sweep always includes `0.0`, so neither guard can reduce the
> validation score.

---

## 🧪 Model Verification (No Dataset Needed)

All model upgrades are covered by a fast, self-contained harness that runs on tiny
synthetic data (safe under low memory):

```bash
python scripts/test_model_improvements.py
```

It checks the phonetic blocking index, hard-negative weighting, per-source threshold
tuning, the singleton barrier, conservative rules, dual-model save/load, the
round-robin candidate cap, the streamed `TargetLookup`, and chunked inference output
formatting, plus the official submission validator end-to-end (45 checks).

To understand *why* the blocker misses true matches on a real sample, run:

```bash
python scripts/diagnose_blocking.py 3000
```

### Streaming, low-memory training

`src/io.py::load_aligned_sample` reads the three sources in chunks and never
materialises the full multi-GB files, so training samples run in a few hundred MB
(measured ~175 MB RSS) rather than requiring tens of GB.

### Measured results (streaming 3,000-S1 aligned sample, 600 held-out S1 entities)

| Configuration | Candidate recall | Pair precision | Pair recall | **Macro F0.5** |
|---|---|---|---|---|
| Before blocking-recall fixes | 0.879 | 0.991 | 0.873 | 0.9465 |
| Single model + hard negatives (default) | 0.985 | 0.990 | 0.963 | **0.9818** |
| Dual models + per-source thresholds | 0.986 | 0.989 | 0.959 | **0.9820** |

These are sample-scale numbers, not the full 2.2M-entity test score.

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

- [Project Human Explainer](docs/PROJECT_EXPLAINER.md) — Intuitive, detailed plain-English guide covering every architectural design choice.
- [Project Tracker](docs/PROJECT.md) — Live experiment ledger, parameter log, and decision tracker.
- [Roadmap & Route Analysis](docs/ROADMAP.md) — Architectural routes and milestone criteria.
- [AI & Team Context](docs/CONTEXT.md) — Engineering rules of engagement, data dictionary, and feature log.
- [Problem Statement](docs/problem_statement.md) — The official challenge prompt.
- [Organizer README](docs/organizer_readme.md) — The resource pack README from the organizers.
