# 🗺️ Comprehensive Project Roadmap & Execution Plan: Business Entity Resolution

> **Challenge:** Amazon ML Challenge — Business Entity Resolution (ER)  
> **Target Metric:** Macro-averaged $F_{0.5}$ (Precision-heavy; false merges penalized $4\times$ more than false dismissals)  
> **Dataset Scale:** 3 Sources (~24.2M records, ~2.4GB raw text), Train: US + India, Test: US + India + France  
> **Key Guardrail:** Submission package must include self-contained reproducible code under `code/business_entity_resolution/`, `output/matching_results.tsv`, and `output/candidate_pairs.tsv` satisfying `utils/validate_submission.py`.

---

## 1. Executive Summary & Problem Realignment

Standard tabular ML pipelines (e.g. basic classification with StratifiedKFold on rows) **do not apply** to this challenge.
This is a **Large-Scale Multi-Source Business Entity Resolution** task:
- **Reference entity source:** `Source 1` (deduplicated clean reference)
- **Noisy sources:** `Source 2` and `Source 3`
- **Output:** Every `S1` entity mapped to zero, one, or multiple matching `S2`/`S3` records.
- **Metric ($F_{0.5}$):**
  $$\beta = 0.5 \implies F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
  - A false positive (incorrect merge) drops the precision heavily and crushes the $F_{0.5}$ score.
  - A singleton (S1 entity with 0 matches) correctly predicted as empty scores **1.0**. Incorrectly predicting a match on a singleton yields **0.0**.
  - **Golden Principle:** *High recall during candidate blocking, high precision during pair classification, conservative thresholding.*

---

## 2. Optimized Project Architecture

The existing repository contained generic tabular ML templates (`src/data.py`, `src/validate.py`) which are incompatible with TSV formatting and ER pairing logic. We restructure and optimize the repository as follows:

```
amazon-ML/
├── ROADMAP.md                  # Comprehensive route and execution plan (THIS FILE)
├── PROJECT.md                  # Project tracker, experiment ledger, EDA insights
├── CONTEXT.md                  # System prompt / rules of engagement for assistants
├── problemstatment.md          # Problem specification from organizers
├── Documentation_template.md   # Official documentation report template
├── requirements.txt            # Pinned environment requirements
├── .gitignore                  # Git hygiene (ignoring large raw data and caches)
│
├── dataset/                    # Authoritative challenge datasets (TSV, sep="\t")
│   ├── train/
│   │   ├── train_source1.tsv       # S1 records (2.2M)
│   │   ├── train_source2.tsv       # S2 records (5.0M)
│   │   ├── train_source3.tsv       # S3 records (5.3M)
│   │   └── train_ground_truth.tsv  # S1 -> comma-separated S2/S3 IDs
│   └── test/
│       ├── test_source1.tsv        # S1 records (1.7M)
│       ├── test_source2.tsv        # S2 records (4.9M)
│       └── test_source3.tsv        # S3 records (5.1M) [contains France!]
│
├── src/                        # Core Entity Resolution Pipeline Modules
│   ├── __init__.py
│   ├── config.py               # Paths, constants, seeds, tuned thresholds
│   ├── io.py                   # Chunked TSV ingestion & Ground Truth parsers
│   ├── normalize.py            # Multilingual/multimodal string & address cleaning
│   ├── blocking.py             # Multi-pass candidate indexing & candidate union
│   ├── features.py             # Pairwise lexical, phonetic, token & cross-field sim
│   ├── train.py                # Pairwise LightGBM / CatBoost model & Optuna tuning
│   ├── evaluate.py             # Macro-F0.5 calculator, threshold sweeper, singleton acc
│   ├── predict.py              # Test inference generator & per-S1 thresholding
│   ├── ensemble.py             # Rank / probability blending across model folds
│   └── pipeline.py             # End-to-end deterministic execution CLI
│
├── scripts/                    # Standalone utility & exploration scripts
│   ├── eda_explore.py          # Fast exploratory profiling script
│   └── run_mini_benchmark.py   # 10k sample rapid testing sandbox
│
├── output/                     # Generated submission artifacts
│   ├── matching_results.tsv    # Target predictions (source1_entity_id \t matched_entity_ids)
│   └── candidate_pairs.tsv     # Candidate pairs (source1_entity_id \t candidate_entity_ids)
│
├── utils/                      # Official challenge verification tools
│   └── validate_submission.py  # Output syntax and integrity validator
│
├── models/                     # Serialized LightGBM models and vectorizers
├── reports/                    # Generated charts, EDA report, threshold curves
└── submissions/                # Ready-to-upload ZIP archives
```

---

## 3. Four Strategic Execution Routes

We analyze 4 possible architectural routes to solve the project, ranked by feasibility, compute complexity, and competitive score potential:

```
Route A: Fast Heuristic & String Distance (Baseline / Floor)
Route B: Blocking + Feature Engineering + Gradient Boosted Trees (Recommended Core Route)
Route C: Source-Disentangled Dual GBDT Pipeline (High Precision Extension)
Route D: Hybrid Dense-Embedding / Bi-Encoder Re-Ranking (Advanced / High Compute)
```

### Route Comparison Matrix

| Route | Complexity | Memory / Compute | Expected Val $F_{0.5}$ | Time to Implement | Risk Profile |
|---|---|---|---|---|---|
| **Route A: Deterministic Rule-Based** | Very Low | Minimal (16GB RAM) | 0.65 – 0.72 | 2–3 hours | High precision, low recall ceiling |
| **Route B: Multi-Block + LightGBM (Selected)** | Moderate | Medium (32GB RAM / chunking) | 0.82 – 0.88 | 1–2 days | **Optimal trade-off of speed, recall & precision** |
| **Route C: Source-Specific Dual-Model (S1-S2 & S1-S3)** | Moderate+ | Medium | 0.84 – 0.89 | +4 hours on Route B | Handles divergence between S2 and S3 schemas |
| **Route D: Dense Embeddings (MiniLM/DeBERTa)** | Very High | GPU intensive, slow on 24M records | 0.85 – 0.90 | 3–4 days | Scale bottleneck on 10M test candidates; OOM risk |

---

## 4. Deep Dive: The Recommended Route (Route B + Route C Hybrid)

### Phase 1: Ingestion & Normalization (`src/io.py`, `src/normalize.py`)
1. **TSV Streaming / Chunking:**
   - Always read files with `sep="\t"`, quoting disabled or controlled (`quoting=csv.QUOTE_NONE` / standard TSV parser) to prevent misaligned commas in address fields.
2. **Text Standardization:**
   - Unicode NFKD normalization to ascii/clean unicode (handles accents in French addresses: `é` $\to$ `e`).
   - Case folding, strip punctuation except critical hyphens/alphanumerics.
   - Legal suffix canonicalization: `pvt ltd` $\to$ `private limited`, `inc` $\to$ `incorporated`, `co` $\to$ `company`, `corp` $\to$ `corporation`, `llc` $\to$ `limited liability company`.
   - Ampersand normalization: `&` $\to$ `and`.
3. **Address Component Parsing:**
   - Postal code extraction: Regex for 6-digit (India), 5-digit/ZIP (US, France).
   - Numeric token set: Extract house numbers, floor numbers, ward numbers, plot codes.
   - City / state token extraction from trailing address tokens.

### Phase 2: High-Recall Multi-Index Candidate Generation (`src/blocking.py`)
Because Cartesian comparison ($2.2\text{M} \times 10.3\text{M} \approx 2.2 \times 10^{13}$ pairs) is impossible, candidate generation sets the **recall ceiling**.
We use a **7-tier inverted index union**:
- **Index 1 (Exact Name):** `country + clean_name`
- **Index 2 (Name Prefix / 4-gram):** `country + clean_name[:6]`
- **Index 3 (Postal / PIN Code):** `country + postal_code` (high-precision geographical anchor)
- **Index 4 (Name Token Inverted Index):** Rare tokens (TF-IDF > threshold) in business name
- **Index 5 (Shared Number + City Token):** For addresses sharing distinct building numbers and postal areas
- **Index 6 (Soundex / Double Metaphone Prefix):** Phonetic variations and transliteration typos (India/US)
- **Index 7 (Relaxed Name Match for Single-Token Entities):**
- **Safety Bounds:** Cap candidate pairs per S1 at top-100 to prevent combinatorial explosion.
- **Evaluation Gate:** Must achieve $\ge 92\%$ Candidate Recall on validation holdout.

### Phase 3: Pairwise Feature Engineering (`src/features.py`)
For every candidate pair $(S_1, S_{2/3})$, construct ~28 dense similarity features:
1. **Lexical Similarities (Name):**
   - Exact match boolean
   - Levenshtein / Damerau-Levenshtein edit ratio (via `rapidfuzz`)
   - Jaro-Winkler similarity (strong for prefix-focused business names)
   - Token Sort Ratio & Token Set Ratio
   - Character 3-gram Jaccard similarity
   - Containment ratio ($\min(|A|,|B|) / |A \cap B|$)
2. **Address Similarities:**
   - Exact address match boolean
   - Address token Jaccard similarity
   - Shared numeric tokens count & Jaccard (crucial for street numbers/PINs)
   - Postal code exact match / mismatch indicator
   - Address length difference ratio
3. **Cross-Field & Semantic Signals:**
   - Country match boolean (strictly handles `US`, `India`, `France`)
   - Joint agreement: `(name_sim > 0.85) AND (addr_sim > 0.70)`
   - Discrepancy indicator: `(name_sim > 0.90) AND (addr_numeric_mismatch)` (prevents merging different branches of chains)
   - Source indicator: `is_s2` vs `is_s3` (allows learning source-specific noise profiles)

### Phase 4: Supervised Classifier & Hard-Negative Mining (`src/train.py`)
1. **Dataset Construction:**
   - Positives: True pairs from `train_ground_truth.tsv`.
   - Negatives: Candidates generated by blocking that are not true matches.
   - **Hard Negatives:** Specifically sample candidates that share identical business names but different addresses (e.g. branch offices, competitor stores), and identical addresses with different business names.
   - Ratio: 1 positive to 4–6 negatives.
2. **Model Choice:**
   - Primary: **LightGBM** (fastest training, native NaN handling, MIT licensed, parameter count $< 1\text{M} \ll 8\text{B}$).
   - Secondary: **CatBoost** (superior categorical handling for city/country interaction).
3. **Loss Function:** Binary logloss with tuned class weight or focal loss.

### Phase 5: Threshold Optimization & Singleton Preservation (`src/evaluate.py`)
1. **Metric Realization:** Macro $F_{0.5}$ computed strictly per S1 entity:
   - For S1 with true matches: compute entity precision and recall, then $F_{0.5}$.
   - For S1 singletons (true matches = 0):
     - If predicted empty: score = **1.0**.
     - If any false positive match predicted: score = **0.0**.
2. **Threshold Sweep:**
   - Grid search decision threshold $t \in [0.30, 0.95]$ with step $0.01$.
   - Because $F_{0.5}$ weights precision at $80\%$ and recall at $20\%$, the optimal threshold is typically higher ($t^* \approx 0.68 - 0.78$) than standard 0.5.
   - Secondary safeguard rule: If a candidate has high name similarity but zero matching numeric address tokens, bump the required threshold by $+0.15$.

### Phase 6: Full Test Inference & Output Generation (`src/predict.py`)
1. Run Test S1 against Test S2 & S3 via Blocking.
2. Predict probabilities using trained LightGBM models.
3. Apply optimal threshold $t^*$.
4. Format:
   - `output/matching_results.tsv` (only IDs passing $t^*$)
   - `output/candidate_pairs.tsv` (all evaluated candidates; superset of matches)
5. Run verification:
   `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
   **Must output: `PASS`**.

---

## 5. Implementation Roadmap & Milestones

| Milestone | Deliverables | Verification Criteria | Target Timeline |
|---|---|---|---|
| **M1: Foundation & IO** | `src/config.py`, `src/io.py`, `src/normalize.py` | Load 10k samples, test normalization speed (>10k rec/sec), clean unicode | Day 1 (Morning) |
| **M2: Blocking Engine** | `src/blocking.py`, inverted index caches | Validate candidate recall $\ge 90\%$ on 50k train subset | Day 1 (Afternoon) |
| **M3: Feature Pipeline** | `src/features.py` with rapidfuzz / token metrics | Compute 28 features on 200k pairs in $<60$ seconds | Day 1 (Evening) |
| **M4: Model Training** | `src/train.py`, baseline LightGBM model | Validate logloss & pair AUC $\ge 0.96$ | Day 2 (Morning) |
| **M5: F0.5 Optimization** | `src/evaluate.py`, threshold sweep | Peak Macro $F_{0.5}$ logged; singleton accuracy $> 95\%$ | Day 2 (Afternoon) |
| **M6: Test Inference & QA** | `src/predict.py`, output files generated | `validate_submission.py` outputs `PASS` | Day 2 (Evening) |
| **M7: Final Packaging** | Clean code under `code/`, `Documentation_template.md`, ZIP | All constraints, license checks, and ZIP integrity verified | Day 3 |

---

## 6. Risk Mitigation Matrix

| Potential Pitfall | Impact | Mitigation Strategy |
|---|---|---|
| **Memory Exhaustion (OOM) on 24M records** | Fatal Crash | Process blocking and feature extraction in streaming chunks of 250k S1 records; use sparse inverted indices. |
| **France Out-of-Distribution Drift** | Severe Drop in Test | Do not hardcode US/India regexes. Use generalized unicode normalization and open-string country matching. |
| **Chain Store False Positive Merges** | Massive $F_{0.5}$ degradation | Explicitly penalize address number mismatches even if name similarity is 1.0. |
| **Format Invalidity (Commas instead of TSV)** | Disqualification | Enforce explicit `sep="\t"` and assert column counts prior to saving. Run `validate_submission.py`. |
| **Licensing / Model Constraints** | Disqualification | Strict adherence to MIT / Apache 2.0 (LightGBM / RapidFuzz / Scikit-learn). Parameter count $< 10\text{M} \ll 8\text{B}$. |
