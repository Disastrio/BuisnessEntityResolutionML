# CONTEXT.md — Amazon ML Challenge: Business Entity Resolution Master Context

> **Purpose:** Single source of truth for all AI assistants and team members.
> Load this file at the start of every session. It consolidates the problem contract,
> data dictionary, ER architecture, feature log, experiment log, validation strategy,
> and operational rules into one authoritative document.

---

## 0. AI Assistant Rules of Engagement

When assisting with this project, always adhere strictly to these rules:

1. **Strictly No External APIs / Lookups:** Never propose external geocoding, Google Maps, entity lookup services, commercial ER tools, or web scraping. This causes immediate disqualification.
2. **Tab-Separated Data (`sep="\t"`):** All datasets and outputs are TSV. Business addresses and comma-separated ID lists contain commas; reading as CSV will corrupt data.
3. **Metric Focus: Macro-Averaged $F_{0.5}$:**
   - $\beta = 0.5 \implies$ Precision is weighted $2\times$ more than Recall (false merges are penalized $4\times$ harder than false dismissals).
   - Singletons (no matches) correctly predicted as empty score **1.0**. Incorrectly predicting a match on a singleton yields **0.0**.
   - Thresholding must be conservative ($t^* \approx 0.65 - 0.78$ rather than default 0.50).
4. **Generalization to France:** The test set introduces `France`, which never appears in the training data (`US` and `India`). Country must be treated as an open string; never hardcode regexes or filters restricted to US/India.
5. **Entity-Level Validation (No Leakage):** Splits must partition on `Source 1` entity IDs (`GroupKFold` or entity-level holdout). Never randomly shuffle candidate pairs across splits.
6. **Deterministic & Reproducible:** Fixed seeds (`SEED=42`), explicit numpy/torch/python random state.
7. **Model Constraints:** Open-source MIT/Apache 2.0 license, model parameter count $\le 8\text{B}$ (LightGBM $\ll 10\text{M}$ parameters fully satisfies this).
8. **Platform Hygiene (Windows/PowerShell):** Use `;` (not `&&`) for command sequencing. Run scripts with `$env:PYTHONIOENCODING='utf-8'` and use ASCII-safe status tags (`[SMOKE]`, `[TRAIN]`, `[EVAL]`).

---

## 1. Problem Contract

| Parameter | Specification |
|---|---|
| **Challenge** | Amazon ML Challenge 2026: Business Entity Resolution |
| **Task Type** | Pairwise Entity Resolution & Linkage across heterogeneous sources |
| **Reference Source** | `Source 1` (clean deduplicated entity references) |
| **Noisy Sources** | `Source 2` and `Source 3` (noisy, partial, unlinked records) |
| **Target Output** | For every test S1 entity, output comma-separated matching S2/S3 entity IDs |
| **Primary Metric** | Macro-Averaged $F_{0.5}$ per S1 entity |
| **Train Scale** | S1: 2,206,821 \| S2: 5,034,616 \| S3: 5,285,603 \| Total: ~12.5M records |
| **Test Scale** | S1: 1,732,544 \| S2: 4,887,273 \| S3: 5,082,316 \| Total: ~11.7M records |
| **Singleton Frequency** | ~5.6% of S1 entities have 0 matches (must predict empty) |
| **Matched Entities** | 94.4% have matches; average 3.46 matches per entity (max 11) |
| **Countries** | Train: `US`, `India` \| Test: `US`, `India`, `France` |
| **External Data** | **STRICTLY PROHIBITED** (Automatic disqualification) |
| **License Requirement** | MIT or Apache 2.0 (LightGBM satisfies) |
| **Model Size Limit** | $\le 8\text{B}$ parameters |

---

## 2. Data Dictionary & Schemas

### 2.1 Entity Sources (`train_source*.tsv`, `test_source*.tsv`)
All source files share an identical 4-column tab-delimited schema:

| Column | Type | Description | Observed Noise & Variations |
|---|---|---|---|
| `entity_id` | String | Unique record ID | Prefix denotes source: `S1-`, `S2-`, `S3-` |
| `business_name` | String | Commercial/Trade name | Typos, abbreviations (`Pvt`/`Private`, `Corp`/`Corporation`, `&`/`and`), transliterations, legal suffixes |
| `business_address` | String | Physical address | Missing postal codes, missing states, landmark-based descriptions ("Near SBI ATM"), reordered tokens |
| `country` | String | Country identifier | Categorical label. Train: `US`, `India`. Test: `US`, `India`, `France` |

### 2.2 Ground Truth (`train_ground_truth.tsv`)

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | String | S1 entity ID (e.g. `S1-00001`) |
| `matched_entity_ids` | String | Comma-separated list of true S2/S3 IDs (e.g. `S2-00047,S3-00812`), or empty for singletons |

### 2.3 Submission Output Formats (`output/`)

1. **`matching_results.tsv`** *(Only file scored on leaderboard)*:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Exactly one row per test S1 entity.
   - Singletons have empty `matched_entity_ids`.
   - No duplicate IDs, no self-references, test IDs only.
2. **`candidate_pairs.tsv`** *(Blocking audit file)*:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Must contain every ID that appears in `matching_results.tsv` (strict superset).
   - Validated by `utils/validate_submission.py`.

---

## 3. Four-Stage Pipeline Architecture

```
Raw TSV Data (S1, S2, S3)
        │
        ▼
[Stage 1: Normalization] (`src/normalize.py`)
  ├── Unicode NFKD normalization (accents stripped: 'é' -> 'e')
  ├── Case folding & ampersand conversion ('&' -> 'and')
  ├── Legal suffix expansion ('pvt ltd' -> 'private limited', 'corp' -> 'corporation')
  ├── Address standardizations ('rd' -> 'road', 'st' -> 'street')
  └── Component extraction: Postal codes (5/6-digit), numbers, city tokens
        │
        ▼
[Stage 2: 7-Pass Blocking] (`src/blocking.py`)
  ├── Pass 1: Exact country + normalized name
  ├── Pass 2: Exact country + 6-char name prefix
  ├── Pass 3: Exact country + postal/PIN code
  ├── Pass 4: Rare name tokens (IDF-filtered inverted index)
  ├── Pass 5: Shared numeric token sets (>= 2) + country
  ├── Pass 5b: Single meaningful numeric token (>= 3 digits) + country
  ├── Pass 6: Soundex / phonetic prefix (transliterations)
  ├── Pass 7: Relaxed name match (single-token entities)
  ├── Pass 8: Rare address tokens (connects blank-name records by address)
  └── Round-robin union across blocks, capped at 100 candidates per S1
      (each block is fetched high-precision-first and capped at BLOCK_FETCH_CAP,
       so a huge block cannot starve the others of candidate slots)
        │
        ▼
[Stage 3: Pairwise Feature Generation] (`src/features.py`)
  └── 28 dense similarity features computed via RapidFuzz:
        ├── Name similarities (12 features): exact, Levenshtein, Jaro-Winkler,
        │   Token Sort, Token Set, 3-gram Jaccard, containment, lengths
        ├── Address similarities (10 features): exact, token Jaccard, numeric
        │   token overlap & Jaccard, postal match, length ratios
        └── Cross-field signals (6 features): same_country, joint high name+address,
            numeric mismatch indicator, source indicators (is_s2, is_s3)
        │
        ▼
[Stage 4: Supervised Classifier & Thresholding] (`src/train.py`, `src/evaluate.py`, `src/predict.py`)
  ├── Model: LightGBM binary classifier (match vs no-match)
  ├── E4 Hard Negatives: Same-name/diff-address etc. detected via `identify_hard_negatives`
  │   and upweighted (x3) with sample weights — not row-oversampled
  ├── E5 Threshold Sweep: Global macro F0.5 search over t in [0.30, 0.95] (optimal t* ≈ 0.70),
  │   or per-source S2/S3 thresholds via coordinate ascent (`threshold_sweep_by_source`)
  ├── E6 Dual Models (optional): separate S1↔S2 and S1↔S3 classifiers
  ├── E7 Conservative Rules: reject near-identical-name pairs with zero address/postal
  │   evidence — adopted only when they improve held-out F0.5
  ├── E8 Singleton Barrier: entity emits matches only if its best probability ≥ barrier
  └── Output Assembly: Per-S1 candidate filtering, singleton preservation, TSV export
```

---

## 4. Feature Log (28 Pairwise Similarity Features)

All computed in `src/features.py` for each candidate pair $(S_1, S_{2/3})$:

### 4.1 Name Similarity Features (12)
1. `name_exact_match`: Boolean exact match on normalized name.
2. `name_levenshtein_ratio`: RapidFuzz normalized Levenshtein ratio $[0, 1]$.
3. `name_jaro_winkler`: Prefix-weighted string distance $[0, 1]$.
4. `name_token_sort_ratio`: Token-sorted Levenshtein (handles word reordering).
5. `name_token_set_ratio`: Set-based token matching (handles extra noise tokens).
6. `name_token_jaccard`: Jaccard similarity of whitespace-token sets.
7. `name_char_3gram_jaccard`: Jaccard similarity of character 3-grams.
8. `name_containment`: $\min(|A|, |B|) / |A \cap B|$ token containment.
9. `name_len_diff_ratio`: Relative length difference $|len_A - len_B| / \max(len_A, len_B)$.
10. `name_token_count_diff`: Absolute difference in word counts.
11. `name_shared_token_count`: Count of intersecting name words.
12. `name_raw_exact_match`: Exact match on raw un-normalized name strings.

### 4.2 Address Similarity Features (10)
13. `addr_exact_match`: Boolean exact match on normalized address.
14. `addr_levenshtein_ratio`: RapidFuzz Levenshtein ratio on full address.
15. `addr_token_jaccard`: Word-level Jaccard similarity on address tokens.
16. `addr_char_3gram_jaccard`: Character 3-gram similarity on address.
17. `addr_shared_numeric_count`: Count of overlapping numeric tokens (street/building/PIN).
18. `addr_numeric_jaccard`: Jaccard similarity of numeric token sets.
19. `postal_exact_match`: 1.0 if postal codes match, 0.0 if both exist but differ, 0.5 if missing.
20. `postal_both_present`: Binary indicator that both records have extracted postal codes.
21. `addr_containment`: Token containment ratio for addresses.
22. `addr_len_diff_ratio`: Relative length difference of address strings.

### 4.3 Cross-Field & Interaction Features (6)
23. `same_country`: 1.0 if country matches, 0.0 otherwise (open string comparison).
24. `joint_name_addr_high`: 1.0 if `name_levenshtein > 0.85` AND `addr_token_jaccard > 0.60`.
25. `name_high_addr_num_mismatch`: 1.0 if `name_levenshtein > 0.90` BUT numeric address tokens conflict (catches chain store branches).
26. `is_source_2`: 1.0 if candidate is from Source 2.
27. `is_source_3`: 1.0 if candidate is from Source 3.
28. `name_addr_geom_mean`: Geometric mean of name Levenshtein and address token Jaccard.

---

## 5. Experiment Log & Benchmarking

| ID | Description | Blocking Recall | Val Precision | Val Recall | Val Macro $F_{0.5}$ | Status | Notes |
|---|---|---|---|---|---|---|---|
| **E1** | Exact Name & Address Baseline | 42.1% | 0.965 | 0.421 | 0.748 | ✅ Completed | Deterministic rule floor |
| **E2** | 7-Pass Blocking + LightGBM (Default $t=0.50$) | 92.4% | 0.841 | 0.886 | 0.849 | ✅ Completed | Solid recall, lower precision |
| **E3** | 7-Pass Blocking + LightGBM + Tuned Threshold ($t^*=0.72$) | 92.4% | 0.932 | 0.824 | **0.908** | ✅ Completed | Peak $F_{0.5}$, protects singletons |
| **E4** | Hard-Negative Mining (Chain store branch differentiation) | — | — | — | — | ✅ Implemented | `identify_hard_negatives` + x3 sample weights in `src/train.py` (full-data numbers pending) |
| **E5** | Source-Specific Dual-Model (Separate S1↔S2 and S1↔S3) | — | — | — | — | ✅ Implemented | `train_dual_models` / `predict_dual_probabilities`; enable with `--dual-model` |
| **E6** | Post-processing / Discrepancy Filter | — | — | — | — | ✅ Implemented | `conservative_reject_mask` (kept only if it improves val F0.5); `--no-rules` to disable |
| **E7** | Per-Source Threshold + Singleton Barrier | 98.5% | 0.989 | 0.959 | 0.982 | ✅ Implemented | `threshold_sweep_by_source` + `sweep_singleton_barrier`; sample-scale measurement |
| **E8** | Phonetic Blocking Index 6 (Soundex) | — | — | — | — | ✅ Implemented | 7th block now actually built |
| **E9** | Recall Fixes: single-numeric + address-token indexes + round-robin cap | 98.5% | 0.990 | 0.963 | **0.982** | ✅ Implemented | Candidate recall 0.879 → 0.985; see `scripts/diagnose_blocking.py` |

---

## 6. Validation Strategy & Metric Computation

### 6.1 Entity-Level Holdout
- **Never split by candidate pair.** If pairs of the same S1 entity are divided across train and validation, the model leaks entity identity.
- Validation split is drawn strictly at the **S1 entity level** (e.g. 20% held-out S1 IDs with all their true S2/S3 matches and candidate negatives).

### 6.2 Macro-Averaged $F_{0.5}$ Calculation
For each S1 entity $i$:
- Let $T_i$ be true matched IDs, and $P_i$ be predicted matched IDs.
- If $|T_i| > 0$:
  $$\text{Precision}_i = \frac{|P_i \cap T_i|}{|P_i|} \quad (\text{0 if } |P_i| = 0)$$
  $$\text{Recall}_i = \frac{|P_i \cap T_i|}{|T_i|}$$
  $$F_{0.5, i} = \frac{1.25 \cdot \text{Precision}_i \cdot \text{Recall}_i}{0.25 \cdot \text{Precision}_i + \text{Recall}_i} \quad (\text{0 if } P_i \cap T_i = \emptyset)$$
- If $|T_i| = 0$ (Singleton):
  $$F_{0.5, i} = \begin{cases} 1.0 & \text{if } |P_i| = 0 \\ 0.0 & \text{if } |P_i| > 0 \end{cases}$$
- Final Score:
  $$\text{Macro } F_{0.5} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} F_{0.5, i}$$

---

## 7. Submission Checklist & Quality Gates

Prior to creating any submission zip:
- [ ] Run `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` $\implies$ Must output `PASS`.
- [ ] Row count of `matching_results.tsv` matches `test_source1.tsv` exactly (1,732,544 rows).
- [ ] Row count of `candidate_pairs.tsv` matches `test_source1.tsv` exactly (1,732,544 rows).
- [ ] Every matched ID in `matching_results.tsv` exists in `candidate_pairs.tsv`.
- [ ] No `NaN`, nulls, trailing spaces, or CSV commas in columns.
- [ ] Singletons correctly represented by an empty string after the tab delimiter (`S1-xxxxx\t`).
- [ ] All IDs verified to exist in test S2 or S3 sets.
- [ ] Code is self-contained under `code/business_entity_resolution/src/` with `requirements.txt`.
- [ ] `Documentation_template.md` filled out and included at archive root.
