# 📋 PROJECT TRACKER — Amazon Business Entity Resolution
### *Last Updated: 2026-09-25 00:05 IST*
> **How to use this file:** This is the human brain of the project.
> Update it every time you run an experiment, make a decision, or notice something.
> Read this FIRST every session before touching code.

---

## 🔴 CURRENT STATUS

```
Phase:         PIPELINE RESTRUCTURED & ROADMAP READY
Best Val F0.5: — (ready for Baseline E1)
Best Model:    —
Next Action:   -> Implement src/normalize.py and run rapid normalization benchmarks
Roadmap:       See ROADMAP.md for phase-by-phase execution plan
```

---

## 🎯 WHAT THIS PROBLEM ACTUALLY IS

**Business Entity Resolution** — NOT a standard classification/regression problem.

Given 3 sources of business records (name + address + country), find which
records across sources refer to the *same real-world business*.

### The Core Task
```
Source 1 (clean reference)  ←→  Source 2 (noisy)
Source 1 (clean reference)  ←→  Source 3 (noisy)
```
For every S1 entity → find all matching S2/S3 records (could be 0, 1, or many).

### Why This Is Hard
- No shared ID between sources
- Business names have typos, abbreviations, transliterations
- Addresses have missing components, landmarks, reorderings
- Common names (e.g. "City Bank") can match many — high false positive risk
- Test set introduces FRANCE (not seen in training) — can't hardcode countries

---

## 📊 METRIC — F0.5 (Precision-Weighted!)

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

**This is NOT F1. Precision matters MORE than recall.**

| Implication | Action |
|---|---|
| False merges are very expensive | Be conservative with threshold |
| Missing a match is less costly than a wrong match | Higher threshold is safer |
| Empty prediction = 1.0 score for singletons | Never blindly predict matches |
| A wrong match on a singleton = 0.0 score | Singleton handling is critical |

**Computed:** per S1 entity, then **macro-averaged**.

---

## 📂 DATA

### Files (TSV — tab-separated, NOT comma!)
```python
pd.read_csv("dataset/train/train_source1.tsv", sep="\t")  # ALWAYS sep="\t"
```

| File | Location | Rows | Size | Status |
|------|----------|------|------|--------|
| `train_source1.tsv` | `dataset/train/` | **2,206,821** | 200 MB | OK |
| `train_source2.tsv` | `dataset/train/` | **5,034,616** | 467 MB | OK |
| `train_source3.tsv` | `dataset/train/` | **5,285,603** | 480 MB | OK |
| `train_ground_truth.tsv` | `dataset/train/` | **2,206,821** | 121 MB | OK |
| `test_source1.tsv` | `dataset/test/` | **1,732,544** | 167 MB | OK |
| `test_source2.tsv` | `dataset/test/` | **4,887,273** | 486 MB | OK |
| `test_source3.tsv` | `dataset/test/` | **5,082,316** | 483 MB | OK |

> **Total: ~24.2 million records, ~2.4 GB raw text**

### What We Know (from EDA)
| Field | Value |
|-------|-------|
| Task type | Pairwise matching (match / no-match per pair) |
| Metric | **F0.5 (precision-heavy), macro-averaged per S1 entity** |
| Train S1 entities | **2,206,821** |
| Test S1 entities | **1,732,544** |
| Singletons (no match) | **123,247 (5.6%)** |
| With matches | **2,083,574 (94.4%)** |
| Avg matches per S1 | **3.46** |
| Max matches per S1 | **11** |
| Total matched IDs | **7,638,365** (S2: 3.69M, S3: 3.94M) |
| Countries (train) | **US, India** |
| Countries (test) | **US, India, France** (NEW!) |

### Schema (same for all 3 sources)
| Column | Description | Notes |
|--------|-------------|-------|
| `entity_id` | Unique ID | Prefix: `S1-`, `S2-`, `S3-` |
| `business_name` | Business name | Noisy: abbreviations, typos, transliteration |
| `business_address` | Address | Noisy: missing parts, landmarks, reordering |
| `country` | Country string | Train: US + India. Test: + France. **Treat as open string.** |

### Ground Truth Format
| Column | Description |
|--------|-------------|
| `source1_entity_id` | S1 record |
| `matched_entity_ids` | Comma-separated S2/S3 IDs (empty if no match) |

### Known Noise Patterns
**Name noise:**
- `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`
- `&` vs `and`
- Word order changes, typos, transliteration differences
- DBA/trade names, punctuation, capitalization

**Address noise:**
- `Rd` vs `Road`, `St` vs `Street`
- Missing PIN, state, city, house number
- Landmark references (`Near SBI ATM`)
- Component reordering

---

## 🏗️ FULL PIPELINE ARCHITECTURE

```
Raw TSV Records
      │
      ▼
① NORMALIZATION
   ├── Name: lowercase → remove punctuation → expand abbreviations
   │         → normalize & vs and → collapse whitespace → keep raw too
   └── Address: lowercase → alphanumeric → extract PIN/postal
                → extract city/state → extract numeric tokens
      │
      ▼
② BLOCKING / CANDIDATE GENERATION  (controls RECALL ceiling)
   Union of all blocks:
   ├── Block A: exact normalized name
   ├── Block B: name prefix / char n-grams
   ├── Block C: distinctive address tokens
   ├── Block D: postal/PIN code (when available)
   ├── Block E: country (open string)
   ├── Block F: rare name tokens
   └── Block G: combined (country + name prefix, country + postal, etc.)
      │
      ▼
③ PAIRWISE FEATURE GENERATION  (for every candidate pair)
   Name features:
   ├── exact normalized match, Jaccard, token-set, token-sort
   ├── Levenshtein ratio, char n-gram similarity
   ├── TF-IDF cosine, containment (A in B, B in A)
   └── token overlap count, rare-token overlap, length diff, shared numeric tokens
   Address features:
   ├── exact match, token Jaccard, token overlap, char n-gram
   ├── TF-IDF cosine, edit similarity
   └── shared numeric tokens, shared PIN, shared city, shared state, containment
   Cross-field features:
   ├── same_country
   ├── name_exact AND address_exact
   ├── name_high AND address_high
   └── name_high AND shared_postcode
      │
      ▼
④ PAIR CLASSIFIER  (supervised, trained from ground truth)
   → LightGBM (primary)
   → Option: separate S1↔S2 and S1↔S3 models if sources differ significantly
   → Hard negatives in training (same name different address, etc.)
      │
      ▼
⑤ THRESHOLD SELECTION  (optimize on held-out val using macro F0.5)
   → Do NOT use 0.5 as default
   → Test range of thresholds, pick best val F0.5
   → Higher threshold = safer (F0.5 is precision-heavy)
      │
      ▼
⑥ PER-S1 MATCHING
   For each S1 entity:
   → collect all candidates scoring above threshold
   → deduplicate IDs
   → if nothing passes: output empty (correct for singletons)
      │
      ▼
⑦ OUTPUT FILES
   ├── output/matching_results.tsv
   └── output/candidate_pairs.tsv
      │
      ▼
⑧ VALIDATE
   python3 utils/validate_submission.py --matching ... --candidate ... --test-dir ...
   → Must print PASS
```

---

## ⚙️ ALGORITHM CHOICES

### Primary Model
**LightGBM** — pairwise binary classifier (match=1 / no-match=0)

Why:
- Tabular similarity features → trees dominate
- Fast, handles mixed feature types
- MIT licensed ✅ (under 8B param limit)
- We already have it installed (v4.7.0)

### Model Options Considered
| Model | Verdict | Reason |
|---|---|---|
| **LightGBM** | ✅ Primary | Fast, accurate on tabular, MIT licensed |
| **XGBoost** | 🔄 Backup/blend | Diversity in ensemble |
| **CatBoost** | 🔄 Try for high-card cats | Good at string categories natively |
| **Logistic Regression** | ✅ Baseline | Simple, interpretable, use as floor |
| **HistGradientBoosting** | ✅ Fallback | scikit-learn native, no extra deps |
| Neural nets / transformers | ❌ Not needed | Overkill for similarity features, license risk |
| External APIs / LLMs | ❌ PROHIBITED | Disqualification risk |

### One Model vs Two Models
- **Start:** One model with `source_pair` (S2/S3) as a feature
- **Experiment 6:** Compare vs separate S1↔S2 and S1↔S3 models
- **Pick by:** validation F0.5

---

## 🧪 EXPERIMENT PLAN (in order)

| # | Experiment | Status | Val F0.5 | Notes |
|---|------------|--------|----------|-------|
| E1 | **Exact name match baseline** — normalized name exact OR name+address exact | ⬜ TODO | — | Floor |
| E2 | **Add fuzzy similarity** — Levenshtein, Jaccard, token overlap, TF-IDF cosine, char n-gram | ⬜ TODO | — | First real model |
| E3 | **Blocking improvements** — add all 7 block types, measure candidate recall | ⬜ TODO | — | Recall ceiling |
| E4 | **Hard-negative training** — same name diff address, same address diff business | ⬜ TODO | — | Precision boost |
| E5 | **Threshold tuning** — sweep 0.3–0.9, pick best macro F0.5 on val | ⬜ TODO | — | Do not use 0.5 |
| E6 | **Source-specific models** — separate S1↔S2 vs S1↔S3 models | ⬜ TODO | — | Compare vs single |
| E7 | **Conservative decision rules** — require name_high AND address_high | ⬜ TODO | — | Precision guard |
| E8 | **Singleton-aware tuning** — measure singleton accuracy separately | ⬜ TODO | — | F0.5 sensitive |

> ✅ = Done and kept | ❌ = Tried and reverted | ⬜ = Not started | 🔄 = In progress

---

## 🔬 METRICS TO TRACK (Not just F0.5)

| Metric | Formula | Why |
|---|---|---|
| **Val macro F0.5** | See above | Primary — optimize this |
| Pair precision | TP / (TP+FP) | Are predicted matches correct? |
| Pair recall | TP / (TP+FN) | Are true matches being found? |
| **Candidate recall** | True matches in candidates / all true matches | Upper bound on final recall |
| Reduction ratio | Candidates / Cartesian product | Blocking efficiency |
| **Singleton accuracy** | % of no-match S1s predicted as empty | Critical for F0.5 |

---

## ⚠️ PRECISION SAFEGUARDS

Because F0.5 punishes false merges hard:

| Signal Strength | Evidence Required | Action |
|---|---|---|
| **Very strong** | Exact normalized name + exact/near-exact address | Accept |
| **Strong** | Very high name sim + strong address sim + same country | Accept |
| **Ambiguous** | High name sim + weak/no address | Require threshold > 0.7 |
| **Dangerous** | Common business name only | Reject unless address confirms |

---

## 🚨 KNOWN FAILURE MODES TO AVOID

| Failure | Symptom | Fix |
|---|---|---|
| Blocking too strict | High precision, low recall | Add more blocks, relax rules |
| Candidate set too broad | Huge candidate count, many FPs | Stronger threshold + hard negatives |
| Name-only matching | Common names merge wrong businesses | Always combine with address evidence |
| Address-only matching | Businesses sharing offices/malls | Always combine with name evidence |
| Over-normalization | Distinct businesses merged | Keep multiple raw+normalized representations |
| Ignoring multiple matches | Missed true positives | Never force 1:1 matching |
| Forgetting singletons | FPs on no-match entities | Always output empty if no pair passes threshold |
| Country hard-coding | France breaks at test time | Treat country as open string |
| Output format errors | TSV with commas, missing rows, wrong cols | Run validator before every submit |

---

## 📁 PROJECT STRUCTURE

```
amazon-ML/
├── PROJECT.md              ← YOU ARE HERE — update every session
├── CONTEXT.md              ← AI assistant reference
├── problemstatment.md      ← Original problem statement (keep for reference)
├── requirements.txt
├── .gitignore
│
├── dataset/                ← Drop data here
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── config.py           ← Paths, seeds, constants, threshold
│   ├── io.py               ← TSV loaders (always sep="\t")
│   ├── normalize.py        ← Name + address normalization
│   ├── blocking.py         ← 7 blocking strategies + union
│   ├── features.py         ← Pairwise similarity features (name + address + cross)
│   ├── train.py            ← LightGBM pair classifier + Optuna tuning
│   ├── predict.py          ← Inference + threshold + per-S1 matching
│   ├── evaluate.py         ← Macro F0.5, candidate recall, singleton accuracy
│   └── pipeline.py         ← End-to-end runner
│
├── output/
│   ├── matching_results.tsv    ← Final submission file
│   └── candidate_pairs.tsv     ← Required alongside matching_results
│
├── utils/
│   └── validate_submission.py  ← Provided by challenge — run before submitting
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_blocking_analysis.ipynb
│   └── 03_feature_analysis.ipynb
│
├── models/                 ← Saved LightGBM .txt artifacts
├── reports/                ← Plots, threshold curves, SHAP outputs
└── submissions/            ← ZIP archives for submission
```

---

## 📤 OUTPUT FORMAT (Critical — get this wrong = FAIL)

### `output/matching_results.tsv`
```
source1_entity_id\tmatched_entity_ids
S1-00001\tS2-00047,S2-00193,S3-00812
S1-00002\tS3-00004
S1-00003\t                            ← empty = no match (singleton)
```
Rules:
- Exactly one row per test S1 entity (no more, no less)
- Comma-separated IDs (no spaces), tab-separated columns
- Empty `matched_entity_ids` for singletons
- IDs must be valid S2/S3 test IDs only
- No duplicate IDs per row

### `output/candidate_pairs.tsv`
- Same format but with CANDIDATE IDs (superset of matched IDs)
- Every matched ID must appear in its candidate list

### Validate Before EVERY Submission
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
# Must print: PASS
```

---

## 📦 FINAL ZIP STRUCTURE

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 🔒 HARD RULES (Violations = Disqualification)

| Rule | Status |
|------|--------|
| ❌ No external entity lookup APIs | ✅ Following |
| ❌ No commercial ER services | ✅ Following |
| ❌ No external business databases | ✅ Following |
| ❌ No geocoding APIs | ✅ Following |
| ❌ No internet-based entity lookup | ✅ Following |
| ✅ Model must be MIT/Apache 2.0 licensed | LightGBM ✅ |
| ✅ Model must be ≤ 8B parameters | LightGBM ✅ (< 1M) |
| ✅ All evidence from provided data only | ✅ Following |

---

## ✅ PRE-SUBMISSION CHECKLIST

- [ ] `validate_submission.py` returns `PASS`
- [ ] Every test S1 entity has exactly one row
- [ ] `matched_entity_ids` contains only valid S2/S3 test IDs
- [ ] No duplicate IDs in any row
- [ ] Empty lists for singletons formatted correctly
- [ ] Every matched ID appears in `candidate_pairs.tsv`
- [ ] France is handled (no hard-coded country filtering)
- [ ] No external data used
- [ ] Threshold selected from validation (not default 0.5)
- [ ] Candidate recall measured (upper bound on final recall)
- [ ] Hard negatives used in training
- [ ] Singleton accuracy evaluated separately
- [ ] README has exact reproduction commands
- [ ] requirements.txt pins versions
- [ ] Methodology doc included
- [ ] ZIP has correct structure

---

## 🔥 KEY DECISIONS MADE

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-25 | LightGBM as primary pair classifier | Tabular similarity features, MIT licensed, fast |
| 2026-09-25 | F0.5 as primary metric (not F1) | Problem requirement — precision-weighted |
| 2026-09-25 | Country treated as open string | Test has France, not seen in training |
| 2026-09-25 | Conservative threshold strategy | F0.5 penalizes false merges heavily |
| 2026-09-25 | Hard negatives in training | Prevent model from accepting weak name-only matches |
| 2026-09-25 | Union of 7 blocking strategies | Maximize candidate recall (recall ceiling) |
| 2026-09-25 | Keep multiple name/address representations | Over-normalization loses signal |
| *(add as you go)* | | |

---

## 📝 NOTES & OBSERVATIONS

```
[2026-09-25 00:05] — Problem statement fully read. This is Entity Resolution, NOT tabular ML.
                     Complete pipeline rebuild needed vs original scaffold.
                     Key insight: F0.5 = be conservative. When in doubt, predict empty.
                     Singletons are a major scoring opportunity — handle explicitly.
                     Country: France in test, never seen in train. Open string only.
                     Data format: TSV always. Business fields contain commas — never read as CSV.
```

---

## ❓ OPEN QUESTIONS

- [ ] How many S1/S2/S3 records in train and test? (need to see data)
- [ ] What % of S1 entities are singletons (no match)?
- [ ] How different are S2 vs S3 noise patterns? (determines if 1 or 2 models)
- [ ] Are addresses in France formatted differently enough to need special handling?
- [ ] What is the expected candidate reduction ratio needed for scalability?

---

*This file is the human-readable brain of the project.*
*Update every session. Read before touching code.*
