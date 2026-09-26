# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Oreo
**Team Members:** Rishank Semalti (Team Leader), Kartikey Tyagi, Stavya Pathak, Vansh Dhama
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
We solve business entity resolution as a supervised pairwise-matching problem:
multi-pass blocking generates a bounded candidate set, 28 lexical/token/numeric
similarity features are computed per candidate pair, and a LightGBM classifier is
trained on ground-truth pairs (with hard negatives) to decide matches. Because the
metric is macro-averaged F0.5 (precision-weighted), the decision threshold and a
singleton confidence barrier are tuned on an entity-level holdout, and all evidence
comes only from the provided TSV files (no external APIs or data).

---

## 2. Methodology

### 2.1 Problem Analysis
- Three sources share **no common identifier**; S1 is the clean reference and S2/S3
  are noisy. A single S1 entity may match 0, 1, or many S2/S3 records.
- **Name noise:** abbreviations (`Pvt`/`Private`, `Corp`/`Corporation`), `&` vs `and`,
  word reordering, typos, transliteration variants.
- **Address noise:** `Rd`/`Road`, `St`/`Street`, missing PIN/state/city/house number,
  landmark references, component reordering.
- **Blank fields:** some noisy records have empty names, so address-only blocking matters.
- **Country is open-set:** train has `US`/`India`; test adds `France` (accented text).
  Country is never hard-coded; Unicode NFKD de-accenting handles it.
- **Singletons matter:** ~5.6% of S1 entities have no match; a false positive on a
  singleton scores 0, while a correct empty scores 1.

### 2.2 Solution Strategy
**Approach Type:** Blocking + supervised pairwise classifier (gradient-boosted trees)
**Core Innovation:** precision-first pipeline — conservative thresholding, a singleton
confidence barrier, hard-negative upweighting, and a hit-count candidate ranking that
keeps true matches inside a bounded candidate cap as the target set grows.

Pipeline: normalize → multi-pass blocking (union, capped) → 28 pairwise features →
LightGBM → F0.5 threshold/barrier optimization → per-S1 output assembly → validation.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used (union, high-precision first):**
  1. `country + exact normalized name`
  2. `country + 6-char name prefix`
  3. `country + postal/PIN code`
  4. rare name tokens (document-frequency filtered inverted index)
  5. address numeric-token combinations (≥2 shared)
  6. Soundex phonetic prefix of the leading token
  7. single meaningful numeric tokens (≥3 digits)
  8. rare address tokens (connects blank-name records by address)
  9. name character trigrams (implemented, optional)
- **Ranking & cap:** candidates are scored by how many independent blocks retrieve
  them (true matches usually co-occur across blocks); multi-block candidates are kept
  first, then single-block candidates by round-robin across blocks, capped at
  `max_candidates` per S1. Each block is fetched high-precision-first and bounded by
  a per-block fetch cap, so no single huge block starves the others.
- **How true matches are preserved:** union of complementary blocks (exact/prefix/
  postal/rare/phonetic/numeric/address), normalization variants kept separately, and
  hit-count ranking instead of arbitrary truncation.
- **Measured candidate recall:** ~0.93 (20k-S1 sample), ~0.89 (120k-S1 sample);
  the candidate set is the recall ceiling for the final matcher.

---

## 4. Matching Model

**Features used (28):**
- **Name (12):** exact match, Levenshtein ratio, Jaro-Winkler, token-sort, token-set,
  token Jaccard, char-3-gram Jaccard, containment, token overlap, length diff,
  prefix match, common-token ratio.
- **Address (10):** exact match, Levenshtein, token Jaccard, token overlap,
  char-3-gram Jaccard, length diff, numeric-token Jaccard/overlap, postal match,
  containment.
- **Cross-field (6):** same country, source indicator, both exact, high-name+high-address,
  high-name + numeric-address mismatch (chain-branch guard), high-name + postal match.

**Model type:** LightGBM binary classifier (MIT-licensed, < 1M parameters).

**Training:**
- Positives = ground-truth pairs; negatives = blocking candidates that are not matches.
- **Hard negatives** (same-name/different-address, same-address/different-name) are
  detected and upweighted ×3 via sample weights.
- **Validation:** split strictly by S1 entity (no pair leakage) — 20% held-out entities.

**Threshold & decision rules:**
- Threshold sweep over [0.30, 0.95] maximizing macro F0.5.
- Optional per-source thresholds (S2 vs S3) and a singleton confidence barrier.
- A conservative "high name similarity + no address/postal evidence" rule, adopted
  only when it improves held-out F0.5.

---

## 5. Results & Error Analysis

- **Best validation macro F0.5 (120k-S1 sample):** **0.905**
  - pair precision 0.959, pair recall 0.864, singleton accuracy 0.917
  - chosen threshold 0.95; singleton barrier 0.98; max candidates/S1 = 250
- Smaller-sample validations (higher blocking recall ⇒ higher ceiling):
  - 20k S1: macro F0.5 0.957 (candidate recall 0.933)
  - 3k S1: macro F0.5 0.982 (candidate recall 0.985)

- **Common false positives:** common business names matched without address
  confirmation; chain branches sharing a name but different street numbers
  (explicitly penalized by the numeric-mismatch feature and rule).
- **Common false negatives:** true matches missed at the blocking stage — pairs with
  no shared blocking key (heavily corrupted/blank names) or buried in very large
  buckets. Candidate recall (~0.89 at 120k) is the dominant recall limiter; the
  classifier's precision is already ~0.96.

---

## 6. Conclusion
A precision-first blocking + LightGBM pipeline, tuned for macro F0.5 with entity-level
validation and conservative decision rules, reproduces the full submission from the
provided TSV files alone. The main remaining lever is blocking recall at scale —
improving discriminative keys (e.g., name trigrams) rather than enlarging the model.

---

## Appendix

### A. Code Artefacts
Runnable code ships under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── src/
│   ├── config.py       # paths, seeds, thresholds, feature flags
│   ├── io.py           # streaming TSV loaders, ground-truth parser, aligned sampling
│   ├── normalize.py    # Unicode NFKD + name/address canonicalization
│   ├── blocking.py     # multi-pass inverted-index blocking + candidate ranking
│   ├── features.py     # 28 pairwise similarity features (RapidFuzz)
│   ├── train.py        # LightGBM classifier, hard negatives, Optuna tuner
│   ├── evaluate.py     # macro-F0.5, threshold/per-source/barrier sweeps
│   ├── predict.py      # chunked streaming inference + TSV writers
│   └── pipeline.py     # CLI entry point (smoke / train / predict)
├── requirements.txt
└── README.md           # reproduction commands
```

**Entry points to reproduce the outputs:**
```bash
python -m src.pipeline --mode train --sample 120000 --max-candidates 250
python -m src.pipeline --mode predict --max-candidates 20 --chunk-size 25000
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results
- Verification harness: `scripts/test_model_improvements.py` (49 synthetic checks:
  blocking, features, dual models, thresholds, chunked inference, parallel paths).
- Diagnostics: `scripts/diagnose_blocking.py` (explains missed true matches).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
