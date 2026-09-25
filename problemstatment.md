# Business Entity Resolution Challenge --- AI-Assisted Solving Guide

## 1. Problem Statement

This is a **Business Entity Resolution (ER)** challenge.

Business identity data arrives from multiple independent sources. Each
source contains partial, noisy, and inconsistent information about
real-world businesses. The records do not share a common identifier.

The goal is to build an ML/data-science pipeline that determines which
records across the sources refer to the same real-world business.

### Sources

-   **Source 1**: deduplicated reference source.
-   **Source 2**: noisy business records.
-   **Source 3**: noisy business records.

For every Source 1 entity, find **all matching records from Source 2 and
Source 3**.

A Source 1 entity may match:

-   zero records,
-   one record,
-   multiple records.

------------------------------------------------------------------------

## 2. Important File Format

All challenge files are **tab-separated values (`.tsv`)**.

Always read them with an explicit tab separator:

``` python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Do **not** read them as ordinary comma-separated CSVs. Business
addresses and ID-list fields contain commas.

------------------------------------------------------------------------

## 3. Data Description

Each source file contains:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `entity_id`                         Unique record identifier. Prefix
                                      indicates source: `S1-`, `S2-`, or
                                      `S3-`.

  `business_name`                     Business name; may contain
                                      abbreviations, legal suffixes,
                                      typos, transliterations, etc.

  `business_address`                  Business address; may contain
                                      partial addresses, formatting
                                      differences, missing components,
                                      landmarks, etc.

  `country`                           Country label. Training contains US
                                      and India; test additionally
                                      contains France. Treat country as
                                      an open string category.
  -----------------------------------------------------------------------

There is **no separate source column**. A record's source is determined
by its file and its `entity_id` prefix.

### Country warning

Do **not** hard-code:

``` text
US / India
```

The test set contains **France** as an additional country.

Treat `country` as an open-set string feature.

------------------------------------------------------------------------

# 4. Training Files

``` text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

### `train_source1.tsv`

Source 1 training records. This is the deduplicated reference source.

### `train_source2.tsv`

Source 2 training records.

### `train_source3.tsv`

Source 3 training records.

### `train_ground_truth.tsv`

Ground-truth matching labels.

Columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `source1_entity_id`                 Entity ID of a Source 1 record

  `matched_entity_ids`                Comma-separated matching Source
                                      2/Source 3 entity IDs. Empty when
                                      there are no matches.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 5. Test Files

``` text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

There is **no ground truth for the test set**.

Generate matches for **every Source 1 test entity**.

------------------------------------------------------------------------

# 6. Expected Noise Patterns

## Business-name variations

Expect:

-   abbreviations:
    -   `Corp` vs `Corporation`
    -   `Pvt` vs `Private`
    -   `Ltd` vs `Limited`
-   legal suffix inconsistencies
-   DBA/trade names
-   punctuation differences
-   `&` vs `and`
-   word-order changes
-   typos
-   transliteration differences
-   capitalization/spacing differences

## Address variations

Expect:

-   abbreviations:
    -   `Rd` vs `Road`
    -   `St` vs `Street`
-   transliteration variants
-   missing components
    -   PIN/postcode
    -   state
    -   city
    -   house number
-   landmark-based references
    -   e.g. `Near SBI ATM`
-   municipal numbering differences
-   component reordering
-   punctuation/spacing differences

------------------------------------------------------------------------

# 7. Evaluation Metric

Submissions are evaluated using **F0.5**.

It is precision-heavy: false merges are penalized more than missed
matches.

Formula:

``` text
F0.5 = (1.25 × Precision × Recall)
       / (0.25 × Precision + Recall)
```

The metric is computed **per Source 1 entity** and then macro-averaged
across Source 1 entities.

### Singletons matter

If a Source 1 entity has no true matches:

-   predicting an empty list scores `1.0`
-   predicting any match scores `0.0`

Therefore, false positives on singleton entities are particularly
harmful.

------------------------------------------------------------------------

# 8. Practical Modeling Objective

Because F0.5 weights precision more heavily than recall:

1.  Candidate generation should have high recall.
2.  Final matching should be conservative.
3.  Do not automatically accept weak similarities.
4.  Correctly identifying "no match" is valuable.
5.  Prefer multiple strong independent signals over one weak signal.

The pipeline should therefore be:

``` text
Raw records
    ↓
Normalization
    ↓
Candidate generation / blocking
    ↓
Pairwise feature generation
    ↓
Matching model / scoring
    ↓
Conservative thresholding
    ↓
Per-S1 matched ID lists
    ↓
TSV output
```

------------------------------------------------------------------------

# 9. Recommended Normalization

Build separate normalized representations rather than destroying the raw
fields.

## Name normalization

Useful operations:

1.  Unicode normalization.
2.  Lowercase.
3.  Replace punctuation with spaces.
4.  Normalize `&` to `and`.
5.  Collapse repeated whitespace.
6.  Normalize common legal suffixes.
7.  Remove or standardize punctuation.
8.  Preserve useful tokens.
9.  Optionally create:
    -   token-sorted representation
    -   alphanumeric-only representation
    -   character n-gram representation

Example:

``` text
"ABC Pvt. Ltd."
→
"abc private limited"
```

Do not over-normalize names in a way that removes
business-distinguishing tokens.

## Address normalization

Create several representations:

-   normalized lowercase string
-   alphanumeric string
-   token set
-   token-sorted string
-   extracted numeric tokens
-   postal/PIN tokens where available
-   state/city tokens where available
-   landmark tokens where available

Do not assume every address contains every component.

------------------------------------------------------------------------

# 10. Candidate Generation / Blocking

Candidate generation determines the **recall ceiling**.

Avoid comparing every S1 record against every S2/S3 record if the
dataset is large.

Use multiple blocking strategies and take their union.

### Suggested blocks

#### Block A --- exact normalized name

``` text
normalized_name
```

#### Block B --- name prefix / character n-gram

Use selected character prefixes or n-grams from normalized names.

#### Block C --- address tokens

Block on distinctive address tokens.

#### Block D --- postal/PIN code

When available, exact PIN/postal code is a strong blocking signal.

#### Block E --- country

Use country as a blocking feature when appropriate, but do not hard-code
the possible countries.

#### Block F --- rare name tokens

Use uncommon tokens in the business name.

#### Block G --- combined blocks

Examples:

``` text
country + normalized name prefix
country + postal code
rare name token + address token
```

### Candidate rule

A candidate can enter the final candidate set if it is retrieved by
**any sufficiently permissive block**.

The final matching model should perform the precision filtering.

------------------------------------------------------------------------

# 11. Pairwise Similarity Features

For every candidate `(S1, S2/S3)` pair, compute multiple features.

## Name features

Recommended:

-   exact normalized-name match
-   Jaccard similarity
-   token-set similarity
-   token-sort similarity
-   edit/Levenshtein similarity
-   character n-gram similarity
-   TF-IDF cosine similarity
-   containment:
    -   name A contained in name B
    -   name B contained in name A
-   token overlap count
-   rare-token overlap
-   length difference
-   number of shared numeric tokens

Example:

``` text
name_jaccard
name_levenshtein_ratio
name_tfidf_cosine
name_token_overlap
name_exact
```

## Address features

Recommended:

-   exact normalized-address match
-   token Jaccard
-   token overlap
-   character n-gram similarity
-   TF-IDF cosine
-   edit similarity
-   shared numeric tokens
-   shared postal/PIN
-   shared city
-   shared state
-   address-token containment
-   address length difference

## Cross-field features

Useful features include:

``` text
same_country
name_exact AND address_exact
name_high AND address_high
name_high AND shared_postcode
name_high AND shared_numeric_tokens
```

The strongest pairs often have agreement across **both name and
address**.

------------------------------------------------------------------------

# 12. Model Options

A strong practical approach is a supervised pairwise classifier trained
from the provided training ground truth.

Possible models:

-   Logistic Regression
-   Random Forest
-   Gradient Boosted Trees
-   XGBoost/LightGBM if permitted by the environment
-   HistGradientBoosting
-   CatBoost if allowed

The final model must comply with the challenge's model-license
restriction:

> Final model should be MIT/Apache 2.0 licensed and no more than 8
> billion parameters.

For a tabular similarity-feature problem, a compact tree-based model or
logistic model is sufficient and easier to reproduce.

------------------------------------------------------------------------

# 13. Creating Training Pair Labels

Convert the ground truth into pair-level labels.

For each Source 1 entity:

``` text
positive pairs:
    (S1, every true S2/S3 match)

negative pairs:
    candidate pairs that are not true matches
```

Important:

**Do not train only on random negatives.**

Hard negatives are especially valuable:

-   same/similar business name but different address
-   same address but different business
-   similar abbreviations
-   same city + similar name
-   businesses sharing common words
-   records differing only in distinguishing numbers

These teach the model to avoid false merges.

------------------------------------------------------------------------

# 14. Validation Strategy

Since the test ground truth is unavailable, create a validation split
from the training data.

Recommended:

``` text
training data
    ↓
train / validation split
    ↓
fit candidate generation on training
    ↓
generate candidates for validation
    ↓
score candidates
    ↓
threshold
    ↓
calculate entity-level F0.5
```

Avoid leakage.

If normalization/blocking statistics are learned from the data, ensure
validation information is not improperly used to train the model.

------------------------------------------------------------------------

# 15. Threshold Selection

Do not automatically use:

``` text
threshold = 0.5
```

Optimize the decision threshold against a held-out validation set using
the **same macro-averaged F0.5 logic** used by the challenge.

Test a range of thresholds.

Because F0.5 is precision-heavy, a higher threshold may be useful when
weak matches create many false merges.

However, select the threshold empirically from validation data rather
than assuming a universal value.

------------------------------------------------------------------------

# 16. Per-Entity Matching

For every S1 entity:

1.  Generate candidates from S2 and S3.
2.  Compute pairwise features.
3.  Predict match probability/score.
4.  Apply the selected threshold.
5.  Collect all accepted S2/S3 IDs.
6.  Remove duplicates.
7.  Preserve valid IDs only.
8.  If no pair passes the threshold, output an empty list.

A Source 1 entity can have **multiple valid matches**.

Do not force exactly one match.

------------------------------------------------------------------------

# 17. Important Precision Safeguards

Because F0.5 is precision-heavy, consider requiring stronger evidence in
ambiguous cases.

Examples:

### Very strong

``` text
exact normalized name
+
exact/near-exact address
```

### Strong

``` text
very high name similarity
+
strong address similarity
+
same country
```

### Potentially ambiguous

``` text
high name similarity
+
weak/no address evidence
```

### Dangerous

``` text
common business name only
```

The final decision should be based on validation performance, not
hand-written intuition alone.

------------------------------------------------------------------------

# 18. Source-Specific Modeling

S2 and S3 may contain different noise patterns.

Possible approaches:

### One model

Train one classifier with:

``` text
source_pair = S2 or S3
```

as a feature.

### Two models

Train:

``` text
S1 ↔ S2 model
S1 ↔ S3 model
```

separately.

Compare both approaches using validation F0.5.

Source-specific models can help when the two sources have substantially
different formatting/noise characteristics.

------------------------------------------------------------------------

# 19. Candidate File Requirement

The final submission must contain:

``` text
output/matching_results.tsv
output/candidate_pairs.tsv
```

`candidate_pairs.tsv` is the candidate set **actually fed into the final
matching model for inference**.

It is not merely an early blocking output if additional filtering occurs
afterward.

Every ID in `matching_results.tsv` must appear in the corresponding
candidate list.

------------------------------------------------------------------------

# 20. `matching_results.tsv`

Columns:

``` text
source1_entity_id
matched_entity_ids
```

Example:

``` text
source1_entity_id   matched_entity_ids
S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003    
```

Rules:

-   exactly one row per test Source 1 entity
-   include every Source 1 test entity
-   empty `matched_entity_ids` for no matches
-   no duplicate IDs
-   IDs must come only from S2/S3
-   IDs must exist in the test set

------------------------------------------------------------------------

# 21. `candidate_pairs.tsv`

Columns:

``` text
source1_entity_id
candidate_entity_ids
```

Example:

``` text
source1_entity_id   candidate_entity_ids
S1-00001    S2-00047,S2-00193,S3-00812,S3-00999
S1-00002    S3-00004
S1-00003    
```

Rules:

-   exactly one row per Source 1 entity
-   empty candidate list is allowed
-   no duplicate candidate IDs
-   candidates must be S2/S3 IDs from the test set
-   every final matched ID must occur in the candidate list

------------------------------------------------------------------------

# 22. Submission Validation

The challenge provides:

``` text
utils/validate_submission.py
```

Run:

``` bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

It prints:

``` text
PASS
```

when the files satisfy the required format.

Otherwise it prints a numbered list of issues.

This validator checks output correctness/format. It does **not**
calculate your leaderboard score.

------------------------------------------------------------------------

# 23. Final Submission Package

The expected ZIP structure is:

``` text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

### `output/`

Contains:

-   final matches
-   candidate-generation set

### `code/business_entity_resolution/`

Must be a self-contained runnable copy of the pipeline.

Put source code under:

``` text
src/
```

The README should contain exact reproduction instructions.

`requirements.txt` should pin dependency/environment versions.

The pipeline should be able to regenerate both output files using only
the provided training/test data and files contained in the submission
package.

------------------------------------------------------------------------

# 24. Methodology Documentation

The methodology document should describe:

1.  Methodology used.
2.  Candidate generation/blocking strategy.
3.  Model architecture.
4.  Feature engineering.
5.  Threshold selection.
6.  Validation methodology.
7.  Any other relevant implementation details.

There is no page limit. Technical clarity is preferred over unnecessary
brevity.

------------------------------------------------------------------------

# 25. Academic Integrity / Fair Play

## Strictly prohibited: external data lookup

Do **not** use:

-   commercial entity-resolution APIs/services
-   external business databases
-   government business-registration databases
-   geocoding APIs
-   internet-based entity lookup
-   external data augmentation

The challenge is intended to be solved using the provided data.

External lookup can result in disqualification.

The model should learn entity relationships from the supplied training
data rather than from external knowledge.

------------------------------------------------------------------------

# 26. AI-Assisted Solving Workflow

Use AI assistance for:

-   designing the pipeline
-   debugging code
-   explaining similarity metrics
-   generating feature-engineering code
-   designing validation experiments
-   reviewing candidate-generation logic
-   checking TSV formatting
-   reviewing leakage risks
-   improving documentation
-   analyzing validation results

AI assistance should still operate within the challenge rules.

Do not use AI or external services to perform prohibited external
business-identity lookups.

------------------------------------------------------------------------

# 27. Recommended Project Structure

``` text
business_entity_resolution/
├── src/
│   ├── config.py
│   ├── io.py
│   ├── normalize.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── README.md
└── requirements.txt
```

Suggested pipeline:

``` text
pipeline.py
    │
    ├── load data
    │
    ├── normalize fields
    │
    ├── train pair model
    │
    ├── generate test candidates
    │
    ├── calculate pair features
    │
    ├── predict pair scores
    │
    ├── apply threshold
    │
    ├── build matching_results.tsv
    │
    ├── build candidate_pairs.tsv
    │
    └── validate outputs
```

------------------------------------------------------------------------

# 28. Suggested Experiment Plan

## Experiment 1 --- Exact matching baseline

Use:

``` text
normalized_name exact
OR
normalized_name + address exact
```

Measure validation F0.5.

## Experiment 2 --- Fuzzy similarity

Add:

-   Levenshtein
-   Jaccard
-   token overlap
-   TF-IDF cosine
-   character n-gram similarity

Measure improvement.

## Experiment 3 --- Blocking improvements

Add multiple blocking rules and measure:

``` text
candidate recall
candidate count
reduction ratio
```

The candidate set must preserve true matches.

## Experiment 4 --- Hard-negative training

Add difficult non-match pairs.

Measure whether false merges decrease.

## Experiment 5 --- Threshold tuning

Evaluate many thresholds using validation F0.5.

## Experiment 6 --- Source-specific models

Compare:

``` text
single model
```

versus:

``` text
S2-specific model
S3-specific model
```

## Experiment 7 --- Conservative decision rules

Test whether additional evidence requirements improve precision without
excessive recall loss.

------------------------------------------------------------------------

# 29. Useful Metrics During Development

Do not only track F0.5.

Track:

### Pair precision

``` text
TP / (TP + FP)
```

### Pair recall

``` text
TP / (TP + FN)
```

### F0.5

Use the challenge formula.

### Candidate recall

Fraction of true matches that appear in the candidate set.

This is critical because:

``` text
candidate recall = maximum possible matching recall
```

if the final model can only select from candidates.

### Reduction ratio

Measure how much the candidate-generation stage reduces the theoretical
Cartesian product.

### Singleton accuracy

Measure how often entities with no true matches are correctly predicted
as having no matches.

------------------------------------------------------------------------

# 30. Key Failure Modes to Watch

## Failure 1 --- Candidate blocking is too strict

Symptom:

``` text
high precision
low recall
```

Cause:

True matches never reach the matching model.

Fix:

-   add more blocking rules
-   relax selected blocks
-   use union of blocks
-   use fuzzy/character-based retrieval

## Failure 2 --- Candidate set is too broad

Symptom:

``` text
huge candidate count
many false positives
```

Fix:

-   improve features
-   use stronger final threshold
-   add hard-negative training
-   introduce source-aware features

## Failure 3 --- Name-only matching

Common business names can refer to multiple businesses.

Fix:

Use address evidence and other fields.

## Failure 4 --- Address-only matching

Different businesses can share addresses, malls, offices, or landmarks.

Fix:

Combine address with name.

## Failure 5 --- Over-normalization

Removing too much information can merge distinct businesses.

Fix:

Keep multiple raw/normalized representations.

## Failure 6 --- Ignoring multiple matches

A Source 1 entity can match multiple S2/S3 records.

Do not force one-to-one matching.

## Failure 7 --- Forgetting singletons

A blank prediction can be the correct answer.

## Failure 8 --- Country hard-coding

France appears in test data.

Do not restrict the pipeline to US/India.

## Failure 9 --- Output mismatch

Typical problems:

-   comma-separated file instead of TSV
-   wrong column names
-   missing S1 rows
-   duplicate IDs
-   invalid S2/S3 IDs
-   matched IDs not present in candidates

Always run the validator.

------------------------------------------------------------------------

# 31. AI Prompt Template for Coding Assistance

Use a prompt such as:

> I am solving a business entity resolution hackathon using only the
> provided train/test TSV files. Source 1 is the reference entity set,
> and Source 2/Source 3 contain noisy records. The evaluation is
> macro-averaged F0.5 per Source 1 entity, so precision is weighted more
> heavily than recall. External entity lookup, geocoding, business
> databases, and external data augmentation are prohibited.
>
> Design/implement a reproducible Python pipeline with:
>
> 1.  TSV loading with `sep="\t"`.
> 2.  Name and address normalization.
> 3.  High-recall multi-rule candidate generation.
> 4.  Pairwise name/address similarity features.
> 5.  Supervised pair classification using training ground truth.
> 6.  Hard-negative generation.
> 7.  Validation-based threshold selection using macro F0.5.
> 8.  Conservative singleton handling.
> 9.  Test inference.
> 10. `matching_results.tsv` and `candidate_pairs.tsv` generation.
> 11. Output validation.
>
> Do not use external business identity data or external lookup
> services. Keep the implementation self-contained and reproducible.

------------------------------------------------------------------------

# 32. AI Prompt Template for Debugging

> Review this entity-resolution pipeline for:
>
> -   data leakage
> -   candidate recall problems
> -   false-positive sources
> -   normalization errors
> -   duplicate output IDs
> -   invalid candidate IDs
> -   incorrect F0.5 calculation
> -   singleton handling
> -   source-specific issues
> -   train/test inconsistencies
> -   output-format violations
>
> Suggest concrete code changes without using external business data.

------------------------------------------------------------------------

# 33. AI Prompt Template for Improving F0.5

> Analyze my validation results for a business entity-resolution task
> evaluated with macro F0.5. False positives are more costly than false
> negatives. Review candidate recall, pairwise precision/recall,
> threshold curves, singleton performance, and hard-negative errors.
> Suggest experiments that can improve validation F0.5 while respecting
> the rule that all entity-resolution evidence must come only from the
> supplied data.

------------------------------------------------------------------------

# 34. Final Checklist

Before submission:

-   [ ] All TSV files are read/written with tab separation.
-   [ ] Every test Source 1 entity appears exactly once.
-   [ ] No duplicate Source 1 rows.
-   [ ] `matched_entity_ids` contains only valid S2/S3 test IDs.
-   [ ] No duplicate matched IDs.
-   [ ] Empty lists are represented correctly.
-   [ ] Every final match occurs in `candidate_pairs.tsv`.
-   [ ] Candidate IDs are valid S2/S3 test IDs.
-   [ ] Candidate lists contain no duplicates.
-   [ ] France is handled without hard-coded country filtering.
-   [ ] External lookup/data augmentation is not used.
-   [ ] Validation uses macro F0.5.
-   [ ] Singleton behavior has been evaluated.
-   [ ] Candidate recall has been measured.
-   [ ] Threshold has been selected using held-out validation.
-   [ ] Hard negatives have been considered.
-   [ ] `utils/validate_submission.py` returns `PASS`.
-   [ ] `README.md` contains exact reproduction commands.
-   [ ] `requirements.txt` pins dependencies.
-   [ ] Methodology documentation is included.
-   [ ] Final ZIP has the required structure.

------------------------------------------------------------------------

# 35. Core Takeaway

The challenge is fundamentally a **high-precision entity-resolution
pipeline**:

``` text
                TRAINING DATA
                     │
                     ▼
              Normalization
                     │
                     ▼
          High-recall Blocking
                     │
                     ▼
             Candidate Pairs
                     │
                     ▼
       Name + Address Features
                     │
                     ▼
        Supervised Pair Model
                     │
                     ▼
         Validation Threshold
                     │
                     ▼
       Conservative Predictions
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
matching_results.tsv     candidate_pairs.tsv
```

The most important principles are:

1.  **Candidate generation controls recall.**
2.  **The final matcher controls precision.**
3.  **F0.5 makes false merges expensive.**
4.  **Singletons must be handled explicitly.**
5.  **Name and address evidence should be combined.**
6.  **Hard negatives are important for precision.**
7.  **Use validation to select thresholds rather than guessing.**
8.  **Never rely on external business-identity lookup.**
9.  **Every final match must exist in the candidate set.**
10. **Validate the exact submission files before uploading.**
