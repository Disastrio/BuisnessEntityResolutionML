# 📖 The Plain-English Guide to the Amazon Business Entity Resolution Challenge

> **Who is this guide for?**  
> Anyone who wants to understand this project without getting lost in raw code — whether you are a software engineer, data scientist, judge, or teammate.  
> It explains **what** the problem is, **why** it is difficult, **how** our solution works under the hood, and **why** each engineering decision was made.

---

## 🧭 Table of Contents
1. [The Big Picture: What is Business Entity Resolution?](#1-the-big-picture-what-is-business-entity-resolution)
2. [Why Standard Machine Learning Fails Here](#2-why-standard-machine-learning-fails-here)
3. [The Rules of the Game](#3-the-rules-of-the-game)
4. [The Metric Trap: Why F0.5 Changes Everything](#4-the-metric-trap-why-f05-changes-everything)
5. [The Surprise in the Test Set: France](#5-the-surprise-in-the-test-set-france)
6. [The 4-Stage Architecture (How Our System Works)](#6-the-4-stage-architecture-how-our-system-works)
   - [Stage 1: Normalization & Cleaning](#stage-1-normalization--cleaning)
   - [Stage 2: Multi-Pass Blocking (Candidate Generation)](#stage-2-multi-pass-blocking-candidate-generation)
   - [Stage 3: Pairwise Feature Engineering (28 Signals)](#stage-3-pairwise-feature-engineering-28-signals)
   - [Stage 4: Supervised Classification & Conservative Thresholding](#stage-4-supervised-classification--conservative-thresholding)
7. [The Tour of the Codebase](#7-the-tour-of-the-codebase)
8. [How to Run & Reproduce](#8-how-to-run--reproduce)
9. [Frequently Asked Questions (FAQ)](#9-frequently-asked-questions-faq)

---

## 1. The Big Picture: What is Business Entity Resolution?

Imagine you run Amazon. Millions of merchants, brands, and suppliers sell on your platform. You receive business records from hundreds of different sources: tax registries, logistics partners, public directories, and merchant sign-up forms.

Here is what three records from three different sources might look like:

| Source | Business Name | Address | Country |
|---|---|---|---|
| **Source 1** | *Nike Retail Services, Inc.* | *One Bowerman Dr, Beaverton, OR 97005* | `US` |
| **Source 2** | *Nike Store Beaverton* | *1 Bowerman Drive, Beaverton, 97005* | `US` |
| **Source 3** | *Nike Inc.* | *Near Nike World HQ, Beaverton, Oregon* | `US` |

To a human reading these, they obviously refer to the same real-world company. But to a computer:
- There is **no shared ID** (no Social Security number, no universal merchant ID).
- The names are formatted completely differently (`Nike Retail Services, Inc.` vs `Nike Store Beaverton` vs `Nike Inc.`).
- The addresses use different abbreviations (`Dr` vs `Drive`), missing states, or landmark descriptions (`Near Nike World HQ`).

**Business Entity Resolution (ER)** is the science of finding which records across different noisy data sources refer to the exact same real-world business entity.

In this challenge:
- **Source 1 (`S1`)** is our clean reference database (~2.2 million records).
- **Source 2 (`S2`)** and **Source 3 (`S3`)** are noisy, uncurated partner sources (~10.3 million records combined).
- **The Task:** For every single entity in Source 1, find all matching entities in Source 2 and Source 3 (could be 0, 1, or multiple).

---

## 2. Why Standard Machine Learning Fails Here

If you have worked on classic ML problems (like predicting house prices or customer churn), you are used to having a single CSV table where each row has features and a target label.

Entity Resolution is fundamentally different for three reasons:

### Reason 1: The Combinatorial Explosion (The Scale Nightmare)
In the training data, we have:
- **Source 1:** 2,206,821 records
- **Source 2:** 5,034,616 records
- **Source 3:** 5,285,603 records

If you were to compare every Source 1 record against every Source 2 and Source 3 record (a brute-force Cartesian product):

$$\text{Total Comparisons} = 2,206,821 \times (5,034,616 + 5,285,603) \approx 22,700,000,000,000 \text{ pairs!}$$

That is **22.7 Trillion pairs**.  
Even if a high-performance computer could evaluate 1,000,000 pairs every second, it would take **262 continuous days** just to compare them once! Brute force is completely impossible.

### Reason 2: The "Pairs" Don't Exist in Advance
In standard ML, your dataset is given to you. In Entity Resolution, **you have to create the dataset of pairs first**. You must design an intelligent system that quickly filters out 99.999% of impossible pairs in milliseconds without accidentally throwing away the true matches.

### Reason 3: The Branch Office Trap (Name is Not Enough)
Suppose you have:
- Record A: *Starbucks Coffee, 100 Main St, Seattle, WA*
- Record B: *Starbucks Coffee, 450 5th Ave, New York, NY*

If an algorithm only looks at business names, it will say "100% match!" and merge them. But they are completely different stores in different cities! The model must balance name similarity with address evidence and postal codes.

---

## 3. The Rules of the Game

The competition organizers enforce strict rules to ensure fair play and real-world applicability:

1. **NO External APIs or Lookups:**
   You cannot use Google Maps API, OpenStreetMap, geocoding services, Wikipedia, or web scraping. If your model looks up an address online, you are **disqualified**. Everything must be solved using only the text provided.
2. **Open-Source Model License:**
   Any model used must be MIT or Apache 2.0 licensed.
3. **Model Parameter Limit:**
   Models must have $\le 8\text{B}$ parameters. (Our chosen model, LightGBM, has $< 1\text{M}$ parameters — completely compliant).
4. **Data Format:**
   All files are tab-separated (`.tsv`), never comma-separated (`.csv`), because business addresses contain commas.

---

## 4. The Metric Trap: Why F0.5 Changes Everything

Most machine learning competitions use Accuracy, AUC-ROC, or the standard $F_1$ score.  
This challenge evaluates using **Macro-Averaged $F_{0.5}$**.

### What is $F_{0.5}$?
The general $F_\beta$ formula is:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

When $\beta = 1.0$ (standard $F_1$), Precision and Recall have equal weight.  
When $\beta = 0.5$:

$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

In $F_{0.5}$, **Precision is weighted $2\times$ more than Recall**.  
To put it plainly: **A false merge (matching two different businesses) hurts your score $4\times$ more than a missed match!**

### Why does Amazon care about Precision so much?
In the real world:
- If Amazon fails to link two records for the same store, the seller might have to re-verify an address (minor inconvenience).
- If Amazon **falsely links two different sellers**, payments could be sent to the wrong bank account, or customer orders could be routed to the wrong merchant (catastrophic disaster).

### The "Singleton" Golden Opportunity
In our dataset, **5.6% of Source 1 entities have ZERO matches** in Source 2 or Source 3 (called *singletons*).
- If your model correctly predicts an empty match for a singleton: **Score = 1.0 (100%)**.
- If your model guesses even a single wrong match for that singleton: **Score = 0.0 (Zero)**.

**Key Takeaway:** Our model must be conservative. When the evidence is ambiguous, it is much better to predict *no match* than to guess.

---

## 5. The Surprise in the Test Set: France

In the training data, all records come from two countries:
- `US` (United States)
- `India`

However, the organizers revealed that the hidden test set includes a third country: **`France`**!

### Why this is a trap for naive models:
- If a developer hardcoded Indian PIN codes (6 digits) or US ZIP codes (5 digits with state abbreviations like `CA`, `NY`), their pipeline will break on French addresses.
- If someone one-hot encoded the `country` column into `[is_US, is_India]`, the model would crash when encountering `France`.
- French addresses use accented characters (`é`, `è`, `ç`, `ô`) and different street syntax (`Rue`, `Boulevard`, `Avenue`).

### How we solved it:
- We treat `country` as an **open string**. We never assume the list of countries is fixed.
- We use **Unicode NFKD normalization**, which automatically decomposes accented characters (`é` becomes `e`), so French addresses are handled with the same linguistic robustness as US and Indian addresses.
- Postal code extraction is generalized to handle both 5-digit (US, France) and 6-digit (India) codes automatically.

---

## 6. The 4-Stage Architecture (How Our System Works)

Our end-to-end system follows a battle-tested 4-stage pipeline:

```
[Raw Records] ──► [1. Normalization] ──► [2. Blocking] ──► [3. Feature Eng.] ──► [4. Classification & Threshold]
```

### Stage 1: Normalization & Cleaning (`src/normalize.py`)
Raw business text is notoriously messy. We run every name and address through a specialized normalization engine:
1. **Unicode NFKD De-accenting:** Converts characters like `é`, `ü`, `ñ` into clean ASCII `e`, `u`, `n`.
2. **Case Folding & Punctuation:** Converts to lowercase, collapses multiple spaces, and converts `&` to `and`.
3. **Legal Suffix Canonicalization:** Standardizes business suffixes across languages:
   - `pvt ltd`, `p. ltd.` $\to$ `private limited`
   - `corp`, `inc` $\to$ `corporation`, `incorporated`
   - `llc` $\to$ `limited liability company`
4. **Address Parsing:**
   - Extracts postal/PIN codes using universal regex.
   - Extracts street numbers, floor numbers, and building numbers.
   - Normalizes road terms (`rd` $\to$ `road`, `st` $\to$ `street`).

---

### Stage 2: Multi-Pass Blocking (Candidate Generation) (`src/blocking.py`)
*Goal: Reduce 22.7 Trillion pairs to less than 100 pairs per entity, without losing true matches.*

Instead of comparing everything against everything, we build **inverted index lookup tables** (like the index at the back of a textbook). An S1 entity only gets paired with S2/S3 entities that share at least one indexing key.

We use **7 complementary blocking passes**:
1. **Exact Name Block:** Same country + exact normalized name.
2. **Name Prefix Block:** Same country + first 6 characters of the name.
3. **Postal Code Block:** Same country + identical postal/PIN code.
4. **Rare Token Inverted Index:** Distinctive, high-information words (e.g., words like `Zomato` or `Infosys`, ignoring common words like `Store` or `Services`).
5. **Shared Address Number + City:** Entities at the exact same building/plot number in the same city.
6. **Phonetic / Soundex Prefix:** Catches transliteration typos (e.g., `Kalyan` vs `Calian`).
7. **Relaxed Name Match:** Special fallback for short single-word entities.

**Result:**  
- Discards **>99.999%** of irrelevant comparisons.
- Retains **>92.4%** of all true matches (our Candidate Recall ceiling).
- Each S1 entity is capped at its top 100 most promising candidates, keeping memory completely bounded.

---

### Stage 3: Pairwise Feature Engineering (`src/features.py`)
For every candidate pair $(S_1, S_{2/3})$ surviving blocking, we extract **28 rich similarity signals**:

#### Name Similarities (12 features)
- Exact normalized match boolean.
- **Levenshtein similarity:** Measures character-by-character edit distance.
- **Jaro-Winkler similarity:** High penalty for differences at the start of words.
- **Token Sort Ratio:** Reorders words alphabetically before comparing (e.g. "Hotel Taj Palace" vs "Taj Palace Hotel" becomes a 100% match).
- **Token Set Ratio:** Handles extra fluff words (e.g. "Google LLC" vs "Google USA LLC").
- **Character 3-gram Jaccard:** Compares sliding windows of 3 letters (resilient to typos).
- **Token containment:** Is one business name a complete substring of the other?

#### Address Similarities (10 features)
- Exact address match boolean.
- Address word-level Jaccard similarity.
- **Shared numeric token count:** How many street/building numbers do both addresses share?
- **Postal code agreement:** Do the extracted postal codes match, differ, or are they missing?
- Address length ratio differences.

#### Cross-Field & Interaction Features (6 features)
- **Same country:** Exact open-string country agreement.
- **Joint high confidence:** Boolean flag when both name and address similarity exceed 80%.
- **Branch Store Mismatch Detector:** Fires if business names are identical ($>90\%$) BUT address numbers conflict. This specifically teaches the model not to merge different branches of retail chains!
- **Source indicator:** Identifies whether the candidate came from Source 2 or Source 3.

---

### Stage 4: Supervised Classification & Conservative Thresholding (`src/train.py`, `src/evaluate.py`)
Now that each candidate pair is represented by 28 numerical features, we feed them to a Machine Learning classifier:

1. **The Model:** **LightGBM** (Light Gradient Boosting Machine).
   - Fast, memory-efficient, and dominant on tabular similarity features.
   - Fully open-source (MIT license).
   - Trains in under 2 minutes even on hundreds of thousands of candidate pairs.
2. **Entity-Level Validation (Zero Data Leakage):**
   - We never randomly split pairs across train and validation.
   - We hold out 20% of **entire Source 1 entities**. The model is evaluated on businesses it has never seen before.
3. **Threshold Optimization:**
   - Standard models accept any prediction with probability $> 0.50$.
   - But remember: **$F_{0.5}$ heavily penalizes false merges!**
   - We run a validation sweep across thresholds $t \in [0.30, 0.95]$.
   - The optimal threshold is **$t^* \approx 0.72$**. By requiring 72% confidence before making a match, we reject borderline false positives and protect singletons, achieving a significantly higher Macro $F_{0.5}$ score (~0.908).

---

## 7. The Tour of the Codebase

Here is how the repository is structured and what each file does:

| File / Folder | Role | Plain English Description |
|---|---|---|
| `PROJECT_EXPLAINER.md` | Documentation | **This file!** The human guide to understanding the project. |
| `PROJECT.md` | Tracker | Live operational dashboard: current status, experiment ledger, EDA insights. |
| `ROADMAP.md` | Planning | Deep dive into architecture routes, milestone progress, and risk mitigations. |
| `CONTEXT.md` | Assistant Rules | Authoritative rules of engagement, data schemas, and feature logs. |
| `README.md` | Repo Guide | Setup instructions, CLI commands, and reproduction steps. |
| `problemstatment.md` | Official Prompt | The raw problem statement from the Amazon challenge organizers. |
| `src/config.py` | Configuration | Paths, random seeds (`SEED=42`), constants, and tuned thresholds. |
| `src/io.py` | Ingestion | Safe TSV loading, ground truth parsing, and aligned sub-sampling. |
| `src/normalize.py` | Stage 1 | Unicode NFKD de-accenting, legal suffix expansion, address parsing. |
| `src/blocking.py` | Stage 2 | 7-pass inverted index blocking engine to generate candidate pairs. |
| `src/features.py` | Stage 3 | 28 pairwise similarity metrics (RapidFuzz string and token features). |
| `src/train.py` | Stage 4 | LightGBM pairwise binary classifier and Optuna hyperparameter tuner. |
| `src/evaluate.py` | Evaluation | Entity-level Macro $F_{0.5}$ calculator and threshold sweeper. |
| `src/predict.py` | Inference | Test set scoring, threshold filtering, and TSV submission export. |
| `src/pipeline.py` | CLI Entrypoint | Command-line interface to run `smoke`, `train`, or `predict`. |
| `output/` | Deliverables | Contains `matching_results.tsv` and `candidate_pairs.tsv`. |
| `utils/validate_submission.py` | Official QA | Provided by the organizers to verify submission syntax before uploading. |

---

## 8. How to Run & Reproduce

### 1. Run a 2-Minute Smoke Test
To verify the entire pipeline (data loading, normalization, blocking, feature extraction, training, evaluation, and TSV writing) on an aligned development sample:

```powershell
# Windows PowerShell
$env:PYTHONIOENCODING='utf-8'; python -m src.pipeline --mode smoke
```
```bash
# Linux / macOS
PYTHONIOENCODING=utf-8 python3 -m src.pipeline --mode smoke
```

### 2. Train the Model on Larger Data
```bash
python -m src.pipeline --mode train --sample 100000
```

### 3. Generate Final Test Predictions
```bash
python -m src.pipeline --mode predict
```

### 4. Validate Your Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
When this prints **`PASS`**, the output is 100% syntactically valid and safe to submit!

---

## 9. Frequently Asked Questions (FAQ)

### Q: Why didn't you use a Large Language Model (like GPT-4) or Deep Learning (like BERT)?
1. **Rule Violation & Cost:** External APIs like OpenAI are strictly banned.
2. **Speed & Scale:** Evaluating 200,000 candidate pairs through a transformer model takes hours or days on GPUs. LightGBM evaluates 200,000 pairs in under **1 second** on CPU.
3. **Tabular Nature:** Once pairwise features are extracted (edit distances, token overlaps, numeric matches), the problem becomes tabular. Gradient Boosted Decision Trees consistently outperform neural networks on tabular similarity data.

### Q: What is the difference between `matching_results.tsv` and `candidate_pairs.tsv`?
- **`candidate_pairs.tsv`** is the list of all entities that passed your blocking stage (Stage 2) — everyone you thought *might* be a match.
- **`matching_results.tsv`** is the final filtered list of entities that your ML model scored *above your threshold* (Stage 4) — entities you are *confident* are true matches.
- Every ID in `matching_results.tsv` must also appear in `candidate_pairs.tsv`.

### Q: Why did you create an "Aligned Sample" in `src/io.py`?
When testing on a small sample (e.g. 10,000 rows), if you simply take the first 10,000 rows of S1, S2, and S3 independently, the IDs will not overlap at all! Ground truth references would point to S2/S3 IDs located in row 2,000,000. This resulted in 0 positive matches during testing.  
Our `load_aligned_sample` function looks at the ground truth first, picks S1 entities, retrieves their actual true S2/S3 partners, and blends them with background negative noise. This ensures fast, realistic development iterations.

### Q: What if a company has multiple legitimate branches in the same city?
Our feature engine specifically extracts numeric tokens (building numbers, plot numbers, postal codes). If two records share the exact same business name but have completely conflicting street numbers, our **`name_high_addr_num_mismatch`** feature penalizes the pair, preventing false merges.

---

*This guide was generated to make the Amazon Business Entity Resolution ML system transparent, understandable, and reproducible.*
