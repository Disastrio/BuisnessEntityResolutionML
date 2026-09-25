# CONTEXT.md — Amazon ML Hackathon Master Context

> **Purpose:** Single source of truth for all AI assistants and team members.
> Load this file at the start of every session. It consolidates the problem contract,
> data dictionary, feature log, experiment log, modeling decision tree, and all
> operational rules into one place.

---

## 0. AI Assistant Rules

When assisting with this project, always follow these rules:

1. Always reference this file and `DATA_DICTIONARY` section before suggesting code.
2. Never propose features that **leak the target** or use test-time-unavailable data.
3. All suggestions must include a **validation plan**.
4. Prefer **gradient boosting** for tabular data; justify any alternative.
5. Keep code **reproducible**: fixed seeds (`SEED=42`), deterministic splits.
6. **Flag every assumption** you make explicitly.
7. When a CV score is shared, state whether the change is within noise (`delta < 1 std`).

**Current State** *(update after every session)*:
- Best CV: `<score>` — Model: `<name>`
- Next experiment: `<hypothesis>`

---

## 1. Problem Contract

> Fill in after reading the problem statement. This is the anchor for everything.

| Field | Value |
|---|---|
| Task type | `<binary classification / regression / ...>` |
| Target column | `<column_name>` |
| Metric | `<F1 / AUC / RMSE / ...>` |
| Submission format | `<file>.csv` with columns `[id, target]` |
| Train rows | `N` |
| Test rows | `M` |
| Time limit | `X hours` |
| Compute | `CPU / GPU` |
| External data | `allowed / not allowed` |
| Time-series? | `Yes / No` |
| Class imbalance? | `Yes (ratio 1:X) / No` |

**Validation Strategy:** `StratifiedKFold k=5` *(change if time-series or grouped)*

**Key Risks:**
- Leakage in `<column>`
- High-cardinality: `<col1>, <col2>`
- Imbalance ratio: `<1:X>`

---

## 2. Data Dictionary

| Column | Type | Range / Categories | Missing % | Notes |
|--------|------|--------------------|-----------|-------|
| id     | int  | 0 – 1M             | 0%        | Drop for training |
| age    | int  | 18 – 90            | 2.1%      | Clip at 99th pct |
| cat_X  | str  | 500 categories     | 0%        | Target-encode |

> **Update this table immediately when a new column is discovered or dropped.**

---

## 3. Feature Engineering Log

### v1 — Baseline
- Raw numeric columns, **median-imputed**
- One-hot encoding for categoricals with < 10 levels

### v2 — Target Encoding
- CV target encoding for **all** categoricals (k=5, smoothing=10)
- Added derived feature: `ratio_A_B = A / (B + 1)`

### v3 — Lags *(time-series only)*
- 7 / 14 / 28-day lags of `<col>`
- Rolling mean & std windows (7 / 14 / 28 days)

---

## 4. Experiment Log

| ID  | Date | Change | CV Mean | CV Std | LB | Notes |
|-----|------|--------|---------|--------|----|-------|
| 001 | —    | Baseline LightGBM (default params) | 0.812 | 0.004 | — | Reference |
| 002 | —    | + target encoding | 0.826 | 0.003 | — | Keep |
| 003 | —    | + log1p target | 0.801 | 0.005 | — | Revert |

> **Rule:** Log every experiment here before moving to the next one.
> A gain of 0.001 with +-0.01 std is **noise** — do not keep it.

---

## 5. Model Selection Decision Tree

```
Is target labeled?
├── No  → clustering / dimensionality reduction / self-supervised
└── Yes
    ├── Target continuous?  → Regression (RMSE / MAE)
    │     ├── Linear         → Ridge / Lasso / ElasticNet
    │     ├── Non-linear tab → LightGBM / XGBoost / CatBoost  ← DEFAULT
    │     └── Time-series    → ARIMA / Prophet / LGBM with lags
    ├── Target categorical?
    │     ├── Binary         → Logistic / LGBM (scale_pos_weight)
    │     ├── Multi-class    → Softmax / LGBM multiclass
    │     └── Imbalanced     → focal loss / class weights / threshold tuning
    ├── Text?               → TF-IDF + Linear, or transformer fine-tune
    ├── Image?              → Pretrained CNN (ResNet / EfficientNet) + head
    └── Sequence?           → LSTM / GRU / Transformer
```

---

## 6. Modeling Workflow

### 6.1 Baseline First (Non-Negotiable)
- Regression → predict the **mean/median**
- Classification → predict **majority class**
- Then one simple model (LogisticRegression / Ridge)
- Every experiment must beat this floor

### 6.2 Model Progression
1. Linear / tree baseline — Logistic, Ridge, DecisionTree
2. Ensemble — RandomForest, ExtraTrees
3. **Gradient boosting** — LightGBM → XGBoost → CatBoost (try all three; pick by CV)
4. Neural nets — only if tabular is huge, or data is text / image / audio
5. Stacking / blending — combine diverse models with OOF predictions

### 6.3 Hyperparameter Tuning
- Tool: **Optuna** (TPE sampler)
- Budget: 50–200 trials for GBDTs
- Tune on **CV**, never on public leaderboard
- Log every trial

### 6.4 Ensembling Options
| Method | Description |
|---|---|
| Bagging | Same model, different seeds → average |
| Blending | Weighted average of diverse models (weights via ridge on OOF) |
| Stacking | Meta-model (Logistic / Ridge) on OOF predictions |
| Hill-climbing | Greedy weight search on OOF |

---

## 7. Validation Strategy Reference

| Data type | CV Strategy |
|---|---|
| i.i.d. tabular | StratifiedKFold(5-10) or RepeatedStratifiedKFold |
| Imbalanced | StratifiedKFold + class weights |
| Time-series | TimeSeriesSplit, expanding window, no shuffle |
| Grouped (users / patients) | GroupKFold |
| Small data (< 1k rows) | Leave-One-Out or Repeated 5-fold |
| Multi-label | IterativeStratification |

> **Never** tune on the public leaderboard — trust your local CV.
> Fix the fold split **once** and reuse for every experiment.

---

## 8. Guardrails — Things That Kill Submissions

- [ ] **Data leakage** — checked every feature for train/test-time availability
- [ ] **Overfitting to CV** — if CV ≈ 0.99 and LB ≈ 0.70, you leaked
- [ ] **Submission format** — row count, column names, ID order, dtypes
- [ ] **NaN in predictions** — always `assert not pred.isna().any()`
- [ ] **Class order** — `predict_proba` column order must match submission spec
- [ ] **Reproducibility** — fixed seeds, saved preprocessing pipeline
- [ ] **Runtime** — inference must fit hackathon limits

---

## 9. Iteration Loop

```
1. Form hypothesis  (e.g., "log1p target helps")
2. Run experiment with FIXED CV folds
3. Compare to current best (CV mean ± std)
4. If better  → keep, log to Experiment Log (Section 4)
5. If worse   → log why, revert, move on
6. Repeat until time budget or plateau
```

---

## 10. Time Budget (24-Hour Hackathon Reference)

| Phase | Time | Output |
|---|---|---|
| Problem framing | 30 min | Problem Contract (Section 1) |
| EDA | 1.5 h | EDA notebook + findings |
| Baseline | 1 h | CV score, submission pipeline |
| Feature engineering | 4 h | Feature set v1 |
| Model tuning | 4 h | Best single model |
| Ensembling | 2 h | Blend / stack |
| Final validation | 1 h | Sanity-checked submission |
| Buffer | remaining | Handle surprises |

---

## 11. EDA Checklist

- [ ] Target distribution — histogram, class balance, skew, outliers
- [ ] Missing values — pattern (MCAR / MAR / MNAR), % per column
- [ ] Cardinality — high-cardinality categoricals need special handling
- [ ] Leakage check — any column that wouldn't exist at inference time? Drop it
- [ ] Train vs test drift — compare distributions (KS test, adversarial validation)
- [ ] Correlations — heatmap for numerics, Cramér's V for categoricals
- [ ] Duplicates & near-duplicates — can inflate CV scores

### Target Leakage Red Flags
- Column perfectly correlated with target
- IDs that encode target
- Future-dated features in time-series
- Aggregates computed over train + test

---

## 12. Project Structure

```
amazon-ML/
├── CONTEXT.md              ← Rules & operational guidelines
├── PROJECT.md              ← Project tracker & experiment ledger
├── ROADMAP.md              ← Comprehensive execution routes & architecture
├── problemstatment.md      ← Official challenge specification
├── Documentation_template.md
├── requirements.txt        ← Pinned environment packages
├── .gitignore
├── dataset/                ← Challenge TSVs (train & test S1/S2/S3)
│   ├── train/
│   └── test/
├── src/                    ← Business Entity Resolution Pipeline
│   ├── config.py           ← Paths, constants, seeds, tuned thresholds
│   ├── io.py               ← Chunked TSV ingestion & GT parsers
│   ├── normalize.py        ← Name & address canonicalization
│   ├── blocking.py         ← 7-pass candidate indexing & candidate union
│   ├── features.py         ← Pairwise similarity features (~28 features)
│   ├── train.py            ← LightGBM pair classifier & Optuna tuning
│   ├── evaluate.py         ← Macro-F0.5 calculator & threshold search
│   ├── predict.py          ← Test candidate evaluation & thresholding
│   ├── ensemble.py         ← Bagging / model blending
│   └── pipeline.py         ← Full end-to-end execution runner
├── output/                 ← Challenge submission TSV files
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py ← Verification tool (must PASS)
├── models/                 ← Saved model binaries & vectorizers
├── reports/                ← EDA and validation performance reports
└── submissions/            ← Final ZIP archives
```

---

## 13. Environment Setup

```bash
# Install all dependencies
pip install -r requirements.txt

# Seeds (already handled in src/config.py)
# random.seed(42), np.random.seed(42), torch.manual_seed(42)
```

**Packages installed:**
`pandas · numpy · scikit-learn · lightgbm · xgboost · catboost · optuna · shap · matplotlib · seaborn · torch (optional)`

---

*Last updated: 2026-09-24 — consolidated from: AI_ASSISTANT_PROMPT.md, DATA_DICTIONARY.md, EXPERIMENTS.md, FEATURES.md, Model-Agnostic Decision Tree.txt, hackathon_basics.md*
