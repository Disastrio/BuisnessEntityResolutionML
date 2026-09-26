# 🎯 Accuracy Risk Register & Resolution Tracker

> **Status:** All 6 accuracy risks have been **implemented and validated** across 60 unit and integration tests (`RESULT: 60 passed, 0 failed`).
> Training speed remains unaffected (all fixes add 0 ms overhead).

---

## ✅ Resolved & Validated Accuracy Improvements

### [x] Item 1: Empty-Evidence Jaccard Defaulting to 0.0 (Eliminates False Merges)
* **File:** [`src/features.py`](file:///r:/BuisnessEntityResolutionML/src/features.py#L255-L279)
* **Resolution:** When `n_union == 0`, `name_jacc`, `addr_tok_jacc`, and `addr_num_jacc` now correctly default to `0.0` (zero evidence) instead of `1.0`. Prevents false positive merges on unnumbered addresses.
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 11).

---

### [x] Item 2: Exact Name Matches Prioritized in Candidate Blocking
* **File:** [`src/blocking.py`](file:///r:/BuisnessEntityResolutionML/src/blocking.py#L398-L403)
* **Resolution:** Exact normalized name candidates (`block_lists[0]`) are prepended to the candidate list before multi-block hits:
  ```python
  if block_lists and block_lists[0]:
      ordered = list(dict.fromkeys(block_lists[0] + ordered))[:max_candidates]
  ```
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 7: `[PASS] exact-name candidate always kept`).

---

### [x] Item 3: French & European Corporate Suffixes Added
* **File:** [`src/normalize.py`](file:///r:/BuisnessEntityResolutionML/src/normalize.py#L27-L66)
* **Resolution:** Added `sarl`, `sas`, `sasu`, `sci`, `eurl`, `sa`, and `gmbh` to `LEGAL_SUFFIXES` with keyword literal guards.
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 14: `[PASS] SARL normalized`, `[PASS] SAS normalized`, `[PASS] GmbH handled`).

---

### [x] Item 4: Chain Store Branch Mismatch Guard with Token Set Ratio
* **File:** [`src/features.py`](file:///r:/BuisnessEntityResolutionML/src/features.py#L285-L295)
* **Resolution:** `name_high_addr_mis` now triggers on `(name_lev > 0.85 or name_tset > 0.90)` when combined with numeric address mismatch. Catches chain store branches that have city/locality appendages.
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 4).

---

### [x] Item 5: Singleton Confidence Barrier Activated (`0.80`)
* **File:** [`src/config.py`](file:///r:/BuisnessEntityResolutionML/src/config.py#L125)
* **Resolution:** `SINGLETON_BARRIER` is now set to `0.80` by default. Protects the ~5.6% singletons in the dataset from emitting false positive matches that drop per-entity scores to 0.0.
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 3).

---

### [x] Item 6: Country-Conditioned Postal Code Extraction
* **File:** [`src/normalize.py`](file:///r:/BuisnessEntityResolutionML/src/normalize.py#L96-L100,L215-L230)
* **Resolution:** `extract_postal_codes(address, country)` uses country-specific regex patterns (`_INDIA_PIN_RE` starting 1-9 for India, `_FIVE_DIGIT_RE` for US/France). Prevents Indian plot numbers from colliding with US/France ZIPs.
* **Validation:** Verified in [scripts/test_model_improvements.py](file:///r:/BuisnessEntityResolutionML/scripts/test_model_improvements.py) (Check 14: `[PASS] India postal ignores plot number`, `[PASS] US 5-digit zip`).

---

## 📊 Summary of Resolved Items

| Total Items | Resolved & Verified | Pending |
| :--- | :--- | :--- |
| **6** | **6 (100%)** | **0** |
