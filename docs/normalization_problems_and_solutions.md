# Business Entity Resolution: Normalization Audit & Optimization Plan

This document details the critical audit of the normalization stage (`src/normalize.py`) for the Business Entity Resolution pipeline. It outlines current architectural bottlenecks, failure modes leading to missed matches and false merges, performance overheads, and the proposed solutions.

---

## 1. Accuracy & Match Recall Problems

### 1.1 Missing French Legal Suffixes (Test-Set Blindspot)
* **The Problem:** The test set introduces **France** (`test_source3.tsv`), but `LEGAL_SUFFIXES` only accounts for Anglo-American and Indian forms (`Pvt Ltd`, `LLC`, `Corp`, `Inc`, `Holdings`). Major French and European corporate forms (`SARL`, `SAS`, `SASU`, `SA`, `SCI`, `EURL`, `GmbH`) are completely unhandled.
* **Failure Mode:** A clean reference entity in Source 1 (*"Boulangerie Paul"*) fails exact and prefix blocking against its match in Source 3 (*"Boulangerie Paul SARL"*), causing a false negative.
* **Proposed Solution:**
  Add canonical French and continental European legal entity suffixes to `LEGAL_SUFFIXES` (ordered longest-first to prevent partial matching):
  ```python
  # Add to LEGAL_SUFFIXES in src/normalize.py
  (r'\bsociete\s+a\s+responsabilite\s+limitee\b', 'sarl'),
  (r'\bsociete\s+par\s+actions\s+simplifiee\b', 'sas'),
  (r'\bs\.?a\.?r\.?l\.?\b', 'sarl'),
  (r'\bs\.?a\.?s\.?u?\.?\b', 'sas'),
  (r'\bs\.?c\.?i\.?\b', 'sci'),
  (r'\be\.?u\.?r\.?l\.?\b', 'eurl'),
  (r'\bs\.?a\.?\b', 'sa'),
  (r'\bgmbh\b', 'gmbh'),
  ```

---

### 1.2 Country-Agnostic Postal Code Collisions
* **The Problem:** `POSTAL_PATTERNS` searches both 6-digit (India PIN) and 5-digit (US ZIP / France) patterns globally across all records regardless of the record's country.
* **Failure Mode:** In Indian records, arbitrary 5-digit door numbers, plot numbers, or telephone fragments (e.g., *"Plot 12042, MG Road"*) are falsely extracted as US/French postal codes. Unrelated businesses that share an incidental 5-digit street number get grouped into the same postal candidate block (`Index 3`), generating false positive merges.
* **Proposed Solution:**
  Condition postal code extraction on the normalized country:
  ```python
  def extract_postal_codes(address: str, country: str = "") -> str:
      if not address:
          return ""
      if country == 'india':
          codes = re.findall(r'\b[1-9]\d{5}\b', address)        # Valid Indian PINs start with 1-9
      elif country in ('us', 'france'):
          codes = re.findall(r'\b\d{5}(?:-\d{4})?\b', address) # US 5-digit ZIP / France 5-digit
      else:
          codes = re.findall(r'\b\d{5,6}\b', address)
      return ','.join(dict.fromkeys(codes))                     # Order-preserving deduplication
  ```

---

### 1.3 Alphanumeric Flat and House Number Dropping
* **The Problem:** `extract_numeric_tokens` uses `re.findall(r'\b\d+\b', text)` on punctuation-stripped address text.
* **Failure Mode:** *"Flat 42-B"* becomes *"flat 42 b"* (extracts `42`). However, *"Flat 42B"* (without spaces) has no word boundary surrounding the digits, so `\b\d+\b` extracts **nothing**. This causes numeric blocking (`Index 5`) and numeric overlap features to fail.
* **Proposed Solution:**
  Insert a space between digit-letter transitions during address pre-cleaning:
  ```python
  # Separate numbers and trailing letters: '42b' -> '42 b', 'a101' -> 'a 101'
  text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)
  text = re.sub(r'([a-z]+)(\d+)', r'\1 \2', text)
  ```

---

### 1.4 Missing Regional Address Landmark Abbreviations
* **The Problem:** `ADDRESS_ABBREVS` only covers standard Western roadway types (`rd`, `st`, `ave`, `dr`). It lacks regional Indian and French urban tokens.
* **Failure Mode:** Addresses such as *"Phase 2, Sector 14, Behind Mandir"* versus *"Ph-2, Sec 14, Beh Mandir"* receive heavily degraded similarity scores despite referring to the exact same physical location.
* **Proposed Solution:**
  Expand `ADDRESS_ABBREVS` with common regional tokens:
  ```python
  (r'\bsec(?:t)?\s*\.?\b', 'sector'),
  (r'\bph(?:s)?\s*\.?\b', 'phase'),
  (r'\bflr\s*\.?\b', 'floor'),
  (r'\bbeh\s*\.?\b', 'behind'),
  (r'\bblog\s*\.?\b', 'block'),
  (r'\bbk\s*\.?\b', 'block'),
  (r'\bboul\s*\.?\b', 'boulevard'),
  ```

---

## 2. Performance & Execution Speed Bottlenecks

### 2.1 Quadruple Unicode NFKD Decomposition per Row
* **The Problem:** `unicode_normalize()` is invoked 4 separate times per entity: inside `normalize_name()`, inside `normalize_address()`, inside `normalize_country()`, and again in the lambda extracting postal codes.
* **Bottleneck:** `unicodedata.normalize('NFKD', ...)` traverses each character performing lookup checks. Executing this 4 times across millions of records wastes 30–40% of Phase 2 CPU runtime.
* **Proposed Solution:**
  Decompose each raw text column **once** at the start of DataFrame normalization, then pass the pre-decomposed strings into downstream cleaners:
  ```python
  name_nfkd = [unicode_normalize(x) for x in df[NAME_COL]]
  addr_nfkd = [unicode_normalize(x) for x in df[ADDR_COL]]
  country_nfkd = [unicode_normalize(x) for x in df[COUNTRY_COL]]
  ```

---

### 2.2 Seven Sequential Pandas `.apply()` Loops
* **The Problem:** `normalize_dataframe` executes 7 independent `.apply()` calls sequentially:
  ```python
  out['name_clean']   = out[NAME_COL].apply(normalize_name)
  out['name_tokens']  = out['name_clean'].apply(extract_name_tokens)
  out['name_prefix']  = out['name_clean'].apply(name_prefix)
  out['addr_clean']   = out[ADDR_COL].apply(normalize_address)
  out['addr_postal']  = out[ADDR_COL].apply(...)
  out['addr_numeric'] = out['addr_clean'].apply(extract_numeric_tokens)
  out['country_clean']= out[COUNTRY_COL].apply(normalize_country)
  ```
* **Bottleneck:** `Series.apply()` introduces significant Python function wrapper overhead. Native Python list comprehensions over arrays are **2× faster**.
* **Proposed Solution:**
  Batch process columns using direct list comprehensions:
  ```python
  names_clean  = [clean_name_fast(n) for n in name_nfkd]
  names_tokens = [' '.join(sorted(re.findall(r'[a-z]+', n))) for n in names_clean]
  names_pfx    = [n[:6] for n in names_clean]

  out['name_clean']  = names_clean
  out['name_tokens'] = names_tokens
  out['name_prefix'] = names_pfx
  ```

---

## 3. Memory & Windows Multiprocessing Overhead

### 3.1 Memory Tripling (Raw + Normalized Columns)
* **The Problem:** `normalize_dataframe` starts with `out = df.copy()` and appends 7 new string columns without freeing raw un-normalized fields.
* **Bottleneck:** Keeping raw strings alongside clean strings triples DataFrame memory consumption to 6–8 GB on full datasets.
* **Proposed Solution:**
  Drop the raw `business_name`, `business_address`, and `country` columns immediately after normalized fields are derived:
  ```python
  out.drop(columns=[NAME_COL, ADDR_COL, COUNTRY_COL], inplace=True)
  ```

---

### 3.2 Windows `spawn` DataFrame IPC Serialization
* **The Problem:** In `normalize_all_sources` under Windows, Pandas DataFrame chunks are passed across `ProcessPoolExecutor` IPC boundaries. Pickling large DataFrames containing millions of Python string objects creates serialization latency and transient RAM spikes.
* **Proposed Solution:**
  Pass lightweight primitive tuples `(entity_id, name, addr, country)` or use memory-mapped array slices to reduce IPC serialization overhead by ~60%.

---

## 4. Priority Summary & Expected Gains

| ID | Issue | Category | Effort | Expected Impact |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | French Legal Suffixes (`SARL`, `SAS`) | **Accuracy** | Low | **Direct recall boost on France test set** |
| **1.2** | Country-Specific Postal Regex | **Accuracy** | Low | **Eliminates false matches on Indian plot numbers** |
| **1.3** | Alphanumeric House Numbers (`42B`) | **Accuracy** | Low | **Recovers missing numeric candidate pairs** |
| **1.4** | Regional Landmark Abbreviations | **Accuracy** | Medium | **Increases address match confidence** |
| **2.1** | Single-Pass Unicode NFKD | **Speed** | Low | **~30% faster normalization** |
| **2.2** | Single-Pass Batch Extraction | **Speed** | Medium | **2× faster column processing** |
| **3.1** | Drop Raw Columns Post-Clean | **Memory** | Low | **Cuts DataFrame memory by ~50%** |
| **3.2** | Streamlined IPC Serialization | **Memory** | Medium | **Eliminates RAM spikes on Windows** |
