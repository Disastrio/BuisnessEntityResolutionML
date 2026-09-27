# VM Runbook — Full-Data Training on `uas-big`

Exact, copy-paste steps to go from a fresh clone to a validated submission on the
A100 VM (12 cores, 167 GB RAM, CUDA 13 driver).

> This pipeline is **CPU LightGBM** by default: blocking + pairwise feature
> generation run on all 12 cores and are the real bottleneck. The helper scripts
> auto-detect cores/RAM and scale to the machine. Optionally, `setup_vm.sh --gpu`
> builds CUDA-enabled LightGBM so the A100 accelerates the training step (see §3).
> 167 GB RAM is comfortable; disk is the real constraint.

---

## 0. Prerequisites (once, on the VM)

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
# optional, only for the A100 / CUDA LightGBM path:
sudo apt-get install -y cmake nvidia-cuda-toolkit
```

Python 3.10+ is required (3.12 recommended).

---

## 1. Clone the repo

```bash
git clone https://github.com/Disastrio/BuisnessEntityResolutionML.git
cd BuisnessEntityResolutionML
```

If the repo is already cloned:

```bash
cd BuisnessEntityResolutionML && git pull
```

---

## 2. Upload the dataset

The challenge files are **tab-separated** and are git-ignored, so they are not in
the repo. Create the exact directory layout and copy the files in.

```bash
mkdir -p dataset/train dataset/test
```

From your local machine (where the files are), `scp` them to the VM:

```bash
# run this on YOUR LOCAL machine, not the VM
scp dataset/train/*.tsv uasdtu28@uas-big:~/BuisnessEntityResolutionML/dataset/train/
scp dataset/test/*.tsv  uasdtu28@uas-big:~/BuisnessEntityResolutionML/dataset/test/
```

Required filenames on the VM:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

`setup_vm.sh` prints a `[ok]/[MISSING]` line per file so you can confirm.

---

## 3. Bootstrap the environment

```bash
bash scripts/setup_vm.sh          # CPU LightGBM
# or, to use the A100:
bash scripts/setup_vm.sh --gpu    # builds CUDA-enabled LightGBM (~20-40 min)
```

This creates `.venv`, installs `requirements.txt`, makes `models/ output/ reports/
submissions/ logs/`, and prints CPU/RAM/GPU plus a dataset presence report. It
auto-detects cores and RAM and reports the concurrency defaults it will use.

### Using the A100 (optional)

By default the PyPI `lightgbm` wheel is **CPU-only**. `setup_vm.sh --gpu` clones
LightGBM `v4.7.0` and builds it with `-DUSE_CUDA=1` against the installed CUDA
toolkit, then drops a `.venv/.lgbm_gpu` marker. If the build succeeds, `train_vm.sh`
automatically sets `ER_LGBM_DEVICE=cuda`; if it fails, the script warns and keeps
CPU LightGBM (nothing breaks). Requires `cmake` and `nvcc`:

```bash
sudo apt-get install -y cmake nvidia-cuda-toolkit
```

> The GPU accelerates only the LightGBM training step. Blocking and pairwise
> feature generation are CPU/RapidFuzz work and always run on all cores. For a
> 28-feature tabular model the CPU path is already fast; treat the GPU build as
> an optional optimization, not a requirement.

Activate the venv for every interactive session:

```bash
source .venv/bin/activate
```

---

## 4. Smoke test (2–3 min)

Verifies the full lifecycle (ingestion → normalize → blocking → features →
training → evaluation → output) on a 10k aligned sample before committing hours.

```bash
PYTHONIOENCODING=utf-8 python -m src.pipeline --mode smoke --workers 12
```

Expect a `Training Complete` line with a validation macro F0.5. If this fails,
do not start full training — see Troubleshooting.

---

## 5. Full-data training

Full data = 2.2M S1, 5.0M S2, 5.3M S3 (~10.3M targets). Run it inside `tmux` so a
dropped SSH session does not kill the job.

```bash
tmux new -s train

# (inside tmux)
cd ~/BuisnessEntityResolutionML
source .venv/bin/activate
bash scripts/train_vm.sh
```

Detach with `Ctrl-b d`, re-attach with `tmux attach -t train`.

`train_vm.sh` auto-detects cores/RAM and scales accordingly. On this VM
(12 cores, 167 GB) it resolves to:

| Setting | Auto value | Effect |
|---|---|---|
| `--workers` | core count (12) | process pool = all cores |
| `ER_LGBM_THREADS` / `OMP_NUM_THREADS` | core count (12) | LightGBM + OpenMP threads |
| `--max-candidates` | 100 (RAM ≥128 GB) | full-recall candidate cap per S1 |
| `--neg-ratio` | 10 (RAM ≥128 GB) | keep all positives + hard negatives + ≤10× easy negatives |
| `--source-thresholds` | on | separate S2/S3 decision thresholds (E5) |
| `ER_LGBM_DEVICE` | `cuda` if `--gpu` build present, else `cpu` | LightGBM backend |

The RAM→`(max-candidates, neg-ratio)` mapping is
`≥128 GB → (100, 10)`, `≥64 GB → (80, 6)`, else `(50, 4)`. Every value is an env
override, so you can dial it without editing scripts.

Blocking now also runs a **BM25 word-n-gram recall pass** (`BM25Index`) unioned
with the inverted-index blocks. Disable it with `ER_USE_BM25=0` to reproduce the
previous blocking behavior; quantify its candidate-recall contribution with
`python scripts/sparse_sweep.py 20000` (prints SPARSE / BM25 / UNION recall).

Outputs: a timestamped log in `reports/`, and `models/lgbm_pair_classifier*` plus
its `*_meta.json` (validation F0.5, chosen threshold, per-source thresholds, barrier).

### Variants

```bash
# Use the A100 for training (after `setup_vm.sh --gpu`)
ER_LGBM_DEVICE=cuda bash scripts/train_vm.sh

# Max-recall at the expense of wall-clock (bump negative sampling too)
MAX_CANDIDATES=150 NEG_RATIO=20 bash scripts/train_vm.sh

# Precision-first: force the conservative rule and require 99% pair precision
bash scripts/train_vm.sh --force-rules --min-precision 0.99

# Faster iteration on a large sample instead of the full set
SAMPLE=300000 bash scripts/train_vm.sh

# Cap CPU usage / RAM if you need to share the box
WORKERS=8 MAX_CANDIDATES=50 NEG_RATIO=4 bash scripts/train_vm.sh
```

### Resource notes (full data)

- Feature matrix is `pairs × 28` float32. At the 100-candidate cap the worst case is
  ~220M pairs (~25 GB); the negative ratio then bounds the LightGBM training matrix
  while keeping all positives and hard negatives.
- Expect **blocking + feature generation to dominate wall-clock**, not LightGBM.
- `MAX_CANDIDATES=40–50` roughly halves feature-generation time with near-identical
  macro F0.5 because ranking keeps the true matches and hard negatives.

---

## 6. Test inference + validation

```bash
tmux new -s predict
# (inside tmux)
cd ~/BuisnessEntityResolutionML && source .venv/bin/activate
bash scripts/predict_vm.sh
```

This streams all test S1 in chunks (bounded memory), writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the
official validator (must print `PASS`). `predict_vm.sh` auto-scales the chunk size
to RAM (`≥128 GB → 50000`, `≥64 GB → 30000`, else `20000`) and uses all cores.

Staged check first, if you prefer (first 5,000 S1 only):

```bash
LIMIT_S1=5000 bash scripts/predict_vm.sh
```

Resume an interrupted full run:

```bash
RESUME=1 bash scripts/predict_vm.sh
```

Manual validation (equivalent to the script's final step):

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Full inference writes ~10 GB; make sure `output/` has that much free space.

---

## 7. Package the submission

```bash
TEAM=yourteam bash scripts/predict_vm.sh   # runs inference + validation + zip
```

or, after inference has already run:

```bash
python scripts/make_submission_zip.py --team yourteam
```

Produces `submissions/yourteam_submission.zip` with the required structure:

```
yourteam_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
```

---

## 8. End-to-end command sequence (TL;DR)

```bash
# on the VM, from ~/BuisnessEntityResolutionML
sudo apt-get install -y git python3 python3-venv python3-pip      # once
# optional for the A100 path: sudo apt-get install -y cmake nvidia-cuda-toolkit
git clone https://github.com/Disastrio/BuisnessEntityResolutionML.git
cd BuisnessEntityResolutionML
mkdir -p dataset/train dataset/test
# ... scp the 7 TSV files into dataset/train and dataset/test ...

bash scripts/setup_vm.sh                 # add --gpu to build CUDA LightGBM
source .venv/bin/activate
PYTHONIOENCODING=utf-8 python -m src.pipeline --mode smoke --workers "$(nproc)"   # verify

tmux new -s train && bash scripts/train_vm.sh          # full training (auto-tuned)
tmux new -s predict && bash scripts/predict_vm.sh      # inference + validate
TEAM=yourteam bash scripts/predict_vm.sh               # add packaging
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `python3-venv` error on setup | `sudo apt-get install -y python3-venv` then re-run `setup_vm.sh`. |
| `[MISSING] dataset/...` | Files not uploaded yet or wrong filenames/paths; re-check step 2. |
| Validator prints issues | Do not submit. Read the numbered issues; common ones are missing S1 rows or token IDs not in candidates. Re-run inference with `RESUME=` unset to rebuild. |
| Out of memory during training | Lower `MAX_CANDIDATES` (e.g. 40) or `NEG_RATIO` (e.g. 4) and re-run. |
| Training very slow | Expected on full data; reduce `MAX_CANDIDATES` to 40, lower `NEG_RATIO`, and use `SAMPLE=300000` for iteration. |
| GPU build failed / still on CPU | `setup_vm.sh --gpu` warns and continues on CPU; ensure `cmake` + `nvcc` are installed, or ignore — CPU training is valid. |
| `device_type=cuda` errors at train time | Remove the marker (`.venv/.lgbm_gpu`) or run with `ER_LGBM_DEVICE=cpu bash scripts/train_vm.sh`. |
| SSH session died mid-training | Use `tmux` (step 5). Partial logs remain in `reports/`. |
| Output row count mismatch | `matching_results.tsv` must have exactly one row per test S1 (~1,732,544). Re-run full inference without `--limit-s1`. |

## Guardrails (do not violate)

- No external APIs, geocoding, business databases, or data augmentation — disqualification.
- Always TSV (`sep="\t"`); addresses and ID lists contain commas.
- `country` is an open string (test includes France); never hard-code US/India.
- Model must stay MIT/Apache-2.0 and ≤8B params (LightGBM is fine).
