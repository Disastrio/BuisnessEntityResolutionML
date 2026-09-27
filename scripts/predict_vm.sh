#!/usr/bin/env bash
# predict_vm.sh — Full test-set inference + submission validation, auto-tuned.
#
# Detects cores/RAM and scales the streaming chunk size accordingly. Loads the
# most recently trained model, scores every test S1 entity, writes
# output/matching_results.tsv + output/candidate_pairs.tsv, then runs the
# official validator (must print PASS).
#
# Override with env vars:
#   CHUNK_SIZE=50000   bash scripts/predict_vm.sh
#   LIMIT_S1=5000      bash scripts/predict_vm.sh   # staged check, first N S1 only
#   RESUME=1           bash scripts/predict_vm.sh   # continue an interrupted run
#   TEAM=mytem         bash scripts/predict_vm.sh   # also build submissions/<team>_submission.zip
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# shellcheck disable=SC1091
. "$REPO_DIR/scripts/_vm_common.sh"

if [ ! -d .venv ]; then
    echo "[FAIL] .venv not found. Run: bash scripts/setup_vm.sh"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

CORES="$(vm_cores)"
RAM_GB="$(vm_ram_gb)"

WORKERS="${WORKERS:-$CORES}"
if [ -z "${CHUNK_SIZE:-}" ]; then
    if   [ "$RAM_GB" -ge 128 ]; then CHUNK_SIZE=50000
    elif [ "$RAM_GB" -ge 64  ]; then CHUNK_SIZE=30000
    else                             CHUNK_SIZE=20000
    fi
fi
LIMIT_S1="${LIMIT_S1:-}"
RESUME="${RESUME:-}"
TEAM="${TEAM:-}"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export ER_WORKERS="$WORKERS"
export ER_LGBM_THREADS="$WORKERS"
export OMP_NUM_THREADS="$WORKERS"

mkdir -p reports output
LOG="reports/predict_$(date +%Y%m%d_%H%M%S).log"

ARGS=(--mode predict --workers "$WORKERS" --chunk-size "$CHUNK_SIZE")
if [ -n "$LIMIT_S1" ]; then ARGS+=(--limit-s1 "$LIMIT_S1"); fi
if [ -n "$RESUME" ]; then ARGS+=(--resume); fi

echo "=============================================================="
echo " Inference (auto-tuned)"
echo "   cores      : $CORES"
echo "   ram        : $RAM_GB GB"
echo "   workers    : $WORKERS"
echo "   chunk_size : $CHUNK_SIZE"
echo "   limit_s1   : ${LIMIT_S1:-<all>}"
echo "   log        : $LOG"
echo "=============================================================="

python -m src.pipeline "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"

if [ -n "$LIMIT_S1" ]; then
    echo
    echo "[skip] validation skipped for a partial (--limit-s1) run."
    exit 0
fi

echo
echo "=============================================================="
echo " Submission validation"
echo "=============================================================="
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

if [ -n "$TEAM" ]; then
    echo
    echo "=============================================================="
    echo " Packaging submission"
    echo "=============================================================="
    python scripts/make_submission_zip.py --team "$TEAM"
fi
