#!/usr/bin/env bash
# train_vm.sh — Full-data training, auto-tuned to the machine's capabilities.
#
# Detects CPU cores and RAM, then:
#   * sets every process pool / LightGBM / OMP thread to the core count,
#   * picks the candidate cap + negative ratio from available RAM (a 167 GB box
#     gets the full-recall, low-subsampling setting),
#   * enables CUDA LightGBM automatically if `setup_vm.sh --gpu` was used.
#
# Defaults (override with env vars):
#   SAMPLE=300000      quick large-sample run instead of full data
#   MAX_CANDIDATES=50  candidate cap per S1 (recall vs time)
#   NEG_RATIO=6        negatives kept per positive (all positives + hard negatives always kept)
#   WORKERS=12         process pool size
#   ER_LGBM_DEVICE=cpu force CPU even if a CUDA build is present
#
# Extra flags pass straight through, e.g.
#   bash scripts/train_vm.sh --force-rules --min-precision 0.99
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
read -r AUTO_MAXCAND AUTO_NEGRATIO <<<"$(vm_train_defaults "$RAM_GB")"

WORKERS="${WORKERS:-$CORES}"
MAX_CANDIDATES="${MAX_CANDIDATES:-$AUTO_MAXCAND}"
NEG_RATIO="${NEG_RATIO:-$AUTO_NEGRATIO}"
SAMPLE="${SAMPLE:-}"
# Pair-precision floor: maximise macro F0.5 subject to validation pair precision
# >= this. Default 0.98 leaves a buffer so the unseen test pair precision stays
# above the 0.97 requirement. Set TARGET_PRECISION= (empty) to fall back to the
# F0.5-only sweep, or TARGET_PRECISION=0.97 for the exact floor.
TARGET_PRECISION="${TARGET_PRECISION-0.98}"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export ER_WORKERS="$WORKERS"
export ER_LGBM_THREADS="$WORKERS"
export OMP_NUM_THREADS="$WORKERS"

DEVICE_NOTE="cpu"
if [ "${ER_LGBM_DEVICE:-}" != "cpu" ] && vm_has_gpu_build; then
    export ER_LGBM_DEVICE="${ER_LGBM_DEVICE:-cuda}"
    DEVICE_NOTE="$ER_LGBM_DEVICE ($(vm_gpu_name))"
fi

mkdir -p reports models
LOG="reports/train_$(date +%Y%m%d_%H%M%S).log"

ARGS=(--mode train --workers "$WORKERS" --max-candidates "$MAX_CANDIDATES" --neg-ratio "$NEG_RATIO")
if [ -n "$TARGET_PRECISION" ]; then
    ARGS+=(--target-precision "$TARGET_PRECISION")
else
    ARGS+=(--source-thresholds)
fi
if [ -n "$SAMPLE" ]; then
    ARGS+=(--sample "$SAMPLE")
fi

echo "=============================================================="
echo " Training (auto-tuned)"
echo "   cores          : $CORES"
echo "   ram            : $RAM_GB GB"
echo "   workers        : $WORKERS"
echo "   lgbm device    : $DEVICE_NOTE"
echo "   max_candidates : $MAX_CANDIDATES"
echo "   neg_ratio      : $NEG_RATIO"
echo "   target_prec    : ${TARGET_PRECISION:-<none: pure F0.5 sweep>}"
echo "   sample         : ${SAMPLE:-<full dataset>}"
echo "   log            : $LOG"
echo "=============================================================="
echo "Tip: run inside tmux/nohup so a dropped SSH session does not kill training."
echo

python -m src.pipeline "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"

echo
echo "=============================================================="
echo " Trained model artifacts"
echo "=============================================================="
ls -lh models/ | grep -v '^total' || true
python - <<'EOF'
import json, pathlib
for p in sorted(pathlib.Path("models").glob("*_meta.json")):
    print(f"\n[{p.name}]")
    print(json.dumps(json.loads(p.read_text()), indent=2))
EOF
echo
echo "Next: bash scripts/predict_vm.sh"
