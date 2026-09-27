#!/usr/bin/env bash
# setup_vm.sh — Bootstrap the Business Entity Resolution repo on a fresh Linux VM.
#
# Idempotent: safe to re-run. Creates a virtualenv, installs pinned deps, makes
# the working directories, and reports CPU / RAM / GPU plus which dataset files
# are present.
#
# Usage:
#   bash scripts/setup_vm.sh              # CPU LightGBM (default)
#   bash scripts/setup_vm.sh --gpu        # also build CUDA-enabled LightGBM for the A100
#
# The dataset is NOT downloaded (challenge rules forbid external data). Upload
# the TSV files yourself into dataset/train and dataset/test (see the runbook).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# shellcheck disable=SC1091
. "$REPO_DIR/scripts/_vm_common.sh"

BUILD_GPU=0
for arg in "$@"; do
    case "$arg" in
        --gpu) BUILD_GPU=1 ;;
        -h|--help)
            cat <<'USAGE'
Usage: bash scripts/setup_vm.sh [--gpu]

  (no flag)   install pinned CPU dependencies into .venv
  --gpu       additionally build CUDA-enabled LightGBM for an NVIDIA GPU

The dataset is not downloaded (challenge rules forbid external data);
upload the TSV files into dataset/train and dataset/test yourself.
USAGE
            exit 0 ;;
        *) echo "[warn] unknown argument: $arg" ;;
    esac
done

CORES="$(vm_cores)"
RAM_GB="$(vm_ram_gb)"
GPU_NAME="$(vm_gpu_name)"

echo "=============================================================="
echo " Business Entity Resolution - VM setup"
echo " repo   : $REPO_DIR"
echo " cpu    : ${CORES} cores"
echo " ram    : ${RAM_GB} GB"
echo " gpu    : ${GPU_NAME:-none detected}"
echo "=============================================================="

# ── 1. Python check ──────────────────────────────────────────────────────────
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[FAIL] '$PY' not found. Install Python 3.10+ (e.g. sudo apt-get install python3 python3-venv)."
    exit 1
fi
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 10), f"Python 3.10+ required, found {sys.version.split()[0]}"
print(f"[ok] Python {sys.version.split()[0]}")
EOF

# ── 2. Virtualenv ────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    echo "[..] creating virtualenv .venv"
    "$PY" -m venv .venv || {
        echo "[FAIL] venv creation failed. Install the venv module: sudo apt-get install python3-venv"
        exit 1
    }
else
    echo "[ok] .venv already exists"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 3. Dependencies ──────────────────────────────────────────────────────────
echo "[..] installing pinned dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ── 3b. Optional: CUDA-enabled LightGBM (uses the A100) ──────────────────────
if [ "$BUILD_GPU" -eq 1 ]; then
    echo "--------------------------------------------------------------"
    echo " Building CUDA-enabled LightGBM (this can take 20-40 min)"
    echo "--------------------------------------------------------------"
    if [ -z "$GPU_NAME" ]; then
        echo "[warn] no NVIDIA GPU detected; skipping the GPU build."
    elif ! python -c "import torch" 2>/dev/null && \
         [ ! -x /usr/local/cuda/bin/nvcc ] && \
         ! ls /usr/local/cuda-*/bin/nvcc >/dev/null 2>&1 && \
         ! command -v nvcc >/dev/null 2>&1; then
        echo "[warn] nvcc not found. Install the CUDA toolkit:"
        echo "       sudo apt-get install -y nvidia-cuda-toolkit   # or the matching cuda-toolkit-XX"
        echo "[warn] skipping the GPU build; CPU LightGBM remains installed."
    elif ! command -v cmake >/dev/null 2>&1; then
        echo "[warn] cmake not found (sudo apt-get install cmake); skipping the GPU build."
    else
        CUDACXX_TMP="$(command -v nvcc || ls /usr/local/cuda-*/bin/nvcc 2>/dev/null | tail -1 || echo /usr/local/cuda/bin/nvcc)"
        export CUDACXX="${CUDACXX:-$CUDACXX_TMP}"
        SRC=/tmp/LightGBM-v4.7.0
        set +e
        rm -rf "$SRC"
        git clone --recursive --branch v4.7.0 --depth 1 \
            https://github.com/microsoft/LightGBM "$SRC"
        cmake -S "$SRC" -B "$SRC/build" \
            -DUSE_CUDA=1 -DCMAKE_CUDA_COMPILER="$CUDACXX" \
            -DBUILD_CPP_TEST=0 -DBUILD_CLI=1
        cmake --build "$SRC/build" -j"$CORES"
        BUILD_RC=$?
        if [ $BUILD_RC -eq 0 ]; then
            python -m pip install --no-build-isolation --force-reinstall --no-deps "$SRC/python-package"
            touch .venv/.lgbm_gpu
            echo "[ok] CUDA LightGBM installed -> GPU training enabled"
        else
            echo "[warn] CUDA LightGBM build failed (rc=$BUILD_RC); staying on CPU LightGBM."
        fi
        set -e
    fi
fi

# ── 4. Directories ───────────────────────────────────────────────────────────
mkdir -p dataset/train dataset/test models output reports submissions logs
echo "[ok] directories: dataset/{train,test} models output reports submissions logs"

# ── 5. Concurrency defaults ──────────────────────────────────────────────────
read -r AUTO_MAXCAND AUTO_NEGRATIO <<<"$(vm_train_defaults "$RAM_GB")"
echo "[ok] env defaults -> WORKERS=${CORES}  MAX_CANDIDATES=${AUTO_MAXCAND}  NEG_RATIO=${AUTO_NEGRATIO}"
if vm_has_gpu_build; then
    echo "[ok] CUDA LightGBM detected: train_vm.sh will set ER_LGBM_DEVICE=cuda automatically."
fi

# ── 6. Dataset presence check ────────────────────────────────────────────────
echo "--------------------------------------------------------------"
echo " Dataset files (upload these; not downloaded automatically)"
echo "--------------------------------------------------------------"
MISSING=0
for f in \
    dataset/train/train_source1.tsv \
    dataset/train/train_source2.tsv \
    dataset/train/train_source3.tsv \
    dataset/train/train_ground_truth.tsv \
    dataset/test/test_source1.tsv \
    dataset/test/test_source2.tsv \
    dataset/test/test_source3.tsv
do
    if [ -f "$f" ]; then
        printf '[ok]      %-48s %s\n' "$f" "$(du -h "$f" | cut -f1)"
    else
        printf '[MISSING] %-48s\n' "$f"
        MISSING=1
    fi
done

echo "--------------------------------------------------------------"
if [ "$MISSING" -eq 1 ]; then
    echo "[!] Some dataset files are missing. Upload them before training."
else
    echo "[ok] all dataset files present."
fi

cat <<'EOF'

--------------------------------------------------------------
 Next steps
--------------------------------------------------------------
  source .venv/bin/activate

  # quick end-to-end verification (10k aligned sample, ~2-3 min)
  PYTHONIOENCODING=utf-8 python -m src.pipeline --mode smoke --workers $(nproc)

  # full training + inference + validation: see docs/VM_RUNBOOK.md
  bash scripts/train_vm.sh
  bash scripts/predict_vm.sh
--------------------------------------------------------------
EOF
