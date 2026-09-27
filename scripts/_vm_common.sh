#!/usr/bin/env bash
# _vm_common.sh — shared hardware detection for the VM scripts.
# Source this after REPO_DIR is set:  . "$REPO_DIR/scripts/_vm_common.sh"

_VM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VM_REPO_DIR="$(cd "$_VM_SCRIPTS_DIR/.." && pwd)"

vm_cores() {
    nproc --all 2>/dev/null || grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 4
}

vm_ram_gb() {
    # whole GB of installed RAM
    awk '/^MemTotal:/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null \
        || free -g 2>/dev/null | awk '/^Mem:/{print $2}' \
        || echo 16
}

vm_gpu_name() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
    fi
}

# True when setup_vm.sh --gpu successfully installed a CUDA-enabled LightGBM.
vm_has_gpu_build() {
    [ -f "$_VM_REPO_DIR/.venv/.lgbm_gpu" ]
}

# Print "WORKERS MAX_CANDIDATES NEG_RATIO" tuned to the detected RAM (167 GB box
# gets the full-recall setting). Overridable via env vars of the same names.
vm_train_defaults() {
    local ram="${1:-$(vm_ram_gb)}"
    if   [ "$ram" -ge 128 ]; then echo "100 10"
    elif [ "$ram" -ge 64  ]; then echo "80 6"
    else                          echo "50 4"
    fi
}
