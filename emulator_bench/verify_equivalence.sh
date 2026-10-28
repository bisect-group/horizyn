#!/usr/bin/env bash
# Checks that the wrapper (emulator_bench) reproduces the original Horizyn code path.
# Expected environment: a Python with torch, lightning, rdkit, drfp, h5py, pandas, pyarrow, pytest and the
# horizyn package (EMULaToR uses /home/adhil/.conda/envs/horizyn_venv). This script does not create it.
# Usage: emulator_bench/verify_equivalence.sh [--gpu | --negative-control]
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
echo "Expected environment: horizyn_venv (python: $PYTHON)"

FILES=(emulator_bench/test_original_equivalence.py emulator_bench/test_fit_patches.py
       emulator_bench/test_wrapper.py emulator_bench/test_end_to_end.py)
PYTEST=("$PYTHON" -m pytest -q -rs -p no:cacheprovider --color=no -o addopts="")
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if [[ "${1:-}" == "--negative-control" ]]; then
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || {
    echo "FAIL: the negative control needs a CUDA device." >&2
    exit 1
  }
  echo "Negative control: the production run starts from weights times 1.05. Every GPU test must FAIL."
  if EQUIV_NEGATIVE_CONTROL=1 "${PYTEST[@]}" -s -m gpu emulator_bench/test_gpu_precision.py | tee "$log"; then
    echo "FAIL: a GPU test passed with the negative control (the bound is vacuous)." >&2
    exit 1
  fi
  grep -q "ratio=" "$log" || { echo "FAIL: the GPU test crashed before it measured a ratio." >&2; exit 1; }
  grep "ratio=" "$log"
  echo "Negative control failed as expected."
  exit 0
fi

echo "CPU mode"
CUDA_VISIBLE_DEVICES="" "${PYTEST[@]}" -m "not gpu" "${FILES[@]}" | tee "$log"
if grep -Eq "[0-9]+ skipped" "$log"; then
  echo "FAIL: a test was skipped in CPU mode" >&2
  exit 1
fi

if [[ "${1:-}" == "--gpu" ]]; then
  echo "GPU mode (bf16 against fp32)"
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || {
    echo "No CUDA device is visible: the GPU tests are skipped." >&2
    exit 0
  }
  "${PYTEST[@]}" -s -m gpu emulator_bench/test_gpu_precision.py | tee "$log"
  if grep -Eq "[0-9]+ skipped" "$log"; then
    echo "FAIL: a GPU test was skipped" >&2
    exit 1
  fi
fi
echo "Equivalence checks passed."
